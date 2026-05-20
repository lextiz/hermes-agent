from unittest.mock import patch

from agent.context_compressor import ContextCompressor, SUMMARY_PREFIX
from agent.model_metadata import estimate_messages_tokens_rough
from plugins.context_engine import load_context_engine
from plugins.context_engine.branch_compressor import (
    BranchAwareContextCompressor,
    _CONTRIBUTING,
    _IRRELEVANT,
)


def _engine() -> BranchAwareContextCompressor:
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        engine = BranchAwareContextCompressor()
    engine.configure(
        {
            "enabled": True,
            "min_branch_chars": 200,
            "max_branch_summary_chars": 500,
            "include_negative_findings": True,
            "preserve_failed_branch_details": False,
        },
        compression_config={
            "threshold": 0.50,
            "protect_first_n": 1,
            "protect_last_n": 3,
            "target_ratio": 0.10,
        },
        quiet_mode=True,
    )
    engine.update_model("test/model", 100000, provider="test")
    return engine


def _tool_call(call_id: str, name: str, args: str = "{}"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": args},
    }


def _conversation():
    long_failed_detail = (
        "exit 1\n"
        "failed because dependency alpha was missing\n"
        "STACKTRACE_UNIQUE_RAW_DETAIL " + ("x" * 800)
    )
    return [
        {"role": "system", "content": "SYSTEM_HEAD_CONTEXT"},
        {"role": "user", "content": "protected opening user context"},
        {"role": "assistant", "content": "protected opening assistant context"},
        {"role": "user", "content": "Attempt A: try dependency alpha resolution. " + ("a" * 260)},
        {
            "role": "assistant",
            "content": "I will run the alpha dependency check.",
            "tool_calls": [_tool_call("call-a", "terminal", '{"command":"check alpha"}')],
        },
        {"role": "tool", "tool_call_id": "call-a", "content": long_failed_detail},
        {"role": "assistant", "content": "This did not fix the failure; switching to another approach."},
        {"role": "user", "content": "Attempt B: inspect beta config and implement the safer fix. " + ("b" * 260)},
        {
            "role": "assistant",
            "content": "I will read beta config and patch it.",
            "tool_calls": [_tool_call("call-b", "terminal", '{"command":"pytest tests/beta"}')],
        },
        {"role": "tool", "tool_call_id": "call-b", "content": "exit 0\n3 passed\nupdated agent/beta.py"},
        {"role": "assistant", "content": "Implemented the beta fix in agent/beta.py and tests passed."},
        {"role": "user", "content": "Recent tail user message remains verbatim."},
        {"role": "assistant", "content": "Recent tail assistant response remains verbatim."},
        {"role": "user", "content": "LATEST USER MESSAGE EXACT"},
    ]


def _summary_text(messages):
    return "\n\n".join(str(m.get("content", "")) for m in messages)


def _assert_valid_tool_pairs(messages):
    call_ids = set()
    result_ids = set()
    for msg in messages:
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                call_ids.add(tc.get("id"))
        if msg.get("role") == "tool":
            result_ids.add(msg.get("tool_call_id"))
    assert result_ids <= call_ids
    assert call_ids <= result_ids


def test_engine_can_be_selected_by_context_engine_loader():
    engine = load_context_engine("branch_compressor")
    assert isinstance(engine, BranchAwareContextCompressor)
    assert engine.name == "branch_compressor"


def test_default_compressor_remains_the_default_engine_class():
    compressor = ContextCompressor(model="test/model", quiet_mode=True, config_context_length=100000)
    assert compressor.name == "compressor"
    assert not isinstance(compressor, BranchAwareContextCompressor)


def test_segmentation_keeps_tool_call_and_result_together():
    engine = _engine()
    messages = _conversation()[3:11]

    branches = engine._segment_attempt_branches(messages)

    branch_with_call = next(b for b in branches if any(m.get("tool_calls") for m in b.messages))
    roles = [m.get("role") for m in branch_with_call.messages]
    assert "assistant" in roles
    assert "tool" in roles
    assistant_idx = next(i for i, m in enumerate(branch_with_call.messages) if m.get("tool_calls"))
    assert branch_with_call.messages[assistant_idx + 1]["role"] == "tool"


def test_compress_preserves_head_tail_and_latest_user_verbatim():
    engine = _engine()
    engine._call_branch_summary_model = lambda prompt, budget: None
    messages = _conversation()

    compressed = engine.compress(messages, current_tokens=90000)

    assert compressed[0]["role"] == "system"
    assert "SYSTEM_HEAD_CONTEXT" in compressed[0]["content"]
    assert any(m.get("content") == "Recent tail user message remains verbatim." for m in compressed)
    assert any(m.get("content") == "Recent tail assistant response remains verbatim." for m in compressed)
    assert compressed[-1] == {"role": "user", "content": "LATEST USER MESSAGE EXACT"}


def test_failed_branch_raw_details_omitted_but_negative_finding_remains():
    engine = _engine()
    engine._call_branch_summary_model = lambda prompt, budget: None

    compressed = engine.compress(_conversation(), current_tokens=90000)
    text = _summary_text(compressed)

    assert "STACKTRACE_UNIQUE_RAW_DETAIL" not in text
    assert "Failed but Relevant Branches" in text
    assert "Negative finding" in text
    assert "failed because" in text


def test_contributing_and_irrelevant_branches_are_structured_separately():
    engine = _engine()
    branches = engine._segment_attempt_branches([
        {"role": "user", "content": "Repeated log inspection with no new findings. " + ("x" * 220)},
        {"role": "assistant", "content": "No new findings; duplicate output."},
        {"role": "user", "content": "Implement useful beta fix. " + ("y" * 260)},
        {
            "role": "assistant",
            "content": "Updated agent/beta.py and pytest passed.",
            "tool_calls": [_tool_call("call-c", "terminal", '{"command":"pytest tests/beta"}')],
        },
        {"role": "tool", "tool_call_id": "call-c", "content": "exit 0\n3 passed"},
    ])
    for branch in branches:
        engine._classify_branch(branch)

    summary = engine._fallback_branch_summary(branches)

    assert any(b.classification == _IRRELEVANT for b in branches)
    assert any(b.classification == _CONTRIBUTING for b in branches)
    assert "## Contributing Branches" in summary
    assert "## Omitted or Superseded Branches" in summary


def test_bad_summarizer_output_falls_back_safely():
    engine = _engine()
    engine._call_branch_summary_model = lambda prompt, budget: "not valid"

    compressed = engine.compress(_conversation(), current_tokens=90000)
    text = _summary_text(compressed)

    assert "[Branch-aware conversation compression summary]" in text
    assert "Contributing Branches" in text
    assert "LATEST USER MESSAGE EXACT" in text


def test_provider_valid_message_sequence_after_compression():
    engine = _engine()
    engine._call_branch_summary_model = lambda prompt, budget: None

    compressed = engine.compress(_conversation(), current_tokens=90000)

    _assert_valid_tool_pairs(compressed)


def test_token_budget_improves_compared_with_uncompressed_middle():
    engine = _engine()
    engine._call_branch_summary_model = lambda prompt, budget: None
    messages = _conversation()

    compressed = engine.compress(messages, current_tokens=90000)

    assert estimate_messages_tokens_rough(compressed) < estimate_messages_tokens_rough(messages)


def test_configure_applies_branch_compressor_settings():
    engine = _engine()
    engine.configure(
        {
            "enabled": False,
            "min_branch_chars": 1500,
            "max_branch_summary_chars": 700,
            "include_negative_findings": False,
            "preserve_failed_branch_details": True,
            "model": "summary/model",
        },
        compression_config={"protect_first_n": 0, "protect_last_n": 4},
        quiet_mode=False,
    )

    assert engine.enabled is False
    assert engine.min_branch_chars == 1500
    assert engine.max_branch_summary_chars == 700
    assert engine.include_negative_findings is False
    assert engine.preserve_failed_branch_details is True
    assert engine.summary_model == "summary/model"
    assert engine.protect_first_n == 0
    assert engine.protect_last_n == 4
    assert engine.quiet_mode is False


def test_branch_summary_uses_existing_reference_prefix():
    engine = _engine()
    engine._call_branch_summary_model = lambda prompt, budget: None

    compressed = engine.compress(_conversation(), current_tokens=90000)
    text = _summary_text(compressed)

    assert SUMMARY_PREFIX in text
    assert "[Branch-aware conversation compression summary]" in text
