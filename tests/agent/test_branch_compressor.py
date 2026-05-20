import json
import re
from types import SimpleNamespace
from unittest.mock import patch

from agent.context_compressor import ContextCompressor, SUMMARY_PREFIX
from agent.model_metadata import estimate_messages_tokens_rough
from plugins.context_engine import load_context_engine
from plugins.context_engine.branch_compressor import (
    BranchSummary,
    BranchAwareContextCompressor,
    _CONTRIBUTING,
    _FAILED,
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


def _planner_json(prompt: str = ""):
    group_ids = [
        int(match.group(1))
        for match in re.finditer(r"^### Atomic group (\d+)", prompt, flags=re.MULTILINE)
    ] or list(range(1, 7))
    split = max(1, len(group_ids) // 2)
    return json.dumps({
        "branches": [
            {
                "group_indices": group_ids[:split],
                "classification": _FAILED,
                "title": "dependency alpha attempt",
                "key_points": ["Checked dependency alpha."],
                "negative_findings": [
                    "Tried dependency alpha resolution; failed because dependency alpha was missing; do not retry unless dependency alpha is installed."
                ],
                "omitted_reason": "",
            },
            {
                "group_indices": group_ids[split:],
                "classification": _CONTRIBUTING,
                "title": "beta config fix",
                "key_points": ["Updated agent/beta.py and pytest tests/beta passed."],
                "negative_findings": [],
                "omitted_reason": "",
            },
        ]
    })


def _install_llm_planner(engine):
    def plan(messages):
        prompt = messages[0]["content"] if isinstance(messages, list) else str(messages)
        return _planner_json(prompt)

    engine._call_branch_plan_model = plan


def _branch_summary_json(title: str, classification: str, *, failed: bool = False):
    payload = {
        "title": title,
        "classification": classification,
        "outcome": "failed" if failed else "contributed",
        "key_findings": [
            "Checked dependency alpha." if failed else "Updated agent/beta.py and pytest tests/beta passed."
        ],
        "files_commands_results": [] if failed else ["agent/beta.py updated", "pytest tests/beta -> 3 passed"],
        "negative_findings": [
            "Tried dependency alpha resolution; failed because dependency alpha was missing; do not retry unless dependency alpha is installed."
        ] if failed else [],
        "omitted_note": "",
        "current_state": [] if failed else ["agent/beta.py contains the safer fix."],
        "remaining_work": [],
    }
    return json.dumps(payload)


def _install_branch_summarizer(engine):
    calls = []

    def summarize(messages, max_tokens):
        calls.append((messages, max_tokens))
        prompt = messages[0]["content"] if isinstance(messages, list) else str(messages)
        if "dependency alpha attempt" in prompt:
            return _branch_summary_json("dependency alpha attempt", _FAILED, failed=True)
        return _branch_summary_json("beta config fix", _CONTRIBUTING)

    engine._call_branch_summary_model = summarize
    return calls


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
    _install_llm_planner(engine)
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
    _install_llm_planner(engine)
    _install_branch_summarizer(engine)
    messages = _conversation()

    compressed = engine.compress(messages, current_tokens=90000)

    assert compressed[0]["role"] == "system"
    assert "SYSTEM_HEAD_CONTEXT" in compressed[0]["content"]
    assert any(m.get("content") == "Recent tail user message remains verbatim." for m in compressed)
    assert any(m.get("content") == "Recent tail assistant response remains verbatim." for m in compressed)
    assert compressed[-1] == {"role": "user", "content": "LATEST USER MESSAGE EXACT"}


def test_failed_branch_raw_details_omitted_but_negative_finding_remains():
    engine = _engine()
    _install_llm_planner(engine)
    _install_branch_summarizer(engine)

    compressed = engine.compress(_conversation(), current_tokens=90000)
    text = _summary_text(compressed)

    assert "STACKTRACE_UNIQUE_RAW_DETAIL" not in text
    assert "Failed but Relevant Branches" in text
    assert "Negative finding" in text
    assert "failed because" in text


def test_summarizes_each_detected_branch_with_its_own_llm_call():
    engine = _engine()
    _install_llm_planner(engine)
    calls = _install_branch_summarizer(engine)

    engine.compress(_conversation(), current_tokens=90000)

    assert len(calls) == 2
    assert all(isinstance(messages, list) for messages, _max_tokens in calls)
    assert any("dependency alpha attempt" in messages[0]["content"] for messages, _ in calls)
    assert any("beta config fix" in messages[0]["content"] for messages, _ in calls)


def test_contributing_and_irrelevant_branches_are_structured_separately():
    engine = _engine()
    groups = engine._build_atomic_groups([
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
    branches = engine._parse_branch_plan(json.dumps({
        "branches": [
            {
                "group_indices": [1, 2],
                "classification": _IRRELEVANT,
                "title": "repeated log inspection",
                "key_points": [],
                "negative_findings": [],
                "omitted_reason": "Repeated inspection produced no new durable findings.",
            },
            {
                "group_indices": [3, 4],
                "classification": _CONTRIBUTING,
                "title": "beta fix",
                "key_points": ["Updated agent/beta.py and pytest passed."],
                "negative_findings": [],
                "omitted_reason": "",
            },
        ]
    }), groups)

    summary = engine._compose_branch_summary([
        BranchSummary(
            branch=branches[0],
            title="repeated log inspection",
            classification=_IRRELEVANT,
            outcome="omitted",
            omitted_note="Repeated inspection produced no new durable findings.",
        ),
        BranchSummary(
            branch=branches[1],
            title="beta fix",
            classification=_CONTRIBUTING,
            outcome="contributed",
            key_findings=["Updated agent/beta.py and pytest passed."],
        ),
    ])

    assert any(b.classification == _IRRELEVANT for b in branches)
    assert any(b.classification == _CONTRIBUTING for b in branches)
    assert "## Contributing Branches" in summary
    assert "## Omitted or Superseded Branches" in summary


def test_bad_branch_summary_output_falls_back_to_default_compressor():
    engine = _engine()
    _install_llm_planner(engine)
    engine._call_branch_summary_model = lambda messages, max_tokens: "not valid"

    with patch.object(
        ContextCompressor,
        "_generate_summary",
        return_value=f"{SUMMARY_PREFIX}\nDEFAULT FALLBACK SUMMARY",
    ):
        compressed = engine.compress(_conversation(), current_tokens=90000)
    text = _summary_text(compressed)

    assert "DEFAULT FALLBACK SUMMARY" in text
    assert "LATEST USER MESSAGE EXACT" in text
    assert engine.get_status()["branch_compressor_telemetry"]["default_fallback_used"] is True


def test_branch_summary_repair_reuses_same_context():
    engine = _engine()
    _install_llm_planner(engine)
    attempts = []

    def summarize(messages, max_tokens):
        attempts.append(messages)
        if len(attempts) == 1:
            return "not json"
        return _branch_summary_json("dependency alpha attempt", _FAILED, failed=True)

    engine._call_branch_summary_model = summarize

    with patch.object(
        ContextCompressor,
        "_generate_summary",
        return_value=f"{SUMMARY_PREFIX}\nDEFAULT FALLBACK SUMMARY",
    ):
        compressed = engine.compress(_conversation(), current_tokens=90000)
    text = _summary_text(compressed)

    assert "DEFAULT FALLBACK SUMMARY" not in text
    assert len(attempts) >= 2
    assert attempts[1][1]["role"] == "assistant"
    assert "not json" in attempts[1][1]["content"]
    assert "invalid_json" in attempts[1][2]["content"]
    assert engine.get_status()["branch_compressor_telemetry"]["branch_summary_repair_attempts_used"] == 1


def test_transient_auxiliary_llm_error_retries_before_failing():
    engine = _engine()
    engine.llm_retry_delay_seconds = 1
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"branches": []}'))]
    )

    with (
        patch(
            "plugins.context_engine.branch_compressor.call_llm",
            side_effect=[Exception("503 Loading model"), response],
        ),
        patch("plugins.context_engine.branch_compressor.time.sleep") as sleep,
    ):
        result = engine._call_auxiliary_llm_with_retries({"messages": []}, "planning")

    assert result is response
    assert sleep.call_count == 1
    assert engine.get_status()["branch_compressor_telemetry"]["llm_retries_used"] == 1


def test_provider_valid_message_sequence_after_compression():
    engine = _engine()
    _install_llm_planner(engine)
    _install_branch_summarizer(engine)

    compressed = engine.compress(_conversation(), current_tokens=90000)

    _assert_valid_tool_pairs(compressed)


def test_token_budget_improves_compared_with_uncompressed_middle():
    engine = _engine()
    _install_llm_planner(engine)
    _install_branch_summarizer(engine)
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
            "planner_model": "planner/model",
            "planner_max_tokens": 1234,
            "planner_repair_attempts": 3,
            "planner_group_max_chars": 456,
            "branch_summary_max_tokens": 2345,
            "branch_summary_repair_attempts": 2,
            "branch_source_max_chars": 3456,
            "llm_timeout": 180,
            "llm_call_retries": 4,
            "llm_retry_delay_seconds": 2,
            "telemetry_enabled": False,
            "log_branch_details": True,
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
    assert engine.planner_model == "planner/model"
    assert engine.planner_max_tokens == 1234
    assert engine.planner_repair_attempts == 3
    assert engine.planner_group_max_chars == 456
    assert engine.branch_summary_max_tokens == 2345
    assert engine.branch_summary_repair_attempts == 2
    assert engine.branch_source_max_chars == 3456
    assert engine.llm_timeout == 180
    assert engine.llm_call_retries == 4
    assert engine.llm_retry_delay_seconds == 2
    assert engine.telemetry_enabled is False
    assert engine.log_branch_details is True
    assert engine.protect_first_n == 0
    assert engine.protect_last_n == 4
    assert engine.quiet_mode is False


def test_branch_summary_uses_existing_reference_prefix():
    engine = _engine()
    _install_llm_planner(engine)
    _install_branch_summarizer(engine)

    compressed = engine.compress(_conversation(), current_tokens=90000)
    text = _summary_text(compressed)

    assert SUMMARY_PREFIX in text
    assert "[Branch-aware conversation compression summary]" in text


def test_bad_planner_output_repair_can_return_valid_plan():
    engine = _engine()
    attempts = []

    def plan(messages):
        attempts.append(messages)
        prompt = messages[0]["content"] if isinstance(messages, list) else str(messages)
        return "not json" if len(attempts) == 1 else _planner_json(prompt)

    engine._call_branch_plan_model = plan
    _install_branch_summarizer(engine)

    compressed = engine.compress(_conversation(), current_tokens=90000)
    text = _summary_text(compressed)

    assert "Failed but Relevant Branches" in text
    assert "dependency alpha" in text
    assert "STACKTRACE_UNIQUE_RAW_DETAIL" not in text
    telemetry = engine.get_status()["branch_compressor_telemetry"]
    assert telemetry["planner_repair_attempts_used"] == 1
    assert telemetry["planner_failed"] is False


def test_unrepairable_planner_output_uses_default_compressor_fallback():
    engine = _engine()
    engine._call_branch_plan_model = lambda messages: "not json"
    _install_branch_summarizer(engine)

    with patch.object(
        ContextCompressor,
        "_generate_summary",
        return_value=f"{SUMMARY_PREFIX}\nDEFAULT FALLBACK SUMMARY",
    ):
        compressed = engine.compress(_conversation(), current_tokens=90000)
    text = _summary_text(compressed)

    assert "DEFAULT FALLBACK SUMMARY" in text
    telemetry = engine.get_status()["branch_compressor_telemetry"]
    assert telemetry["planner_failed"] is True
    assert telemetry["default_fallback_used"] is True


def test_telemetry_records_branch_counts():
    engine = _engine()
    _install_llm_planner(engine)
    _install_branch_summarizer(engine)

    engine.compress(_conversation(), current_tokens=90000)
    telemetry = engine.get_status()["branch_compressor_telemetry"]

    assert telemetry["branches_total"] >= 2
    assert telemetry["classification_counts"][_FAILED] >= 1
    assert telemetry["classification_counts"][_CONTRIBUTING] >= 1
    assert telemetry["atomic_groups"] > 0
    assert telemetry["duration_ms"] >= 0
