import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import agent.tool_result_reducer as reducer_mod
from agent.tool_result_reducer import (
    ToolResultReducer,
    ToolResultReductionConfig,
    _REDUCER_ACTIVE,
    recent_user_intent_from_messages,
)


def _cfg(**overrides):
    values = {
        "enabled": True,
        "min_chars": 1000,
        "model": None,
        "preserve_raw_reference": False,
        "max_reduced_chars": 4000,
        "include_tool_args": True,
        "include_recent_user_intent": True,
        "excluded_tools": (),
        "included_tools": ("*",),
    }
    values.update(overrides)
    return ToolResultReductionConfig(**values)


def _response(text):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(content=text)),
        ],
    )


def _reducer(monkeypatch, config):
    reducer = ToolResultReducer()
    monkeypatch.setattr(reducer, "_load_config", lambda: config)
    return reducer


def test_config_disabled_long_output_passes_through(monkeypatch):
    reducer = _reducer(monkeypatch, _cfg(enabled=False))
    monkeypatch.setattr(
        reducer_mod,
        "call_llm",
        lambda **_: pytest.fail("reducer model should not be called"),
    )

    result = "x" * 5000
    assert reducer.reduce(tool_name="terminal", tool_args={}, result=result) == result


def test_below_threshold_passes_through(monkeypatch):
    reducer = _reducer(monkeypatch, _cfg(min_chars=2000))
    monkeypatch.setattr(
        reducer_mod,
        "call_llm",
        lambda **_: pytest.fail("reducer model should not be called"),
    )

    result = "x" * 1500
    assert reducer.reduce(tool_name="terminal", tool_args={}, result=result) == result


def test_above_threshold_calls_model_and_returns_reduced_output(monkeypatch):
    reducer = _reducer(monkeypatch, _cfg(model="cheap-model"))
    calls = []

    def fake_call_llm(**kwargs):
        calls.append(kwargs)
        return _response(
            "Relevant findings:\n- build succeeded\n\nOmitted:\n- repeated progress logs"
        )

    monkeypatch.setattr(reducer_mod, "call_llm", fake_call_llm)

    result = reducer.reduce(
        tool_name="terminal",
        tool_args={"command": "pytest tests/unit"},
        result="progress\n" + ("x" * 1500),
        tool_call_id="call_1",
        recent_user_intent="Fix the failing unit tests.",
    )

    assert calls
    assert calls[0]["task"] == "tool_result_reduction"
    assert calls[0]["model"] == "cheap-model"
    assert result.startswith("[Tool result reduced from")
    assert "Tool: terminal" in result
    assert "Query/command: pytest tests/unit" in result
    assert "Relevant findings" in result
    assert len(result) < 1509


def test_reducer_failure_falls_back_to_original(monkeypatch):
    reducer = _reducer(monkeypatch, _cfg())

    def fake_call_llm(**_kwargs):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(reducer_mod, "call_llm", fake_call_llm)
    result = "x" * 1500

    assert reducer.reduce(tool_name="terminal", tool_args={}, result=result) == result


def test_prompt_contains_tool_args_and_recent_user_intent(monkeypatch):
    reducer = _reducer(monkeypatch, _cfg())
    captured = {}

    def fake_call_llm(**kwargs):
        captured.update(kwargs)
        return _response("Relevant findings:\n- found TODO\n\nOmitted:\n- noise")

    monkeypatch.setattr(reducer_mod, "call_llm", fake_call_llm)

    reducer.reduce(
        tool_name="search_files",
        tool_args={"query": "TODO", "path": "src"},
        result="match\n" + ("x" * 1500),
        recent_user_intent="Find TODOs in the source tree.",
    )

    prompt = captured["messages"][1]["content"]
    assert "Tool name: search_files" in prompt
    assert '"query": "TODO"' in prompt
    assert "Recent user intent: Find TODOs in the source tree." in prompt


def test_important_error_lines_are_preserved_when_model_omits_them(monkeypatch):
    reducer = _reducer(monkeypatch, _cfg())
    monkeypatch.setattr(
        reducer_mod,
        "call_llm",
        lambda **_: _response("Relevant findings:\n- tests failed\n\nOmitted:\n- setup logs"),
    )
    raw = "\n".join(
        [
            "setup noise",
            "Traceback (most recent call last):",
            'File "/repo/app.py", line 42, in run',
            "ValueError: bad config",
            "noise " + ("x" * 1500),
        ]
    )

    result = reducer.reduce(tool_name="terminal", tool_args={"command": "pytest"}, result=raw)

    assert "Traceback (most recent call last):" in result
    assert 'File "/repo/app.py", line 42, in run' in result
    assert "ValueError: bad config" in result


def test_terminal_json_status_and_output_are_sent_to_prompt(monkeypatch):
    reducer = _reducer(monkeypatch, _cfg())
    captured = {}

    def fake_call_llm(**kwargs):
        captured.update(kwargs)
        return _response("Relevant findings:\n- pytest failed\n\nOmitted:\n- repeated logs")

    monkeypatch.setattr(reducer_mod, "call_llm", fake_call_llm)
    terminal_result = json.dumps(
        {
            "output": "FAILED tests/test_app.py::test_login\nERROR auth.py:17 bad token\n"
            + ("log\n" * 400),
            "exit_code": 1,
            "error": None,
        }
    )

    result = reducer.reduce(
        tool_name="terminal",
        tool_args={"command": "pytest"},
        result=terminal_result,
    )

    prompt = captured["messages"][1]["content"]
    assert "Exit/status: exit_code=1" in prompt
    assert "FAILED tests/test_app.py::test_login" in prompt
    assert "ERROR auth.py:17 bad token" in prompt
    assert "Summarized text result; original tool result was JSON." in result


def test_raw_reference_is_included_when_storage_succeeds(monkeypatch):
    reducer = _reducer(monkeypatch, _cfg(preserve_raw_reference=True))
    monkeypatch.setattr(
        reducer_mod,
        "call_llm",
        lambda **_: _response("Relevant findings:\n- useful line\n\nOmitted:\n- noise"),
    )
    env = MagicMock()
    env.get_temp_dir.return_value = "/tmp"
    env.execute.return_value = {"output": "", "returncode": 0}

    result = reducer.reduce(
        tool_name="terminal",
        tool_args={"command": "make"},
        result="x" * 1500,
        tool_call_id="call/raw:1",
        env=env,
    )

    assert "Raw output saved to: /tmp/hermes-results/call_raw_1.raw.txt" in result
    assert env.execute.call_args.kwargs["stdin_data"] == "x" * 1500


def test_no_reducer_recursion(monkeypatch):
    reducer = _reducer(monkeypatch, _cfg())
    monkeypatch.setattr(
        reducer_mod,
        "call_llm",
        lambda **_: pytest.fail("reducer model should not be called while active"),
    )
    token = _REDUCER_ACTIVE.set(True)
    try:
        result = "x" * 1500
        assert reducer.reduce(tool_name="terminal", tool_args={}, result=result) == result
    finally:
        _REDUCER_ACTIVE.reset(token)


def test_recent_user_intent_extracts_only_last_user_message():
    messages = [
        {"role": "user", "content": "older request"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": [{"type": "text", "text": "current request"}]},
    ]

    assert recent_user_intent_from_messages(messages) == "current request"
