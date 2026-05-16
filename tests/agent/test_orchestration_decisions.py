from types import SimpleNamespace

import pytest

from agent import orchestration_decisions as decisions
from hermes_cli.config import DEFAULT_CONFIG


def _response(content):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
            )
        ],
    )


def _enabled_config(kind="delegation"):
    return {
        "orchestration": {
            "enabled": True,
            kind: {"enabled": True, "max_tokens": 111, "timeout": 7},
        },
        "auxiliary": {
            "orchestration_delegate": {
                "provider": "openrouter",
                "model": "google/gemini-flash",
            },
            "orchestration_validation": {
                "provider": "nous",
                "model": "judge-model",
            },
            "orchestration_escalation": {
                "provider": "custom",
                "model": "local-judge",
            },
        },
    }


class TestParseDecisionJson:
    def test_parses_plain_json_object(self):
        assert decisions.parse_decision_json('{"delegate": true}') == {"delegate": True}

    def test_parses_fenced_json(self):
        raw = '```json\n{"valid": true, "issues": []}\n```'
        assert decisions.parse_decision_json(raw) == {"valid": True, "issues": []}

    def test_extracts_embedded_balanced_object(self):
        raw = 'Here: {"reason": "brace } in string", "escalate": false} done'
        assert decisions.parse_decision_json(raw) == {
            "reason": "brace } in string",
            "escalate": False,
        }

    def test_returns_none_for_non_object_or_invalid_json(self):
        assert decisions.parse_decision_json("[1, 2]") is None
        assert decisions.parse_decision_json("not json") is None


def test_orchestration_enabled_requires_global_and_kind_enable(monkeypatch):
    monkeypatch.setattr(decisions, "load_config", lambda: _enabled_config("validation"))

    assert decisions.orchestration_enabled("validation") is True
    assert decisions.orchestration_enabled("delegation") is False


def test_disabled_delegation_fails_open_without_call(monkeypatch):
    calls = []
    monkeypatch.setattr(decisions, "load_config", lambda: {"orchestration": {"enabled": False}})
    monkeypatch.setattr(decisions, "call_llm", lambda **kwargs: calls.append(kwargs))

    result = decisions.decide_delegation({"task": "build"})

    assert calls == []
    assert result["enabled"] is False
    assert result["called"] is False
    assert result["kind"] == "delegation"
    assert result["decision"]["delegate"] is False
    assert result["decision"]["tasks"] == []
    assert result["raw"] is None
    assert result["error"] is None
    assert result["parse_failed"] is False


def test_delegation_calls_dedicated_aux_task_and_returns_parsed_json(monkeypatch):
    seen = {}

    def fake_call_llm(**kwargs):
        seen.update(kwargs)
        return _response(
            '{"delegate": true, "tasks": [{"goal": "write tests", "role": "leaf"}], '
            '"roles": ["coder"], "model_tier_recommendations": ["cheap"], '
            '"reason": "parallelizable"}'
        )

    monkeypatch.setattr(decisions, "load_config", lambda: _enabled_config("delegation"))
    monkeypatch.setattr(decisions, "call_llm", fake_call_llm)

    result = decisions.decide_delegation({"task": "implement tests"})

    assert result["enabled"] is True
    assert result["called"] is True
    assert result["provider"] == "openrouter"
    assert result["model"] == "google/gemini-flash"
    assert result["decision"]["delegate"] is True
    assert result["decision"]["tasks"] == [{"goal": "write tests", "role": "leaf"}]
    assert result["decision"]["roles"] == ["coder"]
    assert result["decision"]["model_tier_recommendations"] == ["cheap"]
    assert result["decision"]["reason"] == "parallelizable"
    assert result["raw"].startswith("{")
    assert seen["task"] == "orchestration_delegate"
    assert seen["temperature"] == 0
    assert seen["max_tokens"] == 111
    assert seen["timeout"] == 7
    assert "JSON only" in seen["messages"][0]["content"]
    assert "Return delegate=false for small, single-file" in seen["messages"][1]["content"]
    assert "same workspace" in seen["messages"][1]["content"]


def test_validation_parse_failure_is_uncertain_when_enabled(monkeypatch):
    monkeypatch.setattr(decisions, "load_config", lambda: _enabled_config("validation"))
    monkeypatch.setattr(decisions, "call_llm", lambda **kwargs: _response("yes looks fine"))

    result = decisions.judge_validation({"summary": "done"})

    assert result["enabled"] is True
    assert result["called"] is True
    assert result["parse_failed"] is True
    assert result["decision_status"] == "parse_failed"
    assert result["decision"]["verdict"] == "uncertain"
    assert result["decision"]["repair_needed"] is True
    assert "invalid JSON" in result["decision"]["missing_checks"][0]
    assert result["error"] is None
    assert result["provider"] == "nous"
    assert result["model"] == "judge-model"


def test_validation_policy_requires_named_edge_case_evidence(monkeypatch):
    seen = {}

    def fake_call_llm(**kwargs):
        seen.update(kwargs)
        return _response(
            '{"verdict": "uncertain", "missing_checks": ["edge-case evidence"], '
            '"suggested_commands": [], "repair_needed": true, "reason": "missing edge coverage"}'
        )

    monkeypatch.setattr(decisions, "load_config", lambda: _enabled_config("validation"))
    monkeypatch.setattr(decisions, "call_llm", fake_call_llm)

    result = decisions.judge_validation({"task": "implement a parser", "acceptance_criteria": ["handle invalid input"]})

    assert result["decision"]["verdict"] == "uncertain"
    assert "stated requirements" in seen["messages"][1]["content"]
    assert "edge cases" in seen["messages"][1]["content"]


def test_decision_schema_is_normalized(monkeypatch):
    monkeypatch.setattr(decisions, "load_config", lambda: _enabled_config("delegation"))
    monkeypatch.setattr(
        decisions,
        "call_llm",
        lambda **kwargs: _response(
            '{"delegate": "yes", "tasks": [{"goal": "review", "role": "admin", "toolsets": ["file", 3]}]}'
        ),
    )

    result = decisions.decide_delegation({"task": "small fix"})

    assert result["decision_status"] == "parsed"
    assert result["decision"] == {
        "delegate": False,
        "tasks": [],
        "roles": [],
        "model_tier_recommendations": [],
        "reason": "",
    }


def test_escalation_call_error_fails_open_with_error(monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("network unavailable")

    monkeypatch.setattr(decisions, "load_config", lambda: _enabled_config("escalation"))
    monkeypatch.setattr(decisions, "call_llm", boom)

    result = decisions.decide_escalation({"risk": "low"})

    assert result["enabled"] is True
    assert result["called"] is True
    assert result["decision"]["escalate"] is False
    assert result["decision"]["retry_strategy"] == "continue_default"
    assert result["error"] == "network unavailable"
    assert result["parse_failed"] is False
    assert result["provider"] == "custom"
    assert result["model"] == "local-judge"


def test_default_config_registers_disabled_orchestration_and_aux_tasks():
    orchestration = DEFAULT_CONFIG["orchestration"]
    assert orchestration["enabled"] is False
    assert orchestration["delegation"]["enabled"] is False
    assert orchestration["validation"]["enabled"] is False
    assert orchestration["escalation"]["enabled"] is False
    assert orchestration["escalation"]["retry_delegated_tasks"] is True

    auxiliary = DEFAULT_CONFIG["auxiliary"]
    assert "orchestration_delegate" in auxiliary
    assert "orchestration_validation" in auxiliary
    assert "orchestration_escalation" in auxiliary
