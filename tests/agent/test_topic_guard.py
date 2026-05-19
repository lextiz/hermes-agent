import pytest

from agent.topic_guard import (
    FORCE_FALLBACK_ADVISORY,
    SUGGEST_ADVISORY,
    TopicGuard,
    TopicGuardConfig,
    build_topic_snapshot,
    parse_topic_guard_response,
)
from hermes_cli.config import DEFAULT_CONFIG


def _history(user_turns=4):
    messages = []
    for i in range(user_turns):
        messages.append({"role": "user", "content": f"Debug pytest failure {i}"})
        messages.append({"role": "assistant", "content": f"Investigated failure {i}"})
    return messages


class FakeClassifier:
    def __init__(self, response=None, exc=None):
        self.response = response
        self.exc = exc
        self.calls = 0
        self.last_messages = None
        self.last_model = None
        self.last_runtime = None

    def __call__(self, messages, model, runtime):
        self.calls += 1
        self.last_messages = messages
        self.last_model = model
        self.last_runtime = runtime
        if self.exc:
            raise self.exc
        return self.response


def _guard(config=None, classifier=None, force_new_session=None):
    return TopicGuard(
        TopicGuardConfig.from_mapping(config or {}),
        classifier=classifier,
        force_new_session=force_new_session,
        main_runtime_provider=lambda: {"model": "main-model", "provider": "custom"},
    )


def _drift_json(confidence=0.87):
    return (
        '{"drift": true, "confidence": %.2f, "old_topic": "debugging pytest", '
        '"new_topic": "planning a vacation", "reason": "travel planning is unrelated"}'
    ) % confidence


def test_default_config_keeps_topic_guard_disabled():
    cfg = TopicGuardConfig.from_config(DEFAULT_CONFIG)
    assert cfg.enabled is False
    assert cfg.mode == "suggest"
    assert cfg.min_turns == 4
    assert cfg.confidence_threshold == 0.75


def test_enabled_defaults_to_suggest_mode():
    cfg = TopicGuardConfig.from_mapping({"enabled": True})
    assert cfg.enabled is True
    assert cfg.mode == "suggest"


@pytest.mark.parametrize(
    ("config", "message", "history", "reason"),
    [
        ({}, "Plan a vacation to Japan next month", _history(5), "disabled"),
        ({"enabled": True, "min_turns": 4}, "Plan a vacation to Japan next month with hotels and trains", _history(3), "below_min_turns"),
        ({"enabled": True}, "Vacation?", _history(5), "short_message"),
        ({"enabled": True}, "/new travel planning", _history(5), "slash_command"),
    ],
)
def test_prefilters_skip_classifier(config, message, history, reason):
    classifier = FakeClassifier(_drift_json())
    decision = _guard(config, classifier).evaluate(
        user_message=message,
        conversation_history=history,
    )
    assert decision.fired is False
    assert decision.skipped_reason == reason
    assert classifier.calls == 0


def test_no_drift_continues_normally():
    classifier = FakeClassifier(
        '{"drift": false, "confidence": 0.92, "old_topic": "debugging", "new_topic": "debugging"}'
    )
    decision = _guard({"enabled": True, "min_turns": 1}, classifier).evaluate(
        user_message="The pytest failure is still happening after the fixture change",
        conversation_history=_history(2),
    )
    assert decision.fired is False
    assert decision.skipped_reason == "no_drift"
    assert classifier.calls == 1


def test_suggest_mode_advises_once_and_respects_cooldown():
    classifier = FakeClassifier(_drift_json())
    guard = _guard(
        {"enabled": True, "mode": "suggest", "min_turns": 1, "cooldown_turns": 3},
        classifier,
    )

    first = guard.evaluate(
        user_message="Plan a two week vacation to Japan with hotels and trains",
        conversation_history=_history(4),
    )
    second = guard.evaluate(
        user_message="Now plan another unrelated vacation to Greece with islands",
        conversation_history=_history(5),
    )

    assert first.fired is True
    assert first.action == "suggest"
    assert first.advisory == SUGGEST_ADVISORY
    assert second.fired is False
    assert second.skipped_reason == "cooldown"
    assert classifier.calls == 1


def test_force_mode_without_callback_falls_back_to_suggest():
    classifier = FakeClassifier(_drift_json())
    decision = _guard(
        {"enabled": True, "mode": "force", "min_turns": 1},
        classifier,
    ).evaluate(
        user_message="Plan a two week vacation to Japan with hotels and trains",
        conversation_history=_history(4),
    )
    assert decision.fired is True
    assert decision.action == "suggest"
    assert decision.force_fallback is True
    assert decision.advisory == FORCE_FALLBACK_ADVISORY


def test_force_mode_delegates_to_callback_when_supported():
    classifier = FakeClassifier(_drift_json())
    seen = []

    def force_new_session(classification):
        seen.append(classification)
        return True

    decision = _guard(
        {"enabled": True, "mode": "force", "min_turns": 1},
        classifier,
        force_new_session=force_new_session,
    ).evaluate(
        user_message="Plan a two week vacation to Japan with hotels and trains",
        conversation_history=_history(4),
    )
    assert decision.fired is True
    assert decision.action == "force"
    assert decision.started_new_session is True
    assert seen and seen[0].new_topic == "planning a vacation"


def test_bad_classifier_json_fails_open():
    classifier = FakeClassifier("```json\nnot-json\n```")
    decision = _guard({"enabled": True, "min_turns": 1}, classifier).evaluate(
        user_message="Plan a two week vacation to Japan with hotels and trains",
        conversation_history=_history(4),
    )
    assert decision.fired is False
    assert decision.skipped_reason == "parse_failed"


def test_model_failure_fails_open():
    classifier = FakeClassifier(exc=RuntimeError("offline"))
    decision = _guard({"enabled": True, "min_turns": 1}, classifier).evaluate(
        user_message="Plan a two week vacation to Japan with hotels and trains",
        conversation_history=_history(4),
    )
    assert decision.fired is False
    assert decision.skipped_reason == "parse_failed"


def test_parser_tolerates_fenced_json_and_noise():
    parsed = parse_topic_guard_response(
        'Here:\n```json\n{"drift": "true", "confidence": 1.4, "reason": "x"}\n```'
    )
    assert parsed is not None
    assert parsed.drift is True
    assert parsed.confidence == 1.0


def test_snapshot_is_compact_and_uses_recent_messages():
    snapshot = build_topic_snapshot(_history(10), session_title="pytest fixes", max_chars=500)
    assert "Session title: pytest fixes" in snapshot
    assert "Debug pytest failure 9" in snapshot
    assert len(snapshot) <= 500
