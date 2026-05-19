from unittest.mock import patch

from agent.topic_guard import TopicDriftClassification, TopicGuardDecision
from run_agent import AIAgent


class FakeTopicGuard:
    def __init__(self, decision):
        self.decision = decision
        self.calls = []

    def evaluate(self, **kwargs):
        self.calls.append(kwargs)
        return self.decision


class FakeSessionDB:
    def __init__(self):
        self.ended = []
        self.created = []

    def end_session(self, session_id, reason):
        self.ended.append((session_id, reason))

    def create_session(self, **kwargs):
        self.created.append(kwargs)

    def get_session_title(self, session_id):
        return ""


def _make_agent():
    with patch("run_agent.OpenAI"), patch("hermes_cli.config.load_config", return_value={}):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.suppress_status_output = True
    return agent


def test_topic_guard_hook_emits_advisory_through_status_callback():
    agent = _make_agent()
    guard = FakeTopicGuard(
        TopicGuardDecision(fired=True, action="suggest", advisory="topic changed")
    )
    agent._topic_guard = guard
    events = []
    agent.status_callback = lambda kind, message: events.append((kind, message))

    started = agent._maybe_apply_topic_guard(
        "Plan a vacation to Japan with hotels and trains",
        [{"role": "user", "content": "debug pytest"}],
    )

    assert started is False
    assert events == [("lifecycle", "topic changed")]
    assert guard.calls[0]["user_message"].startswith("Plan a vacation")


def test_topic_guard_hook_reports_successful_force_session_start():
    agent = _make_agent()
    agent._topic_guard = FakeTopicGuard(
        TopicGuardDecision(
            fired=True,
            action="force",
            advisory="started a new session",
            started_new_session=True,
        )
    )

    assert agent._maybe_apply_topic_guard("Plan a vacation", []) is True


def test_topic_guard_force_mode_rotates_agent_session_with_db():
    agent = _make_agent()
    db = FakeSessionDB()
    agent._session_db = db
    old_session_id = agent.session_id

    started = agent._topic_guard_force_new_session(
        TopicDriftClassification(
            drift=True,
            confidence=0.9,
            old_topic="debugging pytest",
            new_topic="planning travel",
        )
    )

    assert started is True
    assert agent.session_id != old_session_id
    assert db.ended == [(old_session_id, "topic_guard")]
    assert db.created[-1]["session_id"] == agent.session_id
    assert db.created[-1]["parent_session_id"] == old_session_id
    assert agent._last_flushed_db_idx == 0
