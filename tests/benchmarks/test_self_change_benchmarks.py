"""Static contract tests for the self-change benchmark catalog.

These tests do not invoke Hermes or any model. They keep the local benchmark
metadata valid so the cron eval gate can rely on it.
"""

import pytest

from benchmarks.self_change.tasks import (
    AGENTIC_TASKS,
    INSTRUCTION_TASKS,
    SUITES,
    TASKS,
    _strip_cli_session_prefix,
    selected_tasks,
    task_manifest,
)


def test_cli_session_strip_removes_transport_noise():
    output = "Warning: Unknown toolsets: hermes\n\nsession_id: abc\n{\"red\":2,\"blue\":1}\n"
    assert _strip_cli_session_prefix(output) == '{"red":2,"blue":1}'


def test_catalog_sizes_match_requested_bounds():
    assert 1 <= len(AGENTIC_TASKS) <= 20
    assert 1 <= len(INSTRUCTION_TASKS) <= 20
    assert len(TASKS) == len(AGENTIC_TASKS) + len(INSTRUCTION_TASKS)
    assert len(SUITES["nightly"]) == 20


def test_required_canaries_and_gate_membership():
    assert SUITES["smoke"] == ("self_edit_smoke", "prompt_stop_contract")
    assert "self_edit_smoke" in SUITES["gate"]
    assert "prompt_stop_contract" in SUITES["gate"]
    assert "processing_pipeline" in SUITES["gate"]
    assert "tool_call_extraction" in SUITES["gate"]


def test_all_tasks_have_eval_contract_metadata():
    for task in task_manifest():
        assert task["objective"]
        assert task["fixture"]
        assert task["runner_contract"]
        assert task["pass_fail"]
        assert task["severity"] in {"P0", "P1", "P2"}
        assert task["blocks_auto_commit"] is True
        assert task["difficulty"] in {"smoke", "easy", "medium", "hard"}
        assert task["timeout_seconds"] > 0
        assert task["max_turns"] > 0


def test_selected_tasks_rejects_unknown_task():
    with pytest.raises(KeyError):
        selected_tasks(task_ids=["does-not-exist"])
