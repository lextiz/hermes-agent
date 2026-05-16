import json

import agent.orchestration_decisions as decisions
import run_agent
from run_agent import AIAgent


def _agent(*, enabled=True, depth=0, tools=None):
    agent = object.__new__(AIAgent)
    agent._orchestration_cfg = {
        "enabled": enabled,
        "delegation": {
            "enabled": enabled,
            "max_cost_usd": 0.25,
            "max_latency_seconds": 20,
            "prefer": "cheap_parallel",
        }
    }
    agent._delegate_depth = depth
    agent.valid_tool_names = set(tools or {"delegate_task", "read_file", "web_search"})
    agent.provider = "openrouter"
    agent.model = "anthropic/claude-sonnet-4.6"
    agent.status_callback = None
    agent.log_prefix = ""
    agent._emit_status = lambda message: None
    agent._tool_guardrail_halt_decision = None
    return agent


def test_validation_wrapper_failure_is_uncertain(monkeypatch):
    def boom(packet):
        raise RuntimeError("judge offline")

    monkeypatch.setattr(decisions, "judge_validation", boom)

    result = run_agent.judge_validation({"task": "ship change"})

    assert result["verdict"] == "uncertain"
    assert result["repair_needed"] is True
    assert "judge unavailable" in result["missing_checks"][0]
    assert result["error"] == "judge offline"


def test_pre_turn_delegation_disabled_does_not_call_judge(monkeypatch):
    agent = _agent(enabled=False)

    def fail_if_called(packet):
        raise AssertionError("judge should not be called when config is disabled")

    monkeypatch.setattr(run_agent, "decide_delegation", fail_if_called)

    assert agent._maybe_run_pre_turn_delegation("please split this up") == ""


def test_pre_turn_delegation_skips_subagents(monkeypatch):
    agent = _agent(depth=1)

    def fail_if_called(packet):
        raise AssertionError("judge should not be called inside a subagent")

    monkeypatch.setattr(run_agent, "decide_delegation", fail_if_called)

    assert agent._maybe_run_pre_turn_delegation("please split this up") == ""


def test_pre_turn_delegation_calls_judge_and_delegate_task(monkeypatch):
    agent = _agent()
    packets = []
    delegate_calls = []
    tasks = [
        {"goal": "Inspect the failing tests", "toolsets": ["search"]},
        {"goal": "Summarize the risk", "toolsets": ["safe"]},
    ]

    def fake_decide(packet):
        packets.append(packet)
        return {"delegate": True, "tasks": tasks}

    def fake_delegate_task(**kwargs):
        delegate_calls.append(kwargs)
        return json.dumps(
            {
                "results": [
                    {"goal": tasks[0]["goal"], "status": "success", "summary": "Found one narrow hook."},
                    {"goal": tasks[1]["goal"], "status": "success", "summary": "Low risk."},
                ],
                "total_duration_seconds": 1.2,
            }
        )

    monkeypatch.setattr(run_agent, "decide_delegation", fake_decide)
    monkeypatch.setattr("tools.delegate_tool.delegate_task", fake_delegate_task)

    context = agent._maybe_run_pre_turn_delegation("Find the smallest safe implementation")

    assert packets == [
        {
            "user_task": "Find the smallest safe implementation",
            "available_tools": ["delegate_task", "read_file", "web_search"],
            "runtime": {
                "provider": "openrouter",
                "model": "anthropic/claude-sonnet-4.6",
            },
            "cost_latency_policy": {
                "max_cost_usd": 0.25,
                "max_latency_seconds": 20,
                "prefer": "cheap_parallel",
            },
            "task_hints": {},
            "acceptance_hints": [],
        }
    ]
    assert delegate_calls
    prepared_tasks = delegate_calls[0]["tasks"]
    assert prepared_tasks is not tasks
    assert [task["goal"] for task in prepared_tasks] == [task["goal"] for task in tasks]
    assert "Original user task and constraints" in prepared_tasks[0]["context"]
    assert "Find the smallest safe implementation" in prepared_tasks[0]["context"]
    assert "The parent agent owns final edits" in prepared_tasks[0]["context"]
    assert prepared_tasks[0]["acceptance_criteria"] == "Find the smallest safe implementation"
    assert "context" not in tasks[0]
    assert delegate_calls[0]["parent_agent"] is agent
    assert delegate_calls[0]["acceptance_criteria"] == "Find the smallest safe implementation"
    assert "<orchestration-delegation-context>" in context
    assert "Found one narrow hook." in context
    assert "Low risk." in context


def test_hard_task_delegation_uses_configured_escalation_target():
    agent = _agent()
    agent._orchestration_cfg = {
        "enabled": True,
        "delegation": {"enabled": True},
        "escalation": {
            "enabled": True,
            "target_provider": "openai",
            "target_model": "gpt-5.5",
        },
    }

    prepared = agent._prepare_orchestration_delegation_tasks(
        [{"goal": "Review JSON Patch edge cases"}],
        "Implement JSON Patch.",
        {"difficulty": "hard", "tags": ["nightly", "hard"]},
    )

    assert prepared[0]["provider"] == "openai"
    assert prepared[0]["model"] == "gpt-5.5"
    assert "high difficulty or elevated verifier risk" in prepared[0]["context"]


def test_task_metadata_acceptance_hints_are_added_to_packets_and_delegation_context(monkeypatch):
    agent = _agent()
    task_hints = {
        "id": "acceptance_task",
        "difficulty": "hard",
        "pass_fail": "Task must satisfy its edge cases.",
        "acceptance_hints": [
            "Validate zero-length ranges and boundary insertions.",
            "Validate state-history behavior after reopen transitions.",
        ],
    }
    monkeypatch.setattr(agent, "_orchestration_task_hints", lambda: task_hints)

    packet = agent._build_delegation_decision_packet("Implement a domain-specific task")

    assert packet["task_hints"] == task_hints
    assert packet["acceptance_hints"] == task_hints["acceptance_hints"]

    prepared = agent._prepare_orchestration_delegation_tasks(
        [{"goal": "Review hidden edge cases"}],
        "Implement a domain-specific task",
        task_hints,
    )

    assert "Task-provided acceptance checks" in prepared[0]["context"]
    assert "Validate zero-length ranges" in prepared[0]["context"]
    assert "Validate state-history behavior" in prepared[0]["context"]


def test_pre_turn_delegation_judge_failure_fails_open(monkeypatch):
    agent = _agent()

    def fake_decide(packet):
        raise RuntimeError("judge unavailable")

    def fail_delegate_task(**kwargs):
        raise AssertionError("delegate_task should not run after judge failure")

    monkeypatch.setattr(run_agent, "decide_delegation", fake_decide)
    monkeypatch.setattr("tools.delegate_tool.delegate_task", fail_delegate_task)

    assert agent._maybe_run_pre_turn_delegation("do work") == ""


def test_pre_turn_delegation_requires_delegate_tool(monkeypatch):
    agent = _agent(tools={"read_file", "web_search"})

    def fail_if_called(packet):
        raise AssertionError("judge should not be called without delegate_task")

    monkeypatch.setattr(run_agent, "decide_delegation", fail_if_called)

    assert agent._maybe_run_pre_turn_delegation("do work") == ""


def test_turn_validation_calls_judge_with_compact_packet(monkeypatch):
    agent = _agent()
    agent._orchestration_cfg = {
        "enabled": True,
        "validation": {"enabled": True},
    }
    packets = []

    def fake_judge(packet):
        packets.append(packet)
        return {
            "called": True,
            "model": "judge-model",
            "provider": "openai",
            "decision": {
                "verdict": "pass",
                "repair_needed": False,
                "missing_checks": [],
                "suggested_commands": [],
            },
        }

    monkeypatch.setattr(run_agent, "judge_validation", fake_judge)
    monkeypatch.setattr(
        agent,
        "_git_evidence_for_validation",
        lambda: {"cwd": "/tmp/work", "status": " M app.py"},
    )
    messages = [
        {"role": "user", "content": "Fix the failing test"},
        {"role": "assistant", "tool_calls": [{"id": "1", "function": {"name": "terminal"}}]},
        {"role": "tool", "name": "terminal", "content": "pytest passed"},
        {"role": "assistant", "content": "Done"},
    ]

    result = agent._maybe_run_turn_validation(
        "Fix the failing test",
        "Done",
        messages,
        "text_response(finish_reason=stop)",
        1,
    )

    assert result["decision"]["verdict"] == "pass"
    assert packets[0]["task"] == "Fix the failing test"
    assert packets[0]["acceptance_criteria"] == "Fix the failing test"
    assert packets[0]["final_response"] == "Done"
    assert packets[0]["turn"]["tool_turns"] == 1
    assert packets[0]["tool_results"] == [{"name": "terminal", "content": "pytest passed"}]
    assert packets[0]["workspace"]["status"] == " M app.py"
    assert packets[0]["task_hints"] == {}


def test_turn_validation_skips_subagents(monkeypatch):
    agent = _agent(depth=1)
    agent._orchestration_cfg = {
        "enabled": True,
        "validation": {"enabled": True},
    }

    def fail_if_called(packet):
        raise AssertionError("validation should not run inside a subagent")

    monkeypatch.setattr(run_agent, "judge_validation", fail_if_called)

    assert agent._maybe_run_turn_validation("task", "done", [], "text_response", 0) == {}


def test_validation_changed_file_contents_include_untracked_file(tmp_path):
    agent = _agent()
    (tmp_path / "report.json").write_text('{"north_total": 17}\n', encoding="utf-8")

    changed = agent._changed_file_contents_for_validation(tmp_path, "?? report.json\n")

    assert changed == [
        {"path": "report.json", "content": '{"north_total": 17}'},
    ]


def test_orchestration_task_hints_reads_configured_task_json(tmp_path, monkeypatch):
    task_dir = tmp_path / "task"
    workspace = task_dir / "workspace"
    workspace.mkdir(parents=True)
    (task_dir / "task.json").write_text(
        json.dumps(
            {
                "id": "complex_local_task",
                "difficulty": "hard",
                "tags": ["nightly", "hard"],
                "acceptance_criteria": "All operations work.",
                "acceptance_hints": ["Validate root path operations."],
                "ignored_large_field": "not needed",
            }
        ),
        encoding="utf-8",
    )
    agent = _agent()
    agent.session_cwd = str(workspace)
    monkeypatch.setenv("HERMES_ORCHESTRATION_TASK_CONTEXT_PATH", str(task_dir / "task.json"))

    hints = agent._orchestration_task_hints()

    assert hints["id"] == "complex_local_task"
    assert hints["difficulty"] == "hard"
    assert hints["tags"] == ["nightly", "hard"]
    assert hints["acceptance_criteria"] == "All operations work."
    assert hints["acceptance_hints"] == ["Validate root path operations."]
    assert "ignored_large_field" not in hints


def test_turn_escalation_skips_when_validation_passes(monkeypatch):
    agent = _agent()
    agent._orchestration_cfg = {
        "enabled": True,
        "escalation": {"enabled": True, "retry_main_turns": True},
    }
    agent.max_iterations = 10

    def fail_if_called(packet):
        raise AssertionError("escalation judge should not run for a passing turn")

    monkeypatch.setattr(run_agent, "decide_escalation", fail_if_called)

    result = agent._maybe_run_turn_escalation(
        "Fix the bug",
        "Done",
        [{"role": "assistant", "content": "Done"}],
        "text_response(finish_reason=stop)",
        1,
        {"decision": {"verdict": "pass", "repair_needed": False}},
    )

    assert result == {}


def test_turn_escalation_skips_uncertain_when_judge_says_no_repair(monkeypatch):
    agent = _agent()
    agent._orchestration_cfg = {
        "enabled": True,
        "escalation": {"enabled": True, "retry_main_turns": True},
    }
    agent.max_iterations = 10

    def fail_if_called(packet):
        raise AssertionError("escalation judge should not run when repair is explicitly unnecessary")

    monkeypatch.setattr(run_agent, "decide_escalation", fail_if_called)

    result = agent._maybe_run_turn_escalation(
        "Fix the bug",
        "Done",
        [{"role": "assistant", "content": "Done"}],
        "guardrail_halt",
        1,
        {"decision": {"verdict": "uncertain", "repair_needed": False}},
    )

    assert result == {}
    assert agent._validation_envelope_needs_repair(
        {"decision": {"verdict": "uncertain", "repair_needed": False}}
    ) is False
    assert agent._validation_envelope_needs_repair(
        {"decision": {"verdict": "uncertain", "repair_needed": True}}
    ) is True


def test_turn_escalation_dispatches_compact_retry_on_validation_failure(monkeypatch):
    agent = _agent()
    agent._orchestration_cfg = {
        "enabled": True,
        "escalation": {
            "enabled": True,
            "retry_main_turns": True,
            "target_provider": "openai",
            "target_model": "gpt-5.5",
        },
    }
    agent.max_iterations = 10
    packets = []
    dispatches = []

    def fake_decide(packet):
        packets.append(packet)
        return {
            "called": True,
            "model": "router",
            "provider": "openai",
            "decision": {
                "escalate": True,
                "provider": "openai",
                "model": "gpt-5.4-mini",
                "retry_strategy": "repair",
                "compact_prompt": "Repair the missing report.json file.",
            },
        }

    def fake_dispatch(args):
        dispatches.append(args)
        return '{"status": "success", "summary": "report.json repaired"}'

    monkeypatch.setattr(run_agent, "decide_escalation", fake_decide)
    monkeypatch.setattr(agent, "_dispatch_delegate_task", fake_dispatch)
    monkeypatch.setattr(
        agent,
        "_git_evidence_for_validation",
        lambda: {"cwd": "/tmp/work", "status": "?? report.json"},
    )

    result = agent._maybe_run_turn_escalation(
        "Create report.json",
        "Done",
        [{"role": "tool", "name": "terminal", "content": "missing report.json"}],
        "text_response(finish_reason=stop)",
        1,
        {
            "decision": {
                "verdict": "fail",
                "repair_needed": True,
                "missing_checks": ["report.json was not created"],
            }
        },
    )

    assert packets[0]["task"] == "Create report.json"
    assert packets[0]["verifier_result"]["decision"]["verdict"] == "fail"
    assert dispatches == [
        {
            "goal": "Repair the missing report.json file.",
            "context": dispatches[0]["context"],
            "acceptance_criteria": "Create report.json",
            "provider": "openai",
            "model": "gpt-5.5",
            "allow_escalation_retry": False,
        }
    ]
    assert "Quality escalation retry" in dispatches[0]["context"]
    assert "do not call bare python" in dispatches[0]["context"]
    assert "do not call apply_patch as a terminal command" in dispatches[0]["context"]
    assert "missing report.json" in dispatches[0]["context"]
    assert "report.json repaired" in result["retry_result"]


def test_turn_escalation_revalidates_retry_and_runs_second_round(monkeypatch):
    agent = _agent()
    agent._orchestration_cfg = {
        "enabled": True,
        "validation": {"enabled": True},
        "escalation": {
            "enabled": True,
            "retry_main_turns": True,
            "max_retry_rounds": 2,
            "target_provider": "openai",
            "target_model": "gpt-5.4",
        },
    }
    agent.max_iterations = 10
    decisions = []
    dispatches = []
    validations = []

    def fake_decide(packet):
        decisions.append(packet)
        return {
            "called": True,
            "model": "router",
            "provider": "openai",
            "decision": {
                "escalate": True,
                "provider": "openai",
                "model": "gpt-5.4-mini",
                "compact_prompt": f"Repair round {len(decisions)}.",
            },
        }

    def fake_dispatch(args):
        dispatches.append(args)
        return f"retry result {len(dispatches)}"

    def fake_judge(packet):
        validations.append(packet)
        round_index = packet.get("post_escalation", {}).get("round")
        if round_index == 1:
            return {
                "called": True,
                "model": "judge",
                "provider": "openai",
                "decision": {
                    "verdict": "uncertain",
                    "repair_needed": True,
                    "missing_checks": ["hidden edge case still lacks evidence"],
                },
            }
        return {
            "called": True,
            "model": "judge",
            "provider": "openai",
            "decision": {"verdict": "pass", "repair_needed": False},
        }

    monkeypatch.setattr(run_agent, "decide_escalation", fake_decide)
    monkeypatch.setattr(run_agent, "judge_validation", fake_judge)
    monkeypatch.setattr(agent, "_dispatch_delegate_task", fake_dispatch)
    monkeypatch.setattr(
        agent,
        "_git_evidence_for_validation",
        lambda: {"cwd": "/tmp/work", "status": " M diff_apply.py"},
    )

    result = agent._maybe_run_turn_escalation(
        "Implement the patch application routine",
        "Done",
        [{"role": "tool", "name": "terminal", "content": "visible tests pass"}],
        "text_response(finish_reason=stop)",
        1,
        {"decision": {"verdict": "uncertain", "repair_needed": True}},
    )

    assert [packet["escalation_round"] for packet in decisions] == [1, 2]
    assert len(dispatches) == 2
    assert "Repair round 1" in dispatches[0]["goal"]
    assert "Repair round 2" in dispatches[1]["goal"]
    assert [dispatch["model"] for dispatch in dispatches] == ["gpt-5.4", "gpt-5.4"]
    assert [dispatch["allow_escalation_retry"] for dispatch in dispatches] == [False, False]
    assert [packet["post_escalation"]["round"] for packet in validations] == [1, 2]
    assert [packet["tool_results"] for packet in validations] == [
        [{"name": "escalation_retry_result", "content": "retry result 1"}],
        [{"name": "escalation_retry_result", "content": "retry result 2"}],
    ]
    assert result["retry_result"] == "retry result 2"
    assert len(result["rounds"]) == 2
    assert result["rounds"][0]["post_retry_validation"]["decision"]["verdict"] == "uncertain"
    assert result["rounds"][1]["post_retry_validation"]["decision"]["verdict"] == "pass"


def test_post_success_guard_blocks_new_file_mutation_after_verifier_passes():
    agent = _agent()
    agent._orchestration_cfg = {
        "enabled": True,
        "validation": {"enabled": True, "freeze_after_successful_check": True},
    }
    agent._post_success_changed_paths = set()
    agent._post_success_allowed_mutation_paths = set()
    agent._post_success_mutation_guard_seen = False

    agent._record_post_success_mutation_state(
        "patch",
        {"mode": "replace", "path": "summarize_cli.py"},
        '{"success": true}',
        False,
    )
    agent._record_post_success_mutation_state(
        "terminal",
        {"command": "/venv/bin/python -m unittest -v"},
        '{"exit_code": 0, "output": "OK"}',
        False,
    )

    assert agent._post_success_mutation_block_message(
        "write_file",
        {"path": "sample.txt", "content": "alpha beta"},
    )
    assert agent._tool_guardrail_halt_decision.code == "post_success_mutation_block"
    assert agent._tool_guardrail_halt_decision.count == 1
    assert agent._post_success_mutation_block_message(
        "terminal",
        {"command": "rm sample.txt"},
    )
    assert agent._tool_guardrail_halt_decision.count == 1
    assert agent._post_success_mutation_block_message(
        "patch",
        {"mode": "replace", "path": "summarize_cli.py"},
    ) is None


def test_post_success_guard_disabled_by_default():
    agent = _agent()
    agent._orchestration_cfg = {
        "enabled": True,
        "validation": {"enabled": True},
    }
    agent._post_success_mutation_guard_seen = True
    agent._post_success_allowed_mutation_paths = set()

    assert agent._post_success_mutation_block_message(
        "write_file",
        {"path": "sample.txt", "content": "alpha beta"},
    ) is None


def test_post_success_guard_skips_when_execute_code_mutation_history_is_unknown():
    agent = _agent()
    agent._orchestration_cfg = {
        "enabled": True,
        "validation": {"enabled": True, "freeze_after_successful_check": True},
    }
    agent._post_success_changed_paths = set()
    agent._post_success_allowed_mutation_paths = set()
    agent._post_success_mutation_guard_seen = False
    agent._post_success_mutation_history_unknown = False

    agent._record_post_success_mutation_state(
        "execute_code",
        {"code": "from pathlib import Path\nPath('solver.py').write_text('repair')"},
        '{"status": "ok", "tool_calls_made": 0}',
        False,
    )
    agent._record_post_success_mutation_state(
        "terminal",
        {"command": "/venv/bin/python -m unittest -v"},
        '{"exit_code": 0, "output": "OK"}',
        False,
    )

    assert agent._post_success_mutation_guard_seen is False
    assert agent._post_success_mutation_block_message(
        "write_file",
        {"path": "solver.py", "content": "repair again"},
    ) is None


def test_post_success_guard_still_arms_after_read_only_execute_code():
    agent = _agent()
    agent._orchestration_cfg = {
        "enabled": True,
        "validation": {"enabled": True, "freeze_after_successful_check": True},
    }
    agent._post_success_changed_paths = set()
    agent._post_success_allowed_mutation_paths = set()
    agent._post_success_mutation_guard_seen = False
    agent._post_success_mutation_history_unknown = False

    agent._record_post_success_mutation_state(
        "execute_code",
        {"code": "print('inspection only')"},
        '{"status": "ok", "tool_calls_made": 0}',
        False,
    )
    agent._record_post_success_mutation_state(
        "terminal",
        {"command": "/venv/bin/python -m unittest -v"},
        '{"exit_code": 0, "output": "OK"}',
        False,
    )

    assert agent._post_success_mutation_guard_seen is True
    assert agent._post_success_mutation_block_message(
        "write_file",
        {"path": "new_file.py", "content": "late edit"},
    )


def test_post_success_guard_skips_subagents():
    agent = _agent(depth=1)
    agent._orchestration_cfg = {
        "enabled": True,
        "validation": {"enabled": True, "freeze_after_successful_check": True},
    }
    agent._post_success_mutation_guard_seen = True
    agent._post_success_allowed_mutation_paths = set()

    assert agent._post_success_mutation_block_message(
        "write_file",
        {"path": "jsonpatcher.py", "content": "repair"},
    ) is None
