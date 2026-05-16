"""Local runner for Hermes self-change benchmarks.

This runner is deliberately separate from pytest/CI. It invokes the live Hermes
CLI against temporary fixtures, records the transcript and diff, then runs
deterministic verifiers.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.self_change.tasks import (
    SUITES,
    TASKS,
    BenchmarkTask,
    SetupContext,
    selected_tasks,
    task_manifest,
)


DEFAULT_RESULTS_DIR = Path.home() / ".hermes" / "self-change-benchmarks"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_hermes() -> str:
    local_bin = Path.home() / ".local" / "bin" / "hermes"
    if local_bin.exists():
        return str(local_bin)
    repo_script = _repo_root() / "hermes"
    if repo_script.exists():
        return str(repo_script)
    return "hermes"


def _run(
    cmd: list[str],
    cwd: Path,
    timeout: int,
    env: dict[str, str],
) -> dict[str, Any]:
    started = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            env=env,
        )
        return {
            "cmd": cmd,
            "returncode": proc.returncode,
            "seconds": round(time.time() - started, 2),
            "output": proc.stdout,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout or ""
        if isinstance(out, bytes):
            out = out.decode("utf-8", errors="replace")
        return {
            "cmd": cmd,
            "returncode": 124,
            "seconds": round(time.time() - started, 2),
            "output": out + "\n[TIMEOUT]\n",
            "timed_out": True,
        }
    except OSError as exc:
        return {
            "cmd": cmd,
            "returncode": getattr(exc, "errno", 1) or 1,
            "seconds": round(time.time() - started, 2),
            "output": f"{exc.__class__.__name__}: {exc}\n",
            "timed_out": False,
        }


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _git_diff(path: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "diff", "--", "."],
            cwd=str(path),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
        )
        return proc.stdout
    except Exception as exc:
        return f"<failed to capture git diff: {exc}>"


def _restore_test_files(path: Path) -> None:
    """Keep verifier results independent from agent-added or edited tests."""
    try:
        proc = subprocess.run(
            ["git", "ls-files"],
            cwd=str(path),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
    except Exception:
        return
    if proc.returncode != 0:
        return
    tracked = {line.strip() for line in proc.stdout.splitlines() if line.strip()}
    test_files = [
        rel
        for rel in tracked
        if Path(rel).name.startswith("test") and Path(rel).suffix == ".py"
    ]
    if test_files:
        subprocess.run(
            ["git", "checkout", "--", *test_files],
            cwd=str(path),
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
    for candidate in path.rglob("test*.py"):
        rel = candidate.relative_to(path).as_posix()
        if rel not in tracked:
            try:
                candidate.unlink()
            except OSError:
                pass


def _build_hermes_cmd(args: argparse.Namespace, task: BenchmarkTask, prompt: str) -> list[str]:
    cmd = [args.hermes]
    if args.profile:
        cmd.extend(["--profile", args.profile])
    cmd.extend(["chat", "-Q", "--yolo", "--max-turns", str(_task_max_turns(args, task))])
    if args.force_fallback:
        # `hermes chat` honors CLI model/provider flags reliably. Its
        # interactive path does not use HERMES_INFERENCE_MODEL the same way
        # oneshot mode does, so do not rely on env-only routing for baselines.
        if args.force_fallback_provider:
            cmd.extend(["--provider", args.force_fallback_provider])
        if args.force_fallback_model:
            cmd.extend(["--model", args.force_fallback_model])
    if args.ignore_rules:
        cmd.append("--ignore-rules")
    if args.skills:
        cmd.extend(["-s", args.skills])
    cmd.extend(["-q", prompt])
    return cmd


def _runner_env(args: argparse.Namespace, run_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("TZ", "UTC")
    env.setdefault("LANG", "C.UTF-8")
    env.setdefault("LC_ALL", "C.UTF-8")
    env.setdefault("PYTHONHASHSEED", "0")
    env["HERMES_DISABLE_BACKGROUND_REVIEW"] = "1"
    env["HERMES_SELF_CHANGE_RUN_DIR"] = str(run_dir)
    env["HERMES_ORCHESTRATION_TASK_CONTEXT_PATH"] = str(run_dir / "task.json")
    return env


def _task_timeout_seconds(args: argparse.Namespace, task: BenchmarkTask) -> int:
    multiplier = max(args.timeout_multiplier, 1.0)
    return max(task.timeout_seconds, int(task.timeout_seconds * multiplier))


def _task_max_turns(args: argparse.Namespace, task: BenchmarkTask) -> int:
    multiplier = max(args.max_turns_multiplier, 1.0)
    return max(task.max_turns, int(task.max_turns * multiplier))


def run_task(args: argparse.Namespace, task: BenchmarkTask, base_dir: Path, ctx: SetupContext) -> dict[str, Any]:
    task_dir = base_dir / task.id
    workspace = task_dir / "workspace"
    if task_dir.exists():
        shutil.rmtree(task_dir)
    workspace.mkdir(parents=True)

    prompt = task.setup(workspace, ctx)
    timeout_seconds = _task_timeout_seconds(args, task)
    max_turns = _task_max_turns(args, task)
    _write_text(task_dir / "prompt.txt", prompt)
    _write_json(
        task_dir / "task.json",
        {
            "id": task.id,
            "category": task.category,
            "objective": task.objective,
            "fixture": task.fixture,
            "runner_contract": task.runner_contract,
            "pass_fail": task.pass_fail,
            "acceptance_criteria": task.pass_fail,
            "requirements": [task.fixture, task.runner_contract],
            "severity": task.severity,
            "blocks_auto_commit": task.blocks_auto_commit,
            "difficulty": task.difficulty,
            "timeout_seconds": task.timeout_seconds,
            "effective_timeout_seconds": timeout_seconds,
            "max_turns": task.max_turns,
            "effective_max_turns": max_turns,
            "tags": list(task.tags),
        },
    )

    env = _runner_env(args, task_dir)
    cmd = _build_hermes_cmd(args, task, prompt)
    print(f"== {task.id} ({task.difficulty}, {task.category}) ==", flush=True)
    print(f"   cwd={workspace}", flush=True)

    agent = _run(cmd, workspace, timeout_seconds, env)
    _write_text(task_dir / "agent_output.txt", agent["output"])

    if "calibration" in task.tags:
        _restore_test_files(workspace)

    verification = task.verify(workspace, ctx, agent["output"])
    diff = _git_diff(workspace)
    if diff:
        _write_text(task_dir / "diff.patch", diff)

    passed = agent["returncode"] == 0 and verification.passed
    result = {
        "id": task.id,
        "category": task.category,
        "difficulty": task.difficulty,
        "severity": task.severity,
        "blocks_auto_commit": task.blocks_auto_commit,
        "passed": passed,
        "agent_returncode": agent["returncode"],
        "agent_seconds": agent["seconds"],
        "agent_timed_out": agent["timed_out"],
        "timeout_seconds": timeout_seconds,
        "max_turns": max_turns,
        "verification": {
            "passed": verification.passed,
            "message": verification.message,
            "metrics": verification.metrics or {},
        },
        "run_dir": str(task_dir),
        "workspace": str(workspace),
        "command": agent["cmd"],
    }
    _write_json(task_dir / "result.json", result)
    print(f"   passed={passed} seconds={agent['seconds']} rc={agent['returncode']}", flush=True)
    if not passed:
        print(f"   verifier: {verification.message.splitlines()[0] if verification.message else '<empty>'}", flush=True)
    return result


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    blocking_failed = [
        r["id"]
        for r in results
        if not r["passed"] and r.get("blocks_auto_commit")
    ]
    by_category: dict[str, dict[str, int]] = {}
    for result in results:
        bucket = by_category.setdefault(result["category"], {"passed": 0, "total": 0})
        bucket["total"] += 1
        if result["passed"]:
            bucket["passed"] += 1
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "blocking_failed": blocking_failed,
        "by_category": by_category,
        "score": round(passed / total, 4) if total else 0.0,
    }


def _load_baseline(path: Path | None) -> dict[str, Any] | None:
    if not path or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _compare_baseline(results: list[dict[str, Any]], baseline: dict[str, Any] | None) -> dict[str, Any]:
    if not baseline:
        return {"available": False, "regressions": [], "improvements": []}
    current = {r["id"]: bool(r["passed"]) for r in results}
    old_results = baseline.get("results", [])
    old = {r["id"]: bool(r["passed"]) for r in old_results if "id" in r}
    regressions = sorted(task_id for task_id, was_passed in old.items() if was_passed and current.get(task_id) is False)
    improvements = sorted(task_id for task_id, was_passed in old.items() if not was_passed and current.get(task_id) is True)
    return {"available": True, "regressions": regressions, "improvements": improvements}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Hermes self-change regression benchmarks.")
    parser.add_argument("--suite", choices=sorted(SUITES), default="smoke")
    parser.add_argument("--task", action="append", default=[], help="Run one task id. May be repeated.")
    parser.add_argument("--list", action="store_true", help="List benchmark task metadata and exit.")
    parser.add_argument("--manifest", action="store_true", help="Print full JSON task manifest and exit.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--hermes", default=_default_hermes())
    parser.add_argument("--profile", default="", help="Optional Hermes profile. Empty means live default profile.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used by fixture verifiers.")
    parser.add_argument("--ignore-rules", action="store_true", help="Pass --ignore-rules to Hermes. Default keeps live prompts/rules.")
    parser.add_argument("--skills", default="", help="Optional comma-separated skill list to pass to Hermes.")
    parser.add_argument("--max-seconds", type=int, default=0, help="Global wall-clock budget. 0 means no global cap.")
    parser.add_argument(
        "--timeout-multiplier",
        type=float,
        default=1.0,
        help="Multiply each task timeout. Values below 1 are treated as 1.",
    )
    parser.add_argument(
        "--max-turns-multiplier",
        type=float,
        default=1.0,
        help="Multiply each task max-turn budget. Values below 1 are treated as 1.",
    )
    parser.add_argument("--baseline", type=Path, default=None, help="Optional previous result JSON to compare.")
    parser.add_argument("--write-baseline", type=Path, default=None, help="Write this run as a baseline JSON.")
    parser.add_argument("--force-fallback", action="store_true", help="Manually force the fallback provider/model via env.")
    parser.add_argument("--force-fallback-provider", default="openai")
    parser.add_argument("--force-fallback-model", default="gpt-5.4-mini")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.manifest:
        print(json.dumps(task_manifest(), indent=2, sort_keys=True))
        return 0
    if args.list:
        for task in task_manifest():
            tags = ",".join(task["tags"])
            print(
                f"{task['id']:<26} {task['category']:<26} {task['difficulty']:<8} "
                f"{task['severity']:<3} block={task['blocks_auto_commit']} tags={tags}"
            )
        return 0

    try:
        tasks = selected_tasks(args.suite, args.task or None)
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_name = args.run_name or f"{args.suite}-{stamp}"
    base_dir = args.results_dir / run_name
    base_dir.mkdir(parents=True, exist_ok=True)

    ctx = SetupContext(python_cmd=args.python)
    started = time.time()
    results: list[dict[str, Any]] = []
    budget_exceeded = False

    for task in tasks:
        if args.max_seconds and time.time() - started >= args.max_seconds:
            budget_exceeded = True
            print(f"global budget exceeded before {task.id}; stopping", flush=True)
            break
        result = run_task(args, task, base_dir, ctx)
        results.append(result)

    summary = _summarize(results)
    summary["budget_exceeded"] = budget_exceeded
    summary["seconds"] = round(time.time() - started, 2)
    baseline = _compare_baseline(results, _load_baseline(args.baseline))
    summary["baseline"] = baseline

    run_result = {
        "run_name": run_name,
        "suite": args.suite,
        "profile": args.profile or "<live-default>",
        "hermes": args.hermes,
        "python": args.python,
        "autonomy_variant": os.environ.get("HERMES_AUTONOMY_GUIDANCE_VARIANT", ""),
        "forced_provider": args.force_fallback_provider if args.force_fallback else "",
        "forced_model": args.force_fallback_model if args.force_fallback else "",
        "created_at_utc": stamp,
        "summary": summary,
        "results": results,
    }
    _write_json(base_dir / "summary.json", run_result)
    if args.write_baseline:
        _write_json(args.write_baseline, run_result)

    print("== summary ==", flush=True)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(str(base_dir), flush=True)

    if budget_exceeded:
        return 1
    if summary["blocking_failed"]:
        return 1
    if baseline["regressions"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
