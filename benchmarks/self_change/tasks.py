"""Task catalog for Hermes' self-change regression benchmarks.

The tasks here are intentionally local and fixture-backed. The runner invokes a
live Hermes profile against these fixtures, but verification is deterministic:
each task either leaves the workspace in the expected state or it does not.
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import sqlite3
import stat
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


@dataclass(frozen=True)
class SetupContext:
    python_cmd: str


@dataclass(frozen=True)
class Verification:
    passed: bool
    message: str
    metrics: dict[str, object] | None = None


SetupFn = Callable[[Path, SetupContext], str]
VerifyFn = Callable[[Path, SetupContext, str], Verification]


@dataclass(frozen=True)
class BenchmarkTask:
    id: str
    category: str
    objective: str
    fixture: str
    runner_contract: str
    pass_fail: str
    severity: str
    blocks_auto_commit: bool
    difficulty: str
    timeout_seconds: int
    max_turns: int
    tags: tuple[str, ...]
    setup: SetupFn
    verify: VerifyFn


def _dedent(text: str) -> str:
    return textwrap.dedent(text).strip() + "\n"


def _write(path: Path, text: str, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_dedent(text), encoding="utf-8")
    if mode is not None:
        path.chmod(mode)


def _run(
    cmd: list[str],
    cwd: Path,
    timeout: int = 60,
    env: dict[str, str] | None = None,
) -> tuple[bool, str]:
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
        return proc.returncode == 0, proc.stdout
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout or ""
        if isinstance(out, bytes):
            out = out.decode("utf-8", errors="replace")
        return False, out + "\n[TIMEOUT]\n"
    except OSError as exc:
        return False, f"{exc.__class__.__name__}: {exc}\n"


def _git_init(path: Path) -> None:
    _write(path / ".gitignore", "__pycache__/\n*.pyc\n.pytest_cache/\n")
    _run(["git", "init", "-b", "master"], path, 30)
    _run(["git", "config", "user.email", "bench@example.com"], path, 30)
    _run(["git", "config", "user.name", "Hermes Bench"], path, 30)
    _run(["git", "add", "."], path, 30)
    _run(["git", "commit", "-m", "fixture"], path, 30)


def _changed_files(path: Path) -> list[str]:
    ok, out = _run(["git", "status", "--porcelain"], path, 30)
    if not ok:
        return ["<git-status-failed>"]
    changed: list[str] = []
    for line in out.splitlines():
        if not line.strip():
            continue
        changed.append(line[3:].strip())
    return sorted(changed)


def _verify_unittest(path: Path, ctx: SetupContext, timeout: int = 120) -> tuple[bool, str]:
    return _run([ctx.python_cmd, "-m", "unittest", "-v"], path, timeout)


def _verify_python(path: Path, ctx: SetupContext, script: str, timeout: int = 120) -> tuple[bool, str]:
    return _run([ctx.python_cmd, script], path, timeout)


def _verify_python_code(path: Path, ctx: SetupContext, code: str, timeout: int = 120) -> tuple[bool, str]:
    return _run([ctx.python_cmd, "-c", _dedent(code)], path, timeout)


def _contains_future_intent(text: str) -> bool:
    lowered = text.lower()
    patterns = (
        "let me ",
        "i will ",
        "i'll ",
        "next i ",
        "now i need",
        "going to ",
        "i’m going to",
        "i am going to",
    )
    return any(p in lowered for p in patterns)


def _strip_cli_session_prefix(output: str) -> str:
    """Remove Hermes CLI transport metadata before judging final content."""
    lines = [
        line
        for line in output.splitlines()
        if not line.strip().startswith("session_id:")
        and not line.strip().startswith("Warning: Unknown toolsets:")
    ]
    return "\n".join(lines).strip()


# C1
def setup_self_edit_smoke(path: Path, ctx: SetupContext) -> str:
    _write(
        path / "order_stats.py",
        """
        def median(values):
            values = list(values)
            if not values:
                raise ValueError("median requires at least one value")
            midpoint = len(values) // 2
            return values[midpoint]


        def summarize(values):
            return {
                "count": len(values),
                "median": median(values),
            }
        """,
    )
    _write(
        path / "test_order_stats.py",
        """
        import unittest

        from order_stats import median, summarize


        class OrderStatsTests(unittest.TestCase):
            def test_median_sorts_input(self):
                self.assertEqual(median([9, 1, 5]), 5)

            def test_median_even_count_averages_middle_pair(self):
                self.assertEqual(median([10, 2, 8, 4]), 6)

            def test_summarize_keeps_count_and_median(self):
                self.assertEqual(summarize([4, 1, 9]), {"count": 3, "median": 4})


        if __name__ == "__main__":
            unittest.main()
        """,
    )
    _git_init(path)
    return f"""
    Self-change canary C1.

    Fix the failing tests in this tiny repository. Keep the change minimal:
    only edit order_stats.py unless a test failure proves another file must be
    touched. Run this verifier before final:

        {ctx.python_cmd} -m unittest -v

    Final response: one sentence with the result.
    """


def verify_self_edit_smoke(path: Path, ctx: SetupContext, output: str) -> Verification:
    ok, test_out = _verify_unittest(path, ctx)
    changed = _changed_files(path)
    allowed = changed == ["order_stats.py"]
    passed = ok and allowed
    return Verification(
        passed=passed,
        message=f"unittest_ok={ok}; changed={changed}\n{test_out}",
        metrics={"changed_files": changed},
    )


# C2
def setup_prompt_stop_contract(path: Path, ctx: SetupContext) -> str:
    _write(
        path / "calendar.txt",
        """
        Monday: school pickup 16:30.
        Tuesday: soccer practice 18:00-19:30.
        Thursday: late work call ending 19:15.
        """,
    )
    _write(
        path / "keep_leftovers.md",
        """
        # Fridge notes
        - leftover rice, two portions
        - spinach opened yesterday
        - tofu must be used first
        """,
    )
    _write(
        path / "rewe_snapshot.csv",
        """
        item,price,available
        potatoes,1.99,yes
        fresh basil,2.49,no
        lentils,1.79,yes
        """,
    )
    _write(
        path / "ticktick.txt",
        """
        - Keep dinner prep under 25 minutes on Tuesday.
        - Stay under 35 EUR for the first shop.
        - Ask before starting recipe planning.
        """,
    )
    _git_init(path)
    return """
    Prompt/config canary C2.

    This is a synthetic weekly-meal-planning Phase 1 stop-contract task. All
    required data is in the current directory. Read the local files only. Do not
    edit files and do not touch real Calendar, Keep, REWE, TickTick, browser, or
    network services.

    Produce exactly one final kickoff MESSAGE for the user. It must mention:
    leftovers, schedule constraints, shopping/budget constraints, assumptions,
    and blockers or approvals needed. Stop there. Do not start recipe planning,
    do not propose a meal plan, and do not end with a future-intent sentence such
    as "I will..." or "Let me...".
    """


def verify_prompt_stop_contract(path: Path, ctx: SetupContext, output: str) -> Verification:
    changed = _changed_files(path)
    final_text = _strip_cli_session_prefix(output)
    lowered = final_text.lower()
    required = {
        "leftovers": "leftover" in lowered or "rice" in lowered or "spinach" in lowered,
        "schedule": "soccer" in lowered or "schedule" in lowered or "tuesday" in lowered,
        "budget": "budget" in lowered or "35" in lowered or "shop" in lowered,
        "assumptions": "assumption" in lowered or "assuming" in lowered,
        "approval": "approval" in lowered or "approve" in lowered or "ask" in lowered,
    }
    forbidden = {
        "future_intent": _contains_future_intent(output),
        "recipe_planning": any(
            phrase in lowered
            for phrase in ("meal plan:", "proposed plan:", "recipe 1", "dinner 1:", "start recipe planning now")
        ),
        "file_changes": bool(changed),
    }
    passed = all(required.values()) and not any(forbidden.values())
    return Verification(
        passed=passed,
        message=f"required={required}; forbidden={forbidden}; changed={changed}",
        metrics={"required": required, "forbidden": forbidden, "changed_files": changed},
    )


# Agentic intelligence tasks
def setup_processing_pipeline(path: Path, ctx: SetupContext) -> str:
    _write(path / "data" / "input.txt", "alpha\nbeta\n")
    _write(
        path / "scripts" / "extract.sh",
        """
        #!/usr/bin/env bash
        mkdir -p build
        cp data/input.txt build/extracted.txt
        """,
        0o644,
    )
    _write(
        path / "scripts" / "transform.py",
        """
        from pathlib import Path

        text = Path("build/extracted.txt").read_text(encoding="utf-8")
        Path("build/transformed.txt").write_text(text.upper, encoding="utf-8")
        """,
    )
    _write(
        path / "scripts" / "report.py",
        """
        from pathlib import Path

        lines = [x for x in Path("build/transformed.txt").read_text(encoding="utf-8").splitlines() if x]
        Path("output").mkdir(exist_ok=True)
        Path("output/report.txt").write_text("\\n".join(lines + [f"COUNT={len(lines)}"]) + "\\n", encoding="utf-8")
        """,
    )
    _write(
        path / "run_pipeline.sh",
        """
        #!/usr/bin/env bash
        set -euo pipefail
        ./scripts/extract.sh
        python3 scripts/transform.py
        python3 scripts/report.py
        """,
        0o755,
    )
    _git_init(path)
    return """
    Fix the processing pipeline so ./run_pipeline.sh succeeds and writes
    output/report.txt with uppercase input lines followed by COUNT=2.
    Keep the fix minimal and verify the command before final.
    """


def verify_processing_pipeline(path: Path, ctx: SetupContext, output: str) -> Verification:
    ok, out = _run(["./run_pipeline.sh"], path, 60)
    report = path / "output" / "report.txt"
    content = report.read_text(encoding="utf-8") if report.exists() else ""
    passed = ok and content == "ALPHA\nBETA\nCOUNT=2\n"
    return Verification(passed, f"run_ok={ok}\n{out}\nreport={content!r}")


def setup_git_history_repair(path: Path, ctx: SetupContext) -> str:
    _git_init(path)
    _write(path / "index.html", "<h1>Home</h1>\n")
    _run(["git", "add", "index.html"], path)
    _run(["git", "commit", "-m", "initial site"], path)
    _run(["git", "checkout", "-b", "feature/contact-form"], path)
    _write(path / "index.html", "<h1>Home</h1>\n<p>Contact form enabled</p>\n")
    _run(["git", "commit", "-am", "add contact form"], path)
    _run(["git", "checkout", "master"], path)
    return """
    A useful change was committed on another branch. Bring the contact-form
    change into master cleanly, preserving git history, and leave the repository
    on master. Verify with git log and the file content before final.
    """


def verify_git_history_repair(path: Path, ctx: SetupContext, output: str) -> Verification:
    _, branch = _run(["git", "branch", "--show-current"], path)
    _, log = _run(["git", "log", "--oneline", "--decorate", "-5"], path)
    content = (path / "index.html").read_text(encoding="utf-8")
    passed = branch.strip() == "master" and "Contact form enabled" in content and "add contact form" in log
    return Verification(passed, f"branch={branch}\nlog={log}\ncontent={content}")


def setup_grid_pattern_transform(path: Path, ctx: SetupContext) -> str:
    _write(path / "grid_transform.py", "def transform(grid):\n    return grid\n")
    _write(
        path / "test_grid_transform.py",
        """
        import unittest

        from grid_transform import transform


        class GridTransformTests(unittest.TestCase):
            def test_border_fill(self):
                self.assertEqual(
                    transform([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
                    [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                )

            def test_preserve_existing_nonzero(self):
                self.assertEqual(
                    transform([[0, 2, 0], [0, 0, 0], [3, 0, 0]]),
                    [[2, 2, 2], [3, 2, 2], [3, 3, 0]],
                )


        if __name__ == "__main__":
            unittest.main()
        """,
    )
    _git_init(path)
    return f"""
    Fix grid_transform.py so `{ctx.python_cmd} -m unittest -v` passes.
    The transform expands each nonzero cell to its 8-neighbors without
    overwriting an original nonzero value. If two sources can fill the same
    original zero cell, orthogonal adjacency beats diagonal adjacency. Keep
    edits focused.
    """


def verify_grid_pattern_transform(path: Path, ctx: SetupContext, output: str) -> Verification:
    ok, out = _verify_unittest(path, ctx)
    return Verification(ok, out)


def setup_cross_entropy_method(path: Path, ctx: SetupContext) -> str:
    _write(
        path / "cem.py",
        """
        import math
        import random

        ACTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1), (0, 0))


        class PointEnv:
            def __init__(self, goal=(2, 0), limit=3):
                self.goal = goal
                self.limit = limit
                self.cache = {}

            def rollout(self, plan):
                x = 0
                y = 0
                for action in plan:
                    dx, dy = ACTIONS[int(action)]
                    x = max(-self.limit, min(self.limit, x + dx))
                    y = max(-self.limit, min(self.limit, y + dy))
                return -math.dist((x, y), self.goal)

            def evaluate_plans_memoized(self, plans):
                raise NotImplementedError


        class CrossEntropyMethod:
            def __init__(self, env, horizon=3, samples=120, elite_frac=0.2, seed=0):
                self.env = env
                self.horizon = horizon
                self.samples = samples
                self.elite_frac = elite_frac
                self.rng = random.Random(seed)

            def optimize(self, iterations=4):
                raise NotImplementedError
        """,
    )
    _write(
        path / "test_cem.py",
        """
        import unittest

        from cem import CrossEntropyMethod, PointEnv


        class CEMTests(unittest.TestCase):
            def test_memoized_evaluation_reuses_cache(self):
                env = PointEnv(goal=(2, 0))
                plans = [(0, 0, 0), (0, 0, 0), (2, 2, 2)]
                scores = env.evaluate_plans_memoized(plans)
                self.assertEqual(len(scores), 3)
                self.assertEqual(scores[0], scores[1])
                self.assertEqual(len(env.cache), 2)

            def test_cem_finds_plan_toward_goal(self):
                env = PointEnv(goal=(2, 0))
                cem = CrossEntropyMethod(env, horizon=3, samples=160, elite_frac=0.25, seed=7)
                plan, score = cem.optimize(iterations=5)
                self.assertEqual(len(plan), 3)
                self.assertGreaterEqual(score, -1e-9)
                self.assertGreaterEqual(env.rollout(plan), -1e-9)


        if __name__ == "__main__":
            unittest.main()
        """,
    )
    _git_init(path)
    return f"""
    Implement the missing cross-entropy-method planner in cem.py so
    `{ctx.python_cmd} -m unittest -v` passes. Keep the implementation
    deterministic for the provided seed and avoid overfitting to one exact test.
    """


def verify_cross_entropy_method(path: Path, ctx: SetupContext, output: str) -> Verification:
    ok, out = _verify_unittest(path, ctx, timeout=180)
    return Verification(ok, out)


def setup_broken_python_toolchain(path: Path, ctx: SetupContext) -> str:
    _write(path / "bin" / "python", f"#!/usr/bin/env bash\nexec {ctx.python_cmd} \"$@\"\n", 0o755)
    _write(
        path / "bin" / "pip",
        """
        #!/usr/bin/env bash
        echo 'bad interpreter: /missing/python' >&2
        exit 127
        """,
        0o755,
    )
    _git_init(path)
    return """
    The local Python toolchain under ./bin is broken: ./bin/pip fails even
    though ./bin/python works. Fix it so ./bin/pip --version and
    ./bin/python -m pip --version both succeed. Keep the repair local to ./bin.
    """


def verify_broken_python_toolchain(path: Path, ctx: SetupContext, output: str) -> Verification:
    ok1, out1 = _run(["./bin/pip", "--version"], path)
    ok2, out2 = _run(["./bin/python", "-m", "pip", "--version"], path)
    changed = _changed_files(path)
    passed = ok1 and ok2 and all(p.startswith("bin/") for p in changed)
    return Verification(passed, f"pip={ok1}\n{out1}\npython -m pip={ok2}\n{out2}\nchanged={changed}")


def setup_sqlite_migration(path: Path, ctx: SetupContext) -> str:
    db = path / "app.db"
    conn = sqlite3.connect(db)
    conn.execute("create table users(id integer primary key, name text)")
    conn.execute("insert into users(name) values ('Ada')")
    conn.commit()
    conn.close()
    _write(
        path / "migrate.py",
        """
        import sqlite3

        conn = sqlite3.connect("app.db")
        conn.execute("alter table users add column email text not null")
        conn.commit()
        conn.close()
        """,
    )
    _write(
        path / "verify_migration.py",
        """
        import sqlite3

        import importlib
        import migrate  # first run
        importlib.reload(migrate)  # second run should not break idempotency

        conn = sqlite3.connect("app.db")
        columns = [row[1] for row in conn.execute("pragma table_info(users)")]
        rows = list(conn.execute("select name, email from users"))
        assert "email" in columns, columns
        assert rows == [("Ada", "")], rows
        conn.close()
        """,
    )
    _git_init(path)
    return f"""
    Fix migrate.py so `{ctx.python_cmd} verify_migration.py` passes. The
    migration must be idempotent and preserve existing users with a default
    empty email. Do not replace the database with a fresh one.
    """


def verify_sqlite_migration(path: Path, ctx: SetupContext, output: str) -> Verification:
    ok, out = _verify_python(path, ctx, "verify_migration.py")
    return Verification(ok, out)


def setup_argparse_feature(path: Path, ctx: SetupContext) -> str:
    _write(
        path / "summarize_cli.py",
        """
        import argparse
        from collections import Counter
        from pathlib import Path


        def main(argv=None):
            parser = argparse.ArgumentParser()
            parser.add_argument("path")
            args = parser.parse_args(argv)
            words = Path(args.path).read_text(encoding="utf-8").lower().split()
            for word, count in Counter(words).most_common():
                print(f"{word},{count}")


        if __name__ == "__main__":
            main()
        """,
    )
    _write(path / "sample.txt", "alpha beta alpha gamma beta alpha\n")
    _write(
        path / "test_cli.py",
        """
        import subprocess
        import sys
        import unittest


        class CLITests(unittest.TestCase):
            def test_limit_flag_caps_rows(self):
                out = subprocess.check_output(
                    [sys.executable, "summarize_cli.py", "--limit", "2", "sample.txt"],
                    text=True,
                )
                self.assertEqual(out.splitlines(), ["alpha,3", "beta,2"])


        if __name__ == "__main__":
            unittest.main()
        """,
    )
    _git_init(path)
    return f"""
    Add a --limit N option to summarize_cli.py so `{ctx.python_cmd} -m unittest -v`
    passes. Preserve the existing positional path behavior.
    """


def verify_argparse_feature(path: Path, ctx: SetupContext, output: str) -> Verification:
    ok, out = _verify_unittest(path, ctx)
    return Verification(ok, out)


def setup_async_retry(path: Path, ctx: SetupContext) -> str:
    _write(
        path / "fetcher.py",
        """
        import asyncio


        async def fetch_with_retry(client, url, retries=2):
            for _ in range(retries):
                return await client.fetch(url)
            return await client.fetch(url)


        async def fetch_all(client, urls):
            return [await fetch_with_retry(client, urls[0]) for _ in urls]
        """,
    )
    _write(
        path / "test_fetcher.py",
        """
        import asyncio
        import unittest

        from fetcher import fetch_all, fetch_with_retry


        class FakeClient:
            def __init__(self):
                self.calls = {}

            async def fetch(self, url):
                self.calls[url] = self.calls.get(url, 0) + 1
                if url == "flaky" and self.calls[url] == 1:
                    raise TimeoutError("try again")
                await asyncio.sleep(0)
                return f"ok:{url}"


        class FetcherTests(unittest.IsolatedAsyncioTestCase):
            async def test_retry_then_success(self):
                client = FakeClient()
                self.assertEqual(await fetch_with_retry(client, "flaky", retries=2), "ok:flaky")
                self.assertEqual(client.calls["flaky"], 2)

            async def test_fetch_all_uses_each_url_in_order(self):
                client = FakeClient()
                self.assertEqual(await fetch_all(client, ["a", "b", "c"]), ["ok:a", "ok:b", "ok:c"])


        if __name__ == "__main__":
            unittest.main()
        """,
    )
    _git_init(path)
    return f"""
    Fix the async fetch helpers so `{ctx.python_cmd} -m unittest -v` passes.
    Preserve order, retry transient failures, and keep the code small.
    """


def verify_async_retry(path: Path, ctx: SetupContext, output: str) -> Verification:
    ok, out = _verify_unittest(path, ctx)
    return Verification(ok, out)


def setup_lru_ttl_cache(path: Path, ctx: SetupContext) -> str:
    _write(
        path / "ttl_cache.py",
        """
        class TTLCache:
            def __init__(self, max_size=2, clock=None):
                self.max_size = max_size
                self.clock = clock or (lambda: 0)
                self.data = {}

            def get(self, key):
                value, expires_at = self.data[key]
                return value

            def set(self, key, value, ttl):
                self.data[key] = (value, self.clock() + ttl)
        """,
    )
    _write(
        path / "test_ttl_cache.py",
        """
        import unittest

        from ttl_cache import TTLCache


        class Clock:
            def __init__(self):
                self.now = 0
            def __call__(self):
                return self.now


        class TTLCacheTests(unittest.TestCase):
            def test_expired_keys_miss(self):
                clock = Clock()
                cache = TTLCache(max_size=2, clock=clock)
                cache.set("a", 1, ttl=5)
                clock.now = 6
                self.assertIsNone(cache.get("a"))

            def test_lru_evicts_oldest_live_key(self):
                clock = Clock()
                cache = TTLCache(max_size=2, clock=clock)
                cache.set("a", 1, ttl=10)
                cache.set("b", 2, ttl=10)
                self.assertEqual(cache.get("a"), 1)
                cache.set("c", 3, ttl=10)
                self.assertIsNone(cache.get("b"))
                self.assertEqual(cache.get("a"), 1)
                self.assertEqual(cache.get("c"), 3)


        if __name__ == "__main__":
            unittest.main()
        """,
    )
    _git_init(path)
    return f"""
    Implement ttl_cache.TTLCache so `{ctx.python_cmd} -m unittest -v` passes.
    Requirements: get returns None for missing/expired keys, set honors TTL,
    and when over max_size it evicts the least recently used live key.
    """


def verify_lru_ttl_cache(path: Path, ctx: SetupContext, output: str) -> Verification:
    ok, out = _verify_unittest(path, ctx)
    return Verification(ok, out)


# Instruction following and tool-calling tasks
def setup_json_tool_report(path: Path, ctx: SetupContext) -> str:
    _write(path / "sales.csv", "region,amount\nnorth,12\nsouth,8\nnorth,5\n")
    _write(path / "notes.md", "- north includes online orders\n- south has one refund pending\n")
    _git_init(path)
    return """
    Inspect the local CSV and notes, then write report.json with exactly:
    {"north_total": 17, "south_total": 8, "notes_count": 2}
    Do not include commentary in the file. Final response should be brief.
    """


def verify_json_tool_report(path: Path, ctx: SetupContext, output: str) -> Verification:
    p = path / "report.json"
    if not p.exists():
        return Verification(False, "missing report.json")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return Verification(False, f"invalid JSON: {exc}")
    passed = data == {"north_total": 17, "south_total": 8, "notes_count": 2}
    return Verification(passed, repr(data))


def setup_tool_call_extraction(path: Path, ctx: SetupContext) -> str:
    _write(
        path / "raw_model_output.txt",
        """
        Thinking...
        <tool_call>{"name":"read_file","arguments":{"path":"alpha.txt"}}</tool_call>
        then
        <tool_call>{"name":"terminal","arguments":{"command":"wc -l alpha.txt"}}</tool_call>
        """,
    )
    _git_init(path)
    return """
    Extract deterministic tool call data from raw_model_output.txt and write
    tools.json as a JSON array of objects with keys name and arguments, preserving
    order. Do not execute the extracted commands.
    """


def verify_tool_call_extraction(path: Path, ctx: SetupContext, output: str) -> Verification:
    p = path / "tools.json"
    if not p.exists():
        return Verification(False, "missing tools.json")
    data = json.loads(p.read_text(encoding="utf-8"))
    expected = [
        {"name": "read_file", "arguments": {"path": "alpha.txt"}},
        {"name": "terminal", "arguments": {"command": "wc -l alpha.txt"}},
    ]
    return Verification(data == expected, repr(data))


def setup_context_needle(path: Path, ctx: SetupContext) -> str:
    for i in range(60):
        _write(path / "docs" / f"note_{i:02d}.txt", f"noise line {i}\n" * 12)
    _write(
        path / "docs" / "note_47.txt",
        """
        noise
        RELEASE_GATE_CODE=violet-813
        noise
        """,
    )
    _git_init(path)
    return """
    Search the local docs directory and write answer.txt containing only the
    RELEASE_GATE_CODE value. Do not guess; inspect the files.
    """


def verify_context_needle(path: Path, ctx: SetupContext, output: str) -> Verification:
    p = path / "answer.txt"
    text = p.read_text(encoding="utf-8").strip() if p.exists() else ""
    return Verification(text == "violet-813", f"answer={text!r}")


def setup_readonly_audit(path: Path, ctx: SetupContext) -> str:
    _write(path / "service.conf", "port=8787\nmode=readonly\n")
    _write(path / "events.log", "INFO boot\nWARN retry\nERROR timeout\n")
    _git_init(path)
    return """
    Read service.conf and events.log, then answer with the configured port and
    the count of ERROR lines. This is a read-only audit: do not edit or create
    files.
    """


def verify_readonly_audit(path: Path, ctx: SetupContext, output: str) -> Verification:
    changed = _changed_files(path)
    lowered = output.lower()
    passed = not changed and "8787" in output and ("1" in output or "one" in lowered) and "error" in lowered
    return Verification(passed, f"changed={changed}; output={output}")


def setup_exact_patch_scope(path: Path, ctx: SetupContext) -> str:
    _write(path / "target.txt", "status=old\n")
    _write(path / "protected.txt", "do-not-touch\n")
    _git_init(path)
    return """
    Change target.txt so it contains exactly `status=new` and leave every other
    file untouched. Do not edit protected.txt.
    """


def verify_exact_patch_scope(path: Path, ctx: SetupContext, output: str) -> Verification:
    target = (path / "target.txt").read_text(encoding="utf-8").strip()
    protected = (path / "protected.txt").read_text(encoding="utf-8").strip()
    changed = _changed_files(path)
    passed = target == "status=new" and protected == "do-not-touch" and changed == ["target.txt"]
    return Verification(passed, f"target={target!r}; protected={protected!r}; changed={changed}")


def setup_final_json_only(path: Path, ctx: SetupContext) -> str:
    _write(path / "input.txt", "red\nblue\nred\n")
    _git_init(path)
    return """
    Inspect input.txt and respond with final output as one JSON object only, no
    Markdown and no extra prose. Required object:
    {"red":2,"blue":1}
    Do not write files.
    """


def verify_final_json_only(path: Path, ctx: SetupContext, output: str) -> Verification:
    changed = _changed_files(path)
    stripped = _strip_cli_session_prefix(output)
    match = re.search(r"\{[^{}]*\}\s*$", stripped)
    if not match:
        return Verification(False, f"no final JSON object found; changed={changed}; output={output!r}")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return Verification(False, f"invalid final JSON: {exc}; output={output!r}")
    passed = data == {"red": 2, "blue": 1} and match.group(0) == stripped and not changed
    return Verification(passed, f"data={data}; changed={changed}")


def setup_csv_summary(path: Path, ctx: SetupContext) -> str:
    _write(path / "expenses.csv", "category,amount\nfood,12\ntravel,20\nfood,8\n")
    _git_init(path)
    return """
    Read expenses.csv and create totals.csv with header category,total and rows
    sorted alphabetically by category. Use integer totals.
    """


def verify_csv_summary(path: Path, ctx: SetupContext, output: str) -> Verification:
    p = path / "totals.csv"
    if not p.exists():
        return Verification(False, "missing totals.csv")
    rows = list(csv.reader(p.open(encoding="utf-8", newline="")))
    expected = [["category", "total"], ["food", "20"], ["travel", "20"]]
    return Verification(rows == expected, repr(rows))


def setup_markdown_table_extract(path: Path, ctx: SetupContext) -> str:
    _write(
        path / "README.md",
        """
        | name | status | owner |
        | --- | --- | --- |
        | alpha | passing | Mira |
        | beta | failing | Jo |
        | gamma | passing | Sol |
        """,
    )
    _git_init(path)
    return """
    Parse README.md and write failing_owners.txt containing the owner names for
    rows whose status is failing, one per line, no bullets.
    """


def verify_markdown_table_extract(path: Path, ctx: SetupContext, output: str) -> Verification:
    p = path / "failing_owners.txt"
    text = p.read_text(encoding="utf-8").strip() if p.exists() else ""
    return Verification(text == "Jo", f"text={text!r}")


def setup_config_merge(path: Path, ctx: SetupContext) -> str:
    _write(
        path / "base.json",
        """
        {"model": {"provider": "local", "temperature": 0.2}, "tools": ["file", "terminal"]}
        """,
    )
    _write(
        path / "override.json",
        """
        {"model": {"temperature": 0.0}, "tools": ["file", "terminal", "memory"]}
        """,
    )
    _git_init(path)
    return """
    Merge base.json and override.json into merged.json. Nested objects should be
    merged, with override values winning. Arrays should be replaced by the
    override array. Do not mutate the inputs.
    """


def verify_config_merge(path: Path, ctx: SetupContext, output: str) -> Verification:
    p = path / "merged.json"
    if not p.exists():
        return Verification(False, "missing merged.json")
    data = json.loads(p.read_text(encoding="utf-8"))
    expected = {
        "model": {"provider": "local", "temperature": 0.0},
        "tools": ["file", "terminal", "memory"],
    }
    inputs_ok = "temperature\": 0.2" in (path / "base.json").read_text(encoding="utf-8")
    return Verification(data == expected and inputs_ok, f"data={data}; inputs_ok={inputs_ok}")


# Calibrated hard tasks. These keep visible examples small but verify hidden
# edge cases so the suite remains useful after the easy tasks saturate.
def setup_json_patch_engine(path: Path, ctx: SetupContext) -> str:
    _write(
        path / "jsonpatcher.py",
        """
        class JsonPatchError(Exception):
            pass


        def apply_patch(document, operations):
            return document
        """,
    )
    _write(
        path / "test_jsonpatcher.py",
        """
        import unittest

        from jsonpatcher import apply_patch


        class JsonPatcherTests(unittest.TestCase):
            def test_add_replace_remove_without_mutating_input(self):
                doc = {"items": ["a", "b"], "meta": {"count": 2}}
                result = apply_patch(doc, [
                    {"op": "add", "path": "/items/-", "value": "c"},
                    {"op": "replace", "path": "/meta/count", "value": 3},
                    {"op": "remove", "path": "/items/0"},
                ])
                self.assertEqual(result, {"items": ["b", "c"], "meta": {"count": 3}})
                self.assertEqual(doc, {"items": ["a", "b"], "meta": {"count": 2}})

            def test_json_pointer_escaping(self):
                doc = {"a/b": {"~key": 1}}
                result = apply_patch(doc, [
                    {"op": "test", "path": "/a~1b/~0key", "value": 1},
                    {"op": "replace", "path": "/a~1b/~0key", "value": 2},
                ])
                self.assertEqual(result["a/b"]["~key"], 2)


        if __name__ == "__main__":
            unittest.main()
        """,
    )
    _git_init(path)
    return f"""
    Implement a small JSON Patch engine in jsonpatcher.py and make
    `{ctx.python_cmd} -m unittest -v` pass.

    Requirements: support add, remove, replace, move, copy, and test. Support
    JSON Pointer paths, including the empty root path, /- array append for add,
    and ~0/~1 escaping. Return a patched deep copy without mutating the input.
    Raise JsonPatchError for invalid operations, paths, and failed tests.
    """


def verify_json_patch_engine(path: Path, ctx: SetupContext, output: str) -> Verification:
    ok, visible = _verify_unittest(path, ctx, timeout=180)
    if not ok:
        return Verification(False, visible)
    ok, hidden = _verify_python_code(
        path,
        ctx,
        """
        from jsonpatcher import JsonPatchError, apply_patch

        original = {"items": [{"x": 1}, {"x": 2}], "archive": []}
        patched = apply_patch(original, [
            {"op": "copy", "from": "/items/0", "path": "/archive/-"},
            {"op": "move", "from": "/items/1", "path": "/archive/-"},
        ])
        assert patched == {"items": [{"x": 1}], "archive": [{"x": 1}, {"x": 2}]}, patched
        patched["archive"][0]["x"] = 99
        assert original == {"items": [{"x": 1}, {"x": 2}], "archive": []}, original

        assert apply_patch({"a": 1}, [{"op": "replace", "path": "", "value": [1, 2]}]) == [1, 2]

        for ops in (
            [{"op": "test", "path": "/missing", "value": 1}],
            [{"op": "remove", "path": "/missing"}],
            [{"op": "add", "path": "/items/9", "value": 0}],
        ):
            try:
                apply_patch({"items": []}, ops)
            except JsonPatchError:
                pass
            else:
                raise AssertionError(f"expected JsonPatchError for {ops!r}")
        """,
        timeout=180,
    )
    return Verification(ok, f"visible:\n{visible}\nhidden:\n{hidden}")


def setup_semver_range_resolver(path: Path, ctx: SetupContext) -> str:
    _write(
        path / "semver_resolver.py",
        """
        def select_version(available, constraint):
            return None
        """,
    )
    _write(
        path / "test_semver_resolver.py",
        """
        import unittest

        from semver_resolver import select_version


        class SemverResolverTests(unittest.TestCase):
            def test_caret_and_comparator_ranges(self):
                versions = ["1.0.0", "1.2.0", "1.2.3", "1.9.0", "2.0.0"]
                self.assertEqual(select_version(versions, "^1.2.0"), "1.9.0")
                self.assertEqual(select_version(versions, ">=1.2.0, <1.3.0"), "1.2.3")

            def test_or_range_and_none(self):
                versions = ["1.0.0", "2.0.0", "2.1.0"]
                self.assertEqual(select_version(versions, "<1.0.0 || >=2.0.0, <2.1.0"), "2.0.0")
                self.assertIsNone(select_version(versions, ">=3.0.0"))


        if __name__ == "__main__":
            unittest.main()
        """,
    )
    _git_init(path)
    return f"""
    Implement semver_resolver.select_version so `{ctx.python_cmd} -m unittest -v`
    passes, then run the verifier.

    It should return the highest available version that satisfies the constraint,
    or None. Versions are SemVer major.minor.patch with optional prerelease and
    optional +build metadata. Build metadata must not affect ordering. Support
    comma- or whitespace-separated AND clauses, || OR clauses, bare exact
    versions, =, !=, <, <=, >, >=, caret ranges, tilde ranges, and x/*
    wildcards such as 1.2.x or 1.*. Caret follows SemVer 0.x rules.
    Prerelease versions only satisfy a range when that OR group explicitly
    includes a prerelease comparator.
    """


def verify_semver_range_resolver(path: Path, ctx: SetupContext, output: str) -> Verification:
    ok, visible = _verify_unittest(path, ctx, timeout=180)
    if not ok:
        return Verification(False, visible)
    ok, hidden = _verify_python_code(
        path,
        ctx,
        """
        from semver_resolver import select_version

        versions = [
            "0.0.3", "0.0.4", "0.2.3", "0.2.9", "0.3.0",
            "1.0.0-alpha.1", "1.0.0-alpha.10", "1.0.0",
            "1.0.1+build.7", "1.1.0",
        ]
        assert select_version(versions, "^0.2.3") == "0.2.9"
        assert select_version(versions, "^0.0.3") == "0.0.3"
        assert select_version(versions, "~1.0.0") == "1.0.1+build.7"
        assert select_version(versions, ">=1.0.0-alpha.1, <1.0.0") == "1.0.0-alpha.10"
        assert select_version(versions, ">=1.0.0, !=1.0.1+build.7, <2.0.0") == "1.1.0"
        assert select_version(["1.0.0+one", "1.0.0+two"], "=1.0.0") in {"1.0.0+one", "1.0.0+two"}
        wildcard_versions = ["1.2.0", "1.2.5", "1.3.0", "2.0.0"]
        assert select_version(wildcard_versions, "1.2.x") == "1.2.5"
        assert select_version(wildcard_versions, "1.*") == "1.3.0"
        assert select_version(wildcard_versions, "!=1.2.x, <2.0.0") == "1.3.0"
        assert select_version(["1.1.0-alpha.1", "1.0.0"], ">=1.0.0, <2.0.0") == "1.0.0"
        assert select_version(["1.4.0", "1.5.0", "2.0.0"], ">=1.0.0 <2.0.0") == "1.5.0"
        assert select_version(["1.4.0", "1.5.0", "2.0.0"], "^1.0.0 !=1.5.0") == "1.4.0"
        """,
        timeout=180,
    )
    return Verification(ok, f"visible:\n{visible}\nhidden:\n{hidden}")


def setup_dependency_resolver(path: Path, ctx: SetupContext) -> str:
    _write(
        path / "resolver.py",
        """
        class ResolutionError(Exception):
            pass


        def resolve(catalog, roots):
            return {}
        """,
    )
    _write(
        path / "test_resolver.py",
        """
        import unittest

        from resolver import resolve


        class ResolverTests(unittest.TestCase):
            def test_picks_highest_compatible_versions(self):
                catalog = {
                    "app": {"1.0.0": {"lib": "^1.0.0"}},
                    "lib": {"1.0.0": {}, "1.2.0": {}, "2.0.0": {}},
                }
                self.assertEqual(resolve(catalog, {"app": "^1.0.0"}), {"app": "1.0.0", "lib": "1.2.0"})

            def test_backtracks_away_from_conflict(self):
                catalog = {
                    "app": {"1.0.0": {"ui": "^1.0.0", "db": "^1.0.0"}},
                    "ui": {"1.1.0": {"core": "^2.0.0"}, "1.0.0": {"core": "^1.0.0"}},
                    "db": {"1.0.0": {"core": "^1.0.0"}},
                    "core": {"1.5.0": {}, "2.0.0": {}},
                }
                self.assertEqual(resolve(catalog, {"app": "^1.0.0"}), {
                    "app": "1.0.0", "ui": "1.0.0", "db": "1.0.0", "core": "1.5.0"
                })


        if __name__ == "__main__":
            unittest.main()
        """,
    )
    _git_init(path)
    return f"""
    Implement resolver.resolve so `{ctx.python_cmd} -m unittest -v` passes.

    Catalog shape: package -> version -> dependency mapping. Roots is a package
    -> constraint mapping. Choose the highest compatible version for every root
    and transitive dependency, backtracking when a newer choice conflicts later.
    Support exact versions, =, comma-separated AND constraints, || OR constraints,
    !=, <, <=, >, >=, caret ranges, and tilde ranges. Versions may include
    SemVer prerelease and +build metadata; build metadata must not affect
    ordering. Prerelease versions only satisfy a range when that constraint
    group explicitly includes a prerelease comparator. Return {{"package":
    "version"}}. Raise ResolutionError if no complete solution exists.
    """


def verify_dependency_resolver(path: Path, ctx: SetupContext, output: str) -> Verification:
    ok, visible = _verify_unittest(path, ctx, timeout=180)
    if not ok:
        return Verification(False, visible)
    ok, hidden = _verify_python_code(
        path,
        ctx,
        """
        from resolver import ResolutionError, resolve

        catalog = {
            "cli": {"2.0.0": {"parser": "^2.0.0"}, "1.5.0": {"parser": "^1.0.0"}},
            "plugin": {"1.0.0": {"parser": ">=1.4.0, <2.0.0"}},
            "parser": {"2.1.0": {}, "2.0.0": {}, "1.9.0": {}, "1.4.0": {}},
        }
        assert resolve(catalog, {"cli": ">=1.0.0", "plugin": "^1.0.0"}) == {
            "cli": "1.5.0", "plugin": "1.0.0", "parser": "1.9.0"
        }
        try:
            resolve(catalog, {"cli": "^2.0.0", "plugin": "^1.0.0"})
        except ResolutionError:
            pass
        else:
            raise AssertionError("expected ResolutionError")

        catalog = {
            "app": {"1.0.0": {"lib": "~1.2.0 || ^2.0.0"}},
            "addon": {"1.0.0": {"lib": ">=1.2.0, !=1.2.5, <1.3.0"}},
            "lib": {"2.0.1": {}, "1.2.6": {}, "1.2.5": {}, "1.2.0": {}},
        }
        assert resolve(catalog, {"app": "^1.0.0", "addon": "^1.0.0"}) == {
            "app": "1.0.0", "addon": "1.0.0", "lib": "1.2.6"
        }

        catalog = {
            "app": {"1.0.0": {"lib": ">=2.0.0-alpha.1, <2.0.0"}},
            "lib": {"2.0.0": {}, "2.0.0-beta.1": {}, "2.0.0-alpha.1+build.5": {}},
        }
        assert resolve(catalog, {"app": "^1.0.0"}) == {"app": "1.0.0", "lib": "2.0.0-beta.1"}

        catalog = {
            "a": {"2.0.0": {}, "1.0.0": {}},
            "b": {"1.0.0": {"a": "<2.0.0"}},
        }
        assert resolve(catalog, {"a": ">=1.0.0", "b": "=1.0.0"}) == {"a": "1.0.0", "b": "1.0.0"}

        catalog = {
            "app": {"1.0.0": {"lib": ">=1.0.0, <2.0.0"}},
            "lib": {"2.0.0-alpha.1": {}, "1.9.0": {}},
        }
        assert resolve(catalog, {"app": "^1.0.0"}) == {"app": "1.0.0", "lib": "1.9.0"}
        """,
        timeout=180,
    )
    return Verification(ok, f"visible:\n{visible}\nhidden:\n{hidden}")


def setup_unified_diff_apply(path: Path, ctx: SetupContext) -> str:
    _write(
        path / "diff_apply.py",
        """
        class PatchApplyError(Exception):
            pass


        def apply_unified_diff(files, diff_text):
            return files
        """,
    )
    _write(
        path / "test_diff_apply.py",
        """
        import unittest

        from diff_apply import apply_unified_diff


        class DiffApplyTests(unittest.TestCase):
            def test_updates_existing_file(self):
                files = {"a.txt": "one\\ntwo\\nthree\\n"}
                diff = (
                    "--- a/a.txt\\n"
                    "+++ b/a.txt\\n"
                    "@@ -1,3 +1,3 @@\\n"
                    " one\\n"
                    "-two\\n"
                    "+TWO\\n"
                    " three\\n"
                )
                self.assertEqual(apply_unified_diff(files, diff)["a.txt"], "one\\nTWO\\nthree\\n")
                self.assertEqual(files["a.txt"], "one\\ntwo\\nthree\\n")


        if __name__ == "__main__":
            unittest.main()
        """,
    )
    _git_init(path)
    return f"""
    Implement diff_apply.apply_unified_diff so `{ctx.python_cmd} -m unittest -v`
    passes.

    It receives a dict of path -> text and a unified diff string. Return a new
    dict after applying all hunks exactly. Support ---/+++ file headers, optional
    diff --git lines, multiple files, multiple hunks per file, hunk headers with
    trailing section text, quoted paths with spaces and optional timestamps in
    ---/+++ headers, /dev/null creates and deletes, source/target path changes
    such as renames with edits, "\\ No newline at end of file" markers, and
    context validation. Raise PatchApplyError on mismatch.
    """


def verify_unified_diff_apply(path: Path, ctx: SetupContext, output: str) -> Verification:
    ok, visible = _verify_unittest(path, ctx, timeout=180)
    if not ok:
        return Verification(False, visible)
    ok, hidden = _verify_python_code(
        path,
        ctx,
        """
        from diff_apply import PatchApplyError, apply_unified_diff

        files = {"a.txt": "one\\ntwo\\nthree\\n", "old.txt": "gone\\naway\\n"}
        diff = '''diff --git a/a.txt b/a.txt
        --- a/a.txt
        +++ b/a.txt
        @@ -1,3 +1,4 @@
         one
        -two
        +TWO
         three
        +four
        diff --git a/new.txt b/new.txt
        --- /dev/null
        +++ b/new.txt
        @@ -0,0 +1,2 @@
        +fresh
        +file
        diff --git a/old.txt b/old.txt
        --- a/old.txt
        +++ /dev/null
        @@ -1,2 +0,0 @@
        -gone
        -away
        '''
        out = apply_unified_diff(files, diff)
        assert out["a.txt"] == "one\\nTWO\\nthree\\nfour\\n", out
        assert out["new.txt"] == "fresh\\nfile\\n", out
        assert "old.txt" not in out, out
        assert files["a.txt"] == "one\\ntwo\\nthree\\n", files
        try:
            apply_unified_diff({"a.txt": "different\\n"}, "--- a/a.txt\\n+++ b/a.txt\\n@@ -1 +1 @@\\n-one\\n+two\\n")
        except PatchApplyError:
            pass
        else:
            raise AssertionError("expected PatchApplyError")

        no_newline = apply_unified_diff(
            {"a.txt": "one\\ntwo"},
            "--- a/a.txt\\n+++ b/a.txt\\n@@ -1,2 +1,2 @@ section\\n one\\n-two\\n\\\\ No newline at end of file\\n+TWO\\n\\\\ No newline at end of file\\n",
        )
        assert no_newline["a.txt"] == "one\\nTWO", repr(no_newline["a.txt"])
        renamed = apply_unified_diff(
            {"old.txt": "name=old\\n"},
            "diff --git a/old.txt b/new.txt\\n--- a/old.txt\\n+++ b/new.txt\\n@@ -1 +1 @@\\n-name=old\\n+name=new\\n",
        )
        assert renamed == {"new.txt": "name=new\\n"}, renamed
        quoted = apply_unified_diff(
            {"dir/old name.txt": "alpha\\n"},
            '--- "a/dir/old name.txt"\\t2026-05-12\\n+++ "b/dir/new name.txt"\\t2026-05-12\\n@@ -1 +1 @@\\n-alpha\\n+beta\\n',
        )
        assert quoted == {"dir/new name.txt": "beta\\n"}, quoted
        inserted = apply_unified_diff(
            {"a.txt": "a\\nb\\n"},
            "--- a/a.txt\\n+++ b/a.txt\\n@@ -1,0 +2,1 @@\\n+x\\n",
        )
        assert inserted["a.txt"] == "a\\nx\\nb\\n", inserted
        """,
        timeout=180,
    )
    return Verification(ok, f"visible:\n{visible}\nhidden:\n{hidden}")


def setup_workflow_rule_engine(path: Path, ctx: SetupContext) -> str:
    _write(
        path / "policy_engine.py",
        """
        def evaluate(policy, event):
            return []
        """,
    )
    _write(
        path / "test_policy_engine.py",
        """
        import unittest

        from policy_engine import evaluate


        class PolicyEngineTests(unittest.TestCase):
            def test_nested_conditions_and_actions(self):
                policy = {"rules": [
                    {"name": "docs", "when": {"all": [
                        {"field": "author.role", "eq": "staff"},
                        {"field": "path", "matches": "docs/*.md"},
                    ]}, "actions": ["allow", "label:docs"]},
                    {"name": "block", "when": {"field": "risk", "gte": 7}, "actions": ["manual-review"]},
                ]}
                event = {"author": {"role": "staff"}, "path": "docs/a.md", "risk": 3}
                self.assertEqual(evaluate(policy, event), ["allow", "label:docs"])


        if __name__ == "__main__":
            unittest.main()
        """,
    )
    _git_init(path)
    return f"""
    Implement policy_engine.evaluate so `{ctx.python_cmd} -m unittest -v` passes.

    Policy has a "rules" list. For each matching rule, append its action or
    actions to the result in rule order, removing duplicate actions while keeping
    first occurrence. Conditions support all, any, not, and leaves with
    dot-path field plus eq, neq, in, contains, gt, gte, lt, lte, matches using
    shell-style glob matching, or field-to-field comparisons such as gt_field
    and lte_field. Missing fields make a leaf false.
    """


def verify_workflow_rule_engine(path: Path, ctx: SetupContext, output: str) -> Verification:
    ok, visible = _verify_unittest(path, ctx, timeout=180)
    if not ok:
        return Verification(False, visible)
    ok, hidden = _verify_python_code(
        path,
        ctx,
        """
        from policy_engine import evaluate

        policy = {"rules": [
            {"name": "urgent", "when": {"any": [
                {"field": "labels", "contains": "incident"},
                {"field": "priority", "in": ["p0", "p1"]},
            ]}, "actions": ["page", "manual-review"]},
            {"name": "safe-bot", "when": {"all": [
                {"field": "author.type", "eq": "bot"},
                {"not": {"field": "changed_files", "contains": "prod.yaml"}},
            ]}, "actions": ["allow", "manual-review"]},
            {"name": "prod", "when": {"field": "path", "matches": "infra/prod/*"}, "actions": "manual-review"},
        ]}
        event = {
            "labels": ["incident", "customer"],
            "priority": "p2",
            "author": {"type": "bot"},
            "changed_files": ["README.md"],
            "path": "infra/prod/api.yaml",
        }
        assert evaluate(policy, event) == ["page", "manual-review", "allow"], evaluate(policy, event)
        assert evaluate(policy, {"author": {}, "path": "docs/x.md"}) == []
        dynamic = {"rules": [
            {"when": {"field": "duration_ms", "gt_field": "limits.warn_ms"}, "actions": ["slow"]},
            {"when": {"field": "duration_ms", "lte_field": "limits.max_ms"}, "actions": ["within-max"]},
        ]}
        assert evaluate(dynamic, {"duration_ms": 250, "limits": {"warn_ms": 200, "max_ms": 500}}) == ["slow", "within-max"]
        """,
        timeout=180,
    )
    return Verification(ok, f"visible:\n{visible}\nhidden:\n{hidden}")


def setup_template_renderer(path: Path, ctx: SetupContext) -> str:
    _write(
        path / "renderer.py",
        """
        def render(template, context):
            return template
        """,
    )
    _write(
        path / "test_renderer.py",
        """
        import unittest

        from renderer import render


        class RendererTests(unittest.TestCase):
            def test_escaped_variables_and_list_sections(self):
                template = "Hello {{name}}!{{#items}} {{.}}{{/items}}"
                self.assertEqual(render(template, {"name": "<Ada>", "items": ["x", "y"]}), "Hello &lt;Ada&gt;! x y")

            def test_inverted_section(self):
                self.assertEqual(render("{{^items}}empty{{/items}}", {"items": []}), "empty")


        if __name__ == "__main__":
            unittest.main()
        """,
    )
    _git_init(path)
    return f"""
    Implement renderer.render so `{ctx.python_cmd} -m unittest -v` passes.

    This is a tiny Mustache-style renderer. Support escaped double-brace
    variables, raw triple-brace variables, dot paths, the current-item dot,
    list sections, truthy dict/scalar sections, and inverted sections. Inside a
    section, resolve names against the current item first and then all parent
    contexts. Missing values render as an empty string. Use HTML escaping for
    normal variables, and support ampersand raw variables too.
    """


def verify_template_renderer(path: Path, ctx: SetupContext, output: str) -> Verification:
    ok, visible = _verify_unittest(path, ctx, timeout=180)
    if not ok:
        return Verification(False, visible)
    ok, hidden = _verify_python_code(
        path,
        ctx,
        """
        from renderer import render

        template = "{{#users}}{{name}}:{{#roles}} {{.}}{{/roles}}{{^roles}} none{{/roles}}\\n{{/users}}"
        context = {"users": [{"name": "Ada", "roles": ["admin", "dev"]}, {"name": "Bo", "roles": []}]}
        assert render(template, context) == "Ada: admin dev\\nBo: none\\n"
        assert render("{{title}} {{{title}}}", {"title": "<b>"}) == "&lt;b&gt; <b>"
        assert render("{{#user}}{{name}}/{{team.name}}{{/user}}", {"user": {"name": "Cy", "team": {"name": "Ops"}}}) == "Cy/Ops"
        assert render("{{#teams}}{{name}}:{{#members}}{{.}}@{{name}};{{/members}}{{/teams}}", {
            "teams": [{"name": "core", "members": ["ada", "bo"]}]
        }) == "core:ada@core;bo@core;"
        assert render("{{#orgs}}{{name}}/{{#teams}}{{name}}/{{#members}}{{.}}@{{name}}@{{domain}};{{/members}}{{/teams}}{{/orgs}}", {
            "domain": "example.com",
            "orgs": [{"name": "acme", "teams": [{"name": "core", "members": ["ada"]}]}],
        }) == "acme/core/ada@core@example.com;"
        assert render("{{& title}}", {"title": "<i>"}) == "<i>"
        assert render("{{missing}}", {}) == ""
        """,
        timeout=180,
    )
    return Verification(ok, f"visible:\n{visible}\nhidden:\n{hidden}")


def setup_weighted_interval_scheduler(path: Path, ctx: SetupContext) -> str:
    _write(
        path / "scheduler.py",
        """
        def best_schedule(jobs):
            return {"value": 0, "jobs": []}
        """,
    )
    _write(
        path / "test_scheduler.py",
        """
        import unittest

        from scheduler import best_schedule


        class SchedulerTests(unittest.TestCase):
            def test_weighted_non_overlapping_jobs(self):
                jobs = [
                    {"id": "a", "start": 0, "end": 3, "value": 5},
                    {"id": "b", "start": 3, "end": 5, "value": 6},
                    {"id": "c", "start": 1, "end": 5, "value": 12},
                ]
                self.assertEqual(best_schedule(jobs), {"value": 12, "jobs": ["c"]})

            def test_dependency_requires_compatible_prerequisite(self):
                jobs = [
                    {"id": "prep", "start": 0, "end": 2, "value": 3},
                    {"id": "ship", "start": 2, "end": 4, "value": 8, "requires": ["prep"]},
                    {"id": "rush", "start": 1, "end": 4, "value": 9},
                ]
                self.assertEqual(best_schedule(jobs), {"value": 11, "jobs": ["prep", "ship"]})


        if __name__ == "__main__":
            unittest.main()
        """,
    )
    _git_init(path)
    return f"""
    Implement scheduler.best_schedule so `{ctx.python_cmd} -m unittest -v`
    passes.

    Jobs have id, start, end, value, optional resource, and optional requires
    list. Pick a subset with maximum total value. Intervals are half-open, so
    end == start is compatible. Jobs conflict only when their intervals overlap
    on the same resource; missing resource means "default". If a job is
    selected, all required jobs must also be selected and must finish no later
    than the dependent job starts. Return
    {{"value": total, "jobs": ids}} with ids in chronological order. Break exact
    value ties by the lexicographically smallest id list.
    """


def verify_weighted_interval_scheduler(path: Path, ctx: SetupContext, output: str) -> Verification:
    ok, visible = _verify_unittest(path, ctx, timeout=180)
    if not ok:
        return Verification(False, visible)
    ok, hidden = _verify_python_code(
        path,
        ctx,
        """
        from scheduler import best_schedule

        jobs = [
            {"id": "a", "start": 0, "end": 2, "value": 4},
            {"id": "b", "start": 2, "end": 4, "value": 4},
            {"id": "c", "start": 4, "end": 6, "value": 4},
            {"id": "x", "start": 0, "end": 6, "value": 11},
            {"id": "d", "start": 6, "end": 8, "value": 1, "requires": ["b"]},
        ]
        assert best_schedule(jobs) == {"value": 13, "jobs": ["a", "b", "c", "d"]}, best_schedule(jobs)
        tie = [
            {"id": "b", "start": 0, "end": 2, "value": 5},
            {"id": "a", "start": 0, "end": 2, "value": 5},
        ]
        assert best_schedule(tie) == {"value": 5, "jobs": ["a"]}
        shared = [
            {"id": "prep", "start": 0, "end": 1, "value": 1},
            {"id": "test", "start": 1, "end": 2, "value": 5, "requires": ["prep"]},
            {"id": "deploy", "start": 2, "end": 3, "value": 5, "requires": ["prep"]},
            {"id": "shortcut", "start": 0, "end": 3, "value": 10},
        ]
        assert best_schedule(shared) == {"value": 11, "jobs": ["prep", "test", "deploy"]}, best_schedule(shared)
        resources = [
            {"id": "cpu-a", "start": 0, "end": 3, "value": 5, "resource": "cpu"},
            {"id": "gpu-a", "start": 0, "end": 3, "value": 6, "resource": "gpu"},
            {"id": "cpu-b", "start": 3, "end": 4, "value": 1, "resource": "cpu"},
        ]
        assert best_schedule(resources) == {"value": 12, "jobs": ["cpu-a", "gpu-a", "cpu-b"]}, best_schedule(resources)
        """,
        timeout=180,
    )
    return Verification(ok, f"visible:\n{visible}\nhidden:\n{hidden}")


def setup_event_state_replay(path: Path, ctx: SetupContext) -> str:
    _write(
        path / "tickets.py",
        """
        def replay_events(events):
            return {"status": "new", "assignee": None, "comments": [], "warnings": []}
        """,
    )
    _write(
        path / "test_tickets.py",
        """
        import unittest

        from tickets import replay_events


        class TicketReplayTests(unittest.TestCase):
            def test_sorts_events_and_ignores_duplicates(self):
                events = [
                    {"id": "2", "at": "2026-05-12T10:02:00Z", "type": "assign", "user": "Mira"},
                    {"id": "1", "at": "2026-05-12T10:00:00Z", "type": "open"},
                    {"id": "2", "at": "2026-05-12T10:03:00Z", "type": "assign", "user": "Nope"},
                ]
                self.assertEqual(replay_events(events)["assignee"], "Mira")

            def test_invalid_transition_warns(self):
                state = replay_events([{"id": "x", "at": "2026-05-12T10:00:00Z", "type": "close"}])
                self.assertEqual(state["status"], "new")
                self.assertEqual(state["warnings"], ["invalid:x:close"])


        if __name__ == "__main__":
            unittest.main()
        """,
    )
    _git_init(path)
    return f"""
    Implement tickets.replay_events so `{ctx.python_cmd} -m unittest -v` passes.

    Sort events by ISO timestamp, preserving input order for ties. Ignore
    duplicate event ids after their first chronological occurrence. Valid
    transitions: open from new; assign from open or assigned; unassign from
    assigned; resolve from open or assigned and clears the assignee; reopen from
    resolved; close from resolved. comment is always valid and appends its text.
    Invalid transitions append
    "invalid:<id>:<type>" to warnings and otherwise do nothing. Return status,
    assignee, comments, and warnings.
    """


def verify_event_state_replay(path: Path, ctx: SetupContext, output: str) -> Verification:
    ok, visible = _verify_unittest(path, ctx, timeout=180)
    if not ok:
        return Verification(False, visible)
    ok, hidden = _verify_python_code(
        path,
        ctx,
        """
        from tickets import replay_events

        events = [
            {"id": "4", "at": "2026-05-12T10:04:00Z", "type": "resolve"},
            {"id": "2", "at": "2026-05-12T10:02:00Z", "type": "comment", "text": "first"},
            {"id": "1", "at": "2026-05-12T10:00:00Z", "type": "open"},
            {"id": "3", "at": "2026-05-12T10:03:00Z", "type": "assign", "user": "Ana"},
            {"id": "5", "at": "2026-05-12T10:05:00Z", "type": "reopen"},
            {"id": "6", "at": "2026-05-12T10:06:00Z", "type": "close"},
            {"id": "7", "at": "2026-05-12T10:07:00Z", "type": "comment", "text": "after"},
        ]
        state = replay_events(events)
        assert state == {
            "status": "open",
            "assignee": "Ana",
            "comments": ["first", "after"],
            "warnings": ["invalid:6:close"],
        }, state
        state = replay_events([
            {"id": "a", "at": "2026-05-12T10:00:00Z", "type": "open"},
            {"id": "b", "at": "2026-05-12T10:01:00Z", "type": "assign", "user": "Lee"},
            {"id": "c", "at": "2026-05-12T10:02:00Z", "type": "unassign"},
            {"id": "d", "at": "2026-05-12T10:03:00Z", "type": "assign", "user": "Mira"},
            {"id": "e", "at": "2026-05-12T10:04:00Z", "type": "resolve"},
        ])
        assert state["status"] == "resolved" and state["assignee"] is None and state["warnings"] == [], state
        """,
        timeout=180,
    )
    return Verification(ok, f"visible:\n{visible}\nhidden:\n{hidden}")


def setup_grid_keys_path(path: Path, ctx: SetupContext) -> str:
    _write(
        path / "maze.py",
        """
        def shortest_path(grid):
            return None
        """,
    )
    _write(
        path / "test_maze.py",
        """
        import unittest

        from maze import shortest_path


        class MazeTests(unittest.TestCase):
            def test_simple_path(self):
                self.assertEqual(shortest_path(["S.E"]), 2)

            def test_key_opens_matching_door(self):
                grid = [
                    "S.a",
                    "##A",
                    "..E",
                ]
                self.assertEqual(shortest_path(grid), 4)


        if __name__ == "__main__":
            unittest.main()
        """,
    )
    _git_init(path)
    return f"""
    Implement maze.shortest_path so `{ctx.python_cmd} -m unittest -v` passes.

    Grid rows contain S start, E exit, . floor, # wall, digits 1-9, paired @
    portals, lowercase keys, and uppercase doors. Movement is 4-directional.
    Entering a digit cell costs that digit; entering any other passable cell
    costs 1. Entering an @ cell instantly moves to the other @ with no extra
    cost. A door can be entered only after collecting its matching lowercase key. Return the
    shortest step count to E, or None if unreachable. Treat collected keys as
    part of the search state.
    """


def verify_grid_keys_path(path: Path, ctx: SetupContext, output: str) -> Verification:
    ok, visible = _verify_unittest(path, ctx, timeout=180)
    if not ok:
        return Verification(False, visible)
    ok, hidden = _verify_python_code(
        path,
        ctx,
        """
        from maze import shortest_path

        assert shortest_path([
            "S..B",
            ".##.",
            ".aAb",
            "...E",
        ]) == 6
        assert shortest_path(["S#E", "###", "a.A"]) is None
        assert shortest_path(["Sb.A", ".#.#", ".a.E"]) == 5
        assert shortest_path(["S9E", "111"]) == 4
        assert shortest_path(["S@#", "###", "@.E"]) == 3
        """,
        timeout=180,
    )
    return Verification(ok, f"visible:\n{visible}\nhidden:\n{hidden}")


def setup_formula_engine(path: Path, ctx: SetupContext) -> str:
    _write(
        path / "formula.py",
        """
        class FormulaError(Exception):
            pass


        def evaluate(cells):
            return cells
        """,
    )
    _write(
        path / "test_formula.py",
        """
        import unittest

        from formula import evaluate


        class FormulaTests(unittest.TestCase):
            def test_arithmetic_refs_and_sum(self):
                cells = {"A1": 2, "A2": 3, "B1": "=A1 + A2 * 4", "B2": "=SUM(A1:A2)"}
                self.assertEqual(evaluate(cells), {"A1": 2, "A2": 3, "B1": 14, "B2": 5})

            def test_if_expression(self):
                self.assertEqual(evaluate({"A1": 5, "B1": "=IF(A1>3, 10, 20)"})["B1"], 10)


        if __name__ == "__main__":
            unittest.main()
        """,
    )
    _git_init(path)
    return f"""
    Implement formula.evaluate so `{ctx.python_cmd} -m unittest -v` passes.

    Cells is a dict like {{"A1": 2, "B1": "=A1+1"}}. Return a new dict with all
    formulas evaluated to numbers. Support +, -, *, /, unary +/-, parentheses, cell
    references, SUM(A1:A3) over one row or one column, and IF(condition, a, b)
    where condition uses >, >=, <, <=, ==, or !=. IF must evaluate only the
    chosen branch. Detect dependency cycles and invalid references by raising
    FormulaError.
    """


def verify_formula_engine(path: Path, ctx: SetupContext, output: str) -> Verification:
    ok, visible = _verify_unittest(path, ctx, timeout=180)
    if not ok:
        return Verification(False, visible)
    ok, hidden = _verify_python_code(
        path,
        ctx,
        """
        from formula import FormulaError, evaluate

        cells = {
            "A1": 1, "A2": 2, "A3": 3,
            "B1": "=SUM(A1:A3)",
            "B2": "=IF(B1>=6, B1/2, 0)",
            "C1": "=B2 + A3 * (A2 + 1)",
        }
        result = evaluate(cells)
        assert result["B1"] == 6 and result["B2"] == 3 and result["C1"] == 12, result
        assert evaluate({"A1": 1, "B1": "=IF(A1>0, 7, MISSING+1)"})["B1"] == 7
        assert evaluate({"A1": "=-1 + 2", "B1": "=-(A1 + 2)"}) == {"A1": 1, "B1": -3}
        for bad in ({"A1": "=B1", "B1": "=A1"}, {"A1": "=MISSING+1"}):
            try:
                evaluate(bad)
            except FormulaError:
                pass
            else:
                raise AssertionError(f"expected FormulaError for {bad!r}")
        """,
        timeout=180,
    )
    return Verification(ok, f"visible:\n{visible}\nhidden:\n{hidden}")


def setup_json_schema_validator(path: Path, ctx: SetupContext) -> str:
    _write(
        path / "schema_validator.py",
        """
        class ValidationError(Exception):
            pass


        def validate(schema, data):
            return True
        """,
    )
    _write(
        path / "test_schema_validator.py",
        """
        import unittest

        from schema_validator import ValidationError, validate


        class SchemaValidatorTests(unittest.TestCase):
            def test_object_required_and_types(self):
                schema = {
                    "type": "object",
                    "required": ["name", "age"],
                    "properties": {"name": {"type": "string"}, "age": {"type": "integer", "minimum": 0}},
                    "additionalProperties": False,
                }
                self.assertTrue(validate(schema, {"name": "Ada", "age": 37}))
                with self.assertRaises(ValidationError):
                    validate(schema, {"name": "Ada", "age": -1, "extra": True})


        if __name__ == "__main__":
            unittest.main()
        """,
    )
    _git_init(path)
    return f"""
    Implement schema_validator.validate so `{ctx.python_cmd} -m unittest -v`
    passes.

    Support a useful JSON Schema subset: type, required, properties,
    additionalProperties, items, enum, const, minimum, maximum, multipleOf, minLength,
    maxLength, minItems, maxItems, uniqueItems, minProperties, maxProperties,
    dependentRequired, propertyNames, patternProperties, contains, minContains,
    maxContains, pattern, format=email, allOf, anyOf, oneOf, not, if/then/else, and local refs such as
    #/definitions/User. Return True for valid data. Raise ValidationError for
    invalid data, including a message that mentions the failing path.
    """


def verify_json_schema_validator(path: Path, ctx: SetupContext, output: str) -> Verification:
    ok, visible = _verify_unittest(path, ctx, timeout=180)
    if not ok:
        return Verification(False, visible)
    ok, hidden = _verify_python_code(
        path,
        ctx,
        """
        from schema_validator import ValidationError, validate

        schema = {
            "definitions": {
                "User": {
                    "type": "object",
                    "required": ["id", "email"],
                    "properties": {
                        "id": {"type": "string", "pattern": "^u-[0-9]+$"},
                        "email": {"type": "string", "pattern": "@"},
                        "role": {"enum": ["admin", "member"]},
                    },
                    "additionalProperties": False,
                }
            },
            "type": "object",
            "required": ["owner", "tags"],
            "properties": {
                "owner": {"$ref": "#/definitions/User"},
                "tags": {"type": "array", "items": {"type": "string", "minLength": 2}},
                "mode": {"oneOf": [{"const": "auto"}, {"const": "manual"}]},
            },
        }
        assert validate(schema, {"owner": {"id": "u-42", "email": "a@b", "role": "admin"}, "tags": ["ci"], "mode": "auto"})
        try:
            validate(schema, {"owner": {"id": "bad", "email": "ab", "x": 1}, "tags": ["c"], "mode": "other"})
        except ValidationError as exc:
            assert "owner" in str(exc) or "tags" in str(exc) or "mode" in str(exc), str(exc)
        else:
            raise AssertionError("expected ValidationError")

        assert validate({"type": "array", "minItems": 2, "maxItems": 3, "uniqueItems": True, "items": {"type": "integer"}}, [1, 2])
        try:
            validate({"type": "array", "minItems": 2, "uniqueItems": True, "items": {"type": "integer"}}, [1, 1])
        except ValidationError:
            pass
        else:
            raise AssertionError("expected uniqueItems failure")

        billing = {
            "type": "object",
            "minProperties": 2,
            "maxProperties": 3,
            "dependentRequired": {"credit_card": ["billing_address"]},
        }
        assert validate(billing, {"name": "Ada", "credit_card": "123", "billing_address": "Moon"})
        try:
            validate(billing, {"name": "Ada", "credit_card": "123"})
        except ValidationError:
            pass
        else:
            raise AssertionError("expected dependentRequired failure")

        conditional = {
            "type": "object",
            "properties": {"kind": {"enum": ["point", "range"]}},
            "if": {"properties": {"kind": {"const": "range"}}, "required": ["kind"]},
            "then": {"required": ["min", "max"]},
            "else": {"not": {"required": ["min"]}},
        }
        assert validate(conditional, {"kind": "range", "min": 1, "max": 3})
        try:
            validate(conditional, {"kind": "range", "min": 1})
        except ValidationError:
            pass
        else:
            raise AssertionError("expected then failure")

        patterned = {
            "type": "object",
            "propertyNames": {"pattern": "^(x-|name$)"},
            "patternProperties": {"^x-": {"type": "integer"}},
            "additionalProperties": {"type": "string"},
        }
        assert validate(patterned, {"name": "Ada", "x-score": 7})
        try:
            validate(patterned, {"name": "Ada", "x-score": "high"})
        except ValidationError:
            pass
        else:
            raise AssertionError("expected patternProperties failure")

        array_contains = {"type": "array", "contains": {"type": "integer", "minimum": 10}, "minContains": 2, "maxContains": 3}
        assert validate(array_contains, [3, 10, 12, "x"])
        try:
            validate(array_contains, [3, 10, "x"])
        except ValidationError:
            pass
        else:
            raise AssertionError("expected minContains failure")

        assert validate({"type": "number", "multipleOf": 0.5}, 1.5)
        try:
            validate({"type": "number", "multipleOf": 0.5}, 1.3)
        except ValidationError:
            pass
        else:
            raise AssertionError("expected multipleOf failure")
        try:
            validate({"type": "string", "format": "email"}, "not-an-email")
        except ValidationError:
            pass
        else:
            raise AssertionError("expected email format failure")
        try:
            validate({"not": {"type": "string"}}, "blocked")
        except ValidationError:
            pass
        else:
            raise AssertionError("expected not failure")
        """,
        timeout=180,
    )
    return Verification(ok, f"visible:\n{visible}\nhidden:\n{hidden}")


def setup_cnf_sat_solver(path: Path, ctx: SetupContext) -> str:
    _write(
        path / "sat.py",
        """
        def solve_cnf(num_vars, clauses):
            return None
        """,
    )
    _write(
        path / "test_sat.py",
        """
        import unittest

        from sat import solve_cnf


        def satisfies(assignment, clauses):
            return all(any((lit > 0) == assignment[abs(lit)] for lit in clause) for clause in clauses)


        class SatTests(unittest.TestCase):
            def test_finds_satisfying_assignment(self):
                clauses = [[1, 2], [-1, 2], [1, -2]]
                assignment = solve_cnf(2, clauses)
                self.assertIsInstance(assignment, dict)
                self.assertEqual(set(assignment), {1, 2})
                self.assertTrue(satisfies(assignment, clauses))

            def test_unsat_returns_none(self):
                self.assertIsNone(solve_cnf(1, [[1], [-1]]))


        if __name__ == "__main__":
            unittest.main()
        """,
    )
    _git_init(path)
    return f"""
    Implement sat.solve_cnf so `{ctx.python_cmd} -m unittest -v` passes.

    Clauses are lists of signed ints. Return a complete dict mapping each
    variable 1..num_vars to a bool satisfying all clauses, or None if unsat.
    Literals must reference variables in 1..num_vars; invalid literals raise
    ValueError. Empty clauses make a formula unsat. Empty formulas are satisfiable. For
    deterministic results, return the lexicographically smallest complete
    assignment by variable order with False < True. Use a real
    backtracking/DPLL-style search; do not hard-code the tests.
    """


def verify_cnf_sat_solver(path: Path, ctx: SetupContext, output: str) -> Verification:
    ok, visible = _verify_unittest(path, ctx, timeout=180)
    if not ok:
        return Verification(False, visible)
    ok, hidden = _verify_python_code(
        path,
        ctx,
        """
        from sat import solve_cnf

        def check(n, clauses):
            assignment = solve_cnf(n, clauses)
            assert isinstance(assignment, dict), assignment
            assert set(assignment) == set(range(1, n + 1)), assignment
            assert all(isinstance(v, bool) for v in assignment.values()), assignment
            assert all(any((lit > 0) == assignment[abs(lit)] for lit in clause) for clause in clauses), assignment

        check(4, [[1, 2, 3], [-1, 2], [-2, 3], [-3, 4], [-4, 1]])
        check(6, [[1], [-1, 2], [-2, 3], [-3, 4], [-4, 5], [-5, 6]])
        assert solve_cnf(2, [[1, 2]]) == {1: False, 2: True}
        assert solve_cnf(2, [[], [1]]) is None
        empty = solve_cnf(3, [])
        assert isinstance(empty, dict) and set(empty) == {1, 2, 3}, empty
        try:
            solve_cnf(2, [[3]])
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for out-of-range variable")
        """,
        timeout=180,
    )
    return Verification(ok, f"visible:\n{visible}\nhidden:\n{hidden}")


def setup_log_query_language(path: Path, ctx: SetupContext) -> str:
    _write(
        path / "logql.py",
        """
        def query(lines, expression):
            return []
        """,
    )
    _write(
        path / "test_logql.py",
        """
        import unittest

        from logql import query


        class LogQLTests(unittest.TestCase):
            def test_boolean_expression(self):
                lines = [
                    '{"service":"api","level":"ERROR","duration_ms":120,"message":"timeout"}',
                    '{"service":"web","level":"INFO","duration_ms":12,"message":"ok"}',
                    '{"service":"api","level":"INFO","duration_ms":40,"message":"ok"}',
                ]
                rows = query(lines, 'service = "api" and (level = "ERROR" or duration_ms > 50)')
                self.assertEqual([row["message"] for row in rows], ["timeout"])


        if __name__ == "__main__":
            unittest.main()
        """,
    )
    _git_init(path)
    return f"""
    Implement logql.query so `{ctx.python_cmd} -m unittest -v` passes.

    Each input line is a JSON object. Parse the expression and return matching
    objects in input order. Operators: =, !=, >, >=, <, <=, contains, matches
    with shell globs, and in with array literals. Boolean and/or/not use
    parentheses. contains works on strings and arrays. Field names may be dot
    paths. Strings are double-quoted and support JSON-style escapes; numbers
    compare numerically. Boolean and null literals are supported.
    Boolean precedence is not > and > or.
    """


def verify_log_query_language(path: Path, ctx: SetupContext, output: str) -> Verification:
    ok, visible = _verify_unittest(path, ctx, timeout=180)
    if not ok:
        return Verification(False, visible)
    ok, hidden = _verify_python_code(
        path,
        ctx,
        """
        from logql import query

        lines = [
            '{"service":"api","level":"ERROR","duration_ms":90,"message":"db timeout","meta":{"region":"eu"}}',
            '{"service":"api","level":"WARN","duration_ms":40,"message":"cache warm","meta":{"region":"us"}}',
            '{"service":"worker","level":"ERROR","duration_ms":200,"message":"job failed","meta":{"region":"eu"}}',
            '{"service":"web","level":"INFO","duration_ms":10,"message":"ok","meta":{"region":"eu"}}',
        ]
        expr = 'meta.region = "eu" and not (service = "web") and (message contains "fail" or duration_ms >= 90)'
        assert [row["service"] for row in query(lines, expr)] == ["api", "worker"]
        assert [row["service"] for row in query(lines, 'service matches "a*" or service = "web"')] == ["api", "api", "web"]
        tagged = [
            '{"service":"api","tags":["prod","db"]}',
            '{"service":"web","tags":["edge"]}',
        ]
        assert [row["service"] for row in query(tagged, 'tags contains "prod"')] == ["api"]
        flags = [
            '{"service":"api","ok":true,"meta":{"trace":null}}',
            '{"service":"web","ok":false,"meta":{"trace":"abc"}}',
        ]
        assert [row["service"] for row in query(flags, 'ok = true and meta.trace = null')] == ["api"]
        assert [row["service"] for row in query(lines, 'service in ["api", "worker"] and duration_ms >= 90')] == ["api", "worker"]
        escaped = [
            '{"service":"api","message":"user said \\"hi\\""}',
            '{"service":"web","message":"plain"}',
        ]
        assert [row["service"] for row in query(escaped, 'message contains "\\\"hi\\\""')] == ["api"]
        """,
        timeout=180,
    )
    return Verification(ok, f"visible:\n{visible}\nhidden:\n{hidden}")


def setup_crdt_text_merge(path: Path, ctx: SetupContext) -> str:
    _write(
        path / "crdt.py",
        """
        def merge_ops(ops):
            return ""
        """,
    )
    _write(
        path / "test_crdt.py",
        """
        import unittest

        from crdt import merge_ops


        class CRDTTests(unittest.TestCase):
            def test_concurrent_inserts_are_deterministic(self):
                ops = [
                    {"id": "a1", "site": "a", "seq": 1, "op": "insert", "after": None, "char": "H"},
                    {"id": "b1", "site": "b", "seq": 1, "op": "insert", "after": "a1", "char": "i"},
                    {"id": "c1", "site": "c", "seq": 1, "op": "insert", "after": "a1", "char": "!"},
                ]
                self.assertEqual(merge_ops(ops), "Hi!")

            def test_delete_tombstones_character(self):
                ops = [
                    {"id": "a1", "site": "a", "seq": 1, "op": "insert", "after": None, "char": "A"},
                    {"id": "a2", "site": "a", "seq": 2, "op": "delete", "target": "a1"},
                ]
                self.assertEqual(merge_ops(ops), "")


        if __name__ == "__main__":
            unittest.main()
        """,
    )
    _git_init(path)
    return f"""
    Implement crdt.merge_ops so `{ctx.python_cmd} -m unittest -v` passes.

    Operations form a tiny ordered text CRDT. Insert ops have id, site, seq,
    after, and char. Delete ops have id, site, seq, and target. Apply operations
    after their referenced ids exist, regardless of input order. Missing refs
    make an op pending until resolved; unresolved ops are ignored. Concurrent
    inserts after the same id are ordered by (site, seq, id), before later
    descendants. Deletes create tombstones and are idempotent. Return visible
    text.
    """


def verify_crdt_text_merge(path: Path, ctx: SetupContext, output: str) -> Verification:
    ok, visible = _verify_unittest(path, ctx, timeout=180)
    if not ok:
        return Verification(False, visible)
    ok, hidden = _verify_python_code(
        path,
        ctx,
        """
        from crdt import merge_ops

        ops = [
            {"id": "b2", "site": "b", "seq": 2, "op": "insert", "after": "b1", "char": "y"},
            {"id": "a1", "site": "a", "seq": 1, "op": "insert", "after": None, "char": "h"},
            {"id": "b1", "site": "b", "seq": 1, "op": "insert", "after": "a1", "char": "e"},
            {"id": "c1", "site": "c", "seq": 1, "op": "insert", "after": "a1", "char": "a"},
            {"id": "a2", "site": "a", "seq": 2, "op": "insert", "after": "b2", "char": "!"},
            {"id": "d1", "site": "d", "seq": 1, "op": "delete", "target": "c1"},
            {"id": "d2", "site": "d", "seq": 2, "op": "delete", "target": "missing"},
        ]
        assert merge_ops(ops) == "hey!", merge_ops(ops)
        siblings = [
            {"id": "a1", "site": "a", "seq": 1, "op": "insert", "after": None, "char": "h"},
            {"id": "b1", "site": "b", "seq": 1, "op": "insert", "after": "a1", "char": "e"},
            {"id": "c1", "site": "c", "seq": 1, "op": "insert", "after": "a1", "char": "a"},
            {"id": "b2", "site": "b", "seq": 2, "op": "insert", "after": "b1", "char": "y"},
        ]
        assert merge_ops(siblings) == "heay", merge_ops(siblings)
        assert merge_ops([{"id": "x", "site": "x", "seq": 1, "op": "insert", "after": "missing", "char": "?"}]) == ""
        """,
        timeout=180,
    )
    return Verification(ok, f"visible:\n{visible}\nhidden:\n{hidden}")


AGENTIC_TASKS: tuple[BenchmarkTask, ...] = (
    BenchmarkTask(
        id="self_edit_smoke",
        category="canary",
        objective="Verify Hermes can safely make a small code change after prompt/config/model/code changes.",
        fixture="Tiny Python repo with failing unittest coverage around median calculation.",
        runner_contract="Run live Hermes in a temp git repo; capture transcript, diff, and unittest verifier.",
        pass_fail="Unittest passes and only order_stats.py changes.",
        severity="P0",
        blocks_auto_commit=True,
        difficulty="smoke",
        timeout_seconds=300,
        max_turns=35,
        tags=("c1", "smoke", "agentic", "gate"),
        setup=setup_self_edit_smoke,
        verify=verify_self_edit_smoke,
    ),
    BenchmarkTask(
        id="processing_pipeline",
        category="agentic-intelligence",
        objective="Exercise multi-file shell/Python pipeline debugging.",
        fixture="Broken extract/transform/report pipeline with executable and Python bugs.",
        runner_contract="Run live Hermes, then execute ./run_pipeline.sh as verifier.",
        pass_fail="Pipeline succeeds and output/report.txt exactly matches expected report.",
        severity="P1",
        blocks_auto_commit=True,
        difficulty="easy",
        timeout_seconds=420,
        max_turns=45,
        tags=("c4", "agentic", "gate", "nightly"),
        setup=setup_processing_pipeline,
        verify=verify_processing_pipeline,
    ),
    BenchmarkTask(
        id="argparse_feature",
        category="agentic-intelligence",
        objective="Measure small feature implementation with CLI tests.",
        fixture="Word-count CLI missing --limit behavior.",
        runner_contract="Run live Hermes; verifier runs unittest invoking subprocess CLI.",
        pass_fail="CLI preserves old behavior and supports --limit N.",
        severity="P1",
        blocks_auto_commit=True,
        difficulty="easy",
        timeout_seconds=360,
        max_turns=40,
        tags=("c4", "agentic", "gate", "nightly"),
        setup=setup_argparse_feature,
        verify=verify_argparse_feature,
    ),
    BenchmarkTask(
        id="json_patch_engine",
        category="agentic-calibration",
        objective="Measure robust implementation against hidden JSON Pointer and JSON Patch edge cases.",
        fixture="JSON Patch skeleton with visible examples and hidden mutation/error checks.",
        runner_contract="Run live Hermes; verifier runs visible unittest plus hidden Python assertions.",
        pass_fail="All JSON Patch operations work without mutating inputs and invalid cases raise JsonPatchError.",
        severity="P1",
        blocks_auto_commit=True,
        difficulty="hard",
        timeout_seconds=720,
        max_turns=70,
        tags=("c5", "agentic", "nightly", "hard", "calibration"),
        setup=setup_json_patch_engine,
        verify=verify_json_patch_engine,
    ),
    BenchmarkTask(
        id="semver_range_resolver",
        category="agentic-calibration",
        objective="Measure range parsing, semantic version ordering, and hidden pre-release/build edge cases.",
        fixture="SemVer resolver skeleton with visible range tests and hidden edge tests.",
        runner_contract="Run live Hermes; verifier runs visible unittest plus hidden Python assertions.",
        pass_fail="select_version returns the highest satisfying version across exact, comparator, caret, tilde, AND, and OR constraints.",
        severity="P1",
        blocks_auto_commit=True,
        difficulty="hard",
        timeout_seconds=720,
        max_turns=70,
        tags=("c5", "agentic", "nightly", "hard", "calibration"),
        setup=setup_semver_range_resolver,
        verify=verify_semver_range_resolver,
    ),
    BenchmarkTask(
        id="dependency_resolver",
        category="agentic-calibration",
        objective="Measure constraint solving with transitive dependencies and backtracking.",
        fixture="Package resolver skeleton with visible and hidden conflict cases.",
        runner_contract="Run live Hermes; verifier runs visible unittest plus hidden Python assertions.",
        pass_fail="resolve chooses highest compatible versions and raises ResolutionError when no solution exists.",
        severity="P1",
        blocks_auto_commit=True,
        difficulty="hard",
        timeout_seconds=720,
        max_turns=70,
        tags=("c5", "agentic", "nightly", "hard", "calibration"),
        setup=setup_dependency_resolver,
        verify=verify_dependency_resolver,
    ),
    BenchmarkTask(
        id="unified_diff_apply",
        category="agentic-calibration",
        objective="Measure exact parser/patch application behavior over multi-file unified diffs.",
        fixture="Unified diff applier skeleton with visible update and hidden create/delete/error checks.",
        runner_contract="Run live Hermes; verifier runs visible unittest plus hidden Python assertions.",
        pass_fail="apply_unified_diff applies multiple hunks/files exactly, creates/deletes files, and raises PatchApplyError on context mismatch.",
        severity="P1",
        blocks_auto_commit=True,
        difficulty="hard",
        timeout_seconds=720,
        max_turns=70,
        tags=("c5", "agentic", "nightly", "hard", "calibration"),
        setup=setup_unified_diff_apply,
        verify=verify_unified_diff_apply,
    ),
    BenchmarkTask(
        id="workflow_rule_engine",
        category="agentic-calibration",
        objective="Measure nested logical evaluation and deterministic action aggregation.",
        fixture="Policy engine skeleton with visible rules and hidden nested-condition cases.",
        runner_contract="Run live Hermes; verifier runs visible unittest plus hidden Python assertions.",
        pass_fail="evaluate returns deduplicated actions in rule order for all supported condition operators.",
        severity="P1",
        blocks_auto_commit=True,
        difficulty="hard",
        timeout_seconds=600,
        max_turns=60,
        tags=("c5", "agentic", "nightly", "hard", "calibration"),
        setup=setup_workflow_rule_engine,
        verify=verify_workflow_rule_engine,
    ),
    BenchmarkTask(
        id="template_renderer",
        category="agentic-calibration",
        objective="Measure recursive parsing, HTML escaping, and section semantics.",
        fixture="Mustache-style renderer skeleton with visible and hidden nested-section tests.",
        runner_contract="Run live Hermes; verifier runs visible unittest plus hidden Python assertions.",
        pass_fail="render handles escaped/raw variables, dot paths, current item, truthy/list sections, and inverted sections.",
        severity="P1",
        blocks_auto_commit=True,
        difficulty="hard",
        timeout_seconds=720,
        max_turns=70,
        tags=("c5", "agentic", "nightly", "hard", "calibration"),
        setup=setup_template_renderer,
        verify=verify_template_renderer,
    ),
    BenchmarkTask(
        id="weighted_interval_scheduler",
        category="agentic-calibration",
        objective="Measure optimization with interval compatibility, prerequisites, and tie-breaking.",
        fixture="Scheduler skeleton with visible and hidden dependency/tie cases.",
        runner_contract="Run live Hermes; verifier runs visible unittest plus hidden Python assertions.",
        pass_fail="best_schedule returns the maximum-value compatible job set and deterministic ties.",
        severity="P1",
        blocks_auto_commit=True,
        difficulty="hard",
        timeout_seconds=720,
        max_turns=70,
        tags=("c5", "agentic", "nightly", "hard", "calibration"),
        setup=setup_weighted_interval_scheduler,
        verify=verify_weighted_interval_scheduler,
    ),
    BenchmarkTask(
        id="event_state_replay",
        category="agentic-calibration",
        objective="Measure event ordering, idempotency, transition validation, and warning generation.",
        fixture="Ticket event reducer skeleton with visible and hidden replay cases.",
        runner_contract="Run live Hermes; verifier runs visible unittest plus hidden Python assertions.",
        pass_fail="replay_events returns deterministic status, assignee, comments, and invalid-transition warnings.",
        severity="P1",
        blocks_auto_commit=True,
        difficulty="hard",
        timeout_seconds=600,
        max_turns=60,
        tags=("c5", "agentic", "nightly", "hard", "calibration"),
        setup=setup_event_state_replay,
        verify=verify_event_state_replay,
    ),
    BenchmarkTask(
        id="grid_keys_path",
        category="agentic-calibration",
        objective="Measure shortest-path search with key inventory as part of state.",
        fixture="Key-and-door maze solver skeleton with visible and hidden reachability cases.",
        runner_contract="Run live Hermes; verifier runs visible unittest plus hidden Python assertions.",
        pass_fail="shortest_path returns the shortest route to E or None while respecting keys, doors, walls, and revisits.",
        severity="P1",
        blocks_auto_commit=True,
        difficulty="hard",
        timeout_seconds=600,
        max_turns=60,
        tags=("c5", "agentic", "nightly", "hard", "calibration"),
        setup=setup_grid_keys_path,
        verify=verify_grid_keys_path,
    ),
    BenchmarkTask(
        id="formula_engine",
        category="agentic-calibration",
        objective="Measure expression evaluation, dependency ordering, ranges, IF logic, and cycle detection.",
        fixture="Spreadsheet formula engine skeleton with visible and hidden dependency tests.",
        runner_contract="Run live Hermes; verifier runs visible unittest plus hidden Python assertions.",
        pass_fail="evaluate computes all formulas and raises FormulaError for cycles and invalid references.",
        severity="P1",
        blocks_auto_commit=True,
        difficulty="hard",
        timeout_seconds=720,
        max_turns=70,
        tags=("c5", "agentic", "nightly", "hard", "calibration"),
        setup=setup_formula_engine,
        verify=verify_formula_engine,
    ),
    BenchmarkTask(
        id="json_schema_validator",
        category="agentic-calibration",
        objective="Measure recursive JSON Schema validation with refs and composition.",
        fixture="JSON Schema validator skeleton with visible object checks and hidden ref/composition cases.",
        runner_contract="Run live Hermes; verifier runs visible unittest plus hidden Python assertions.",
        pass_fail="validate returns True for valid data and raises ValidationError with useful paths for invalid data.",
        severity="P1",
        blocks_auto_commit=True,
        difficulty="hard",
        timeout_seconds=720,
        max_turns=70,
        tags=("c5", "agentic", "nightly", "hard", "calibration"),
        setup=setup_json_schema_validator,
        verify=verify_json_schema_validator,
    ),
    BenchmarkTask(
        id="cnf_sat_solver",
        category="agentic-calibration",
        objective="Measure complete boolean satisfiability search and unsat handling.",
        fixture="CNF SAT solver skeleton with visible and hidden formulas.",
        runner_contract="Run live Hermes; verifier runs visible unittest plus hidden Python assertions.",
        pass_fail="solve_cnf returns a complete satisfying assignment or None for unsatisfiable formulas.",
        severity="P1",
        blocks_auto_commit=True,
        difficulty="hard",
        timeout_seconds=720,
        max_turns=70,
        tags=("c5", "agentic", "nightly", "hard", "calibration"),
        setup=setup_cnf_sat_solver,
        verify=verify_cnf_sat_solver,
    ),
    BenchmarkTask(
        id="log_query_language",
        category="agentic-calibration",
        objective="Measure expression parsing, precedence, dot-path lookup, and typed comparisons.",
        fixture="JSONL query engine skeleton with visible and hidden boolean query cases.",
        runner_contract="Run live Hermes; verifier runs visible unittest plus hidden Python assertions.",
        pass_fail="query returns matching JSON objects in input order for the supported query language.",
        severity="P1",
        blocks_auto_commit=True,
        difficulty="hard",
        timeout_seconds=720,
        max_turns=70,
        tags=("c5", "agentic", "nightly", "hard", "calibration"),
        setup=setup_log_query_language,
        verify=verify_log_query_language,
    ),
    BenchmarkTask(
        id="crdt_text_merge",
        category="agentic-calibration",
        objective="Measure deterministic merge behavior for out-of-order collaborative text operations.",
        fixture="CRDT text merge skeleton with visible and hidden ordering/tombstone cases.",
        runner_contract="Run live Hermes; verifier runs visible unittest plus hidden Python assertions.",
        pass_fail="merge_ops applies resolvable inserts/deletes deterministically and ignores unresolved operations.",
        severity="P1",
        blocks_auto_commit=True,
        difficulty="hard",
        timeout_seconds=720,
        max_turns=70,
        tags=("c5", "agentic", "nightly", "hard", "calibration"),
        setup=setup_crdt_text_merge,
        verify=verify_crdt_text_merge,
    ),
)


INSTRUCTION_TASKS: tuple[BenchmarkTask, ...] = (
    BenchmarkTask(
        id="prompt_stop_contract",
        category="canary",
        objective="Catch corrected behavioral regressions: execute Phase 1 gathering then stop cleanly.",
        fixture="Synthetic Calendar/Keep/REWE/TickTick files with read-only constraints.",
        runner_contract="Run live Hermes in read-only fixture; inspect final output and git diff.",
        pass_fail="Final kickoff MESSAGE covers required fields, no future-intent ending, no file edits.",
        severity="P0",
        blocks_auto_commit=True,
        difficulty="smoke",
        timeout_seconds=300,
        max_turns=35,
        tags=("c2", "smoke", "instruction", "gate"),
        setup=setup_prompt_stop_contract,
        verify=verify_prompt_stop_contract,
    ),
    BenchmarkTask(
        id="json_tool_report",
        category="instruction-tool-calling",
        objective="Measure exact local data extraction and JSON file production.",
        fixture="CSV plus Markdown notes.",
        runner_contract="Run live Hermes; verifier parses report.json.",
        pass_fail="report.json exactly matches expected schema and values.",
        severity="P1",
        blocks_auto_commit=True,
        difficulty="easy",
        timeout_seconds=300,
        max_turns=30,
        tags=("c4", "instruction", "gate", "nightly"),
        setup=setup_json_tool_report,
        verify=verify_json_tool_report,
    ),
    BenchmarkTask(
        id="tool_call_extraction",
        category="instruction-tool-calling",
        objective="Measure deterministic extraction of textual tool calls without executing them.",
        fixture="Raw model output containing Hermes-style <tool_call> blocks.",
        runner_contract="Run live Hermes; verifier parses tools.json.",
        pass_fail="Tool names, arguments, and order exactly match fixture.",
        severity="P1",
        blocks_auto_commit=True,
        difficulty="medium",
        timeout_seconds=300,
        max_turns=30,
        tags=("c4", "instruction", "gate", "nightly"),
        setup=setup_tool_call_extraction,
        verify=verify_tool_call_extraction,
    ),
)


TASKS: dict[str, BenchmarkTask] = {task.id: task for task in (*AGENTIC_TASKS, *INSTRUCTION_TASKS)}

SUITES: dict[str, tuple[str, ...]] = {
    "smoke": ("self_edit_smoke", "prompt_stop_contract"),
    "gate": (
        "self_edit_smoke",
        "prompt_stop_contract",
        "processing_pipeline",
        "argparse_feature",
        "json_tool_report",
        "tool_call_extraction",
    ),
    "agentic": tuple(task.id for task in AGENTIC_TASKS),
    "instruction": tuple(task.id for task in INSTRUCTION_TASKS),
    "nightly": tuple(TASKS),
}


def selected_tasks(suite: str | None = None, task_ids: Iterable[str] | None = None) -> list[BenchmarkTask]:
    if task_ids:
        ids = list(task_ids)
    else:
        ids = list(SUITES[suite or "smoke"])

    unknown = [task_id for task_id in ids if task_id not in TASKS]
    if unknown:
        raise KeyError(f"unknown benchmark task(s): {', '.join(unknown)}")
    return [TASKS[task_id] for task_id in ids]


def task_manifest() -> list[dict[str, object]]:
    return [
        {
            "id": task.id,
            "category": task.category,
            "objective": task.objective,
            "fixture": task.fixture,
            "runner_contract": task.runner_contract,
            "pass_fail": task.pass_fail,
            "severity": task.severity,
            "blocks_auto_commit": task.blocks_auto_commit,
            "difficulty": task.difficulty,
            "timeout_seconds": task.timeout_seconds,
            "max_turns": task.max_turns,
            "tags": list(task.tags),
        }
        for task in TASKS.values()
    ]

