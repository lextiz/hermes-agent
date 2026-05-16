# Hermes Self-Change Benchmarks

Local live-profile regression suite for Hermes changes that affect prompts,
config, model selection, code, memory behavior, context handling, tool calling,
and end-to-end task quality.

This suite is not wired into CI yet. It runs Hermes against temporary fixtures,
captures transcripts/diffs, and uses deterministic verifiers to decide pass/fail.

## Quick Gate

Run the two P0 canaries within the 5-minute local budget:

```bash
python scripts/run_self_change_benchmarks.py --suite smoke --max-seconds 300
```

The runner uses the live default profile by default. Do not pass `--profile`
unless you intentionally want to compare another profile.

The self-improvement cron gate calls the same runner through:

```bash
~/.hermes/self-improvement/eval-suite/run.sh --scope all
```

Scope mapping used by `eval_gate.py` for the 5-minute local gate:

- `prompt`, `config`, `model`, `code`, and `all` -> `smoke`
- set `HERMES_SELF_CHANGE_NIGHTLY=1` or pass `--nightly` to run `nightly`

## Broader Gates

Fast representative gate:

```bash
python scripts/run_self_change_benchmarks.py --suite gate --max-seconds 300
```

Nightly/full suite:

```bash
python scripts/run_self_change_benchmarks.py --suite nightly
```

Agentic-only and instruction/tool-only subsets:

```bash
python scripts/run_self_change_benchmarks.py --suite agentic
python scripts/run_self_change_benchmarks.py --suite instruction
```

List task contracts:

```bash
python scripts/run_self_change_benchmarks.py --list
python scripts/run_self_change_benchmarks.py --manifest
```

## Baselines

Once the Gemma 4 baseline stabilizes, save a baseline:

```bash
python scripts/run_self_change_benchmarks.py --suite nightly \
  --write-baseline benchmarks/self_change/baselines/gemma4-26b-q6.json
```

Compare future runs:

```bash
python scripts/run_self_change_benchmarks.py --suite nightly \
  --baseline benchmarks/self_change/baselines/gemma4-26b-q6.json
```

A blocking task failure or a regression against a passed baseline task exits
non-zero.

OpenAI fallback note: as of the current OpenAI model docs, the mini model in
the GPT-5.5 family comparison is `gpt-5.4-mini`; there is no documented
`gpt-5.5-mini` model ID. Keep the live Hermes fallback config explicit rather
than relying on an invented alias.

## Suites

- `smoke`: C1 self-edit smoke and C2 prompt stop-contract.
- `gate`: C1/C2 plus a small representative C4 sample intended for a short local gate.
- `agentic`: C1 plus representative Terminal-Bench/SWE-Bench-style tasks.
- `instruction`: C2 plus instruction-following and tool-calling tasks.
- `nightly`: all tasks.

All current tasks are blocking because the desired policy is: block any change
when the selected benchmark suite fails.

