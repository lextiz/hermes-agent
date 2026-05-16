# Delegate / Validate / Escalate Notes

## Local Benchmark Runs

Final quality-fixed run:

- Run: `20260516-dve-code-nightly-final-blue-complete20-r2`
- Branch: `codex/hermes-dve-benchmark-generous`
- Profile: `bench-20260515-dve-code-combined-optimized`
- Suite: `nightly`
- Effective timeouts: `--timeout-multiplier 3`
- Effective max turns: `--max-turns-multiplier 4`
- Score: `15/20`
- Duration: `5306.48s`
- Failed: `json_patch_engine`, `workflow_rule_engine`, `event_state_replay`, `formula_engine`, `log_query_language`
- Timed out tasks: none

Observed DVE activity in the final run:

- Delegation decision calls: `20`
- Delegation accepted: `13`
- Delegation declined: `7`
- Validation decision calls: `20`
- Validation verdicts: `pass=7`, `uncertain=11`, `fail=2`
- Main-turn escalation decision calls: `4`
- Main-turn escalation retries: `4`
- Delegated escalation retry targets: `22`
- Escalation target used: `gpt-5.5/openai` (`26` retry target log entries)
- Post-escalation validations: `8`
- Guardrail halts: `22`
- Post-success mutation guard armed: `17`
- Background review skipped: `1`

Quality fixes after the first 2026-05-16 observational run:

- Added a configurable same-read-only-tool success streak guardrail. This is generic and catches long runs of varied read-only calls, such as repeated `session_search`, without relying on benchmark task ids.
- Made the post-success mutation guard conservative when earlier `execute_code` may have changed files without exposing exact changed paths. In that case Hermes leaves the freeze guard unarmed instead of blocking later legitimate repairs based on incomplete mutation history.
- Focused validation after the fixes: `236 passed`; `ruff check` passed for touched files; touched files compiled.

Observational run before the post-success guard fix:

- Run: `20260516-dve-code-nightly-final-blue-complete20-r1`
- Score: `14/20`
- Duration: `5765.74s`
- Failed: `semver_range_resolver`, `unified_diff_apply`, `event_state_replay`, `formula_engine`, `json_schema_validator`, `log_query_language`
- Timed out tasks: none

Final clean generic run:

- Run: `20260515-dve-code-nightly-clean-generic-complete20-r1`
- Profile: `bench-20260515-dve-code-combined-optimized`
- Suite: `nightly`
- Effective timeouts: `--timeout-multiplier 3`
- Effective max turns: `--max-turns-multiplier 4`
- Score: `15/20`
- Failed: `workflow_rule_engine`, `event_state_replay`, `formula_engine`, `json_schema_validator`, `log_query_language`

Diagnostic pre-cleanup run:

- Run: `20260515-dve-code-nightly-combined-optimized-complete20-r8`
- Score: `18/20`
- Failed: `json_schema_validator`, `log_query_language`
- Caveat: this run happened before the final generic cleanup that removed benchmark-flavored task hints.

Observed DVE activity in the clean run:

- Delegation decision calls: `22`
- Delegation accepted: `20`
- Validation decision calls: `21`
- Main-turn escalation decision calls: `8`
- Delegated escalation retry targets: `27`
- Escalation target used: `gpt-5.5/openai`
- Subagent timeouts: `1`

## Failure Notes

- `workflow_rule_engine`: validation missed singular string `action` versus list `actions`; hidden verifier observed characters from `"manual-review"` being appended individually.
- `event_state_replay`: hidden verifier expected `reopen` to restore the last assignee, but the task prompt does not state that behavior.
- `formula_engine`: validation missed the stated lazy `IF` branch requirement.
- `json_schema_validator`: delegation and escalation fired, but one subagent looped on repeated search and timed out before the parent also timed out.
- `log_query_language`: recurring parser/evaluator miss from prior runs.

## Draft Upstream Comments

PR #25530:

> I tested this locally with Gemma 4 as the default model plus configured stronger escalation. Per-call delegate routing is useful, but quality-based escalation still needs a compact decision packet and validation hook so stronger models are used only for the failed slice.

Issue #25699:

> A config-level delegation pool plus per-call override seems like the right split. In my local runs, the important observability was recording the model/provider actually used for each decision call, delegated worker, and escalation retry.

Issue #25700:

> Per-call base_url override matters for mixed local/default plus stronger remote escalation setups. I also hit the related need to log effective base_url/provider/model for delegated escalation, otherwise benchmark wins are hard to interpret.

Issue #356:

> I prototyped a compact validation judge path using only task criteria, diff/output, tool logs, and verifier output. The judge caught some failures and triggered repair, but missed singular-vs-list and lazy-branch semantics until the validation policy was made more explicit.

Issue #479:

> Best-of-N would pair well with the validation judge, but the key piece is making the judge packet small and evidence-based. In local Gemma 4 runs, judging without observable repair/escalation hooks was not enough to explain score movement.

Issue #25689:

> I reproduced the broader shape: task-quality failures and stale/no-progress behavior need an escalation decision path separate from provider transport fallback. A timed-out delegated worker should emit a compact failure packet for retry.

Issue #24782:

> In local mixed-provider tests, delegated escalation must use the fallback/escalation target provider and base_url, not inherit the parent local base_url. Logging the effective retry target made this easy to verify.
