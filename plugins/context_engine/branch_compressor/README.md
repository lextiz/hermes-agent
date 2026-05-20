# Branch-Aware Context Compressor

Experimental opt-in context engine for long exploratory conversations.

Enable it with:

```yaml
context:
  engine: "branch_compressor"
  branch_compressor:
    enabled: true
    model: null
    planner_model: null
    planner_max_tokens: 12000
    planner_repair_attempts: 2
    planner_group_max_chars: 700
    branch_summary_max_tokens: 4000
    branch_summary_repair_attempts: 1
    branch_source_max_chars: 6000
    llm_timeout: 300
    llm_call_retries: 3
    llm_retry_delay_seconds: 5
    min_branch_chars: 1000
    max_branch_summary_chars: 1200
    include_negative_findings: true
    preserve_failed_branch_details: false
    telemetry_enabled: true
    log_branch_details: false
```

The engine reuses the default compressor's protected head, protected recent
tail, token counting, redaction, auxiliary model calls, tool-pair sanitizer,
and media pruning. It only changes the summary strategy for the compacted
middle: an auxiliary LLM plans attempt branches and classifies them as
contributing, failed-but-relevant, omitted, or unknown branches. Invalid planner
JSON is repaired in the same short conversation when possible. Each planned
branch is then summarized by its own auxiliary LLM call, and Hermes composes
the final handoff summary deterministically. If planner or branch-summary repair
still fails, the engine falls back to Hermes' default compressor instead of
emitting a weak branch-aware summary.

`preserve_failed_branch_details: false` does not delete session history. It
only keeps raw failed-branch logs out of active model context and retains a
short negative finding instead.
