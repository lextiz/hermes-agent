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
    planner_max_tokens: 8000
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
contributing, failed-but-relevant, omitted, or unknown branches. If the planner
returns invalid JSON, the engine falls back to a single unknown branch instead
of making semantic guesses.

`preserve_failed_branch_details: false` does not delete session history. It
only keeps raw failed-branch logs out of active model context and retains a
short negative finding instead.
