# Branch-Aware Context Compressor

Experimental opt-in context engine for long exploratory conversations.

Enable it with:

```yaml
context:
  engine: "branch_compressor"
  branch_compressor:
    enabled: true
    model: null
    min_branch_chars: 1000
    max_branch_summary_chars: 1200
    include_negative_findings: true
    preserve_failed_branch_details: false
```

The engine reuses the default compressor's protected head, protected recent
tail, token counting, redaction, auxiliary model calls, tool-pair sanitizer,
and media pruning. It only changes the summary strategy for the compacted
middle: messages are grouped into attempt branches, classified with simple
heuristics, and summarized as contributing, failed-but-relevant, omitted, or
unknown branches.

`preserve_failed_branch_details: false` does not delete session history. It
only keeps raw failed-branch logs out of active model context and retains a
short negative finding instead.
