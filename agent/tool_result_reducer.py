"""Configurable reduction for large tool results before prompt insertion."""

from __future__ import annotations

import contextvars
import fnmatch
import json
import logging
import re
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

from agent.auxiliary_client import call_llm

logger = logging.getLogger(__name__)

_REDUCER_ACTIVE: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "tool_result_reducer_active",
    default=False,
)

_DEFAULT_MIN_CHARS = 4_000
_DEFAULT_MAX_REDUCED_CHARS = 4_000
_MAX_ARGS_CHARS = 2_000
_MAX_USER_INTENT_CHARS = 1_000
_MAX_RAW_PROMPT_CHARS = 120_000
_MAX_IMPORTANT_LINES = 40
_MAX_IMPORTANT_LINE_CHARS = 800

_IMPORTANT_LINE_RE = re.compile(
    r"("
    r"\b(?:ERROR|Error|FAILED|FAIL|Failure|Exception|Traceback|panic|fatal|[A-Za-z_]*Error)\b"
    r"|File \".+?\", line \d+"
    r"|[\w./\\-]+:\d+(?::\d+)?"
    r"|https?://\S+"
    r"|exit code\s+\d+"
    r")"
)


@dataclass(frozen=True)
class ToolResultReductionConfig:
    """Runtime settings for large tool-result reduction."""

    enabled: bool = False
    min_chars: int = _DEFAULT_MIN_CHARS
    model: str | None = None
    preserve_raw_reference: bool = True
    max_reduced_chars: int = _DEFAULT_MAX_REDUCED_CHARS
    include_tool_args: bool = True
    include_recent_user_intent: bool = True
    excluded_tools: tuple[str, ...] = ()
    included_tools: tuple[str, ...] = ("*",)

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> "ToolResultReductionConfig":
        context = config.get("context", {}) if isinstance(config, Mapping) else {}
        raw = context.get("tool_result_reduction", {}) if isinstance(context, Mapping) else {}
        if not isinstance(raw, Mapping):
            raw = {}

        def _int(name: str, default: int, minimum: int = 1) -> int:
            try:
                value = int(raw.get(name, default))
            except (TypeError, ValueError):
                return default
            return value if value >= minimum else default

        def _str_list(name: str, default: Iterable[str]) -> tuple[str, ...]:
            value = raw.get(name, list(default))
            if not isinstance(value, list):
                return tuple(default)
            items = tuple(
                str(item).strip()
                for item in value
                if item is not None and str(item).strip()
            )
            return items

        def _bool(name: str, default: bool) -> bool:
            value = raw.get(name, default)
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "on"}
            return bool(value)

        model = raw.get("model")
        if isinstance(model, str):
            model = model.strip() or None
        else:
            model = None

        return cls(
            enabled=_bool("enabled", False),
            min_chars=_int("min_chars", _DEFAULT_MIN_CHARS),
            model=model,
            preserve_raw_reference=_bool("preserve_raw_reference", True),
            max_reduced_chars=_int(
                "max_reduced_chars",
                _DEFAULT_MAX_REDUCED_CHARS,
                minimum=500,
            ),
            include_tool_args=_bool("include_tool_args", True),
            include_recent_user_intent=_bool("include_recent_user_intent", True),
            excluded_tools=_str_list("excluded_tools", ()),
            included_tools=_str_list("included_tools", ("*",)),
        )

    def allows_tool(self, tool_name: str) -> bool:
        """Return True when the include/exclude lists allow this tool."""
        if any(fnmatch.fnmatchcase(tool_name, pattern) for pattern in self.excluded_tools):
            return False
        if not self.included_tools:
            return False
        return any(fnmatch.fnmatchcase(tool_name, pattern) for pattern in self.included_tools)


class ToolResultReducer:
    """Reduce large text tool results with an auxiliary model, fail-open."""

    def reduce(
        self,
        *,
        tool_name: str,
        tool_args: Mapping[str, Any] | None,
        result: Any,
        tool_call_id: str | None = None,
        task_id: str | None = None,
        recent_user_intent: str | None = None,
        env: Any = None,
    ) -> Any:
        """Return a reduced tool result when enabled and applicable.

        Non-string and small results pass through unchanged. Any reducer error
        also returns the original result so tool execution is never blocked.
        """
        if not isinstance(result, str):
            return result
        try:
            config = self._load_config()
            if not self._should_reduce(tool_name, result, config):
                return result

            raw_reference = self._preserve_raw_reference(
                config=config,
                tool_name=tool_name,
                result=result,
                tool_call_id=tool_call_id,
                task_id=task_id,
                env=env,
            )

            return self._reduce_with_model(
                config=config,
                tool_name=tool_name,
                tool_args=tool_args or {},
                result=result,
                raw_reference=raw_reference,
                recent_user_intent=recent_user_intent,
            )
        except Exception as exc:
            logger.debug("tool result reducer failed for %s: %s", tool_name, exc, exc_info=True)
            return result

    def _load_config(self) -> ToolResultReductionConfig:
        try:
            from hermes_cli.config import load_config

            return ToolResultReductionConfig.from_config(load_config())
        except Exception as exc:
            logger.debug("tool result reducer config load failed: %s", exc)
            return ToolResultReductionConfig()

    def _should_reduce(
        self,
        tool_name: str,
        result: str,
        config: ToolResultReductionConfig,
    ) -> bool:
        if not config.enabled:
            return False
        if _REDUCER_ACTIVE.get():
            return False
        if len(result) < config.min_chars:
            return False
        if not config.allows_tool(tool_name):
            return False
        if _looks_binary(result):
            return False
        return True

    def _preserve_raw_reference(
        self,
        *,
        config: ToolResultReductionConfig,
        tool_name: str,
        result: str,
        tool_call_id: str | None,
        task_id: str | None,
        env: Any,
    ) -> str | None:
        if not config.preserve_raw_reference:
            return None
        if env is None and task_id:
            try:
                from tools.terminal_tool import get_active_env

                env = get_active_env(task_id)
            except Exception:
                env = None
        if env is None:
            # TODO(tool-result-reducer): add a session-artifact store for
            # non-environment tools so raw references are available outside
            # terminal-backed sandbox storage.
            return None
        try:
            from tools.tool_result_storage import persist_tool_result_reference

            return persist_tool_result_reference(
                content=result,
                tool_name=tool_name,
                tool_use_id=tool_call_id or "tool-result",
                env=env,
            )
        except Exception as exc:
            logger.debug("raw tool-result reference persistence failed: %s", exc)
            return None

    def _reduce_with_model(
        self,
        *,
        config: ToolResultReductionConfig,
        tool_name: str,
        tool_args: Mapping[str, Any],
        result: str,
        raw_reference: str | None,
        recent_user_intent: str | None,
    ) -> str:
        prompt_result = _prompt_text_from_result(result)
        redacted_result = _redact_for_prompt(prompt_result)
        prompt = _build_reduction_prompt(
            tool_name=tool_name,
            tool_args=tool_args if config.include_tool_args else None,
            result=redacted_result,
            status=_extract_status(result),
            recent_user_intent=(
                recent_user_intent if config.include_recent_user_intent else None
            ),
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You reduce large tool outputs for an autonomous coding agent. "
                    "Preserve exact task-critical facts and do not invent details."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        token = _REDUCER_ACTIVE.set(True)
        try:
            response = call_llm(
                task="tool_result_reduction",
                model=config.model,
                messages=messages,
                temperature=0,
                max_tokens=_max_tokens_for_chars(config.max_reduced_chars),
            )
        finally:
            _REDUCER_ACTIVE.reset(token)

        reduced = _extract_response_text(response).strip()
        if not reduced:
            raise RuntimeError("empty reducer response")

        important_lines = _extract_important_lines(redacted_result)
        reduced = _append_missing_important_lines(reduced, important_lines)

        if _looks_like_json(result):
            reduced = (
                "Summarized text result; original tool result was JSON.\n"
                f"{reduced}"
            )

        query_line = _query_or_command(tool_args)
        header_lines = [
            f"[Tool result reduced from {len(result):,} chars]",
            f"Tool: {tool_name}",
        ]
        if query_line:
            header_lines.append(f"Query/command: {_truncate_middle(query_line, 500)}")
        if raw_reference:
            header_lines.append(f"Raw output saved to: {raw_reference}")
        elif config.preserve_raw_reference:
            header_lines.append("Raw output reference: unavailable for this tool/backend")

        final = "\n".join(header_lines) + "\n\n" + reduced
        final = _cap_text(final, config.max_reduced_chars)
        first_line, _, rest = final.partition("\n")
        final = (
            f"{first_line[:-1]} to {len(final):,} chars]"
            if first_line.endswith("]")
            else first_line
        ) + ("\n" + rest if rest else "")

        if len(final) >= len(result):
            return result

        logger.info(
            "Reduced tool result for %s from %d to %d chars",
            tool_name,
            len(result),
            len(final),
        )
        return final


def recent_user_intent_from_messages(messages: list[dict[str, Any]] | None) -> str | None:
    """Extract only the most recent user content from a message list."""
    if not messages:
        return None
    for msg in reversed(messages):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = _content_to_text(msg.get("content"))
        if content.strip():
            return _truncate_middle(content.strip(), _MAX_USER_INTENT_CHARS)
    return None


def _build_reduction_prompt(
    *,
    tool_name: str,
    tool_args: Mapping[str, Any] | None,
    result: str,
    status: str,
    recent_user_intent: str | None,
) -> str:
    raw_for_prompt = _truncate_middle(result, _MAX_RAW_PROMPT_CHARS)
    raw_note = ""
    if raw_for_prompt != result:
        raw_note = (
            "\nNote: the raw output below was clipped head/tail for reducer "
            "prompt size; preserve any explicit omitted-section note."
        )

    parts = [
        "Reduce this large tool result before it is inserted into active model context.",
        "",
        f"Tool name: {tool_name}",
    ]
    if tool_args is not None:
        parts.append(f"Tool arguments: {_format_args(tool_args)}")
    if status:
        parts.append(f"Exit/status: {status}")
    if recent_user_intent:
        parts.append(
            f"Recent user intent: {_truncate_middle(recent_user_intent, _MAX_USER_INTENT_CHARS)}"
        )

    important_lines = _extract_important_lines(result)
    if important_lines:
        parts.extend(
            [
                "",
                "Important raw lines to preserve exactly when relevant:",
                "\n".join(f"- {line}" for line in important_lines),
            ]
        )

    parts.extend(
        [
            "",
            "Write a concise reduced tool result. Include:",
            "- exact relevant findings",
            "- exact errors, failing test names, file paths, line numbers, commands, URLs, IDs, and code snippets that matter",
            "- negative findings such as failed commands or failed searches",
            "- a short Omitted section naming irrelevant or repetitive sections left out",
            "",
            "If the original result is JSON and you summarize it as prose, clearly say it is summarized text. Do not emit malformed JSON while implying it is valid JSON.",
            raw_note,
            "",
            "Raw tool output:",
            raw_for_prompt,
        ]
    )
    return "\n".join(parts)


def _format_args(args: Mapping[str, Any]) -> str:
    try:
        text = json.dumps(args, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        text = str(args)
    return _truncate_middle(text, _MAX_ARGS_CHARS)


def _extract_status(result: str) -> str:
    try:
        data = json.loads(result)
    except Exception:
        return ""
    if not isinstance(data, dict):
        return ""
    fields: list[str] = []
    for key in ("exit_code", "returncode", "status", "error", "exit_code_meaning"):
        value = data.get(key)
        if value is not None and value != "":
            fields.append(f"{key}={value!r}")
    return ", ".join(fields)


def _prompt_text_from_result(result: str) -> str:
    """Return the most useful text payload for the reducer prompt."""
    try:
        data = json.loads(result)
    except Exception:
        return result
    if not isinstance(data, dict):
        return result
    output = data.get("output")
    if isinstance(output, str) and output:
        return output
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)


def _extract_important_lines(text: str) -> list[str]:
    lines: list[str] = []
    seen: set[str] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line in seen:
            continue
        if not _IMPORTANT_LINE_RE.search(line):
            continue
        line = _truncate_middle(line, _MAX_IMPORTANT_LINE_CHARS)
        lines.append(line)
        seen.add(line)
        if len(lines) >= _MAX_IMPORTANT_LINES:
            break
    return lines


def _append_missing_important_lines(reduced: str, important_lines: list[str]) -> str:
    missing = [line for line in important_lines if line not in reduced]
    if not missing:
        return reduced
    return (
        f"{reduced.rstrip()}\n\n"
        "Important raw lines preserved:\n"
        + "\n".join(f"- {line}" for line in missing)
    )


def _query_or_command(args: Mapping[str, Any]) -> str:
    for key in ("command", "query", "url", "urls", "path", "file_path"):
        if key in args and args[key] not in (None, ""):
            value = args[key]
            if isinstance(value, (list, tuple)):
                return ", ".join(str(item) for item in value)
            return str(value)
    return ""


def _redact_for_prompt(text: str) -> str:
    try:
        from agent.redact import redact_sensitive_text

        return redact_sensitive_text(text)
    except Exception:
        return text


def _extract_response_text(response: Any) -> str:
    try:
        message = response.choices[0].message
    except Exception:
        return ""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, Mapping):
                text = part.get("text") or part.get("content")
                if text:
                    parts.append(str(text))
            else:
                parts.append(str(part))
        return "\n".join(parts)
    return str(content or "")


def _max_tokens_for_chars(max_chars: int) -> int:
    return max(256, min(8192, int(max_chars / 3) + 256))


def _looks_like_json(text: str) -> bool:
    stripped = text.lstrip()
    return stripped.startswith("{") or stripped.startswith("[")


def _looks_binary(text: str) -> bool:
    sample = text[:4096]
    if "\x00" in sample:
        return True
    if not sample:
        return False
    bad = sum(
        1
        for ch in sample
        if ord(ch) < 32 and ch not in "\n\r\t\b\f"
    )
    return (bad / max(1, len(sample))) > 0.05


def _truncate_middle(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars <= 20:
        return text[:max_chars]
    head = max_chars // 2
    tail = max_chars - head - 32
    omitted = len(text) - head - tail
    return f"{text[:head]}\n...[{omitted:,} chars omitted]...\n{text[-tail:]}"


def _cap_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    notice = "\n\n[Reducer note: reduced output exceeded max_reduced_chars; middle omitted.]\n\n"
    budget = max_chars - len(notice)
    if budget <= 0:
        return text[:max_chars]
    head = budget // 2
    tail = budget - head
    return text[:head] + notice + text[-tail:]


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, Mapping):
                if item.get("type") in {"text", "input_text"}:
                    parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    if isinstance(content, SimpleNamespace):
        return str(getattr(content, "content", "") or "")
    return str(content or "")
