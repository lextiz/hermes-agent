"""Compact orchestration decision calls for delegation, validation, escalation."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from agent.auxiliary_client import call_llm
from hermes_cli.config import load_config

logger = logging.getLogger(__name__)

DECISION_KINDS = frozenset({"delegation", "validation", "escalation"})

_AUX_TASK_BY_KIND = {
    "delegation": "orchestration_delegate",
    "validation": "orchestration_validation",
    "escalation": "orchestration_escalation",
}

_FAIL_OPEN_DECISIONS = {
    "delegation": {
        "delegate": False,
        "tasks": [],
        "roles": [],
        "model_tier_recommendations": [],
        "reason": "orchestration disabled",
    },
    "validation": {
        "verdict": "pass",
        "missing_checks": [],
        "suggested_commands": [],
        "repair_needed": False,
        "reason": "orchestration disabled",
    },
    "escalation": {
        "escalate": False,
        "target_tier": None,
        "provider": None,
        "model": None,
        "profile": None,
        "retry_strategy": "continue_default",
        "compact_prompt": "",
        "reason": "orchestration disabled",
    },
}

_SCHEMA_HINT_BY_KIND = {
    "delegation": (
        '{"delegate": boolean, "tasks": [{"goal": string, "context": string, '
        '"toolsets": string[], "role": "leaf|orchestrator", "model": string|null, '
        '"provider": string|null}], "roles": string[], '
        '"model_tier_recommendations": string[], "reason": string}'
    ),
    "validation": (
        '{"verdict": "pass|fail|uncertain", "missing_checks": string[], '
        '"suggested_commands": string[], "repair_needed": boolean, "reason": string}'
    ),
    "escalation": (
        '{"escalate": boolean, "target_tier": string|null, "provider": string|null, '
        '"model": string|null, "profile": string|null, "retry_strategy": string, '
        '"compact_prompt": string, "reason": string}'
    ),
}

_QUESTION_BY_KIND = {
    "delegation": "Should this work be delegated into compact subagent tasks?",
    "validation": "Does this work satisfy the task and acceptance criteria?",
    "escalation": "Should this failed or uncertain slice be retried with a stronger configured model?",
}

_POLICY_BY_KIND = {
    "delegation": (
        "Policy:\n"
        "- Delegate only when independent subtasks materially reduce risk or latency.\n"
        "- Return delegate=false for small, single-file, or straightforward test-fix tasks.\n"
        "- For task_hints showing high difficulty or explicit verification risk, delegate a compact edge-case review, design, or repair slice when it can reduce verifier risk.\n"
        "- Automatic subagents run against the same workspace; never create tasks that can modify files disallowed by the user task.\n"
        "- Prefer diagnostic/read-review tasks unless the original task explicitly permits parallel implementation edits.\n"
        "- Each task must restate relevant file-change constraints and acceptance criteria."
    ),
    "validation": (
        "Policy:\n"
        "- Judge only the supplied task, acceptance criteria, diff/output, and logs.\n"
        "- Mark uncertain when required evidence is missing instead of assuming success.\n"
        "- When a task names edge cases, invariants, invalid inputs, mutation behavior, state transitions, or required errors, visible tests passing is not enough: check those stated requirements and mark fail or uncertain when the supplied evidence does not clearly satisfy them."
        "\n- Pay special attention to singular-vs-collection inputs, ordered deduplication, and lazy or short-circuit behavior when those semantics are part of the task."
    ),
    "escalation": (
        "Policy:\n"
        "- Escalate only the failed or uncertain slice, not the whole conversation.\n"
        "- Prefer a compact retry prompt with exact errors, attempts, files, verifier output, task_hints, acceptance_hints, and the smallest missing requirement.\n"
        "- If acceptance_hints identify missing edge cases, include the most specific missing requirement in compact_prompt."
    ),
}


def parse_decision_json(raw: Any) -> Optional[Dict[str, Any]]:
    """Extract the first JSON object from plain or fenced model output."""
    if isinstance(raw, dict):
        return raw
    if raw is None:
        return None

    text = str(raw).strip()
    if not text:
        return None

    direct = _try_json_object(text)
    if direct is not None:
        return direct

    for fenced in _iter_fenced_blocks(text):
        parsed = _try_json_object(fenced.strip())
        if parsed is not None:
            return parsed

    for candidate in _iter_json_object_candidates(text):
        parsed = _try_json_object(candidate)
        if parsed is not None:
            return parsed

    return None


def orchestration_enabled(kind: str) -> bool:
    """Return True only when the top-level orchestration config enables kind."""
    kind = _normalize_kind(kind)
    cfg = _load_orchestration_config()
    if not cfg.get("enabled", False):
        return False

    kind_cfg = cfg.get(kind, {})
    if isinstance(kind_cfg, dict):
        return bool(kind_cfg.get("enabled", False))
    return bool(kind_cfg)


def decide_delegation(packet: Dict[str, Any]) -> Dict[str, Any]:
    """Decide whether the current packet should be delegated."""
    return _decide("delegation", packet)


def judge_validation(packet: Dict[str, Any]) -> Dict[str, Any]:
    """Judge whether the completed packet validates."""
    return _decide("validation", packet)


def decide_escalation(packet: Dict[str, Any]) -> Dict[str, Any]:
    """Decide whether the current packet should be escalated."""
    return _decide("escalation", packet)


def _decide(kind: str, packet: Dict[str, Any]) -> Dict[str, Any]:
    kind = _normalize_kind(kind)
    cfg = _safe_load_config()
    orch_cfg = cfg.get("orchestration", {}) if isinstance(cfg, dict) else {}
    if not isinstance(orch_cfg, dict):
        orch_cfg = {}

    provider, model = _configured_aux_provider_model(cfg, kind)
    result = _base_result(kind, provider, model)
    result["auxiliary_task"] = _AUX_TASK_BY_KIND[kind]

    if not _kind_enabled_from_config(orch_cfg, kind):
        result["decision_status"] = "disabled"
        result["decision"] = _fail_open_decision(kind, "orchestration disabled")
        return result

    task = _AUX_TASK_BY_KIND[kind]
    messages = _build_messages(kind, packet)
    result["enabled"] = True
    result["called"] = True
    result["decision_status"] = "called"

    try:
        response = call_llm(
            task=task,
            messages=messages,
            temperature=0,
            max_tokens=int(_kind_config(orch_cfg, kind).get("max_tokens", 300)),
            timeout=_kind_config(orch_cfg, kind).get("timeout"),
        )
        raw = _response_content(response)
        result["raw"] = raw
        parsed = parse_decision_json(raw)
        if parsed is None:
            result["parse_failed"] = True
            result["decision_status"] = "parse_failed"
            result["decision"] = _fail_open_decision(kind, "decision JSON parse failed", enabled_failure=True)
        else:
            result["decision_status"] = "parsed"
            result["decision"] = _normalize_decision(kind, parsed)
    except Exception as exc:
        logger.debug("Orchestration %s decision failed: %s", kind, exc)
        result["error"] = str(exc)
        result["decision_status"] = "call_failed"
        result["decision"] = _fail_open_decision(kind, "decision call failed", enabled_failure=True)

    return result


def _base_result(kind: str, provider: Optional[str], model: Optional[str]) -> Dict[str, Any]:
    return {
        "enabled": False,
        "called": False,
        "kind": kind,
        "decision": None,
        "raw": None,
        "error": None,
        "provider": provider,
        "model": model,
        "parse_failed": False,
        "decision_status": "not_called",
        "auxiliary_task": None,
    }


def _build_messages(kind: str, packet: Dict[str, Any]) -> list[dict[str, str]]:
    packet_json = json.dumps(packet or {}, ensure_ascii=False, sort_keys=True, default=str)
    system = (
        "You are a compact Hermes orchestration judge. "
        "Return JSON only. No markdown, no prose."
    )
    user = (
        f"{_QUESTION_BY_KIND[kind]}\n"
        f"{_POLICY_BY_KIND.get(kind, '')}\n"
        f"Schema: {_SCHEMA_HINT_BY_KIND[kind]}\n"
        f"Packet JSON:\n{packet_json}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _response_content(response: Any) -> str:
    if isinstance(response, str):
        return response
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError):
        return str(response)
    return "" if content is None else str(content)


def _fail_open_decision(kind: str, reason: str, *, enabled_failure: bool = False) -> Dict[str, Any]:
    decision = dict(_FAIL_OPEN_DECISIONS[kind])
    if kind == "validation" and enabled_failure:
        decision.update(
            {
                "verdict": "uncertain",
                "missing_checks": ["validation judge unavailable or returned invalid JSON"],
                "suggested_commands": [],
                "repair_needed": True,
            }
        )
    decision["reason"] = reason
    return decision


def _normalize_decision(kind: str, parsed: Dict[str, Any]) -> Dict[str, Any]:
    if kind == "delegation":
        tasks = []
        raw_tasks = parsed.get("tasks")
        if isinstance(raw_tasks, list):
            for item in raw_tasks:
                if not isinstance(item, dict):
                    continue
                goal = _optional_str(item.get("goal"))
                if not goal:
                    continue
                normalized = {"goal": goal}
                context = _optional_str(item.get("context"))
                if context:
                    normalized["context"] = context
                toolsets = _string_list(item.get("toolsets"))
                if toolsets:
                    normalized["toolsets"] = toolsets
                role = _optional_str(item.get("role"))
                normalized["role"] = role if role in {"leaf", "orchestrator"} else "leaf"
                for key in ("model", "provider"):
                    value = _optional_str(item.get(key))
                    if value:
                        normalized[key] = value
                tasks.append(normalized)
        delegate = parsed.get("delegate") is True and bool(tasks)
        return {
            "delegate": delegate,
            "tasks": tasks if delegate else [],
            "roles": _string_list(parsed.get("roles")),
            "model_tier_recommendations": _string_list(parsed.get("model_tier_recommendations")),
            "reason": _optional_str(parsed.get("reason")) or "",
        }

    if kind == "validation":
        verdict = _optional_str(parsed.get("verdict"))
        if verdict not in {"pass", "fail", "uncertain"}:
            if parsed.get("valid") is True:
                verdict = "pass"
            elif parsed.get("valid") is False:
                verdict = "fail"
            else:
                verdict = "uncertain"
        repair_needed = parsed.get("repair_needed")
        if not isinstance(repair_needed, bool):
            repair_needed = verdict != "pass"
        return {
            "verdict": verdict,
            "missing_checks": _string_list(parsed.get("missing_checks")),
            "suggested_commands": _string_list(parsed.get("suggested_commands")),
            "repair_needed": repair_needed,
            "reason": _optional_str(parsed.get("reason")) or "",
        }

    return {
        "escalate": parsed.get("escalate") is True,
        "target_tier": _optional_str(parsed.get("target_tier")),
        "provider": _optional_str(parsed.get("provider")),
        "model": _optional_str(parsed.get("model")),
        "profile": _optional_str(parsed.get("profile")),
        "retry_strategy": _optional_str(parsed.get("retry_strategy")) or "continue_default",
        "compact_prompt": _optional_str(parsed.get("compact_prompt")) or "",
        "reason": _optional_str(parsed.get("reason")) or "",
    }


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    value = value.strip()
    return value or None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    items = []
    for item in value:
        text = _optional_str(item)
        if text:
            items.append(text)
    return items


def _safe_load_config() -> Dict[str, Any]:
    try:
        cfg = load_config()
    except Exception as exc:
        logger.debug("Could not load orchestration config: %s", exc)
        return {}
    return cfg if isinstance(cfg, dict) else {}


def _load_orchestration_config() -> Dict[str, Any]:
    cfg = _safe_load_config()
    orch_cfg = cfg.get("orchestration", {})
    return orch_cfg if isinstance(orch_cfg, dict) else {}


def _kind_enabled_from_config(orchestration_cfg: Dict[str, Any], kind: str) -> bool:
    if not orchestration_cfg.get("enabled", False):
        return False
    kind_cfg = orchestration_cfg.get(kind, {})
    if isinstance(kind_cfg, dict):
        return bool(kind_cfg.get("enabled", False))
    return bool(kind_cfg)


def _kind_config(orchestration_cfg: Dict[str, Any], kind: str) -> Dict[str, Any]:
    kind_cfg = orchestration_cfg.get(kind, {})
    return kind_cfg if isinstance(kind_cfg, dict) else {}


def _configured_aux_provider_model(
    cfg: Dict[str, Any],
    kind: str,
) -> tuple[Optional[str], Optional[str]]:
    auxiliary = cfg.get("auxiliary", {}) if isinstance(cfg, dict) else {}
    if not isinstance(auxiliary, dict):
        return None, None
    task_cfg = auxiliary.get(_AUX_TASK_BY_KIND[kind], {})
    if not isinstance(task_cfg, dict):
        return None, None
    return task_cfg.get("provider"), task_cfg.get("model")


def _normalize_kind(kind: str) -> str:
    normalized = (kind or "").strip().lower()
    if normalized not in DECISION_KINDS:
        raise ValueError(f"Unknown orchestration decision kind: {kind!r}")
    return normalized


def _try_json_object(text: str) -> Optional[Dict[str, Any]]:
    try:
        parsed = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _iter_fenced_blocks(text: str):
    marker = "```"
    start = 0
    while True:
        open_idx = text.find(marker, start)
        if open_idx < 0:
            return
        content_start = text.find("\n", open_idx + len(marker))
        if content_start < 0:
            return
        close_idx = text.find(marker, content_start + 1)
        if close_idx < 0:
            return
        yield text[content_start + 1:close_idx]
        start = close_idx + len(marker)


def _iter_json_object_candidates(text: str):
    in_string = False
    escape = False
    depth = 0
    start = None

    for idx, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue
        if char == "{":
            if depth == 0:
                start = idx
            depth += 1
            continue
        if char == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                yield text[start:idx + 1]
                start = None
