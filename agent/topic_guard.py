"""Optional session topic-drift guard.

The guard is intentionally small: cheap local prefilters first, then one
structured auxiliary LLM call only when a session has enough prior turns to
make drift detection useful. All failures are fail-open.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

SUGGEST_ADVISORY = (
    "This looks like a new topic. Consider starting a new session with /new "
    "so the current session stays focused."
)
FORCE_FALLBACK_ADVISORY = (
    "This looks like a new topic. Topic guard force mode is enabled, but this "
    "entry point cannot safely start a new session automatically. Consider "
    "starting a new session with /new so the current session stays focused."
)
FORCE_STARTED_ADVISORY = (
    "This looks like a new topic, so a new session was started for it."
)

_CONTINUATION_PREFIXES = (
    "continue",
    "keep going",
    "same issue",
    "same problem",
    "try again",
    "that failed",
    "this failed",
    "still failing",
    "do that",
    "go ahead",
)

_JSON_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class TopicGuardConfig:
    enabled: bool = False
    mode: str = "suggest"
    model: Optional[str] = None
    min_turns: int = 4
    confidence_threshold: float = 0.75
    cooldown_turns: int = 3
    max_summary_chars: int = 2000
    ignore_slash_commands: bool = True
    ignore_short_messages_chars: int = 40

    @classmethod
    def from_config(cls, config: Optional[Dict[str, Any]]) -> "TopicGuardConfig":
        section: Dict[str, Any] = {}
        if isinstance(config, dict):
            session_cfg = config.get("session", {})
            if isinstance(session_cfg, dict) and isinstance(session_cfg.get("topic_guard"), dict):
                section = session_cfg["topic_guard"]
            elif isinstance(config.get("topic_guard"), dict):
                section = config["topic_guard"]
        return cls.from_mapping(section)

    @classmethod
    def from_mapping(cls, raw: Optional[Dict[str, Any]]) -> "TopicGuardConfig":
        raw = raw if isinstance(raw, dict) else {}
        mode = str(raw.get("mode", "suggest") or "suggest").strip().lower()
        if mode not in {"off", "suggest", "force"}:
            logger.warning("Invalid session.topic_guard.mode=%r; using suggest", mode)
            mode = "suggest"
        enabled = _truthy(raw.get("enabled", False)) and mode != "off"
        model = raw.get("model")
        if isinstance(model, str):
            model = model.strip() or None
        elif model is not None:
            model = str(model).strip() or None
        return cls(
            enabled=enabled,
            mode=mode,
            model=model,
            min_turns=_bounded_int(raw.get("min_turns"), 4, minimum=1),
            confidence_threshold=_bounded_float(
                raw.get("confidence_threshold"), 0.75, minimum=0.0, maximum=1.0
            ),
            cooldown_turns=_bounded_int(raw.get("cooldown_turns"), 3, minimum=0),
            max_summary_chars=_bounded_int(raw.get("max_summary_chars"), 2000, minimum=400),
            ignore_slash_commands=_truthy(raw.get("ignore_slash_commands", True)),
            ignore_short_messages_chars=_bounded_int(
                raw.get("ignore_short_messages_chars"), 40, minimum=0
            ),
        )


@dataclass(frozen=True)
class TopicDriftClassification:
    drift: bool
    confidence: float
    old_topic: str = ""
    new_topic: str = ""
    reason: str = ""


@dataclass(frozen=True)
class TopicGuardDecision:
    fired: bool = False
    action: str = "none"
    classification: Optional[TopicDriftClassification] = None
    advisory: str = ""
    started_new_session: bool = False
    force_fallback: bool = False
    skipped_reason: str = ""


Classifier = Callable[[List[Dict[str, str]], Optional[str], Optional[Dict[str, str]]], str]
ForceNewSession = Callable[[TopicDriftClassification], bool]
MainRuntimeProvider = Callable[[], Dict[str, str]]


class TopicGuard:
    def __init__(
        self,
        config: TopicGuardConfig,
        *,
        classifier: Optional[Classifier] = None,
        force_new_session: Optional[ForceNewSession] = None,
        main_runtime_provider: Optional[MainRuntimeProvider] = None,
    ) -> None:
        self.config = config
        self._classifier = classifier
        self._force_new_session = force_new_session
        self._main_runtime_provider = main_runtime_provider
        self._last_trigger_user_turn: Optional[int] = None

    def evaluate(
        self,
        *,
        user_message: Any,
        conversation_history: List[Dict[str, Any]],
        session_title: str = "",
    ) -> TopicGuardDecision:
        reason = self._prefilter_reason(user_message, conversation_history)
        if reason:
            return TopicGuardDecision(skipped_reason=reason)

        prompt_messages = self._build_classifier_messages(
            user_message=str(user_message),
            conversation_history=conversation_history,
            session_title=session_title,
        )
        raw = self._classify(prompt_messages)
        classification = parse_topic_guard_response(raw)
        if classification is None:
            logger.info("topic guard classifier returned unparsable response")
            return TopicGuardDecision(skipped_reason="parse_failed")

        if (
            not classification.drift
            or classification.confidence < self.config.confidence_threshold
        ):
            return TopicGuardDecision(
                classification=classification,
                skipped_reason="no_drift",
            )

        self._last_trigger_user_turn = _prior_user_turn_count(conversation_history)
        if self.config.mode == "force":
            if self._force_new_session is None:
                logger.info("topic guard force mode unsupported; falling back to suggest")
                return TopicGuardDecision(
                    fired=True,
                    action="suggest",
                    classification=classification,
                    advisory=FORCE_FALLBACK_ADVISORY,
                    force_fallback=True,
                )
            try:
                if self._force_new_session(classification):
                    return TopicGuardDecision(
                        fired=True,
                        action="force",
                        classification=classification,
                        advisory=FORCE_STARTED_ADVISORY,
                        started_new_session=True,
                    )
            except Exception as exc:
                logger.warning("topic guard force callback failed open: %s", exc, exc_info=True)
            return TopicGuardDecision(
                fired=True,
                action="suggest",
                classification=classification,
                advisory=FORCE_FALLBACK_ADVISORY,
                force_fallback=True,
            )

        return TopicGuardDecision(
            fired=True,
            action="suggest",
            classification=classification,
            advisory=SUGGEST_ADVISORY,
        )

    def _prefilter_reason(
        self,
        user_message: Any,
        conversation_history: List[Dict[str, Any]],
    ) -> str:
        if not self.config.enabled:
            return "disabled"
        if not isinstance(user_message, str):
            return "non_text"
        stripped = user_message.strip()
        if not stripped:
            return "empty"
        if self.config.ignore_slash_commands and stripped.startswith("/"):
            return "slash_command"
        if len(stripped) < self.config.ignore_short_messages_chars:
            return "short_message"
        prior_turns = _prior_user_turn_count(conversation_history)
        if prior_turns < self.config.min_turns:
            return "below_min_turns"
        if (
            self._last_trigger_user_turn is not None
            and prior_turns - self._last_trigger_user_turn < self.config.cooldown_turns
        ):
            return "cooldown"
        lowered = stripped.lower()
        if any(lowered.startswith(prefix) for prefix in _CONTINUATION_PREFIXES):
            return "continuation_cue"
        return ""

    def _build_classifier_messages(
        self,
        *,
        user_message: str,
        conversation_history: List[Dict[str, Any]],
        session_title: str,
    ) -> List[Dict[str, str]]:
        snapshot = build_topic_snapshot(
            conversation_history,
            session_title=session_title,
            max_chars=self.config.max_summary_chars,
        )
        system_prompt = (
            "You are a conservative topic-drift classifier for an AI assistant. "
            "Decide whether the new user message clearly switches away from the "
            "current session topic. Do not flag normal follow-ups, corrections, "
            "short asides, or implementation details. Return only JSON with keys: "
            "drift, confidence, old_topic, new_topic, reason."
        )
        user_prompt = (
            "Current session topic snapshot:\n"
            f"{snapshot or '(no compact topic snapshot available)'}\n\n"
            "New user message:\n"
            f"{user_message.strip()}\n\n"
            "Return JSON like:\n"
            '{"drift": true, "confidence": 0.87, "old_topic": "...", '
            '"new_topic": "...", "reason": "..."}'
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _classify(self, messages: List[Dict[str, str]]) -> str:
        try:
            runtime = self._main_runtime_provider() if self._main_runtime_provider else None
        except Exception:
            runtime = None
        try:
            if self._classifier is not None:
                return self._classifier(messages, self.config.model, runtime)
            from agent.auxiliary_client import call_llm

            response = call_llm(
                "topic_guard",
                model=self.config.model,
                main_runtime=runtime,
                messages=messages,
                temperature=0,
                max_tokens=300,
            )
            return _response_text(response)
        except Exception as exc:
            logger.info("topic guard classifier failed open: %s", exc)
            return ""


def parse_topic_guard_response(raw: Any) -> Optional[TopicDriftClassification]:
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = _JSON_FENCE_RE.sub("", raw.strip())
    first = text.find("{")
    last = text.rfind("}")
    if first == -1 or last == -1 or last <= first:
        return None
    try:
        data = json.loads(text[first : last + 1])
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(confidence, 1.0))
    return TopicDriftClassification(
        drift=_truthy(data.get("drift", False)),
        confidence=confidence,
        old_topic=_clean_short_text(data.get("old_topic", "")),
        new_topic=_clean_short_text(data.get("new_topic", "")),
        reason=_clean_short_text(data.get("reason", "")),
    )


def build_topic_snapshot(
    conversation_history: List[Dict[str, Any]],
    *,
    session_title: str = "",
    max_chars: int = 2000,
) -> str:
    max_chars = max(400, int(max_chars or 2000))
    lines: List[str] = []
    title = str(session_title or "").strip()
    if title:
        lines.append(f"Session title: {_truncate(title, 200)}")

    latest_summary = _latest_compaction_summary(conversation_history)
    if latest_summary:
        lines.append(f"Latest context summary: {_truncate(latest_summary, min(800, max_chars // 2))}")

    recent: List[str] = []
    for msg in reversed(conversation_history or []):
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "").strip().lower()
        if role not in {"user", "assistant"}:
            continue
        content = _content_to_text(msg.get("content"))
        if not content:
            continue
        recent.append(f"{role}: {_truncate(content, 320)}")
        if len(recent) >= 8:
            break
    if recent:
        lines.append("Recent turns:\n" + "\n".join(reversed(recent)))

    snapshot = "\n\n".join(lines).strip()
    return _truncate(snapshot, max_chars)


def _latest_compaction_summary(conversation_history: List[Dict[str, Any]]) -> str:
    markers = (
        "[CONTEXT COMPACTION",
        "[CONTEXT SUMMARY]",
        "handoff from a previous context",
    )
    for msg in reversed(conversation_history or []):
        if not isinstance(msg, dict):
            continue
        content = _content_to_text(msg.get("content"))
        if content and any(marker in content for marker in markers):
            return content
    return ""


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return re.sub(r"\s+", " ", content).strip()
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item.get("text_summary"), str):
                    parts.append(item["text_summary"])
        return re.sub(r"\s+", " ", " ".join(parts)).strip()
    if content is None:
        return ""
    return re.sub(r"\s+", " ", str(content)).strip()


def _prior_user_turn_count(conversation_history: List[Dict[str, Any]]) -> int:
    return sum(
        1
        for msg in (conversation_history or [])
        if isinstance(msg, dict) and msg.get("role") == "user"
    )


def _response_text(response: Any) -> str:
    try:
        return response.choices[0].message.content or ""
    except Exception:
        return ""


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on", "enabled"}


def _bounded_int(value: Any, default: int, *, minimum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, parsed)


def _bounded_float(value: Any, default: float, *, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _truncate(text: str, limit: int) -> str:
    text = str(text or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 14)].rstrip() + "...[truncated]"


def _clean_short_text(value: Any) -> str:
    return _truncate(re.sub(r"\s+", " ", str(value or "")).strip(), 300)
