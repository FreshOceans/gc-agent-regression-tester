"""Journey-mode helpers: category resolution and transcript URL journey parsing."""

from __future__ import annotations

import re
from typing import Any, Callable, Optional

from .journey_mode import (
    CATEGORY_STRATEGY_LLM_FIRST,
    CATEGORY_STRATEGY_RULES_FIRST,
    DEFAULT_PRIMARY_CATEGORIES,
    normalize_category_strategy,
)

_CUSTOMER_ROLES = {
    "customer",
    "user",
    "guest",
    "traveler",
    "external",
    "externaluser",
    "consumer",
    "client",
    "webuser",
}
_AGENT_ROLES = {
    "agent",
    "assistant",
    "bot",
    "ivr",
    "acd",
    "workflow",
    "flow",
    "architect",
    "system",
    "server",
}
_NOISE_PREFIXES = (
    "conversation_id:",
    "participant_id:",
    "detected_intent:",
    "conversationid:",
    "participantid:",
)
_NOISE_EXACT = {
    "conversation ended by user",
    "conversation ended",
    "typing",
    "typing...",
    "presence events are not supported in this configuration",
}


def normalize_category_name(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip().lower().replace(" ", "_")
    return normalized or None


def resolve_primary_categories(
    *,
    suite_categories: Optional[list[Any]] = None,
    config_overrides: Optional[list[dict[str, Any]]] = None,
) -> list[dict[str, Any]]:
    """Resolve primary categories from defaults + config + suite overrides."""
    merged = {
        normalize_category_name(item.get("name")): {
            "name": normalize_category_name(item.get("name")),
            "keywords": _normalize_keywords(item.get("keywords")),
            "rubric": str(item.get("rubric") or "").strip() or None,
        }
        for item in DEFAULT_PRIMARY_CATEGORIES
    }

    for override in config_overrides or []:
        _upsert_category(merged, override)

    for raw in suite_categories or []:
        if isinstance(raw, dict):
            _upsert_category(merged, raw)
            continue
        # Support pydantic objects with model_dump.
        model_dump = getattr(raw, "model_dump", None)
        if callable(model_dump):
            payload = model_dump(exclude_none=True)
            if isinstance(payload, dict):
                _upsert_category(merged, payload)

    ordered = [item for item in merged.values() if item.get("name")]
    ordered.sort(key=lambda row: row["name"])
    return ordered


def categorize_message_by_rules(
    message: str,
    *,
    categories: list[dict[str, Any]],
) -> Optional[str]:
    normalized_message = _normalize_message(message)
    if not normalized_message:
        return None

    best_name: Optional[str] = None
    best_score = 0
    for category in categories:
        name = normalize_category_name(category.get("name"))
        keywords = _normalize_keywords(category.get("keywords"))
        if not name or not keywords:
            continue
        score = 0
        for keyword in keywords:
            token = keyword.strip().lower()
            if not token:
                continue
            if token in normalized_message:
                score += max(1, len(token.split()))
        if score > best_score:
            best_score = score
            best_name = name
    return best_name


def resolve_category_with_strategy(
    message: str,
    *,
    categories: list[dict[str, Any]],
    strategy: str,
    llm_classifier: Optional[Callable[[str, list[dict[str, Any]]], dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Resolve category with rules-first or llm-first strategy."""
    normalized_strategy = normalize_category_strategy(strategy)
    rules_category = categorize_message_by_rules(message, categories=categories)

    def _llm_result() -> dict[str, Any]:
        if llm_classifier is None:
            return {"category": None, "source": "llm_unavailable", "confidence": None}
        try:
            llm = llm_classifier(message, categories)
        except Exception:
            return {"category": None, "source": "llm_error", "confidence": None}
        category = normalize_category_name(llm.get("category"))
        names = {normalize_category_name(item.get("name")) for item in categories}
        if category not in names:
            category = None
        return {
            "category": category,
            "source": "llm",
            "confidence": llm.get("confidence"),
            "explanation": llm.get("explanation"),
        }

    if normalized_strategy == CATEGORY_STRATEGY_RULES_FIRST:
        if rules_category:
            return {"category": rules_category, "source": "rules", "confidence": 1.0}
        llm = _llm_result()
        if llm.get("category"):
            return llm
        return {"category": None, "source": "unclassified", "confidence": None}

    llm = _llm_result()
    if llm.get("category"):
        return llm
    if rules_category:
        return {"category": rules_category, "source": "rules_fallback", "confidence": 1.0}
    return {"category": None, "source": "unclassified", "confidence": None}


def extract_journey_seed_candidates(payload: Any) -> list[dict[str, Any]]:
    """Extract one journey candidate per conversation from transcript payload."""
    candidates: list[dict[str, Any]] = []
    for convo in _iter_conversation_payloads(payload):
        conversation_id = (
            str(
                convo.get("conversation_id")
                or convo.get("conversationId")
                or convo.get("id")
                or ""
            ).strip().lower()
            or None
        )
        messages = _extract_messages(convo)
        if not messages:
            continue
        first_customer = _first_meaningful_customer_message(messages)
        if not first_customer:
            continue
        contained = infer_containment_from_payload_metadata(convo)
        candidates.append(
            {
                "conversation_id": conversation_id,
                "first_customer_message": first_customer,
                "messages": messages,
                "metadata_contained": contained,
            }
        )
    return candidates


def infer_containment_from_payload_metadata(payload: Any) -> Optional[bool]:
    """Return containment from transcript metadata, or None if unknown."""
    participants = None
    if isinstance(payload, dict):
        participants = payload.get("participants")
    if not isinstance(participants, list) or not participants:
        return None

    saw_known_participant = False
    for participant in participants:
        if not isinstance(participant, dict):
            continue
        purpose = str(
            participant.get("purpose")
            or participant.get("participantPurpose")
            or participant.get("role")
            or ""
        ).strip().lower()
        if purpose:
            saw_known_participant = True
        if purpose == "agent" and str(participant.get("userId") or "").strip():
            return False
    if saw_known_participant:
        return True
    return None


def _iter_conversation_payloads(payload: Any):
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item
        return

    if not isinstance(payload, dict):
        return

    conversations = payload.get("conversations")
    if isinstance(conversations, list) and conversations:
        for convo in conversations:
            if isinstance(convo, dict):
                merged = dict(convo)
                for inherited_key in ("participants", "conversationId", "conversation_id"):
                    if inherited_key not in merged and inherited_key in payload:
                        merged[inherited_key] = payload[inherited_key]
                yield merged
        return

    yield payload


def _extract_messages(convo: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []

    raw_messages = convo.get("messages")
    if isinstance(raw_messages, list):
        for message in raw_messages:
            if not isinstance(message, dict):
                continue
            text = _extract_text(message)
            if not text:
                continue
            role = _normalize_role(
                message.get("speaker")
                or message.get("role")
                or message.get("participantPurpose")
                or message.get("purpose")
                or message.get("direction")
            )
            messages.append(
                {
                    "role": role,
                    "text": text,
                    "timestamp": (
                        message.get("timestamp")
                        or message.get("time")
                        or message.get("eventTime")
                        or message.get("startTime")
                    ),
                }
            )

    transcripts = convo.get("transcripts")
    if isinstance(transcripts, list):
        for transcript in transcripts:
            if not isinstance(transcript, dict):
                continue
            phrases = transcript.get("phrases")
            if not isinstance(phrases, list):
                continue
            for phrase in phrases:
                if not isinstance(phrase, dict):
                    continue
                text = _extract_text(phrase)
                if not text:
                    continue
                role = _normalize_role(phrase.get("participantPurpose"))
                messages.append(
                    {
                        "role": role,
                        "text": text,
                        "timestamp": phrase.get("startTimeMs") or phrase.get("timestamp"),
                    }
                )

    return messages


def _extract_text(node: dict[str, Any]) -> Optional[str]:
    for key in ("text", "body", "message", "messageText", "content", "utterance"):
        value = node.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _normalize_role(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in _CUSTOMER_ROLES:
        return "customer"
    if normalized in _AGENT_ROLES:
        return "agent"
    return "system"


def _first_meaningful_customer_message(messages: list[dict[str, Any]]) -> Optional[str]:
    for msg in messages:
        if str(msg.get("role") or "").strip().lower() != "customer":
            continue
        text = str(msg.get("text") or "").strip()
        if not text:
            continue
        normalized = _normalize_message(text)
        if not normalized:
            continue
        if normalized in _NOISE_EXACT:
            continue
        if any(normalized.startswith(prefix) for prefix in _NOISE_PREFIXES):
            continue
        # Skip one-letter or non-alpha noise.
        if len(normalized) < 2 or not re.search(r"[a-zA-Z]", normalized):
            continue
        return text
    return None


def _normalize_message(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _upsert_category(target: dict[str, dict[str, Any]], payload: dict[str, Any]) -> None:
    name = normalize_category_name(payload.get("name"))
    if not name:
        return
    keywords = _normalize_keywords(payload.get("keywords"))
    rubric = str(payload.get("rubric") or "").strip() or None
    target[name] = {
        "name": name,
        "keywords": keywords,
        "rubric": rubric,
    }


def _normalize_keywords(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw = [item.strip() for item in value.split(",")]
    elif isinstance(value, (list, tuple, set)):
        raw = [str(item).strip() for item in value]
    else:
        return []

    deduped: list[str] = []
    seen: set[str] = set()
    for item in raw:
        normalized = item.strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped
