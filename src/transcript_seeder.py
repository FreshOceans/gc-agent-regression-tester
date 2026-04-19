"""Transcript-to-suite seeding helpers for Phase 4 MVP.

This module converts uploaded transcript content into a draft TestSuite.
The MVP intentionally focuses on deterministic parsing and clear defaults:
- Extract customer utterances from JSON, YAML, or plain text transcript content.
- Build one-attempt scenarios from those utterances.
- Apply known behavior overrides (for example pricing-guidance/guideline flows).
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

import yaml

from .models import TestScenario, TestSuite


class TranscriptSeedError(ValueError):
    """Raised when transcript content cannot be seeded into scenarios."""


_CUSTOMER_SPEAKER_TOKENS = {
    "customer",
    "user",
    "traveler",
    "guest",
    "consumer",
    "client",
    "enduser",
    "end-user",
    "inbound",
    "visitor",
}
_AGENT_SPEAKER_TOKENS = {
    "agent",
    "assistant",
    "bot",
    "system",
    "outbound",
    "server",
}

_PRICING_KEYWORDS = {
    "price",
    "pricing",
    "cost",
    "costs",
    "fee",
    "fees",
    "charge",
    "charges",
    "how much",
}
_BAGGAGE_KEYWORDS = {
    "bag",
    "bags",
    "baggage",
    "checked bag",
    "checked bags",
    "carry on",
    "carry-on",
}


def seed_test_suite_from_transcript(
    content: str,
    *,
    format_hint: Optional[str] = None,
    suite_name: Optional[str] = None,
    max_scenarios: int = 50,
) -> TestSuite:
    """Generate a draft TestSuite from transcript content.

    Args:
        content: Raw transcript text.
        format_hint: Optional input hint ("json", "yaml", or "text").
        suite_name: Optional output suite name override.
        max_scenarios: Maximum number of generated scenarios.

    Returns:
        A validated TestSuite containing draft scenarios.
    """
    if max_scenarios < 1:
        raise TranscriptSeedError("max_scenarios must be at least 1")

    utterances = _extract_customer_utterances(content, format_hint=format_hint)
    if not utterances:
        raise TranscriptSeedError(
            "No customer/user utterances were found in the uploaded transcript."
        )

    limited = utterances[:max_scenarios]
    scenarios = _build_seed_scenarios(limited)
    if not scenarios:
        raise TranscriptSeedError(
            "Transcript parsing succeeded, but no scenarios could be generated."
        )

    seeded_suite_name = (
        (suite_name or "").strip() or "Seeded Transcript Regression Suite"
    )
    return TestSuite(name=seeded_suite_name, scenarios=scenarios)


def _extract_customer_utterances(
    content: str, *, format_hint: Optional[str] = None
) -> list[dict[str, Optional[str]]]:
    parsed = _parse_content(content, format_hint=format_hint)

    if isinstance(parsed, (dict, list)):
        return _extract_customer_utterances_from_structured(parsed)

    return _extract_customer_utterances_from_text(content)


def _parse_content(content: str, *, format_hint: Optional[str]) -> Any:
    hint = (format_hint or "").strip().lower()
    if hint == "json":
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise TranscriptSeedError(f"Invalid JSON transcript: {e}") from e
    if hint in {"yaml", "yml"}:
        try:
            return yaml.safe_load(content)
        except yaml.YAMLError as e:
            raise TranscriptSeedError(f"Invalid YAML transcript: {e}") from e
    if hint in {"txt", "text", "log"}:
        return content

    # Auto-detect when no hint was provided.
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    try:
        loaded_yaml = yaml.safe_load(content)
    except yaml.YAMLError:
        return content

    if isinstance(loaded_yaml, (dict, list)):
        return loaded_yaml
    return content


def _extract_customer_utterances_from_structured(
    payload: dict[str, Any] | list[Any],
) -> list[dict[str, Optional[str]]]:
    candidates: list[dict[str, Optional[str]]] = []
    _walk_for_message_candidates(payload, candidates)

    utterances: list[dict[str, Optional[str]]] = []
    seen: set[tuple[str, str, str]] = set()
    for candidate in candidates:
        text = _normalize_whitespace(candidate.get("text") or "")
        if not text:
            continue
        speaker = _normalize_whitespace(candidate.get("speaker") or "")
        if speaker and not _is_customer_speaker(speaker):
            continue
        if not speaker and not _looks_like_customer_text(text):
            continue
        intent = _normalize_whitespace(candidate.get("intent") or "").lower() or None
        key = (speaker.lower(), text.lower(), intent or "")
        if key in seen:
            continue
        seen.add(key)
        utterances.append({"text": text, "intent": intent})
    return utterances


def _walk_for_message_candidates(node: Any, out: list[dict[str, Optional[str]]]) -> None:
    if isinstance(node, list):
        for item in node:
            _walk_for_message_candidates(item, out)
        return

    if not isinstance(node, dict):
        return

    candidate = _candidate_from_message_object(node)
    if candidate is not None:
        out.append(candidate)

    for value in node.values():
        _walk_for_message_candidates(value, out)


def _candidate_from_message_object(node: dict[str, Any]) -> Optional[dict[str, Optional[str]]]:
    text = _first_non_empty_str(
        node,
        [
            "text",
            "body",
            "message",
            "messageText",
            "content",
            "utterance",
            "userMessage",
        ],
    )
    if not text:
        return None

    speaker = _first_non_empty_str(
        node,
        [
            "speaker",
            "role",
            "sender",
            "senderRole",
            "participantPurpose",
            "purpose",
            "direction",
            "participantType",
            "authorRole",
        ],
    )

    # Nested speaker hints (for example {"from": {"name": "...", "role": "..."}})
    if not speaker:
        nested = node.get("from") or node.get("author") or node.get("participant")
        if isinstance(nested, dict):
            speaker = _first_non_empty_str(
                nested,
                ["role", "purpose", "type", "name", "displayName"],
            )
        elif isinstance(nested, str):
            speaker = nested

    intent = _first_non_empty_str(
        node,
        [
            "detected_intent",
            "detectedIntent",
            "intent",
            "intentName",
            "nlu_intent",
            "nluIntent",
        ],
    )

    return {
        "speaker": _normalize_whitespace(speaker or ""),
        "text": _normalize_whitespace(text),
        "intent": _normalize_whitespace(intent or ""),
    }


def _extract_customer_utterances_from_text(content: str) -> list[dict[str, Optional[str]]]:
    utterances: list[dict[str, Optional[str]]] = []
    line_pattern = re.compile(
        r"^\s*(?:\[[^\]]+\]\s*)?(?P<speaker>[A-Za-z][A-Za-z _-]{1,40})\s*[:|-]\s*(?P<text>.+?)\s*$"
    )
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = line_pattern.match(line)
        if not match:
            continue
        speaker = _normalize_whitespace(match.group("speaker"))
        text = _normalize_whitespace(match.group("text"))
        if not text or not _is_customer_speaker(speaker):
            continue
        utterances.append({"text": text, "intent": None})
    return utterances


def _build_seed_scenarios(
    utterances: list[dict[str, Optional[str]]],
) -> list[TestScenario]:
    scenarios: list[TestScenario] = []
    label_counts: dict[str, int] = {}
    for utterance in utterances:
        first_message = _normalize_whitespace(utterance.get("text") or "")
        if not first_message:
            continue

        inferred = _infer_seed_fields(
            first_message=first_message,
            detected_intent=(utterance.get("intent") or None),
        )
        label = inferred["label"]
        label_counts[label] = label_counts.get(label, 0) + 1
        scenario_name = f"{label} - Seed {label_counts[label]:02d}"

        scenario_kwargs: dict[str, Any] = {
            "name": scenario_name,
            "persona": (
                "A traveler contacting the WestJet Travel Agent for flight-related help. "
                "They explain their request clearly and provide needed details when asked."
            ),
            "goal": inferred["goal"],
            "first_message": first_message,
            "attempts": 1,
        }
        if inferred["expected_intent"]:
            scenario_kwargs["expected_intent"] = inferred["expected_intent"]

        scenarios.append(TestScenario(**scenario_kwargs))
    return scenarios


def _infer_seed_fields(
    *,
    first_message: str,
    detected_intent: Optional[str],
) -> dict[str, Optional[str]]:
    normalized_message = first_message.lower()
    expected_intent = (detected_intent or "").strip().lower() or None

    # Guideline pricing questions are behavior-evaluated, not strict intent judged.
    if expected_intent in {"guideline", "guidelines"}:
        expected_intent = None

    if _is_guideline_pricing_question(normalized_message):
        return {
            "label": "guideline",
            "expected_intent": None,
            "goal": (
                "Help the traveler with a pricing-guidance request. The goal is achieved "
                "when the WestJet Travel Agent clearly explains it does not provide "
                "specific baggage fee or pricing details in chat, then directs the traveler "
                "to the WestJet website or app for current costs."
            ),
        }

    if not expected_intent:
        expected_intent = _infer_intent_from_message(normalized_message)

    if expected_intent:
        pretty = expected_intent.replace("_", " ")
        return {
            "label": pretty,
            "expected_intent": expected_intent,
            "goal": (
                f"Help the traveler with a {pretty} request. The goal is achieved when the "
                "WestJet Travel Agent provides a relevant, policy-aligned response and clear "
                "next steps for this request."
            ),
        }

    return {
        "label": "general",
        "expected_intent": None,
        "goal": (
            "Help the traveler with their request. The goal is achieved when the WestJet "
            "Travel Agent provides a relevant, policy-aligned response and clear next steps."
        ),
    }


def _infer_intent_from_message(normalized_message: str) -> Optional[str]:
    if "vacation" in normalized_message:
        if "hotel" in normalized_message:
            return "vacation_flight_and_hotel"
        if "flight" in normalized_message or "package" in normalized_message:
            return "vacation_inquiry_flight_only"
    return None


def _is_guideline_pricing_question(normalized_message: str) -> bool:
    has_pricing = any(token in normalized_message for token in _PRICING_KEYWORDS)
    has_baggage = any(token in normalized_message for token in _BAGGAGE_KEYWORDS)
    return has_pricing and has_baggage


def _looks_like_customer_text(text: str) -> bool:
    # Fallback heuristic only used when speaker metadata is missing.
    return len(text.split()) >= 2


def _is_customer_speaker(speaker: str) -> bool:
    normalized = speaker.strip().lower()
    if not normalized:
        return False
    if any(token in normalized for token in _AGENT_SPEAKER_TOKENS):
        return False
    return any(token in normalized for token in _CUSTOMER_SPEAKER_TOKENS)


def _normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _first_non_empty_str(node: dict[str, Any], keys: list[str]) -> Optional[str]:
    for key in keys:
        value = node.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None
