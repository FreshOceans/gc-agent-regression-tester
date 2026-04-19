"""Transcript-to-suite seeding helpers for transcript-driven suite generation.

This module converts uploaded transcript content into a draft TestSuite.
It focuses on deterministic parsing and clear defaults:
- Extract customer utterances from JSON, YAML, CSV/TSV, or plain text transcript content.
- Build one-attempt scenarios from those utterances.
- Apply known behavior overrides (for example pricing-guidance/guideline flows).
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from io import StringIO
from typing import Any, Optional

import yaml

from .language_profiles import get_language_profile, normalize_language_code
from .models import TestScenario, TestSuite


class TranscriptSeedError(ValueError):
    """Raised when transcript content cannot be seeded into scenarios."""


@dataclass
class TranscriptSeedDiagnostics:
    """Simple extraction diagnostics for transcript preview UX."""

    total_candidates: int = 0
    utterances_found: int = 0
    scenarios_generated: int = 0
    skipped_non_customer: int = 0
    skipped_empty_or_noise: int = 0
    skipped_duplicate: int = 0
    truncated_by_max_scenarios: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def skipped_messages(self) -> int:
        return (
            self.skipped_non_customer
            + self.skipped_empty_or_noise
            + self.skipped_duplicate
        )


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
    "external",
    "externaluser",
    "customerparticipant",
    "webuser",
    "messengeruser",
}
_AGENT_SPEAKER_TOKENS = {
    "agent",
    "assistant",
    "bot",
    "system",
    "outbound",
    "server",
    "ivr",
    "workflow",
    "architect",
    "flow",
    "acd",
    "queue",
    "script",
    "auto",
    "automation",
    "agentparticipant",
    "virtualassistant",
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

_TEXT_FIELDS = {
    "text",
    "body",
    "message",
    "messagetext",
    "content",
    "utterance",
    "usermessage",
    "input",
    "prompt",
    "query",
    "question",
}
_SPEAKER_FIELDS = {
    "speaker",
    "role",
    "sender",
    "senderrole",
    "participantpurpose",
    "purpose",
    "direction",
    "participanttype",
    "authorrole",
    "author",
    "from",
}
_INTENT_FIELDS = {
    "detected_intent",
    "detectedintent",
    "intent",
    "intentname",
    "nlu_intent",
    "nluintent",
}

_IGNORED_EXACT_MESSAGES = {
    "conversation ended by user",
    "conversation ended",
    "presence events are not supported in this configuration",
    "typing",
    "typing...",
}
_IGNORED_PREFIXES = (
    "conversation_id:",
    "participant_id:",
    "detected_intent:",
    "conversationid:",
    "participantid:",
)


def seed_test_suite_from_transcript(
    content: str,
    *,
    format_hint: Optional[str] = None,
    suite_name: Optional[str] = None,
    max_scenarios: int = 50,
    language_code: str = "en",
) -> TestSuite:
    """Generate a draft TestSuite from transcript content."""
    suite, _ = seed_test_suite_from_transcript_with_diagnostics(
        content,
        format_hint=format_hint,
        suite_name=suite_name,
        max_scenarios=max_scenarios,
        language_code=language_code,
    )
    return suite


def seed_test_suite_from_transcript_with_diagnostics(
    content: str,
    *,
    format_hint: Optional[str] = None,
    suite_name: Optional[str] = None,
    max_scenarios: int = 50,
    language_code: str = "en",
) -> tuple[TestSuite, TranscriptSeedDiagnostics]:
    """Generate a draft TestSuite plus extraction diagnostics."""
    if max_scenarios < 1:
        raise TranscriptSeedError("max_scenarios must be at least 1")

    canonical_language = normalize_language_code(language_code, default="en")
    utterances, diagnostics = _extract_customer_utterances(
        content,
        format_hint=format_hint,
        language_code=canonical_language,
    )
    if not utterances:
        raise TranscriptSeedError(
            "No customer/user utterances were found in the uploaded transcript."
        )

    limited = utterances[:max_scenarios]
    diagnostics.truncated_by_max_scenarios = max(0, len(utterances) - len(limited))
    scenarios = _build_seed_scenarios(
        limited,
        language_code=canonical_language,
    )
    if not scenarios:
        raise TranscriptSeedError(
            "Transcript parsing succeeded, but no scenarios could be generated."
        )

    diagnostics.utterances_found = len(utterances)
    diagnostics.scenarios_generated = len(scenarios)

    if diagnostics.truncated_by_max_scenarios > 0:
        diagnostics.warnings.append(
            "Generated scenarios were truncated by max scenario limit. "
            f"Ignored {diagnostics.truncated_by_max_scenarios} additional utterance(s)."
        )

    if diagnostics.total_candidates >= 10 and diagnostics.skipped_messages >= (
        diagnostics.total_candidates * 0.4
    ):
        diagnostics.warnings.append(
            "A large portion of transcript messages were skipped. "
            "Verify speaker labels and transcript format if output looks incomplete."
        )

    seeded_suite_name = (
        (suite_name or "").strip() or "Transcript Regression Suite"
    )
    return (
        TestSuite(
            name=seeded_suite_name,
            language=canonical_language,
            scenarios=scenarios,
        ),
        diagnostics,
    )


def _extract_customer_utterances(
    content: str,
    *,
    format_hint: Optional[str] = None,
    language_code: str = "en",
) -> tuple[list[dict[str, Optional[str]]], TranscriptSeedDiagnostics]:
    diagnostics = TranscriptSeedDiagnostics()
    hint = (format_hint or "").strip().lower()

    if hint == "csv":
        utterances = _extract_customer_utterances_from_delimited_text(
            content,
            delimiter=",",
            diagnostics=diagnostics,
            auto_detect=False,
            language_code=language_code,
        )
        return utterances, diagnostics

    if hint == "tsv":
        utterances = _extract_customer_utterances_from_delimited_text(
            content,
            delimiter="\t",
            diagnostics=diagnostics,
            auto_detect=False,
            language_code=language_code,
        )
        return utterances, diagnostics

    parsed = _parse_content(content, format_hint=hint)

    if isinstance(parsed, (dict, list)):
        utterances = _extract_customer_utterances_from_structured(
            parsed,
            diagnostics=diagnostics,
            language_code=language_code,
        )
        return utterances, diagnostics

    if _looks_tabular(content, delimiter="\t"):
        utterances = _extract_customer_utterances_from_delimited_text(
            content,
            delimiter="\t",
            diagnostics=diagnostics,
            auto_detect=True,
            language_code=language_code,
        )
        if utterances:
            return utterances, diagnostics

    if _looks_tabular(content, delimiter=","):
        utterances = _extract_customer_utterances_from_delimited_text(
            content,
            delimiter=",",
            diagnostics=diagnostics,
            auto_detect=True,
            language_code=language_code,
        )
        if utterances:
            return utterances, diagnostics

    utterances = _extract_customer_utterances_from_text(
        content,
        diagnostics=diagnostics,
        language_code=language_code,
    )
    return utterances, diagnostics


def _parse_content(content: str, *, format_hint: str) -> Any:
    if format_hint == "json":
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise TranscriptSeedError(f"Invalid JSON transcript: {e}") from e
    if format_hint in {"yaml", "yml"}:
        try:
            return yaml.safe_load(content)
        except yaml.YAMLError as e:
            raise TranscriptSeedError(f"Invalid YAML transcript: {e}") from e
    if format_hint in {"txt", "text", "log", "csv", "tsv"}:
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
    *,
    diagnostics: TranscriptSeedDiagnostics,
    language_code: str,
) -> list[dict[str, Optional[str]]]:
    candidates: list[dict[str, Optional[str]]] = []
    _walk_for_message_candidates(payload, candidates)

    utterances: list[dict[str, Optional[str]]] = []
    seen: set[tuple[str, str]] = set()
    for candidate in candidates:
        diagnostics.total_candidates += 1
        text = _normalize_whitespace(candidate.get("text") or "")
        if not text or _is_noise_or_boilerplate(text, language_code=language_code):
            diagnostics.skipped_empty_or_noise += 1
            continue

        speaker = _normalize_whitespace(candidate.get("speaker") or "")
        if not speaker:
            diagnostics.skipped_non_customer += 1
            continue
        if not _is_customer_speaker(speaker, language_code=language_code):
            diagnostics.skipped_non_customer += 1
            continue

        intent = _normalize_whitespace(candidate.get("intent") or "").lower() or None
        dedupe_key = (_normalize_for_dedupe(text), intent or "")
        if dedupe_key in seen:
            diagnostics.skipped_duplicate += 1
            continue

        seen.add(dedupe_key)
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
    text = _extract_message_text(node)
    if not text:
        return None

    speaker = _extract_speaker_label(node)
    intent = _extract_intent_label(node)

    return {
        "speaker": _normalize_whitespace(speaker or ""),
        "text": _normalize_whitespace(text),
        "intent": _normalize_whitespace(intent or ""),
    }


def _extract_message_text(node: dict[str, Any]) -> Optional[str]:
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
            "normalizedText",
            "query",
            "prompt",
            "input",
        ],
    )
    if text:
        return text

    for nested_key in ("payload", "data", "eventBody", "message", "content"):
        nested = node.get(nested_key)
        nested_text = _extract_text_from_nested(nested)
        if nested_text:
            return nested_text

    return None


def _extract_text_from_nested(node: Any) -> Optional[str]:
    if isinstance(node, str) and node.strip():
        return node

    if not isinstance(node, dict):
        return None

    direct = _first_non_empty_str(
        node,
        [
            "text",
            "body",
            "message",
            "messageText",
            "content",
            "utterance",
            "query",
            "prompt",
            "input",
            "value",
        ],
    )
    if direct:
        return direct

    for key in ("payload", "data", "message", "content"):
        value = node.get(key)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, dict):
            inner = _first_non_empty_str(
                value,
                ["text", "body", "message", "messageText", "content", "utterance"],
            )
            if inner:
                return inner

    return None


def _extract_speaker_label(node: dict[str, Any]) -> Optional[str]:
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
    if speaker:
        return speaker

    for nested_key in ("from", "author", "participant", "sender", "participantInfo", "metadata"):
        nested = node.get(nested_key)
        nested_speaker = _extract_speaker_from_nested(nested)
        if nested_speaker:
            return nested_speaker

    return None


def _extract_speaker_from_nested(node: Any) -> Optional[str]:
    if isinstance(node, str) and node.strip():
        return node

    if not isinstance(node, dict):
        return None

    return _first_non_empty_str(
        node,
        [
            "role",
            "purpose",
            "type",
            "name",
            "displayName",
            "participantPurpose",
            "direction",
        ],
    )


def _extract_intent_label(node: dict[str, Any]) -> Optional[str]:
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
    if intent:
        return intent

    for nested_key in ("nlu", "intent", "metadata", "analysis"):
        nested = node.get(nested_key)
        if isinstance(nested, dict):
            nested_intent = _first_non_empty_str(
                nested,
                [
                    "detected_intent",
                    "detectedIntent",
                    "intent",
                    "intentName",
                    "name",
                ],
            )
            if nested_intent:
                return nested_intent

    return None


def _extract_customer_utterances_from_delimited_text(
    content: str,
    *,
    delimiter: str,
    diagnostics: TranscriptSeedDiagnostics,
    auto_detect: bool,
    language_code: str,
) -> list[dict[str, Optional[str]]]:
    rows = list(csv.reader(StringIO(content), delimiter=delimiter))
    if not rows:
        return []

    if auto_detect and not _rows_look_tabular(rows):
        return []

    header_tokens = [_normalize_header_token(value) for value in rows[0]]
    text_idx = _find_column_index(header_tokens, _TEXT_FIELDS)
    speaker_idx = _find_column_index(header_tokens, _SPEAKER_FIELDS)
    intent_idx = _find_column_index(header_tokens, _INTENT_FIELDS)

    has_header = (
        text_idx is not None
        or speaker_idx is not None
        or intent_idx is not None
    )
    data_rows = rows[1:] if has_header else rows

    utterances: list[dict[str, Optional[str]]] = []
    seen: set[tuple[str, str]] = set()

    for row in data_rows:
        if not row:
            continue
        diagnostics.total_candidates += 1

        if has_header:
            text = _safe_cell(row, text_idx)
            speaker = _safe_cell(row, speaker_idx)
            intent = _safe_cell(row, intent_idx)
        else:
            speaker = row[0] if len(row) >= 2 else ""
            text = row[1] if len(row) >= 2 else row[0]
            intent = row[2] if len(row) >= 3 else ""

        normalized_text = _normalize_whitespace(text)
        if not normalized_text or _is_noise_or_boilerplate(
            normalized_text,
            language_code=language_code,
        ):
            diagnostics.skipped_empty_or_noise += 1
            continue

        normalized_speaker = _normalize_whitespace(speaker)
        if normalized_speaker:
            if not _is_customer_speaker(
                normalized_speaker,
                language_code=language_code,
            ):
                diagnostics.skipped_non_customer += 1
                continue
        elif not _looks_like_customer_text(normalized_text):
            diagnostics.skipped_non_customer += 1
            continue

        normalized_intent = _normalize_whitespace(intent).lower() or None
        dedupe_key = (_normalize_for_dedupe(normalized_text), normalized_intent or "")
        if dedupe_key in seen:
            diagnostics.skipped_duplicate += 1
            continue

        seen.add(dedupe_key)
        utterances.append(
            {
                "text": normalized_text,
                "intent": normalized_intent,
            }
        )

    return utterances


def _safe_cell(row: list[str], index: Optional[int]) -> str:
    if index is None:
        return ""
    if index < 0 or index >= len(row):
        return ""
    return row[index]


def _rows_look_tabular(rows: list[list[str]]) -> bool:
    if len(rows) < 2:
        return False

    multi_column_rows = sum(1 for row in rows if len(row) >= 2)
    return multi_column_rows >= max(2, int(len(rows) * 0.4))


def _looks_tabular(content: str, *, delimiter: str) -> bool:
    rows = list(csv.reader(StringIO(content), delimiter=delimiter))
    if not _rows_look_tabular(rows):
        return False

    header_tokens = [_normalize_header_token(value) for value in rows[0]]
    if any(token in _TEXT_FIELDS or token in _SPEAKER_FIELDS for token in header_tokens):
        return True

    # Even without headers, rows with multiple columns across the sample are likely tabular.
    return True


def _normalize_header_token(value: str) -> str:
    return re.sub(r"[^a-z0-9_]", "", (value or "").strip().lower())


def _find_column_index(tokens: list[str], candidates: set[str]) -> Optional[int]:
    for index, token in enumerate(tokens):
        if token in candidates:
            return index
    return None


def _extract_customer_utterances_from_text(
    content: str,
    *,
    diagnostics: TranscriptSeedDiagnostics,
    language_code: str,
) -> list[dict[str, Optional[str]]]:
    utterances: list[dict[str, Optional[str]]] = []
    seen: set[tuple[str, str]] = set()
    line_pattern = re.compile(
        r"^\s*(?:\[[^\]]+\]\s*)?(?P<speaker>[^\W\d_][\w \-']{1,50})\s*[:|-]\s*(?P<text>.+?)\s*$",
        flags=re.UNICODE,
    )

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        match = line_pattern.match(line)
        if not match:
            continue

        diagnostics.total_candidates += 1
        speaker = _normalize_whitespace(match.group("speaker"))
        text = _normalize_whitespace(match.group("text"))

        if not text or _is_noise_or_boilerplate(text, language_code=language_code):
            diagnostics.skipped_empty_or_noise += 1
            continue
        if not _is_customer_speaker(speaker, language_code=language_code):
            diagnostics.skipped_non_customer += 1
            continue

        dedupe_key = (_normalize_for_dedupe(text), "")
        if dedupe_key in seen:
            diagnostics.skipped_duplicate += 1
            continue

        seen.add(dedupe_key)
        utterances.append({"text": text, "intent": None})

    return utterances


def _build_seed_scenarios(
    utterances: list[dict[str, Optional[str]]],
    *,
    language_code: str,
) -> list[TestScenario]:
    profile = get_language_profile(language_code)
    seeded_persona = str(profile.get("seeded_persona") or "")
    seeded_goal_guideline = str(profile.get("seeded_goal_guideline") or "")
    seeded_goal_intent_template = str(profile.get("seeded_goal_intent_template") or "")
    seeded_goal_general = str(profile.get("seeded_goal_general") or "")

    scenarios: list[TestScenario] = []
    label_counts: dict[str, int] = {}
    for utterance in utterances:
        first_message = _normalize_whitespace(utterance.get("text") or "")
        if not first_message:
            continue

        inferred = _infer_seed_fields(
            first_message=first_message,
            detected_intent=(utterance.get("intent") or None),
            seeded_goal_guideline=seeded_goal_guideline,
            seeded_goal_intent_template=seeded_goal_intent_template,
            seeded_goal_general=seeded_goal_general,
        )
        label = inferred["label"]
        label_counts[label] = label_counts.get(label, 0) + 1
        scenario_name = f"{label} - Seed {label_counts[label]:02d}"

        scenario_kwargs: dict[str, Any] = {
            "name": scenario_name,
            "persona": seeded_persona,
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
    seeded_goal_guideline: str,
    seeded_goal_intent_template: str,
    seeded_goal_general: str,
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
            "goal": seeded_goal_guideline,
        }

    if not expected_intent:
        expected_intent = _infer_intent_from_message(normalized_message)

    if expected_intent:
        pretty = expected_intent.replace("_", " ")
        return {
            "label": pretty,
            "expected_intent": expected_intent,
            "goal": seeded_goal_intent_template.format(intent_label=pretty),
        }

    return {
        "label": "general",
        "expected_intent": None,
        "goal": seeded_goal_general,
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


def _is_noise_or_boilerplate(text: str, *, language_code: str) -> bool:
    profile = get_language_profile(language_code)
    ignored_exact = set(_IGNORED_EXACT_MESSAGES)
    ignored_exact.update(
        str(item).strip().lower()
        for item in profile.get("transcript_ignored_exact_messages", set())
        if str(item).strip()
    )
    ignored_prefixes = tuple(_IGNORED_PREFIXES) + tuple(
        str(item).strip().lower()
        for item in profile.get("transcript_ignored_prefixes", tuple())
        if str(item).strip()
    )

    normalized = _normalize_whitespace(text).lower()
    if not normalized:
        return True
    if normalized in ignored_exact:
        return True
    if any(normalized.startswith(prefix) for prefix in ignored_prefixes):
        return True
    if re.fullmatch(r"[_\W]+", normalized):
        return True
    if re.search(r"\b(message us|agent joined|joined the conversation|left the conversation)\b", normalized):
        return True
    if "virtual assistant" in normalized and "how may i help" in normalized:
        return True
    return False


def _looks_like_customer_text(text: str) -> bool:
    # Fallback heuristic only used when speaker metadata is missing.
    return len(text.split()) >= 2


def _is_customer_speaker(speaker: str, *, language_code: str) -> bool:
    profile = get_language_profile(language_code)
    customer_tokens = set(_CUSTOMER_SPEAKER_TOKENS)
    customer_tokens.update(
        str(item).strip().lower()
        for item in profile.get("transcript_customer_speaker_tokens", set())
        if str(item).strip()
    )
    agent_tokens = set(_AGENT_SPEAKER_TOKENS)
    agent_tokens.update(
        str(item).strip().lower()
        for item in profile.get("transcript_agent_speaker_tokens", set())
        if str(item).strip()
    )

    normalized = speaker.strip().lower()
    if not normalized:
        return False
    compact = re.sub(r"[^a-z0-9]", "", normalized)

    if any(token in normalized or token in compact for token in agent_tokens):
        return False

    return any(token in normalized or token in compact for token in customer_tokens)


def _normalize_for_dedupe(value: str) -> str:
    lowered = (value or "").strip().lower()
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _first_non_empty_str(node: dict[str, Any], keys: list[str]) -> Optional[str]:
    for key in keys:
        value = node.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None
