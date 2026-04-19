"""Helpers for journey-mode harness behavior and primary category resolution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


HARNESS_STANDARD = "standard"
HARNESS_JOURNEY = "journey"
VALID_HARNESS_MODES = {HARNESS_STANDARD, HARNESS_JOURNEY}

CATEGORY_STRATEGY_RULES_FIRST = "rules_first"
CATEGORY_STRATEGY_LLM_FIRST = "llm_first"
VALID_CATEGORY_STRATEGIES = {
    CATEGORY_STRATEGY_RULES_FIRST,
    CATEGORY_STRATEGY_LLM_FIRST,
}


DEFAULT_PRIMARY_CATEGORIES: list[dict[str, Any]] = [
    {
        "name": "flight_cancel",
        "keywords": ["cancel", "refund", "cancel my booking", "cancel flight"],
        "rubric": "The journey should progress through flight cancellation flow and provide cancellation/refund next steps.",
    },
    {
        "name": "flight_change",
        "keywords": ["change flight", "reschedule", "flight changed", "new flight time"],
        "rubric": "The journey should route through flight-change handling and collect required itinerary details.",
    },
    {
        "name": "flight_status",
        "keywords": ["flight status", "is my flight on time", "flight delayed", "departure time"],
        "rubric": "The journey should provide status/operational guidance, not unrelated booking flows.",
    },
    {
        "name": "baggage",
        "keywords": ["baggage", "bag", "carry on", "checked bag"],
        "rubric": "The journey should answer baggage policies and baggage process questions accurately.",
    },
    {
        "name": "pets",
        "keywords": ["pet", "dog", "cat", "animal travel"],
        "rubric": "The journey should provide pet-travel policies and restrictions with compliant guidance.",
    },
    {
        "name": "vacation",
        "keywords": ["vacation", "package", "flight and hotel", "trip package"],
        "rubric": "The journey should route into vacation/package qualification and branch correctly by traveler choice.",
    },
    {
        "name": "speak_to_agent",
        "keywords": ["human agent", "live agent", "representative", "talk to someone"],
        "rubric": "The journey should handle escalation intent correctly and confirm transfer path when requested.",
    },
    {
        "name": "guidelines",
        "keywords": ["price", "pricing", "fee", "charge", "cost"],
        "rubric": "The journey should provide policy-safe guidance for pricing/cost questions and compliant redirection.",
    },
]


def normalize_harness_mode(
    value: Optional[str],
    *,
    allow_none: bool = False,
) -> Optional[str]:
    if value is None:
        if allow_none:
            return None
        return HARNESS_STANDARD
    normalized = str(value).strip().lower()
    if not normalized:
        if allow_none:
            return None
        return HARNESS_STANDARD
    if normalized not in VALID_HARNESS_MODES:
        raise ValueError(
            "harness_mode must be one of: standard, journey"
        )
    return normalized


def normalize_category_strategy(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if not normalized:
        return CATEGORY_STRATEGY_RULES_FIRST
    if normalized not in VALID_CATEGORY_STRATEGIES:
        raise ValueError(
            "journey_category_strategy must be one of: rules_first, llm_first"
        )
    return normalized


def resolve_effective_harness_mode(
    *,
    runtime_override: Optional[str],
    suite_mode: Optional[str],
    config_mode: Optional[str],
) -> str:
    for candidate in (runtime_override, suite_mode, config_mode, HARNESS_STANDARD):
        try:
            normalized = normalize_harness_mode(candidate, allow_none=True)
        except ValueError:
            continue
        if normalized:
            return normalized
    return HARNESS_STANDARD


def load_category_overrides(
    *,
    categories_json: str = "",
    categories_file: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Load optional category overrides from JSON text and/or JSON file."""
    merged: list[dict[str, Any]] = []

    raw_json = str(categories_json or "").strip()
    if raw_json:
        try:
            loaded = json.loads(raw_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid journey_primary_categories_json: {e}") from e
        if not isinstance(loaded, list):
            raise ValueError(
                "journey_primary_categories_json must be a JSON array of category objects"
            )
        merged.extend(item for item in loaded if isinstance(item, dict))

    file_path = str(categories_file or "").strip()
    if file_path:
        path = Path(file_path)
        if not path.exists():
            raise ValueError(
                f"journey_primary_categories_file not found: {file_path}"
            )
        try:
            loaded_file = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON in journey_primary_categories_file: {e}"
            ) from e
        if not isinstance(loaded_file, list):
            raise ValueError(
                "journey_primary_categories_file must contain a JSON array of category objects"
            )
        merged.extend(item for item in loaded_file if isinstance(item, dict))

    return merged

