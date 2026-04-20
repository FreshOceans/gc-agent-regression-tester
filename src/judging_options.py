"""Normalization helpers and defaults for Phase 11 judging mechanics."""

from __future__ import annotations

from typing import Optional


OBJECTIVE_PROFILE_INTENT = "intent_focused"
OBJECTIVE_PROFILE_JOURNEY = "journey_focused"
OBJECTIVE_PROFILE_BLENDED = "blended"

OBJECTIVE_PROFILE_ALIASES = {
    OBJECTIVE_PROFILE_INTENT,
    "intent",
    OBJECTIVE_PROFILE_JOURNEY,
    "journey",
    OBJECTIVE_PROFILE_BLENDED,
}

STRICTNESS_STRICT = "strict"
STRICTNESS_BALANCED = "balanced"
STRICTNESS_LENIENT = "lenient"

STRICTNESS_THRESHOLDS = {
    STRICTNESS_STRICT: 0.88,
    STRICTNESS_BALANCED: 0.76,
    STRICTNESS_LENIENT: 0.62,
}

EXPLANATION_MODE_CONCISE = "concise"
EXPLANATION_MODE_STANDARD = "standard"
EXPLANATION_MODE_VERBOSE = "verbose"

EXPLANATION_MODES = {
    EXPLANATION_MODE_CONCISE,
    EXPLANATION_MODE_STANDARD,
    EXPLANATION_MODE_VERBOSE,
}


def normalize_objective_profile(
    value: Optional[str],
    *,
    default: str = OBJECTIVE_PROFILE_BLENDED,
) -> str:
    raw = str(value or default).strip().lower()
    if raw in {"intent", OBJECTIVE_PROFILE_INTENT}:
        return OBJECTIVE_PROFILE_INTENT
    if raw in {"journey", OBJECTIVE_PROFILE_JOURNEY}:
        return OBJECTIVE_PROFILE_JOURNEY
    if raw == OBJECTIVE_PROFILE_BLENDED:
        return OBJECTIVE_PROFILE_BLENDED
    raise ValueError(
        "judging_objective_profile must be one of: "
        "intent_focused, journey_focused, blended"
    )


def normalize_judging_strictness(
    value: Optional[str],
    *,
    default: str = STRICTNESS_BALANCED,
) -> str:
    raw = str(value or default).strip().lower()
    if raw not in STRICTNESS_THRESHOLDS:
        raise ValueError(
            "judging_strictness must be one of: strict, balanced, lenient"
        )
    return raw


def normalize_explanation_mode(
    value: Optional[str],
    *,
    default: str = EXPLANATION_MODE_STANDARD,
) -> str:
    raw = str(value or default).strip().lower()
    if raw not in EXPLANATION_MODES:
        raise ValueError(
            "judging_explanation_mode must be one of: concise, standard, verbose"
        )
    return raw


def threshold_for_strictness(strictness: str, tolerance: float) -> float:
    normalized = normalize_judging_strictness(strictness)
    base = STRICTNESS_THRESHOLDS[normalized]
    adjusted = base - float(tolerance or 0.0)
    if adjusted < 0.0:
        return 0.0
    if adjusted > 1.0:
        return 1.0
    return adjusted
