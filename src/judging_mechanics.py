"""Deterministic scoring helpers for Phase 11 judging mechanics."""

from __future__ import annotations

from typing import Optional

from .judging_options import (
    EXPLANATION_MODE_CONCISE,
    EXPLANATION_MODE_VERBOSE,
    OBJECTIVE_PROFILE_BLENDED,
    OBJECTIVE_PROFILE_INTENT,
    OBJECTIVE_PROFILE_JOURNEY,
    normalize_explanation_mode,
    normalize_judging_strictness,
    normalize_objective_profile,
    threshold_for_strictness,
)
from .models import GoalEvaluation, JourneyValidationResult


def resolve_judging_mechanics_config(raw: Optional[dict]) -> dict:
    payload = raw if isinstance(raw, dict) else {}
    enabled = bool(payload.get("enabled", False))
    objective_profile = normalize_objective_profile(
        payload.get("objective_profile"),
        default=OBJECTIVE_PROFILE_BLENDED,
    )
    strictness = normalize_judging_strictness(
        payload.get("strictness"),
        default="balanced",
    )
    tolerance_raw = payload.get("tolerance", 0.0)
    try:
        tolerance = float(tolerance_raw)
    except (TypeError, ValueError):
        tolerance = 0.0
    tolerance = max(0.0, min(0.45, tolerance))

    explanation_mode = normalize_explanation_mode(
        payload.get("explanation_mode"),
        default="standard",
    )

    containment_weight = _as_weight(payload.get("containment_weight", 0.35))
    fulfillment_weight = _as_weight(payload.get("fulfillment_weight", 0.45))
    path_weight = _as_weight(payload.get("path_weight", 0.20))

    return {
        "enabled": enabled,
        "objective_profile": objective_profile,
        "strictness": strictness,
        "tolerance": tolerance,
        "threshold": threshold_for_strictness(strictness, tolerance),
        "explanation_mode": explanation_mode,
        "containment_weight": containment_weight,
        "fulfillment_weight": fulfillment_weight,
        "path_weight": path_weight,
    }


def score_goal_evaluation(
    *,
    evaluation: GoalEvaluation,
    config: dict,
    hard_gate_passed: bool,
) -> dict:
    """Score standard (goal-eval) attempts.

    Deterministic heuristic so operators can tune strictness/tolerance.
    """
    mechanics = resolve_judging_mechanics_config(config)
    explanation = str(evaluation.explanation or "").strip().lower()

    score = 0.60 if evaluation.success else 0.35
    positive_cues = ["achieved", "completed", "success", "resolved", "done"]
    uncertain_cues = ["partial", "partially", "unclear", "maybe", "not sure"]
    negative_cues = ["not achieved", "failed", "unable", "cannot", "did not"]

    if any(token in explanation for token in positive_cues):
        score += 0.20
    if any(token in explanation for token in uncertain_cues):
        score -= 0.20
    if any(token in explanation for token in negative_cues):
        score -= 0.15

    score = _clamp(score)
    threshold = mechanics["threshold"]
    passed_threshold = score >= threshold
    final_gate_passed = bool(hard_gate_passed and passed_threshold)

    return {
        "enabled": mechanics["enabled"],
        "objective_profile": mechanics["objective_profile"],
        "strictness": mechanics["strictness"],
        "tolerance": mechanics["tolerance"],
        "threshold": threshold,
        "score": score,
        "passed_threshold": passed_threshold,
        "hard_gate_passed": bool(hard_gate_passed),
        "final_gate_passed": final_gate_passed,
        "explanation_mode": mechanics["explanation_mode"],
        "criteria": {
            "goal_success_signal": 1.0 if evaluation.success else 0.0,
            "positive_cues": 1.0 if any(token in explanation for token in positive_cues) else 0.0,
            "uncertainty_penalty": 1.0 if any(token in explanation for token in uncertain_cues) else 0.0,
            "negative_penalty": 1.0 if any(token in explanation for token in negative_cues) else 0.0,
        },
    }


def score_journey_evaluation(
    *,
    journey_result: JourneyValidationResult,
    config: dict,
    hard_gate_passed: bool,
) -> dict:
    """Score journey attempts using configurable containment/fulfillment/path weights."""
    mechanics = resolve_judging_mechanics_config(config)

    containment_score = 1.0 if journey_result.contained is True else 0.0
    fulfillment_score = 1.0 if journey_result.fulfilled else 0.0
    path_score = 1.0 if journey_result.path_correct else 0.0

    weight_total = (
        mechanics["containment_weight"]
        + mechanics["fulfillment_weight"]
        + mechanics["path_weight"]
    )
    if weight_total <= 0:
        weighted_journey = 0.0
    else:
        weighted_journey = (
            containment_score * mechanics["containment_weight"]
            + fulfillment_score * mechanics["fulfillment_weight"]
            + path_score * mechanics["path_weight"]
        ) / weight_total

    if journey_result.category_match is None and not journey_result.expected_category:
        category_score = 1.0
    else:
        category_score = 1.0 if journey_result.category_match is True else 0.0

    profile = mechanics["objective_profile"]
    if profile == OBJECTIVE_PROFILE_INTENT:
        score = (category_score * 0.65) + (weighted_journey * 0.35)
    elif profile == OBJECTIVE_PROFILE_JOURNEY:
        score = (weighted_journey * 0.85) + (category_score * 0.15)
    else:
        score = (weighted_journey * 0.65) + (category_score * 0.35)

    score = _clamp(score)
    threshold = mechanics["threshold"]
    passed_threshold = score >= threshold
    final_gate_passed = bool(hard_gate_passed and passed_threshold)

    return {
        "enabled": mechanics["enabled"],
        "objective_profile": profile,
        "strictness": mechanics["strictness"],
        "tolerance": mechanics["tolerance"],
        "threshold": threshold,
        "score": score,
        "passed_threshold": passed_threshold,
        "hard_gate_passed": bool(hard_gate_passed),
        "final_gate_passed": final_gate_passed,
        "explanation_mode": mechanics["explanation_mode"],
        "criteria": {
            "contained": containment_score,
            "fulfilled": fulfillment_score,
            "path_correct": path_score,
            "category_match": category_score,
            "weighted_journey": weighted_journey,
        },
    }


def format_mechanics_summary(result: dict) -> str:
    """Render explanation snippet according to explanation mode."""
    if not isinstance(result, dict):
        return ""

    mode = normalize_explanation_mode(result.get("explanation_mode"), default="standard")
    score = float(result.get("score", 0.0) or 0.0)
    threshold = float(result.get("threshold", 0.0) or 0.0)
    passed = bool(result.get("passed_threshold", False))
    status = "PASS" if passed else "FAIL"

    if mode == EXPLANATION_MODE_CONCISE:
        return (
            "Judging Mechanics: "
            f"{status} (score={score:.3f}, threshold={threshold:.3f})."
        )

    profile = str(result.get("objective_profile") or OBJECTIVE_PROFILE_BLENDED)
    strictness = str(result.get("strictness") or "balanced")
    tolerance = float(result.get("tolerance", 0.0) or 0.0)
    header = (
        "Judging Mechanics: "
        f"{status} (score={score:.3f}, threshold={threshold:.3f}, "
        f"profile={profile}, strictness={strictness}, tolerance={tolerance:.2f})."
    )

    if mode != EXPLANATION_MODE_VERBOSE:
        return header

    criteria = result.get("criteria")
    if not isinstance(criteria, dict) or not criteria:
        return header
    criteria_bits = ", ".join(
        f"{name}={float(value):.3f}"
        for name, value in criteria.items()
        if isinstance(value, (int, float))
    )
    if not criteria_bits:
        return header
    return f"{header}\nCriteria: {criteria_bits}."


def _as_weight(raw: object) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = 0.0
    if value < 0:
        return 0.0
    if value > 100.0:
        return 100.0
    return value


def _clamp(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value
