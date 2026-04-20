"""Unit tests for Phase 11 judging mechanics helpers."""

from src.judging_mechanics import (
    format_mechanics_summary,
    resolve_judging_mechanics_config,
    score_goal_evaluation,
    score_journey_evaluation,
)
from src.models import GoalEvaluation, JourneyValidationResult


def test_resolve_judging_mechanics_config_normalizes_and_clamps():
    config = resolve_judging_mechanics_config(
        {
            "enabled": True,
            "objective_profile": "journey",
            "strictness": "strict",
            "tolerance": 0.8,
            "explanation_mode": "verbose",
            "containment_weight": -1,
            "fulfillment_weight": 150,
            "path_weight": 0.25,
        }
    )

    assert config["enabled"] is True
    assert config["objective_profile"] == "journey_focused"
    assert config["strictness"] == "strict"
    # tolerance is capped at 0.45
    assert config["tolerance"] == 0.45
    assert 0.0 <= config["threshold"] <= 1.0
    assert config["containment_weight"] == 0.0
    assert config["fulfillment_weight"] == 100.0
    assert config["path_weight"] == 0.25


def test_score_goal_evaluation_passes_lenient_threshold():
    result = score_goal_evaluation(
        evaluation=GoalEvaluation(
            success=True,
            explanation="Goal achieved and fully resolved.",
        ),
        config={
            "enabled": True,
            "strictness": "lenient",
            "objective_profile": "blended",
            "tolerance": 0.0,
            "explanation_mode": "standard",
        },
        hard_gate_passed=True,
    )

    assert result["passed_threshold"] is True
    assert result["final_gate_passed"] is True
    assert result["score"] >= result["threshold"]


def test_score_journey_evaluation_respects_hard_gate_and_threshold():
    result = score_journey_evaluation(
        journey_result=JourneyValidationResult(
            category_match=False,
            fulfilled=False,
            path_correct=False,
            contained=True,
            expected_category="flight_change",
        ),
        config={
            "enabled": True,
            "strictness": "strict",
            "objective_profile": "journey_focused",
            "tolerance": 0.0,
            "containment_weight": 0.35,
            "fulfillment_weight": 0.45,
            "path_weight": 0.20,
            "explanation_mode": "standard",
        },
        hard_gate_passed=True,
    )

    assert result["score"] < result["threshold"]
    assert result["passed_threshold"] is False
    assert result["final_gate_passed"] is False


def test_format_mechanics_summary_respects_modes():
    payload = {
        "score": 0.83,
        "threshold": 0.76,
        "passed_threshold": True,
        "objective_profile": "blended",
        "strictness": "balanced",
        "tolerance": 0.1,
        "criteria": {"contained": 1.0, "fulfilled": 1.0},
    }

    concise = format_mechanics_summary({**payload, "explanation_mode": "concise"})
    verbose = format_mechanics_summary({**payload, "explanation_mode": "verbose"})

    assert "Judging Mechanics: PASS" in concise
    assert "Criteria:" not in concise
    assert "Judging Mechanics: PASS" in verbose
    assert "Criteria:" in verbose
