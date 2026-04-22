"""Unit tests for Gemma judge execution routing."""

from unittest.mock import patch

from src.judge_execution import (
    build_judge_execution_client,
    resolve_effective_judge_model_name,
)
from src.models import AppConfig, GoalEvaluation


def test_resolve_effective_judge_model_name_uses_single_override():
    config = AppConfig(
        gc_region="us-east-1",
        gc_deployment_id="deploy-123",
        ollama_model="llama3.2",
        judge_execution_mode="single",
        judge_single_model="gemma4:e4b",
    )

    assert resolve_effective_judge_model_name(config) == "llama3.2"


def test_resolve_effective_judge_model_name_uses_fixed_dual_primary():
    config = AppConfig(
        gc_region="us-east-1",
        gc_deployment_id="deploy-123",
        ollama_model="llama3.2",
        judge_execution_mode="dual_strict_fallback",
        judge_single_model="gemma4:31b",
    )

    assert resolve_effective_judge_model_name(config) == "gemma4:e4b"


def test_dual_fallback_triggers_on_low_confidence_classification():
    config = AppConfig(
        gc_region="us-east-1",
        gc_deployment_id="deploy-123",
        judge_execution_mode="dual_strict_fallback",
    )
    client = build_judge_execution_client(config)
    client.reset_attempt_diagnostics()

    def _fake_classify(self, *, first_message, categories, language_code="en"):
        if self.model == "gemma4:e4b":
            return {
                "category": "flight_cancel",
                "confidence": 0.42,
                "explanation": "low confidence primary",
            }
        return {
            "category": "flight_cancel",
            "confidence": 0.93,
            "explanation": "fallback match",
        }

    with patch(
        "src.judge_llm.JudgeLLMClient.classify_primary_category",
        new=_fake_classify,
    ):
        result = client.classify_primary_category(
            first_message="I need to cancel my booking",
            categories=[{"name": "flight_cancel"}],
            language_code="en",
        )

    assert result["category"] == "flight_cancel"
    assert result["confidence"] == 0.93

    status_messages = client.consume_pending_status_messages()
    diagnostics = client.consume_attempt_diagnostics()

    assert any("Judge fallback triggered for classify_primary_category" in item for item in status_messages)
    assert len(diagnostics) == 1
    assert diagnostics[0].fallback_used is True
    assert diagnostics[0].fallback_reason == "low_confidence"
    assert diagnostics[0].primary_model == "gemma4:e4b"
    assert diagnostics[0].fallback_model == "gemma4:31b"


def test_dual_mode_does_not_fallback_on_valid_negative_goal_result():
    config = AppConfig(
        gc_region="us-east-1",
        gc_deployment_id="deploy-123",
        judge_execution_mode="dual_strict_fallback",
    )
    client = build_judge_execution_client(config)
    client.reset_attempt_diagnostics()
    called_models = []

    def _fake_evaluate_goal(self, *, persona, goal, conversation_history, language_code="en"):
        called_models.append(self.model)
        return GoalEvaluation(success=False, explanation="Goal not achieved.")

    with patch(
        "src.judge_llm.JudgeLLMClient.evaluate_goal",
        new=_fake_evaluate_goal,
    ):
        result = client.evaluate_goal(
            persona="Traveler",
            goal="Cancel the flight",
            conversation_history=[],
            language_code="en",
        )

    diagnostics = client.consume_attempt_diagnostics()

    assert result.success is False
    assert called_models == ["gemma4:e4b"]
    assert len(diagnostics) == 1
    assert diagnostics[0].fallback_used is False

