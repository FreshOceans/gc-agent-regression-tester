"""Unit tests for analytics journey runner helpers."""

from datetime import datetime, timezone

import pytest

from src.analytics_journey_runner import (
    AnalyticsJourneyRunRequest,
    AnalyticsJourneyRunner,
    evaluate_gate,
    infer_auth_evidence,
    infer_transfer_evidence,
    load_analytics_policy_map,
    resolve_policy_for_category,
)
from src.models import AppConfig, JourneyValidationResult, Message, MessageRole
from src.progress import ProgressEmitter


def test_load_analytics_policy_map_merges_defaults_with_overrides(tmp_path):
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        '{"flight_cancel":{"auth_behavior":"required","transfer_behavior":"forbidden"}}',
        encoding="utf-8",
    )

    policy = load_analytics_policy_map(policy_json="", policy_file=str(policy_path))

    assert "default" in policy
    assert policy["flight_cancel"]["auth_behavior"] == "required"
    assert policy["flight_cancel"]["transfer_behavior"] == "forbidden"


def test_resolve_policy_for_category_falls_back_to_default():
    policy = {
        "default": {"auth_behavior": "optional", "transfer_behavior": "optional"},
        "speak_to_agent": {"auth_behavior": "optional", "transfer_behavior": "required"},
    }

    key, resolved = resolve_policy_for_category("unknown_intent", policy)
    assert key == "default"
    assert resolved["transfer_behavior"] == "optional"


def test_evaluate_gate_required_and_optional_behaviors():
    assert evaluate_gate(expected_behavior="required", observed=True) == (True, False)
    assert evaluate_gate(expected_behavior="required", observed=False) == (False, False)
    assert evaluate_gate(expected_behavior="required", observed=None) == (None, True)
    assert evaluate_gate(expected_behavior="optional", observed=None) == (True, False)
    assert evaluate_gate(expected_behavior="forbidden", observed=False) == (True, False)


def test_infer_auth_evidence_from_transcript_tokens():
    messages = [
        Message(
            role=MessageRole.AGENT,
            content="Authentication successful. You're now verified.",
            timestamp=datetime.now(timezone.utc),
        )
    ]

    observed, notes = infer_auth_evidence(messages, raw_payload=None)
    assert observed is True
    assert any("transcript" in note for note in notes)


def test_infer_transfer_evidence_uses_containment_hint_first():
    messages = [Message(role=MessageRole.AGENT, content="Let's continue here")]

    observed_true, notes_true = infer_transfer_evidence(
        messages,
        raw_payload=None,
        contained_hint=False,
    )
    observed_false, notes_false = infer_transfer_evidence(
        messages,
        raw_payload=None,
        contained_hint=True,
    )

    assert observed_true is True
    assert observed_false is False
    assert notes_true
    assert notes_false


@pytest.mark.asyncio
async def test_runner_populates_analytics_run_diagnostics(monkeypatch):
    class FakeAnalyticsClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fetch_conversation_units(
            self,
            *,
            bot_flow_id,
            interval,
            page_size,
            max_conversations,
            divisions=None,
            language_filter=None,
            extra_params=None,
            observer=None,
        ):
            if observer is not None:
                observer({"event": "page_fetch_started", "page_number": 1})
                observer(
                    {
                        "event": "request_retry",
                        "page_number": 1,
                        "attempt": 1,
                        "max_attempts": 3,
                        "status_code": 429,
                        "backoff_seconds": 1.0,
                        "duration_ms": 10.0,
                    }
                )
                observer(
                    {
                        "event": "page_fetch_completed",
                        "page_number": 1,
                        "rows_count": 2,
                        "new_unique_conversations": 1,
                        "total_unique_conversations": 1,
                        "duration_ms": 12.5,
                    }
                )
            return {
                "conversations": [
                    {
                        "conversation_id": "11111111-1111-1111-1111-111111111111",
                        "rows": [],
                    }
                ],
                "page_payloads": [{"results": []}],
                "page_count": 1,
            }

    class FakeTranscriptClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fetch_conversation_payload(self, conversation_id):
            return {
                "participants": [
                    {"purpose": "customer"},
                    {"purpose": "agent", "userId": "agent-1"},
                ]
            }

        def normalize_conversation_payload(self, payload, *, conversation_id):
            return {
                "messages": [
                    {
                        "role": "user",
                        "text": "I want to speak to an agent",
                        "timestamp": "2026-04-19T00:00:00Z",
                    },
                    {
                        "role": "agent",
                        "text": "Sure, I can transfer you now.",
                        "timestamp": "2026-04-19T00:00:05Z",
                    },
                ]
            }

    class FakeJudge:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def warm_up(self, **kwargs):
            return None

        def classify_primary_category(self, *, first_message, categories, language_code):
            return {
                "category": "speak_to_agent",
                "confidence": 0.92,
                "explanation": "matched",
            }

        def evaluate_journey(
            self,
            *,
            persona,
            goal,
            expected_category,
            path_rubric,
            category_rubric,
            conversation_history,
            language_code,
            known_contained,
        ):
            return JourneyValidationResult(
                category_match=True,
                fulfilled=True,
                path_correct=True,
                contained=False,
                expected_category=expected_category,
                actual_category=expected_category,
                containment_source="metadata",
                explanation="ok",
            )

    monkeypatch.setattr(
        "src.analytics_journey_runner.GenesysAnalyticsJourneyClient",
        FakeAnalyticsClient,
    )
    monkeypatch.setattr(
        "src.analytics_journey_runner.GenesysTranscriptImportClient",
        FakeTranscriptClient,
    )
    monkeypatch.setattr("src.analytics_journey_runner.JudgeLLMClient", FakeJudge)

    config = AppConfig(
        gc_region="cac1.pure.cloud",
        gc_client_id="client-id",
        gc_client_secret="client-secret",
        ollama_model="gemma3:12b",
        judge_warmup_enabled=True,
    )
    runner = AnalyticsJourneyRunner(
        config=config,
        progress_emitter=ProgressEmitter(),
        stop_event=None,
        artifact_store=None,
    )
    request = AnalyticsJourneyRunRequest(
        bot_flow_id="flow-123",
        interval="2026-04-19T00:00:00.000Z/2026-04-20T00:00:00.000Z",
        page_size=50,
        max_conversations=1,
    )

    report = await runner.run(request)

    assert report.analytics_run_diagnostics is not None
    diagnostics = report.analytics_run_diagnostics
    assert diagnostics.request.bot_flow_id == "flow-123"
    assert diagnostics.summary.pages_fetched == 1
    assert diagnostics.summary.rows_scanned == 2
    assert diagnostics.summary.retry_count == 1
    assert diagnostics.summary.http_429_count == 1
    assert diagnostics.summary.evaluated == 1
    assert diagnostics.summary.passed == 1
    stages = [entry.stage for entry in diagnostics.timeline]
    assert "analytics_fetch_start" in stages
    assert "analytics_page_complete" in stages
    assert "conversation_eval_complete" in stages
    attempt = report.scenario_results[0].attempt_results[0]
    assert attempt.step_log
    assert any(entry.get("stage") == "conversation_enrichment_start" for entry in attempt.step_log)
    assert any(entry.get("stage") == "conversation_journey_evaluation_complete" for entry in attempt.step_log)
