"""Unit tests for Phase 12 journey taxonomy helpers."""

from datetime import datetime, timezone

from src.journey_taxonomy import (
    TOTAL_CALLS_LABEL,
    build_journey_taxonomy_rollups,
    classify_attempt_taxonomy,
    normalize_journey_view,
)
from src.models import AttemptResult, Message, MessageRole, ScenarioResult, TestReport


def _attempt(*, content: str, success: bool = True, timed_out: bool = False) -> AttemptResult:
    return AttemptResult(
        attempt_number=1,
        success=success,
        timed_out=timed_out,
        conversation=[Message(role=MessageRole.AGENT, content=content)],
        explanation="ok",
        detected_intent="speak_to_agent" if success else None,
    )


def _report(attempt: AttemptResult) -> TestReport:
    scenario = ScenarioResult(
        scenario_name="Scenario A",
        expected_intent="speak_to_agent",
        attempts=1,
        successes=1 if attempt.success else 0,
        failures=0 if attempt.success else 1,
        timeouts=1 if attempt.timed_out else 0,
        skipped=0,
        success_rate=1.0 if attempt.success else 0.0,
        is_regression=not attempt.success,
        attempt_results=[attempt],
    )
    return TestReport(
        suite_name="Taxonomy Suite",
        timestamp=datetime(2026, 4, 20, 12, 0, tzinfo=timezone.utc),
        duration_seconds=1.0,
        scenario_results=[scenario],
        overall_attempts=1,
        overall_successes=1 if attempt.success else 0,
        overall_failures=0 if attempt.success else 1,
        overall_timeouts=1 if attempt.timed_out else 0,
        overall_skipped=0,
        overall_success_rate=1.0 if attempt.success else 0.0,
        has_regressions=not attempt.success,
        regression_threshold=0.8,
    )


def test_normalize_journey_view_fallbacks_to_overview():
    assert normalize_journey_view("live_agent_transfer") == "live_agent_transfer"
    assert normalize_journey_view("unknown") == "overview"


def test_classify_attempt_taxonomy_honors_override_first():
    attempt = _attempt(content="generic text")
    label = classify_attempt_taxonomy(
        expected_intent="speak_to_agent",
        attempt=attempt,
        overrides={"generic text": "Wrong Number/Marketing"},
    )
    assert label == "Wrong Number/Marketing"


def test_classify_attempt_taxonomy_detects_agent_transfer_flow():
    attempt = _attempt(content="I can transfer to live agent now.")
    label = classify_attempt_taxonomy(
        expected_intent="speak_to_agent",
        attempt=attempt,
        overrides=None,
    )
    assert label == "Agent Request - Successful Transfer To Agent"


def test_build_journey_taxonomy_rollups_includes_total_and_delta():
    current = _report(_attempt(content="I can transfer to live agent now."))
    baseline = _report(_attempt(content="Guest hung up during call", success=False))

    rollup = build_journey_taxonomy_rollups(
        current,
        baseline_report=baseline,
        active_view="live_agent_transfer",
    )

    labels = {row["label"]: row for row in rollup["labels"]}
    assert labels[TOTAL_CALLS_LABEL]["count"] == 1
    assert labels[TOTAL_CALLS_LABEL]["delta"] == 0
    assert labels["Agent Request - Successful Transfer To Agent"]["count"] == 1
    # baseline had no transfer classification, so delta is +1
    assert labels["Agent Request - Successful Transfer To Agent"]["delta"] == 1

    active_view = next(v for v in rollup["views"] if v["key"] == "live_agent_transfer")
    assert active_view["total"] == 1
