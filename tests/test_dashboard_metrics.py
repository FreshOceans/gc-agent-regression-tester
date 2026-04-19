"""Tests for dashboard metric aggregation."""

from datetime import datetime, timezone

from src.dashboard_metrics import build_dashboard_metrics, summarize_entry_for_compare
from src.models import AttemptResult, Message, MessageRole, ScenarioResult, TestReport


def _attempt(number: int, *, success: bool, duration: float, timed_out: bool = False, skipped: bool = False) -> AttemptResult:
    return AttemptResult(
        attempt_number=number,
        success=success,
        conversation=[Message(role=MessageRole.USER, content="hello")],
        explanation="ok",
        duration_seconds=duration,
        timed_out=timed_out,
        skipped=skipped,
    )


def _report(
    *,
    suite_name: str,
    timestamp: datetime,
    attempts: list[AttemptResult],
) -> TestReport:
    successes = sum(1 for a in attempts if a.success)
    timeouts = sum(1 for a in attempts if a.timed_out)
    skipped = sum(1 for a in attempts if a.skipped)
    failures = len(attempts) - successes - timeouts - skipped
    scenario = ScenarioResult(
        scenario_name="Scenario A",
        attempts=len(attempts),
        successes=successes,
        failures=failures,
        timeouts=timeouts,
        skipped=skipped,
        success_rate=successes / len(attempts),
        is_regression=(successes / len(attempts)) < 0.8,
        attempt_results=attempts,
    )
    return TestReport(
        suite_name=suite_name,
        timestamp=timestamp,
        duration_seconds=sum(a.duration_seconds or 0 for a in attempts),
        scenario_results=[scenario],
        overall_attempts=len(attempts),
        overall_successes=successes,
        overall_failures=failures,
        overall_timeouts=timeouts,
        overall_skipped=skipped,
        overall_success_rate=successes / len(attempts),
        has_regressions=(successes / len(attempts)) < 0.8,
        regression_threshold=0.8,
    )


def test_build_dashboard_metrics_core_values():
    report = _report(
        suite_name="Suite A",
        timestamp=datetime(2026, 4, 18, 12, 0, tzinfo=timezone.utc),
        attempts=[
            _attempt(1, success=True, duration=1.0),
            _attempt(2, success=False, duration=3.0),
            _attempt(3, success=False, duration=5.0, timed_out=True),
            _attempt(4, success=False, duration=7.0, skipped=True),
        ],
    )

    metrics = build_dashboard_metrics(report)

    assert metrics["kpis"]["attempts"] == 4
    assert metrics["kpis"]["successes"] == 1
    assert metrics["kpis"]["failures"] == 1
    assert metrics["kpis"]["timeouts"] == 1
    assert metrics["kpis"]["skipped"] == 1
    assert metrics["duration"]["average_seconds"] == 4.0
    assert metrics["duration"]["median_seconds"] == 4.0
    assert metrics["duration"]["p95_seconds"] > 6.0
    assert len(metrics["outcome_mix"]) == 4
    assert metrics["tool_effectiveness"]["validated_attempts"] == 0
    assert metrics["scenario_tool_health"] == []


def test_build_dashboard_metrics_compare_and_trend():
    current = _report(
        suite_name="Suite A",
        timestamp=datetime(2026, 4, 18, 13, 0, tzinfo=timezone.utc),
        attempts=[
            _attempt(1, success=True, duration=1.0),
            _attempt(2, success=True, duration=1.2),
        ],
    )
    baseline = _report(
        suite_name="Suite A",
        timestamp=datetime(2026, 4, 18, 12, 0, tzinfo=timezone.utc),
        attempts=[
            _attempt(1, success=False, duration=2.0),
            _attempt(2, success=True, duration=3.0),
        ],
    )
    trend_entries = [
        {
            "run_id": "a",
            "timestamp": "2026-04-18T12:00:00+00:00",
            "overall_attempts": 2,
            "overall_success_rate": 0.5,
            "overall_failures": 1,
            "overall_timeouts": 0,
            "overall_skipped": 0,
            "duration_seconds": 5.0,
        },
        {
            "run_id": "b",
            "timestamp": "2026-04-18T13:00:00+00:00",
            "overall_attempts": 2,
            "overall_success_rate": 1.0,
            "overall_failures": 0,
            "overall_timeouts": 0,
            "overall_skipped": 0,
            "duration_seconds": 2.2,
        },
    ]

    metrics = build_dashboard_metrics(
        current,
        baseline_report=baseline,
        trend_entries=trend_entries,
        current_run_id="b",
    )

    assert metrics["compare"] is not None
    assert metrics["compare"]["deltas"]["success_rate"]["delta"] > 0
    assert "tool_loose_pass_rate" in metrics["compare"]["deltas"]
    assert len(metrics["trend"]) == 2
    assert metrics["trend"][-1]["is_current"] is True


def test_build_dashboard_metrics_tool_effectiveness_summary():
    report = _report(
        suite_name="Suite A",
        timestamp=datetime(2026, 4, 18, 12, 0, tzinfo=timezone.utc),
        attempts=[
            _attempt(1, success=True, duration=1.0),
            _attempt(2, success=True, duration=1.2),
        ],
    )
    scenario = report.scenario_results[0]
    scenario.tool_validated_attempts = 2
    scenario.tool_loose_passes = 2
    scenario.tool_strict_passes = 1
    scenario.tool_missing_signal_count = 0
    scenario.tool_order_mismatch_count = 1
    scenario.tool_loose_pass_rate = 1.0
    scenario.tool_strict_pass_rate = 0.5
    report.overall_tool_validated_attempts = 2
    report.overall_tool_loose_passes = 2
    report.overall_tool_strict_passes = 1
    report.overall_tool_missing_signal_count = 0
    report.overall_tool_order_mismatch_count = 1
    report.overall_tool_loose_pass_rate = 1.0
    report.overall_tool_strict_pass_rate = 0.5

    metrics = build_dashboard_metrics(report)
    assert metrics["tool_effectiveness"]["validated_attempts"] == 2
    assert metrics["tool_effectiveness"]["strict_pass_rate"] == 0.5
    assert len(metrics["scenario_tool_health"]) == 1
    assert metrics["scenario_tool_health"][0]["name"] == "Scenario A"


def test_build_dashboard_metrics_compare_from_summary_only_baseline():
    current = _report(
        suite_name="Suite A",
        timestamp=datetime(2026, 4, 18, 13, 0, tzinfo=timezone.utc),
        attempts=[
            _attempt(1, success=True, duration=1.0),
            _attempt(2, success=True, duration=1.1),
        ],
    )
    baseline_entry = {
        "suite_name": "Suite A",
        "timestamp": "2026-04-18T12:00:00+00:00",
        "storage_type": "summary_only",
        "overall_attempts": 2,
        "overall_successes": 0,
        "overall_failures": 2,
        "overall_timeouts": 0,
        "overall_skipped": 0,
        "overall_success_rate": 0.0,
        "duration_seconds": 5.0,
        "scenario_summaries": [
            {
                "name": "Scenario A",
                "attempts": 2,
                "successes": 0,
                "failures": 2,
                "timeouts": 0,
                "skipped": 0,
                "success_rate": 0.0,
                "is_regression": True,
            }
        ],
    }

    baseline_summary = summarize_entry_for_compare(baseline_entry)
    metrics = build_dashboard_metrics(current, baseline_summary=baseline_summary)

    assert metrics["compare"] is not None
    assert metrics["compare"]["baseline_storage_type"] == "summary_only"
    assert metrics["compare"]["deltas"]["success_rate"]["delta"] > 0


def test_build_dashboard_metrics_flakiness_identifies_unstable_scenarios():
    current = _report(
        suite_name="Suite A",
        timestamp=datetime(2026, 4, 18, 13, 0, tzinfo=timezone.utc),
        attempts=[_attempt(1, success=True, duration=1.0)],
    )
    trend_entries = [
        {
            "run_id": "run-1",
            "suite_name": "Suite A",
            "timestamp": "2026-04-18T10:00:00+00:00",
            "overall_attempts": 2,
            "overall_success_rate": 1.0,
            "overall_failures": 0,
            "overall_timeouts": 0,
            "overall_skipped": 0,
            "duration_seconds": 2.0,
            "scenario_summaries": [
                {"name": "Scenario A", "success_rate": 1.0},
                {"name": "Scenario B", "success_rate": 1.0},
            ],
        },
        {
            "run_id": "run-2",
            "suite_name": "Suite A",
            "timestamp": "2026-04-18T11:00:00+00:00",
            "overall_attempts": 2,
            "overall_success_rate": 0.5,
            "overall_failures": 1,
            "overall_timeouts": 0,
            "overall_skipped": 0,
            "duration_seconds": 3.0,
            "scenario_summaries": [
                {"name": "Scenario A", "success_rate": 0.0},
                {"name": "Scenario B", "success_rate": 1.0},
            ],
        },
        {
            "run_id": "run-3",
            "suite_name": "Suite A",
            "timestamp": "2026-04-18T12:00:00+00:00",
            "overall_attempts": 2,
            "overall_success_rate": 1.0,
            "overall_failures": 0,
            "overall_timeouts": 0,
            "overall_skipped": 0,
            "duration_seconds": 2.2,
            "scenario_summaries": [
                {"name": "Scenario A", "success_rate": 1.0},
                {"name": "Scenario B", "success_rate": 1.0},
            ],
        },
    ]

    metrics = build_dashboard_metrics(
        current,
        trend_entries=trend_entries,
        current_run_id="run-3",
    )

    flakiness = metrics["flakiness"]
    assert flakiness["evaluated_runs"] == 3
    assert flakiness["scenarios_evaluated"] == 2
    assert flakiness["unstable_scenarios"]
    assert flakiness["unstable_scenarios"][0]["name"] == "Scenario A"
