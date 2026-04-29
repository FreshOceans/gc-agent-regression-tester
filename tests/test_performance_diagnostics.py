"""Tests for run-level performance diagnostics aggregation."""

from datetime import datetime, timezone

import pytest

from src.models import (
    AttemptResult,
    JudgeDiagnosticEntry,
    Message,
    MessageRole,
    ScenarioResult,
    TestReport,
)
from src.performance_diagnostics import build_performance_diagnostics


def _report() -> TestReport:
    attempts = [
        AttemptResult(
            attempt_number=1,
            success=True,
            conversation=[Message(role=MessageRole.USER, content="hello")],
            explanation="ok",
            duration_seconds=1.0,
            warmup_stage_durations_ms={"connect": 10.0, "response_wait": 100.0},
            step_log=[{"stage": "judge_evaluation", "duration_ms": 50.0}],
            judge_diagnostics=[
                JudgeDiagnosticEntry(
                    operation_name="evaluate_goal",
                    primary_model="gemma4:e4b",
                    duration_ms=40.0,
                )
            ],
        ),
        AttemptResult(
            attempt_number=2,
            success=False,
            timed_out=True,
            conversation=[Message(role=MessageRole.USER, content="hello")],
            explanation="timeout",
            duration_seconds=3.0,
            warmup_stage_durations_ms={"connect": 30.0, "response_wait": 300.0},
            judge_diagnostics=[
                JudgeDiagnosticEntry(
                    operation_name="evaluate_goal",
                    primary_model="gemma4:e4b",
                    duration_ms=80.0,
                )
            ],
        ),
    ]
    scenario = ScenarioResult(
        scenario_name="Scenario A",
        attempts=2,
        successes=1,
        failures=0,
        timeouts=1,
        success_rate=0.5,
        is_regression=True,
        attempt_results=attempts,
    )
    return TestReport(
        suite_name="Perf Suite",
        timestamp=datetime.now(timezone.utc),
        duration_seconds=4.0,
        scenario_results=[scenario],
        overall_attempts=2,
        overall_successes=1,
        overall_failures=0,
        overall_timeouts=1,
        overall_success_rate=0.5,
        has_regressions=True,
        regression_threshold=0.8,
    )


def test_build_performance_diagnostics_aggregates_attempt_stage_and_judge_timings():
    diagnostics = build_performance_diagnostics(
        _report(),
        run_type="test_run",
        planned_attempts=2,
        worker_count=2,
        pacing_seconds=5.0,
        queue_wait_ms=[5.0, 10.0],
    )

    stage_names = {summary.stage for summary in diagnostics.stage_summaries}
    judge_names = {summary.stage for summary in diagnostics.judge_operation_summaries}

    assert diagnostics.run_type == "test_run"
    assert diagnostics.completed_attempts == 2
    assert diagnostics.attempts_per_second == pytest.approx(0.5)
    assert diagnostics.timeout_error_rate == pytest.approx(0.5)
    assert diagnostics.worker_count == 2
    assert "attempt_total" in stage_names
    assert "attempt_queue_wait" in stage_names
    assert "web_messaging_response_wait" in stage_names
    assert "evaluate_goal" in judge_names
    assert diagnostics.slowest_stages[0].stage == "attempt_total"
