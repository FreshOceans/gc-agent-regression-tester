"""Tests for dashboard PDF export."""

from datetime import datetime, timezone

from src.dashboard_metrics import build_dashboard_metrics
from src.dashboard_pdf import export_dashboard_pdf
from src.models import AttemptResult, Message, MessageRole, ScenarioResult, TestReport


def _sample_report() -> TestReport:
    attempt = AttemptResult(
        attempt_number=1,
        success=True,
        conversation=[Message(role=MessageRole.USER, content="hello")],
        explanation="ok",
        duration_seconds=1.5,
    )
    scenario = ScenarioResult(
        scenario_name="Scenario A",
        attempts=1,
        successes=1,
        failures=0,
        timeouts=0,
        skipped=0,
        success_rate=1.0,
        is_regression=False,
        attempt_results=[attempt],
    )
    return TestReport(
        suite_name="PDF Suite",
        timestamp=datetime(2026, 4, 18, 12, 0, tzinfo=timezone.utc),
        duration_seconds=1.5,
        scenario_results=[scenario],
        overall_attempts=1,
        overall_successes=1,
        overall_failures=0,
        overall_timeouts=0,
        overall_skipped=0,
        overall_success_rate=1.0,
        has_regressions=False,
        regression_threshold=0.8,
    )


def test_dashboard_pdf_export_non_empty_and_pdf_header():
    report = _sample_report()
    metrics = build_dashboard_metrics(report)
    pdf_bytes = export_dashboard_pdf(report, metrics)

    assert len(pdf_bytes) > 200
    assert pdf_bytes.startswith(b"%PDF-")
    assert b"Executive KPI Summary" in pdf_bytes
    assert b"Scenario Deep Dive" in pdf_bytes


def test_dashboard_pdf_uses_adaptive_duration_units():
    report = _sample_report()
    report.duration_seconds = 130.0
    report.scenario_results[0].attempt_results[0].duration_seconds = 131.0
    metrics = build_dashboard_metrics(report)
    pdf_bytes = export_dashboard_pdf(report, metrics)

    assert b"Run Duration: 2m 10s" in pdf_bytes
