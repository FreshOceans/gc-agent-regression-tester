"""Tests for dashboard PDF export."""

from datetime import datetime, timezone

from src.dashboard_metrics import build_dashboard_metrics
from src.dashboard_pdf import export_dashboard_pdf
from src.models import AttemptResult, Message, MessageRole, ScenarioResult, TestReport


def _sample_report(*, success: bool = True, suite_name: str = "PDF Suite") -> TestReport:
    attempts = 1
    successes = 1 if success else 0
    failures = 0 if success else 1
    attempt = AttemptResult(
        attempt_number=1,
        success=success,
        conversation=[Message(role=MessageRole.USER, content="hello")],
        explanation="ok",
        duration_seconds=1.5,
    )
    scenario = ScenarioResult(
        scenario_name="Scenario A",
        attempts=attempts,
        successes=successes,
        failures=failures,
        timeouts=0,
        skipped=0,
        success_rate=successes / attempts,
        is_regression=not success,
        attempt_results=[attempt],
    )
    return TestReport(
        suite_name=suite_name,
        timestamp=datetime(2026, 4, 18, 12, 0, tzinfo=timezone.utc),
        duration_seconds=1.5,
        scenario_results=[scenario],
        overall_attempts=attempts,
        overall_successes=successes,
        overall_failures=failures,
        overall_timeouts=0,
        overall_skipped=0,
        overall_success_rate=successes / attempts,
        has_regressions=not success,
        regression_threshold=0.8,
    )


def _scenario_heavy_report(total_scenarios: int = 30) -> TestReport:
    scenario_results = []
    for index in range(total_scenarios):
        success = index % 4 != 0
        attempts = 1
        successes = 1 if success else 0
        failures = 0 if success else 1
        attempt = AttemptResult(
            attempt_number=1,
            success=success,
            conversation=[Message(role=MessageRole.USER, content=f"hello {index}")],
            explanation="ok",
            duration_seconds=2.0 + (index * 0.1),
        )
        scenario_results.append(
            ScenarioResult(
                scenario_name=f"Scenario {index:02d}",
                attempts=attempts,
                successes=successes,
                failures=failures,
                timeouts=0,
                skipped=0,
                success_rate=successes / attempts,
                is_regression=not success,
                attempt_results=[attempt],
            )
        )

    overall_attempts = len(scenario_results)
    overall_successes = sum(s.successes for s in scenario_results)
    overall_failures = sum(s.failures for s in scenario_results)
    return TestReport(
        suite_name="Scenario Heavy",
        timestamp=datetime(2026, 4, 18, 12, 0, tzinfo=timezone.utc),
        duration_seconds=300.0,
        scenario_results=scenario_results,
        overall_attempts=overall_attempts,
        overall_successes=overall_successes,
        overall_failures=overall_failures,
        overall_timeouts=0,
        overall_skipped=0,
        overall_success_rate=overall_successes / overall_attempts,
        has_regressions=True,
        regression_threshold=0.8,
    )


def test_dashboard_pdf_export_non_empty_and_pdf_header_with_infographic_sections():
    report = _sample_report()
    metrics = build_dashboard_metrics(report)
    pdf_bytes = export_dashboard_pdf(report, metrics)

    assert len(pdf_bytes) > 200
    assert pdf_bytes.startswith(b"%PDF-")
    assert b"Regression Test Harness Dashboard Report" in pdf_bytes
    assert b"Executive KPI Summary" in pdf_bytes
    assert b"Outcome Mix" in pdf_bytes
    assert b"Tool Effectiveness" in pdf_bytes
    assert b"Scenario League Table" in pdf_bytes
    assert b"Top Failing/Timeout Scenarios" in pdf_bytes
    assert b"Unstable Scenarios" in pdf_bytes


def test_dashboard_pdf_uses_adaptive_duration_units():
    report = _sample_report()
    report.duration_seconds = 130.0
    report.scenario_results[0].attempt_results[0].duration_seconds = 131.0
    metrics = build_dashboard_metrics(report)
    pdf_bytes = export_dashboard_pdf(report, metrics)

    assert b"Run Duration: 2m 10s" in pdf_bytes


def test_dashboard_pdf_handles_no_baseline_and_empty_trend():
    report = _sample_report()
    metrics = build_dashboard_metrics(report, trend_entries=[])
    pdf_bytes = export_dashboard_pdf(report, metrics)

    assert b"No previous same-suite baseline found." in pdf_bytes
    assert b"No trend history available." in pdf_bytes


def test_dashboard_pdf_renders_baseline_compare_and_trend_when_available():
    current = _sample_report(success=True, suite_name="Compare Suite")
    baseline = _sample_report(success=False, suite_name="Compare Suite")

    trend_entries = [
        {
            "run_id": "run-old",
            "timestamp": "2026-04-17T12:00:00+00:00",
            "overall_attempts": 1,
            "overall_success_rate": 0.0,
            "overall_failures": 1,
            "overall_timeouts": 0,
            "overall_skipped": 0,
            "duration_seconds": 2.0,
        },
        {
            "run_id": "run-current",
            "timestamp": "2026-04-18T12:00:00+00:00",
            "overall_attempts": 1,
            "overall_success_rate": 1.0,
            "overall_failures": 0,
            "overall_timeouts": 0,
            "overall_skipped": 0,
            "duration_seconds": 1.0,
        },
    ]

    metrics = build_dashboard_metrics(
        current,
        baseline_report=baseline,
        trend_entries=trend_entries,
        current_run_id="run-current",
    )
    pdf_bytes = export_dashboard_pdf(current, metrics)

    assert b"Baseline timestamp" in pdf_bytes
    assert b"Baseline suite" in pdf_bytes
    assert b"Recent Same-Suite Trend" in pdf_bytes
    assert b"No previous same-suite baseline found." not in pdf_bytes


def test_dashboard_pdf_renders_summary_only_baseline_storage_note():
    current = _sample_report(success=True, suite_name="Compare Suite")
    baseline_summary = {
        "suite_name": "Compare Suite",
        "timestamp": "2026-04-17T12:00:00+00:00",
        "storage_type": "summary_only",
        "summary": {
            "kpis": {
                "attempts": 1,
                "successes": 0,
                "failures": 1,
                "timeouts": 0,
                "skipped": 0,
                "success_rate": 0.0,
            },
            "duration": {
                "average_seconds": 2.0,
                "median_seconds": 2.0,
                "p95_seconds": 2.0,
            },
            "rates": {
                "failure_rate": 1.0,
                "timeout_rate": 0.0,
                "skipped_rate": 0.0,
            },
        },
    }

    metrics = build_dashboard_metrics(current, baseline_summary=baseline_summary)
    pdf_bytes = export_dashboard_pdf(current, metrics)

    assert b"Baseline storage: summary_only" in pdf_bytes


def test_dashboard_pdf_scenario_heavy_report_still_exports_valid_pdf():
    report = _scenario_heavy_report(total_scenarios=30)
    metrics = build_dashboard_metrics(report)
    pdf_bytes = export_dashboard_pdf(report, metrics)

    assert pdf_bytes.startswith(b"%PDF-")
    assert len(pdf_bytes) > 400
    assert b"Scenario League Table" in pdf_bytes


def test_dashboard_pdf_localizes_labels_for_selected_language():
    report = _sample_report()
    metrics = build_dashboard_metrics(report)
    pdf_bytes = export_dashboard_pdf(report, metrics, language_code="es")

    assert pdf_bytes.startswith(b"%PDF-")
    assert b"Dashboard Ejecutivo" in pdf_bytes
    assert b"Resumen Ejecutivo de KPIs" in pdf_bytes


def test_dashboard_pdf_includes_journey_taxonomy_section_when_enabled():
    report = _sample_report()
    report.scenario_results[0].expected_intent = "speak_to_agent"
    report.scenario_results[0].attempt_results[0].detected_intent = "speak_to_agent"
    report.scenario_results[0].attempt_results[0].conversation = [
        Message(role=MessageRole.AGENT, content="I can transfer to live agent now."),
    ]
    metrics = build_dashboard_metrics(
        report,
        journey_dashboard_enabled=True,
        journey_active_view="live_agent_transfer",
    )
    pdf_bytes = export_dashboard_pdf(
        report,
        metrics,
        selected_journey_view="live_agent_transfer",
    )

    assert b"Journey Taxonomy" in pdf_bytes
    assert b"Agent Request - Successful Transfer To Agent" in pdf_bytes
