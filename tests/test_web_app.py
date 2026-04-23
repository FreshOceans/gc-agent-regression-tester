"""Integration tests for web app result export routes."""

from datetime import datetime, timezone
import io
import re

from src.models import (
    AnalyticsJourneyResult,
    AnalyticsRunDiagnostics,
    AnalyticsRunDiagnosticsRequest,
    AnalyticsRunDiagnosticsSummary,
    AnalyticsRunDiagnosticsTimelineEntry,
    AppConfig,
    AttemptResult,
    Message,
    MessageRole,
    ScenarioResult,
    TestReport,
    TestScenario,
    TestSuite,
    ProgressEventType,
)
from src.run_history import RunHistoryStore
from src.progress import ProgressEmitter, ProgressEvent
from src.web_app import ActiveRunControl, create_app


class _FakeWebJudgeClient:
    def __init__(self, *, verify_error=None, classify_result=None):
        self.verify_error = verify_error
        self.classify_result = classify_result or {
            "category": "flight_cancel",
            "confidence": 0.9,
            "explanation": "category match",
        }

    def verify_connection(self):
        if self.verify_error is not None:
            raise self.verify_error
        return None

    def classify_primary_category(self, *args, **kwargs):
        return dict(self.classify_result)

    def warm_up(self, *args, **kwargs):
        return "OK"


def _patch_web_judge_builder(monkeypatch, *, verify_error=None, classify_result=None):
    judge = _FakeWebJudgeClient(
        verify_error=verify_error,
        classify_result=classify_result,
    )
    monkeypatch.setattr(
        "src.web_app.build_judge_execution_client",
        lambda *args, **kwargs: judge,
    )
    return judge


def _sample_report() -> TestReport:
    attempt = AttemptResult(
        attempt_number=1,
        success=True,
        conversation=[Message(role=MessageRole.USER, content="hello")],
        explanation="ok",
        duration_seconds=1.0,
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
        suite_name="Web Suite",
        timestamp=datetime(2026, 4, 18, 12, 0, tzinfo=timezone.utc),
        duration_seconds=1.0,
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


def _large_report(total_attempts: int = 25) -> TestReport:
    attempts = [
        AttemptResult(
            attempt_number=index + 1,
            success=True,
            conversation=[Message(role=MessageRole.USER, content=f"hello {index + 1}")],
            explanation="ok",
            duration_seconds=3.0 + index,
        )
        for index in range(total_attempts)
    ]
    scenario = ScenarioResult(
        scenario_name="Scenario Large",
        attempts=total_attempts,
        successes=total_attempts,
        failures=0,
        timeouts=0,
        skipped=0,
        success_rate=1.0,
        is_regression=False,
        attempt_results=attempts,
    )
    return TestReport(
        suite_name="Large Suite",
        timestamp=datetime(2026, 4, 18, 12, 0, tzinfo=timezone.utc),
        duration_seconds=180.0,
        scenario_results=[scenario],
        overall_attempts=total_attempts,
        overall_successes=total_attempts,
        overall_failures=0,
        overall_timeouts=0,
        overall_skipped=0,
        overall_success_rate=1.0,
        has_regressions=False,
        regression_threshold=0.8,
    )


def _suite_for_web() -> TestSuite:
    return TestSuite(
        name="Web Suite",
        scenarios=[
            TestScenario(
                name="Scenario A",
                persona="Traveler",
                goal="Get help",
                first_message="hello",
                attempts=1,
            )
        ],
    )


def _latest_report_with_failed_bucket() -> TestReport:
    success_attempt = AttemptResult(
        attempt_number=1,
        success=True,
        conversation=[Message(role=MessageRole.USER, content="hello")],
        explanation="ok",
        duration_seconds=1.0,
    )
    failure_attempt = AttemptResult(
        attempt_number=1,
        success=False,
        conversation=[Message(role=MessageRole.USER, content="help")],
        explanation="failed",
        duration_seconds=1.2,
    )
    scenario_ok = ScenarioResult(
        scenario_name="Scenario A",
        attempts=1,
        successes=1,
        failures=0,
        timeouts=0,
        skipped=0,
        success_rate=1.0,
        is_regression=False,
        attempt_results=[success_attempt],
    )
    scenario_failed = ScenarioResult(
        scenario_name="Scenario B",
        attempts=1,
        successes=0,
        failures=1,
        timeouts=0,
        skipped=0,
        success_rate=0.0,
        is_regression=True,
        attempt_results=[failure_attempt],
    )
    return TestReport(
        suite_name="Web Suite",
        timestamp=datetime(2026, 4, 18, 12, 0, tzinfo=timezone.utc),
        duration_seconds=2.2,
        scenario_results=[scenario_ok, scenario_failed],
        overall_attempts=2,
        overall_successes=1,
        overall_failures=1,
        overall_timeouts=0,
        overall_skipped=0,
        overall_success_rate=0.5,
        has_regressions=True,
        regression_threshold=0.8,
    )


def _intent_grouped_report() -> TestReport:
    grouped_attempt = AttemptResult(
        attempt_number=1,
        success=True,
        conversation=[Message(role=MessageRole.USER, content="cancel")],
        explanation="ok",
        duration_seconds=1.0,
    )
    fallback_attempt = AttemptResult(
        attempt_number=1,
        success=False,
        timed_out=True,
        conversation=[Message(role=MessageRole.USER, content="help")],
        explanation="timeout",
        duration_seconds=1.5,
    )
    grouped_scenario = ScenarioResult(
        scenario_name="Scenario Intent",
        expected_intent="flight_cancel",
        attempts=1,
        successes=1,
        failures=0,
        timeouts=0,
        skipped=0,
        success_rate=1.0,
        is_regression=False,
        attempt_results=[grouped_attempt],
    )
    fallback_scenario = ScenarioResult(
        scenario_name="Scenario Journey",
        expected_intent=None,
        attempts=1,
        successes=0,
        failures=0,
        timeouts=1,
        skipped=0,
        success_rate=0.0,
        is_regression=True,
        attempt_results=[fallback_attempt],
    )
    return TestReport(
        suite_name="Grouped Suite",
        timestamp=datetime(2026, 4, 18, 12, 0, tzinfo=timezone.utc),
        duration_seconds=2.5,
        scenario_results=[grouped_scenario, fallback_scenario],
        overall_attempts=2,
        overall_successes=1,
        overall_failures=0,
        overall_timeouts=1,
        overall_skipped=0,
        overall_success_rate=0.5,
        has_regressions=True,
        regression_threshold=0.8,
    )


def _journey_dashboard_report() -> TestReport:
    attempt = AttemptResult(
        attempt_number=1,
        success=True,
        conversation=[
            Message(role=MessageRole.AGENT, content="I can transfer to live agent now."),
            Message(role=MessageRole.USER, content="yes"),
        ],
        explanation="Journey complete",
        detected_intent="speak_to_agent",
        duration_seconds=2.0,
    )
    scenario = ScenarioResult(
        scenario_name="Scenario Transfer",
        expected_intent="speak_to_agent",
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
        suite_name="Journey Suite",
        timestamp=datetime(2026, 4, 18, 12, 0, tzinfo=timezone.utc),
        duration_seconds=2.0,
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


def _analytics_diagnostics_report() -> TestReport:
    attempt = AttemptResult(
        attempt_number=1,
        success=True,
        conversation=[Message(role=MessageRole.USER, content="hello")],
        explanation="ok",
        duration_seconds=1.0,
    )
    scenario = ScenarioResult(
        scenario_name="Scenario Analytics",
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
        suite_name="Analytics Diagnostics Suite",
        timestamp=datetime(2026, 4, 18, 12, 0, tzinfo=timezone.utc),
        duration_seconds=5.0,
        scenario_results=[scenario],
        overall_attempts=1,
        overall_successes=1,
        overall_failures=0,
        overall_timeouts=0,
        overall_skipped=0,
        overall_success_rate=1.0,
        has_regressions=False,
        regression_threshold=0.8,
        analytics_run_diagnostics=AnalyticsRunDiagnostics(
            request=AnalyticsRunDiagnosticsRequest(
                bot_flow_id="flow-abc",
                interval="2026-04-19T00:00:00.000Z/2026-04-20T00:00:00.000Z",
                page_size=50,
                max_conversations=100,
                divisions_count=2,
                language_filter="es",
                extra_query_param_keys=["mediaType"],
            ),
            summary=AnalyticsRunDiagnosticsSummary(
                pages_fetched=2,
                rows_scanned=120,
                unique_conversations=100,
                evaluated=100,
                passed=88,
                failed=8,
                skipped=4,
                retry_count=2,
                http_429_count=1,
                http_5xx_count=1,
                fetch_duration_seconds=2.5,
                evaluation_duration_seconds=11.2,
                total_duration_seconds=13.7,
            ),
            timeline=[
                AnalyticsRunDiagnosticsTimelineEntry(
                    timestamp=datetime(2026, 4, 18, 12, 0, tzinfo=timezone.utc),
                    elapsed_seconds=0.1,
                    stage="run_init",
                    message="Analytics journey run initialized",
                ),
                AnalyticsRunDiagnosticsTimelineEntry(
                    timestamp=datetime(2026, 4, 18, 12, 1, tzinfo=timezone.utc),
                    elapsed_seconds=1.1,
                    stage="analytics_page_complete",
                    message="Fetched analytics page 1",
                    page_number=1,
                    duration_ms=12.5,
                ),
            ],
        ),
    )


def test_results_export_dashboard_pdf_route():
    app = create_app()
    app.config["TESTING"] = True
    app.config["latest_report"] = _sample_report()

    client = app.test_client()
    response = client.get("/results/export?format=dashboard_pdf")

    assert response.status_code == 200
    assert response.mimetype == "application/pdf"
    assert response.headers.get("Content-Disposition", "").endswith("dashboard-report.pdf")
    assert response.data.startswith(b"%PDF-")


def test_results_export_existing_formats_unchanged():
    app = create_app()
    app.config["TESTING"] = True
    app.config["latest_report"] = _sample_report()

    client = app.test_client()

    csv_response = client.get("/results/export?format=csv")
    assert csv_response.status_code == 200
    assert csv_response.mimetype == "text/csv"

    failures_csv_response = client.get("/results/export?format=failures_csv")
    assert failures_csv_response.status_code == 200
    assert failures_csv_response.mimetype == "text/csv"
    assert failures_csv_response.headers.get("Content-Disposition", "").endswith(
        "report-failures.csv"
    )

    json_response = client.get("/results/export?format=json")
    assert json_response.status_code == 200
    assert json_response.mimetype == "application/json"

    bundle_response = client.get("/results/export?format=bundle")
    assert bundle_response.status_code == 200
    assert bundle_response.mimetype == "application/zip"


def test_results_page_shows_compare_fallback_when_no_baseline(tmp_path, monkeypatch):
    monkeypatch.setenv("GC_TESTER_HISTORY_DIR", str(tmp_path / "history"))
    app = create_app()
    app.config["TESTING"] = True
    app.config["latest_report"] = _sample_report()
    app.config["latest_run_history_entry"] = None

    client = app.test_client()
    response = client.get("/results")

    text = response.get_data(as_text=True)
    assert response.status_code == 200
    assert "Current vs Baseline" in text
    assert "No previous same-suite run found yet." in text


def test_results_page_renders_operations_bar_and_sectioned_shell():
    app = create_app()
    app.config["TESTING"] = True
    app.config["latest_report"] = _sample_report()
    app.config["last_run_config"] = AppConfig(gc_region="example.com", gc_deployment_id="deploy-id")
    app.config["last_run_suite"] = _suite_for_web()

    client = app.test_client()
    response = client.get("/results")
    text = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Metrics Legend &amp; Definitions" in text
    assert "Tool Effectiveness" in text
    assert 'id="results-ops-bar"' in text
    assert 'id="results-export-menu"' in text
    assert 'id="results-rerun-toggle"' in text
    assert 'id="results-rerun-panel"' in text
    assert 'id="results-rerun-scenarios-toggle"' in text
    assert 'id="results-rerun-scenarios-panel"' in text
    assert 'id="results-section-toolbar"' in text
    assert 'data-results-section="overview"' in text
    assert 'data-results-section="risk"' in text
    assert 'data-results-section="attempts"' in text
    assert 'data-results-section="diagnostics"' in text
    assert 'data-results-section="exports"' in text
    assert 'id="results-panel-overview"' in text
    assert 'id="results-panel-risk"' in text
    assert 'id="results-panel-attempts"' in text
    assert 'id="results-panel-diagnostics"' in text
    assert 'id="results-panel-exports"' in text
    assert 'id="attempts-back-to-top"' in text
    assert 'id="current-attempt-step"' in text
    assert 'id="attempt-step-log"' in text
    assert "Current vs Baseline" in text
    assert '<summary class="rerun-btn">' not in text
    assert "dashboard-png-export-btn" in text
    assert "js/dashboard_capture.js" in text

    overview_panel = re.search(r'<section id="results-panel-overview"[^>]*>', text)
    assert overview_panel is not None
    assert "hidden" not in overview_panel.group(0)

    risk_panel = re.search(r'<section id="results-panel-risk"[^>]*>', text)
    assert risk_panel is not None
    assert "hidden" in risk_panel.group(0)

    attempts_panel = re.search(r'<section id="results-panel-attempts"[^>]*>', text)
    assert attempts_panel is not None
    assert "hidden" in attempts_panel.group(0)

    diagnostics_panel = re.search(r'<section id="results-panel-diagnostics"[^>]*>', text)
    assert diagnostics_panel is not None
    assert "hidden" in diagnostics_panel.group(0)

    exports_panel = re.search(r'<section id="results-panel-exports"[^>]*>', text)
    assert exports_panel is not None
    assert "hidden" in exports_panel.group(0)
    assert "Download machine-readable results, transcripts, and dashboard captures for this run." in text
    assert text.count('data-export-format="csv"') >= 2
    assert text.count('data-export-format="failures_csv"') >= 2


def test_results_page_initial_render_uses_attempt_chunking():
    app = create_app()
    app.config["TESTING"] = True
    app.config["latest_report"] = _large_report(total_attempts=25)

    client = app.test_client()
    response = client.get("/results")
    text = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Attempt #1" in text
    assert "Attempt #20" in text
    assert "Attempt #21" not in text
    assert "Load more attempts (5)" in text


def test_results_page_groups_attempts_by_expected_intent():
    app = create_app()
    app.config["TESTING"] = True
    app.config["latest_report"] = _intent_grouped_report()

    client = app.test_client()
    response = client.get("/results")
    text = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Intent:" in text
    assert "flight_cancel" in text
    assert "Behavior / Journey" in text
    assert "Scenario Intent" in text
    assert "Scenario Journey" in text
    assert 'id="all-attempts-panel-static"' in text
    assert 'id="all-attempts-panel-live"' in text
    assert 'id="all-attempts-static-tree"' in text
    assert 'id="live-attempts-list"' in text
    assert "Expand All" in text
    assert "Collapse All" in text
    assert 'data-attempt-tree-id="all-attempts-static-tree"' in text
    assert 'data-attempt-tree-id="live-attempts-list"' in text
    assert "class=\"intent-group\"" in text
    assert "class=\"intent-scenario-details\"" in text
    assert 'data-results-panel="attempts"' in text
    assert 'data-results-panel="diagnostics"' in text

    static_panel = re.search(r'<details class="all-attempts-panel" id="all-attempts-panel-static"([^>]*)>', text)
    assert static_panel is not None
    assert "open" not in static_panel.group(1)

    live_panel = re.search(r'<details class="all-attempts-panel" id="all-attempts-panel-live"([^>]*)>', text)
    assert live_panel is not None
    assert "open" not in live_panel.group(1)


def test_results_page_hides_journey_dashboard_when_flag_disabled():
    app = create_app()
    app.config["TESTING"] = True
    app.config["latest_report"] = _journey_dashboard_report()
    app.config["last_run_config"] = AppConfig(journey_dashboard_enabled=False)

    client = app.test_client()
    response = client.get("/results")
    text = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "<h3>Journey Evaluation Dashboard</h3>" not in text


def test_results_page_renders_journey_dashboard_with_toolbar_when_enabled():
    app = create_app()
    app.config["TESTING"] = True
    app.config["latest_report"] = _journey_dashboard_report()
    app.config["last_run_config"] = AppConfig(journey_dashboard_enabled=True)

    client = app.test_client()
    response = client.get("/results?journey_view=live_agent_transfer")
    text = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Journey Evaluation Dashboard" in text
    assert "Live Agent Transfer" in text
    assert "journey-view-btn-active" in text
    assert 'data-journey-view="live_agent_transfer"' in text
    assert 'data-journey-panel="live_agent_transfer"' in text
    assert '<a href="/results?journey_view=' not in text
    assert "Agent Request - Successful Transfer To Agent" in text


def test_results_page_renders_analytics_run_diagnostics_panel_when_present():
    app = create_app()
    app.config["TESTING"] = True
    app.config["latest_report"] = _analytics_diagnostics_report()

    client = app.test_client()
    response = client.get("/results")
    text = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Analytics Run Diagnostics" in text
    assert "Pages Fetched" in text
    assert "Rows Scanned" in text
    assert "Timeline Preview" in text
    assert "Raw Diagnostics JSON" in text
    assert 'data-results-section="diagnostics"' in text
    assert 'id="results-panel-diagnostics"' in text


def test_results_page_recent_step_log_retains_more_than_twenty_events():
    app = create_app()
    app.config["TESTING"] = True
    app.config["latest_report"] = _sample_report()
    emitter = ProgressEmitter()
    for idx in range(30):
        emitter.emit(
            ProgressEvent(
                event_type=ProgressEventType.ATTEMPT_STATUS,
                suite_name="Web Suite",
                message=f"status-{idx + 1}",
                scenario_name="Scenario A",
                attempt_number=1,
            )
        )
    app.config["progress_emitter"] = emitter

    client = app.test_client()
    response = client.get("/results")
    text = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "status-1" in text
    assert "status-30" in text
    assert 'id="current-attempt-step"' in text
    assert 'id="attempt-step-log"' in text


def test_results_page_journey_toolbar_preserves_baseline_and_export_context(
    tmp_path, monkeypatch
):
    history_dir = tmp_path / "history"
    monkeypatch.setenv("GC_TESTER_HISTORY_DIR", str(history_dir))
    app = create_app()
    app.config["TESTING"] = True
    app.config["last_run_config"] = AppConfig(journey_dashboard_enabled=True)

    store = RunHistoryStore(str(history_dir), max_runs=50)
    baseline = _journey_dashboard_report()
    baseline.timestamp = datetime(2026, 4, 18, 10, 0, tzinfo=timezone.utc)
    baseline.overall_success_rate = 0.0
    baseline.overall_successes = 0
    baseline.overall_failures = 1
    baseline.scenario_results[0].success_rate = 0.0
    baseline.scenario_results[0].successes = 0
    baseline.scenario_results[0].failures = 1
    baseline.scenario_results[0].attempt_results[0].success = False
    baseline_entry = store.save_report(baseline)

    current = _journey_dashboard_report()
    current.timestamp = datetime(2026, 4, 18, 12, 0, tzinfo=timezone.utc)
    current_entry = store.save_report(current)

    app.config["history_store"] = store
    app.config["latest_report"] = current
    app.config["latest_run_history_entry"] = current_entry

    client = app.test_client()
    response = client.get(
        f"/results?journey_view=live_agent_transfer&baseline_run_id={baseline_entry['run_id']}"
    )
    text = response.get_data(as_text=True)

    assert response.status_code == 200
    assert (
        'id="baseline-journey-view" name="journey_view" value="live_agent_transfer"'
        in text
    )
    assert 'data-export-format="csv"' in text
    assert f"baseline_run_id={baseline_entry['run_id']}" in text
    assert "journey_view=live_agent_transfer" in text


def test_results_attempt_chunk_endpoint_returns_html_and_paging_state():
    app = create_app()
    app.config["TESTING"] = True
    app.config["latest_report"] = _large_report(total_attempts=25)

    client = app.test_client()
    response = client.get("/results/attempts?scenario_index=0&offset=20&limit=20")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["has_more"] is False
    assert payload["remaining"] == 0
    assert payload["next_offset"] == 25
    assert "Attempt #21" in payload["html"]
    assert "Attempt #25" in payload["html"]


def test_results_attempt_chunk_endpoint_out_of_range_is_safe():
    app = create_app()
    app.config["TESTING"] = True
    app.config["latest_report"] = _sample_report()

    client = app.test_client()
    response = client.get("/results/attempts?scenario_index=99&offset=0&limit=20")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload == {
        "html": "",
        "next_offset": 0,
        "has_more": False,
        "remaining": 0,
    }


def test_results_page_shows_compare_panel_with_baseline(tmp_path, monkeypatch):
    history_dir = tmp_path / "history"
    monkeypatch.setenv("GC_TESTER_HISTORY_DIR", str(history_dir))
    app = create_app()
    app.config["TESTING"] = True

    store = RunHistoryStore(str(history_dir), max_runs=50)
    baseline = _sample_report()
    baseline.timestamp = datetime(2026, 4, 18, 11, 0, tzinfo=timezone.utc)
    baseline.overall_success_rate = 0.0
    baseline.overall_successes = 0
    baseline.overall_failures = 1
    baseline.scenario_results[0].success_rate = 0.0
    baseline.scenario_results[0].successes = 0
    baseline.scenario_results[0].failures = 1
    baseline.scenario_results[0].attempt_results[0].success = False
    baseline_entry = store.save_report(baseline)

    current = _sample_report()
    current.timestamp = datetime(2026, 4, 18, 12, 0, tzinfo=timezone.utc)
    current_entry = store.save_report(current)

    app.config["history_store"] = store
    app.config["latest_report"] = current
    app.config["latest_run_history_entry"] = current_entry

    client = app.test_client()
    response = client.get("/results")
    text = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Current vs Baseline" in text
    assert "Baseline Suite" in text
    assert baseline_entry["timestamp"] in text


def test_results_history_endpoint_returns_ordered_suite_runs(tmp_path, monkeypatch):
    history_dir = tmp_path / "history"
    monkeypatch.setenv("GC_TESTER_HISTORY_DIR", str(history_dir))
    app = create_app()
    app.config["TESTING"] = True

    store = RunHistoryStore(str(history_dir), max_runs=50)
    first = _sample_report()
    first.timestamp = datetime(2026, 4, 18, 10, 0, tzinfo=timezone.utc)
    store.save_report(first)
    second = _sample_report()
    second.timestamp = datetime(2026, 4, 18, 11, 0, tzinfo=timezone.utc)
    store.save_report(second)

    other = _sample_report()
    other.suite_name = "Other Suite"
    other.timestamp = datetime(2026, 4, 18, 12, 0, tzinfo=timezone.utc)
    store.save_report(other)

    app.config["history_store"] = store
    client = app.test_client()
    response = client.get("/results/history?suite_name=Web%20Suite&limit=2")

    assert response.status_code == 200
    payload = response.get_json()
    assert len(payload["runs"]) == 2
    assert payload["runs"][0]["suite_name"] == "Web Suite"
    assert payload["runs"][0]["timestamp"] > payload["runs"][1]["timestamp"]
    assert payload["runs"][0]["storage_type"] in {"full_json", "gz_json", "summary_only"}


def test_results_page_applies_selected_baseline_run_id(tmp_path, monkeypatch):
    history_dir = tmp_path / "history"
    monkeypatch.setenv("GC_TESTER_HISTORY_DIR", str(history_dir))
    app = create_app()
    app.config["TESTING"] = True

    store = RunHistoryStore(str(history_dir), max_runs=50)
    baseline_old = _sample_report()
    baseline_old.timestamp = datetime(2026, 4, 18, 9, 0, tzinfo=timezone.utc)
    baseline_old.overall_success_rate = 0.0
    baseline_old.overall_successes = 0
    baseline_old.overall_failures = 1
    baseline_old.scenario_results[0].success_rate = 0.0
    baseline_old.scenario_results[0].successes = 0
    baseline_old.scenario_results[0].failures = 1
    baseline_old.scenario_results[0].attempt_results[0].success = False
    baseline_old_entry = store.save_report(baseline_old)

    baseline_newer = _sample_report()
    baseline_newer.timestamp = datetime(2026, 4, 18, 10, 0, tzinfo=timezone.utc)
    store.save_report(baseline_newer)

    current = _sample_report()
    current.timestamp = datetime(2026, 4, 18, 12, 0, tzinfo=timezone.utc)
    current_entry = store.save_report(current)

    app.config["history_store"] = store
    app.config["latest_report"] = current
    app.config["latest_run_history_entry"] = current_entry

    client = app.test_client()
    response = client.get(f"/results?baseline_run_id={baseline_old_entry['run_id']}")
    text = response.get_data(as_text=True)

    assert response.status_code == 200
    assert f'value="{baseline_old_entry["run_id"]}" selected' in text
    assert baseline_old_entry["timestamp"] in text


def test_results_export_dashboard_pdf_honors_selected_baseline(tmp_path, monkeypatch):
    history_dir = tmp_path / "history"
    monkeypatch.setenv("GC_TESTER_HISTORY_DIR", str(history_dir))
    app = create_app()
    app.config["TESTING"] = True

    store = RunHistoryStore(str(history_dir), max_runs=50)
    baseline = _sample_report()
    baseline.timestamp = datetime(2026, 4, 18, 10, 0, tzinfo=timezone.utc)
    baseline.overall_success_rate = 0.0
    baseline.overall_successes = 0
    baseline.overall_failures = 1
    baseline.scenario_results[0].success_rate = 0.0
    baseline.scenario_results[0].successes = 0
    baseline.scenario_results[0].failures = 1
    baseline.scenario_results[0].attempt_results[0].success = False
    baseline_entry = store.save_report(baseline)

    current = _sample_report()
    current.timestamp = datetime(2026, 4, 18, 12, 0, tzinfo=timezone.utc)
    current_entry = store.save_report(current)

    app.config["history_store"] = store
    app.config["latest_report"] = current
    app.config["latest_run_history_entry"] = current_entry

    client = app.test_client()
    response = client.get(
        f"/results/export?format=dashboard_pdf&baseline_run_id={baseline_entry['run_id']}"
    )

    assert response.status_code == 200
    assert response.mimetype == "application/pdf"
    assert baseline_entry["timestamp"].encode("utf-8") in response.data


def test_rerun_subset_failed_bucket_no_eligible_scenarios_shows_message():
    app = create_app()
    app.config["TESTING"] = True
    app.config["latest_report"] = _sample_report()
    app.config["last_run_config"] = AppConfig(
        gc_region="example.com",
        gc_deployment_id="deploy-id",
        ollama_model="llama3",
    )
    app.config["last_run_suite"] = _suite_for_web()

    client = app.test_client()
    response = client.post(
        "/run/rerun_subset",
        data={"mode": "failed_bucket"},
        follow_redirects=True,
    )

    text = response.get_data(as_text=True)
    assert response.status_code == 200
    assert "No failed/timeout/skipped scenarios were found in the latest results." in text


def test_rerun_subset_selected_mode_requires_selected_scenarios():
    app = create_app()
    app.config["TESTING"] = True
    app.config["latest_report"] = _sample_report()
    app.config["last_run_config"] = AppConfig(
        gc_region="example.com",
        gc_deployment_id="deploy-id",
        ollama_model="llama3",
    )
    app.config["last_run_suite"] = _suite_for_web()

    client = app.test_client()
    response = client.post(
        "/run/rerun_subset",
        data={"mode": "selected"},
        follow_redirects=True,
    )

    text = response.get_data(as_text=True)
    assert response.status_code == 200
    assert "Select at least one scenario to rerun." in text


def test_rerun_subset_failed_bucket_builds_filtered_suite(monkeypatch):
    class _ImmediateThread:
        def __init__(self, target=None, daemon=None):
            self._target = target

        def start(self):
            if self._target:
                self._target()

    class _FakeOrchestrator:
        captured_suite = None

        def __init__(self, config, progress_emitter, stop_event):
            pass

        async def run_suite(self, suite):
            _FakeOrchestrator.captured_suite = suite
            report = _sample_report()
            report.suite_name = suite.name
            return report

    monkeypatch.setattr("src.web_app.threading.Thread", _ImmediateThread)
    monkeypatch.setattr("src.web_app.TestOrchestrator", _FakeOrchestrator)
    _patch_web_judge_builder(monkeypatch)

    app = create_app()
    app.config["TESTING"] = True
    app.config["latest_report"] = _latest_report_with_failed_bucket()
    app.config["last_run_config"] = AppConfig(
        gc_region="example.com",
        gc_deployment_id="deploy-id",
        ollama_model="llama3",
    )
    app.config["last_run_suite"] = TestSuite(
        name="Web Suite",
        scenarios=[
            TestScenario(name="Scenario A", persona="Traveler", goal="Goal A", first_message="A", attempts=1),
            TestScenario(name="Scenario B", persona="Traveler", goal="Goal B", first_message="B", attempts=1),
        ],
    )

    client = app.test_client()
    response = client.post(
        "/run/rerun_subset",
        data={"mode": "failed_bucket"},
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert _FakeOrchestrator.captured_suite is not None
    assert [s.name for s in _FakeOrchestrator.captured_suite.scenarios] == ["Scenario B"]


def test_rerun_subset_selected_builds_filtered_suite(monkeypatch):
    class _ImmediateThread:
        def __init__(self, target=None, daemon=None):
            self._target = target

        def start(self):
            if self._target:
                self._target()

    class _FakeOrchestrator:
        captured_suite = None

        def __init__(self, config, progress_emitter, stop_event):
            pass

        async def run_suite(self, suite):
            _FakeOrchestrator.captured_suite = suite
            report = _sample_report()
            report.suite_name = suite.name
            return report

    monkeypatch.setattr("src.web_app.threading.Thread", _ImmediateThread)
    monkeypatch.setattr("src.web_app.TestOrchestrator", _FakeOrchestrator)
    _patch_web_judge_builder(monkeypatch)

    app = create_app()
    app.config["TESTING"] = True
    app.config["latest_report"] = _latest_report_with_failed_bucket()
    app.config["last_run_config"] = AppConfig(
        gc_region="example.com",
        gc_deployment_id="deploy-id",
        ollama_model="llama3",
    )
    app.config["last_run_suite"] = TestSuite(
        name="Web Suite",
        scenarios=[
            TestScenario(name="Scenario A", persona="Traveler", goal="Goal A", first_message="A", attempts=1),
            TestScenario(name="Scenario B", persona="Traveler", goal="Goal B", first_message="B", attempts=1),
        ],
    )

    client = app.test_client()
    response = client.post(
        "/run/rerun_subset",
        data={"mode": "selected", "scenario_names": ["Scenario A"]},
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert _FakeOrchestrator.captured_suite is not None
    assert [s.name for s in _FakeOrchestrator.captured_suite.scenarios] == ["Scenario A"]


def test_home_page_shows_transcript_suite_renamed_labels():
    app = create_app()
    app.config["TESTING"] = True

    client = app.test_client()
    response = client.get("/")
    text = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Regression Test Harness" in text
    assert 'id="theme-toggle"' in text
    assert "rth_theme_preference" in text
    assert "home-tab-language" in text
    assert "home-tab-harness" in text
    assert "home-tab-analytics" in text
    assert "home-tab-transcript" in text
    assert ">Harness</button>" in text
    assert ">Analytics</button>" in text
    assert ">Transcript</button>" in text
    assert ">Defaults</button>" in text
    assert "class=\"home-tab utility-tab" in text
    assert "quick-start-grid" not in text
    assert "analytics-journey-form" in text
    assert "analytics_region" in text
    assert "analytics_auth_mode" in text
    assert "judge_execution_mode" in text
    assert "judge_single_model" in text
    assert "analytics_judge_execution_mode" in text
    assert "analytics_judge_single_model" in text
    assert "Effective primary judge" in text
    assert "gemma4:e4b" in text
    assert "gemma4:31b" in text
    assert "OAuth Client Credentials" in text
    assert "Manual Bearer Token" in text
    assert "analytics_gc_client_id" in text
    assert "analytics_gc_client_secret" in text
    assert "analytics_bearer_token" in text
    assert "analytics-get-token-button" in text
    assert "analytics-token-output" in text
    assert "analytics-token-copy-button" in text
    assert "analytics-token-reveal-button" in text
    assert "analytics-test-api-button" in text
    assert "analytics-test-api-cc-button" in text
    assert "analytics-api-test-output" in text
    assert "analytics_ollama_model" in text
    assert "analytics_bot_flow_id" in text
    assert "Used directly in <code>/api/v2/analytics/botflows/{botFlowId}/divisions/reportingturns</code>." in text
    assert "analytics_interval" in text
    assert "analytics-interval-group" in text
    assert "analytics-interval-trigger" not in text
    assert "Choose a local date/time range; we auto-convert to canonical UTC ISO-8601 interval format for Genesys." in text
    assert "Rows requested per Analytics API page (1-100)." in text
    assert "Hard cap of conversations evaluated in this run." in text
    assert 'id="analytics_page_size"' in text
    assert 'max="100"' in text
    assert 'vendor/flatpickr/flatpickr.min.css' in text
    assert 'vendor/flatpickr/flatpickr.min.js' in text
    assert 'data-analytics-interval-preset="today"' in text
    assert 'data-analytics-interval-preset="yesterday"' in text
    assert 'data-analytics-interval-preset="last_7_days"' in text
    assert 'data-analytics-interval-preset="last_24_hours"' in text
    assert 'data-analytics-interval-preset="clear"' in text
    assert "initAnalyticsIntervalPicker" in text
    assert "validateAnalyticsIntervalField" in text
    assert "openAnalyticsIntervalPicker" in text
    assert "resolveAnalyticsIntervalPicker" in text
    assert "applyAnalyticsIntervalMonthLayout" in text
    assert "verifyAnalyticsIntervalVerticalFit" in text
    assert "preferredAnalyticsIntervalMonths" in text
    assert "layoutAnalyticsIntervalPicker" in text
    assert "positionAnalyticsIntervalPicker" in text
    assert "analyticsIntervalTrigger" not in text
    assert "const preferredMonths = preferredAnalyticsIntervalMonths(picker);" in text
    assert "applyAnalyticsIntervalMonthLayout(picker);" in text
    assert "picker.open(undefined, picker.altInput || analyticsIntervalInput)" in text
    assert "appendTo: analyticsIntervalGroup || undefined" not in text
    assert "position: 'below left'" in text
    assert "picker.set('positionElement', picker.altInput);" in text
    assert "instance.set('positionElement', instance.altInput)" in text
    assert "onOpen: function(selectedDates, dateStr, instance)" in text
    assert "onMonthChange: function()" in text
    assert "if (picker.config.showMonths !== preferredMonths)" in text
    assert "window.requestAnimationFrame(function() {" in text
    assert "verifyAnalyticsIntervalVerticalFit(instance);" in text
    assert "background: rgba(15, 23, 42, 0.86);" in text
    assert "html[data-theme=\"dark\"] .flatpickr-time input:hover" in text
    assert "html[data-theme=\"dark\"] .flatpickr-time .flatpickr-am-pm:hover" in text
    assert "html[data-theme=\"dark\"] .flatpickr-time input:active" in text
    assert "html[data-theme=\"dark\"] .flatpickr-time .numInputWrapper:focus-within" in text
    assert "copyTextToClipboard" in text
    assert "/run/analytics_journey/token" in text
    assert "/run/analytics_journey/test" in text
    assert "/run/analytics_journey/test/client_credentials" in text
    assert "cdn.jsdelivr" not in text
    assert "action=\"/run/analytics_journey\"" in text
    assert "Transcript Suite Name" in text
    assert "Seed From Uploaded Transcript" in text
    assert "Conversation IDs" in text
    assert "Transcript URL" in text
    assert "Automation" in text
    assert "Seed From Transcript URL" in text
    assert "transcript_url" in text
    assert "action=\"/seed/url\"" in text
    assert "action=\"/transcript/import/settings\"" in text
    assert "harness_mode" in text
    assert "journey_category_strategy" in text
    assert "judging_mechanics_enabled" in text
    assert "judging_objective_profile" in text
    assert "judging_strictness" in text
    assert "judging_tolerance" in text
    assert "judging_containment_weight" in text
    assert "judging_fulfillment_weight" in text
    assert "judging_path_weight" in text
    assert "judging_explanation_mode" in text
    assert "journey_dashboard_enabled" in text
    assert "attempt_parallel_enabled" in text
    assert "max_parallel_attempt_workers" in text
    assert "knowledge_mode_timeout_seconds" in text
    assert "judge-single-model-group" in text
    assert "journey-strategy-group" in text
    assert "analytics-judge-single-model-group" in text
    assert 'id="run-mode-settings"' in text
    assert 'id="advanced-run-performance"' in text
    assert 'id="advanced-run-scoring"' in text
    assert 'id="advanced-run-api-fallback"' in text
    assert 'id="advanced-run-diagnostics"' in text
    assert 'id="advanced-analytics-model-override"' in text
    assert 'id="advanced-analytics-filters"' in text
    assert 'id="advanced-analytics-connection-tools"' in text
    assert 'max="3"' in text
    assert "Valid range is <code>1..3</code>" in text
    assert "evaluation_results_language" in text
    assert "seed_strategy" in text
    assert 'name="csrf_token"' in text
    assert "What this field means" not in text
    assert text.count('class="help-icon"') >= 10
    assert text.count('class="field-label-row"') >= 10
    assert "closeLegendPopovers" in text
    assert "closeLegendPopover" in text
    assert "placeLegendPopover" in text
    assert "shouldUseHoverFieldLegends" in text
    assert "(hover: hover) and (pointer: fine)" in text
    assert "legend.addEventListener('mouseenter'" in text
    assert "legend.addEventListener('mouseleave'" in text
    assert "legend.addEventListener('focusin'" in text
    assert "legend.addEventListener('focusout'" in text
    assert "summary.addEventListener('click'" in text
    assert "summary.addEventListener('keydown'" in text
    assert "hoverPointerQuery.addEventListener('change'" in text
    assert "updateHarnessModeFields" in text
    assert "judgeSingleModelGroup" in text
    assert "journeyStrategyGroup" in text
    assert "analyticsJudgeSingleModelGroup" in text
    assert text.count('class="field-legend-content"') >= 10
    assert ".advanced-panel > summary" in text
    assert ".advanced-panel[open] > summary" in text
    assert ".advanced-panel .field-legend summary" in text
    assert ".advanced-panel summary {" not in text
    assert 'aria-label="Field help for Deployment ID"' in text
    assert 'aria-label="Field help for Genesys OAuth Client ID"' in text
    assert 'id="legend-deployment_id"' in text
    assert 'id="legend-region"' in text
    assert 'id="legend-ollama_model"' in text
    assert 'id="legend-max_turns"' in text
    assert 'id="legend-harness_mode"' in text
    assert 'id="legend-journey_category_strategy"' in text
    assert 'id="legend-judging_mechanics_enabled"' in text
    assert 'id="legend-judging_objective_profile"' in text
    assert 'id="legend-judging_strictness"' in text
    assert 'id="legend-judging_tolerance"' in text
    assert 'id="legend-judging_containment_weight"' in text
    assert 'id="legend-judging_fulfillment_weight"' in text
    assert 'id="legend-judging_path_weight"' in text
    assert 'id="legend-judging_explanation_mode"' in text
    assert 'id="legend-journey_dashboard_enabled"' in text
    assert 'id="legend-attempt_parallel_enabled"' in text
    assert 'id="legend-max_parallel_attempt_workers"' in text
    assert 'id="legend-knowledge_mode_timeout_seconds"' in text
    assert 'id="legend-test_suite_file"' in text
    assert 'id="legend-gc_client_id"' in text
    assert 'id="legend-gc_client_secret"' in text
    assert 'id="legend-intent_attribute_name"' in text
    assert 'id="legend-debug_capture_frames"' in text
    assert 'id="legend-debug_capture_frame_limit"' in text
    assert 'id="legend-analytics_journey_enabled"' in text
    assert 'id="legend-analytics_auth_mode"' in text
    assert 'id="legend-analytics_region"' in text
    assert 'id="legend-analytics_gc_client_id"' in text
    assert 'id="legend-analytics_gc_client_secret"' in text
    assert 'id="legend-analytics_bearer_token"' in text
    assert 'id="legend-analytics_judge_execution_mode"' in text
    assert 'id="legend-analytics_judge_single_model"' in text
    assert 'id="legend-analytics_bot_flow_id"' in text
    assert 'id="legend-analytics_interval"' in text
    assert 'id="legend-analytics_ollama_model"' in text
    assert 'id="legend-analytics_divisions"' in text
    assert 'id="legend-analytics_page_size"' in text
    assert 'id="legend-analytics_max_conversations"' in text
    assert 'id="legend-analytics_filter_json"' in text
    assert 'id="legend-analytics_token_capture"' in text
    assert 'id="legend-analytics_api_connectivity_test"' in text
    assert "journey validation and ignores" in text
    assert "rules_first" in text
    assert "llm_first" in text
    assert 'value="inherit"' in text
    assert 'value="en"' in text
    assert 'value="fr"' in text
    assert 'value="fr-CA"' in text
    assert 'value="es"' in text
    assert 'id="run-language-select"' in text
    assert 'id="transcript-language-select"' in text
    assert 'id="evaluation-results-language-select"' in text
    assert "Run Language" in text
    assert "Transcript Language" in text
    assert 'value="fr-CA"' in text
    assert 'name="language"' in text
    assert "id_source_mode" in text
    assert "transcript_import_time" in text
    assert "rth_home_active_tab" in text
    assert "rth_transcript_active_tab" in text
    assert "Seed Suite From Transcript (Phase 4 MVP)" not in text
    assert "Seeded Suite Name (Optional)" not in text

    harness_section = re.search(r'<section id="home-panel-harness"[^>]*>', text)
    assert harness_section is not None
    assert "hidden" not in harness_section.group(0)

    language_section = re.search(r'<section id="home-panel-language"[^>]*>', text)
    assert language_section is not None
    assert "hidden" in language_section.group(0)

    transcript_section = re.search(r'<section id="home-panel-transcript"[^>]*>', text)
    assert transcript_section is not None
    assert "hidden" in transcript_section.group(0)

    analytics_section = re.search(r'<section id="home-panel-analytics"[^>]*>', text)
    assert analytics_section is not None
    assert "hidden" in analytics_section.group(0)

    upload_panel = re.search(r'<div id="transcript-subtab-upload"[^>]*>', text)
    assert upload_panel is not None
    assert "hidden" not in upload_panel.group(0)

    for panel_id in [
        "run-mode-settings",
        "advanced-run-performance",
        "advanced-run-scoring",
        "advanced-run-api-fallback",
        "advanced-run-diagnostics",
        "advanced-analytics-model-override",
        "advanced-analytics-filters",
        "advanced-analytics-connection-tools",
    ]:
        panel = re.search(rf'<details class="advanced-panel(?: [^"]*)?" id="{panel_id}"[^>]*>', text)
        assert panel is not None
        assert "open" not in panel.group(0)


def test_run_error_preserves_language_and_evaluation_results_selection():
    app = create_app()
    app.config["TESTING"] = True

    client = app.test_client()
    response = client.post(
        "/run",
        data={
            "deployment_id": "dep-id",
            "region": "mypurecloud.com",
            "ollama_model": "llama3",
            "language": "fr-CA",
            "evaluation_results_language": "es",
        },
    )
    text = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Please upload a test suite file" in text
    assert 'name="language" class="run-language-bound-input" value="fr-CA"' in text
    assert '<option value="es" selected>Spanish</option>' in text


def test_home_page_query_tabs_render_selected_panes_visible():
    app = create_app()
    app.config["TESTING"] = True

    client = app.test_client()
    response = client.get("/?home_tab=transcript&transcript_tab=automation")
    text = response.get_data(as_text=True)

    assert response.status_code == 200

    transcript_section = re.search(r'<section id="home-panel-transcript"[^>]*>', text)
    assert transcript_section is not None
    assert "hidden" not in transcript_section.group(0)

    harness_section = re.search(r'<section id="home-panel-harness"[^>]*>', text)
    assert harness_section is not None
    assert "hidden" in harness_section.group(0)

    automation_panel = re.search(r'<div id="transcript-subtab-automation"[^>]*>', text)
    assert automation_panel is not None
    assert "hidden" not in automation_panel.group(0)

    upload_panel = re.search(r'<div id="transcript-subtab-upload"[^>]*>', text)
    assert upload_panel is not None
    assert "hidden" in upload_panel.group(0)


def test_run_analytics_journey_route_starts_background_run(monkeypatch):
    class _ImmediateThread:
        def __init__(self, target=None, daemon=None):
            self._target = target

        def start(self):
            if self._target:
                self._target()

    class _FakeAnalyticsRunner:
        captured_request = None

        def __init__(self, config, progress_emitter, stop_event, artifact_store=None):
            pass

        async def run(self, run_request):
            _FakeAnalyticsRunner.captured_request = run_request
            report = _sample_report()
            report.suite_name = "Analytics Journey Regression - flow-123"
            return report

    monkeypatch.setenv("GC_REGION", "usw2.pure.cloud")
    monkeypatch.setenv("GC_CLIENT_ID", "client-id")
    monkeypatch.setenv("GC_CLIENT_SECRET", "client-secret")
    monkeypatch.setenv("OLLAMA_MODEL", "llama3")
    monkeypatch.setattr("src.web_app.threading.Thread", _ImmediateThread)
    monkeypatch.setattr("src.web_app.AnalyticsJourneyRunner", _FakeAnalyticsRunner)
    _patch_web_judge_builder(monkeypatch)

    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()
    response = client.post(
        "/run/analytics_journey",
        data={
            "analytics_journey_enabled": "on",
            "analytics_bot_flow_id": "flow-123",
            "analytics_interval": "2026-04-19T00:00:00.000Z/2026-04-20T00:00:00.000Z",
            "analytics_page_size": "25",
            "analytics_max_conversations": "40",
            "analytics_divisions": "div-1,div-2",
            "analytics_language_filter": "fr-CA",
            "language": "en",
            "evaluation_results_language": "fr-CA",
        },
        follow_redirects=False,
    )

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/results")
    assert _FakeAnalyticsRunner.captured_request is not None
    assert _FakeAnalyticsRunner.captured_request.bot_flow_id == "flow-123"
    assert _FakeAnalyticsRunner.captured_request.page_size == 25
    assert _FakeAnalyticsRunner.captured_request.max_conversations == 40
    assert _FakeAnalyticsRunner.captured_request.divisions == ["div-1", "div-2"]


def test_run_analytics_journey_route_accepts_analytics_form_overrides(monkeypatch):
    class _ImmediateThread:
        def __init__(self, target=None, daemon=None):
            self._target = target

        def start(self):
            if self._target:
                self._target()

    class _FakeAnalyticsRunner:
        captured_request = None

        def __init__(self, config, progress_emitter, stop_event, artifact_store=None):
            pass

        async def run(self, run_request):
            _FakeAnalyticsRunner.captured_request = run_request
            report = _sample_report()
            report.suite_name = "Analytics Journey Regression - flow-abc"
            return report

    monkeypatch.delenv("GC_REGION", raising=False)
    monkeypatch.delenv("GC_CLIENT_ID", raising=False)
    monkeypatch.delenv("GC_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.setattr("src.web_app.threading.Thread", _ImmediateThread)
    monkeypatch.setattr("src.web_app.AnalyticsJourneyRunner", _FakeAnalyticsRunner)
    _patch_web_judge_builder(monkeypatch)

    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()
    response = client.post(
        "/run/analytics_journey",
        data={
            "analytics_journey_enabled": "on",
            "analytics_region": "usw2.pure.cloud",
            "analytics_gc_client_id": "form-client-id",
            "analytics_gc_client_secret": "form-client-secret",
            "analytics_ollama_model": "llama3",
            "analytics_bot_flow_id": "flow-abc",
            "analytics_interval": "2026-04-19T00:00:00.000Z/2026-04-20T00:00:00.000Z",
            "analytics_page_size": "20",
            "analytics_max_conversations": "35",
        },
        follow_redirects=False,
    )

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/results")
    assert _FakeAnalyticsRunner.captured_request is not None
    assert _FakeAnalyticsRunner.captured_request.bot_flow_id == "flow-abc"
    assert _FakeAnalyticsRunner.captured_request.page_size == 20
    assert _FakeAnalyticsRunner.captured_request.max_conversations == 35


def test_run_analytics_journey_route_supports_manual_bearer_mode(monkeypatch):
    class _ImmediateThread:
        def __init__(self, target=None, daemon=None):
            self._target = target

        def start(self):
            if self._target:
                self._target()

    class _FakeAnalyticsRunner:
        captured_request = None

        def __init__(self, config, progress_emitter, stop_event, artifact_store=None):
            pass

        async def run(self, run_request):
            _FakeAnalyticsRunner.captured_request = run_request
            report = _sample_report()
            report.suite_name = "Analytics Journey Regression - reporting-turns"
            return report

    monkeypatch.delenv("GC_REGION", raising=False)
    monkeypatch.delenv("GC_CLIENT_ID", raising=False)
    monkeypatch.delenv("GC_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.setattr("src.web_app.threading.Thread", _ImmediateThread)
    monkeypatch.setattr("src.web_app.AnalyticsJourneyRunner", _FakeAnalyticsRunner)
    _patch_web_judge_builder(monkeypatch)

    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()
    response = client.post(
        "/run/analytics_journey",
        data={
            "analytics_journey_enabled": "on",
            "analytics_auth_mode": "manual_bearer",
            "analytics_bearer_token": "token-123",
            "analytics_region": "usw2.pure.cloud",
            "analytics_ollama_model": "llama3",
            "analytics_bot_flow_id": "flow-manual",
            "analytics_interval": "2026-04-19T00:00:00.000Z/2026-04-20T00:00:00.000Z",
            "analytics_page_size": "20",
            "analytics_max_conversations": "35",
        },
        follow_redirects=False,
    )

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/results")
    assert _FakeAnalyticsRunner.captured_request is not None
    assert _FakeAnalyticsRunner.captured_request.auth_mode == "manual_bearer"
    assert _FakeAnalyticsRunner.captured_request.manual_bearer_token == "token-123"


def test_run_analytics_journey_route_reports_missing_required_config(monkeypatch):
    monkeypatch.delenv("GC_REGION", raising=False)
    monkeypatch.delenv("GC_CLIENT_ID", raising=False)
    monkeypatch.delenv("GC_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    _patch_web_judge_builder(monkeypatch)

    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()
    response = client.post(
        "/run/analytics_journey",
        data={
            "analytics_journey_enabled": "on",
            "analytics_bot_flow_id": "flow-xyz",
            "analytics_interval": "2026-04-19T00:00:00.000Z/2026-04-20T00:00:00.000Z",
            "analytics_page_size": "10",
            "analytics_max_conversations": "10",
        },
        follow_redirects=True,
    )

    text = response.get_data(as_text=True)
    assert response.status_code == 200
    assert "Missing required configuration for analytics journey: gc_region, gc_client_id, gc_client_secret" in text


def test_run_analytics_journey_route_requires_bearer_token_in_manual_mode(monkeypatch):
    monkeypatch.delenv("GC_REGION", raising=False)
    monkeypatch.delenv("GC_CLIENT_ID", raising=False)
    monkeypatch.delenv("GC_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    _patch_web_judge_builder(monkeypatch)

    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()
    response = client.post(
        "/run/analytics_journey",
        data={
            "analytics_journey_enabled": "on",
            "analytics_auth_mode": "manual_bearer",
            "analytics_region": "usw2.pure.cloud",
            "analytics_ollama_model": "llama3",
            "analytics_bot_flow_id": "flow-manual",
            "analytics_interval": "2026-04-19T00:00:00.000Z/2026-04-20T00:00:00.000Z",
            "analytics_page_size": "10",
            "analytics_max_conversations": "10",
        },
        follow_redirects=True,
    )

    text = response.get_data(as_text=True)
    assert response.status_code == 200
    assert "Missing required configuration for analytics journey: manual_bearer_token" in text


def test_capture_analytics_token_route_success(monkeypatch):
    class _FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "access_token": "captured-token-123",
                "token_type": "Bearer",
                "expires_in": 3600,
            }

    monkeypatch.setattr("src.web_app.requests.post", lambda *args, **kwargs: _FakeResponse())

    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()
    home = client.get("/")
    csrf_match = re.search(
        r'name="csrf_token" value="([^"]+)"',
        home.get_data(as_text=True),
    )
    assert csrf_match is not None

    response = client.post(
        "/run/analytics_journey/token",
        headers={"X-CSRF-Token": csrf_match.group(1)},
        data={
            "analytics_region": "usw2.pure.cloud",
            "analytics_gc_client_id": "client-id",
            "analytics_gc_client_secret": "client-secret",
        },
        follow_redirects=False,
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload is not None
    assert payload.get("access_token") == "captured-token-123"
    assert payload.get("token_type") == "Bearer"
    assert payload.get("expires_in") == 3600
    assert payload.get("issued_at_utc")
    assert app.config.get("latest_report") is None


def test_capture_analytics_token_route_invalid_credentials(monkeypatch):
    class _FakeResponse:
        status_code = 401

        def raise_for_status(self):
            import requests

            raise requests.HTTPError("401 unauthorized")

    monkeypatch.setattr("src.web_app.requests.post", lambda *args, **kwargs: _FakeResponse())

    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()
    home = client.get("/")
    csrf_match = re.search(
        r'name="csrf_token" value="([^"]+)"',
        home.get_data(as_text=True),
    )
    assert csrf_match is not None

    response = client.post(
        "/run/analytics_journey/token",
        headers={"X-CSRF-Token": csrf_match.group(1)},
        data={
            "analytics_region": "usw2.pure.cloud",
            "analytics_gc_client_id": "bad-id",
            "analytics_gc_client_secret": "bad-secret",
        },
        follow_redirects=False,
    )

    assert response.status_code == 401
    payload = response.get_json()
    assert payload is not None
    assert "authorized" in str(payload.get("error") or "").lower()


def test_capture_analytics_token_route_requires_csrf():
    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()

    response = client.post(
        "/run/analytics_journey/token",
        data={
            "analytics_region": "usw2.pure.cloud",
            "analytics_gc_client_id": "client-id",
            "analytics_gc_client_secret": "client-secret",
        },
        follow_redirects=False,
    )
    assert response.status_code == 400


def test_test_analytics_journey_api_route_success_manual_bearer(monkeypatch):
    class _FakeAnalyticsClient:
        init_kwargs = None

        def __init__(self, **kwargs):
            _FakeAnalyticsClient.init_kwargs = kwargs

        @staticmethod
        def sanitize_extra_query_params(extra_params):
            return {"pageSize": 25}, ["ignoredKey"]

        def fetch_reporting_turns_page(self, **kwargs):
            return {
                "entities": [
                    {
                        "conversation": {"id": "conv-1"},
                        "language": "en",
                    },
                    {
                        "conversation": {"id": "conv-2"},
                        "language": "fr",
                    },
                ],
                "nextUri": "/api/v2/analytics/botflows/next",
            }

        @staticmethod
        def extract_rows(payload):
            return list(payload.get("entities") or [])

        @staticmethod
        def row_matches_language(row, language_filter):
            if not language_filter:
                return True
            return str(row.get("language") or "").lower() == str(language_filter).lower()

        @classmethod
        def filter_rows_by_language(cls, rows, language_filter):
            matching_rows = [
                row for row in rows if cls.row_matches_language(row, language_filter)
            ]
            return (
                matching_rows,
                {
                    "language_filter": language_filter,
                    "eligible_conversations": 1,
                    "selected_conversations": 1,
                    "excluded_missing_language_conversations": 0,
                    "excluded_mismatched_conversations": 1,
                },
            )

        @staticmethod
        def extract_conversation_id(row):
            conversation = row.get("conversation") if isinstance(row, dict) else None
            if isinstance(conversation, dict):
                return conversation.get("id")
            return None

        @staticmethod
        def extract_next_uri(payload):
            return payload.get("nextUri")

    monkeypatch.setattr("src.web_app.GenesysAnalyticsJourneyClient", _FakeAnalyticsClient)

    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()

    home = client.get("/")
    csrf_match = re.search(
        r'name="csrf_token" value="([^"]+)"',
        home.get_data(as_text=True),
    )
    assert csrf_match is not None

    response = client.post(
        "/run/analytics_journey/test",
        headers={"X-CSRF-Token": csrf_match.group(1)},
        data={
            "analytics_auth_mode": "manual_bearer",
            "analytics_bearer_token": "token-abc",
            "analytics_region": "usw2.pure.cloud",
            "analytics_bot_flow_id": "flow-123",
            "analytics_interval": "2026-04-19T00:00:00.000Z/2026-04-20T00:00:00.000Z",
            "analytics_page_size": "25",
            "analytics_language_filter": "en",
            "analytics_filter_json": '{"pageSize":25,"ignoredKey":"x"}',
        },
        follow_redirects=False,
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload is not None
    assert payload.get("ok") is True
    assert payload.get("request", {}).get("auth_mode") == "manual_bearer"
    assert payload.get("request", {}).get("applied_query_param_keys") == ["pageSize"]
    assert payload.get("request", {}).get("ignored_query_param_keys") == ["ignoredKey"]
    assert payload.get("result", {}).get("rows_count") == 2
    assert payload.get("result", {}).get("matching_rows_count") == 1
    assert payload.get("result", {}).get("unique_conversations") == 1
    assert payload.get("result", {}).get("language_filter_stats", {}).get(
        "excluded_mismatched_conversations"
    ) == 1
    assert payload.get("result", {}).get("next_page_available") is True
    assert _FakeAnalyticsClient.init_kwargs is not None
    assert _FakeAnalyticsClient.init_kwargs.get("manual_bearer_token") == "token-abc"
    assert _FakeAnalyticsClient.init_kwargs.get("auth_mode") == "manual_bearer"


def test_test_analytics_journey_api_route_client_credentials_mode(monkeypatch):
    class _FakeAnalyticsClient:
        init_kwargs = None

        def __init__(self, **kwargs):
            _FakeAnalyticsClient.init_kwargs = kwargs

        @staticmethod
        def sanitize_extra_query_params(extra_params):
            return {}, []

        def fetch_reporting_turns_page(self, **kwargs):
            return {"entities": [{"conversation": {"id": "conv-10"}, "userInput": "hello", "botPrompts": ["hi"]}]}

        @staticmethod
        def extract_rows(payload):
            return list(payload.get("entities") or [])

        @staticmethod
        def row_matches_language(row, language_filter):
            return True

        @classmethod
        def filter_rows_by_language(cls, rows, language_filter):
            return (
                list(rows),
                {
                    "language_filter": language_filter,
                    "eligible_conversations": 1,
                    "selected_conversations": 1,
                    "excluded_missing_language_conversations": 0,
                    "excluded_mismatched_conversations": 0,
                },
            )

        @staticmethod
        def extract_conversation_id(row):
            conversation = row.get("conversation") if isinstance(row, dict) else None
            return conversation.get("id") if isinstance(conversation, dict) else None

        @staticmethod
        def extract_next_uri(payload):
            return None

    monkeypatch.setattr("src.web_app.GenesysAnalyticsJourneyClient", _FakeAnalyticsClient)

    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()

    home = client.get("/")
    csrf_match = re.search(
        r'name="csrf_token" value="([^"]+)"',
        home.get_data(as_text=True),
    )
    assert csrf_match is not None

    response = client.post(
        "/run/analytics_journey/test/client_credentials",
        headers={"X-CSRF-Token": csrf_match.group(1)},
        data={
            "analytics_auth_mode": "manual_bearer",
            "analytics_bearer_token": "ignored-manual-token",
            "analytics_region": "usw2.pure.cloud",
            "analytics_gc_client_id": "client-id",
            "analytics_gc_client_secret": "client-secret",
            "analytics_bot_flow_id": "flow-123",
            "analytics_interval": "2026-04-19T00:00:00.000Z/2026-04-20T00:00:00.000Z",
            "analytics_page_size": "25",
        },
        follow_redirects=False,
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload is not None
    request_ctx = payload.get("request") or {}
    assert request_ctx.get("auth_mode") == "client_credentials"
    assert request_ctx.get("forced_auth_mode") == "client_credentials"
    turn_parsing = ((payload.get("result") or {}).get("turn_parsing")) or {}
    assert turn_parsing.get("rows_with_user_input") == 1
    assert turn_parsing.get("rows_with_bot_prompts") == 1
    assert _FakeAnalyticsClient.init_kwargs is not None
    assert _FakeAnalyticsClient.init_kwargs.get("auth_mode") == "client_credentials"


def test_test_analytics_journey_api_route_warns_when_language_filter_excludes_all_conversations(
    monkeypatch,
):
    class _FakeAnalyticsClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        @staticmethod
        def sanitize_extra_query_params(extra_params):
            return {}, []

        def fetch_reporting_turns_page(self, **kwargs):
            return {
                "entities": [
                    {
                        "conversation": {"id": "conv-fr"},
                        "language": "fr",
                        "userInput": "bonjour",
                    }
                ]
            }

        @staticmethod
        def extract_rows(payload):
            return list(payload.get("entities") or [])

        @staticmethod
        def extract_conversation_id(row):
            conversation = row.get("conversation") if isinstance(row, dict) else None
            return conversation.get("id") if isinstance(conversation, dict) else None

        @staticmethod
        def extract_next_uri(payload):
            return None

        @staticmethod
        def filter_rows_by_language(rows, language_filter):
            return (
                [],
                {
                    "language_filter": language_filter,
                    "eligible_conversations": 0,
                    "selected_conversations": 0,
                    "excluded_missing_language_conversations": 0,
                    "excluded_mismatched_conversations": 1,
                },
            )

    monkeypatch.setattr("src.web_app.GenesysAnalyticsJourneyClient", _FakeAnalyticsClient)

    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()

    home = client.get("/")
    csrf_match = re.search(
        r'name="csrf_token" value="([^"]+)"',
        home.get_data(as_text=True),
    )
    assert csrf_match is not None

    response = client.post(
        "/run/analytics_journey/test",
        headers={"X-CSRF-Token": csrf_match.group(1)},
        data={
            "analytics_auth_mode": "manual_bearer",
            "analytics_bearer_token": "token-abc",
            "analytics_region": "usw2.pure.cloud",
            "analytics_bot_flow_id": "flow-123",
            "analytics_interval": "2026-04-19T00:00:00.000Z/2026-04-20T00:00:00.000Z",
            "analytics_page_size": "25",
            "analytics_language_filter": "en",
        },
        follow_redirects=False,
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload is not None
    assert payload.get("warnings") == [
        "Rows were returned, but no complete conversations matched the selected language filter."
    ]
    assert payload.get("result", {}).get("matching_rows_count") == 0
    assert payload.get("result", {}).get("unique_conversations") == 0


def test_test_analytics_journey_api_route_requires_expected_fields():
    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()

    home = client.get("/")
    csrf_match = re.search(
        r'name="csrf_token" value="([^"]+)"',
        home.get_data(as_text=True),
    )
    assert csrf_match is not None

    response = client.post(
        "/run/analytics_journey/test",
        headers={"X-CSRF-Token": csrf_match.group(1)},
        data={
            "analytics_auth_mode": "manual_bearer",
            "analytics_region": "",
            "analytics_bearer_token": "",
            "analytics_bot_flow_id": "",
            "analytics_interval": "",
        },
        follow_redirects=False,
    )
    assert response.status_code == 400
    payload = response.get_json()
    assert payload is not None
    assert "missing" in payload
    missing = payload.get("missing") or []
    assert "analytics_region" in missing
    assert "analytics_bot_flow_id" in missing
    assert "analytics_interval" in missing
    assert "analytics_bearer_token" in missing


def test_test_analytics_journey_api_route_client_credentials_requires_credentials():
    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()

    home = client.get("/")
    csrf_match = re.search(
        r'name="csrf_token" value="([^"]+)"',
        home.get_data(as_text=True),
    )
    assert csrf_match is not None

    response = client.post(
        "/run/analytics_journey/test/client_credentials",
        headers={"X-CSRF-Token": csrf_match.group(1)},
        data={
            "analytics_region": "usw2.pure.cloud",
            "analytics_bot_flow_id": "flow-123",
            "analytics_interval": "2026-04-19T00:00:00.000Z/2026-04-20T00:00:00.000Z",
            "analytics_gc_client_id": "",
            "analytics_gc_client_secret": "",
        },
        follow_redirects=False,
    )
    assert response.status_code == 400
    payload = response.get_json()
    assert payload is not None
    missing = payload.get("missing") or []
    assert "analytics_gc_client_id" in missing
    assert "analytics_gc_client_secret" in missing


def test_test_analytics_journey_api_route_403_returns_permission_guidance(monkeypatch):
    from src.web_app import GenesysAnalyticsJourneyError

    class _FakeAnalyticsClient:
        def __init__(self, **kwargs):
            pass

        @staticmethod
        def sanitize_extra_query_params(extra_params):
            return {}, []

        def fetch_reporting_turns_page(self, **kwargs):
            raise GenesysAnalyticsJourneyError(
                "Request failed for /api/v2/analytics/botflows/flow/divisions/reportingturns: "
                "403 Client Error: Forbidden for url: https://api.usw2.pure.cloud/...",
                metadata={
                    "status_code": 403,
                    "correlation_id": "corr-123",
                    "path": "/api/v2/analytics/botflows/flow/divisions/reportingturns",
                    "method": "GET",
                    "response_body_excerpt": '{"message":"forbidden"}',
                },
            )

    monkeypatch.setattr("src.web_app.GenesysAnalyticsJourneyClient", _FakeAnalyticsClient)

    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()

    home = client.get("/")
    csrf_match = re.search(
        r'name="csrf_token" value="([^"]+)"',
        home.get_data(as_text=True),
    )
    assert csrf_match is not None

    response = client.post(
        "/run/analytics_journey/test",
        headers={"X-CSRF-Token": csrf_match.group(1)},
        data={
            "analytics_auth_mode": "manual_bearer",
            "analytics_bearer_token": "token-abc",
            "analytics_region": "usw2.pure.cloud",
            "analytics_bot_flow_id": "flow-123",
            "analytics_interval": "2026-04-19T00:00:00.000Z/2026-04-20T00:00:00.000Z",
            "analytics_page_size": "25",
        },
        follow_redirects=False,
    )

    assert response.status_code == 403
    payload = response.get_json()
    assert payload is not None
    assert payload.get("error_class") == "permission_or_division_access"
    assert payload.get("status_code") == 403
    assert "Access denied" in str(payload.get("user_message") or "")
    guidance = payload.get("guidance") or []
    assert isinstance(guidance, list)
    assert any("botFlowReportingTurn" in str(item) for item in guidance)
    assert any("OAuth > [your client] > Roles" in str(item) for item in guidance)
    context = payload.get("request_context") or {}
    assert context.get("auth_mode") == "manual_bearer"
    assert context.get("region") == "usw2.pure.cloud"
    upstream_debug = payload.get("upstream_debug") or {}
    assert upstream_debug.get("correlation_id") == "corr-123"
    assert upstream_debug.get("status_code") == 403


def test_test_analytics_journey_api_route_requires_csrf():
    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()

    response = client.post(
        "/run/analytics_journey/test",
        data={
            "analytics_auth_mode": "manual_bearer",
            "analytics_bearer_token": "token-123",
            "analytics_region": "usw2.pure.cloud",
            "analytics_bot_flow_id": "flow-123",
            "analytics_interval": "2026-04-19T00:00:00.000Z/2026-04-20T00:00:00.000Z",
        },
        follow_redirects=False,
    )
    assert response.status_code == 400


def test_run_analytics_journey_route_requires_bot_flow_id(monkeypatch):
    monkeypatch.setenv("GC_REGION", "usw2.pure.cloud")
    monkeypatch.setenv("GC_CLIENT_ID", "client-id")
    monkeypatch.setenv("GC_CLIENT_SECRET", "client-secret")
    monkeypatch.setenv("OLLAMA_MODEL", "llama3")
    _patch_web_judge_builder(monkeypatch)

    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()
    response = client.post(
        "/run/analytics_journey",
        data={
            "analytics_journey_enabled": "on",
            "analytics_interval": "2026-04-19T00:00:00.000Z/2026-04-20T00:00:00.000Z",
            "analytics_page_size": "10",
            "analytics_max_conversations": "10",
        },
        follow_redirects=True,
    )

    text = response.get_data(as_text=True)
    assert response.status_code == 200
    assert "Bot Flow ID is required for analytics journey runs." in text


def test_seed_url_route_success(monkeypatch):
    class _FakeResponse:
        def __init__(self, payload):
            import json

            self.encoding = "utf-8"
            self._body = json.dumps(payload).encode("utf-8")

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size=65536):
            for index in range(0, len(self._body), chunk_size):
                yield self._body[index : index + chunk_size]

    def _fake_get(url, timeout, stream):
        return _FakeResponse(
            {
                "conversations": [
                    {
                        "messages": [
                            {
                                "speaker": "customer",
                                "text": "I want to cancel my booking",
                            }
                        ]
                    }
                ]
            }
        )

    monkeypatch.setattr("src.transcript_url_importer.requests.get", _fake_get)

    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()
    response = client.post(
        "/seed/url",
        data={
            "language": "en",
            "seed_suite_name": "URL Seeded Suite",
            "seed_max_scenarios": "5",
            "transcript_url": "https://api-downloads.cac1.pure.cloud/transcript.json?token=abc",
        },
    )
    text = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Transcript Suite Preview" in text
    assert "URL Seeded Suite" in text
    assert "Source URL:" in text
    assert "token=abc" not in text


def test_seed_url_route_journey_strategy_generates_journey_suite(monkeypatch):
    class _FakeResponse:
        def __init__(self, payload):
            import json

            self.encoding = "utf-8"
            self._body = json.dumps(payload).encode("utf-8")

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size=65536):
            for index in range(0, len(self._body), chunk_size):
                yield self._body[index : index + chunk_size]

    def _fake_get(url, timeout, stream):
        return _FakeResponse(
            {
                "conversations": [
                    {
                        "conversationId": "11111111-1111-1111-1111-111111111111",
                        "participants": [
                            {"purpose": "customer"},
                        ],
                        "messages": [
                            {
                                "speaker": "customer",
                                "text": "I need to cancel my booking",
                            },
                            {
                                "speaker": "agent",
                                "text": "I can help with that.",
                            },
                        ],
                    }
                ]
            }
        )

    monkeypatch.setattr("src.transcript_url_importer.requests.get", _fake_get)
    _patch_web_judge_builder(
        monkeypatch,
        classify_result={
            "category": "flight_cancel",
            "confidence": 0.9,
            "explanation": "category match",
        },
    )

    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()
    response = client.post(
        "/seed/url",
        data={
            "language": "en",
            "seed_suite_name": "Journey URL Suite",
            "seed_max_scenarios": "10",
            "seed_strategy": "journey",
            "journey_category_strategy": "rules_first",
            "transcript_url": "https://api-downloads.cac1.pure.cloud/transcript.json?token=abc",
        },
    )
    text = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Transcript Suite Preview" in text
    assert "Journey URL Suite" in text
    assert "harness_mode: journey" in text
    assert "journey_validation:" in text
    assert "journey_category: flight_cancel" in text
    assert "token=abc" not in text


def test_seed_url_route_rejects_disallowed_domain():
    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()
    response = client.post(
        "/seed/url",
        data={
            "language": "en",
            "seed_suite_name": "URL Seeded Suite",
            "seed_max_scenarios": "5",
            "transcript_url": "https://example.org/transcript.json",
        },
    )
    text = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Could not seed suite from transcript URL" in text
    assert "host is not allowed" in text


def test_transcript_import_settings_route_saves_and_redirects():
    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()

    response = client.post(
        "/transcript/import/settings",
        data={
            "language": "en",
            "transcript_import_enabled": "on",
            "transcript_import_time": "03:15",
            "transcript_import_timezone": "America/New_York",
            "transcript_import_max_ids": "77",
            "transcript_import_filter_json": "{\"mediaType\":\"message\"}",
        },
        follow_redirects=False,
    )

    assert response.status_code == 302
    assert "/?home_tab=transcript&transcript_tab=automation" in response.headers.get(
        "Location", ""
    )


def test_transcript_import_settings_route_invalid_values_stays_in_automation_tab():
    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()

    response = client.post(
        "/transcript/import/settings",
        data={
            "language": "en",
            "transcript_import_enabled": "on",
            "transcript_import_time": "25:99",
            "transcript_import_timezone": "America/New_York",
            "transcript_import_max_ids": "20",
            "transcript_import_filter_json": "{}",
        },
        follow_redirects=True,
    )
    text = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Could not save automation settings" in text
    assert 'data-initial-home-tab=\"transcript\"' in text
    assert 'data-initial-transcript-tab=\"automation\"' in text


def test_seed_preview_includes_extraction_summary_and_warnings():
    app = create_app()
    app.config["TESTING"] = True

    transcript = (
        "Customer: I need help with my booking\n"
        "Customer: conversation_id: 123\n"
        "Customer: I need help with my booking\n"
        "Agent: Hi there\n"
    )

    client = app.test_client()
    response = client.post(
        "/seed",
        data={
            "seed_suite_name": "My Transcript Suite",
            "seed_max_scenarios": "10",
            "transcript_file": (io.BytesIO(transcript.encode("utf-8")), "sample.txt"),
        },
        content_type="multipart/form-data",
    )
    text = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Extraction Summary" in text
    assert "Utterances Found" in text
    assert "Scenarios Generated" in text
    assert "Messages Skipped" in text
    assert "Transcript Suite Preview - Regression Test Harness" in text
    assert 'id="theme-toggle"' in text
    assert "rth_theme_preference" in text


def test_seed_accepts_language_override_for_localized_suite():
    app = create_app()
    app.config["TESTING"] = True

    transcript = "Client: Je veux annuler ma reservation\nAgent: D'accord"
    client = app.test_client()
    response = client.post(
        "/seed",
        data={
            "language": "fr-CA",
            "seed_suite_name": "Suite FR",
            "seed_max_scenarios": "10",
            "transcript_file": (io.BytesIO(transcript.encode("utf-8")), "sample.txt"),
        },
        content_type="multipart/form-data",
    )
    text = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "language: fr-CA" in text


def test_seed_import_ids_paste_generates_preview(monkeypatch, tmp_path):
    monkeypatch.setenv("GC_REGION", "usw2.pure.cloud")
    monkeypatch.setenv("GC_CLIENT_ID", "client-id")
    monkeypatch.setenv("GC_CLIENT_SECRET", "client-secret")
    monkeypatch.setenv(
        "GC_TESTER_TRANSCRIPT_IMPORT_DIR",
        str(tmp_path / "imports"),
    )

    def _fake_import(self, conversation_ids):
        assert conversation_ids == ["11111111-2222-4333-8444-555555555555"]
        return {
            "fetched": [
                {
                    "conversation_id": "11111111-2222-4333-8444-555555555555",
                    "transcript": {
                        "conversation_id": "11111111-2222-4333-8444-555555555555",
                        "messages": [
                            {
                                "role": "customer",
                                "text": "I need help with my booking",
                                "timestamp": "2026-04-19T01:00:00Z",
                            }
                        ],
                    },
                    "raw_payload": {"id": "11111111-2222-4333-8444-555555555555"},
                }
            ],
            "failed": [],
            "skipped": [],
        }

    monkeypatch.setattr(
        "src.web_app.GenesysTranscriptImportClient.import_transcripts_by_ids",
        _fake_import,
    )

    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()
    response = client.post(
        "/seed/import",
        data={
            "id_source_mode": "ids_paste",
            "conversation_ids_paste": "11111111-2222-4333-8444-555555555555",
            "seed_suite_name": "Imported Suite",
            "seed_max_scenarios": "10",
            "transcript_import_max_ids": "10",
            "transcript_import_filter_json": "{}",
        },
        follow_redirects=True,
    )
    text = response.get_data(as_text=True)
    assert response.status_code == 200
    assert "Import Summary" in text
    assert "Imported Suite" in text
    assert "Scenarios Generated" in text


def test_seed_import_failure_manifest_download(monkeypatch, tmp_path):
    monkeypatch.setenv("GC_REGION", "usw2.pure.cloud")
    monkeypatch.setenv("GC_CLIENT_ID", "client-id")
    monkeypatch.setenv("GC_CLIENT_SECRET", "client-secret")
    monkeypatch.setenv(
        "GC_TESTER_TRANSCRIPT_IMPORT_DIR",
        str(tmp_path / "imports"),
    )

    def _fake_import(self, conversation_ids):
        return {
            "fetched": [
                {
                    "conversation_id": "11111111-2222-4333-8444-555555555555",
                    "transcript": {
                        "conversation_id": "11111111-2222-4333-8444-555555555555",
                        "messages": [
                            {
                                "role": "customer",
                                "text": "I need help with my booking",
                                "timestamp": "2026-04-19T01:00:00Z",
                            }
                        ],
                    },
                    "raw_payload": {"id": "11111111-2222-4333-8444-555555555555"},
                }
            ],
            "failed": [
                {
                    "conversation_id": "66666666-7777-4888-9999-aaaaaaaaaaaa",
                    "reason": "not found",
                }
            ],
            "skipped": [],
        }

    monkeypatch.setattr(
        "src.web_app.GenesysTranscriptImportClient.import_transcripts_by_ids",
        _fake_import,
    )

    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()
    response = client.post(
        "/seed/import",
        data={
            "id_source_mode": "ids_paste",
            "conversation_ids_paste": (
                "11111111-2222-4333-8444-555555555555\n"
                "66666666-7777-4888-9999-aaaaaaaaaaaa"
            ),
            "seed_suite_name": "Imported Suite",
            "seed_max_scenarios": "10",
            "transcript_import_max_ids": "10",
            "transcript_import_filter_json": "{}",
        },
        follow_redirects=True,
    )
    text = response.get_data(as_text=True)
    assert response.status_code == 200
    assert "Download Failure Manifest" in text
    marker = "/seed/import/failures?run_id="
    start = text.find(marker)
    assert start != -1
    run_id = text[start + len(marker) :].split('"', 1)[0]
    download = client.get(f"/seed/import/failures?run_id={run_id}")
    payload = download.get_data(as_text=True)
    assert download.status_code == 200
    assert '"reason": "not found"' in payload


def test_results_page_includes_theme_toggle_and_theme_storage_hook():
    app = create_app()
    app.config["TESTING"] = True
    app.config["latest_report"] = _sample_report()

    client = app.test_client()
    response = client.get("/results")
    text = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Test Results - Regression Test Harness" in text
    assert 'id="theme-toggle"' in text
    assert "rth_theme_preference" in text


def test_results_page_localizes_labels_from_evaluation_results_language():
    app = create_app()
    app.config["TESTING"] = True
    app.config["latest_report"] = _sample_report()
    app.config["last_run_config"] = AppConfig(
        gc_region="usw2.pure.cloud",
        gc_deployment_id="dep-id",
        ollama_model="llama3",
        language="en",
        evaluation_results_language="es",
    )

    client = app.test_client()
    response = client.get("/results")
    text = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Resultados de Pruebas" in text
    assert "Intentos Totales" in text


def test_results_page_shows_na_for_non_applicable_analytics_gates():
    app = create_app()
    app.config["TESTING"] = True
    report = _sample_report()
    report.scenario_results[0].attempt_results[0].analytics_journey_result = AnalyticsJourneyResult(
        conversation_id="22222222-2222-2222-2222-222222222222",
        category="flight_change",
        expected_auth_behavior="optional",
        auth_gate=None,
        auth_gate_applicable=False,
        expected_transfer_behavior="optional",
        transfer_gate=None,
        transfer_gate_applicable=False,
        category_gate=True,
        journey_quality_gate=True,
    )
    app.config["latest_report"] = report

    client = app.test_client()
    response = client.get("/results")
    text = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Auth Gate: N/A" in text
    assert "Transfer Gate: N/A" in text


def test_web_auth_redirects_unauthenticated_requests(monkeypatch):
    monkeypatch.setenv("GC_TESTER_WEB_AUTH_ENABLED", "true")
    monkeypatch.setenv("GC_TESTER_WEB_AUTH_USERNAME", "operator")
    monkeypatch.setenv("GC_TESTER_WEB_AUTH_PASSWORD", "secret")

    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()

    response = client.get("/", follow_redirects=False)
    assert response.status_code in {301, 302}
    assert "/login" in response.headers.get("Location", "")

    login_page = client.get("/login")
    assert login_page.status_code == 200
    text = login_page.get_data(as_text=True)
    assert "Sign In" in text
    assert 'name="csrf_token"' in text


def test_web_auth_login_and_csrf_guard(monkeypatch):
    monkeypatch.setenv("GC_TESTER_WEB_AUTH_ENABLED", "true")
    monkeypatch.setenv("GC_TESTER_WEB_AUTH_USERNAME", "operator")
    monkeypatch.setenv("GC_TESTER_WEB_AUTH_PASSWORD", "secret")

    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()

    login_page = client.get("/login")
    login_text = login_page.get_data(as_text=True)
    csrf_match = re.search(r'name="csrf_token" value="([^"]+)"', login_text)
    assert csrf_match is not None

    login_response = client.post(
        "/login",
        data={
            "username": "operator",
            "password": "secret",
            "csrf_token": csrf_match.group(1),
            "next": "/",
        },
        follow_redirects=False,
    )
    assert login_response.status_code in {301, 302}

    home_response = client.get("/")
    assert home_response.status_code == 200
    home_text = home_response.get_data(as_text=True)
    home_csrf_match = re.search(r'name="csrf_token" value="([^"]+)"', home_text)
    assert home_csrf_match is not None

    missing_csrf = client.post("/run/stop", follow_redirects=False)
    assert missing_csrf.status_code == 400
    assert "CSRF token" in missing_csrf.get_data(as_text=True)

    with_csrf = client.post(
        "/run/stop",
        headers={"X-CSRF-Token": home_csrf_match.group(1)},
        follow_redirects=False,
    )
    assert with_csrf.status_code in {301, 302}


def test_stop_route_force_finalizes_active_run_when_worker_hangs():
    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()

    home = client.get("/")
    csrf_match = re.search(
        r'name="csrf_token" value="([^"]+)"',
        home.get_data(as_text=True),
    )
    assert csrf_match is not None
    csrf_token = csrf_match.group(1)

    progress = ProgressEmitter()
    progress.emit(
        ProgressEvent(
            event_type=ProgressEventType.SUITE_STARTED,
            suite_name="Kill Switch Suite",
            message="Starting test suite: Kill Switch Suite",
            planned_attempts=10,
            completed_attempts=0,
        )
    )

    class _HangingThread:
        def __init__(self):
            self.join_calls: list[float] = []

        def is_alive(self):
            return True

        def join(self, timeout=None):
            self.join_calls.append(float(timeout or 0.0))

    hanging_thread = _HangingThread()
    control = ActiveRunControl(run_id="run-kill-switch")
    control.thread = hanging_thread

    app.config["progress_emitter"] = progress
    app.config["run_active"] = True
    app.config["stop_requested"] = False
    app.config["active_run_control"] = control
    app.config["active_run_id"] = control.run_id
    app.config["stop_event"] = control.stop_event

    response = client.post(
        "/run/stop",
        headers={"X-CSRF-Token": csrf_token},
        follow_redirects=False,
    )
    assert response.status_code in {301, 302}
    assert app.config["run_active"] is False
    assert app.config["active_run_id"] is None

    report = app.config.get("latest_report")
    assert isinstance(report, TestReport)
    assert report.stopped_by_user is True
    assert report.force_finalized is True
    assert report.stop_mode == "immediate"
    assert app.config.get("stop_requested") is False


def test_stop_route_json_kill_switch_response():
    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()

    home = client.get("/")
    csrf_match = re.search(
        r'name="csrf_token" value="([^"]+)"',
        home.get_data(as_text=True),
    )
    assert csrf_match is not None
    csrf_token = csrf_match.group(1)

    progress = ProgressEmitter()
    progress.emit(
        ProgressEvent(
            event_type=ProgressEventType.SUITE_STARTED,
            suite_name="Kill Switch JSON Suite",
            message="Starting test suite: Kill Switch JSON Suite",
            planned_attempts=3,
            completed_attempts=0,
        )
    )

    class _HangingThread:
        def is_alive(self):
            return True

    control = ActiveRunControl(run_id="run-kill-switch-json")
    control.thread = _HangingThread()

    app.config["progress_emitter"] = progress
    app.config["run_active"] = True
    app.config["stop_requested"] = False
    app.config["active_run_control"] = control
    app.config["active_run_id"] = control.run_id
    app.config["stop_event"] = control.stop_event

    response = client.post(
        "/run/stop",
        headers={
            "X-CSRF-Token": csrf_token,
            "Accept": "application/json",
        },
        follow_redirects=False,
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload is not None
    assert payload.get("stopped") is True
    assert payload.get("force_finalized") is True
    assert payload.get("run_active") is False
    assert payload.get("stop_mode") == "immediate"
