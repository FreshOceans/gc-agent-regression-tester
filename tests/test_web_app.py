"""Integration tests for web app result export routes."""

from datetime import datetime, timezone
import io
import re

from src.models import (
    AppConfig,
    AttemptResult,
    Message,
    MessageRole,
    ScenarioResult,
    TestReport,
    TestScenario,
    TestSuite,
)
from src.run_history import RunHistoryStore
from src.web_app import create_app


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


def test_results_page_includes_collapsed_legend_and_responsive_export_actions():
    app = create_app()
    app.config["TESTING"] = True
    app.config["latest_report"] = _sample_report()

    client = app.test_client()
    response = client.get("/results")
    text = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Metrics Legend &amp; Definitions" in text
    assert "Tool Effectiveness" in text
    assert "class=\"export-actions\"" in text
    assert "class=\"export-link export-link-pdf\"" in text
    assert "dashboard-png-export-btn" in text
    assert "js/dashboard_capture.js" in text


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
    monkeypatch.setattr("src.web_app.JudgeLLMClient.verify_connection", lambda self: None)

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
    monkeypatch.setattr("src.web_app.JudgeLLMClient.verify_connection", lambda self: None)

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
    assert "Harness Configuration" in text
    assert "Analytics Journey Regression" in text
    assert "Transcript Suite" in text
    assert "analytics-journey-form" in text
    assert "analytics_region" in text
    assert "analytics_gc_client_id" in text
    assert "analytics_gc_client_secret" in text
    assert "analytics_ollama_model" in text
    assert "analytics_bot_flow_id" in text
    assert "analytics_interval" in text
    assert "analytics-interval-group" in text
    assert "analytics-interval-trigger" not in text
    assert "Choose a local date/time range; we auto-convert to canonical UTC ISO-8601 interval format for Genesys." in text
    assert "Rows requested per Analytics API page (1-250)." in text
    assert "Hard cap of conversations evaluated in this run." in text
    assert 'id="analytics_page_size"' in text
    assert 'max="250"' in text
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
    assert 'id="legend-test_suite_file"' in text
    assert 'id="legend-gc_client_id"' in text
    assert 'id="legend-gc_client_secret"' in text
    assert 'id="legend-intent_attribute_name"' in text
    assert 'id="legend-debug_capture_frames"' in text
    assert 'id="legend-debug_capture_frame_limit"' in text
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
    monkeypatch.setattr("src.web_app.JudgeLLMClient.verify_connection", lambda self: None)

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
    monkeypatch.setattr("src.web_app.JudgeLLMClient.verify_connection", lambda self: None)

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


def test_run_analytics_journey_route_reports_missing_required_config(monkeypatch):
    monkeypatch.delenv("GC_REGION", raising=False)
    monkeypatch.delenv("GC_CLIENT_ID", raising=False)
    monkeypatch.delenv("GC_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.setattr("src.web_app.JudgeLLMClient.verify_connection", lambda self: None)

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
    assert "Missing required configuration for analytics journey: gc_region, gc_client_id, gc_client_secret, ollama_model" in text


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
    monkeypatch.setattr(
        "src.web_app.JudgeLLMClient.classify_primary_category",
        lambda self, first_message, categories, language_code="en": {
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
