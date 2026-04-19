"""Integration tests for web app result export routes."""

from datetime import datetime, timezone

from src.models import AttemptResult, Message, MessageRole, ScenarioResult, TestReport
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
    assert "Current vs Previous Same Suite" in text
    assert "No previous same-suite run found yet." in text


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
    assert "Current vs Previous Same Suite" in text
    assert "No previous same-suite run found yet." not in text
    assert baseline_entry["timestamp"] in text
