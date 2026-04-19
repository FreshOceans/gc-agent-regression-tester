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
    assert "No previous same-suite run found yet." not in text
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
    assert "home-tab-transcript" in text
    assert "Harness Configuration" in text
    assert "Transcript Suite" in text
    assert "Transcript Suite Name" in text
    assert "Seed From Uploaded Transcript" in text
    assert "Conversation IDs" in text
    assert "Transcript URL" in text
    assert "Automation" in text
    assert "Seed From Transcript URL" in text
    assert "transcript_url" in text
    assert "action=\"/seed/url\"" in text
    assert "action=\"/transcript/import/settings\"" in text
    assert 'id="global-language-select"' in text
    assert "Run &amp; Transcript Language" in text
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

    upload_panel = re.search(r'<div id="transcript-subtab-upload"[^>]*>', text)
    assert upload_panel is not None
    assert "hidden" not in upload_panel.group(0)


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
