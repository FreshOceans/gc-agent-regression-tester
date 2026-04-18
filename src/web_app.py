"""Flask web application for the GC Agent Regression Tester.

Provides a web UI for uploading test suites, triggering test execution,
viewing results, and streaming progress via SSE.
"""

import asyncio
import json
import os
import queue
import threading
from datetime import datetime, timezone
from typing import Optional

from flask import (
    Flask,
    Response,
    flash,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)
from pydantic import ValidationError

from .app_config import load_app_config, merge_config, validate_required_config
from .config_loader import load_test_suite_from_string, validate_test_suite
from .judge_llm import JudgeLLMClient, JudgeLLMError
from .models import AppConfig, ProgressEventType, TestReport
from .orchestrator import TestOrchestrator
from .progress import ProgressEmitter
from .report import (
    export_csv,
    export_json,
    export_junit_xml,
    export_report_bundle_zip,
    export_transcripts_zip,
)


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(
        __name__,
        template_folder=os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "templates"
        ),
    )
    app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")

    # App state
    app.config["latest_report"]: Optional[TestReport] = None
    app.config["progress_emitter"] = ProgressEmitter()
    app.config["run_active"] = False
    app.config["stop_event"] = threading.Event()
    app.config["stop_requested"] = False
    app.config["last_run_config"]: Optional[AppConfig] = None
    app.config["last_run_suite"] = None

    def build_partial_report_from_history() -> Optional[TestReport]:
        """Build a best-effort partial report from progress history."""
        progress_emitter = app.config.get("progress_emitter")
        if not isinstance(progress_emitter, ProgressEmitter):
            return None

        history = progress_emitter.get_history()
        if not history:
            return None

        threshold = 0.8
        last_config = app.config.get("last_run_config")
        if isinstance(last_config, AppConfig):
            threshold = float(last_config.success_threshold)

        suite_name = "In-progress suite"
        started_at: Optional[datetime] = None
        for event in history:
            if event.event_type == ProgressEventType.SUITE_STARTED:
                started_at = event.emitted_at
                if event.suite_name:
                    suite_name = event.suite_name
                elif event.message.startswith("Starting test suite: "):
                    suite_name = event.message.replace("Starting test suite: ", "", 1)
                break

        if started_at is None:
            started_at = datetime.now(timezone.utc)

        scenario_order: list[str] = []
        scenario_attempts: dict[str, list] = {}
        for event in history:
            if (
                event.event_type == ProgressEventType.ATTEMPT_COMPLETED
                and event.scenario_name
                and event.attempt_result is not None
            ):
                if event.scenario_name not in scenario_attempts:
                    scenario_attempts[event.scenario_name] = []
                    scenario_order.append(event.scenario_name)
                scenario_attempts[event.scenario_name].append(event.attempt_result)

        if not scenario_attempts:
            return None

        scenario_results = []
        for scenario_name in scenario_order:
            attempts = scenario_attempts[scenario_name]
            attempts_count = len(attempts)
            successes = sum(1 for attempt in attempts if attempt.success)
            timeouts = sum(1 for attempt in attempts if attempt.timed_out)
            failures = attempts_count - successes
            success_rate = successes / attempts_count if attempts_count else 0.0
            is_regression = success_rate < threshold
            scenario_results.append(
                {
                    "scenario_name": scenario_name,
                    "attempts": attempts_count,
                    "successes": successes,
                    "failures": failures,
                    "timeouts": timeouts,
                    "success_rate": success_rate,
                    "is_regression": is_regression,
                    "attempt_results": attempts,
                }
            )

        overall_attempts = sum(item["attempts"] for item in scenario_results)
        overall_successes = sum(item["successes"] for item in scenario_results)
        overall_failures = sum(item["failures"] for item in scenario_results)
        overall_timeouts = sum(item["timeouts"] for item in scenario_results)
        overall_success_rate = (
            overall_successes / overall_attempts if overall_attempts else 0.0
        )
        has_regressions = any(item["is_regression"] for item in scenario_results)
        duration_seconds = max(
            0.0,
            (datetime.now(timezone.utc) - started_at).total_seconds(),
        )

        return TestReport(
            suite_name=suite_name,
            timestamp=datetime.now(timezone.utc),
            duration_seconds=duration_seconds,
            scenario_results=scenario_results,
            overall_attempts=overall_attempts,
            overall_successes=overall_successes,
            overall_failures=overall_failures,
            overall_timeouts=overall_timeouts,
            overall_success_rate=overall_success_rate,
            has_regressions=has_regressions,
            regression_threshold=threshold,
        )

    def start_background_run(merged_config: AppConfig, test_suite) -> None:
        """Start a test run in a background thread with fresh run state."""
        progress_emitter = ProgressEmitter()
        app.config["progress_emitter"] = progress_emitter
        app.config["latest_report"] = None
        app.config["run_active"] = True
        app.config["stop_event"] = threading.Event()
        app.config["stop_requested"] = False

        def run_tests():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                orchestrator = TestOrchestrator(
                    config=merged_config,
                    progress_emitter=progress_emitter,
                    stop_event=app.config["stop_event"],
                )
                report = loop.run_until_complete(
                    orchestrator.run_suite(test_suite)
                )
                app.config["latest_report"] = report
            finally:
                app.config["run_active"] = False
                loop.close()

        thread = threading.Thread(target=run_tests, daemon=True)
        thread.start()

    @app.route("/")
    def home():
        """Home page with config inputs and file upload."""
        base_config = load_app_config()
        return render_template(
            "home.html",
            config=base_config,
            errors=None,
        )

    @app.route("/run", methods=["POST"])
    def run():
        """Trigger test execution from form submission."""
        if app.config.get("run_active", False):
            return redirect(url_for("results"))

        base_config = load_app_config()

        # Read form fields
        deployment_id = request.form.get("deployment_id", "").strip()
        region = request.form.get("region", "").strip()
        ollama_model = request.form.get("ollama_model", "").strip()
        max_turns = request.form.get("max_turns", "").strip()
        gc_client_id = request.form.get("gc_client_id", "").strip()
        gc_client_secret = request.form.get("gc_client_secret", "").strip()
        intent_attribute_name = request.form.get("intent_attribute_name", "").strip()
        debug_capture_frames = request.form.get("debug_capture_frames") is not None
        debug_capture_frame_limit = request.form.get("debug_capture_frame_limit", "").strip()

        # Read uploaded file
        uploaded_file = request.files.get("test_suite_file")
        if not uploaded_file or uploaded_file.filename == "":
            return render_template(
                "home.html",
                config=base_config,
                errors=["Please upload a test suite file (JSON or YAML)."],
            )

        # Determine format from filename
        filename = uploaded_file.filename.lower()
        if filename.endswith(".json"):
            fmt = "json"
        elif filename.endswith((".yaml", ".yml")):
            fmt = "yaml"
        else:
            return render_template(
                "home.html",
                config=base_config,
                errors=["Unsupported file format. Use .json, .yaml, or .yml"],
            )

        # Read and validate file content
        try:
            content = uploaded_file.read().decode("utf-8")
        except UnicodeDecodeError:
            return render_template(
                "home.html",
                config=base_config,
                errors=["File must be valid UTF-8 text."],
            )

        try:
            test_suite = load_test_suite_from_string(content, fmt)
        except (ValueError, ValidationError) as e:
            error_msg = str(e)
            return render_template(
                "home.html",
                config=base_config,
                errors=[f"Invalid test suite: {error_msg}"],
            )

        # Merge web overrides with base config
        web_overrides = {}
        if deployment_id:
            web_overrides["gc_deployment_id"] = deployment_id
        if region:
            web_overrides["gc_region"] = region
        if ollama_model:
            web_overrides["ollama_model"] = ollama_model
        if max_turns:
            web_overrides["max_turns"] = max_turns
        if gc_client_id:
            web_overrides["gc_client_id"] = gc_client_id
        if gc_client_secret:
            web_overrides["gc_client_secret"] = gc_client_secret
        if intent_attribute_name:
            web_overrides["intent_attribute_name"] = intent_attribute_name
        web_overrides["debug_capture_frames"] = debug_capture_frames
        if debug_capture_frame_limit:
            web_overrides["debug_capture_frame_limit"] = debug_capture_frame_limit

        merged_config = merge_config(base_config, web_overrides)

        # Validate required config
        missing = validate_required_config(merged_config)
        if missing:
            errors = [
                f"Missing required configuration: {', '.join(missing)}"
            ]
            return render_template(
                "home.html",
                config=base_config,
                errors=errors,
            )

        # Validate Ollama connectivity and model before starting long test runs.
        try:
            JudgeLLMClient(
                base_url=merged_config.ollama_base_url,
                model=merged_config.ollama_model or "",
                timeout=merged_config.response_timeout,
            ).verify_connection()
        except JudgeLLMError as e:
            return render_template(
                "home.html",
                config=base_config,
                errors=[str(e)],
            )

        app.config["last_run_config"] = merged_config.model_copy(deep=True)
        app.config["last_run_suite"] = test_suite.model_copy(deep=True)
        start_background_run(merged_config, test_suite)

        return redirect(url_for("results"))

    @app.route("/run/rerun", methods=["POST"])
    def rerun():
        """Re-run the latest uploaded test suite with the last merged config."""
        if app.config.get("run_active", False):
            flash("A run is already active.")
            return redirect(url_for("results"))

        last_config = app.config.get("last_run_config")
        last_suite = app.config.get("last_run_suite")
        if last_config is None or last_suite is None:
            flash("No previous run configuration found. Start a run from the home page first.")
            return redirect(url_for("results"))

        merged_config = last_config.model_copy(deep=True)
        test_suite = last_suite.model_copy(deep=True)

        try:
            JudgeLLMClient(
                base_url=merged_config.ollama_base_url,
                model=merged_config.ollama_model or "",
                timeout=merged_config.response_timeout,
            ).verify_connection()
        except JudgeLLMError as e:
            flash(str(e))
            return redirect(url_for("results"))

        start_background_run(merged_config, test_suite)
        flash(f"Re-running suite: {test_suite.name}")
        return redirect(url_for("results"))

    @app.route("/run/stop", methods=["POST"])
    def stop_run():
        """Request the active run to stop gracefully."""
        if app.config.get("run_active", False):
            app.config["stop_requested"] = True
            stop_event = app.config.get("stop_event")
            if isinstance(stop_event, threading.Event):
                stop_event.set()
            flash("Stop requested. Finishing current attempt, then stopping.")
        else:
            flash("No active run to stop.")

        return redirect(url_for("results"))

    @app.route("/results")
    def results():
        """Results page displaying the latest TestReport."""
        report = app.config.get("latest_report")
        partial_report = False
        if report is None:
            report = build_partial_report_from_history()
            partial_report = report is not None
        run_active = app.config.get("run_active", False)
        stop_requested = app.config.get("stop_requested", False)
        progress_emitter = app.config.get("progress_emitter")
        progress_history = []
        if isinstance(progress_emitter, ProgressEmitter):
            progress_history = [
                event.model_dump(mode="json")
                for event in progress_emitter.get_history(limit=200)
                if event.event_type.value in {"attempt_started", "attempt_status", "attempt_completed"}
            ]
        has_rerun = (
            app.config.get("last_run_config") is not None
            and app.config.get("last_run_suite") is not None
        )
        return render_template(
            "results.html",
            report=report,
            partial_report=partial_report,
            run_active=run_active,
            stop_requested=stop_requested,
            has_rerun=has_rerun,
            progress_history=progress_history,
        )

    @app.route("/results/export")
    def export():
        """Download report in supported export formats."""
        report = app.config.get("latest_report")
        if report is None:
            report = build_partial_report_from_history()
        if report is None:
            return redirect(url_for("results"))

        fmt = request.args.get("format", "json").lower()

        if fmt == "csv":
            content = export_csv(report)
            return Response(
                content,
                mimetype="text/csv",
                headers={
                    "Content-Disposition": "attachment; filename=report.csv"
                },
            )
        elif fmt == "junit":
            content = export_junit_xml(report)
            return Response(
                content,
                mimetype="application/xml",
                headers={
                    "Content-Disposition": "attachment; filename=report.junit.xml"
                },
            )
        elif fmt == "transcripts":
            content = export_transcripts_zip(report)
            return Response(
                content,
                mimetype="application/zip",
                headers={
                    "Content-Disposition": "attachment; filename=report-transcripts.zip"
                },
            )
        elif fmt == "bundle":
            content = export_report_bundle_zip(report)
            return Response(
                content,
                mimetype="application/zip",
                headers={
                    "Content-Disposition": "attachment; filename=report-bundle.zip"
                },
            )
        else:
            content = export_json(report)
            return Response(
                content,
                mimetype="application/json",
                headers={
                    "Content-Disposition": "attachment; filename=report.json"
                },
            )

    @app.route("/progress")
    def progress():
        """SSE endpoint streaming ProgressEvent data to the browser."""
        emitter: ProgressEmitter = app.config["progress_emitter"]

        def event_stream():
            q = emitter.subscribe()
            try:
                while True:
                    try:
                        event = q.get(timeout=30)
                        data = event.model_dump(mode="json")
                        yield f"data: {json.dumps(data)}\n\n"
                        # Stop streaming after suite_completed
                        if event.event_type.value == "suite_completed":
                            break
                    except queue.Empty:
                        # Send keepalive comment
                        yield ": keepalive\n\n"
            finally:
                emitter.unsubscribe(q)

        return Response(
            event_stream(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    return app


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=5000, debug=True)
