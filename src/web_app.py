"""Flask web application for the GC Agent Regression Tester.

Provides a web UI for uploading test suites, triggering test execution,
viewing results, and streaming progress via SSE.
"""

import asyncio
import json
import os
import queue
import threading
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
from .judge_llm import JudgeLLMClient
from .models import AppConfig, TestReport
from .orchestrator import TestOrchestrator
from .progress import ProgressEmitter
from .report import export_csv, export_json, export_junit_xml, export_transcripts_zip


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
        base_config = load_app_config()

        # Read form fields
        deployment_id = request.form.get("deployment_id", "").strip()
        region = request.form.get("region", "").strip()
        ollama_model = request.form.get("ollama_model", "").strip()
        origin_url = request.form.get("origin_url", "").strip()
        max_turns = request.form.get("max_turns", "").strip()

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
        if origin_url:
            web_overrides["gc_origin"] = origin_url
        if max_turns:
            web_overrides["max_turns"] = max_turns

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

        # Create a fresh progress emitter for this run
        progress_emitter = ProgressEmitter()
        app.config["progress_emitter"] = progress_emitter
        app.config["latest_report"] = None
        app.config["run_active"] = True

        # Start test execution in a background thread
        def run_tests():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                orchestrator = TestOrchestrator(
                    config=merged_config,
                    progress_emitter=progress_emitter,
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

        return redirect(url_for("results"))

    @app.route("/results")
    def results():
        """Results page displaying the latest TestReport."""
        report = app.config.get("latest_report")
        run_active = app.config.get("run_active", False)
        return render_template("results.html", report=report, run_active=run_active)

    @app.route("/results/export")
    def export():
        """Download report in CSV, JSON, JUnit XML, or transcript ZIP format."""
        report = app.config.get("latest_report")
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
