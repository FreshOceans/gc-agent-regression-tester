"""Flask web application for the GC Agent Regression Tester.

Provides a web UI for uploading test suites, triggering test execution,
viewing results, and streaming progress via SSE.
"""

import asyncio
import io
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
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)
from pydantic import ValidationError

from .app_config import load_app_config, merge_config, validate_required_config
from .config_loader import (
    load_test_suite_from_string,
    print_test_suite,
    validate_test_suite,
)
from .judge_llm import JudgeLLMClient, JudgeLLMError
from .journey_mode import (
    CATEGORY_STRATEGY_LLM_FIRST,
    CATEGORY_STRATEGY_RULES_FIRST,
    HARNESS_JOURNEY,
    HARNESS_STANDARD,
    load_category_overrides,
    normalize_category_strategy,
    normalize_harness_mode,
)
from .journey_regression import (
    extract_journey_seed_candidates,
    resolve_category_with_strategy,
    resolve_primary_categories,
)
from .language_profiles import (
    EVALUATION_RESULTS_LANGUAGE_OPTIONS,
    get_language_profile,
    SUPPORTED_LANGUAGE_OPTIONS,
    normalize_evaluation_results_language,
    normalize_language_code,
    resolve_effective_evaluation_results_language,
    resolve_effective_language,
)
from .models import (
    AppConfig,
    JourneyValidationConfig,
    PrimaryCategoryConfig,
    ProgressEventType,
    TestScenario,
    TestSuite,
    TestReport,
)
from .orchestrator import TestOrchestrator
from .progress import ProgressEmitter
from .run_history import RunHistoryStore
from .dashboard_metrics import build_dashboard_metrics, summarize_entry_for_compare
from .dashboard_pdf import export_dashboard_pdf
from .duration_format import format_duration, format_duration_delta
from .results_i18n import get_results_i18n
from .genesys_transcript_import_client import (
    GenesysTranscriptImportClient,
    GenesysTranscriptImportError,
)
from .report import (
    export_csv,
    export_json,
    export_junit_xml,
    export_report_bundle_zip,
    export_transcripts_zip,
)
from .transcript_seeder import (
    TranscriptSeedError,
    seed_test_suite_from_transcript_with_diagnostics,
)
from .transcript_import_scheduler import TranscriptImportScheduler
from .transcript_import_store import TranscriptImportStore
from .transcript_importer import (
    build_last_24h_interval,
    build_transcript_seeder_payload,
    dedupe_and_cap_conversation_ids,
    parse_conversation_ids_from_file,
    parse_conversation_ids_from_paste,
    parse_filter_json,
)
from .transcript_url_importer import (
    TranscriptUrlImportError,
    TranscriptUrlImportService,
    redact_url_for_display,
)

ATTEMPT_CHUNK_SIZE = 20
BASELINE_OPTIONS_LIMIT = 30


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(
        __name__,
        template_folder=os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "templates"
        ),
    )
    app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")
    app.jinja_env.globals["format_duration"] = format_duration
    app.jinja_env.globals["format_duration_delta"] = format_duration_delta

    # App state
    app.config["latest_report"]: Optional[TestReport] = None
    app.config["progress_emitter"] = ProgressEmitter()
    app.config["run_active"] = False
    app.config["stop_event"] = threading.Event()
    app.config["stop_requested"] = False
    app.config["last_run_config"]: Optional[AppConfig] = None
    app.config["last_run_suite"] = None
    base_config = load_app_config()
    app.config["history_store"] = RunHistoryStore(
        history_dir=base_config.history_dir,
        max_runs=base_config.history_max_runs,
        full_json_runs=base_config.history_full_json_runs,
        gzip_runs=base_config.history_gzip_runs,
    )
    app.config["latest_run_history_entry"] = None
    app.config["transcript_import_active"] = False

    def build_transcript_import_settings(config: AppConfig) -> dict:
        return {
            "enabled": bool(config.transcript_import_enabled),
            "time_hhmm": str(config.transcript_import_time or "02:00"),
            "timezone_name": str(
                config.transcript_import_timezone
                or "UTC"
            ),
            "max_ids": int(config.transcript_import_max_ids),
            "filter_json": str(config.transcript_import_filter_json or "{}"),
            "language_code": str(config.language or "en"),
        }

    app.config["transcript_import_runtime_settings"] = (
        build_transcript_import_settings(base_config)
    )
    app.config["transcript_import_store"] = TranscriptImportStore(
        import_dir=base_config.transcript_import_dir
    )
    app.config["transcript_url_import_store"] = TranscriptImportStore(
        import_dir=os.path.join(base_config.transcript_import_dir, "url_mode")
    )
    transcript_import_store = app.config.get("transcript_import_store")
    if isinstance(transcript_import_store, TranscriptImportStore):
        app.config["transcript_import_last_status"] = (
            transcript_import_store.load_latest_status()
        )
    else:
        app.config["transcript_import_last_status"] = None
    transcript_url_import_store = app.config.get("transcript_url_import_store")
    if isinstance(transcript_url_import_store, TranscriptImportStore):
        app.config["transcript_url_import_last_status"] = (
            transcript_url_import_store.load_latest_status()
        )
    else:
        app.config["transcript_url_import_last_status"] = None
    app.config["transcript_import_scheduler"] = None

    def home_template_context(
        base_cfg: AppConfig,
        *,
        errors: Optional[list[str]] = None,
        active_home_tab: Optional[str] = None,
        active_transcript_tab: Optional[str] = None,
        selected_language_override: Optional[str] = None,
        selected_evaluation_results_language_override: Optional[str] = None,
    ) -> dict:
        runtime_settings = app.config.get("transcript_import_runtime_settings")
        if not isinstance(runtime_settings, dict):
            runtime_settings = build_transcript_import_settings(base_cfg)
        selected_language_source = (
            selected_language_override
            if selected_language_override is not None
            else str(runtime_settings.get("language_code") or base_cfg.language or "en")
        )
        selected_language = normalize_language_code(selected_language_source, default="en")
        selected_eval_source = (
            selected_evaluation_results_language_override
            if selected_evaluation_results_language_override is not None
            else str(base_cfg.evaluation_results_language or "inherit")
        )
        selected_evaluation_results_language = (
            normalize_evaluation_results_language(
                selected_eval_source,
                default="inherit",
            )
            or "inherit"
        )
        home_tab_requested = str(request.args.get("home_tab", "")).strip().lower()
        transcript_tab_requested = str(
            request.args.get("transcript_tab", "")
        ).strip().lower()
        home_tab = (active_home_tab or home_tab_requested or "harness").strip().lower()
        transcript_tab = (
            active_transcript_tab or transcript_tab_requested or "upload"
        ).strip().lower()
        if home_tab not in {"language", "harness", "transcript"}:
            home_tab = "harness"
        if transcript_tab not in {"upload", "ids", "url", "automation"}:
            transcript_tab = "upload"
        return {
            "config": base_cfg,
            "errors": errors,
            "transcript_import_settings": runtime_settings,
            "transcript_import_last_status": app.config.get("transcript_import_last_status"),
            "transcript_url_import_last_status": app.config.get(
                "transcript_url_import_last_status"
            ),
            "language_options": SUPPORTED_LANGUAGE_OPTIONS,
            "evaluation_results_language_options": EVALUATION_RESULTS_LANGUAGE_OPTIONS,
            "selected_language": selected_language,
            "selected_evaluation_results_language": selected_evaluation_results_language,
            "active_home_tab": home_tab,
            "active_transcript_tab": transcript_tab,
        }

    def render_home(
        base_cfg: AppConfig,
        errors: Optional[list[str]] = None,
        *,
        active_home_tab: Optional[str] = None,
        active_transcript_tab: Optional[str] = None,
        selected_language_override: Optional[str] = None,
        selected_evaluation_results_language_override: Optional[str] = None,
    ):
        return render_template(
            "home.html",
            **home_template_context(
                base_cfg,
                errors=errors,
                active_home_tab=active_home_tab,
                active_transcript_tab=active_transcript_tab,
                selected_language_override=selected_language_override,
                selected_evaluation_results_language_override=(
                    selected_evaluation_results_language_override
                ),
            ),
        )

    def set_transcript_import_status(status_payload: dict) -> None:
        app.config["transcript_import_last_status"] = status_payload

    def set_transcript_url_import_status(status_payload: dict) -> None:
        app.config["transcript_url_import_last_status"] = status_payload

    def resolve_results_language_code() -> str:
        """Resolve language used for judge explanations and results labels."""
        last_run_config = app.config.get("last_run_config")
        if isinstance(last_run_config, AppConfig):
            return resolve_effective_evaluation_results_language(
                runtime_override=last_run_config.evaluation_results_language,
                config_value=last_run_config.evaluation_results_language,
                run_language=last_run_config.language,
            )
        base_cfg = load_app_config()
        return resolve_effective_evaluation_results_language(
            runtime_override=None,
            config_value=base_cfg.evaluation_results_language,
            run_language=base_cfg.language,
        )

    def build_dashboard_context(
        report: Optional[TestReport],
        *,
        baseline_run_id: Optional[str] = None,
    ) -> dict:
        """Build dashboard metrics plus optional baseline/trend context."""
        if report is None:
            return {
                "metrics": None,
                "baseline_entry": None,
                "history_count": 0,
                "baseline_options": [],
                "selected_baseline_run_id": None,
            }

        history_store = app.config.get("history_store")
        latest_entry = app.config.get("latest_run_history_entry")
        baseline_entry = None
        baseline_report = None
        baseline_summary = None
        trend_entries: list[dict] = []
        baseline_options: list[dict] = []
        history_count = 0

        if isinstance(history_store, RunHistoryStore):
            exclude_run_id = None
            if isinstance(latest_entry, dict):
                exclude_run_id = latest_entry.get("run_id")
            trend_entries = history_store.list_entries(
                suite_name=report.suite_name,
                limit=25,
            )
            history_count = len(trend_entries)
            baseline_options = [
                entry
                for entry in history_store.list_entries(
                    suite_name=report.suite_name,
                    limit=BASELINE_OPTIONS_LIMIT,
                )
                if not exclude_run_id or entry.get("run_id") != exclude_run_id
            ]

            selected_baseline = str(baseline_run_id or "").strip()
            if selected_baseline:
                candidate = history_store.get_entry_by_run_id(selected_baseline)
                if (
                    isinstance(candidate, dict)
                    and str(candidate.get("suite_name", "")).strip().lower()
                    == report.suite_name.strip().lower()
                ):
                    baseline_entry = candidate
            if baseline_entry is None:
                baseline_entry = history_store.get_previous_same_suite(
                    report.suite_name,
                    exclude_run_id=exclude_run_id,
                )
            if baseline_entry is not None:
                baseline_report = history_store.load_report_from_entry(baseline_entry)
                if baseline_report is None:
                    baseline_summary = summarize_entry_for_compare(baseline_entry)

        metrics = build_dashboard_metrics(
            report,
            baseline_report=baseline_report,
            baseline_summary=baseline_summary,
            trend_entries=trend_entries,
            current_run_id=(
                latest_entry.get("run_id")
                if isinstance(latest_entry, dict)
                else None
            ),
        )
        return {
            "metrics": metrics,
            "baseline_entry": baseline_entry,
            "history_count": history_count,
            "baseline_options": baseline_options,
            "selected_baseline_run_id": (
                baseline_entry.get("run_id")
                if isinstance(baseline_entry, dict)
                else None
            ),
        }

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
            skipped = sum(1 for attempt in attempts if attempt.skipped)
            failures = attempts_count - successes - timeouts - skipped
            success_rate = successes / attempts_count if attempts_count else 0.0
            is_regression = success_rate < threshold
            scenario_results.append(
                {
                    "scenario_name": scenario_name,
                    "attempts": attempts_count,
                    "successes": successes,
                    "failures": failures,
                    "timeouts": timeouts,
                    "skipped": skipped,
                    "success_rate": success_rate,
                    "is_regression": is_regression,
                    "attempt_results": attempts,
                }
            )

        overall_attempts = sum(item["attempts"] for item in scenario_results)
        overall_successes = sum(item["successes"] for item in scenario_results)
        overall_failures = sum(item["failures"] for item in scenario_results)
        overall_timeouts = sum(item["timeouts"] for item in scenario_results)
        overall_skipped = sum(item["skipped"] for item in scenario_results)
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
            overall_skipped=overall_skipped,
            overall_success_rate=overall_success_rate,
            has_regressions=has_regressions,
            regression_threshold=threshold,
        )

    def start_background_run(merged_config: AppConfig, test_suite) -> None:
        """Start a test run in a background thread with fresh run state."""
        progress_emitter = ProgressEmitter()
        app.config["progress_emitter"] = progress_emitter
        app.config["latest_report"] = None
        app.config["latest_run_history_entry"] = None
        app.config["run_active"] = True
        app.config["stop_event"] = threading.Event()
        app.config["stop_requested"] = False
        app.config["history_store"] = RunHistoryStore(
            history_dir=merged_config.history_dir,
            max_runs=merged_config.history_max_runs,
            full_json_runs=merged_config.history_full_json_runs,
            gzip_runs=merged_config.history_gzip_runs,
        )

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
                history_store = app.config.get("history_store")
                if isinstance(history_store, RunHistoryStore):
                    try:
                        entry = history_store.save_report(report)
                        app.config["latest_run_history_entry"] = entry
                    except Exception:
                        # History persistence should never block result availability.
                        app.config["latest_run_history_entry"] = None
            finally:
                app.config["run_active"] = False
                loop.close()

        thread = threading.Thread(target=run_tests, daemon=True)
        thread.start()

    def _parse_positive_int(raw: str, *, fallback: int) -> int:
        try:
            value = int(str(raw).strip())
        except (TypeError, ValueError):
            return fallback
        return max(1, value)

    def _update_runtime_transcript_settings_from_form(
        base_cfg: AppConfig,
        form,
    ) -> dict:
        current = app.config.get("transcript_import_runtime_settings")
        if not isinstance(current, dict):
            current = build_transcript_import_settings(base_cfg)

        enabled = bool(current.get("enabled", False))
        if "transcript_import_enabled" in form:
            enabled = form.get("transcript_import_enabled") == "on"

        time_hhmm = str(current.get("time_hhmm", "02:00") or "02:00").strip() or "02:00"
        if "transcript_import_time" in form:
            time_hhmm = str(form.get("transcript_import_time", time_hhmm)).strip() or "02:00"

        timezone_name = str(current.get("timezone_name", "") or "").strip()
        if "transcript_import_timezone" in form:
            timezone_name = str(form.get("transcript_import_timezone", timezone_name)).strip()

        max_ids = int(current.get("max_ids", 50) or 50)
        if "transcript_import_max_ids" in form:
            max_ids = _parse_positive_int(
                form.get("transcript_import_max_ids", max_ids),
                fallback=max_ids,
            )

        filter_json = str(current.get("filter_json", "{}") or "{}").strip() or "{}"
        if "transcript_import_filter_json" in form:
            filter_json = (
                str(form.get("transcript_import_filter_json", filter_json)).strip()
                or "{}"
            )
        language_raw = str(
            form.get(
                "language",
                current.get("language_code", base_cfg.language),
            )
        ).strip()
        language_code = normalize_language_code(
            language_raw,
            default=(base_cfg.language or "en"),
        )

        updated = {
            "enabled": enabled,
            "time_hhmm": time_hhmm,
            "timezone_name": timezone_name,
            "max_ids": max_ids,
            "filter_json": filter_json,
            "language_code": language_code,
        }
        app.config["transcript_import_runtime_settings"] = updated
        return updated

    def _merge_transcript_settings_into_config(
        base_cfg: AppConfig,
        settings: dict,
    ) -> AppConfig:
        return merge_config(
            base_cfg,
            {
                "transcript_import_enabled": bool(settings.get("enabled")),
                "transcript_import_time": str(settings.get("time_hhmm") or "02:00"),
                "transcript_import_timezone": str(settings.get("timezone_name") or ""),
                "transcript_import_max_ids": int(settings.get("max_ids") or 50),
                "transcript_import_filter_json": str(settings.get("filter_json") or "{}"),
                "language": str(settings.get("language_code") or base_cfg.language or "en"),
            },
        )

    def run_transcript_import_workflow(
        *,
        merged_config: AppConfig,
        id_mode: str,
        suite_name: Optional[str],
        max_scenarios: int,
        max_ids: int,
        ids_file_content: Optional[str] = None,
        ids_file_name: str = "",
        ids_paste_text: str = "",
        auto_filter_json: str = "",
        interval: Optional[str] = None,
        source_label: str = "manual",
    ) -> dict:
        if not merged_config.gc_region:
            raise TranscriptSeedError("GC region is required for transcript import.")
        if not merged_config.gc_client_id or not merged_config.gc_client_secret:
            raise TranscriptSeedError(
                "Genesys OAuth Client ID and Client Secret are required for transcript import."
            )

        client = GenesysTranscriptImportClient(
            region=merged_config.gc_region,
            client_id=merged_config.gc_client_id,
            client_secret=merged_config.gc_client_secret,
            timeout=merged_config.response_timeout,
        )

        requested_ids: list[str] = []
        parsed_filter_payload: dict = {}
        active_interval = interval

        if id_mode == "ids_file":
            if ids_file_content is None:
                raise TranscriptSeedError("Upload a conversation IDs file for file mode.")
            requested_ids = parse_conversation_ids_from_file(
                content=ids_file_content,
                filename=ids_file_name,
            )
        elif id_mode == "ids_paste":
            requested_ids = parse_conversation_ids_from_paste(ids_paste_text)
        elif id_mode == "auto_query":
            parsed_filter_payload = parse_filter_json(auto_filter_json)
            active_interval = active_interval or build_last_24h_interval()
            query_rows = client.query_conversation_ids(
                filter_payload=parsed_filter_payload,
                interval=active_interval,
                page_size=100,
                max_results=max_ids,
            )
            requested_ids = [
                str(row.get("conversation_id") or "").strip().lower()
                for row in query_rows
                if str(row.get("conversation_id") or "").strip()
            ]
        else:
            raise TranscriptSeedError(f"Unsupported transcript import mode: {id_mode}")

        selected_ids = dedupe_and_cap_conversation_ids(requested_ids, max_ids=max_ids)
        if not selected_ids:
            raise TranscriptSeedError(
                "No valid conversation IDs were found for transcript import."
            )

        outcomes = client.import_transcripts_by_ids(selected_ids)
        fetched = outcomes.get("fetched", [])
        failed = outcomes.get("failed", [])
        skipped = outcomes.get("skipped", [])
        if not fetched:
            raise TranscriptSeedError(
                "No transcripts were fetched successfully. "
                "Review failure details and OAuth/API permissions."
            )

        transcripts_for_seeder = [item["transcript"] for item in fetched]
        transcript_payload = build_transcript_seeder_payload(transcripts_for_seeder)
        seeded_suite, seed_diagnostics = seed_test_suite_from_transcript_with_diagnostics(
            json.dumps(transcript_payload),
            format_hint="json",
            suite_name=(suite_name or "").strip() or None,
            max_scenarios=max_scenarios,
            language_code=merged_config.language,
        )
        suite_yaml = print_test_suite(seeded_suite, format="yaml")

        warnings = list(seed_diagnostics.warnings)
        if failed or skipped:
            warnings.append(
                "Transcript import completed with partial success: "
                f"{len(failed)} failed, {len(skipped)} skipped."
            )

        manifest = {
            "status": "completed",
            "source": source_label,
            "mode": id_mode,
            "language": merged_config.language,
            "requested_ids": len(requested_ids),
            "selected_ids": len(selected_ids),
            "fetched_ids": len(fetched),
            "failed_ids": len(failed),
            "skipped_ids": len(skipped),
            "scenarios_generated": seed_diagnostics.scenarios_generated,
            "interval": active_interval,
            "filter_json": auto_filter_json or "{}",
            "failures": failed + skipped,
        }
        transcripts_by_id = {
            item["conversation_id"]: item["raw_payload"]
            for item in fetched
            if item.get("conversation_id")
        }
        transcript_store = app.config.get("transcript_import_store")
        stored_manifest = manifest
        if isinstance(transcript_store, TranscriptImportStore):
            stored_manifest = transcript_store.save_run(
                manifest=manifest,
                transcripts_by_id=transcripts_by_id,
                suite_yaml=suite_yaml,
            )
            set_transcript_import_status(
                transcript_store.load_latest_status() or manifest
            )

        return {
            "seeded_suite": seeded_suite,
            "suite_yaml": suite_yaml,
            "seed_diagnostics": seed_diagnostics,
            "warnings": warnings,
            "stored_manifest": stored_manifest,
            "import_summary": {
                "requested_ids": len(requested_ids),
                "selected_ids": len(selected_ids),
                "fetched_ids": len(fetched),
                "failed_ids": len(failed),
                "skipped_ids": len(skipped),
                "scenarios_generated": seed_diagnostics.scenarios_generated,
                "mode": id_mode,
                "source": source_label,
                "language": merged_config.language,
            },
            "failure_details": failed + skipped,
        }

    def run_scheduled_transcript_import(settings: dict) -> None:
        if app.config.get("run_active", False) or app.config.get(
            "transcript_import_active", False
        ):
            skipped_manifest = {
                "status": "skipped",
                "source": "scheduler",
                "mode": "auto_query",
                "requested_ids": 0,
                "selected_ids": 0,
                "fetched_ids": 0,
                "failed_ids": 0,
                "skipped_ids": 0,
                "scenarios_generated": 0,
                "failures": [
                    {
                        "conversation_id": "",
                        "reason": "Skipped because another run/import is active.",
                    }
                ],
            }
            transcript_store = app.config.get("transcript_import_store")
            if isinstance(transcript_store, TranscriptImportStore):
                transcript_store.save_run(
                    manifest=skipped_manifest,
                    transcripts_by_id={},
                    suite_yaml=None,
                )
                set_transcript_import_status(
                    transcript_store.load_latest_status() or skipped_manifest
                )
            else:
                set_transcript_import_status(skipped_manifest)
            return

        app.config["transcript_import_active"] = True
        try:
            base_cfg = load_app_config()
            merged_cfg = _merge_transcript_settings_into_config(base_cfg, settings)
            suite_name = f"Daily Transcript Suite {datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
            run_transcript_import_workflow(
                merged_config=merged_cfg,
                id_mode="auto_query",
                suite_name=suite_name,
                max_scenarios=max(1, int(merged_cfg.transcript_import_max_ids)),
                max_ids=max(1, int(merged_cfg.transcript_import_max_ids)),
                auto_filter_json=str(settings.get("filter_json") or "{}"),
                interval=build_last_24h_interval(),
                source_label="scheduler",
            )
        except Exception as e:
            failed_manifest = {
                "status": "failed",
                "source": "scheduler",
                "mode": "auto_query",
                "requested_ids": 0,
                "selected_ids": 0,
                "fetched_ids": 0,
                "failed_ids": 1,
                "skipped_ids": 0,
                "scenarios_generated": 0,
                "failures": [{"conversation_id": "", "reason": str(e)}],
            }
            transcript_store = app.config.get("transcript_import_store")
            if isinstance(transcript_store, TranscriptImportStore):
                transcript_store.save_run(
                    manifest=failed_manifest,
                    transcripts_by_id={},
                    suite_yaml=None,
                )
                set_transcript_import_status(
                    transcript_store.load_latest_status() or failed_manifest
                )
            else:
                set_transcript_import_status(failed_manifest)
        finally:
            app.config["transcript_import_active"] = False

    def ensure_transcript_scheduler_state() -> None:
        settings = app.config.get("transcript_import_runtime_settings")
        scheduler = app.config.get("transcript_import_scheduler")
        enabled = isinstance(settings, dict) and bool(settings.get("enabled"))
        if enabled and not isinstance(scheduler, TranscriptImportScheduler):
            scheduler = TranscriptImportScheduler(
                settings_getter=lambda: app.config.get(
                    "transcript_import_runtime_settings", {}
                ),
                run_job=run_scheduled_transcript_import,
            )
            scheduler.start()
            app.config["transcript_import_scheduler"] = scheduler
            return
        if not enabled and isinstance(scheduler, TranscriptImportScheduler):
            scheduler.stop()
            app.config["transcript_import_scheduler"] = None

    ensure_transcript_scheduler_state()

    @app.route("/")
    def home():
        """Home page with config inputs and file upload."""
        base_config = load_app_config()
        return render_home(base_config, errors=None)

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
        language_raw = request.form.get("language", "").strip()
        evaluation_results_language_raw = request.form.get(
            "evaluation_results_language",
            "",
        ).strip()
        harness_mode_raw = request.form.get("harness_mode", "").strip()
        journey_category_strategy_raw = request.form.get(
            "journey_category_strategy",
            "",
        ).strip()
        debug_capture_frames = request.form.get("debug_capture_frames") is not None
        debug_capture_frame_limit = request.form.get("debug_capture_frame_limit", "").strip()

        selected_language_for_home = normalize_language_code(
            language_raw or base_config.language or "en",
            default="en",
        )
        selected_eval_for_home = normalize_evaluation_results_language(
            evaluation_results_language_raw or base_config.evaluation_results_language or "inherit",
            default="inherit",
        ) or "inherit"

        # Read uploaded file
        uploaded_file = request.files.get("test_suite_file")
        if not uploaded_file or uploaded_file.filename == "":
            return render_home(
                base_config,
                errors=["Please upload a test suite file (JSON or YAML)."],
                active_home_tab="harness",
                selected_language_override=selected_language_for_home,
                selected_evaluation_results_language_override=selected_eval_for_home,
            )

        # Determine format from filename
        filename = uploaded_file.filename.lower()
        if filename.endswith(".json"):
            fmt = "json"
        elif filename.endswith((".yaml", ".yml")):
            fmt = "yaml"
        else:
            return render_home(
                base_config,
                errors=["Unsupported file format. Use .json, .yaml, or .yml"],
                active_home_tab="harness",
                selected_language_override=selected_language_for_home,
                selected_evaluation_results_language_override=selected_eval_for_home,
            )

        # Read and validate file content
        try:
            content = uploaded_file.read().decode("utf-8")
        except UnicodeDecodeError:
            return render_home(
                base_config,
                errors=["File must be valid UTF-8 text."],
                active_home_tab="harness",
                selected_language_override=selected_language_for_home,
                selected_evaluation_results_language_override=selected_eval_for_home,
            )

        try:
            test_suite = load_test_suite_from_string(content, fmt)
        except (ValueError, ValidationError) as e:
            error_msg = str(e)
            return render_home(
                base_config,
                errors=[f"Invalid test suite: {error_msg}"],
                active_home_tab="harness",
                selected_language_override=selected_language_for_home,
                selected_evaluation_results_language_override=selected_eval_for_home,
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
        harness_mode_override: Optional[str] = None
        if harness_mode_raw:
            try:
                harness_mode_override = normalize_harness_mode(
                    harness_mode_raw,
                    allow_none=True,
                )
            except ValueError as e:
                return render_home(
                    base_config,
                    errors=[str(e)],
                    active_home_tab="harness",
                    selected_language_override=selected_language_for_home,
                    selected_evaluation_results_language_override=selected_eval_for_home,
                )
        if journey_category_strategy_raw:
            try:
                web_overrides["journey_category_strategy"] = normalize_category_strategy(
                    journey_category_strategy_raw
                )
            except ValueError as e:
                return render_home(
                    base_config,
                    errors=[str(e)],
                    active_home_tab="harness",
                    selected_language_override=selected_language_for_home,
                    selected_evaluation_results_language_override=selected_eval_for_home,
                )
        language_override: Optional[str] = None
        if language_raw:
            try:
                language_override = normalize_language_code(language_raw, default="en")
            except ValueError as e:
                return render_home(
                    base_config,
                    errors=[str(e)],
                    active_home_tab="harness",
                    selected_language_override=selected_language_for_home,
                    selected_evaluation_results_language_override=selected_eval_for_home,
                )
            web_overrides["language"] = language_override
        evaluation_results_language_override: Optional[str] = None
        if evaluation_results_language_raw:
            try:
                evaluation_results_language_override = (
                    normalize_evaluation_results_language(
                        evaluation_results_language_raw,
                        default="inherit",
                    )
                )
            except ValueError as e:
                return render_home(
                    base_config,
                    errors=[str(e)],
                    active_home_tab="harness",
                    selected_language_override=selected_language_for_home,
                    selected_evaluation_results_language_override=selected_eval_for_home,
                )
            if evaluation_results_language_override is not None:
                web_overrides["evaluation_results_language"] = (
                    evaluation_results_language_override
                )
        web_overrides["debug_capture_frames"] = debug_capture_frames
        if debug_capture_frame_limit:
            web_overrides["debug_capture_frame_limit"] = debug_capture_frame_limit

        merged_config = merge_config(base_config, web_overrides)
        effective_language = resolve_effective_language(
            runtime_override=language_override,
            suite_language=test_suite.language,
            config_language=merged_config.language,
        )
        effective_evaluation_results_language = (
            resolve_effective_evaluation_results_language(
                runtime_override=evaluation_results_language_override,
                config_value=merged_config.evaluation_results_language,
                run_language=effective_language,
            )
        )
        merged_config = merge_config(
            merged_config,
            {
                "language": effective_language,
                "evaluation_results_language": effective_evaluation_results_language,
            },
        )
        if harness_mode_override:
            test_suite = test_suite.model_copy(deep=True)
            test_suite.harness_mode = harness_mode_override

        # Validate required config
        missing = validate_required_config(merged_config)
        if missing:
            errors = [
                f"Missing required configuration: {', '.join(missing)}"
            ]
            return render_home(
                base_config,
                errors=errors,
                active_home_tab="harness",
                selected_language_override=selected_language_for_home,
                selected_evaluation_results_language_override=selected_eval_for_home,
            )

        # Validate Ollama connectivity and model before starting long test runs.
        try:
            JudgeLLMClient(
                base_url=merged_config.ollama_base_url,
                model=merged_config.ollama_model or "",
                timeout=merged_config.response_timeout,
            ).verify_connection()
        except JudgeLLMError as e:
            return render_home(
                base_config,
                errors=[str(e)],
                active_home_tab="harness",
                selected_language_override=selected_language_for_home,
                selected_evaluation_results_language_override=selected_eval_for_home,
            )

        app.config["last_run_config"] = merged_config.model_copy(deep=True)
        app.config["last_run_suite"] = test_suite.model_copy(deep=True)
        start_background_run(merged_config, test_suite)

        return redirect(url_for("results"))

    @app.route("/seed", methods=["POST"])
    def seed():
        """Generate a draft test suite from an uploaded transcript file."""
        base_config = load_app_config()
        uploaded_file = request.files.get("transcript_file")
        if not uploaded_file or uploaded_file.filename == "":
            return render_home(
                base_config,
                errors=["Please upload a transcript file to seed a suite."],
                active_home_tab="transcript",
                active_transcript_tab="upload",
            )

        suite_name = request.form.get("seed_suite_name", "").strip()
        language_raw = request.form.get("language", "").strip()
        try:
            selected_language = normalize_language_code(
                language_raw or base_config.language,
                default=base_config.language or "en",
            )
        except ValueError as e:
            return render_home(
                base_config,
                errors=[str(e)],
                active_home_tab="transcript",
                active_transcript_tab="upload",
            )
        max_scenarios_raw = request.form.get("seed_max_scenarios", "50").strip()
        try:
            max_scenarios = max(1, int(max_scenarios_raw))
        except ValueError:
            return render_home(
                base_config,
                errors=["Max seeded scenarios must be a positive integer."],
                active_home_tab="transcript",
                active_transcript_tab="upload",
            )

        filename = uploaded_file.filename.lower()
        if filename.endswith(".json"):
            fmt = "json"
        elif filename.endswith((".yaml", ".yml")):
            fmt = "yaml"
        elif filename.endswith(".csv"):
            fmt = "csv"
        elif filename.endswith(".tsv"):
            fmt = "tsv"
        else:
            # Transcript exports are often .txt/.log/.csv; treat as text by default.
            fmt = "text"

        try:
            content = uploaded_file.read().decode("utf-8")
        except UnicodeDecodeError:
            return render_home(
                base_config,
                errors=["Transcript file must be valid UTF-8 text."],
                active_home_tab="transcript",
                active_transcript_tab="upload",
            )

        try:
            seeded_suite, seed_diagnostics = seed_test_suite_from_transcript_with_diagnostics(
                content,
                format_hint=fmt,
                suite_name=suite_name or None,
                max_scenarios=max_scenarios,
                language_code=selected_language,
            )
        except (TranscriptSeedError, ValidationError, ValueError) as e:
            return render_home(
                base_config,
                errors=[f"Could not seed suite from transcript: {e}"],
                active_home_tab="transcript",
                active_transcript_tab="upload",
            )

        suite_yaml = print_test_suite(seeded_suite, format="yaml")
        return render_template(
            "seed_preview.html",
            seeded_suite=seeded_suite,
            suite_yaml=suite_yaml,
            transcript_filename=uploaded_file.filename,
            extraction_summary={
                "utterances_found": seed_diagnostics.utterances_found,
                "scenarios_generated": seed_diagnostics.scenarios_generated,
                "messages_skipped": seed_diagnostics.skipped_messages,
            },
            extraction_warnings=seed_diagnostics.warnings,
        )

    @app.route("/seed/url", methods=["POST"])
    def seed_url():
        """Generate a draft test suite from a transcript URL."""
        base_config = load_app_config()
        transcript_url = request.form.get("transcript_url", "").strip()
        if not transcript_url:
            return render_home(
                base_config,
                errors=["Please provide a transcript URL."],
                active_home_tab="transcript",
                active_transcript_tab="url",
            )

        suite_name = request.form.get("seed_suite_name", "").strip()
        language_raw = request.form.get("language", "").strip()
        try:
            selected_language = normalize_language_code(
                language_raw or base_config.language,
                default=base_config.language or "en",
            )
        except ValueError as e:
            return render_home(
                base_config,
                errors=[str(e)],
                active_home_tab="transcript",
                active_transcript_tab="url",
            )

        max_scenarios_raw = request.form.get("seed_max_scenarios", "50").strip()
        try:
            max_scenarios = max(1, int(max_scenarios_raw))
        except ValueError:
            return render_home(
                base_config,
                errors=["Max seeded scenarios must be a positive integer."],
                active_home_tab="transcript",
                active_transcript_tab="url",
            )

        seed_strategy = str(
            request.form.get("seed_strategy", "utterance")
        ).strip().lower() or "utterance"
        if seed_strategy not in {"utterance", "journey"}:
            return render_home(
                base_config,
                errors=["Seed strategy must be either 'utterance' or 'journey'."],
                active_home_tab="transcript",
                active_transcript_tab="url",
            )
        journey_category_strategy_raw = str(
            request.form.get("journey_category_strategy", ""),
        ).strip()
        journey_category_strategy = (
            normalize_category_strategy(journey_category_strategy_raw)
            if journey_category_strategy_raw
            else normalize_category_strategy(base_config.journey_category_strategy)
        )
        merged_config = merge_config(
            base_config,
            {
                "language": selected_language,
                "journey_category_strategy": journey_category_strategy,
            },
        )
        importer = TranscriptUrlImportService(
            allowlist_domains=merged_config.transcript_url_allowlist,
            timeout_seconds=merged_config.transcript_url_timeout_seconds,
            max_bytes=merged_config.transcript_url_max_bytes,
        )
        try:
            fetched = importer.fetch_transcript_json(transcript_url)
            payload_text = json.dumps(fetched.payload, ensure_ascii=False)
            warnings: list[str] = []
            if seed_strategy == "journey":
                category_overrides = load_category_overrides(
                    categories_json=merged_config.journey_primary_categories_json,
                    categories_file=merged_config.journey_primary_categories_file,
                )
                primary_categories = resolve_primary_categories(
                    suite_categories=None,
                    config_overrides=category_overrides,
                )
                language_profile = get_language_profile(merged_config.language)
                persona_template = str(
                    language_profile.get("seeded_persona")
                    or "A traveler contacting the WestJet Travel Agent."
                ).strip()
                candidates = extract_journey_seed_candidates(fetched.payload)
                if not candidates:
                    raise TranscriptSeedError(
                        "No valid conversation journeys were found in the transcript payload."
                    )
                selected_candidates = candidates[:max_scenarios]
                dropped_candidates = max(0, len(candidates) - len(selected_candidates))

                classifier_error_logged = False

                def llm_classifier(message: str, categories: list[dict]) -> dict:
                    nonlocal classifier_error_logged
                    judge = JudgeLLMClient(
                        base_url=merged_config.ollama_base_url,
                        model=merged_config.ollama_model or "",
                        timeout=merged_config.response_timeout,
                    )
                    try:
                        return judge.classify_primary_category(
                            first_message=message,
                            categories=categories,
                            language_code=merged_config.language,
                        )
                    except JudgeLLMError as e:
                        if not classifier_error_logged:
                            warnings.append(
                                (
                                    "Primary-category LLM classification fallback failed; "
                                    f"using deterministic rules where possible. Details: {e}"
                                )
                            )
                            classifier_error_logged = True
                        return {
                            "category": None,
                            "confidence": None,
                            "explanation": str(e),
                        }

                scenarios: list[TestScenario] = []
                for index, candidate in enumerate(selected_candidates, start=1):
                    first_message = str(
                        candidate.get("first_customer_message") or ""
                    ).strip()
                    if not first_message:
                        continue
                    category_resolution = resolve_category_with_strategy(
                        first_message,
                        categories=primary_categories,
                        strategy=merged_config.journey_category_strategy,
                        llm_classifier=(
                            llm_classifier
                            if merged_config.journey_category_strategy
                            in {CATEGORY_STRATEGY_RULES_FIRST, CATEGORY_STRATEGY_LLM_FIRST}
                            else None
                        ),
                    )
                    resolved_category = category_resolution.get("category")
                    category_label = (
                        str(resolved_category).replace("_", " ").strip()
                        if resolved_category
                        else "general journey"
                    )
                    scenario_name = (
                        f"{resolved_category or 'journey'} - Conversation {index:02d}"
                    )
                    path_rubric = ""
                    if resolved_category:
                        for item in primary_categories:
                            if str(item.get("name") or "").strip().lower() == str(
                                resolved_category
                            ).strip().lower():
                                path_rubric = str(item.get("rubric") or "").strip()
                                break
                    goal_text = (
                        f"Validate the end-to-end customer journey for {category_label}. "
                        "The journey succeeds only when the customer is contained in automation "
                        "and the request is fulfilled through the correct path."
                    )
                    if path_rubric:
                        goal_text = f"{goal_text} {path_rubric}"
                    scenarios.append(
                        TestScenario(
                            name=scenario_name,
                            persona=persona_template,
                            goal=goal_text,
                            first_message=first_message,
                            attempts=1,
                            journey_category=resolved_category,
                            journey_validation=JourneyValidationConfig(
                                require_containment=True,
                                require_fulfillment=True,
                                path_rubric=path_rubric or None,
                            ),
                        )
                    )

                if not scenarios:
                    raise TranscriptSeedError(
                        "Journey parsing succeeded, but no scenarios could be generated."
                    )

                if dropped_candidates > 0:
                    warnings.append(
                        (
                            "Generated scenarios were truncated by max scenario limit. "
                            f"Ignored {dropped_candidates} additional conversation(s)."
                        )
                    )
                seeded_suite = TestSuite(
                    name=(suite_name or "").strip() or "Transcript Journey Regression Suite",
                    language=merged_config.language,
                    harness_mode=HARNESS_JOURNEY,
                    primary_categories=[
                        PrimaryCategoryConfig.model_validate(category)
                        for category in primary_categories
                    ],
                    scenarios=scenarios,
                )
                seed_diagnostics = None
            else:
                seeded_suite, seed_diagnostics = (
                    seed_test_suite_from_transcript_with_diagnostics(
                        payload_text,
                        format_hint="json",
                        suite_name=suite_name or None,
                        max_scenarios=max_scenarios,
                        language_code=merged_config.language,
                    )
                )
            suite_yaml = print_test_suite(seeded_suite, format="yaml")
        except (TranscriptUrlImportError, TranscriptSeedError, ValidationError, ValueError) as e:
            return render_home(
                base_config,
                errors=[f"Could not seed suite from transcript URL: {e}"],
                active_home_tab="transcript",
                active_transcript_tab="url",
            )

        if seed_strategy == "journey":
            warnings = warnings
        else:
            warnings = list(seed_diagnostics.warnings)
        if fetched.followed_wrapper_url:
            warnings.append(
                "Resolved transcript from a wrapper URL response before seeding."
            )

        redacted_source_url = redact_url_for_display(fetched.source_url)
        redacted_resolved_url = redact_url_for_display(fetched.resolved_url)
        manifest = {
            "status": "completed",
            "source": "manual",
            "mode": "url",
            "language": merged_config.language,
            "requested_ids": 1,
            "selected_ids": 1,
            "fetched_ids": 1,
            "failed_ids": 0,
            "skipped_ids": 0,
            "scenarios_generated": len(seeded_suite.scenarios),
            "source_url_redacted": redacted_source_url,
            "resolved_url_redacted": redacted_resolved_url,
            "followed_wrapper_url": bool(fetched.followed_wrapper_url),
            "seed_strategy": seed_strategy,
            "journey_category_strategy": merged_config.journey_category_strategy,
            "failures": [],
        }

        transcript_store = app.config.get("transcript_url_import_store")
        if isinstance(transcript_store, TranscriptImportStore):
            transcript_store.save_run(
                manifest=manifest,
                transcripts_by_id={"url_payload": fetched.payload},
                suite_yaml=suite_yaml,
            )
            set_transcript_url_import_status(
                transcript_store.load_latest_status() or manifest
            )
        else:
            set_transcript_url_import_status(manifest)

        return render_template(
            "seed_preview.html",
            seeded_suite=seeded_suite,
            suite_yaml=suite_yaml,
            transcript_filename=redacted_source_url,
            extraction_summary={
                "utterances_found": (
                    seed_diagnostics.utterances_found
                    if seed_diagnostics is not None
                    else len(seeded_suite.scenarios)
                ),
                "scenarios_generated": (
                    seed_diagnostics.scenarios_generated
                    if seed_diagnostics is not None
                    else len(seeded_suite.scenarios)
                ),
                "messages_skipped": (
                    seed_diagnostics.skipped_messages
                    if seed_diagnostics is not None
                    else 0
                ),
            },
            extraction_warnings=warnings,
            import_summary={
                "requested_ids": 1,
                "selected_ids": 1,
                "fetched_ids": 1,
                "failed_ids": 0,
                "skipped_ids": 0,
                "scenarios_generated": len(seeded_suite.scenarios),
                "mode": "url",
                "seed_strategy": seed_strategy,
                "journey_category_strategy": merged_config.journey_category_strategy,
                "source_url": redacted_source_url,
                "resolved_url": redacted_resolved_url,
            },
            failure_details=[],
            failure_manifest_url=None,
        )

    @app.route("/seed/import", methods=["POST"])
    def seed_import():
        """Import transcripts by conversation ID and generate seeded suite."""
        base_config = load_app_config()
        try:
            settings = _update_runtime_transcript_settings_from_form(
                base_config,
                request.form,
            )
        except ValueError as e:
            return render_home(
                base_config,
                errors=[str(e)],
                active_home_tab="transcript",
                active_transcript_tab="ids",
            )
        ensure_transcript_scheduler_state()

        mode = str(request.form.get("id_source_mode", "ids_file")).strip() or "ids_file"
        suite_name = request.form.get("seed_suite_name", "").strip()
        max_scenarios = _parse_positive_int(
            request.form.get("seed_max_scenarios", "50"),
            fallback=50,
        )
        max_ids = _parse_positive_int(
            request.form.get(
                "transcript_import_max_ids",
                str(settings.get("max_ids", base_config.transcript_import_max_ids)),
            ),
            fallback=int(base_config.transcript_import_max_ids),
        )

        ids_file_content = None
        ids_file_name = ""
        ids_paste_text = request.form.get("conversation_ids_paste", "")
        auto_filter_json = request.form.get(
            "transcript_import_filter_json",
            str(settings.get("filter_json") or "{}"),
        )

        if mode == "ids_file":
            ids_file = request.files.get("conversation_ids_file")
            if not ids_file or not ids_file.filename:
                return render_home(
                    base_config,
                    errors=["Upload a conversation IDs file for file import mode."],
                    active_home_tab="transcript",
                    active_transcript_tab="ids",
                )
            ids_file_name = ids_file.filename
            try:
                ids_file_content = ids_file.read().decode("utf-8")
            except UnicodeDecodeError:
                return render_home(
                    base_config,
                    errors=["Conversation IDs file must be valid UTF-8 text."],
                    active_home_tab="transcript",
                    active_transcript_tab="ids",
                )

        merged_config = _merge_transcript_settings_into_config(base_config, settings)
        app.config["transcript_import_active"] = True
        try:
            result = run_transcript_import_workflow(
                merged_config=merged_config,
                id_mode=mode,
                suite_name=suite_name or None,
                max_scenarios=max_scenarios,
                max_ids=max_ids,
                ids_file_content=ids_file_content,
                ids_file_name=ids_file_name,
                ids_paste_text=ids_paste_text,
                auto_filter_json=auto_filter_json,
                interval=build_last_24h_interval() if mode == "auto_query" else None,
                source_label="manual",
            )
        except (TranscriptSeedError, ValidationError, ValueError, GenesysTranscriptImportError) as e:
            app.config["transcript_import_active"] = False
            return render_home(
                base_config,
                errors=[f"Could not import transcripts by conversation ID: {e}"],
                active_home_tab="transcript",
                active_transcript_tab="ids",
            )
        except Exception as e:
            app.config["transcript_import_active"] = False
            return render_home(
                base_config,
                errors=[f"Unexpected transcript import error: {e}"],
                active_home_tab="transcript",
                active_transcript_tab="ids",
            )
        app.config["transcript_import_active"] = False

        stored_manifest = result.get("stored_manifest", {})
        run_id = stored_manifest.get("run_id")
        failure_details = result.get("failure_details", [])
        failure_manifest_url = None
        if run_id and failure_details:
            failure_manifest_url = url_for(
                "seed_import_failures",
                run_id=run_id,
            )

        transcript_source_name = {
            "ids_file": "Conversation IDs file",
            "ids_paste": "Pasted conversation IDs",
            "auto_query": "Genesys auto-query",
        }.get(mode, "Conversation ID import")

        return render_template(
            "seed_preview.html",
            seeded_suite=result["seeded_suite"],
            suite_yaml=result["suite_yaml"],
            transcript_filename=transcript_source_name,
            extraction_summary={
                "utterances_found": result["seed_diagnostics"].utterances_found,
                "scenarios_generated": result["seed_diagnostics"].scenarios_generated,
                "messages_skipped": result["seed_diagnostics"].skipped_messages,
            },
            extraction_warnings=result["warnings"],
            import_summary=result.get("import_summary"),
            failure_manifest_url=failure_manifest_url,
            failure_details=failure_details,
        )

    @app.route("/seed/import/failures")
    def seed_import_failures():
        """Download transcript import failure manifest for a run."""
        run_id = request.args.get("run_id", "").strip()
        if not run_id:
            return redirect(url_for("home"))

        transcript_store = app.config.get("transcript_import_store")
        if not isinstance(transcript_store, TranscriptImportStore):
            return redirect(url_for("home"))

        manifest = transcript_store.load_manifest(run_id)
        if not manifest:
            return redirect(url_for("home"))

        failures = manifest.get("failures", [])
        content = json.dumps(
            {
                "run_id": run_id,
                "status": manifest.get("status"),
                "mode": manifest.get("mode"),
                "failures": failures,
            },
            indent=2,
            ensure_ascii=False,
        )
        return send_file(
            io.BytesIO(content.encode("utf-8")),
            mimetype="application/json",
            as_attachment=True,
            download_name=f"transcript-import-failures-{run_id}.json",
        )

    @app.route("/transcript/import/settings", methods=["POST"])
    def transcript_import_settings():
        """Save transcript import automation settings without running imports."""
        base_config = load_app_config()
        try:
            settings = _update_runtime_transcript_settings_from_form(
                base_config,
                request.form,
            )
            # Validate scheduler values through AppConfig validators.
            _merge_transcript_settings_into_config(base_config, settings)
        except (ValueError, ValidationError) as e:
            return render_home(
                base_config,
                errors=[f"Could not save automation settings: {e}"],
                active_home_tab="transcript",
                active_transcript_tab="automation",
            )

        ensure_transcript_scheduler_state()
        flash("Transcript automation settings saved.")
        return redirect(
            url_for("home", home_tab="transcript", transcript_tab="automation")
        )

    @app.route("/seed/export", methods=["POST"])
    def seed_export():
        """Download the seeded suite YAML from preview."""
        suite_yaml = request.form.get("suite_yaml", "").strip()
        if not suite_yaml:
            return redirect(url_for("home"))

        # Validate before download so users don't export malformed edits.
        try:
            seeded_suite = load_test_suite_from_string(suite_yaml, "yaml")
        except (ValidationError, ValueError) as e:
            base_config = load_app_config()
            return render_home(
                base_config,
                errors=[f"Seeded suite validation failed: {e}"],
            )

        filename = (
            seeded_suite.name.strip().lower().replace(" ", "_") or "seeded_suite"
        )
        safe_filename = "".join(
            ch for ch in filename if ch.isalnum() or ch in {"_", "-"}
        )[:80]
        safe_filename = safe_filename or "seeded_suite"
        return send_file(
            io.BytesIO(suite_yaml.encode("utf-8")),
            mimetype="application/x-yaml",
            as_attachment=True,
            download_name=f"{safe_filename}.yaml",
        )

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

    @app.route("/run/rerun_subset", methods=["POST"])
    def rerun_subset():
        """Re-run a subset of scenarios from the last suite/config snapshot."""
        if app.config.get("run_active", False):
            flash("A run is already active.")
            return redirect(url_for("results"))

        last_config = app.config.get("last_run_config")
        last_suite = app.config.get("last_run_suite")
        if last_config is None or last_suite is None:
            flash("No previous run configuration found. Start a run from the home page first.")
            return redirect(url_for("results"))

        latest_report = app.config.get("latest_report")
        if latest_report is None:
            latest_report = build_partial_report_from_history()
        if latest_report is None:
            flash("No completed attempts available yet to build a rerun subset.")
            return redirect(url_for("results"))

        mode = str(request.form.get("mode", "")).strip()
        selected_names: list[str]
        if mode == "failed_bucket":
            selected_names = [
                scenario.scenario_name
                for scenario in latest_report.scenario_results
                if (scenario.failures > 0 or scenario.timeouts > 0 or scenario.skipped > 0)
            ]
            if not selected_names:
                flash("No failed/timeout/skipped scenarios were found in the latest results.")
                return redirect(url_for("results"))
        elif mode == "selected":
            selected_names = [
                value.strip()
                for value in request.form.getlist("scenario_names")
                if value and value.strip()
            ]
            if not selected_names:
                flash("Select at least one scenario to rerun.")
                return redirect(url_for("results"))
        else:
            flash("Invalid rerun subset mode.")
            return redirect(url_for("results"))

        selected_set = set(selected_names)
        filtered_scenarios = [
            scenario.model_copy(deep=True)
            for scenario in last_suite.scenarios
            if scenario.name in selected_set
        ]
        if not filtered_scenarios:
            flash("No matching scenarios were found in the last uploaded suite.")
            return redirect(url_for("results"))

        merged_config = last_config.model_copy(deep=True)
        filtered_suite = last_suite.model_copy(deep=True)
        filtered_suite.scenarios = filtered_scenarios

        try:
            JudgeLLMClient(
                base_url=merged_config.ollama_base_url,
                model=merged_config.ollama_model or "",
                timeout=merged_config.response_timeout,
            ).verify_connection()
        except JudgeLLMError as e:
            flash(str(e))
            return redirect(url_for("results"))

        start_background_run(merged_config, filtered_suite)
        flash(
            f"Re-running subset: {len(filtered_suite.scenarios)} scenario(s) from {filtered_suite.name}."
        )
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
        baseline_run_id = request.args.get("baseline_run_id", "").strip() or None
        dashboard_context = build_dashboard_context(report, baseline_run_id=baseline_run_id)
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
        results_language = resolve_results_language_code()
        results_i18n = get_results_i18n(results_language)
        return render_template(
            "results.html",
            report=report,
            partial_report=partial_report,
            run_active=run_active,
            stop_requested=stop_requested,
            has_rerun=has_rerun,
            progress_history=progress_history,
            dashboard_metrics=dashboard_context.get("metrics"),
            dashboard_history_count=dashboard_context.get("history_count", 0),
            baseline_options=dashboard_context.get("baseline_options", []),
            selected_baseline_run_id=dashboard_context.get("selected_baseline_run_id"),
            attempt_chunk_size=ATTEMPT_CHUNK_SIZE,
            results_language=results_language,
            results_i18n=results_i18n,
        )

    @app.route("/results/history")
    def results_history():
        """Return run history entries for results baseline selection."""
        history_store = app.config.get("history_store")
        if not isinstance(history_store, RunHistoryStore):
            return jsonify({"runs": []})

        suite_name = request.args.get("suite_name", "").strip() or None
        limit = request.args.get("limit", type=int)
        if limit is None:
            limit = BASELINE_OPTIONS_LIMIT
        limit = max(1, min(limit, 100))

        entries = history_store.list_entries(suite_name=suite_name, limit=limit)
        runs = [
            {
                "run_id": entry.get("run_id"),
                "suite_name": entry.get("suite_name"),
                "timestamp": entry.get("timestamp"),
                "overall_attempts": entry.get("overall_attempts"),
                "overall_success_rate": entry.get("overall_success_rate"),
                "storage_type": entry.get("storage_type", "full_json"),
            }
            for entry in entries
        ]
        return jsonify({"runs": runs})

    @app.route("/results/attempts")
    def results_attempts():
        """Load paginated attempt cards for a scenario in the results view."""
        report = app.config.get("latest_report")
        if report is None:
            report = build_partial_report_from_history()
        if report is None:
            return jsonify({
                "html": "",
                "next_offset": 0,
                "has_more": False,
                "remaining": 0,
            })

        scenario_index = request.args.get("scenario_index", type=int)
        offset = request.args.get("offset", type=int)
        limit = request.args.get("limit", type=int)

        if scenario_index is None or scenario_index < 0:
            return jsonify({
                "html": "",
                "next_offset": 0,
                "has_more": False,
                "remaining": 0,
            })
        if offset is None or offset < 0:
            offset = 0
        if limit is None or limit < 1:
            limit = ATTEMPT_CHUNK_SIZE
        limit = min(limit, 100)

        scenarios = report.scenario_results
        if scenario_index >= len(scenarios):
            return jsonify({
                "html": "",
                "next_offset": offset,
                "has_more": False,
                "remaining": 0,
            })

        scenario = scenarios[scenario_index]
        attempts = scenario.attempt_results[offset : offset + limit]
        next_offset = offset + len(attempts)
        total = len(scenario.attempt_results)

        html = render_template(
            "_attempt_cards_chunk.html",
            attempts=attempts,
            results_i18n=get_results_i18n(resolve_results_language_code()),
        )
        return jsonify({
            "html": html,
            "next_offset": next_offset,
            "has_more": next_offset < total,
            "remaining": max(0, total - next_offset),
        })

    @app.route("/results/export")
    def export():
        """Download report in supported export formats."""
        report = app.config.get("latest_report")
        if report is None:
            report = build_partial_report_from_history()
        if report is None:
            return redirect(url_for("results"))

        fmt = request.args.get("format", "json").lower()
        baseline_run_id = request.args.get("baseline_run_id", "").strip() or None

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
        elif fmt == "dashboard_pdf":
            dashboard_context = build_dashboard_context(report, baseline_run_id=baseline_run_id)
            metrics = dashboard_context.get("metrics") or build_dashboard_metrics(report)
            content = export_dashboard_pdf(
                report,
                metrics,
                language_code=resolve_results_language_code(),
            )
            return Response(
                content,
                mimetype="application/pdf",
                headers={
                    "Content-Disposition": "attachment; filename=dashboard-report.pdf"
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
    debug_mode = os.environ.get("GC_TESTER_DEBUG", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    create_app().run(
        host="0.0.0.0",
        port=5000,
        debug=debug_mode,
        use_reloader=debug_mode,
    )
