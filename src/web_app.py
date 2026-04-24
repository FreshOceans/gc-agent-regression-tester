"""Flask web application for the Regression Test Harness.

Provides a web UI for uploading test suites, triggering test execution,
viewing results, and streaming progress via SSE.
"""

import asyncio
import io
import json
import os
import queue
import re
import secrets
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests
from flask import (
    Flask,
    Response,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    session,
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
from .judge_execution import (
    build_judge_execution_client,
    resolve_effective_judge_model_name,
)
from .judge_llm import JudgeLLMError
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
from .journey_taxonomy import (
    load_taxonomy_overrides as load_journey_taxonomy_overrides,
    normalize_journey_view,
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
    ANALYTICS_AUTH_MODE_CLIENT_CREDENTIALS,
    ANALYTICS_AUTH_MODE_MANUAL_BEARER,
    AppConfig,
    JUDGE_EXECUTION_MODE_DUAL_STRICT_FALLBACK,
    JUDGE_EXECUTION_MODE_SINGLE,
    JourneyValidationConfig,
    ModelWarmupRunMetadata,
    PrimaryCategoryConfig,
    ProgressEvent,
    ProgressEventType,
    TestScenario,
    TestSuite,
    TestReport,
    normalize_gemma_single_model,
    normalize_analytics_auth_mode,
    normalize_judge_execution_mode,
)
from .orchestrator import TestOrchestrator
from .progress import ProgressEmitter
from .run_history import RunHistoryStore
from .dashboard_metrics import build_dashboard_metrics, summarize_entry_for_compare
from .dashboard_pdf import export_dashboard_pdf
from .duration_format import format_duration, format_duration_delta
from .results_i18n import get_results_i18n
from .genesys_analytics_journey_client import (
    GenesysAnalyticsJourneyClient,
    GenesysAnalyticsJourneyError,
)
from .genesys_transcript_import_client import (
    GenesysTranscriptImportClient,
    GenesysTranscriptImportError,
)
from .report import (
    export_csv,
    export_failures_csv,
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
from .analytics_journey_runner import (
    AnalyticsJourneyRunRequest,
    AnalyticsJourneyRunner,
)
from .model_warmup_runner import (
    ModelWarmUpRunRequest,
    ModelWarmUpRunner,
    build_model_warmup_metadata,
    normalize_model_warmup_execution_mode,
    normalize_model_warmup_pacing,
    normalize_model_warmup_workers,
)

ATTEMPT_CHUNK_SIZE = 20
BASELINE_OPTIONS_LIMIT = 30


@dataclass
class ActiveRunControl:
    """Lifecycle state for one active run."""

    run_id: str
    stop_event: threading.Event = field(default_factory=threading.Event)
    stop_requested_at: Optional[datetime] = None
    stop_finalized_at: Optional[datetime] = None
    force_finalized: bool = False
    finalized: bool = False
    thread: Optional[threading.Thread] = None


ANALYTICS_AUTH_MODE_OPTIONS = [
    (
        ANALYTICS_AUTH_MODE_CLIENT_CREDENTIALS,
        "OAuth Client Credentials",
    ),
    (
        ANALYTICS_AUTH_MODE_MANUAL_BEARER,
        "Manual Bearer Token",
    ),
]
JUDGE_EXECUTION_MODE_OPTIONS = [
    (JUDGE_EXECUTION_MODE_SINGLE, "single"),
    (JUDGE_EXECUTION_MODE_DUAL_STRICT_FALLBACK, "dual_strict_fallback"),
]
GEMMA_SINGLE_MODEL_OPTIONS = [
    ("gemma4:e4b", "gemma4:e4b"),
    ("gemma4:31b", "gemma4:31b"),
]
_AUTH_SESSION_KEY = "rth_auth_ok"
_AUTH_LAST_ACTIVITY_TS_KEY = "rth_auth_last_activity_ts"
_CSRF_SESSION_KEY = "rth_csrf_token"


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(
        __name__,
        template_folder=os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "templates"
        ),
    )
    app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Strict"
    app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=1)
    app.jinja_env.globals["format_duration"] = format_duration
    app.jinja_env.globals["format_duration_delta"] = format_duration_delta

    # App state
    app.config["latest_report"]: Optional[TestReport] = None
    app.config["progress_emitter"] = ProgressEmitter()
    app.config["run_active"] = False
    app.config["stop_event"] = threading.Event()
    app.config["stop_requested"] = False
    app.config["active_run_control"]: Optional[ActiveRunControl] = None
    app.config["active_run_id"]: Optional[str] = None
    app.config["run_state_lock"] = threading.Lock()
    app.config["last_run_config"]: Optional[AppConfig] = None
    app.config["last_run_suite"] = None
    app.config["active_model_warmup_metadata"] = None
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
    app.config["analytics_journey_store"] = TranscriptImportStore(
        import_dir=base_config.analytics_journey_artifact_dir
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
    analytics_journey_store = app.config.get("analytics_journey_store")
    if isinstance(analytics_journey_store, TranscriptImportStore):
        app.config["analytics_journey_last_status"] = (
            analytics_journey_store.load_latest_status()
        )
    else:
        app.config["analytics_journey_last_status"] = None
    app.config["transcript_import_scheduler"] = None

    def _get_web_auth_settings() -> tuple[bool, str, str, int]:
        cfg = load_app_config()
        enabled = bool(cfg.web_auth_enabled)
        username = str(cfg.web_auth_username or "").strip()
        password = str(cfg.web_auth_password or "")
        idle_minutes = max(1, int(cfg.web_session_idle_minutes))
        return enabled, username, password, idle_minutes

    def _is_web_auth_enabled() -> bool:
        enabled, _, _, _ = _get_web_auth_settings()
        return enabled

    def _ensure_csrf_token() -> str:
        token = str(session.get(_CSRF_SESSION_KEY, "") or "")
        if not token:
            token = secrets.token_urlsafe(32)
            session[_CSRF_SESSION_KEY] = token
        return token

    def _validate_csrf_token() -> bool:
        expected = str(session.get(_CSRF_SESSION_KEY, "") or "")
        if not expected:
            return False
        submitted = (
            request.form.get("csrf_token")
            or request.headers.get("X-CSRF-Token")
            or ""
        ).strip()
        if not submitted:
            return False
        return secrets.compare_digest(expected, submitted)

    def _safe_next_path(next_path: str) -> str:
        raw = str(next_path or "").strip()
        if not raw:
            return url_for("home")
        if raw.startswith("http://") or raw.startswith("https://"):
            return url_for("home")
        if not raw.startswith("/"):
            return url_for("home")
        if raw.startswith("//"):
            return url_for("home")
        return raw

    @app.context_processor
    def inject_security_context() -> dict:
        return {
            "csrf_token": _ensure_csrf_token(),
            "web_auth_enabled": _is_web_auth_enabled(),
        }

    @app.before_request
    def enforce_web_auth_and_csrf():
        endpoint = str(request.endpoint or "")
        if endpoint.startswith("static"):
            return None

        auth_enabled, _, _, idle_minutes = _get_web_auth_settings()
        app.config["SESSION_COOKIE_SECURE"] = bool(auth_enabled and request.is_secure)
        now_ts = datetime.now(timezone.utc).timestamp()

        if auth_enabled:
            if endpoint == "login":
                return None

            if not bool(session.get(_AUTH_SESSION_KEY)):
                next_path = request.full_path if request.query_string else request.path
                return redirect(url_for("login", next=_safe_next_path(next_path)))

            last_activity = float(session.get(_AUTH_LAST_ACTIVITY_TS_KEY, 0.0) or 0.0)
            if last_activity and (now_ts - last_activity) > (idle_minutes * 60):
                session.clear()
                flash("Session timed out after inactivity. Please sign in again.")
                return redirect(url_for("login"))

            session[_AUTH_LAST_ACTIVITY_TS_KEY] = now_ts

        _ensure_csrf_token()
        if auth_enabled and request.method == "POST" and endpoint != "login":
            if not _validate_csrf_token():
                return (
                    "Invalid or missing CSRF token. Refresh the page and try again.",
                    400,
                )
        return None

    @app.after_request
    def apply_security_headers(response: Response):
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        response.headers.setdefault(
            "Content-Security-Policy",
            (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data:; "
                "font-src 'self' data:; "
                "connect-src 'self'; "
                "frame-ancestors 'none'; "
                "base-uri 'self'; "
                "form-action 'self';"
            ),
        )
        return response

    def home_template_context(
        base_cfg: AppConfig,
        *,
        errors: Optional[list[str]] = None,
        active_home_tab: Optional[str] = None,
        active_transcript_tab: Optional[str] = None,
        selected_run_language_override: Optional[str] = None,
        selected_transcript_language_override: Optional[str] = None,
        selected_evaluation_results_language_override: Optional[str] = None,
        selected_analytics_auth_mode_override: Optional[str] = None,
    ) -> dict:
        runtime_settings = app.config.get("transcript_import_runtime_settings")
        if not isinstance(runtime_settings, dict):
            runtime_settings = build_transcript_import_settings(base_cfg)
        selected_run_language_source = (
            selected_run_language_override
            if selected_run_language_override is not None
            else str(base_cfg.language or "en")
        )
        selected_run_language = normalize_language_code(
            selected_run_language_source,
            default="en",
        )
        selected_transcript_language_source = (
            selected_transcript_language_override
            if selected_transcript_language_override is not None
            else str(runtime_settings.get("language_code") or base_cfg.language or "en")
        )
        selected_transcript_language = normalize_language_code(
            selected_transcript_language_source,
            default="en",
        )
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
        selected_analytics_auth_mode = normalize_analytics_auth_mode(
            selected_analytics_auth_mode_override
            if selected_analytics_auth_mode_override is not None
            else base_cfg.analytics_journey_auth_mode
        )
        home_tab_requested = str(request.args.get("home_tab", "")).strip().lower()
        transcript_tab_requested = str(
            request.args.get("transcript_tab", "")
        ).strip().lower()
        home_tab = (active_home_tab or home_tab_requested or "harness").strip().lower()
        transcript_tab = (
            active_transcript_tab or transcript_tab_requested or "upload"
        ).strip().lower()
        if home_tab not in {
            "language",
            "harness",
            "model_warm_up",
            "transcript",
            "analytics",
        }:
            home_tab = "harness"
        if transcript_tab not in {"upload", "ids", "url", "automation"}:
            transcript_tab = "upload"
        analytics_page_size_cap = int(base_cfg.analytics_journey_details_page_size_cap)
        analytics_default_page_size = max(
            1,
            min(
                int(base_cfg.analytics_journey_default_page_size),
                analytics_page_size_cap,
            ),
        )
        return {
            "config": base_cfg,
            "errors": errors,
            "transcript_import_settings": runtime_settings,
            "transcript_import_last_status": app.config.get("transcript_import_last_status"),
            "transcript_url_import_last_status": app.config.get(
                "transcript_url_import_last_status"
            ),
            "analytics_journey_last_status": app.config.get(
                "analytics_journey_last_status"
            ),
            "language_options": SUPPORTED_LANGUAGE_OPTIONS,
            "evaluation_results_language_options": EVALUATION_RESULTS_LANGUAGE_OPTIONS,
            "selected_run_language": selected_run_language,
            "selected_transcript_language": selected_transcript_language,
            "selected_evaluation_results_language": selected_evaluation_results_language,
            "active_home_tab": home_tab,
            "active_transcript_tab": transcript_tab,
            "analytics_auth_mode_options": ANALYTICS_AUTH_MODE_OPTIONS,
            "judge_execution_mode_options": JUDGE_EXECUTION_MODE_OPTIONS,
            "gemma_single_model_options": GEMMA_SINGLE_MODEL_OPTIONS,
            "analytics_selected_auth_mode": selected_analytics_auth_mode,
            "analytics_default_page_size": analytics_default_page_size,
            "analytics_page_size_cap": analytics_page_size_cap,
            "analytics_default_max_conversations": int(
                base_cfg.analytics_journey_default_max_conversations
            ),
            "effective_harness_judge_model": resolve_effective_judge_model_name(
                base_cfg,
                analytics=False,
            ),
            "effective_analytics_judge_model": resolve_effective_judge_model_name(
                base_cfg,
                analytics=True,
            ),
            "analytics_default_language_filter": (
                base_cfg.analytics_journey_default_language_filter or ""
            ),
            "model_warmup_model_default": (
                base_cfg.ollama_model or base_cfg.judge_single_model or ""
            ),
        }

    def render_home(
        base_cfg: AppConfig,
        errors: Optional[list[str]] = None,
        *,
        active_home_tab: Optional[str] = None,
        active_transcript_tab: Optional[str] = None,
        selected_run_language_override: Optional[str] = None,
        selected_transcript_language_override: Optional[str] = None,
        selected_evaluation_results_language_override: Optional[str] = None,
        selected_analytics_auth_mode_override: Optional[str] = None,
    ):
        return render_template(
            "home.html",
            **home_template_context(
                base_cfg,
                errors=errors,
                active_home_tab=active_home_tab,
                active_transcript_tab=active_transcript_tab,
                selected_run_language_override=selected_run_language_override,
                selected_transcript_language_override=selected_transcript_language_override,
                selected_evaluation_results_language_override=(
                    selected_evaluation_results_language_override
                ),
                selected_analytics_auth_mode_override=(
                    selected_analytics_auth_mode_override
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
        journey_view: str = "overview",
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

        last_run_config = app.config.get("last_run_config")
        journey_dashboard_enabled = False
        taxonomy_overrides: dict[str, str] = {}
        if isinstance(last_run_config, AppConfig):
            journey_dashboard_enabled = bool(last_run_config.journey_dashboard_enabled)
            try:
                taxonomy_overrides = load_journey_taxonomy_overrides(
                    overrides_json=last_run_config.journey_taxonomy_overrides_json,
                    overrides_file=last_run_config.journey_taxonomy_overrides_file,
                )
            except ValueError:
                taxonomy_overrides = {}

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
            journey_dashboard_enabled=journey_dashboard_enabled,
            journey_active_view=journey_view,
            taxonomy_overrides=taxonomy_overrides,
        )
        return {
            "metrics": metrics,
            "baseline_entry": baseline_entry,
            "history_count": history_count,
            "baseline_options": baseline_options,
            "journey_view": journey_view,
            "selected_baseline_run_id": (
                baseline_entry.get("run_id")
                if isinstance(baseline_entry, dict)
                else None
            ),
        }

    def build_intent_groups(report: Optional[TestReport]) -> list[dict]:
        """Build deterministic intent groups for results rendering."""
        if report is None:
            return []

        groups: list[dict] = []
        groups_by_key: dict[str, dict] = {}

        for scenario_index, scenario in enumerate(report.scenario_results):
            normalized_intent = (scenario.expected_intent or "").strip().lower()
            intent_key = normalized_intent or "behavior_journey"
            group = groups_by_key.get(intent_key)
            if group is None:
                group = {
                    "intent_key": intent_key,
                    "display_intent": normalized_intent or None,
                    "is_fallback": not bool(normalized_intent),
                    "attempts": 0,
                    "successes": 0,
                    "success_rate": 0.0,
                    "scenarios": [],
                }
                groups_by_key[intent_key] = group
                groups.append(group)

            group["attempts"] += int(scenario.attempts)
            group["successes"] += int(scenario.successes)
            group["scenarios"].append({
                "scenario": scenario,
                "scenario_index": scenario_index,
            })

        for group in groups:
            attempts = int(group.get("attempts", 0))
            successes = int(group.get("successes", 0))
            group["success_rate"] = (successes / attempts) if attempts > 0 else 0.0

        return groups

    def build_partial_report_from_history(
        *,
        include_empty: bool = False,
    ) -> Optional[TestReport]:
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

        if not scenario_attempts and not include_empty:
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

        partial_report = TestReport(
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
        warmup_metadata = app.config.get("active_model_warmup_metadata")
        if isinstance(warmup_metadata, ModelWarmupRunMetadata):
            partial_report.model_warmup_run = warmup_metadata.model_copy(
                update={"completed_attempts": overall_attempts}
            )
        return partial_report

    def _fallback_empty_report(suite_name: str) -> TestReport:
        last_config = app.config.get("last_run_config")
        threshold = (
            float(last_config.success_threshold)
            if isinstance(last_config, AppConfig)
            else 0.8
        )
        empty_report = TestReport(
            suite_name=suite_name,
            timestamp=datetime.now(timezone.utc),
            duration_seconds=0.0,
            scenario_results=[],
            overall_attempts=0,
            overall_successes=0,
            overall_failures=0,
            overall_timeouts=0,
            overall_skipped=0,
            overall_success_rate=0.0,
            has_regressions=False,
            regression_threshold=threshold,
        )
        warmup_metadata = app.config.get("active_model_warmup_metadata")
        if isinstance(warmup_metadata, ModelWarmupRunMetadata):
            empty_report.model_warmup_run = warmup_metadata.model_copy(
                update={"completed_attempts": 0}
            )
        return empty_report

    def _extract_suite_name_from_history() -> str:
        progress_emitter = app.config.get("progress_emitter")
        if not isinstance(progress_emitter, ProgressEmitter):
            return "In-progress suite"
        for event in progress_emitter.get_history(limit=200):
            if event.event_type == ProgressEventType.SUITE_STARTED and event.suite_name:
                return event.suite_name
        return "In-progress suite"

    def _extract_progress_counters() -> tuple[int, int]:
        progress_emitter = app.config.get("progress_emitter")
        if not isinstance(progress_emitter, ProgressEmitter):
            return 0, 0
        planned_attempts = 0
        completed_attempts = 0
        for event in progress_emitter.get_history(limit=500):
            if (
                event.event_type == ProgressEventType.SUITE_STARTED
                and isinstance(event.planned_attempts, int)
                and event.planned_attempts > 0
            ):
                planned_attempts = int(event.planned_attempts)
            if event.event_type == ProgressEventType.ATTEMPT_COMPLETED:
                completed_attempts += 1
        if planned_attempts <= 0:
            planned_attempts = completed_attempts
        return planned_attempts, completed_attempts

    def _with_stop_metadata(
        report: TestReport,
        *,
        control: ActiveRunControl,
        force_finalized: bool,
    ) -> TestReport:
        enriched = report.model_copy(deep=True)
        finalized_at = control.stop_finalized_at or datetime.now(timezone.utc)
        requested_at = control.stop_requested_at or finalized_at
        enriched.stopped_by_user = True
        enriched.stop_mode = "immediate"
        enriched.force_finalized = bool(force_finalized)
        enriched.stop_requested_at = requested_at
        enriched.stop_finalized_at = finalized_at
        return enriched

    def _save_report_history(report: TestReport) -> None:
        history_store = app.config.get("history_store")
        if isinstance(history_store, RunHistoryStore):
            try:
                entry = history_store.save_report(report)
                app.config["latest_run_history_entry"] = entry
            except Exception:
                app.config["latest_run_history_entry"] = None

    def _publish_suite_completed_for_stop(report: TestReport) -> None:
        progress_emitter = app.config.get("progress_emitter")
        if not isinstance(progress_emitter, ProgressEmitter):
            return
        planned_attempts, completed_attempts = _extract_progress_counters()
        message = (
            f"Suite stopped early: {report.suite_name} "
            f"after {max(0.0, float(report.duration_seconds)):.1f}s"
        )
        progress_emitter.emit(
            ProgressEvent(
                event_type=ProgressEventType.SUITE_COMPLETED,
                suite_name=report.suite_name,
                message=message,
                duration_seconds=max(0.0, float(report.duration_seconds)),
                planned_attempts=planned_attempts,
                completed_attempts=completed_attempts,
            )
        )

    def _get_active_run_control() -> Optional[ActiveRunControl]:
        candidate = app.config.get("active_run_control")
        if isinstance(candidate, ActiveRunControl):
            return candidate
        return None

    def _is_current_run(control: ActiveRunControl) -> bool:
        return app.config.get("active_run_id") == control.run_id

    def _complete_run_if_current(
        control: ActiveRunControl,
        report: TestReport,
    ) -> bool:
        report_to_store: Optional[TestReport] = None
        with app.config["run_state_lock"]:
            if not _is_current_run(control) or control.finalized:
                return False
            if control.stop_requested_at is not None or report.stopped_by_user:
                if control.stop_finalized_at is None:
                    control.stop_finalized_at = datetime.now(timezone.utc)
                report_to_store = _with_stop_metadata(
                    report,
                    control=control,
                    force_finalized=control.force_finalized,
                )
            else:
                report_to_store = report
            app.config["latest_report"] = report_to_store
            control.finalized = True
            app.config["run_active"] = False
            app.config["stop_requested"] = False
            app.config["active_run_id"] = None
            app.config["active_run_control"] = None
            app.config["stop_event"] = threading.Event()
            app.config["active_model_warmup_metadata"] = None
        if report_to_store is not None:
            _save_report_history(report_to_store)
        return True

    def _force_finalize_run(control: ActiveRunControl) -> TestReport:
        now = datetime.now(timezone.utc)
        with app.config["run_state_lock"]:
            control.stop_requested_at = control.stop_requested_at or now
            control.stop_finalized_at = now
            control.force_finalized = True
        partial_report = build_partial_report_from_history(include_empty=True)
        if partial_report is None:
            partial_report = _fallback_empty_report(_extract_suite_name_from_history())
        elapsed = max(
            0.0,
            (
                now - (control.stop_requested_at or now)
            ).total_seconds(),
        )
        partial_report.duration_seconds = max(
            float(partial_report.duration_seconds),
            elapsed,
        )
        finalized_report = _with_stop_metadata(
            partial_report,
            control=control,
            force_finalized=True,
        )
        with app.config["run_state_lock"]:
            app.config["latest_report"] = finalized_report
            control.finalized = True
            app.config["run_active"] = False
            app.config["stop_requested"] = False
            if _is_current_run(control):
                app.config["active_run_id"] = None
                app.config["active_run_control"] = None
                app.config["stop_event"] = threading.Event()
            app.config["active_model_warmup_metadata"] = None
        _save_report_history(finalized_report)
        _publish_suite_completed_for_stop(finalized_report)
        return finalized_report

    def start_background_run(merged_config: AppConfig, test_suite) -> None:
        """Start a test run in a background thread with fresh run state."""
        progress_emitter = ProgressEmitter()
        run_control = ActiveRunControl(run_id=secrets.token_urlsafe(8))
        with app.config["run_state_lock"]:
            app.config["progress_emitter"] = progress_emitter
            app.config["latest_report"] = None
            app.config["latest_run_history_entry"] = None
            app.config["run_active"] = True
            app.config["stop_requested"] = False
            app.config["stop_event"] = run_control.stop_event
            app.config["active_run_control"] = run_control
            app.config["active_run_id"] = run_control.run_id
            app.config["history_store"] = RunHistoryStore(
                history_dir=merged_config.history_dir,
                max_runs=merged_config.history_max_runs,
                full_json_runs=merged_config.history_full_json_runs,
                gzip_runs=merged_config.history_gzip_runs,
            )
            app.config["active_model_warmup_metadata"] = None

        def run_tests():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                orchestrator = TestOrchestrator(
                    config=merged_config,
                    progress_emitter=progress_emitter,
                    stop_event=run_control.stop_event,
                )
                report = loop.run_until_complete(
                    orchestrator.run_suite(test_suite)
                )
                _complete_run_if_current(run_control, report)
            finally:
                with app.config["run_state_lock"]:
                    if _is_current_run(run_control) and not run_control.finalized:
                        app.config["run_active"] = False
                        app.config["stop_requested"] = False
                        app.config["active_run_id"] = None
                        app.config["active_run_control"] = None
                        app.config["stop_event"] = threading.Event()
                loop.close()

        thread = threading.Thread(target=run_tests, daemon=True)
        run_control.thread = thread
        thread.start()

    def start_background_analytics_run(
        merged_config: AppConfig,
        run_request: AnalyticsJourneyRunRequest,
    ) -> None:
        """Start an analytics-journey run in a background thread."""
        progress_emitter = ProgressEmitter()
        run_control = ActiveRunControl(run_id=secrets.token_urlsafe(8))
        with app.config["run_state_lock"]:
            app.config["progress_emitter"] = progress_emitter
            app.config["latest_report"] = None
            app.config["latest_run_history_entry"] = None
            app.config["run_active"] = True
            app.config["stop_requested"] = False
            app.config["stop_event"] = run_control.stop_event
            app.config["active_run_control"] = run_control
            app.config["active_run_id"] = run_control.run_id
            app.config["history_store"] = RunHistoryStore(
                history_dir=merged_config.history_dir,
                max_runs=merged_config.history_max_runs,
                full_json_runs=merged_config.history_full_json_runs,
                gzip_runs=merged_config.history_gzip_runs,
            )
            app.config["active_model_warmup_metadata"] = None

        def run_tests():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                analytics_store = app.config.get("analytics_journey_store")
                runner = AnalyticsJourneyRunner(
                    config=merged_config,
                    progress_emitter=progress_emitter,
                    stop_event=run_control.stop_event,
                    artifact_store=(
                        analytics_store
                        if isinstance(analytics_store, TranscriptImportStore)
                        else None
                    ),
                )
                report = loop.run_until_complete(runner.run(run_request))
                _complete_run_if_current(run_control, report)

                if isinstance(analytics_store, TranscriptImportStore):
                    app.config["analytics_journey_last_status"] = (
                        analytics_store.load_latest_status()
                    )
            finally:
                with app.config["run_state_lock"]:
                    if _is_current_run(run_control) and not run_control.finalized:
                        app.config["run_active"] = False
                        app.config["stop_requested"] = False
                        app.config["active_run_id"] = None
                        app.config["active_run_control"] = None
                        app.config["stop_event"] = threading.Event()
                loop.close()

        thread = threading.Thread(target=run_tests, daemon=True)
        run_control.thread = thread
        thread.start()

    def start_background_model_warmup_run(
        merged_config: AppConfig,
        run_request: ModelWarmUpRunRequest,
    ) -> None:
        """Start a model warm-up run in a background thread."""
        progress_emitter = ProgressEmitter()
        run_control = ActiveRunControl(run_id=secrets.token_urlsafe(8))
        with app.config["run_state_lock"]:
            app.config["progress_emitter"] = progress_emitter
            app.config["latest_report"] = None
            app.config["latest_run_history_entry"] = None
            app.config["run_active"] = True
            app.config["stop_requested"] = False
            app.config["stop_event"] = run_control.stop_event
            app.config["active_run_control"] = run_control
            app.config["active_run_id"] = run_control.run_id
            app.config["active_model_warmup_metadata"] = build_model_warmup_metadata(
                run_request
            )
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
                runner = ModelWarmUpRunner(
                    config=merged_config,
                    progress_emitter=progress_emitter,
                    stop_event=run_control.stop_event,
                )
                report = loop.run_until_complete(runner.run(run_request))
                _complete_run_if_current(run_control, report)
            finally:
                with app.config["run_state_lock"]:
                    if _is_current_run(run_control) and not run_control.finalized:
                        app.config["run_active"] = False
                        app.config["stop_requested"] = False
                        app.config["active_run_id"] = None
                        app.config["active_run_control"] = None
                        app.config["stop_event"] = threading.Event()
                    if not app.config.get("run_active", False):
                        app.config["active_model_warmup_metadata"] = None
                loop.close()

        thread = threading.Thread(target=run_tests, daemon=True)
        run_control.thread = thread
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

    @app.route("/login", methods=["GET", "POST"])
    def login():
        """Single-operator login for protected web sessions."""
        auth_enabled, expected_username, expected_password, _ = _get_web_auth_settings()
        if not auth_enabled:
            return redirect(url_for("home"))

        next_path = _safe_next_path(request.args.get("next", request.form.get("next", "")))
        if request.method == "POST":
            username = request.form.get("username", "").strip()
            password = request.form.get("password", "")
            valid_username = (
                bool(expected_username)
                and secrets.compare_digest(username, expected_username)
            )
            valid_password = (
                bool(expected_password)
                and secrets.compare_digest(password, expected_password)
            )
            if valid_username and valid_password:
                session.clear()
                session[_AUTH_SESSION_KEY] = True
                session[_AUTH_LAST_ACTIVITY_TS_KEY] = datetime.now(timezone.utc).timestamp()
                session[_CSRF_SESSION_KEY] = secrets.token_urlsafe(32)
                return redirect(next_path or url_for("home"))
            flash("Invalid username or password.")

        return render_template(
            "login.html",
            next_path=next_path or url_for("home"),
            username_hint=expected_username,
        )

    @app.route("/logout", methods=["POST"])
    def logout():
        """Clear authenticated web session."""
        session.clear()
        if _is_web_auth_enabled():
            return redirect(url_for("login"))
        return redirect(url_for("home"))

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
        judge_execution_mode_raw = request.form.get(
            "judge_execution_mode",
            base_config.judge_execution_mode,
        ).strip()
        judge_single_model_raw = request.form.get(
            "judge_single_model",
            base_config.judge_single_model,
        ).strip()
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
        judging_mechanics_enabled = request.form.get("judging_mechanics_enabled") is not None
        journey_dashboard_enabled = request.form.get("journey_dashboard_enabled") is not None
        judging_objective_profile_raw = request.form.get(
            "judging_objective_profile",
            "",
        ).strip()
        judging_strictness_raw = request.form.get("judging_strictness", "").strip()
        judging_tolerance_raw = request.form.get("judging_tolerance", "").strip()
        judging_containment_weight_raw = request.form.get(
            "judging_containment_weight",
            "",
        ).strip()
        judging_fulfillment_weight_raw = request.form.get(
            "judging_fulfillment_weight",
            "",
        ).strip()
        judging_path_weight_raw = request.form.get("judging_path_weight", "").strip()
        judging_explanation_mode_raw = request.form.get(
            "judging_explanation_mode",
            "",
        ).strip()
        attempt_parallel_enabled = request.form.get("attempt_parallel_enabled") is not None
        max_parallel_attempt_workers_raw = request.form.get(
            "max_parallel_attempt_workers",
            "",
        ).strip()
        knowledge_mode_timeout_seconds_raw = request.form.get(
            "knowledge_mode_timeout_seconds",
            "",
        ).strip()
        debug_capture_frames = request.form.get("debug_capture_frames") is not None
        debug_capture_frame_limit = request.form.get("debug_capture_frame_limit", "").strip()

        selected_run_language_for_home = normalize_language_code(
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
                selected_run_language_override=selected_run_language_for_home,
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
                selected_run_language_override=selected_run_language_for_home,
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
                selected_run_language_override=selected_run_language_for_home,
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
                selected_run_language_override=selected_run_language_for_home,
                selected_evaluation_results_language_override=selected_eval_for_home,
            )

        # Merge web overrides with base config
        web_overrides = {}
        if deployment_id:
            web_overrides["gc_deployment_id"] = deployment_id
        if region:
            web_overrides["gc_region"] = region
        try:
            web_overrides["judge_execution_mode"] = normalize_judge_execution_mode(
                judge_execution_mode_raw or base_config.judge_execution_mode
            )
        except ValueError as e:
            return render_home(
                base_config,
                errors=[str(e)],
                active_home_tab="harness",
                selected_run_language_override=selected_run_language_for_home,
                selected_evaluation_results_language_override=selected_eval_for_home,
            )
        try:
            web_overrides["judge_single_model"] = normalize_gemma_single_model(
                judge_single_model_raw or base_config.judge_single_model
            )
        except ValueError as e:
            return render_home(
                base_config,
                errors=[str(e)],
                active_home_tab="harness",
                selected_run_language_override=selected_run_language_for_home,
                selected_evaluation_results_language_override=selected_eval_for_home,
            )
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
                    selected_run_language_override=selected_run_language_for_home,
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
                    selected_run_language_override=selected_run_language_for_home,
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
                    selected_run_language_override=selected_run_language_for_home,
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
                    selected_run_language_override=selected_run_language_for_home,
                    selected_evaluation_results_language_override=selected_eval_for_home,
                )
            if evaluation_results_language_override is not None:
                web_overrides["evaluation_results_language"] = (
                    evaluation_results_language_override
                )
        web_overrides["debug_capture_frames"] = debug_capture_frames
        if (
            "attempt_parallel_enabled" in request.form
            or "max_parallel_attempt_workers" in request.form
        ):
            web_overrides["attempt_parallel_enabled"] = attempt_parallel_enabled
        web_overrides["judging_mechanics_enabled"] = judging_mechanics_enabled
        web_overrides["journey_dashboard_enabled"] = journey_dashboard_enabled
        if debug_capture_frame_limit:
            web_overrides["debug_capture_frame_limit"] = debug_capture_frame_limit
        if max_parallel_attempt_workers_raw:
            web_overrides["max_parallel_attempt_workers"] = max_parallel_attempt_workers_raw
        if knowledge_mode_timeout_seconds_raw:
            web_overrides["knowledge_mode_timeout_seconds"] = (
                knowledge_mode_timeout_seconds_raw
            )
        if judging_objective_profile_raw:
            web_overrides["judging_objective_profile"] = judging_objective_profile_raw
        if judging_strictness_raw:
            web_overrides["judging_strictness"] = judging_strictness_raw
        if judging_tolerance_raw:
            web_overrides["judging_tolerance"] = judging_tolerance_raw
        if judging_containment_weight_raw:
            web_overrides["judging_containment_weight"] = judging_containment_weight_raw
        if judging_fulfillment_weight_raw:
            web_overrides["judging_fulfillment_weight"] = judging_fulfillment_weight_raw
        if judging_path_weight_raw:
            web_overrides["judging_path_weight"] = judging_path_weight_raw
        if judging_explanation_mode_raw:
            web_overrides["judging_explanation_mode"] = judging_explanation_mode_raw

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
                selected_run_language_override=selected_run_language_for_home,
                selected_evaluation_results_language_override=selected_eval_for_home,
            )

        # Validate Ollama connectivity and model before starting long test runs.
        try:
            build_judge_execution_client(merged_config).verify_connection()
        except JudgeLLMError as e:
            return render_home(
                base_config,
                errors=[str(e)],
                active_home_tab="harness",
                selected_run_language_override=selected_run_language_for_home,
                selected_evaluation_results_language_override=selected_eval_for_home,
            )

        app.config["last_run_config"] = merged_config.model_copy(deep=True)
        app.config["last_run_suite"] = test_suite.model_copy(deep=True)
        start_background_run(merged_config, test_suite)

        return redirect(url_for("results"))

    @app.route("/run/model_warm_up", methods=["POST"])
    def run_model_warm_up():
        """Trigger a transport-only model warm-up run."""
        if app.config.get("run_active", False):
            return redirect(url_for("results"))

        base_config = load_app_config()
        deployment_id = request.form.get("model_warmup_deployment_id", "").strip()
        region = request.form.get("model_warmup_region", "").strip()
        recorded_model = request.form.get("model_warmup_llm_model", "").strip()
        execution_mode_raw = request.form.get(
            "model_warmup_execution_mode",
            "serial",
        ).strip()
        worker_count_raw = request.form.get("model_warmup_parallel_workers", "1").strip()
        pacing_raw = request.form.get("model_warmup_pacing_seconds", "2.5").strip()

        errors: list[str] = []
        if not deployment_id:
            errors.append("Deployment ID is required for Model Warm Up.")
        if not region:
            errors.append("Region is required for Model Warm Up.")
        try:
            execution_mode = normalize_model_warmup_execution_mode(execution_mode_raw)
        except ValueError as e:
            errors.append(str(e))
            execution_mode = "serial"
        try:
            worker_count_unclamped = int(worker_count_raw)
        except (TypeError, ValueError):
            errors.append("Model Warm Up parallel workers must be a number.")
            worker_count = 1
        else:
            if worker_count_unclamped < 1 or worker_count_unclamped > 5:
                errors.append("Model Warm Up parallel workers must be between 1 and 5.")
            worker_count = normalize_model_warmup_workers(worker_count_unclamped)
        try:
            pacing_seconds = normalize_model_warmup_pacing(pacing_raw)
        except ValueError as e:
            errors.append(str(e))
            pacing_seconds = 2.5

        if errors:
            return render_home(
                base_config,
                errors=errors,
                active_home_tab="model_warm_up",
            )

        merged_config = merge_config(
            base_config,
            {
                "gc_deployment_id": deployment_id,
                "gc_region": region,
            },
        )
        run_request = ModelWarmUpRunRequest(
            deployment_id=deployment_id,
            region=region,
            recorded_model=recorded_model or None,
            execution_mode=execution_mode,
            worker_count=worker_count,
            pacing_seconds=pacing_seconds,
        )
        app.config["last_run_config"] = merged_config.model_copy(deep=True)
        app.config["last_run_suite"] = None
        start_background_model_warmup_run(merged_config, run_request)

        return redirect(url_for("results"))

    @app.route("/run/analytics_journey", methods=["POST"])
    def run_analytics_journey():
        """Trigger analytics journey evaluate-now regression run."""
        if app.config.get("run_active", False):
            return redirect(url_for("results"))

        base_config = load_app_config()
        bot_flow_id = request.form.get("analytics_bot_flow_id", "").strip()
        interval = request.form.get("analytics_interval", "").strip()
        divisions_raw = request.form.get("analytics_divisions", "").strip()
        analytics_auth_mode_raw = request.form.get(
            "analytics_auth_mode",
            base_config.analytics_journey_auth_mode,
        ).strip()
        analytics_bearer_token_raw = request.form.get(
            "analytics_bearer_token",
            "",
        ).strip()
        analytics_region_raw = request.form.get("analytics_region", "").strip()
        analytics_client_id_raw = request.form.get(
            "analytics_gc_client_id",
            "",
        ).strip()
        analytics_client_secret_raw = request.form.get(
            "analytics_gc_client_secret",
            "",
        ).strip()
        analytics_judge_execution_mode_raw = request.form.get(
            "analytics_judge_execution_mode",
            base_config.analytics_judge_execution_mode,
        ).strip()
        analytics_judge_single_model_raw = request.form.get(
            "analytics_judge_single_model",
            base_config.analytics_judge_single_model,
        ).strip()
        analytics_ollama_model_raw = request.form.get(
            "analytics_ollama_model",
            "",
        ).strip()
        page_size_raw = request.form.get(
            "analytics_page_size",
            str(base_config.analytics_journey_default_page_size),
        ).strip()
        max_conversations_raw = request.form.get(
            "analytics_max_conversations",
            str(base_config.analytics_journey_default_max_conversations),
        ).strip()
        language_filter_raw = request.form.get("analytics_language_filter", "").strip()
        filter_json_raw = request.form.get("analytics_filter_json", "").strip()
        analytics_enabled = request.form.get("analytics_journey_enabled") is not None
        language_raw = request.form.get("language", "").strip()
        evaluation_results_language_raw = request.form.get(
            "evaluation_results_language",
            "",
        ).strip()

        selected_run_language_for_home = normalize_language_code(
            language_raw or base_config.language or "en",
            default="en",
        )
        selected_eval_for_home = normalize_evaluation_results_language(
            evaluation_results_language_raw
            or base_config.evaluation_results_language
            or "inherit",
            default="inherit",
        ) or "inherit"
        try:
            selected_analytics_auth_mode_for_home = normalize_analytics_auth_mode(
                analytics_auth_mode_raw or base_config.analytics_journey_auth_mode
            )
        except ValueError:
            selected_analytics_auth_mode_for_home = (
                ANALYTICS_AUTH_MODE_CLIENT_CREDENTIALS
            )

        if not interval:
            return render_home(
                base_config,
                errors=["Interval/date range is required for analytics journey runs."],
                active_home_tab="analytics",
                selected_run_language_override=selected_run_language_for_home,
                selected_evaluation_results_language_override=selected_eval_for_home,
                selected_analytics_auth_mode_override=(
                    selected_analytics_auth_mode_for_home
                ),
            )
        if not bot_flow_id:
            return render_home(
                base_config,
                errors=["Bot Flow ID is required for analytics journey runs."],
                active_home_tab="analytics",
                selected_run_language_override=selected_run_language_for_home,
                selected_evaluation_results_language_override=selected_eval_for_home,
                selected_analytics_auth_mode_override=(
                    selected_analytics_auth_mode_for_home
                ),
            )

        try:
            analytics_auth_mode = normalize_analytics_auth_mode(
                analytics_auth_mode_raw or base_config.analytics_journey_auth_mode
            )
        except ValueError as e:
            return render_home(
                base_config,
                errors=[str(e)],
                active_home_tab="analytics",
                selected_run_language_override=selected_run_language_for_home,
                selected_evaluation_results_language_override=selected_eval_for_home,
                selected_analytics_auth_mode_override=(
                    selected_analytics_auth_mode_for_home
                ),
            )

        try:
            page_size = max(1, int(page_size_raw))
            max_conversations = max(1, int(max_conversations_raw))
        except ValueError:
            return render_home(
                base_config,
                errors=["Page size and max conversations must be positive integers."],
                active_home_tab="analytics",
                selected_run_language_override=selected_run_language_for_home,
                selected_evaluation_results_language_override=selected_eval_for_home,
                selected_analytics_auth_mode_override=(
                    selected_analytics_auth_mode_for_home
                ),
            )

        language_override: Optional[str] = None
        if language_raw:
            try:
                language_override = normalize_language_code(language_raw, default="en")
            except ValueError as e:
                return render_home(
                    base_config,
                    errors=[str(e)],
                    active_home_tab="analytics",
                    selected_run_language_override=selected_run_language_for_home,
                    selected_evaluation_results_language_override=selected_eval_for_home,
                    selected_analytics_auth_mode_override=(
                        selected_analytics_auth_mode_for_home
                    ),
                )
        evaluation_results_language_override: Optional[str] = None
        if evaluation_results_language_raw:
            try:
                evaluation_results_language_override = normalize_evaluation_results_language(
                    evaluation_results_language_raw,
                    default="inherit",
                )
            except ValueError as e:
                return render_home(
                    base_config,
                    errors=[str(e)],
                    active_home_tab="analytics",
                    selected_run_language_override=selected_run_language_for_home,
                    selected_evaluation_results_language_override=selected_eval_for_home,
                    selected_analytics_auth_mode_override=(
                        selected_analytics_auth_mode_for_home
                    ),
                )

        parsed_filter: dict[str, object] = {}
        if filter_json_raw:
            try:
                parsed_filter = parse_filter_json(filter_json_raw)
            except ValueError as e:
                return render_home(
                    base_config,
                    errors=[f"Advanced analytics filter JSON is invalid: {e}"],
                    active_home_tab="analytics",
                    selected_run_language_override=selected_run_language_for_home,
                    selected_evaluation_results_language_override=selected_eval_for_home,
                    selected_analytics_auth_mode_override=(
                        selected_analytics_auth_mode_for_home
                    ),
                )

        normalized_language_filter: Optional[str] = None
        if language_filter_raw:
            try:
                normalized_language_filter = normalize_language_code(
                    language_filter_raw,
                    allow_none=True,
                )
            except ValueError as e:
                return render_home(
                    base_config,
                    errors=[str(e)],
                    active_home_tab="analytics",
                    selected_run_language_override=selected_run_language_for_home,
                    selected_evaluation_results_language_override=selected_eval_for_home,
                    selected_analytics_auth_mode_override=(
                        selected_analytics_auth_mode_for_home
                    ),
                )
            normalized_language_filter = (
                str(normalized_language_filter)
                if normalized_language_filter
                else None
            )

        web_overrides: dict[str, object] = {
            "analytics_journey_enabled": analytics_enabled,
            "analytics_journey_auth_mode": analytics_auth_mode,
            "analytics_journey_default_page_size": page_size,
            "analytics_journey_default_max_conversations": max_conversations,
        }
        if analytics_region_raw:
            web_overrides["gc_region"] = analytics_region_raw
        if analytics_client_id_raw:
            web_overrides["gc_client_id"] = analytics_client_id_raw
        if analytics_client_secret_raw:
            web_overrides["gc_client_secret"] = analytics_client_secret_raw
        try:
            web_overrides["analytics_judge_execution_mode"] = (
                normalize_judge_execution_mode(
                    analytics_judge_execution_mode_raw
                    or base_config.analytics_judge_execution_mode
                )
            )
        except ValueError as e:
            return render_home(
                base_config,
                errors=[str(e)],
                active_home_tab="analytics",
                selected_run_language_override=selected_run_language_for_home,
                selected_evaluation_results_language_override=selected_eval_for_home,
                selected_analytics_auth_mode_override=(
                    selected_analytics_auth_mode_for_home
                ),
            )
        try:
            web_overrides["analytics_judge_single_model"] = (
                normalize_gemma_single_model(
                    analytics_judge_single_model_raw
                    or base_config.analytics_judge_single_model
                )
            )
        except ValueError as e:
            return render_home(
                base_config,
                errors=[str(e)],
                active_home_tab="analytics",
                selected_run_language_override=selected_run_language_for_home,
                selected_evaluation_results_language_override=selected_eval_for_home,
                selected_analytics_auth_mode_override=(
                    selected_analytics_auth_mode_for_home
                ),
            )
        if analytics_ollama_model_raw:
            web_overrides["analytics_journey_judge_model"] = analytics_ollama_model_raw
        if language_override:
            web_overrides["language"] = language_override
        if evaluation_results_language_override is not None:
            web_overrides["evaluation_results_language"] = (
                evaluation_results_language_override
            )
        if normalized_language_filter:
            web_overrides["analytics_journey_default_language_filter"] = (
                normalized_language_filter
            )

        merged_config = merge_config(base_config, web_overrides)
        page_size = max(
            1,
            min(
                int(page_size),
                int(merged_config.analytics_journey_details_page_size_cap),
            ),
        )
        effective_language = resolve_effective_language(
            runtime_override=language_override,
            suite_language=None,
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

        if not merged_config.analytics_journey_enabled:
            return render_home(
                base_config,
                errors=[
                    (
                        "Analytics Journey mode is disabled. "
                        "Enable the feature toggle in this form to run."
                    )
                ],
                active_home_tab="analytics",
                selected_run_language_override=selected_run_language_for_home,
                selected_evaluation_results_language_override=selected_eval_for_home,
                selected_analytics_auth_mode_override=(
                    selected_analytics_auth_mode_for_home
                ),
            )

        missing = []
        if not merged_config.gc_region:
            missing.append("gc_region")
        if analytics_auth_mode == ANALYTICS_AUTH_MODE_CLIENT_CREDENTIALS:
            if not merged_config.gc_client_id:
                missing.append("gc_client_id")
            if not merged_config.gc_client_secret:
                missing.append("gc_client_secret")
        if (
            analytics_auth_mode == ANALYTICS_AUTH_MODE_MANUAL_BEARER
            and not analytics_bearer_token_raw
        ):
            missing.append("analytics_bearer_token")
        if missing:
            if "analytics_bearer_token" in missing:
                missing = [
                    "manual_bearer_token"
                    if field_name == "analytics_bearer_token"
                    else field_name
                    for field_name in missing
                ]
            return render_home(
                base_config,
                errors=[
                    "Missing required configuration for analytics journey: "
                    + ", ".join(missing)
                ],
                active_home_tab="analytics",
                selected_run_language_override=selected_run_language_for_home,
                selected_evaluation_results_language_override=selected_eval_for_home,
                selected_analytics_auth_mode_override=(
                    selected_analytics_auth_mode_for_home
                ),
            )

        try:
            build_judge_execution_client(merged_config, analytics=True).verify_connection()
        except JudgeLLMError as e:
            return render_home(
                base_config,
                errors=[str(e)],
                active_home_tab="analytics",
                selected_run_language_override=selected_run_language_for_home,
                selected_evaluation_results_language_override=selected_eval_for_home,
                selected_analytics_auth_mode_override=(
                    selected_analytics_auth_mode_for_home
                ),
            )

        divisions = [
            token.strip()
            for token in divisions_raw.split(",")
            if token.strip()
        ]
        run_request = AnalyticsJourneyRunRequest(
            bot_flow_id=bot_flow_id,
            interval=interval,
            page_size=page_size,
            max_conversations=max_conversations,
            auth_mode=analytics_auth_mode,
            manual_bearer_token=analytics_bearer_token_raw or None,
            divisions=divisions,
            language_filter=(
                normalized_language_filter
                or merged_config.analytics_journey_default_language_filter
            ),
            extra_query_params=parsed_filter,
        )

        app.config["last_run_config"] = merged_config.model_copy(deep=True)
        app.config["last_run_suite"] = None
        start_background_analytics_run(merged_config, run_request)
        return redirect(url_for("results"))

    @app.route("/run/analytics_journey/token", methods=["POST"])
    def capture_analytics_journey_token():
        """Capture an AJR bearer token from region/client credentials."""
        if not _validate_csrf_token():
            return jsonify({"error": "Invalid or missing CSRF token."}), 400

        region = request.form.get("analytics_region", "").strip()
        client_id = request.form.get("analytics_gc_client_id", "").strip()
        client_secret = request.form.get("analytics_gc_client_secret", "").strip()

        missing: list[str] = []
        if not region:
            missing.append("analytics_region")
        if not client_id:
            missing.append("analytics_gc_client_id")
        if not client_secret:
            missing.append("analytics_gc_client_secret")
        if missing:
            return (
                jsonify(
                    {
                        "error": "Missing required fields for token capture.",
                        "missing": missing,
                    }
                ),
                400,
            )

        oauth_url = f"https://login.{region}/oauth/token"
        timeout_seconds = int(load_app_config().response_timeout or 30)
        try:
            response = requests.post(
                oauth_url,
                data={"grant_type": "client_credentials"},
                auth=(client_id, client_secret),
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json",
                },
                timeout=timeout_seconds,
            )
            response.raise_for_status()
        except requests.HTTPError:
            status_code = int(getattr(response, "status_code", 0) or 0)
            if status_code in {401, 403}:
                return (
                    jsonify(
                        {
                            "error": (
                                "Token request was not authorized. "
                                "Verify region, client ID, client secret, and permissions."
                            )
                        }
                    ),
                    status_code,
                )
            return (
                jsonify(
                    {
                        "error": "Token request failed. Verify region and OAuth credentials.",
                    }
                ),
                502,
            )
        except requests.RequestException:
            return (
                jsonify(
                    {
                        "error": "Token request failed. Verify region and OAuth credentials.",
                    }
                ),
                502,
            )

        try:
            payload = response.json()
        except ValueError:
            return jsonify({"error": "Token response was not valid JSON."}), 502

        access_token = payload.get("access_token")
        if not isinstance(access_token, str) or not access_token.strip():
            return jsonify({"error": "Token response missing access_token."}), 502

        token_type = str(payload.get("token_type") or "Bearer").strip() or "Bearer"
        raw_expires = payload.get("expires_in", 0)
        try:
            expires_in = max(0, int(raw_expires))
        except (TypeError, ValueError):
            expires_in = 0

        return jsonify(
            {
                "access_token": access_token.strip(),
                "token_type": token_type,
                "expires_in": expires_in,
                "issued_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )

    def _run_analytics_api_connectivity_test(
        *,
        forced_auth_mode: Optional[str] = None,
    ):
        if not _validate_csrf_token():
            return jsonify({"error": "Invalid or missing CSRF token."}), 400

        base_config = load_app_config()
        region = request.form.get("analytics_region", "").strip()
        bot_flow_id = request.form.get("analytics_bot_flow_id", "").strip()
        interval = request.form.get("analytics_interval", "").strip()
        divisions_raw = request.form.get("analytics_divisions", "").strip()
        page_size_raw = request.form.get(
            "analytics_page_size",
            str(base_config.analytics_journey_default_page_size),
        ).strip()
        language_filter_raw = request.form.get("analytics_language_filter", "").strip()
        filter_json_raw = request.form.get("analytics_filter_json", "").strip()
        analytics_auth_mode_raw = request.form.get(
            "analytics_auth_mode",
            base_config.analytics_journey_auth_mode,
        ).strip()
        analytics_bearer_token_raw = request.form.get(
            "analytics_bearer_token",
            "",
        ).strip()
        analytics_client_id_raw = request.form.get(
            "analytics_gc_client_id",
            "",
        ).strip()
        analytics_client_secret_raw = request.form.get(
            "analytics_gc_client_secret",
            "",
        ).strip()

        resolved_auth_mode_raw = (
            forced_auth_mode
            if forced_auth_mode is not None
            else (analytics_auth_mode_raw or base_config.analytics_journey_auth_mode)
        )
        try:
            analytics_auth_mode = normalize_analytics_auth_mode(resolved_auth_mode_raw)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        try:
            page_size = max(1, int(page_size_raw))
        except ValueError:
            return jsonify({"error": "Page size must be a positive integer."}), 400
        page_size = max(
            1,
            min(
                int(page_size),
                int(base_config.analytics_journey_details_page_size_cap),
            ),
        )

        missing: list[str] = []
        if not region:
            missing.append("analytics_region")
        if not bot_flow_id:
            missing.append("analytics_bot_flow_id")
        if not interval:
            missing.append("analytics_interval")
        if analytics_auth_mode == ANALYTICS_AUTH_MODE_MANUAL_BEARER:
            if not analytics_bearer_token_raw:
                missing.append("analytics_bearer_token")
        else:
            if not analytics_client_id_raw:
                missing.append("analytics_gc_client_id")
            if not analytics_client_secret_raw:
                missing.append("analytics_gc_client_secret")
        if missing:
            return (
                jsonify(
                    {
                        "error": "Missing required fields for analytics API test.",
                        "missing": missing,
                    }
                ),
                400,
            )

        parsed_filter: dict[str, object] = {}
        if filter_json_raw:
            try:
                parsed_filter = parse_filter_json(filter_json_raw)
            except ValueError as e:
                return (
                    jsonify(
                        {"error": f"Advanced analytics filter JSON is invalid: {e}"}
                    ),
                    400,
                )

        normalized_language_filter: Optional[str] = None
        if language_filter_raw:
            try:
                normalized_language_filter = normalize_language_code(
                    language_filter_raw,
                    allow_none=True,
                )
            except ValueError as e:
                return jsonify({"error": str(e)}), 400
            normalized_language_filter = (
                str(normalized_language_filter)
                if normalized_language_filter
                else None
            )

        divisions = [
            token.strip()
            for token in divisions_raw.split(",")
            if token.strip()
        ]

        client = GenesysAnalyticsJourneyClient(
            region=region,
            client_id=analytics_client_id_raw,
            client_secret=analytics_client_secret_raw,
            auth_mode=analytics_auth_mode,
            manual_bearer_token=analytics_bearer_token_raw or None,
            timeout=int(base_config.response_timeout),
            page_size_cap=int(base_config.analytics_journey_details_page_size_cap),
        )
        sanitized_extra_params, ignored_extra_params = (
            client.sanitize_extra_query_params(parsed_filter)
        )

        request_context = {
            "region": region,
            "auth_mode": analytics_auth_mode,
            "forced_auth_mode": forced_auth_mode,
            "bot_flow_id": bot_flow_id,
            "interval": interval,
            "page_size": page_size,
            "divisions_count": len(divisions),
            "language_filter": normalized_language_filter,
            "applied_query_param_keys": sorted(sanitized_extra_params.keys()),
            "ignored_query_param_keys": ignored_extra_params,
        }

        def _resolve_api_error_status(raw_error: str) -> int:
            match = re.search(r"\b([45]\d{2})\b", str(raw_error or ""))
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    pass
            normalized = str(raw_error or "").lower()
            if "unauthorized" in normalized:
                return 401
            if "forbidden" in normalized:
                return 403
            if "not found" in normalized:
                return 404
            if "rate limit" in normalized or "too many requests" in normalized:
                return 429
            return 502

        def _build_api_error_guidance(status_code: int) -> tuple[str, list[str]]:
            if status_code == 401:
                return (
                    "OAuth token/authentication failed (401).",
                    [
                        "Verify Region exactly matches your org (for example, usw2.pure.cloud).",
                        "Verify Client ID and Client Secret are valid and active.",
                        "If using manual bearer, capture a fresh token and test again.",
                    ],
                )
            if status_code == 403:
                return (
                    "Access denied by Genesys API (403).",
                    [
                        "In Admin > OAuth > [your client] > Roles, ensure this exact role is assigned to the OAuth client.",
                        "Grant permission `Analytics > botFlowReportingTurn > View` to the OAuth client/user role.",
                        "Confirm role has access to the divisions containing this bot flow's data.",
                        "Verify botFlowId belongs to the same org and region as the token.",
                    ],
                )
            if status_code == 404:
                return (
                    "Bot flow endpoint/resource was not found (404).",
                    [
                        "Confirm Bot Flow ID is correct and exists in this region/org.",
                        "Confirm Region input is correct for that bot flow.",
                    ],
                )
            if status_code == 429:
                return (
                    "Genesys API rate-limited this request (429).",
                    [
                        "Retry after a short delay.",
                        "Reduce page size or concurrency when running repeated tests.",
                    ],
                )
            if status_code >= 500:
                return (
                    "Genesys API returned a server error (5xx).",
                    [
                        "Retry the test; this may be transient upstream instability.",
                        "If persistent, capture timestamp + request context and open a Genesys support case.",
                    ],
                )
            return (
                "Analytics API test failed with an upstream error.",
                [
                    "Review details and verify region, auth mode, interval, and botFlowId.",
                    "Retry with a smaller interval to isolate payload/data availability issues.",
                ],
            )

        def _to_text(value) -> str:
            if isinstance(value, str):
                return value.strip()
            if isinstance(value, dict):
                for key in ("text", "message", "prompt", "value", "utterance"):
                    token = _to_text(value.get(key))
                    if token:
                        return token
                return ""
            if isinstance(value, list):
                for item in value:
                    token = _to_text(item)
                    if token:
                        return token
            return ""

        def _to_text_list(value) -> list[str]:
            if isinstance(value, str):
                text = value.strip()
                return [text] if text else []
            if isinstance(value, dict):
                for key in ("prompts", "items", "messages"):
                    if key in value:
                        return _to_text_list(value.get(key))
                token = _to_text(value)
                return [token] if token else []
            if isinstance(value, list):
                result: list[str] = []
                for item in value:
                    result.extend(_to_text_list(item))
                return [item for item in result if item]
            return []

        def _extract_user_input(row: dict) -> str:
            for key in ("userInput", "userinput", "input", "utterance"):
                token = _to_text(row.get(key))
                if token:
                    return token
            for key in ("turn", "event", "data"):
                nested = row.get(key)
                if not isinstance(nested, dict):
                    continue
                for nested_key in ("userInput", "userinput", "input", "utterance"):
                    token = _to_text(nested.get(nested_key))
                    if token:
                        return token
            return ""

        def _extract_bot_prompts(row: dict) -> list[str]:
            for key in ("botPrompts", "botprompts", "prompts", "botPrompt"):
                values = _to_text_list(row.get(key))
                if values:
                    return values
            for key in ("turn", "event", "data"):
                nested = row.get(key)
                if not isinstance(nested, dict):
                    continue
                for nested_key in ("botPrompts", "botprompts", "prompts", "botPrompt"):
                    values = _to_text_list(nested.get(nested_key))
                    if values:
                        return values
            return []

        started_at = time.monotonic()
        try:
            payload = client.fetch_reporting_turns_page(
                bot_flow_id=bot_flow_id,
                interval=interval,
                page_size=page_size,
                page_number=1,
                divisions=divisions,
                language_filter=normalized_language_filter,
                extra_params=parsed_filter,
            )
        except GenesysAnalyticsJourneyError as e:
            raw_error = str(e)
            upstream_debug_raw = dict(getattr(e, "metadata", {}) or {})
            status_code = int(
                upstream_debug_raw.get("status_code")
                or _resolve_api_error_status(raw_error)
            )
            summary, guidance = _build_api_error_guidance(status_code)
            error_class = {
                401: "oauth_or_token_invalid",
                403: "permission_or_division_access",
                404: "bot_flow_or_region_not_found",
                429: "rate_limited",
            }.get(
                status_code,
                "upstream_api_error",
            )
            upstream_debug: dict[str, object] = {}
            for key in (
                "status_code",
                "method",
                "path",
                "url",
                "correlation_id",
                "content_type",
                "retry_after",
                "attempt",
                "max_attempts",
                "response_body_excerpt",
            ):
                value = upstream_debug_raw.get(key)
                if value in (None, ""):
                    continue
                upstream_debug[key] = value
            return (
                jsonify(
                    {
                        "error": "Analytics API test failed.",
                        "user_message": summary,
                        "error_class": error_class,
                        "status_code": status_code,
                        "guidance": guidance,
                        "request_context": request_context,
                        "upstream_debug": upstream_debug,
                        "details": raw_error,
                    }
                ),
                status_code,
            )

        duration_ms = round((time.monotonic() - started_at) * 1000.0, 2)
        rows = client.extract_rows(payload)
        if hasattr(client, "filter_rows_by_language"):
            matching_rows, language_filter_stats = client.filter_rows_by_language(
                rows,
                normalized_language_filter,
            )
        else:
            matching_rows = [
                row
                for row in rows
                if client.row_matches_language(row, normalized_language_filter)
            ]
            language_filter_stats = {
                "language_filter": normalized_language_filter or None,
                "eligible_conversations": len(
                    {
                        conversation_id
                        for row in matching_rows
                        for conversation_id in [client.extract_conversation_id(row)]
                        if conversation_id
                    }
                ),
                "selected_conversations": len(
                    {
                        conversation_id
                        for row in matching_rows
                        for conversation_id in [client.extract_conversation_id(row)]
                        if conversation_id
                    }
                ),
                "excluded_missing_language_conversations": 0,
                "excluded_mismatched_conversations": 0,
            }
        conversation_ids = sorted(
            {
                conversation_id
                for row in matching_rows
                for conversation_id in [client.extract_conversation_id(row)]
                if conversation_id
            }
        )
        next_uri = client.extract_next_uri(payload)
        warnings: list[str] = []
        if len(rows) == 0:
            warnings.append(
                "No reporting-turn rows returned for page 1. Check interval, botFlowId, divisions, and language filter."
            )
        elif len(matching_rows) == 0 and normalized_language_filter:
            warnings.append(
                "Rows were returned, but no complete conversations matched the selected language filter."
            )

        turns_with_user_input = 0
        turns_with_bot_prompts = 0
        turns_with_both = 0
        sample_turn_pairs: list[dict[str, object]] = []
        for row in matching_rows:
            user_input = _extract_user_input(row)
            bot_prompts = _extract_bot_prompts(row)
            if user_input:
                turns_with_user_input += 1
            if bot_prompts:
                turns_with_bot_prompts += 1
            if user_input and bot_prompts:
                turns_with_both += 1
            if len(sample_turn_pairs) >= 5:
                continue
            if not user_input and not bot_prompts:
                continue
            sample_turn_pairs.append(
                {
                    "conversation_id": client.extract_conversation_id(row),
                    "user_input": user_input,
                    "bot_prompts": bot_prompts[:3],
                    "intent": _to_text(row.get("intent")),
                    "ask_action": _to_text(row.get("askAction")),
                }
            )

        return jsonify(
            {
                "ok": True,
                "message": (
                    f"Analytics API test succeeded. Page 1 returned "
                    f"{len(rows)} rows ({len(conversation_ids)} unique conversations)."
                ),
                "request": request_context,
                "result": {
                    "duration_ms": duration_ms,
                    "rows_count": len(rows),
                    "matching_rows_count": len(matching_rows),
                    "unique_conversations": len(conversation_ids),
                    "language_filter_stats": language_filter_stats,
                    "sample_conversation_ids": conversation_ids[:5],
                    "next_page_available": bool(next_uri),
                    "turn_parsing": {
                        "rows_scanned": len(matching_rows),
                        "rows_with_user_input": turns_with_user_input,
                        "rows_with_bot_prompts": turns_with_bot_prompts,
                        "rows_with_both": turns_with_both,
                        "sample_turn_pairs": sample_turn_pairs,
                    },
                },
                "warnings": warnings,
            }
        )

    @app.route("/run/analytics_journey/test", methods=["POST"])
    def test_analytics_journey_api():
        """Execute a single-page AJR API connectivity test with current form auth."""
        return _run_analytics_api_connectivity_test()

    @app.route("/run/analytics_journey/test/client_credentials", methods=["POST"])
    def test_analytics_journey_api_client_credentials():
        """Execute a single-page AJR API test forcing OAuth client_credentials mode."""
        return _run_analytics_api_connectivity_test(
            forced_auth_mode=ANALYTICS_AUTH_MODE_CLIENT_CREDENTIALS
        )

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
                judge = build_judge_execution_client(merged_config)

                def llm_classifier(message: str, categories: list[dict]) -> dict:
                    nonlocal classifier_error_logged
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
            build_judge_execution_client(merged_config).verify_connection()
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
            build_judge_execution_client(merged_config).verify_connection()
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
        """Force-stop the active run immediately (kill-switch semantics)."""
        wants_json = "application/json" in str(request.headers.get("Accept") or "")
        if app.config.get("run_active", False):
            control = _get_active_run_control()
            if control is None:
                message = "No active run control found."
                if wants_json:
                    return jsonify({"stopped": False, "message": message}), 409
                flash(message)
                return redirect(url_for("results"))

            now = datetime.now(timezone.utc)
            with app.config["run_state_lock"]:
                if control.stop_requested_at is None:
                    control.stop_requested_at = now
                app.config["stop_requested"] = True
            control.stop_event.set()
            finalized_report = _force_finalize_run(control)
            message = (
                "Run stopped by user. Partial results were finalized immediately."
            )
            if wants_json:
                return jsonify(
                    {
                        "stopped": True,
                        "message": message,
                        "run_active": bool(app.config.get("run_active", False)),
                        "force_finalized": bool(finalized_report.force_finalized),
                        "stop_mode": finalized_report.stop_mode,
                    }
                )
            flash(message)
        else:
            message = "No active run to stop."
            if wants_json:
                return jsonify({"stopped": False, "message": message}), 409
            flash(message)

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
        journey_view = normalize_journey_view(
            request.args.get("journey_view", "overview")
        )
        dashboard_context = build_dashboard_context(
            report,
            baseline_run_id=baseline_run_id,
            journey_view=journey_view,
        )
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
            intent_groups=build_intent_groups(report),
            partial_report=partial_report,
            run_active=run_active,
            stop_requested=stop_requested,
            has_rerun=has_rerun,
            progress_history=progress_history,
            dashboard_metrics=dashboard_context.get("metrics"),
            dashboard_history_count=dashboard_context.get("history_count", 0),
            baseline_options=dashboard_context.get("baseline_options", []),
            selected_baseline_run_id=dashboard_context.get("selected_baseline_run_id"),
            selected_journey_view=dashboard_context.get("journey_view", journey_view),
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
        journey_view = normalize_journey_view(
            request.args.get("journey_view", "overview")
        )

        if fmt == "csv":
            content = export_csv(report)
            return Response(
                content,
                mimetype="text/csv",
                headers={
                    "Content-Disposition": "attachment; filename=report.csv"
                },
            )
        elif fmt == "failures_csv":
            content = export_failures_csv(report)
            return Response(
                content,
                mimetype="text/csv",
                headers={
                    "Content-Disposition": "attachment; filename=report-failures.csv"
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
            dashboard_context = build_dashboard_context(
                report,
                baseline_run_id=baseline_run_id,
                journey_view=journey_view,
            )
            metrics = dashboard_context.get("metrics") or build_dashboard_metrics(report)
            content = export_dashboard_pdf(
                report,
                metrics,
                language_code=resolve_results_language_code(),
                selected_journey_view=journey_view,
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
