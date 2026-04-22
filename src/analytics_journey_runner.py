"""Analytics Journey Regression runner (reporting-turns evaluate-now)."""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

from .genesys_analytics_journey_client import (
    GenesysAnalyticsJourneyClient,
    GenesysAnalyticsJourneyError,
)
from .journey_mode import (
    CATEGORY_STRATEGY_LLM_FIRST,
    load_category_overrides,
)
from .journey_regression import (
    resolve_category_with_strategy,
    resolve_primary_categories,
)
from .journey_taxonomy import build_journey_taxonomy_rollups, load_taxonomy_overrides
from .judge_llm import JudgeLLMClient, JudgeLLMError
from .models import (
    ANALYTICS_AUTH_MODE_CLIENT_CREDENTIALS,
    AnalyticsJourneyResult,
    AnalyticsRunDiagnostics,
    AppConfig,
    AttemptResult,
    JourneyTaxonomyRollup,
    JourneyValidationResult,
    Message,
    MessageRole,
    ProgressEvent,
    ProgressEventType,
    ScenarioResult,
    TestReport,
    normalize_analytics_auth_mode,
)
from .progress import ProgressEmitter
from .transcript_import_store import TranscriptImportStore

_AUTH_BEHAVIORS = {"required", "forbidden", "optional"}
_TRANSFER_BEHAVIORS = {"required", "forbidden", "optional"}
_ANALYTICS_TIMELINE_LIMIT = 400

_DEFAULT_POLICY_MAP: dict[str, dict[str, Any]] = {
    "default": {
        "auth_behavior": "optional",
        "transfer_behavior": "optional",
    },
    "speak_to_agent": {
        "auth_behavior": "optional",
        "transfer_behavior": "required",
    },
    "guidelines": {
        "auth_behavior": "forbidden",
        "transfer_behavior": "forbidden",
    },
}


@dataclass
class AnalyticsJourneyRunRequest:
    """Input payload for one analytics journey run."""

    interval: str
    page_size: int
    max_conversations: int
    bot_flow_id: str = ""
    auth_mode: str = ANALYTICS_AUTH_MODE_CLIENT_CREDENTIALS
    manual_bearer_token: Optional[str] = None
    divisions: list[str] = field(default_factory=list)
    language_filter: Optional[str] = None
    extra_query_params: dict[str, Any] = field(default_factory=dict)


class AnalyticsJourneyStopRequested(Exception):
    """Raised when stop is requested during analytics journey execution."""


class AnalyticsJourneyRunner:
    """Evaluate Botflow reporting turns as journey-regression attempts."""

    def __init__(
        self,
        *,
        config: AppConfig,
        progress_emitter: ProgressEmitter,
        stop_event=None,
        artifact_store: Optional[TranscriptImportStore] = None,
    ):
        self.config = config
        self.progress_emitter = progress_emitter
        self.stop_event = stop_event
        self.artifact_store = artifact_store

    def _stop_requested(self) -> bool:
        return bool(self.stop_event is not None and self.stop_event.is_set())

    async def _run_sync_interruptible(self, func, *args, **kwargs):
        """Run blocking sync work while honoring stop requests promptly."""
        task = asyncio.create_task(asyncio.to_thread(func, *args, **kwargs))
        try:
            while True:
                if self._stop_requested():
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
                    raise AnalyticsJourneyStopRequested("Stopped by user request")
                try:
                    return await asyncio.wait_for(asyncio.shield(task), timeout=0.2)
                except asyncio.TimeoutError:
                    if task.done():
                        return await task
                    continue
        finally:
            if self._stop_requested() and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    async def run(self, request: AnalyticsJourneyRunRequest) -> TestReport:
        start_time = time.time()
        run_started_monotonic = time.monotonic()
        request_auth_mode = normalize_analytics_auth_mode(request.auth_mode)
        completed_attempts = 0
        planned_attempts = max(1, int(request.max_conversations))
        suite_name = (
            "Analytics Journey Regression - "
            f"{request.bot_flow_id.strip() or 'reporting-turns'}"
        )
        diagnostics_summary_skips: dict[str, int] = {}
        analytics_diagnostics: dict[str, Any] = {
            "request": {
                "bot_flow_id": request.bot_flow_id,
                "interval": request.interval,
                "page_size": int(request.page_size),
                "max_conversations": int(request.max_conversations),
                "auth_mode": request_auth_mode,
                "divisions_count": len(request.divisions),
                "language_filter": request.language_filter,
                "extra_query_param_keys": sorted(
                    str(key).strip()
                    for key in (request.extra_query_params or {}).keys()
                    if str(key).strip()
                ),
            },
            "summary": {
                "pages_fetched": 0,
                "rows_scanned": 0,
                "unique_conversations": 0,
                "evaluated": 0,
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "retry_count": 0,
                "http_429_count": 0,
                "http_5xx_count": 0,
                "fetch_duration_seconds": 0.0,
                "evaluation_duration_seconds": 0.0,
                "total_duration_seconds": 0.0,
            },
            "timeline": [],
            "dropped_timeline_entries": 0,
        }

        def log_analytics_stage(
            stage: str,
            message: str,
            *,
            page_number: Optional[int] = None,
            conversation_id: Optional[str] = None,
            duration_ms: Optional[float] = None,
            details: Optional[dict[str, Any]] = None,
        ) -> None:
            entry: dict[str, Any] = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": round(
                    max(0.0, time.monotonic() - run_started_monotonic),
                    3,
                ),
                "stage": str(stage or "").strip().lower() or "unknown",
                "message": message,
            }
            if page_number is not None:
                entry["page_number"] = int(page_number)
            if conversation_id:
                entry["conversation_id"] = str(conversation_id).strip().lower()
            if duration_ms is not None:
                entry["duration_ms"] = round(max(0.0, float(duration_ms)), 2)
            if details:
                entry["details"] = details

            timeline = analytics_diagnostics["timeline"]
            if len(timeline) < _ANALYTICS_TIMELINE_LIMIT:
                timeline.append(entry)
            else:
                analytics_diagnostics["dropped_timeline_entries"] = (
                    int(analytics_diagnostics.get("dropped_timeline_entries", 0)) + 1
                )

            self.progress_emitter.emit(
                ProgressEvent(
                    event_type=ProgressEventType.ATTEMPT_STATUS,
                    suite_name=suite_name,
                    message=message,
                    planned_attempts=planned_attempts,
                    completed_attempts=completed_attempts,
                )
            )

        self.progress_emitter.emit(
            ProgressEvent(
                event_type=ProgressEventType.SUITE_STARTED,
                suite_name=suite_name,
                message=f"Starting analytics journey regression: {request.bot_flow_id}",
                planned_attempts=planned_attempts,
                completed_attempts=0,
            )
        )
        log_analytics_stage(
            "run_init",
            (
                "Analytics reporting-turns run initialized: "
                f"bot_flow_id={request.bot_flow_id or 'n/a'}, "
                f"auth_mode={request_auth_mode}, page_size={request.page_size}, "
                f"max_conversations={request.max_conversations}"
            ),
            details={
                "divisions_count": len(request.divisions),
                "language_filter": request.language_filter,
            },
        )

        analytics_judge_model = (
            self.config.analytics_journey_judge_model
            or self.config.ollama_model
            or ""
        ).strip()
        judge = JudgeLLMClient(
            base_url=self.config.ollama_base_url,
            model=analytics_judge_model,
            timeout=self.config.response_timeout,
        )
        if self.config.judge_warmup_enabled:
            if self._stop_requested():
                log_analytics_stage(
                    "run_stop_requested",
                    "Stop requested before Judge warm-up.",
                )
            else:
                log_analytics_stage("judge_warmup_start", "Warming up Judge LLM model")
                warmup_started_at = time.monotonic()
                try:
                    await self._run_sync_interruptible(
                        judge.warm_up,
                        language_code=self.config.evaluation_results_language,
                    )
                    log_analytics_stage(
                        "judge_warmup_complete",
                        "Judge LLM warm-up complete",
                        duration_ms=(time.monotonic() - warmup_started_at) * 1000.0,
                    )
                except AnalyticsJourneyStopRequested:
                    log_analytics_stage(
                        "run_stop_requested",
                        "Stop requested during Judge warm-up.",
                        duration_ms=(time.monotonic() - warmup_started_at) * 1000.0,
                    )
                except Exception as e:  # pragma: no cover - defensive fallback
                    log_analytics_stage(
                        "judge_warmup_failed",
                        f"Judge warm-up failed; continuing. Details: {e}",
                        duration_ms=(time.monotonic() - warmup_started_at) * 1000.0,
                    )

        analytics_client = GenesysAnalyticsJourneyClient(
            region=self.config.gc_region or "",
            client_id=self.config.gc_client_id or "",
            client_secret=self.config.gc_client_secret or "",
            auth_mode=request_auth_mode,
            manual_bearer_token=request.manual_bearer_token,
            timeout=self.config.response_timeout,
            page_size_cap=self.config.analytics_journey_details_page_size_cap,
        )

        def analytics_observer(event: dict[str, Any]) -> None:
            event_name = str(event.get("event") or "").strip().lower()
            page_number = event.get("page_number")
            duration_ms = event.get("duration_ms")
            summary = analytics_diagnostics["summary"]
            if event_name == "page_fetch_started":
                log_analytics_stage(
                    "analytics_page_start",
                    f"Fetching reporting-turns page {page_number}",
                    page_number=int(page_number) if isinstance(page_number, int) else None,
                )
                return
            if event_name == "page_fetch_completed":
                rows_count = int(event.get("rows_count") or 0)
                new_ids = int(event.get("new_unique_conversations") or 0)
                summary["rows_scanned"] = int(summary["rows_scanned"]) + rows_count
                log_analytics_stage(
                    "analytics_page_complete",
                    (
                        f"Fetched reporting-turns page {page_number}: "
                        f"{rows_count} rows, {new_ids} new conversations"
                    ),
                    page_number=int(page_number) if isinstance(page_number, int) else None,
                    duration_ms=duration_ms if isinstance(duration_ms, (int, float)) else None,
                    details={
                        "rows_count": rows_count,
                        "new_unique_conversations": new_ids,
                        "total_unique_conversations": int(
                            event.get("total_unique_conversations") or 0
                        ),
                    },
                )
                return
            if event_name == "request_retry":
                summary["retry_count"] = int(summary["retry_count"]) + 1
                status_code = event.get("status_code")
                if status_code == 429:
                    summary["http_429_count"] = int(summary["http_429_count"]) + 1
                if isinstance(status_code, int) and 500 <= status_code <= 599:
                    summary["http_5xx_count"] = int(summary["http_5xx_count"]) + 1
                log_analytics_stage(
                    "analytics_request_retry",
                    (
                        "Analytics API retry triggered"
                        f" (attempt {event.get('attempt')}/{event.get('max_attempts')})"
                    ),
                    page_number=int(page_number) if isinstance(page_number, int) else None,
                    duration_ms=duration_ms if isinstance(duration_ms, (int, float)) else None,
                    details={
                        "status_code": status_code,
                        "error_type": event.get("error_type"),
                        "backoff_seconds": event.get("backoff_seconds"),
                    },
                )

        try:
            fetch_started_at = time.monotonic()
            log_analytics_stage(
                "analytics_fetch_start",
                "Starting bot reporting-turns ingestion",
            )
            fetched = await self._run_sync_interruptible(
                analytics_client.fetch_conversation_units,
                bot_flow_id=request.bot_flow_id,
                interval=request.interval,
                page_size=request.page_size,
                max_conversations=request.max_conversations,
                divisions=request.divisions,
                language_filter=request.language_filter,
                extra_params=request.extra_query_params,
                observer=analytics_observer,
                stop_requested=self._stop_requested,
            )
            analytics_diagnostics["summary"]["fetch_duration_seconds"] = round(
                max(0.0, time.monotonic() - fetch_started_at),
                3,
            )
        except AnalyticsJourneyStopRequested:
            log_analytics_stage(
                "run_stop_requested",
                "Stop requested during reporting-turns ingestion.",
            )
            fetched = {
                "conversations": [],
                "page_payloads": [],
                "page_count": 0,
                "ignored_query_params": [],
                "applied_query_params": [],
            }
        except GenesysAnalyticsJourneyError as e:
            raise RuntimeError(f"Analytics API ingestion failed: {e}") from e

        ignored_query_params = list(fetched.get("ignored_query_params") or [])
        if ignored_query_params:
            log_analytics_stage(
                "analytics_query_params_ignored",
                "Ignored unsupported advanced query keys for reporting-turns call.",
                details={"ignored_keys": ignored_query_params},
            )

        planned_attempts = max(1, len(fetched.get("conversations", [])))
        analytics_diagnostics["summary"]["pages_fetched"] = int(
            fetched.get("page_count", 0)
        )
        analytics_diagnostics["summary"]["unique_conversations"] = len(
            fetched.get("conversations", [])
        )
        log_analytics_stage(
            "analytics_fetch_complete",
            (
                "Fetched bot reporting-turns pages: "
                f"{int(fetched.get('page_count', 0))} page(s), "
                f"{len(fetched.get('conversations', []))} conversation(s)"
            ),
            details={
                "page_count": int(fetched.get("page_count", 0)),
                "conversation_count": len(fetched.get("conversations", [])),
                "rows_scanned": int(analytics_diagnostics["summary"]["rows_scanned"]),
            },
        )

        try:
            category_overrides = load_category_overrides(
                categories_json=self.config.journey_primary_categories_json,
                categories_file=self.config.journey_primary_categories_file,
            )
        except ValueError:
            category_overrides = []
        primary_categories = resolve_primary_categories(
            suite_categories=None,
            config_overrides=category_overrides,
        )
        policy_map = load_analytics_policy_map(
            policy_json=self.config.analytics_journey_policy_map_json,
            policy_file=self.config.analytics_journey_policy_map_file,
        )

        seeded_suite_yaml = self._build_seeded_suite_yaml(
            suite_name=suite_name,
            language_code=self.config.language,
            units=list(fetched.get("conversations", [])),
            diagnostics_skip_counter=diagnostics_summary_skips,
        )

        scenario_results: list[ScenarioResult] = []
        evaluation_started_at = time.monotonic()
        raw_artifacts: dict[str, dict[str, Any]] = {}
        for page_index, payload in enumerate(fetched.get("page_payloads", []), start=1):
            raw_artifacts[f"analytics-page-{page_index:03d}"] = payload

        for index, unit in enumerate(fetched.get("conversations", []), start=1):
            if self.stop_event is not None and self.stop_event.is_set():
                log_analytics_stage(
                    "run_stop_requested",
                    "Stop requested. Ending analytics evaluation loop.",
                )
                break

            conversation_id = str(unit.get("conversation_id") or "").strip().lower()
            scenario_name = f"{conversation_id} - Analytics Journey"
            raw_artifacts[f"conversation-{conversation_id}"] = {
                "conversation_id": conversation_id,
                "rows": list(unit.get("rows") or []),
            }
            log_analytics_stage(
                "conversation_eval_start",
                (
                    f"Starting analytics conversation {index}/"
                    f"{len(fetched.get('conversations', []))}: {conversation_id}"
                ),
                conversation_id=conversation_id,
                details={"conversation_index": index},
            )
            self.progress_emitter.emit(
                ProgressEvent(
                    event_type=ProgressEventType.SCENARIO_STARTED,
                    suite_name=suite_name,
                    scenario_name=scenario_name,
                    message=(
                        f"Starting analytics conversation {index}/"
                        f"{len(fetched.get('conversations', []))}: {conversation_id}"
                    ),
                )
            )

            self.progress_emitter.emit(
                ProgressEvent(
                    event_type=ProgressEventType.ATTEMPT_STARTED,
                    suite_name=suite_name,
                    scenario_name=scenario_name,
                    attempt_number=1,
                    message="Attempt 1 started",
                    planned_attempts=len(fetched.get("conversations", [])),
                    completed_attempts=completed_attempts,
                )
            )

            step_log: list[dict[str, Any]] = []

            def emit_status(
                status: str,
                *,
                stage: Optional[str] = None,
                duration_ms: Optional[float] = None,
                details: Optional[dict[str, Any]] = None,
            ) -> None:
                self.progress_emitter.emit(
                    ProgressEvent(
                        event_type=ProgressEventType.ATTEMPT_STATUS,
                        suite_name=suite_name,
                        scenario_name=scenario_name,
                        attempt_number=1,
                        message=status,
                        planned_attempts=len(fetched.get("conversations", [])),
                        completed_attempts=completed_attempts,
                    )
                )
                step_log.append(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "step": status,
                        "stage": stage or "conversation_step",
                        "conversation_id": conversation_id,
                        "duration_ms": (
                            round(max(0.0, float(duration_ms)), 2)
                            if duration_ms is not None
                            else None
                        ),
                        "details": details or None,
                    }
                )

            conversation_started_at = time.monotonic()
            try:
                attempt, _resolved_category = await self._run_sync_interruptible(
                    self._evaluate_conversation_unit,
                    unit,
                    judge,
                    primary_categories,
                    policy_map,
                    emit_status,
                    step_log,
                    self._stop_requested,
                )
            except AnalyticsJourneyStopRequested:
                emit_status(
                    "Attempt interrupted by stop request",
                    stage="conversation_result_stopped",
                )
                attempt = AttemptResult(
                    attempt_number=1,
                    success=False,
                    conversation=[],
                    explanation="Attempt stopped by user request.",
                    error="Stopped by user request",
                    timed_out=False,
                    skipped=True,
                    started_at=datetime.now(timezone.utc),
                    completed_at=datetime.now(timezone.utc),
                    step_log=step_log,
                )

            completed_attempts += 1
            analytics_diagnostics["summary"]["evaluated"] = int(
                analytics_diagnostics["summary"]["evaluated"]
            ) + 1
            if attempt.success:
                analytics_diagnostics["summary"]["passed"] = int(
                    analytics_diagnostics["summary"]["passed"]
                ) + 1
            elif attempt.skipped:
                analytics_diagnostics["summary"]["skipped"] = int(
                    analytics_diagnostics["summary"]["skipped"]
                ) + 1
                analytics_result = attempt.analytics_journey_result
                skip_key = str(
                    (
                        analytics_result.skipped_reason
                        if analytics_result is not None
                        else attempt.error or "unknown"
                    )
                    or "unknown"
                ).strip()
                diagnostics_summary_skips[skip_key] = diagnostics_summary_skips.get(skip_key, 0) + 1
            else:
                analytics_diagnostics["summary"]["failed"] = int(
                    analytics_diagnostics["summary"]["failed"]
                ) + 1

            self.progress_emitter.emit(
                ProgressEvent(
                    event_type=ProgressEventType.ATTEMPT_COMPLETED,
                    suite_name=suite_name,
                    scenario_name=scenario_name,
                    expected_intent=None,
                    attempt_number=1,
                    success=attempt.success,
                    message=(
                        f"Attempt 1: {'success' if attempt.success else 'failure'} "
                        f"({completed_attempts}/{len(fetched.get('conversations', []))})"
                    ),
                    attempt_result=attempt,
                    planned_attempts=len(fetched.get("conversations", [])),
                    completed_attempts=completed_attempts,
                )
            )

            successes = 1 if attempt.success else 0
            timeouts = 1 if attempt.timed_out else 0
            skipped = 1 if attempt.skipped else 0
            failures = 1 - successes - timeouts - skipped
            success_rate = float(successes)
            analytics_result = attempt.analytics_journey_result
            scenario_result = ScenarioResult(
                scenario_name=scenario_name,
                expected_intent=None,
                attempts=1,
                successes=successes,
                failures=failures,
                timeouts=timeouts,
                skipped=skipped,
                success_rate=success_rate,
                is_regression=success_rate < self.config.success_threshold,
                analytics_evaluated_attempts=1 if analytics_result is not None else 0,
                analytics_gate_passes=(
                    1
                    if analytics_result is not None and attempt.success
                    else 0
                ),
                analytics_skipped_unknown=(
                    1
                    if analytics_result is not None and attempt.skipped
                    else 0
                ),
                journey_validated_attempts=(
                    1 if attempt.journey_validation_result is not None else 0
                ),
                journey_passes=(
                    1
                    if attempt.journey_validation_result is not None and attempt.success
                    else 0
                ),
                journey_contained_passes=(
                    1
                    if attempt.journey_validation_result is not None
                    and attempt.journey_validation_result.contained is True
                    else 0
                ),
                journey_fulfillment_passes=(
                    1
                    if attempt.journey_validation_result is not None
                    and attempt.journey_validation_result.fulfilled
                    else 0
                ),
                journey_path_passes=(
                    1
                    if attempt.journey_validation_result is not None
                    and attempt.journey_validation_result.path_correct
                    else 0
                ),
                journey_category_match_passes=(
                    1
                    if attempt.journey_validation_result is not None
                    and attempt.journey_validation_result.category_match is True
                    else 0
                ),
                attempt_results=[attempt],
            )
            scenario_results.append(scenario_result)

            self.progress_emitter.emit(
                ProgressEvent(
                    event_type=ProgressEventType.SCENARIO_COMPLETED,
                    suite_name=suite_name,
                    scenario_name=scenario_name,
                    expected_intent=None,
                    success_rate=scenario_result.success_rate,
                    message=(
                        f"Scenario completed: {scenario_name} — "
                        f"{scenario_result.success_rate:.0%} success rate"
                    ),
                )
            )
            log_analytics_stage(
                "conversation_eval_complete",
                (
                    f"Completed analytics conversation {index}/"
                    f"{len(fetched.get('conversations', []))}: "
                    f"{'success' if attempt.success else ('skipped' if attempt.skipped else 'failure')}"
                ),
                conversation_id=conversation_id,
                duration_ms=(time.monotonic() - conversation_started_at) * 1000.0,
                details={
                    "attempt_success": attempt.success,
                    "attempt_skipped": attempt.skipped,
                    "attempt_timed_out": attempt.timed_out,
                },
            )

        duration = time.time() - start_time
        analytics_diagnostics["summary"]["evaluation_duration_seconds"] = round(
            max(0.0, time.monotonic() - evaluation_started_at),
            3,
        )
        analytics_diagnostics["summary"]["total_duration_seconds"] = round(
            max(0.0, duration),
            3,
        )
        overall_attempts = sum(item.attempts for item in scenario_results)
        overall_successes = sum(item.successes for item in scenario_results)
        overall_failures = sum(item.failures for item in scenario_results)
        overall_timeouts = sum(item.timeouts for item in scenario_results)
        overall_skipped = sum(item.skipped for item in scenario_results)
        overall_success_rate = (
            (overall_successes / overall_attempts)
            if overall_attempts > 0
            else 0.0
        )
        overall_analytics_evaluated_attempts = sum(
            item.analytics_evaluated_attempts for item in scenario_results
        )
        overall_analytics_gate_passes = sum(
            item.analytics_gate_passes for item in scenario_results
        )
        overall_analytics_skipped_unknown = sum(
            item.analytics_skipped_unknown for item in scenario_results
        )
        run_diagnostics = AnalyticsRunDiagnostics.model_validate(analytics_diagnostics)
        report = TestReport(
            suite_name=suite_name,
            timestamp=datetime.now(timezone.utc),
            duration_seconds=duration,
            scenario_results=scenario_results,
            overall_attempts=overall_attempts,
            overall_successes=overall_successes,
            overall_failures=overall_failures,
            overall_timeouts=overall_timeouts,
            overall_skipped=overall_skipped,
            overall_success_rate=overall_success_rate,
            overall_journey_validated_attempts=sum(
                item.journey_validated_attempts for item in scenario_results
            ),
            overall_journey_passes=sum(item.journey_passes for item in scenario_results),
            overall_journey_contained_passes=sum(
                item.journey_contained_passes for item in scenario_results
            ),
            overall_journey_fulfillment_passes=sum(
                item.journey_fulfillment_passes for item in scenario_results
            ),
            overall_journey_path_passes=sum(
                item.journey_path_passes for item in scenario_results
            ),
            overall_journey_category_match_passes=sum(
                item.journey_category_match_passes for item in scenario_results
            ),
            overall_analytics_evaluated_attempts=overall_analytics_evaluated_attempts,
            overall_analytics_gate_passes=overall_analytics_gate_passes,
            overall_analytics_skipped_unknown=overall_analytics_skipped_unknown,
            analytics_run_diagnostics=run_diagnostics,
            stopped_by_user=self._stop_requested(),
            stop_mode="immediate" if self._stop_requested() else None,
            has_regressions=any(item.is_regression for item in scenario_results),
            regression_threshold=self.config.success_threshold,
        )

        if self.config.journey_dashboard_enabled:
            try:
                taxonomy_overrides = load_taxonomy_overrides(
                    overrides_json=self.config.journey_taxonomy_overrides_json,
                    overrides_file=self.config.journey_taxonomy_overrides_file,
                )
            except ValueError:
                taxonomy_overrides = {}
            taxonomy_rollups = build_journey_taxonomy_rollups(
                report,
                overrides=taxonomy_overrides,
                active_view="overview",
            )
            report.journey_taxonomy_rollups = [
                JourneyTaxonomyRollup.model_validate(row)
                for row in taxonomy_rollups["labels"]
            ]

        if self.artifact_store is not None:
            manifest = {
                "status": "completed",
                "source": "analytics_journey",
                "mode": "analytics_bot_reporting_turns",
                "bot_flow_id": request.bot_flow_id,
                "interval": request.interval,
                "divisions": request.divisions,
                "language_filter": request.language_filter,
                "requested_ids": request.max_conversations,
                "selected_ids": len(fetched.get("conversations", [])),
                "fetched_ids": len(fetched.get("conversations", [])),
                "failed_ids": 0,
                "skipped_ids": overall_skipped,
                "scenarios_generated": len(scenario_results),
                "page_count": fetched.get("page_count", 0),
                "ignored_query_params": ignored_query_params,
                "skip_reason_counts": diagnostics_summary_skips,
                "failures": [],
            }
            stored_manifest = self.artifact_store.save_run(
                manifest=manifest,
                transcripts_by_id=raw_artifacts,
                suite_yaml=seeded_suite_yaml,
            )
            seeded_path = str(stored_manifest.get("seeded_suite_path") or "").strip()
            if seeded_path:
                log_analytics_stage(
                    "seed_suite_saved",
                    "Saved seeded analytics suite artifact.",
                    details={"seeded_suite_path": seeded_path},
                )

        completed_message = f"Suite completed: {suite_name} in {duration:.1f}s"
        if self.stop_event is not None and self.stop_event.is_set():
            completed_message = (
                f"Suite stopped early: {suite_name} after {duration:.1f}s"
            )
        log_analytics_stage(
            "run_complete",
            completed_message,
            details={
                "evaluated": overall_analytics_evaluated_attempts,
                "passed": overall_analytics_gate_passes,
                "failed_or_skipped": (
                    overall_analytics_evaluated_attempts - overall_analytics_gate_passes
                ),
                "skipped": overall_analytics_skipped_unknown,
            },
        )
        self.progress_emitter.emit(
            ProgressEvent(
                event_type=ProgressEventType.SUITE_COMPLETED,
                suite_name=suite_name,
                message=completed_message,
                duration_seconds=duration,
                planned_attempts=planned_attempts,
                completed_attempts=completed_attempts,
            )
        )

        return report

    def _evaluate_conversation_unit(
        self,
        unit: dict[str, Any],
        judge: JudgeLLMClient,
        categories: list[dict[str, Any]],
        policy_map: dict[str, dict[str, Any]],
        status_callback,
        step_log: list[dict[str, Any]],
        stop_requested,
    ) -> tuple[AttemptResult, Optional[str]]:
        if callable(stop_requested) and stop_requested():
            raise AnalyticsJourneyStopRequested("Stopped before conversation evaluation")

        started_at = datetime.now(timezone.utc)
        conversation_id = str(unit.get("conversation_id") or "").strip().lower()
        raw_rows = [
            row for row in (unit.get("rows") or [])
            if isinstance(row, dict)
        ]
        status_callback(
            "Collecting reporting-turn conversation data",
            stage="conversation_collect_data",
            details={"conversation_id": conversation_id, "turn_count": len(raw_rows)},
        )

        if not raw_rows:
            return self._build_skipped_attempt(
                started_at=started_at,
                conversation=[],
                reason="no_reporting_turn_rows",
                explanation="Skipped: no reporting-turn rows were returned for this conversation.",
                conversation_id=conversation_id,
                step_log=step_log,
            ), None

        conversation_messages = self._build_message_history(raw_rows=raw_rows)
        status_callback(
            f"Prepared conversation history with {len(conversation_messages)} message(s)",
            stage="conversation_history_ready",
            details={"message_count": len(conversation_messages)},
        )
        if not conversation_messages:
            return self._build_skipped_attempt(
                started_at=started_at,
                conversation=conversation_messages,
                reason="no_conversation_messages",
                explanation=(
                    "Skipped: no usable userInput/botPrompts were found in reporting-turn rows."
                ),
                conversation_id=conversation_id,
                step_log=step_log,
            ), None

        status_callback(
            "Resolving expected category/path (policy-map first, LLM fallback)",
            stage="conversation_category_resolution_start",
        )
        category_resolution_started_at = time.monotonic()
        resolved_category, classification_source, classification_confidence = (
            self._resolve_expected_category(
                raw_rows=raw_rows,
                conversation_messages=conversation_messages,
                categories=categories,
                policy_map=policy_map,
                judge=judge,
            )
        )
        status_callback(
            (
                "Category resolution complete: "
                f"{resolved_category or 'unknown'} (source={classification_source})"
            ),
            stage="conversation_category_resolution_complete",
            duration_ms=(time.monotonic() - category_resolution_started_at) * 1000.0,
            details={
                "resolved_category": resolved_category,
                "classification_source": classification_source,
                "classification_confidence": classification_confidence,
            },
        )

        if not resolved_category:
            return self._build_skipped_attempt(
                started_at=started_at,
                conversation=conversation_messages,
                reason="classification_unknown",
                explanation=(
                    "Skipped: the expected path could not be resolved from policy hints "
                    "or LLM classification."
                ),
                conversation_id=conversation_id,
                step_log=step_log,
                category=None,
                classification_source=classification_source,
                classification_confidence=classification_confidence,
            ), None

        policy_key, policy = resolve_policy_for_category(resolved_category, policy_map)
        expected_auth = str(policy.get("auth_behavior") or "optional")
        expected_transfer = str(policy.get("transfer_behavior") or "optional")

        contained_from_metadata = infer_containment_from_reporting_turns(
            raw_rows=raw_rows,
            conversation=conversation_messages,
        )

        status_callback(
            "Running journey quality evaluation",
            stage="conversation_journey_evaluation_start",
        )
        path_rubric = self._rubric_for_category(resolved_category, categories)
        journey_eval_started_at = time.monotonic()
        try:
            if callable(stop_requested) and stop_requested():
                raise AnalyticsJourneyStopRequested("Stopped before journey evaluation")
            journey_result = judge.evaluate_journey(
                persona="Customer journey captured from analytics reporting turns.",
                goal=(
                    "Determine if this customer journey followed the correct path and "
                    "was fulfilled appropriately."
                ),
                expected_category=resolved_category,
                path_rubric=path_rubric,
                category_rubric=None,
                conversation_history=conversation_messages,
                language_code=self.config.evaluation_results_language,
                known_contained=contained_from_metadata,
            )
            status_callback(
                "Journey quality evaluation complete",
                stage="conversation_journey_evaluation_complete",
                duration_ms=(time.monotonic() - journey_eval_started_at) * 1000.0,
            )
        except JudgeLLMError as e:
            return self._build_skipped_attempt(
                started_at=started_at,
                conversation=conversation_messages,
                reason="journey_judge_unavailable",
                explanation=(
                    "Skipped: journey evaluation was inconclusive because Judge LLM "
                    f"evaluation failed ({e})."
                ),
                conversation_id=conversation_id,
                step_log=step_log,
                category=resolved_category,
                classification_source=classification_source,
                classification_confidence=classification_confidence,
            ), resolved_category

        status_callback(
            "Evaluating authentication and escalation gates",
            stage="conversation_gate_evaluation_start",
        )
        gate_eval_started_at = time.monotonic()
        observed_auth, auth_notes = infer_auth_evidence(
            conversation_messages,
            raw_rows=raw_rows,
        )
        observed_transfer, transfer_notes = infer_transfer_evidence(
            conversation_messages,
            raw_rows=raw_rows,
            contained_hint=contained_from_metadata,
        )

        auth_gate, auth_unknown, auth_gate_applicable = evaluate_gate(
            expected_behavior=expected_auth,
            observed=observed_auth,
        )
        transfer_gate, transfer_unknown, transfer_gate_applicable = evaluate_gate(
            expected_behavior=expected_transfer,
            observed=observed_transfer,
        )
        status_callback(
            "Authentication and escalation gate evaluation complete",
            stage="conversation_gate_evaluation_complete",
            duration_ms=(time.monotonic() - gate_eval_started_at) * 1000.0,
            details={
                "auth_gate": auth_gate,
                "auth_gate_applicable": auth_gate_applicable,
                "transfer_gate": transfer_gate,
                "transfer_gate_applicable": transfer_gate_applicable,
                "auth_unknown": auth_unknown,
                "transfer_unknown": transfer_unknown,
            },
        )

        category_gate = journey_result.category_match
        journey_quality_gate = bool(journey_result.fulfilled and journey_result.path_correct)

        skip_reasons: list[str] = []
        if category_gate is None:
            skip_reasons.append("category_gate_unknown")
        if auth_unknown:
            skip_reasons.append("auth_evidence_unknown")
        if transfer_unknown:
            skip_reasons.append("transfer_evidence_unknown")

        gate_result = AnalyticsJourneyResult(
            conversation_id=conversation_id,
            category=resolved_category,
            classification_source=classification_source,
            classification_confidence=(
                float(classification_confidence)
                if isinstance(classification_confidence, (int, float))
                else None
            ),
            policy_key=policy_key,
            expected_auth_behavior=expected_auth,
            observed_auth=observed_auth,
            auth_gate=auth_gate,
            auth_gate_applicable=auth_gate_applicable,
            expected_transfer_behavior=expected_transfer,
            observed_transfer=observed_transfer,
            transfer_gate=transfer_gate,
            transfer_gate_applicable=transfer_gate_applicable,
            category_gate=category_gate,
            journey_quality_gate=journey_quality_gate,
            enrichment_used=False,
            skipped_reason=(";".join(skip_reasons) if skip_reasons else None),
            evidence_notes=auth_notes + transfer_notes,
        )

        if skip_reasons:
            explanation = (
                "Skipped: missing/inconclusive evidence for required analytics gates "
                f"({', '.join(skip_reasons)})."
            )
            status_callback(
                f"Conversation skipped due to inconclusive gate evidence ({', '.join(skip_reasons)})",
                stage="conversation_result_skipped",
                details={"skip_reasons": list(skip_reasons)},
            )
            return self._finalize_attempt(
                started_at=started_at,
                success=False,
                skipped=True,
                timed_out=False,
                conversation=conversation_messages,
                explanation=explanation,
                error="Analytics gate evidence inconclusive",
                detected_intent=resolved_category,
                step_log=step_log,
                journey_result=journey_result,
                analytics_result=gate_result,
            ), resolved_category

        failed_gates: list[str] = []
        if auth_gate is False:
            failed_gates.append("authentication")
        if transfer_gate is False:
            failed_gates.append("escalation")
        if category_gate is False:
            failed_gates.append("classification_path")
        if not journey_quality_gate:
            failed_gates.append("journey_quality")

        if failed_gates:
            explanation = (
                "Journey failed analytics gate validation: "
                f"{', '.join(failed_gates)}."
            )
            status_callback(
                f"Conversation failed analytics gates ({', '.join(failed_gates)})",
                stage="conversation_result_failed",
                details={"failed_gates": list(failed_gates)},
            )
            return self._finalize_attempt(
                started_at=started_at,
                success=False,
                skipped=False,
                timed_out=False,
                conversation=conversation_messages,
                explanation=explanation,
                error=f"Gate failure: {', '.join(failed_gates)}",
                detected_intent=resolved_category,
                step_log=step_log,
                journey_result=journey_result,
                analytics_result=gate_result,
            ), resolved_category

        explanation = (
            "Journey passed analytics gates: path correctness, journey quality, "
            "authentication policy, and escalation policy checks."
        )
        status_callback(
            "Conversation passed all analytics gates",
            stage="conversation_result_passed",
        )
        return self._finalize_attempt(
            started_at=started_at,
            success=True,
            skipped=False,
            timed_out=False,
            conversation=conversation_messages,
            explanation=explanation,
            error=None,
            detected_intent=resolved_category,
            step_log=step_log,
            journey_result=journey_result,
            analytics_result=gate_result,
        ), resolved_category

    def _resolve_expected_category(
        self,
        *,
        raw_rows: list[dict[str, Any]],
        conversation_messages: list[Message],
        categories: list[dict[str, Any]],
        policy_map: dict[str, dict[str, Any]],
        judge: JudgeLLMClient,
    ) -> tuple[Optional[str], str, Optional[float]]:
        policy_hint = self._resolve_category_from_policy_hints(raw_rows, policy_map)
        if policy_hint:
            return policy_hint, "policy_map_intent", 1.0

        if not categories:
            categories = [
                {"name": key, "keywords": [key.replace("_", " ")], "rubric": None}
                for key in policy_map.keys()
                if key != "default"
            ]

        classification_text = self._build_classification_text(
            conversation_messages,
            raw_rows,
        )
        if not classification_text.strip() or not categories:
            return None, "unresolved", None

        resolution = resolve_category_with_strategy(
            classification_text,
            categories=categories,
            strategy=CATEGORY_STRATEGY_LLM_FIRST,
            llm_classifier=lambda msg, cat: judge.classify_primary_category(
                first_message=msg,
                categories=cat,
                language_code=self.config.evaluation_results_language,
            ),
        )
        resolved = str(resolution.get("category") or "").strip().lower()
        confidence = resolution.get("confidence")
        if not resolved:
            return None, str(resolution.get("source") or "llm_first"), confidence
        return resolved, str(resolution.get("source") or "llm_first"), confidence

    def _resolve_category_from_policy_hints(
        self,
        raw_rows: list[dict[str, Any]],
        policy_map: dict[str, dict[str, Any]],
    ) -> Optional[str]:
        candidate_tokens: list[str] = []
        for row in raw_rows:
            for value in (
                row.get("intent"),
                row.get("askAction"),
                row.get("path"),
            ):
                token = _normalize_category_token(value)
                if token:
                    candidate_tokens.append(token)
        policy_keys = [
            key for key in policy_map.keys()
            if key and key != "default"
        ]
        if not policy_keys:
            return None

        for token in candidate_tokens:
            if token in policy_map:
                return token

        for token in candidate_tokens:
            for key in policy_keys:
                if token == key:
                    return key
                if token in key or key in token:
                    return key

        return None

    def _build_message_history(
        self,
        *,
        raw_rows: list[dict[str, Any]],
    ) -> list[Message]:
        entries: list[tuple[float, int, Message]] = []
        fallback_order = 0
        for row_index, row in enumerate(raw_rows):
            if not isinstance(row, dict):
                continue
            created_ts = _parse_timestamp(
                row.get("dateCreated") or row.get("timestamp") or row.get("time")
            )
            completed_ts = _parse_timestamp(
                row.get("dateCompleted") or row.get("completedAt")
            )

            for text in _extract_user_inputs(row):
                sort_key = (
                    created_ts.timestamp()
                    if created_ts is not None
                    else float(row_index * 100 + fallback_order)
                )
                entries.append(
                    (
                        sort_key,
                        fallback_order,
                        Message(
                            role=MessageRole.USER,
                            content=text,
                            timestamp=created_ts,
                        ),
                    )
                )
                fallback_order += 1

            prompts = _extract_bot_prompts(row)
            for prompt_index, text in enumerate(prompts):
                prompt_ts = completed_ts or created_ts
                if prompt_ts is not None:
                    sort_key = prompt_ts.timestamp() + (prompt_index * 0.0001)
                else:
                    sort_key = float(row_index * 100 + fallback_order)
                entries.append(
                    (
                        sort_key,
                        fallback_order,
                        Message(
                            role=MessageRole.AGENT,
                            content=text,
                            timestamp=prompt_ts,
                        ),
                    )
                )
                fallback_order += 1

        entries.sort(key=lambda item: (item[0], item[1]))
        return [item[2] for item in entries]

    def _build_classification_text(
        self,
        conversation_messages: list[Message],
        raw_rows: list[dict[str, Any]],
    ) -> str:
        lines: list[str] = []
        for message in conversation_messages[:80]:
            role = "USER" if message.role == MessageRole.USER else "AGENT"
            text = str(message.content or "").strip()
            if not text:
                continue
            lines.append(f"{role}: {text}")

        metadata_notes: list[str] = []
        for row in raw_rows[:120]:
            intent = str(row.get("intent") or "").strip()
            ask_action = str(row.get("askAction") or "").strip()
            if intent:
                metadata_notes.append(f"intent={intent}")
            if ask_action:
                metadata_notes.append(f"askAction={ask_action}")

        if metadata_notes:
            lines.append("METADATA: " + "; ".join(metadata_notes[:40]))

        text = "\n".join(lines)
        if len(text) > 6000:
            return text[:6000]
        return text

    def _build_seeded_suite_yaml(
        self,
        *,
        suite_name: str,
        language_code: str,
        units: list[dict[str, Any]],
        diagnostics_skip_counter: dict[str, int],
    ) -> Optional[str]:
        scenarios: list[dict[str, Any]] = []
        for unit in units:
            conversation_id = str(unit.get("conversation_id") or "").strip().lower()
            raw_rows = [row for row in (unit.get("rows") or []) if isinstance(row, dict)]
            messages = self._build_message_history(raw_rows=raw_rows)
            if not messages:
                key = "seed_no_turn_messages"
                diagnostics_skip_counter[key] = diagnostics_skip_counter.get(key, 0) + 1
                continue
            first_user_message = next(
                (
                    str(message.content or "").strip()
                    for message in messages
                    if message.role == MessageRole.USER
                    and str(message.content or "").strip()
                ),
                None,
            )
            transcript_lines = []
            for message in messages[:24]:
                role = "USER" if message.role == MessageRole.USER else "AGENT"
                transcript_lines.append(f"- {role}: {message.content}")
            if len(messages) > 24:
                transcript_lines.append(f"- ... ({len(messages) - 24} more turn lines)")
            goal = (
                "Evaluate whether this journey followed the correct path, including "
                "conditional authentication and conditional escalation outcomes.\n"
                f"Conversation ID: {conversation_id}\n"
                "Turn context:\n"
                + "\n".join(transcript_lines)
            )
            scenarios.append(
                {
                    "name": f"{conversation_id} - Analytics Journey",
                    "persona": "Customer from analytics reporting turns",
                    "goal": goal,
                    "first_message": first_user_message,
                    "attempts": 1,
                }
            )

        if not scenarios:
            return None

        suite_payload = {
            "name": f"{suite_name} Seed",
            "language": language_code,
            "harness_mode": "journey",
            "scenarios": scenarios,
        }
        return yaml.safe_dump(
            suite_payload,
            sort_keys=False,
            allow_unicode=True,
        )

    def _rubric_for_category(
        self,
        category: str,
        categories: list[dict[str, Any]],
    ) -> Optional[str]:
        normalized = str(category or "").strip().lower()
        if not normalized:
            return None
        for item in categories:
            if str(item.get("name") or "").strip().lower() == normalized:
                rubric = str(item.get("rubric") or "").strip()
                return rubric or None
        return None

    def _build_skipped_attempt(
        self,
        *,
        started_at: datetime,
        conversation: list[Message],
        reason: str,
        explanation: str,
        conversation_id: str,
        step_log: list[dict[str, Any]],
        category: Optional[str] = None,
        classification_source: str = "unknown",
        classification_confidence: Optional[float] = None,
    ) -> AttemptResult:
        analytics = AnalyticsJourneyResult(
            conversation_id=conversation_id,
            category=category,
            classification_source=classification_source,
            classification_confidence=(
                float(classification_confidence)
                if isinstance(classification_confidence, (int, float))
                else None
            ),
            expected_auth_behavior="optional",
            auth_gate_applicable=False,
            expected_transfer_behavior="optional",
            transfer_gate_applicable=False,
            enrichment_used=False,
            skipped_reason=reason,
        )
        return self._finalize_attempt(
            started_at=started_at,
            success=False,
            skipped=True,
            timed_out=False,
            conversation=conversation,
            explanation=explanation,
            error=f"Skipped: {reason}",
            detected_intent=category,
            step_log=step_log,
            analytics_result=analytics,
        )

    def _finalize_attempt(
        self,
        *,
        started_at: datetime,
        success: bool,
        skipped: bool,
        timed_out: bool,
        conversation: list[Message],
        explanation: str,
        error: Optional[str],
        detected_intent: Optional[str],
        step_log: list[dict[str, Any]],
        journey_result: Optional[JourneyValidationResult] = None,
        analytics_result: Optional[AnalyticsJourneyResult] = None,
    ) -> AttemptResult:
        completed_at = datetime.now(timezone.utc)
        duration_seconds = max(0.0, (completed_at - started_at).total_seconds())
        return AttemptResult(
            attempt_number=1,
            success=success,
            skipped=skipped,
            timed_out=timed_out,
            conversation=conversation,
            explanation=explanation,
            error=error,
            detected_intent=detected_intent,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=duration_seconds,
            step_log=list(step_log),
            journey_validation_result=journey_result,
            analytics_journey_result=analytics_result,
        )


def load_analytics_policy_map(
    *,
    policy_json: Optional[str],
    policy_file: Optional[str],
) -> dict[str, dict[str, Any]]:
    """Load analytics auth/transfer policy map from JSON/file overrides."""
    merged: dict[str, dict[str, Any]] = {
        key: dict(value)
        for key, value in _DEFAULT_POLICY_MAP.items()
    }

    payloads: list[dict[str, Any]] = []
    raw_json = str(policy_json or "").strip()
    if raw_json:
        try:
            decoded = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "analytics journey policy JSON is invalid"
            ) from exc
        if not isinstance(decoded, dict):
            raise ValueError("analytics journey policy JSON must be an object")
        payloads.append(decoded)

    file_path = str(policy_file or "").strip()
    if file_path:
        path = Path(file_path)
        if not path.exists():
            raise ValueError(f"analytics journey policy file not found: {file_path}")
        try:
            decoded_file = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                "analytics journey policy file must contain valid JSON"
            ) from exc
        if not isinstance(decoded_file, dict):
            raise ValueError(
                "analytics journey policy file must contain a JSON object"
            )
        payloads.append(decoded_file)

    for payload in payloads:
        for raw_key, raw_policy in payload.items():
            key = str(raw_key or "").strip().lower()
            if not key or not isinstance(raw_policy, dict):
                continue
            auth_behavior = normalize_policy_behavior(
                raw_policy.get("auth_behavior"),
                valid=_AUTH_BEHAVIORS,
                default="optional",
            )
            transfer_behavior = normalize_policy_behavior(
                raw_policy.get("transfer_behavior"),
                valid=_TRANSFER_BEHAVIORS,
                default="optional",
            )
            merged[key] = {
                "auth_behavior": auth_behavior,
                "transfer_behavior": transfer_behavior,
            }

    return merged


def normalize_policy_behavior(
    value: Any,
    *,
    valid: set[str],
    default: str,
) -> str:
    normalized = str(value or "").strip().lower()
    if not normalized:
        return default
    return normalized if normalized in valid else default


def resolve_policy_for_category(
    category: str,
    policy_map: dict[str, dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    normalized = str(category or "").strip().lower()
    if normalized and normalized in policy_map:
        return normalized, policy_map[normalized]
    return "default", policy_map.get("default", dict(_DEFAULT_POLICY_MAP["default"]))


def infer_containment_from_reporting_turns(
    *,
    raw_rows: list[dict[str, Any]],
    conversation: list[Message],
) -> Optional[bool]:
    corpus = _build_signal_corpus(conversation, raw_rows)
    transfer_tokens = [
        "transfer to agent",
        "transfer to live agent",
        "connect you to an agent",
        "speak with a live agent",
        "handoff",
        "escalate",
    ]
    containment_tokens = [
        "resolved",
        "self-service",
        "issue is fixed",
        "anything else i can help",
        "contained",
    ]
    if any(token in corpus for token in transfer_tokens):
        return False
    if any(token in corpus for token in containment_tokens):
        return True

    bool_signals = _find_boolean_signals(
        raw_rows,
        positive_keys={
            "contained",
            "containment",
            "selfservice",
        },
        negative_keys={
            "transferred",
            "escalated",
            "liveagent",
        },
    )
    if bool_signals["negative"]:
        return False
    if bool_signals["positive"]:
        return True
    return None


def infer_auth_evidence(
    conversation: list[Message],
    raw_rows: Optional[list[dict[str, Any]]] = None,
) -> tuple[Optional[bool], list[str]]:
    corpus = _build_signal_corpus(conversation, raw_rows or [])
    notes: list[str] = []

    success_tokens = [
        "authentication successful",
        "successfully authenticated",
        "verified your identity",
        "guest authenticated",
        "auth success",
        "otp verified",
    ]
    not_required_tokens = [
        "authentication not required",
        "no authentication required",
        "auth not required",
    ]
    if any(token in corpus for token in success_tokens):
        notes.append("auth evidence from reporting-turn text metadata")
        return True, notes
    if any(token in corpus for token in not_required_tokens):
        notes.append("auth-not-required evidence from reporting-turn text metadata")
        return False, notes

    bool_signals = _find_boolean_signals(
        raw_rows,
        positive_keys={
            "authenticated",
            "authsuccess",
            "authentication_success",
            "verificationpassed",
        },
        negative_keys={
            "authenticationrequired",
            "authrequired",
        },
    )
    if bool_signals["positive"]:
        notes.append("auth evidence from structured reporting-turn fields")
        return True, notes
    if bool_signals["negative"]:
        notes.append("auth-not-required evidence from structured reporting-turn fields")
        return False, notes

    return None, notes


def infer_transfer_evidence(
    conversation: list[Message],
    raw_rows: Optional[list[dict[str, Any]]] = None,
    *,
    contained_hint: Optional[bool],
) -> tuple[Optional[bool], list[str]]:
    notes: list[str] = []
    if contained_hint is True:
        notes.append("containment metadata indicates no live transfer")
        return False, notes
    if contained_hint is False:
        notes.append("containment metadata indicates live transfer")
        return True, notes

    corpus = _build_signal_corpus(conversation, raw_rows or [])
    transfer_tokens = [
        "transfer to live agent",
        "transfer to agent",
        "connecting you to an agent",
        "live agent",
        "escalate",
        "handoff",
    ]
    if any(token in corpus for token in transfer_tokens):
        notes.append("transfer evidence from reporting-turn text metadata")
        return True, notes

    bool_signals = _find_boolean_signals(
        raw_rows,
        positive_keys={
            "transferred",
            "escalated",
            "liveagent",
            "handoff",
        },
        negative_keys={
            "contained",
            "selfservice",
        },
    )
    if bool_signals["positive"]:
        notes.append("transfer evidence from structured reporting-turn fields")
        return True, notes
    if bool_signals["negative"]:
        notes.append("containment evidence from structured reporting-turn fields")
        return False, notes

    return None, notes


def evaluate_gate(
    *,
    expected_behavior: str,
    observed: Optional[bool],
) -> tuple[Optional[bool], bool, bool]:
    """Return (gate_pass, unknown_required, gate_applicable)."""
    expected = normalize_policy_behavior(
        expected_behavior,
        valid=_AUTH_BEHAVIORS,
        default="optional",
    )
    if expected == "optional":
        return None, False, False
    if observed is None:
        return None, True, True
    if expected == "required":
        return (observed is True), False, True
    # forbidden
    return (observed is False), False, True


def _extract_user_inputs(row: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("userInput", "input", "utterance", "customerText", "text"):
        value = row.get(key)
        values.extend(_extract_text_values(value))
    return _dedupe_preserve_order(values)


def _extract_bot_prompts(row: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("botPrompts", "botPrompt", "prompts", "responses", "botResponse"):
        value = row.get(key)
        values.extend(_extract_text_values(value))
    return _dedupe_preserve_order(values)


def _extract_text_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        values: list[str] = []
        for item in value:
            values.extend(_extract_text_values(item))
        return values
    if isinstance(value, dict):
        values: list[str] = []
        for key in (
            "text",
            "prompt",
            "content",
            "message",
            "value",
            "utterance",
            "body",
        ):
            values.extend(_extract_text_values(value.get(key)))
        return values
    return []


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        normalized = value.strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


def _build_signal_corpus(
    conversation: list[Message],
    raw_rows: list[dict[str, Any]],
) -> str:
    parts: list[str] = []
    for message in conversation:
        content = str(message.content or "").strip().lower()
        if content:
            parts.append(content)
    for row in raw_rows:
        if not isinstance(row, dict):
            continue
        for key in ("intent", "askAction", "path", "sessionId"):
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                parts.append(value.strip().lower())
        for value in _extract_user_inputs(row):
            if value:
                parts.append(value.lower())
        for value in _extract_bot_prompts(row):
            if value:
                parts.append(value.lower())
    return "\n".join(parts)


def _normalize_category_token(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    normalized = text.replace("-", "_").replace(" ", "_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized.strip("_")


def _parse_timestamp(value: Any) -> Optional[datetime]:
    if not value:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    try:
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        parsed = datetime.fromisoformat(raw)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed
    except Exception:
        return None


def _find_boolean_signals(
    payload: Any,
    *,
    positive_keys: set[str],
    negative_keys: set[str],
) -> dict[str, bool]:
    normalized_positive = {str(key).strip().lower() for key in positive_keys if str(key).strip()}
    normalized_negative = {str(key).strip().lower() for key in negative_keys if str(key).strip()}

    found_positive = False
    found_negative = False

    def walk(node: Any):
        nonlocal found_positive, found_negative
        if isinstance(node, dict):
            for raw_key, raw_value in node.items():
                key = str(raw_key or "").strip().lower().replace(" ", "_")
                key_compact = key.replace("_", "")
                value_str = str(raw_value).strip().lower() if raw_value is not None else ""
                truthy = value_str in {"true", "1", "yes", "on"}
                falsy = value_str in {"false", "0", "no", "off"}
                if key in normalized_positive or key_compact in normalized_positive:
                    if truthy:
                        found_positive = True
                    if falsy:
                        found_negative = True
                if key in normalized_negative or key_compact in normalized_negative:
                    if truthy:
                        found_negative = True
                    if falsy:
                        found_positive = True
                walk(raw_value)
            return
        if isinstance(node, list):
            for item in node:
                walk(item)

    walk(payload)
    return {"positive": found_positive, "negative": found_negative}
