"""Analytics Journey Regression runner (Phase 13 evaluate-now)."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .genesys_analytics_journey_client import (
    GenesysAnalyticsJourneyClient,
    GenesysAnalyticsJourneyError,
)
from .genesys_transcript_import_client import (
    GenesysTranscriptImportClient,
    GenesysTranscriptImportError,
)
from .journey_mode import (
    CATEGORY_STRATEGY_LLM_FIRST,
    load_category_overrides,
)
from .journey_regression import (
    infer_containment_from_payload_metadata,
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


class AnalyticsJourneyRunner:
    """Evaluate Botflow analytics conversations as journey regression attempts."""

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

    async def run(self, request: AnalyticsJourneyRunRequest) -> TestReport:
        start_time = time.time()
        run_started_monotonic = time.monotonic()
        request_auth_mode = normalize_analytics_auth_mode(request.auth_mode)
        completed_attempts = 0
        planned_attempts = max(1, int(request.max_conversations))
        suite_name = (
            "Analytics Journey Regression - "
            f"{request.bot_flow_id.strip() or 'details-query'}"
        )
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
                    "Analytics journey run initialized: "
                    f"bot_flow_id={request.bot_flow_id or 'n/a'}, "
                    f"auth_mode={request_auth_mode}, page_size={request.page_size}, "
                    f"max_conversations={request.max_conversations}"
                ),
            details={
                "divisions_count": len(request.divisions),
                "language_filter": request.language_filter,
            },
        )

        judge = JudgeLLMClient(
            base_url=self.config.ollama_base_url,
            model=self.config.ollama_model or "",
            timeout=self.config.response_timeout,
        )
        if self.config.judge_warmup_enabled:
            log_analytics_stage("judge_warmup_start", "Warming up Judge LLM model")
            warmup_started_at = time.monotonic()
            try:
                await asyncio.to_thread(
                    judge.warm_up,
                    language_code=self.config.evaluation_results_language,
                )
                log_analytics_stage(
                    "judge_warmup_complete",
                    "Judge LLM warm-up complete",
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
        enrichment_client = GenesysTranscriptImportClient(
            region=self.config.gc_region or "",
            client_id=self.config.gc_client_id or "",
            client_secret=self.config.gc_client_secret or "",
            auth_mode=request_auth_mode,
            manual_bearer_token=request.manual_bearer_token,
            timeout=self.config.response_timeout,
        )

        def analytics_observer(event: dict[str, Any]) -> None:
            event_name = str(event.get("event") or "").strip().lower()
            page_number = event.get("page_number")
            duration_ms = event.get("duration_ms")
            summary = analytics_diagnostics["summary"]
            if event_name == "page_fetch_started":
                log_analytics_stage(
                    "analytics_page_start",
                    f"Fetching analytics page {page_number}",
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
                        f"Fetched analytics page {page_number}: "
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
            log_analytics_stage("analytics_fetch_start", "Starting analytics API ingestion")
            fetched = analytics_client.fetch_conversation_units(
                bot_flow_id=request.bot_flow_id,
                interval=request.interval,
                page_size=request.page_size,
                max_conversations=request.max_conversations,
                divisions=request.divisions,
                language_filter=request.language_filter,
                extra_params=request.extra_query_params,
                observer=analytics_observer,
            )
            analytics_diagnostics["summary"]["fetch_duration_seconds"] = round(
                max(0.0, time.monotonic() - fetch_started_at),
                3,
            )
        except GenesysAnalyticsJourneyError as e:
            raise RuntimeError(f"Analytics API ingestion failed: {e}") from e

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
                "Fetched analytics reporting-turn pages: "
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
            attempt, expected_intent, raw_payload = await asyncio.to_thread(
                self._evaluate_conversation_unit,
                unit,
                judge,
                enrichment_client,
                primary_categories,
                policy_map,
                emit_status,
                step_log,
            )

            if raw_payload is not None and conversation_id:
                raw_artifacts[f"conversation-{conversation_id}"] = raw_payload

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
            else:
                analytics_diagnostics["summary"]["failed"] = int(
                    analytics_diagnostics["summary"]["failed"]
                ) + 1

            self.progress_emitter.emit(
                ProgressEvent(
                    event_type=ProgressEventType.ATTEMPT_COMPLETED,
                    suite_name=suite_name,
                    scenario_name=scenario_name,
                    expected_intent=expected_intent,
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
                expected_intent=expected_intent,
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
                    expected_intent=expected_intent,
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
                "mode": "analytics_conversation_details_query",
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
                "failures": [],
            }
            self.artifact_store.save_run(
                manifest=manifest,
                transcripts_by_id=raw_artifacts,
                suite_yaml=None,
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
        enrichment_client: GenesysTranscriptImportClient,
        categories: list[dict[str, Any]],
        policy_map: dict[str, dict[str, Any]],
        status_callback,
        step_log: list[dict[str, Any]],
    ) -> tuple[AttemptResult, Optional[str], Optional[dict[str, Any]]]:
        started_at = datetime.now(timezone.utc)
        conversation_id = str(unit.get("conversation_id") or "").strip().lower()
        status_callback(
            "Collecting analytics conversation data",
            stage="conversation_collect_data",
            details={"conversation_id": conversation_id},
        )
        raw_rows = list(unit.get("rows") or [])

        raw_payload: Optional[dict[str, Any]] = None
        normalized_payload: Optional[dict[str, Any]] = None
        enrichment_used = False
        enrichment_started_at = time.monotonic()
        try:
            status_callback(
                "Enriching with conversation payload",
                stage="conversation_enrichment_start",
            )
            raw_payload = enrichment_client.fetch_conversation_payload(conversation_id)
            normalized_payload = enrichment_client.normalize_conversation_payload(
                raw_payload,
                conversation_id=conversation_id,
            )
            enrichment_used = True
            status_callback(
                "Conversation enrichment complete",
                stage="conversation_enrichment_complete",
                duration_ms=(time.monotonic() - enrichment_started_at) * 1000.0,
            )
        except GenesysTranscriptImportError as e:
            status_callback(
                f"Enrichment unavailable; continuing with analytics rows ({e})",
                stage="conversation_enrichment_unavailable",
                duration_ms=(time.monotonic() - enrichment_started_at) * 1000.0,
            )

        conversation_messages = self._build_message_history(
            normalized_payload=normalized_payload,
            raw_rows=raw_rows,
        )
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
                    "Skipped: no usable conversation messages were found from "
                    "analytics payload or enrichment."
                ),
                conversation_id=conversation_id,
                step_log=step_log,
                enrichment_used=enrichment_used,
            ), None, raw_payload

        status_callback(
            "Resolving initial category/path (LLM-first)",
            stage="conversation_category_resolution_start",
        )
        first_customer_message = self._first_customer_message(conversation_messages)
        if not first_customer_message:
            return self._build_skipped_attempt(
                started_at=started_at,
                conversation=conversation_messages,
                reason="missing_first_customer_message",
                explanation=(
                    "Skipped: could not determine the first meaningful customer utterance."
                ),
                conversation_id=conversation_id,
                step_log=step_log,
                enrichment_used=enrichment_used,
            ), None, raw_payload

        category_resolution_started_at = time.monotonic()
        category_resolution = resolve_category_with_strategy(
            first_customer_message,
            categories=categories,
            strategy=CATEGORY_STRATEGY_LLM_FIRST,
            llm_classifier=lambda msg, cat: judge.classify_primary_category(
                first_message=msg,
                categories=cat,
                language_code=self.config.evaluation_results_language,
            ),
        )
        resolved_category = category_resolution.get("category")
        classification_source = str(category_resolution.get("source") or "unknown")
        classification_confidence = category_resolution.get("confidence")
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
                    "Skipped: the initial category/path could not be resolved from analytics evidence."
                ),
                conversation_id=conversation_id,
                step_log=step_log,
                enrichment_used=enrichment_used,
                category=None,
                classification_source=classification_source,
                classification_confidence=classification_confidence,
            ), None, raw_payload

        policy_key, policy = resolve_policy_for_category(resolved_category, policy_map)
        expected_auth = str(policy.get("auth_behavior") or "optional")
        expected_transfer = str(policy.get("transfer_behavior") or "optional")

        contained_from_metadata = (
            infer_containment_from_payload_metadata(raw_payload)
            if isinstance(raw_payload, dict)
            else None
        )

        status_callback(
            "Running journey quality evaluation",
            stage="conversation_journey_evaluation_start",
        )
        path_rubric = self._rubric_for_category(resolved_category, categories)
        journey_eval_started_at = time.monotonic()
        try:
            journey_result = judge.evaluate_journey(
                persona="Customer journey captured from analytics conversation details.",
                goal=(
                    "Determine if this customer journey was fulfilled correctly and "
                    "followed the expected path."
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
                enrichment_used=enrichment_used,
                category=resolved_category,
                classification_source=classification_source,
                classification_confidence=classification_confidence,
            ), resolved_category, raw_payload

        status_callback(
            "Evaluating auth and transfer gates",
            stage="conversation_gate_evaluation_start",
        )
        gate_eval_started_at = time.monotonic()
        observed_auth, auth_notes = infer_auth_evidence(
            conversation_messages,
            raw_payload,
        )
        observed_transfer, transfer_notes = infer_transfer_evidence(
            conversation_messages,
            raw_payload,
            contained_hint=contained_from_metadata,
        )

        auth_gate, auth_unknown = evaluate_gate(
            expected_behavior=expected_auth,
            observed=observed_auth,
        )
        transfer_gate, transfer_unknown = evaluate_gate(
            expected_behavior=expected_transfer,
            observed=observed_transfer,
        )
        status_callback(
            "Auth and transfer gate evaluation complete",
            stage="conversation_gate_evaluation_complete",
            duration_ms=(time.monotonic() - gate_eval_started_at) * 1000.0,
            details={
                "auth_gate": auth_gate,
                "transfer_gate": transfer_gate,
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
            expected_transfer_behavior=expected_transfer,
            observed_transfer=observed_transfer,
            transfer_gate=transfer_gate,
            category_gate=category_gate,
            journey_quality_gate=journey_quality_gate,
            enrichment_used=enrichment_used,
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
            ), resolved_category, raw_payload

        failed_gates: list[str] = []
        if auth_gate is False:
            failed_gates.append("authentication")
        if transfer_gate is False:
            failed_gates.append("transfer")
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
            ), resolved_category, raw_payload

        explanation = (
            "Journey passed analytics gates: category/path, authentication policy, "
            "transfer policy, and journey quality checks."
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
        ), resolved_category, raw_payload

    def _build_message_history(
        self,
        *,
        normalized_payload: Optional[dict[str, Any]],
        raw_rows: list[dict[str, Any]],
    ) -> list[Message]:
        messages: list[Message] = []
        if isinstance(normalized_payload, dict):
            payload_messages = normalized_payload.get("messages")
            if isinstance(payload_messages, list):
                for row in payload_messages:
                    if not isinstance(row, dict):
                        continue
                    text = str(row.get("text") or "").strip()
                    if not text:
                        continue
                    role = _normalize_role(row.get("role"))
                    messages.append(
                        Message(
                            role=MessageRole.AGENT
                            if role == "agent"
                            else MessageRole.USER,
                            content=text,
                            timestamp=_parse_timestamp(row.get("timestamp")),
                        )
                    )

        if messages:
            return messages

        for row in raw_rows:
            text = _extract_row_text(row)
            if not text:
                continue
            role = _normalize_role(
                row.get("role")
                or row.get("speaker")
                or row.get("direction")
                or row.get("participantPurpose")
            )
            messages.append(
                Message(
                    role=MessageRole.AGENT if role == "agent" else MessageRole.USER,
                    content=text,
                    timestamp=_parse_timestamp(
                        row.get("timestamp")
                        or row.get("time")
                        or row.get("turnStart")
                    ),
                )
            )

        return messages

    def _first_customer_message(self, conversation: list[Message]) -> Optional[str]:
        for message in conversation:
            if message.role != MessageRole.USER:
                continue
            content = str(message.content or "").strip()
            if not content:
                continue
            if content.lower().startswith("conversation_id:"):
                continue
            return content
        return None

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
        enrichment_used: bool,
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
            expected_transfer_behavior="optional",
            enrichment_used=enrichment_used,
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


def infer_auth_evidence(
    conversation: list[Message],
    raw_payload: Optional[dict[str, Any]],
) -> tuple[Optional[bool], list[str]]:
    corpus = "\n".join(message.content.lower() for message in conversation)
    notes: list[str] = []

    success_tokens = [
        "authentication successful",
        "successfully authenticated",
        "verified your identity",
        "guest authenticated",
        "auth success",
    ]
    not_required_tokens = [
        "authentication not required",
        "no authentication required",
        "auth not required",
    ]
    if any(token in corpus for token in success_tokens):
        notes.append("auth evidence from transcript text")
        return True, notes
    if any(token in corpus for token in not_required_tokens):
        notes.append("auth-not-required evidence from transcript text")
        return False, notes

    if isinstance(raw_payload, dict):
        bool_signals = _find_boolean_signals(
            raw_payload,
            positive_keys={
                "authenticated",
                "authsuccess",
                "authentication_success",
            },
            negative_keys={
                "authenticationrequired",
                "authrequired",
            },
        )
        if bool_signals["positive"]:
            notes.append("auth evidence from structured payload fields")
            return True, notes
        if bool_signals["negative"]:
            notes.append("auth-not-required evidence from structured payload fields")
            return False, notes

    return None, notes


def infer_transfer_evidence(
    conversation: list[Message],
    raw_payload: Optional[dict[str, Any]],
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

    corpus = "\n".join(message.content.lower() for message in conversation)
    transfer_tokens = [
        "transfer to live agent",
        "transfer to agent",
        "connecting you to an agent",
        "live agent",
        "escalate",
    ]
    if any(token in corpus for token in transfer_tokens):
        notes.append("transfer evidence from transcript text")
        return True, notes

    if isinstance(raw_payload, dict):
        participants = raw_payload.get("participants")
        if isinstance(participants, list):
            for participant in participants:
                if not isinstance(participant, dict):
                    continue
                purpose = str(participant.get("purpose") or "").strip().lower()
                if purpose == "agent" and str(participant.get("userId") or "").strip():
                    notes.append("transfer evidence from participant metadata")
                    return True, notes

    return None, notes


def evaluate_gate(
    *,
    expected_behavior: str,
    observed: Optional[bool],
) -> tuple[Optional[bool], bool]:
    """Return (gate_pass, unknown_required)."""
    expected = normalize_policy_behavior(
        expected_behavior,
        valid=_AUTH_BEHAVIORS,
        default="optional",
    )
    if expected == "optional":
        return True, False
    if observed is None:
        return None, True
    if expected == "required":
        return (observed is True), False
    # forbidden
    return (observed is False), False


def _extract_row_text(row: dict[str, Any]) -> str:
    for key in (
        "text",
        "message",
        "utterance",
        "customerText",
        "botResponse",
        "content",
        "input",
    ):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    nested = row.get("message")
    if isinstance(nested, dict):
        for key in ("text", "content", "utterance", "body"):
            value = nested.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _normalize_role(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {
        "agent",
        "assistant",
        "bot",
        "ivr",
        "flow",
        "architect",
        "outbound",
    }:
        return "agent"
    return "user"


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
    signals = {"positive": False, "negative": False}

    def walk(node: Any):
        if isinstance(node, dict):
            for key, value in node.items():
                key_norm = str(key or "").strip().lower().replace("-", "").replace("_", "")
                if isinstance(value, bool):
                    if key_norm in positive_keys and value:
                        signals["positive"] = True
                    if key_norm in negative_keys and value is False:
                        signals["negative"] = True
                walk(value)
            return
        if isinstance(node, list):
            for item in node:
                walk(item)

    walk(payload)
    return signals
