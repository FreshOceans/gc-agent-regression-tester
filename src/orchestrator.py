"""Test Orchestrator for coordinating test suite execution."""

import asyncio
import time
from datetime import datetime, timezone
from threading import Event
from typing import Optional

from .conversation_runner import ConversationRunner
from .journey_mode import (
    load_category_overrides,
    normalize_category_strategy,
    resolve_effective_harness_mode,
)
from .journey_regression import resolve_primary_categories
from .journey_taxonomy import build_journey_taxonomy_rollups, load_taxonomy_overrides
from .judge_llm import JudgeLLMClient
from .models import (
    AppConfig,
    AttemptResult,
    JourneyTaxonomyRollup,
    ProgressEvent,
    ProgressEventType,
    ScenarioResult,
    TestReport,
    TestSuite,
)
from .progress import ProgressEmitter


class TestOrchestrator:
    """Coordinates execution of all scenarios in a test suite.

    Iterates through scenarios, runs configured attempts sequentially via
    ConversationRunner, collects results, emits progress events, and builds
    the final TestReport.
    """

    def __init__(
        self,
        config: AppConfig,
        progress_emitter: ProgressEmitter,
        stop_event: Optional[Event] = None,
    ):
        """Initialize with app config and progress emitter.

        Args:
            config: Application configuration with connection details and defaults.
            progress_emitter: Emitter for publishing progress events.
            stop_event: Optional event used to request a graceful stop.
        """
        self.config = config
        self.progress_emitter = progress_emitter
        self.stop_event = stop_event

    def _build_origin_from_region(self, region: str) -> str:
        """Derive an allowed origin URL from the configured Genesys Cloud region."""
        normalized = (region or "").strip().lower()
        if normalized.startswith("https://"):
            normalized = normalized[len("https://") :]
        elif normalized.startswith("http://"):
            normalized = normalized[len("http://") :]
        normalized = normalized.split("/", 1)[0]
        if normalized.startswith("apps."):
            return f"https://{normalized}"
        if normalized.startswith("webmessaging."):
            normalized = normalized[len("webmessaging.") :]
        if not normalized:
            normalized = "mypurecloud.com"
        return f"https://apps.{normalized}"

    async def run_suite(self, suite: TestSuite) -> TestReport:
        """Execute all scenarios in the suite, return the complete TestReport.

        For each scenario:
        1. Emit suite_started event
        2. Emit scenario_started, run configured attempts sequentially,
           emit attempt_completed for each, compute success rate, emit scenario_completed
        3. Build TestReport with aggregated results
        4. Emit suite_completed with duration
        5. Apply default attempt count from config when scenario.attempts is None

        Args:
            suite: The test suite to execute.

        Returns:
            A TestReport with all scenario results and overall statistics.
        """
        start_time = time.time()
        planned_attempts = sum(
            scenario.attempts if scenario.attempts is not None else self.config.default_attempts
            for scenario in suite.scenarios
        )
        completed_attempts = 0

        # Emit suite_started
        self.progress_emitter.emit(ProgressEvent(
            event_type=ProgressEventType.SUITE_STARTED,
            suite_name=suite.name,
            message=f"Starting test suite: {suite.name}",
            planned_attempts=planned_attempts,
            completed_attempts=completed_attempts,
        ))

        # Create internal dependencies
        judge = JudgeLLMClient(
            base_url=self.config.ollama_base_url,
            model=self.config.ollama_model or "",
            timeout=self.config.response_timeout,
        )
        if self.config.judge_warmup_enabled:
            self.progress_emitter.emit(ProgressEvent(
                event_type=ProgressEventType.ATTEMPT_STATUS,
                suite_name=suite.name,
                message="Warming up Judge LLM model",
                planned_attempts=planned_attempts,
                completed_attempts=completed_attempts,
            ))
            try:
                await asyncio.to_thread(judge.warm_up, language_code=self.config.language)
                self.progress_emitter.emit(ProgressEvent(
                    event_type=ProgressEventType.ATTEMPT_STATUS,
                    suite_name=suite.name,
                    message="Judge LLM warm-up complete",
                    planned_attempts=planned_attempts,
                    completed_attempts=completed_attempts,
                ))
            except Exception as e:
                self.progress_emitter.emit(ProgressEvent(
                    event_type=ProgressEventType.ATTEMPT_STATUS,
                    suite_name=suite.name,
                    message=(
                        "Judge LLM warm-up failed; continuing run. "
                        f"Details: {e}"
                    ),
                    planned_attempts=planned_attempts,
                    completed_attempts=completed_attempts,
                ))

        harness_mode = resolve_effective_harness_mode(
            runtime_override=None,
            suite_mode=suite.harness_mode,
            config_mode=self.config.harness_mode,
        )
        category_strategy = normalize_category_strategy(
            self.config.journey_category_strategy
        )
        category_overrides: list[dict] = []
        try:
            category_overrides = load_category_overrides(
                categories_json=self.config.journey_primary_categories_json,
                categories_file=self.config.journey_primary_categories_file,
            )
        except ValueError as e:
            self.progress_emitter.emit(ProgressEvent(
                event_type=ProgressEventType.ATTEMPT_STATUS,
                suite_name=suite.name,
                message=(
                    "Journey category override config is invalid. "
                    f"Falling back to built-in defaults. Details: {e}"
                ),
                planned_attempts=planned_attempts,
                completed_attempts=completed_attempts,
            ))
            category_overrides = []
        primary_categories = resolve_primary_categories(
            suite_categories=suite.primary_categories,
            config_overrides=category_overrides,
        )

        web_msg_config = {
            "region": self.config.gc_region or "",
            "deployment_id": self.config.gc_deployment_id or "",
            "timeout": self.config.response_timeout,
            "step_skip_timeout_seconds": self.config.step_skip_timeout_seconds,
            "stop_event": self.stop_event,
            "origin": self._build_origin_from_region(self.config.gc_region or ""),
            "expected_greeting": self.config.expected_greeting,
            "gc_client_id": self.config.gc_client_id or "",
            "gc_client_secret": self.config.gc_client_secret or "",
            "intent_attribute_name": self.config.intent_attribute_name,
            "debug_capture_frames": self.config.debug_capture_frames,
            "debug_capture_frame_limit": self.config.debug_capture_frame_limit,
            "tool_attribute_keys": self.config.tool_attribute_keys,
            "tool_marker_prefixes": self.config.tool_marker_prefixes,
            "language": self.config.language,
            "evaluation_results_language": self.config.evaluation_results_language,
            "harness_mode": harness_mode,
            "journey_category_strategy": category_strategy,
            "primary_categories": primary_categories,
            "judging_mechanics": {
                "enabled": bool(self.config.judging_mechanics_enabled),
                "objective_profile": self.config.judging_objective_profile,
                "strictness": self.config.judging_strictness,
                "tolerance": self.config.judging_tolerance,
                "containment_weight": self.config.judging_containment_weight,
                "fulfillment_weight": self.config.judging_fulfillment_weight,
                "path_weight": self.config.judging_path_weight,
                "explanation_mode": self.config.judging_explanation_mode,
            },
        }
        runner = ConversationRunner(
            judge=judge,
            web_msg_config=web_msg_config,
            max_turns=self.config.max_turns,
        )

        scenario_results: list[ScenarioResult] = []
        last_attempt_start_monotonic: Optional[float] = None

        for scenario in suite.scenarios:
            if self.stop_event is not None and self.stop_event.is_set():
                break

            # Apply default attempt count if not specified
            attempt_count = scenario.attempts if scenario.attempts is not None else self.config.default_attempts
            scenario_expected_intent = (
                scenario.expected_intent.strip().lower()
                if scenario.expected_intent
                else None
            )

            # Emit scenario_started
            self.progress_emitter.emit(ProgressEvent(
                event_type=ProgressEventType.SCENARIO_STARTED,
                suite_name=suite.name,
                scenario_name=scenario.name,
                expected_intent=scenario_expected_intent,
                message=f"Starting scenario: {scenario.name} ({attempt_count} attempts)",
            ))

            attempt_results: list[AttemptResult] = []
            successes = 0
            timeouts = 0
            skipped = 0

            for attempt_num in range(1, attempt_count + 1):
                if self.stop_event is not None and self.stop_event.is_set():
                    break

                min_interval = max(0, self.config.min_attempt_interval_seconds)
                if (
                    min_interval > 0
                    and last_attempt_start_monotonic is not None
                ):
                    elapsed = time.monotonic() - last_attempt_start_monotonic
                    if elapsed < min_interval:
                        await asyncio.sleep(min_interval - elapsed)

                last_attempt_start_monotonic = time.monotonic()
                self.progress_emitter.emit(ProgressEvent(
                    event_type=ProgressEventType.ATTEMPT_STARTED,
                    suite_name=suite.name,
                    scenario_name=scenario.name,
                    expected_intent=scenario_expected_intent,
                    attempt_number=attempt_num,
                    message=f"Attempt {attempt_num} started",
                    planned_attempts=planned_attempts,
                    completed_attempts=completed_attempts,
                ))

                def emit_attempt_status(status_message: str) -> None:
                    self.progress_emitter.emit(ProgressEvent(
                        event_type=ProgressEventType.ATTEMPT_STATUS,
                        suite_name=suite.name,
                        scenario_name=scenario.name,
                        expected_intent=scenario_expected_intent,
                        attempt_number=attempt_num,
                        message=status_message,
                        planned_attempts=planned_attempts,
                        completed_attempts=completed_attempts,
                    ))

                result = await runner.run_attempt(
                    scenario,
                    attempt_num,
                    status_callback=emit_attempt_status,
                )
                attempt_results.append(result)
                completed_attempts += 1

                if result.success:
                    successes += 1
                if result.timed_out:
                    timeouts += 1
                if result.skipped:
                    skipped += 1

                # Emit attempt_completed
                self.progress_emitter.emit(ProgressEvent(
                    event_type=ProgressEventType.ATTEMPT_COMPLETED,
                    suite_name=suite.name,
                    scenario_name=scenario.name,
                    expected_intent=scenario_expected_intent,
                    attempt_number=result.attempt_number,
                    success=result.success,
                    message=(
                        f"Attempt {result.attempt_number}: "
                        f"{'success' if result.success else 'failure'} "
                        f"({completed_attempts}/{planned_attempts})"
                    ),
                    attempt_result=result,
                    planned_attempts=planned_attempts,
                    completed_attempts=completed_attempts,
                ))

            attempts_run = len(attempt_results)
            if attempts_run == 0:
                continue

            failures = attempts_run - successes - timeouts - skipped
            success_rate = successes / attempts_run if attempts_run > 0 else 0.0
            tool_validated_attempts = sum(
                1
                for attempt in attempt_results
                if attempt.tool_validation_result is not None
            )
            tool_loose_passes = sum(
                1
                for attempt in attempt_results
                if attempt.tool_validation_result is not None
                and attempt.tool_validation_result.loose_pass
            )
            tool_strict_passes = sum(
                1
                for attempt in attempt_results
                if attempt.tool_validation_result is not None
                and attempt.tool_validation_result.strict_pass is True
            )
            tool_missing_signal_count = sum(
                1
                for attempt in attempt_results
                if attempt.tool_validation_result is not None
                and attempt.tool_validation_result.missing_signal
            )
            tool_order_mismatch_count = sum(
                1
                for attempt in attempt_results
                if attempt.tool_validation_result is not None
                and attempt.tool_validation_result.strict_pass is False
                and bool(attempt.tool_validation_result.order_violations)
            )
            tool_loose_pass_rate = (
                tool_loose_passes / tool_validated_attempts
                if tool_validated_attempts > 0
                else 0.0
            )
            tool_strict_pass_rate = (
                tool_strict_passes / tool_validated_attempts
                if tool_validated_attempts > 0
                else 0.0
            )
            journey_validated_attempts = sum(
                1
                for attempt in attempt_results
                if attempt.journey_validation_result is not None
            )
            journey_passes = sum(
                1
                for attempt in attempt_results
                if attempt.journey_validation_result is not None
                and attempt.success
            )
            journey_contained_passes = sum(
                1
                for attempt in attempt_results
                if attempt.journey_validation_result is not None
                and attempt.journey_validation_result.contained is True
            )
            journey_fulfillment_passes = sum(
                1
                for attempt in attempt_results
                if attempt.journey_validation_result is not None
                and attempt.journey_validation_result.fulfilled
            )
            journey_path_passes = sum(
                1
                for attempt in attempt_results
                if attempt.journey_validation_result is not None
                and attempt.journey_validation_result.path_correct
            )
            journey_category_match_passes = sum(
                1
                for attempt in attempt_results
                if attempt.journey_validation_result is not None
                and attempt.journey_validation_result.category_match is True
            )
            judging_scored_attempts = sum(
                1
                for attempt in attempt_results
                if attempt.judging_mechanics_result is not None
            )
            judging_threshold_passes = sum(
                1
                for attempt in attempt_results
                if attempt.judging_mechanics_result is not None
                and attempt.judging_mechanics_result.passed_threshold
            )
            judging_threshold_failures = max(
                0,
                judging_scored_attempts - judging_threshold_passes,
            )
            judging_scores = [
                float(attempt.judging_mechanics_result.score)
                for attempt in attempt_results
                if attempt.judging_mechanics_result is not None
            ]
            judging_average_score = (
                sum(judging_scores) / len(judging_scores)
                if judging_scores
                else 0.0
            )
            analytics_evaluated_attempts = sum(
                1
                for attempt in attempt_results
                if attempt.analytics_journey_result is not None
            )
            analytics_gate_passes = sum(
                1
                for attempt in attempt_results
                if attempt.analytics_journey_result is not None
                and attempt.success
            )
            analytics_skipped_unknown = sum(
                1
                for attempt in attempt_results
                if attempt.analytics_journey_result is not None
                and attempt.skipped
            )

            # Determine if this scenario is a regression
            is_regression = success_rate < self.config.success_threshold

            scenario_result = ScenarioResult(
                scenario_name=scenario.name,
                expected_intent=scenario_expected_intent,
                attempts=attempts_run,
                successes=successes,
                failures=failures,
                timeouts=timeouts,
                skipped=skipped,
                success_rate=success_rate,
                is_regression=is_regression,
                tool_validated_attempts=tool_validated_attempts,
                tool_loose_passes=tool_loose_passes,
                tool_strict_passes=tool_strict_passes,
                tool_missing_signal_count=tool_missing_signal_count,
                tool_order_mismatch_count=tool_order_mismatch_count,
                tool_loose_pass_rate=tool_loose_pass_rate,
                tool_strict_pass_rate=tool_strict_pass_rate,
                journey_validated_attempts=journey_validated_attempts,
                journey_passes=journey_passes,
                journey_contained_passes=journey_contained_passes,
                journey_fulfillment_passes=journey_fulfillment_passes,
                journey_path_passes=journey_path_passes,
                journey_category_match_passes=journey_category_match_passes,
                judging_scored_attempts=judging_scored_attempts,
                judging_threshold_passes=judging_threshold_passes,
                judging_threshold_failures=judging_threshold_failures,
                judging_average_score=judging_average_score,
                analytics_evaluated_attempts=analytics_evaluated_attempts,
                analytics_gate_passes=analytics_gate_passes,
                analytics_skipped_unknown=analytics_skipped_unknown,
                attempt_results=attempt_results,
            )
            scenario_results.append(scenario_result)

            # Emit scenario_completed
            self.progress_emitter.emit(ProgressEvent(
                event_type=ProgressEventType.SCENARIO_COMPLETED,
                suite_name=suite.name,
                scenario_name=scenario.name,
                expected_intent=scenario_expected_intent,
                success_rate=success_rate,
                message=f"Scenario completed: {scenario.name} — {success_rate:.0%} success rate",
            ))

        # Build the report
        duration = time.time() - start_time
        overall_attempts = sum(r.attempts for r in scenario_results)
        overall_successes = sum(r.successes for r in scenario_results)
        overall_failures = sum(r.failures for r in scenario_results)
        overall_timeouts = sum(r.timeouts for r in scenario_results)
        overall_skipped = sum(r.skipped for r in scenario_results)
        overall_success_rate = overall_successes / overall_attempts if overall_attempts > 0 else 0.0
        overall_tool_validated_attempts = sum(
            scenario.tool_validated_attempts for scenario in scenario_results
        )
        overall_tool_loose_passes = sum(
            scenario.tool_loose_passes for scenario in scenario_results
        )
        overall_tool_strict_passes = sum(
            scenario.tool_strict_passes for scenario in scenario_results
        )
        overall_tool_missing_signal_count = sum(
            scenario.tool_missing_signal_count for scenario in scenario_results
        )
        overall_tool_order_mismatch_count = sum(
            scenario.tool_order_mismatch_count for scenario in scenario_results
        )
        overall_tool_loose_pass_rate = (
            overall_tool_loose_passes / overall_tool_validated_attempts
            if overall_tool_validated_attempts > 0
            else 0.0
        )
        overall_tool_strict_pass_rate = (
            overall_tool_strict_passes / overall_tool_validated_attempts
            if overall_tool_validated_attempts > 0
            else 0.0
        )
        overall_journey_validated_attempts = sum(
            scenario.journey_validated_attempts for scenario in scenario_results
        )
        overall_journey_passes = sum(
            scenario.journey_passes for scenario in scenario_results
        )
        overall_journey_contained_passes = sum(
            scenario.journey_contained_passes for scenario in scenario_results
        )
        overall_journey_fulfillment_passes = sum(
            scenario.journey_fulfillment_passes for scenario in scenario_results
        )
        overall_journey_path_passes = sum(
            scenario.journey_path_passes for scenario in scenario_results
        )
        overall_journey_category_match_passes = sum(
            scenario.journey_category_match_passes for scenario in scenario_results
        )
        overall_judging_scored_attempts = sum(
            scenario.judging_scored_attempts for scenario in scenario_results
        )
        overall_judging_threshold_passes = sum(
            scenario.judging_threshold_passes for scenario in scenario_results
        )
        overall_judging_threshold_failures = sum(
            scenario.judging_threshold_failures for scenario in scenario_results
        )
        weighted_score_total = sum(
            scenario.judging_average_score * scenario.judging_scored_attempts
            for scenario in scenario_results
        )
        overall_judging_average_score = (
            weighted_score_total / overall_judging_scored_attempts
            if overall_judging_scored_attempts > 0
            else 0.0
        )
        overall_analytics_evaluated_attempts = sum(
            scenario.analytics_evaluated_attempts for scenario in scenario_results
        )
        overall_analytics_gate_passes = sum(
            scenario.analytics_gate_passes for scenario in scenario_results
        )
        overall_analytics_skipped_unknown = sum(
            scenario.analytics_skipped_unknown for scenario in scenario_results
        )
        has_regressions = any(r.is_regression for r in scenario_results)

        report = TestReport(
            suite_name=suite.name,
            timestamp=datetime.now(timezone.utc),
            duration_seconds=duration,
            scenario_results=scenario_results,
            overall_attempts=overall_attempts,
            overall_successes=overall_successes,
            overall_failures=overall_failures,
            overall_timeouts=overall_timeouts,
            overall_skipped=overall_skipped,
            overall_success_rate=overall_success_rate,
            overall_tool_validated_attempts=overall_tool_validated_attempts,
            overall_tool_loose_passes=overall_tool_loose_passes,
            overall_tool_strict_passes=overall_tool_strict_passes,
            overall_tool_missing_signal_count=overall_tool_missing_signal_count,
            overall_tool_order_mismatch_count=overall_tool_order_mismatch_count,
            overall_tool_loose_pass_rate=overall_tool_loose_pass_rate,
            overall_tool_strict_pass_rate=overall_tool_strict_pass_rate,
            overall_journey_validated_attempts=overall_journey_validated_attempts,
            overall_journey_passes=overall_journey_passes,
            overall_journey_contained_passes=overall_journey_contained_passes,
            overall_journey_fulfillment_passes=overall_journey_fulfillment_passes,
            overall_journey_path_passes=overall_journey_path_passes,
            overall_journey_category_match_passes=overall_journey_category_match_passes,
            overall_judging_scored_attempts=overall_judging_scored_attempts,
            overall_judging_threshold_passes=overall_judging_threshold_passes,
            overall_judging_threshold_failures=overall_judging_threshold_failures,
            overall_judging_average_score=overall_judging_average_score,
            overall_analytics_evaluated_attempts=overall_analytics_evaluated_attempts,
            overall_analytics_gate_passes=overall_analytics_gate_passes,
            overall_analytics_skipped_unknown=overall_analytics_skipped_unknown,
            has_regressions=has_regressions,
            regression_threshold=self.config.success_threshold,
        )

        if self.config.journey_dashboard_enabled:
            taxonomy_overrides: dict[str, str] = {}
            try:
                taxonomy_overrides = load_taxonomy_overrides(
                    overrides_json=self.config.journey_taxonomy_overrides_json,
                    overrides_file=self.config.journey_taxonomy_overrides_file,
                )
            except ValueError as e:
                self.progress_emitter.emit(ProgressEvent(
                    event_type=ProgressEventType.ATTEMPT_STATUS,
                    suite_name=suite.name,
                    message=(
                        "Journey taxonomy override config is invalid. "
                        f"Using built-in taxonomy rules. Details: {e}"
                    ),
                    planned_attempts=planned_attempts,
                    completed_attempts=completed_attempts,
                ))
            taxonomy_rollups = build_journey_taxonomy_rollups(
                report,
                overrides=taxonomy_overrides,
                active_view="overview",
            )
            report.journey_taxonomy_rollups = [
                JourneyTaxonomyRollup.model_validate(row)
                for row in taxonomy_rollups["labels"]
            ]

        # Emit suite_completed
        completed_message = f"Suite completed: {suite.name} in {duration:.1f}s"
        if self.stop_event is not None and self.stop_event.is_set():
            completed_message = f"Suite stopped early: {suite.name} after {duration:.1f}s"
        self.progress_emitter.emit(ProgressEvent(
            event_type=ProgressEventType.SUITE_COMPLETED,
            suite_name=suite.name,
            message=completed_message,
            duration_seconds=duration,
            planned_attempts=planned_attempts,
            completed_attempts=completed_attempts,
        ))

        return report

    def determine_regressions(self, report: TestReport, threshold: float) -> list[str]:
        """Return list of scenario names with success_rate below the threshold.

        Args:
            report: The completed test report.
            threshold: The success rate threshold (0.0 to 1.0).

        Returns:
            List of scenario names that are flagged as regressions.
        """
        return [
            result.scenario_name
            for result in report.scenario_results
            if result.success_rate < threshold
        ]
