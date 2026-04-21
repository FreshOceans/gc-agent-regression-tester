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
    AdaptivePacingAdjustment,
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
            "knowledge_mode_timeout_seconds": self.config.knowledge_mode_timeout_seconds,
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
            "attempt_parallel_enabled": self.config.attempt_parallel_enabled,
            "max_parallel_attempt_workers": self.config.max_parallel_attempt_workers,
            "min_attempt_interval_seconds": self.config.min_attempt_interval_seconds,
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
        scenario_states: list[dict] = []
        attempt_queue: asyncio.Queue[tuple[int, int]] = asyncio.Queue()
        for scenario_index, scenario in enumerate(suite.scenarios):
            attempt_count = (
                scenario.attempts
                if scenario.attempts is not None
                else self.config.default_attempts
            )
            scenario_expected_intent = (
                scenario.expected_intent.strip().lower()
                if scenario.expected_intent
                else None
            )
            scenario_states.append({
                "index": scenario_index,
                "scenario": scenario,
                "attempt_count": attempt_count,
                "expected_intent": scenario_expected_intent,
                "attempt_results": [],
                "successes": 0,
                "timeouts": 0,
                "skipped": 0,
                "started_attempts": 0,
                "completed_attempts": 0,
                "started_emitted": False,
                "completed_emitted": False,
            })
            for attempt_number in range(1, attempt_count + 1):
                attempt_queue.put_nowait((scenario_index, attempt_number))

        max_workers = max(1, min(int(self.config.max_parallel_attempt_workers), 3))
        worker_count = (
            max_workers
            if bool(self.config.attempt_parallel_enabled)
            else 1
        )
        adaptive_window_size = 20
        adaptive_signal_threshold_high = 0.15
        adaptive_signal_threshold_low = 0.05
        adaptive_increase_step = 1.0
        adaptive_decrease_step = 0.5
        adaptive_interval_floor = 5.0
        adaptive_interval_ceiling = 10.0
        adaptive_enabled = bool(self.config.adaptive_attempt_pacing_enabled)
        adaptive_base_interval = max(0.0, float(self.config.min_attempt_interval_seconds))
        adaptive_current_interval = adaptive_base_interval
        adaptive_signals_in_window = 0
        adaptive_attempts_in_window = 0
        adaptive_healthy_window_streak = 0
        adaptive_adjustments: list[AdaptivePacingAdjustment] = []
        event_lock = asyncio.Lock()
        start_rate_lock = asyncio.Lock()
        global_last_start_monotonic: Optional[float] = None

        def is_adaptive_pressure_signal(attempt: AttemptResult) -> bool:
            timeout_diag = attempt.timeout_diagnostics
            if (
                timeout_diag is not None
                and str(timeout_diag.timeout_class or "").strip().lower()
                == "greeting_gate"
            ):
                return True
            failure_diag = attempt.failure_diagnostics
            if (
                failure_diag is not None
                and str(failure_diag.failure_class or "").strip().lower()
                == "upstream_agent_error_before_greeting"
            ):
                return True
            return False

        async def acquire_attempt_start_slot() -> None:
            """Global pacing gate for attempt starts across all workers."""
            nonlocal global_last_start_monotonic, adaptive_current_interval
            current_interval = max(0.0, float(adaptive_current_interval))
            if current_interval <= 0:
                return
            while True:
                async with start_rate_lock:
                    now = time.monotonic()
                    if global_last_start_monotonic is None:
                        global_last_start_monotonic = now
                        return
                    elapsed = now - global_last_start_monotonic
                    if elapsed >= current_interval:
                        global_last_start_monotonic = now
                        return
                    sleep_for = current_interval - elapsed
                await asyncio.sleep(sleep_for)

        async def run_worker() -> None:
            nonlocal adaptive_attempts_in_window
            nonlocal adaptive_current_interval
            nonlocal adaptive_healthy_window_streak
            nonlocal adaptive_signals_in_window
            nonlocal completed_attempts

            while True:
                if self.stop_event is not None and self.stop_event.is_set():
                    break
                try:
                    scenario_index, attempt_num = attempt_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

                state = scenario_states[scenario_index]
                scenario = state["scenario"]
                scenario_name = scenario.name
                scenario_expected_intent = state["expected_intent"]

                await acquire_attempt_start_slot()

                async with event_lock:
                    if not state["started_emitted"]:
                        self.progress_emitter.emit(ProgressEvent(
                            event_type=ProgressEventType.SCENARIO_STARTED,
                            suite_name=suite.name,
                            scenario_name=scenario_name,
                            expected_intent=scenario_expected_intent,
                            message=(
                                f"Starting scenario: {scenario_name} "
                                f"({state['attempt_count']} attempts)"
                            ),
                        ))
                        state["started_emitted"] = True

                    state["started_attempts"] += 1
                    self.progress_emitter.emit(ProgressEvent(
                        event_type=ProgressEventType.ATTEMPT_STARTED,
                        suite_name=suite.name,
                        scenario_name=scenario_name,
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
                        scenario_name=scenario_name,
                        expected_intent=scenario_expected_intent,
                        attempt_number=attempt_num,
                        message=status_message,
                        planned_attempts=planned_attempts,
                        completed_attempts=completed_attempts,
                    ))

                runner = ConversationRunner(
                    judge=judge,
                    web_msg_config=web_msg_config,
                    max_turns=self.config.max_turns,
                )
                result = await runner.run_attempt(
                    scenario,
                    attempt_num,
                    status_callback=emit_attempt_status,
                )

                async with event_lock:
                    state["attempt_results"].append(result)
                    state["completed_attempts"] += 1
                    completed_attempts += 1

                    if result.success:
                        state["successes"] += 1
                    if result.timed_out:
                        state["timeouts"] += 1
                    if result.skipped:
                        state["skipped"] += 1

                    self.progress_emitter.emit(ProgressEvent(
                        event_type=ProgressEventType.ATTEMPT_COMPLETED,
                        suite_name=suite.name,
                        scenario_name=scenario_name,
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

                    if adaptive_enabled:
                        adaptive_attempts_in_window += 1
                        if is_adaptive_pressure_signal(result):
                            adaptive_signals_in_window += 1
                        if adaptive_attempts_in_window >= adaptive_window_size:
                            signal_rate = (
                                adaptive_signals_in_window / adaptive_window_size
                            )
                            previous_interval = adaptive_current_interval
                            adjustment_reason: Optional[str] = None
                            if signal_rate > adaptive_signal_threshold_high:
                                adaptive_current_interval = min(
                                    adaptive_interval_ceiling,
                                    adaptive_current_interval + adaptive_increase_step,
                                )
                                adaptive_healthy_window_streak = 0
                                if adaptive_current_interval > previous_interval:
                                    adjustment_reason = "pressure_window_high"
                            elif signal_rate < adaptive_signal_threshold_low:
                                adaptive_healthy_window_streak += 1
                                if adaptive_healthy_window_streak >= 2:
                                    adaptive_current_interval = max(
                                        adaptive_interval_floor,
                                        adaptive_current_interval - adaptive_decrease_step,
                                    )
                                    adaptive_healthy_window_streak = 0
                                    if adaptive_current_interval < previous_interval:
                                        adjustment_reason = "pressure_window_low"
                            else:
                                adaptive_healthy_window_streak = 0

                            if adjustment_reason:
                                adjustment = AdaptivePacingAdjustment(
                                    attempt_window_end=completed_attempts,
                                    window_size=adaptive_window_size,
                                    signal_count=adaptive_signals_in_window,
                                    signal_rate=signal_rate,
                                    from_interval_seconds=previous_interval,
                                    to_interval_seconds=adaptive_current_interval,
                                    reason=adjustment_reason,
                                )
                                adaptive_adjustments.append(adjustment)
                                self.progress_emitter.emit(ProgressEvent(
                                    event_type=ProgressEventType.ATTEMPT_STATUS,
                                    suite_name=suite.name,
                                    scenario_name=scenario_name,
                                    expected_intent=scenario_expected_intent,
                                    attempt_number=result.attempt_number,
                                    message=(
                                        "Adaptive pacing adjusted interval: "
                                        f"{previous_interval:.1f}s -> {adaptive_current_interval:.1f}s "
                                        f"(window={adaptive_window_size}, signals={adaptive_signals_in_window}, "
                                        f"rate={signal_rate:.1%}, reason={adjustment_reason})"
                                    ),
                                    planned_attempts=planned_attempts,
                                    completed_attempts=completed_attempts,
                                ))

                            adaptive_attempts_in_window = 0
                            adaptive_signals_in_window = 0

                    scenario_done = (
                        state["started_emitted"]
                        and not state["completed_emitted"]
                        and state["completed_attempts"] >= state["started_attempts"]
                        and (
                            state["started_attempts"] >= state["attempt_count"]
                            or (
                                self.stop_event is not None
                                and self.stop_event.is_set()
                            )
                        )
                    )
                    if scenario_done:
                        attempts_run = len(state["attempt_results"])
                        success_rate = (
                            state["successes"] / attempts_run
                            if attempts_run > 0
                            else 0.0
                        )
                        self.progress_emitter.emit(ProgressEvent(
                            event_type=ProgressEventType.SCENARIO_COMPLETED,
                            suite_name=suite.name,
                            scenario_name=scenario_name,
                            expected_intent=scenario_expected_intent,
                            success_rate=success_rate,
                            message=(
                                f"Scenario completed: {scenario_name} — "
                                f"{success_rate:.0%} success rate"
                            ),
                        ))
                        state["completed_emitted"] = True

        workers = [asyncio.create_task(run_worker()) for _ in range(worker_count)]
        await asyncio.gather(*workers)

        scenario_results: list[ScenarioResult] = []
        for state in scenario_states:
            scenario = state["scenario"]
            scenario_expected_intent = state["expected_intent"]
            attempt_results: list[AttemptResult] = sorted(
                state["attempt_results"],
                key=lambda attempt: attempt.attempt_number,
            )
            attempts_run = len(attempt_results)
            if attempts_run == 0:
                continue

            successes = int(state["successes"])
            timeouts = int(state["timeouts"])
            skipped = int(state["skipped"])
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
            adaptive_attempt_pacing_enabled=adaptive_enabled,
            adaptive_attempt_pacing_base_interval_seconds=adaptive_base_interval,
            adaptive_attempt_pacing_final_interval_seconds=adaptive_current_interval,
            adaptive_attempt_pacing_adjustment_count=len(adaptive_adjustments),
            adaptive_attempt_pacing_adjustments=adaptive_adjustments,
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
