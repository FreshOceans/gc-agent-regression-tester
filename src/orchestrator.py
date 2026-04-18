"""Test Orchestrator for coordinating test suite execution."""

import asyncio
import time
from datetime import datetime, timezone
from threading import Event
from typing import Optional

from .conversation_runner import ConversationRunner
from .judge_llm import JudgeLLMClient
from .models import (
    AppConfig,
    AttemptResult,
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
                await asyncio.to_thread(judge.warm_up)
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
        web_msg_config = {
            "region": self.config.gc_region or "",
            "deployment_id": self.config.gc_deployment_id or "",
            "timeout": self.config.response_timeout,
            "origin": self._build_origin_from_region(self.config.gc_region or ""),
            "expected_greeting": self.config.expected_greeting,
            "gc_client_id": self.config.gc_client_id or "",
            "gc_client_secret": self.config.gc_client_secret or "",
            "intent_attribute_name": self.config.intent_attribute_name,
            "debug_capture_frames": self.config.debug_capture_frames,
            "debug_capture_frame_limit": self.config.debug_capture_frame_limit,
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

            # Emit scenario_started
            self.progress_emitter.emit(ProgressEvent(
                event_type=ProgressEventType.SCENARIO_STARTED,
                suite_name=suite.name,
                scenario_name=scenario.name,
                message=f"Starting scenario: {scenario.name} ({attempt_count} attempts)",
            ))

            attempt_results: list[AttemptResult] = []
            successes = 0
            timeouts = 0

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

                # Emit attempt_completed
                self.progress_emitter.emit(ProgressEvent(
                    event_type=ProgressEventType.ATTEMPT_COMPLETED,
                    suite_name=suite.name,
                    scenario_name=scenario.name,
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

            failures = attempts_run - successes
            success_rate = successes / attempts_run if attempts_run > 0 else 0.0

            # Determine if this scenario is a regression
            is_regression = success_rate < self.config.success_threshold

            scenario_result = ScenarioResult(
                scenario_name=scenario.name,
                attempts=attempts_run,
                successes=successes,
                failures=failures,
                timeouts=timeouts,
                success_rate=success_rate,
                is_regression=is_regression,
                attempt_results=attempt_results,
            )
            scenario_results.append(scenario_result)

            # Emit scenario_completed
            self.progress_emitter.emit(ProgressEvent(
                event_type=ProgressEventType.SCENARIO_COMPLETED,
                suite_name=suite.name,
                scenario_name=scenario.name,
                success_rate=success_rate,
                message=f"Scenario completed: {scenario.name} — {success_rate:.0%} success rate",
            ))

        # Build the report
        duration = time.time() - start_time
        overall_attempts = sum(r.attempts for r in scenario_results)
        overall_successes = sum(r.successes for r in scenario_results)
        overall_failures = sum(r.failures for r in scenario_results)
        overall_timeouts = sum(r.timeouts for r in scenario_results)
        overall_success_rate = overall_successes / overall_attempts if overall_attempts > 0 else 0.0
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
            overall_success_rate=overall_success_rate,
            has_regressions=has_regressions,
            regression_threshold=self.config.success_threshold,
        )

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
