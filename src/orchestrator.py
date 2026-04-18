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

        # Emit suite_started
        self.progress_emitter.emit(ProgressEvent(
            event_type=ProgressEventType.SUITE_STARTED,
            message=f"Starting test suite: {suite.name}",
        ))

        # Create internal dependencies
        judge = JudgeLLMClient(
            base_url=self.config.ollama_base_url,
            model=self.config.ollama_model or "",
            timeout=self.config.response_timeout,
        )
        web_msg_config = {
            "region": self.config.gc_region or "",
            "deployment_id": self.config.gc_deployment_id or "",
            "timeout": self.config.response_timeout,
            "origin": self.config.gc_origin,
            "expected_greeting": self.config.expected_greeting,
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
                result = await runner.run_attempt(scenario, attempt_num)
                attempt_results.append(result)

                if result.success:
                    successes += 1
                if result.timed_out:
                    timeouts += 1

                # Emit attempt_completed
                self.progress_emitter.emit(ProgressEvent(
                    event_type=ProgressEventType.ATTEMPT_COMPLETED,
                    scenario_name=scenario.name,
                    attempt_number=result.attempt_number,
                    success=result.success,
                    message=f"Attempt {result.attempt_number}: {'success' if result.success else 'failure'}",
                    attempt_result=result,
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
            message=completed_message,
            duration_seconds=duration,
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
