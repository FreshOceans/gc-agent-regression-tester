"""Unit tests for the Test Orchestrator."""

import asyncio
import threading
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.models import (
    AppConfig,
    AttemptResult,
    FailureDiagnostics,
    Message,
    MessageRole,
    ProgressEvent,
    ProgressEventType,
    ScenarioResult,
    TestReport,
    TestScenario,
    TestSuite,
    TimeoutDiagnostics,
)
from src.orchestrator import TestOrchestrator
from src.progress import ProgressEmitter


@pytest.fixture
def app_config():
    """Create a basic AppConfig for testing."""
    return AppConfig(
        gc_region="us-east-1",
        gc_deployment_id="deploy-123",
        ollama_base_url="http://localhost:11434",
        ollama_model="llama3",
        default_attempts=3,
        max_turns=10,
        min_attempt_interval_seconds=0,
        attempt_parallel_enabled=False,
        max_parallel_attempt_workers=1,
        response_timeout=30,
        success_threshold=0.8,
        judge_warmup_enabled=False,
    )


@pytest.fixture
def progress_emitter():
    """Create a ProgressEmitter for testing."""
    return ProgressEmitter()


@pytest.fixture
def simple_suite():
    """Create a simple test suite with one scenario."""
    return TestSuite(
        name="Test Suite",
        scenarios=[
            TestScenario(
                name="Scenario A",
                persona="A helpful customer",
                goal="Book a meeting",
                attempts=2,
            )
        ],
    )


@pytest.fixture
def multi_scenario_suite():
    """Create a test suite with multiple scenarios."""
    return TestSuite(
        name="Multi Suite",
        scenarios=[
            TestScenario(
                name="Scenario A",
                persona="Customer A",
                goal="Goal A",
                attempts=2,
            ),
            TestScenario(
                name="Scenario B",
                persona="Customer B",
                goal="Goal B",
                attempts=3,
            ),
        ],
    )


def make_attempt_result(
    attempt_number: int, success: bool, timed_out: bool = False
) -> AttemptResult:
    """Helper to create an AttemptResult."""
    return AttemptResult(
        attempt_number=attempt_number,
        success=success,
        timed_out=timed_out,
        conversation=[
            Message(role=MessageRole.AGENT, content="Welcome!"),
            Message(role=MessageRole.USER, content="Hi"),
        ],
        explanation="Goal achieved" if success else "Goal not achieved",
    )


def make_greeting_pressure_timeout_result(attempt_number: int) -> AttemptResult:
    return AttemptResult(
        attempt_number=attempt_number,
        success=False,
        timed_out=True,
        conversation=[
            Message(role=MessageRole.AGENT, content="What is your language preference?"),
            Message(role=MessageRole.USER, content="english"),
        ],
        explanation="Attempt failed due to timeout",
        error="Expected greeting was not received before sending first scenario user message (13.0s)",
        timeout_diagnostics=TimeoutDiagnostics(
            timeout_class="greeting_gate",
            step_name="Waiting for expected greeting before sending first user message",
            step_timeout_seconds=13.0,
        ),
    )


def make_pregreeting_failure_result(attempt_number: int) -> AttemptResult:
    return AttemptResult(
        attempt_number=attempt_number,
        success=False,
        conversation=[
            Message(role=MessageRole.AGENT, content="What is your language preference?"),
            Message(role=MessageRole.USER, content="english"),
            Message(
                role=MessageRole.AGENT,
                content="Sorry, an error occurred. One moment, please, while I put you through to someone who can help.",
            ),
        ],
        explanation="Attempt failed before greeting because the agent returned a terminal error or handoff response.",
        error="upstream_agent_error_before_greeting (en_error_handoff)",
        failure_diagnostics=FailureDiagnostics(
            failure_class="upstream_agent_error_before_greeting",
            gate_step="Waiting for expected greeting before sending first user message",
            matched_pattern_id="en_error_handoff",
        ),
    )


class TestTestOrchestrator:
    """Tests for TestOrchestrator."""

    def test_init(self, app_config, progress_emitter):
        """Test orchestrator initialization."""
        orchestrator = TestOrchestrator(config=app_config, progress_emitter=progress_emitter)
        assert orchestrator.config == app_config
        assert orchestrator.progress_emitter == progress_emitter

    @pytest.mark.asyncio
    async def test_run_suite_emits_suite_started(self, app_config, progress_emitter, simple_suite):
        """Test that run_suite emits a suite_started event."""
        orchestrator = TestOrchestrator(config=app_config, progress_emitter=progress_emitter)
        q = progress_emitter.subscribe()

        with patch("src.orchestrator.ConversationRunner") as MockRunner:
            mock_runner_instance = MockRunner.return_value
            mock_runner_instance.run_attempt = AsyncMock(
                return_value=make_attempt_result(1, True)
            )
            await orchestrator.run_suite(simple_suite)

        # First event should be suite_started
        first_event = q.get_nowait()
        assert first_event.event_type == ProgressEventType.SUITE_STARTED
        assert "Test Suite" in first_event.message
        assert first_event.planned_attempts == 2
        assert first_event.completed_attempts == 0

    @pytest.mark.asyncio
    async def test_run_suite_emits_scenario_started(self, app_config, progress_emitter, simple_suite):
        """Test that run_suite emits scenario_started events."""
        simple_suite.scenarios[0].expected_intent = "flight_cancel"
        orchestrator = TestOrchestrator(config=app_config, progress_emitter=progress_emitter)
        q = progress_emitter.subscribe()

        with patch("src.orchestrator.ConversationRunner") as MockRunner:
            mock_runner_instance = MockRunner.return_value
            mock_runner_instance.run_attempt = AsyncMock(
                return_value=make_attempt_result(1, True)
            )
            await orchestrator.run_suite(simple_suite)

        events = []
        while not q.empty():
            events.append(q.get_nowait())

        scenario_started_events = [
            e for e in events if e.event_type == ProgressEventType.SCENARIO_STARTED
        ]
        assert len(scenario_started_events) == 1
        event = scenario_started_events[0]
        assert event.event_type == ProgressEventType.SCENARIO_STARTED
        assert event.scenario_name == "Scenario A"
        assert event.expected_intent == "flight_cancel"

    @pytest.mark.asyncio
    async def test_run_suite_emits_judge_warmup_status(self, app_config, progress_emitter, simple_suite):
        """Test that run_suite emits warm-up status events before attempts."""
        app_config.judge_warmup_enabled = True
        orchestrator = TestOrchestrator(config=app_config, progress_emitter=progress_emitter)
        q = progress_emitter.subscribe()
        mock_judge_instance = MagicMock()
        mock_judge_instance.warm_up = MagicMock(return_value="OK")

        with patch("src.orchestrator.ConversationRunner") as MockRunner, patch(
            "src.orchestrator.build_judge_execution_client",
            return_value=mock_judge_instance,
        ):
            mock_runner_instance = MockRunner.return_value
            mock_runner_instance.run_attempt = AsyncMock(
                return_value=make_attempt_result(1, True)
            )
            await orchestrator.run_suite(simple_suite)

        events = []
        while not q.empty():
            events.append(q.get_nowait())

        warmup_events = [
            e for e in events
            if e.event_type == ProgressEventType.ATTEMPT_STATUS
            and e.scenario_name is None
        ]
        warmup_messages = [e.message for e in warmup_events]
        assert "Warming up Judge LLM model" in warmup_messages
        assert any("Judge LLM warm-up complete" in message for message in warmup_messages)
        mock_judge_instance.warm_up.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_suite_emits_attempt_completed(self, app_config, progress_emitter, simple_suite):
        """Test that run_suite emits attempt_completed events for each attempt."""
        simple_suite.scenarios[0].expected_intent = "flight_cancel"
        orchestrator = TestOrchestrator(config=app_config, progress_emitter=progress_emitter)
        q = progress_emitter.subscribe()

        with patch("src.orchestrator.ConversationRunner") as MockRunner:
            mock_runner_instance = MockRunner.return_value
            mock_runner_instance.run_attempt = AsyncMock(
                side_effect=[
                    make_attempt_result(1, True),
                    make_attempt_result(2, False),
                ]
            )
            await orchestrator.run_suite(simple_suite)

        events = []
        while not q.empty():
            events.append(q.get_nowait())

        attempt_events = [e for e in events if e.event_type == ProgressEventType.ATTEMPT_COMPLETED]
        assert len(attempt_events) == 2
        assert attempt_events[0].attempt_number == 1
        assert attempt_events[0].success is True
        assert attempt_events[0].expected_intent == "flight_cancel"
        assert attempt_events[0].planned_attempts == 2
        assert attempt_events[0].completed_attempts == 1
        assert attempt_events[1].attempt_number == 2
        assert attempt_events[1].success is False
        assert attempt_events[1].expected_intent == "flight_cancel"
        assert attempt_events[1].planned_attempts == 2
        assert attempt_events[1].completed_attempts == 2

    @pytest.mark.asyncio
    async def test_run_suite_emits_attempt_started(self, app_config, progress_emitter, simple_suite):
        """Test that run_suite emits attempt_started before attempts execute."""
        orchestrator = TestOrchestrator(config=app_config, progress_emitter=progress_emitter)
        q = progress_emitter.subscribe()

        with patch("src.orchestrator.ConversationRunner") as MockRunner:
            mock_runner_instance = MockRunner.return_value
            mock_runner_instance.run_attempt = AsyncMock(
                return_value=make_attempt_result(1, True)
            )
            await orchestrator.run_suite(simple_suite)

        events = []
        while not q.empty():
            events.append(q.get_nowait())

        started = [e for e in events if e.event_type == ProgressEventType.ATTEMPT_STARTED]
        assert len(started) == 2
        assert started[0].attempt_number == 1
        assert started[0].planned_attempts == 2
        assert started[0].completed_attempts == 0
        assert started[1].attempt_number == 2

    @pytest.mark.asyncio
    async def test_run_suite_emits_scenario_completed(self, app_config, progress_emitter, simple_suite):
        """Test that run_suite emits scenario_completed with success rate."""
        orchestrator = TestOrchestrator(config=app_config, progress_emitter=progress_emitter)
        q = progress_emitter.subscribe()

        with patch("src.orchestrator.ConversationRunner") as MockRunner:
            mock_runner_instance = MockRunner.return_value
            mock_runner_instance.run_attempt = AsyncMock(
                side_effect=[
                    make_attempt_result(1, True),
                    make_attempt_result(2, False),
                ]
            )
            await orchestrator.run_suite(simple_suite)

        events = []
        while not q.empty():
            events.append(q.get_nowait())

        scenario_completed = [e for e in events if e.event_type == ProgressEventType.SCENARIO_COMPLETED]
        assert len(scenario_completed) == 1
        assert scenario_completed[0].scenario_name == "Scenario A"
        assert scenario_completed[0].success_rate == 0.5

    @pytest.mark.asyncio
    async def test_run_suite_emits_suite_completed(self, app_config, progress_emitter, simple_suite):
        """Test that run_suite emits suite_completed with duration."""
        orchestrator = TestOrchestrator(config=app_config, progress_emitter=progress_emitter)
        q = progress_emitter.subscribe()

        with patch("src.orchestrator.ConversationRunner") as MockRunner:
            mock_runner_instance = MockRunner.return_value
            mock_runner_instance.run_attempt = AsyncMock(
                return_value=make_attempt_result(1, True)
            )
            await orchestrator.run_suite(simple_suite)

        events = []
        while not q.empty():
            events.append(q.get_nowait())

        suite_completed = [e for e in events if e.event_type == ProgressEventType.SUITE_COMPLETED]
        assert len(suite_completed) == 1
        assert suite_completed[0].duration_seconds is not None
        assert suite_completed[0].duration_seconds >= 0
        assert suite_completed[0].planned_attempts == 2
        assert suite_completed[0].completed_attempts == 2

    @pytest.mark.asyncio
    async def test_run_suite_returns_test_report(self, app_config, progress_emitter, simple_suite):
        """Test that run_suite returns a valid TestReport."""
        simple_suite.scenarios[0].expected_intent = "flight_cancel"
        orchestrator = TestOrchestrator(config=app_config, progress_emitter=progress_emitter)

        with patch("src.orchestrator.ConversationRunner") as MockRunner:
            mock_runner_instance = MockRunner.return_value
            mock_runner_instance.run_attempt = AsyncMock(
                side_effect=[
                    make_attempt_result(1, True),
                    make_attempt_result(2, False),
                ]
            )
            report = await orchestrator.run_suite(simple_suite)

        assert isinstance(report, TestReport)
        assert report.suite_name == "Test Suite"
        assert report.overall_attempts == 2
        assert report.overall_successes == 1
        assert report.overall_failures == 1
        assert report.overall_success_rate == 0.5
        assert report.has_regressions is True

    @pytest.mark.asyncio
    async def test_run_suite_journey_taxonomy_disabled_by_default(
        self,
        app_config,
        progress_emitter,
        simple_suite,
    ):
        """Journey taxonomy rollups should remain empty when dashboard flag is disabled."""
        simple_suite.scenarios[0].expected_intent = "speak_to_agent"
        attempt = AttemptResult(
            attempt_number=1,
            success=True,
            conversation=[
                Message(role=MessageRole.AGENT, content="I can transfer to live agent now."),
                Message(role=MessageRole.USER, content="yes please"),
            ],
            explanation="ok",
            detected_intent="speak_to_agent",
        )
        orchestrator = TestOrchestrator(config=app_config, progress_emitter=progress_emitter)

        with patch("src.orchestrator.ConversationRunner") as MockRunner:
            MockRunner.return_value.run_attempt = AsyncMock(side_effect=[attempt, attempt])
            report = await orchestrator.run_suite(simple_suite)

        assert report.journey_taxonomy_rollups == []
        assert all(
            result.journey_taxonomy_label is None
            for scenario in report.scenario_results
            for result in scenario.attempt_results
        )

    @pytest.mark.asyncio
    async def test_run_suite_journey_taxonomy_populates_when_enabled(
        self,
        app_config,
        progress_emitter,
        simple_suite,
    ):
        """Journey taxonomy rollups should populate deterministic labels when enabled."""
        app_config.journey_dashboard_enabled = True
        simple_suite.scenarios[0].expected_intent = "speak_to_agent"
        attempt = AttemptResult(
            attempt_number=1,
            success=True,
            conversation=[
                Message(role=MessageRole.AGENT, content="I can transfer to live agent now."),
                Message(role=MessageRole.USER, content="yes please"),
            ],
            explanation="ok",
            detected_intent="speak_to_agent",
        )
        orchestrator = TestOrchestrator(config=app_config, progress_emitter=progress_emitter)

        with patch("src.orchestrator.ConversationRunner") as MockRunner:
            MockRunner.return_value.run_attempt = AsyncMock(side_effect=[attempt, attempt])
            report = await orchestrator.run_suite(simple_suite)

        labels = {row.label: row.count for row in report.journey_taxonomy_rollups}
        assert "Total Calls" in labels
        assert labels["Total Calls"] == 2
        assert labels["Agent Request - Successful Transfer To Agent"] == 2
        assert all(
            result.journey_taxonomy_label == "Agent Request - Successful Transfer To Agent"
            for scenario in report.scenario_results
            for result in scenario.attempt_results
        )
        assert report.regression_threshold == 0.8
        assert report.duration_seconds >= 0
        assert len(report.scenario_results) == 1

    @pytest.mark.asyncio
    async def test_run_suite_stops_early_when_stop_event_is_set(
        self, app_config, progress_emitter, multi_scenario_suite
    ):
        """Test graceful stop behavior when a stop event is requested."""
        stop_event = threading.Event()
        orchestrator = TestOrchestrator(
            config=app_config,
            progress_emitter=progress_emitter,
            stop_event=stop_event,
        )

        async def run_attempt_and_stop(*args, **kwargs):
            stop_event.set()
            return make_attempt_result(1, False)

        with patch("src.orchestrator.ConversationRunner") as MockRunner:
            mock_runner_instance = MockRunner.return_value
            mock_runner_instance.run_attempt = AsyncMock(
                side_effect=run_attempt_and_stop
            )
            report = await orchestrator.run_suite(multi_scenario_suite)

        # First scenario should only run one attempt and second scenario should not run.
        assert report.overall_attempts == 1
        assert len(report.scenario_results) == 1
        assert report.scenario_results[0].scenario_name == "Scenario A"

    @pytest.mark.asyncio
    async def test_run_suite_counts_timeouts(self, app_config, progress_emitter, simple_suite):
        """Timeout attempts should be counted in scenario and report totals."""
        orchestrator = TestOrchestrator(config=app_config, progress_emitter=progress_emitter)

        with patch("src.orchestrator.ConversationRunner") as MockRunner:
            mock_runner_instance = MockRunner.return_value
            mock_runner_instance.run_attempt = AsyncMock(
                side_effect=[
                    make_attempt_result(1, False, timed_out=True),
                    make_attempt_result(2, False, timed_out=False),
                ]
            )
            report = await orchestrator.run_suite(simple_suite)

        assert report.overall_timeouts == 1
        assert report.scenario_results[0].timeouts == 1

    @pytest.mark.asyncio
    async def test_run_suite_counts_skipped(self, app_config, progress_emitter, simple_suite):
        """Skipped attempts should be counted in scenario and report totals."""
        orchestrator = TestOrchestrator(config=app_config, progress_emitter=progress_emitter)
        first = make_attempt_result(1, False, timed_out=False)
        first.skipped = True
        second = make_attempt_result(2, False, timed_out=False)

        with patch("src.orchestrator.ConversationRunner") as MockRunner:
            mock_runner_instance = MockRunner.return_value
            mock_runner_instance.run_attempt = AsyncMock(
                side_effect=[first, second]
            )

            report = await orchestrator.run_suite(simple_suite)

        assert report.overall_skipped == 1
        assert report.scenario_results[0].skipped == 1
        assert report.scenario_results[0].failures == 1

    @pytest.mark.asyncio
    async def test_run_suite_throttles_attempt_start_rate(
        self, app_config, progress_emitter, simple_suite
    ):
        """Attempts should be globally rate-limited across parallel workers."""
        app_config.min_attempt_interval_seconds = 0.02
        app_config.attempt_parallel_enabled = True
        app_config.max_parallel_attempt_workers = 2
        orchestrator = TestOrchestrator(config=app_config, progress_emitter=progress_emitter)
        progress_queue = progress_emitter.subscribe()

        with patch("src.orchestrator.ConversationRunner") as MockRunner:
            mock_runner_instance = MockRunner.return_value
            mock_runner_instance.run_attempt = AsyncMock(
                side_effect=[
                    make_attempt_result(1, True),
                    make_attempt_result(2, True),
                ]
            )
            await orchestrator.run_suite(simple_suite)

        started_events = []
        while not progress_queue.empty():
            event = progress_queue.get_nowait()
            if event.event_type == ProgressEventType.ATTEMPT_STARTED:
                started_events.append(event)

        assert len(started_events) == 2
        delta_seconds = (
            started_events[1].emitted_at - started_events[0].emitted_at
        ).total_seconds()
        assert delta_seconds >= 0.015

    @pytest.mark.asyncio
    async def test_run_suite_adaptive_pacing_increases_after_high_pressure_window(
        self, app_config, progress_emitter
    ):
        suite = TestSuite(
            name="Adaptive Increase Suite",
            scenarios=[
                TestScenario(
                    name="Scenario High Pressure",
                    persona="Traveler",
                    goal="Cancel booking",
                    attempts=20,
                )
            ],
        )
        app_config.min_attempt_interval_seconds = 5.0
        app_config.adaptive_attempt_pacing_enabled = True
        app_config.attempt_parallel_enabled = True
        app_config.max_parallel_attempt_workers = 2
        orchestrator = TestOrchestrator(config=app_config, progress_emitter=progress_emitter)
        progress_queue = progress_emitter.subscribe()

        with patch("src.orchestrator.ConversationRunner") as MockRunner, patch(
            "src.orchestrator.time.monotonic", side_effect=range(1000, 2000, 10)
        ):
            mock_runner_instance = MockRunner.return_value
            mock_runner_instance.run_attempt = AsyncMock(
                side_effect=[
                    (
                        make_greeting_pressure_timeout_result(i)
                        if i % 2 == 0
                        else make_pregreeting_failure_result(i)
                    )
                    for i in range(1, 21)
                ]
            )
            report = await orchestrator.run_suite(suite)

        assert report.adaptive_attempt_pacing_enabled is True
        assert report.adaptive_attempt_pacing_base_interval_seconds == pytest.approx(5.0)
        assert report.adaptive_attempt_pacing_final_interval_seconds == pytest.approx(6.0)
        assert report.adaptive_attempt_pacing_adjustment_count == 1
        assert len(report.adaptive_attempt_pacing_adjustments) == 1
        adjustment = report.adaptive_attempt_pacing_adjustments[0]
        assert adjustment.reason == "pressure_window_high"
        assert adjustment.signal_count == 20
        assert adjustment.signal_rate == pytest.approx(1.0)

        status_messages = []
        while not progress_queue.empty():
            event = progress_queue.get_nowait()
            if event.event_type == ProgressEventType.ATTEMPT_STATUS:
                status_messages.append(event.message)
        assert any("Adaptive pacing adjusted interval" in msg for msg in status_messages)

    @pytest.mark.asyncio
    async def test_run_suite_adaptive_pacing_decreases_after_two_healthy_windows(
        self, app_config, progress_emitter
    ):
        suite = TestSuite(
            name="Adaptive Decrease Suite",
            scenarios=[
                TestScenario(
                    name="Scenario Healthy",
                    persona="Traveler",
                    goal="Flight status",
                    attempts=40,
                )
            ],
        )
        app_config.min_attempt_interval_seconds = 6.0
        app_config.adaptive_attempt_pacing_enabled = True
        app_config.attempt_parallel_enabled = True
        app_config.max_parallel_attempt_workers = 2
        orchestrator = TestOrchestrator(config=app_config, progress_emitter=progress_emitter)

        with patch("src.orchestrator.ConversationRunner") as MockRunner, patch(
            "src.orchestrator.time.monotonic", side_effect=range(2000, 4000, 10)
        ):
            mock_runner_instance = MockRunner.return_value
            mock_runner_instance.run_attempt = AsyncMock(
                side_effect=[make_attempt_result(i, True) for i in range(1, 41)]
            )
            report = await orchestrator.run_suite(suite)

        assert report.adaptive_attempt_pacing_enabled is True
        assert report.adaptive_attempt_pacing_base_interval_seconds == pytest.approx(6.0)
        assert report.adaptive_attempt_pacing_final_interval_seconds == pytest.approx(5.5)
        assert report.adaptive_attempt_pacing_adjustment_count == 1
        assert len(report.adaptive_attempt_pacing_adjustments) == 1
        adjustment = report.adaptive_attempt_pacing_adjustments[0]
        assert adjustment.reason == "pressure_window_low"
        assert adjustment.signal_count == 0
        assert adjustment.signal_rate == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_run_suite_scenario_result_fields(self, app_config, progress_emitter, simple_suite):
        """Test that scenario results have correct fields."""
        orchestrator = TestOrchestrator(config=app_config, progress_emitter=progress_emitter)

        with patch("src.orchestrator.ConversationRunner") as MockRunner:
            mock_runner_instance = MockRunner.return_value
            mock_runner_instance.run_attempt = AsyncMock(
                side_effect=[
                    make_attempt_result(1, True),
                    make_attempt_result(2, True),
                ]
            )
            report = await orchestrator.run_suite(simple_suite)

        result = report.scenario_results[0]
        assert result.scenario_name == "Scenario A"
        assert result.attempts == 2
        assert result.successes == 2
        assert result.failures == 0
        assert result.success_rate == 1.0
        assert result.is_regression is False
        assert len(result.attempt_results) == 2

    @pytest.mark.asyncio
    async def test_run_suite_applies_default_attempts(self, app_config, progress_emitter):
        """Test that default_attempts from config is used when scenario.attempts is None."""
        suite = TestSuite(
            name="Default Attempts Suite",
            scenarios=[
                TestScenario(
                    name="No Attempts Specified",
                    persona="Customer",
                    goal="Do something",
                    attempts=None,
                )
            ],
        )
        orchestrator = TestOrchestrator(config=app_config, progress_emitter=progress_emitter)

        with patch("src.orchestrator.ConversationRunner") as MockRunner:
            mock_runner_instance = MockRunner.return_value
            mock_runner_instance.run_attempt = AsyncMock(
                return_value=make_attempt_result(1, True)
            )
            report = await orchestrator.run_suite(suite)

        # default_attempts is 3 in our fixture
        assert report.scenario_results[0].attempts == 3
        assert mock_runner_instance.run_attempt.call_count == 3

    @pytest.mark.asyncio
    async def test_run_suite_multiple_scenarios(self, app_config, progress_emitter, multi_scenario_suite):
        """Test that run_suite handles multiple scenarios correctly."""
        orchestrator = TestOrchestrator(config=app_config, progress_emitter=progress_emitter)

        with patch("src.orchestrator.ConversationRunner") as MockRunner:
            mock_runner_instance = MockRunner.return_value
            mock_runner_instance.run_attempt = AsyncMock(
                return_value=make_attempt_result(1, True)
            )
            report = await orchestrator.run_suite(multi_scenario_suite)

        assert len(report.scenario_results) == 2
        assert report.scenario_results[0].scenario_name == "Scenario A"
        assert report.scenario_results[0].attempts == 2
        assert report.scenario_results[1].scenario_name == "Scenario B"
        assert report.scenario_results[1].attempts == 3
        assert report.overall_attempts == 5
        assert report.overall_successes == 5

    @pytest.mark.asyncio
    async def test_run_suite_creates_runner_with_correct_config(self, app_config, progress_emitter, simple_suite):
        """Test that the orchestrator creates ConversationRunner with correct config."""
        orchestrator = TestOrchestrator(config=app_config, progress_emitter=progress_emitter)
        mock_judge_instance = MagicMock()

        with patch("src.orchestrator.ConversationRunner") as MockRunner, patch(
            "src.orchestrator.build_judge_execution_client",
            return_value=mock_judge_instance,
        ):
            mock_runner_instance = MockRunner.return_value
            mock_runner_instance.run_attempt = AsyncMock(
                return_value=make_attempt_result(1, True)
            )
            await orchestrator.run_suite(simple_suite)

            assert MockRunner.call_count == 2
            call_kwargs = MockRunner.call_args[1]
            assert call_kwargs["judge"] is mock_judge_instance
            assert call_kwargs["web_msg_config"]["region"] == "us-east-1"
            assert call_kwargs["web_msg_config"]["deployment_id"] == "deploy-123"
            assert call_kwargs["web_msg_config"]["timeout"] == 30
            assert call_kwargs["web_msg_config"]["step_skip_timeout_seconds"] == 90
            assert call_kwargs["web_msg_config"]["knowledge_mode_timeout_seconds"] == 120
            assert call_kwargs["web_msg_config"]["language"] == "en"
            assert call_kwargs["max_turns"] == 10


class TestDetermineRegressions:
    """Tests for determine_regressions method."""

    def test_no_regressions(self, app_config, progress_emitter):
        """Test that no regressions are returned when all scenarios pass."""
        orchestrator = TestOrchestrator(config=app_config, progress_emitter=progress_emitter)
        report = TestReport(
            suite_name="Test",
            timestamp=datetime.now(),
            duration_seconds=10.0,
            scenario_results=[
                ScenarioResult(
                    scenario_name="Good Scenario",
                    attempts=5,
                    successes=5,
                    failures=0,
                    success_rate=1.0,
                    is_regression=False,
                    attempt_results=[],
                )
            ],
            overall_attempts=5,
            overall_successes=5,
            overall_failures=0,
            overall_success_rate=1.0,
            has_regressions=False,
            regression_threshold=0.8,
        )

        regressions = orchestrator.determine_regressions(report, threshold=0.8)
        assert regressions == []

    def test_with_regressions(self, app_config, progress_emitter):
        """Test that scenarios below threshold are flagged."""
        orchestrator = TestOrchestrator(config=app_config, progress_emitter=progress_emitter)
        report = TestReport(
            suite_name="Test",
            timestamp=datetime.now(),
            duration_seconds=10.0,
            scenario_results=[
                ScenarioResult(
                    scenario_name="Good Scenario",
                    attempts=5,
                    successes=5,
                    failures=0,
                    success_rate=1.0,
                    is_regression=False,
                    attempt_results=[],
                ),
                ScenarioResult(
                    scenario_name="Bad Scenario",
                    attempts=5,
                    successes=2,
                    failures=3,
                    success_rate=0.4,
                    is_regression=True,
                    attempt_results=[],
                ),
            ],
            overall_attempts=10,
            overall_successes=7,
            overall_failures=3,
            overall_success_rate=0.7,
            has_regressions=True,
            regression_threshold=0.8,
        )

        regressions = orchestrator.determine_regressions(report, threshold=0.8)
        assert regressions == ["Bad Scenario"]

    def test_at_threshold_not_regression(self, app_config, progress_emitter):
        """Test that a scenario exactly at threshold is NOT a regression."""
        orchestrator = TestOrchestrator(config=app_config, progress_emitter=progress_emitter)
        report = TestReport(
            suite_name="Test",
            timestamp=datetime.now(),
            duration_seconds=10.0,
            scenario_results=[
                ScenarioResult(
                    scenario_name="Borderline",
                    attempts=5,
                    successes=4,
                    failures=1,
                    success_rate=0.8,
                    is_regression=False,
                    attempt_results=[],
                )
            ],
            overall_attempts=5,
            overall_successes=4,
            overall_failures=1,
            overall_success_rate=0.8,
            has_regressions=False,
            regression_threshold=0.8,
        )

        regressions = orchestrator.determine_regressions(report, threshold=0.8)
        assert regressions == []

    def test_just_below_threshold_is_regression(self, app_config, progress_emitter):
        """Test that a scenario just below threshold IS a regression."""
        orchestrator = TestOrchestrator(config=app_config, progress_emitter=progress_emitter)
        report = TestReport(
            suite_name="Test",
            timestamp=datetime.now(),
            duration_seconds=10.0,
            scenario_results=[
                ScenarioResult(
                    scenario_name="Almost Good",
                    attempts=10,
                    successes=7,
                    failures=3,
                    success_rate=0.7,
                    is_regression=True,
                    attempt_results=[],
                )
            ],
            overall_attempts=10,
            overall_successes=7,
            overall_failures=3,
            overall_success_rate=0.7,
            has_regressions=True,
            regression_threshold=0.8,
        )

        regressions = orchestrator.determine_regressions(report, threshold=0.8)
        assert regressions == ["Almost Good"]

    def test_custom_threshold(self, app_config, progress_emitter):
        """Test determine_regressions with a custom threshold."""
        orchestrator = TestOrchestrator(config=app_config, progress_emitter=progress_emitter)
        report = TestReport(
            suite_name="Test",
            timestamp=datetime.now(),
            duration_seconds=10.0,
            scenario_results=[
                ScenarioResult(
                    scenario_name="Scenario X",
                    attempts=10,
                    successes=5,
                    failures=5,
                    success_rate=0.5,
                    is_regression=True,
                    attempt_results=[],
                )
            ],
            overall_attempts=10,
            overall_successes=5,
            overall_failures=5,
            overall_success_rate=0.5,
            has_regressions=True,
            regression_threshold=0.5,
        )

        # With threshold 0.5, success_rate 0.5 is NOT a regression (not strictly below)
        regressions = orchestrator.determine_regressions(report, threshold=0.5)
        assert regressions == []

        # With threshold 0.6, success_rate 0.5 IS a regression
        regressions = orchestrator.determine_regressions(report, threshold=0.6)
        assert regressions == ["Scenario X"]


@pytest.mark.asyncio
async def test_run_suite_parallel_preserves_ordering(
    app_config,
    progress_emitter,
    multi_scenario_suite,
):
    """Parallel mode should keep report ordering deterministic."""
    app_config.attempt_parallel_enabled = True
    app_config.max_parallel_attempt_workers = 3
    orchestrator = TestOrchestrator(config=app_config, progress_emitter=progress_emitter)

    async def fake_attempt_run(scenario, attempt_num, status_callback=None):
        # Force variable completion ordering across scenarios.
        if scenario.name == "Scenario A":
            await asyncio.sleep(0.01)
        else:
            await asyncio.sleep(0.0)
        return make_attempt_result(attempt_num, True)

    with patch("src.orchestrator.ConversationRunner") as MockRunner:
        MockRunner.return_value.run_attempt = AsyncMock(side_effect=fake_attempt_run)
        report = await orchestrator.run_suite(multi_scenario_suite)

    assert [scenario.scenario_name for scenario in report.scenario_results] == [
        "Scenario A",
        "Scenario B",
    ]
    assert [attempt.attempt_number for attempt in report.scenario_results[0].attempt_results] == [1, 2]
    assert [attempt.attempt_number for attempt in report.scenario_results[1].attempt_results] == [1, 2, 3]
    assert MockRunner.call_count == 5


@pytest.mark.asyncio
async def test_run_suite_forces_serial_workers_for_knowledge_judging(
    app_config,
    progress_emitter,
):
    """Knowledge-evaluation suites should run serially even when parallel mode is enabled."""
    suite = TestSuite(
        name="Knowledge Serial Suite",
        scenarios=[
            TestScenario(
                name="Knowledge Scenario",
                persona="Traveler",
                goal="Get baggage policy answer",
                expected_intent="knowledge",
                attempts=3,
            )
        ],
    )
    app_config.attempt_parallel_enabled = True
    app_config.max_parallel_attempt_workers = 3
    app_config.min_attempt_interval_seconds = 0
    orchestrator = TestOrchestrator(config=app_config, progress_emitter=progress_emitter)
    progress_queue = progress_emitter.subscribe()

    active_runs = 0
    max_active_runs = 0

    async def fake_attempt_run(scenario, attempt_num, status_callback=None):
        nonlocal active_runs, max_active_runs
        active_runs += 1
        max_active_runs = max(max_active_runs, active_runs)
        await asyncio.sleep(0.01)
        active_runs -= 1
        return make_attempt_result(attempt_num, True)

    with patch("src.orchestrator.ConversationRunner") as MockRunner:
        MockRunner.return_value.run_attempt = AsyncMock(side_effect=fake_attempt_run)
        report = await orchestrator.run_suite(suite)

    status_messages = []
    while not progress_queue.empty():
        event = progress_queue.get_nowait()
        if event.event_type == ProgressEventType.ATTEMPT_STATUS:
            status_messages.append(event.message)

    assert report.overall_attempts == 3
    assert max_active_runs == 1
    assert any(
        "forcing serial execution (1 worker)" in message
        for message in status_messages
    )
