"""Unit tests for the Test Orchestrator."""

import asyncio
import threading
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.models import (
    AppConfig,
    AttemptResult,
    Message,
    MessageRole,
    ProgressEvent,
    ProgressEventType,
    ScenarioResult,
    TestReport,
    TestScenario,
    TestSuite,
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

    @pytest.mark.asyncio
    async def test_run_suite_emits_judge_warmup_status(self, app_config, progress_emitter, simple_suite):
        """Test that run_suite emits warm-up status events before attempts."""
        app_config.judge_warmup_enabled = True
        orchestrator = TestOrchestrator(config=app_config, progress_emitter=progress_emitter)
        q = progress_emitter.subscribe()

        with patch("src.orchestrator.ConversationRunner") as MockRunner, patch(
            "src.orchestrator.JudgeLLMClient"
        ) as MockJudge:
            mock_judge_instance = MockJudge.return_value
            mock_judge_instance.warm_up = MagicMock(return_value="OK")
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
        assert attempt_events[0].planned_attempts == 2
        assert attempt_events[0].completed_attempts == 1
        assert attempt_events[1].attempt_number == 2
        assert attempt_events[1].success is False
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
    async def test_run_suite_throttles_attempt_start_rate(
        self, app_config, progress_emitter, simple_suite
    ):
        """Attempts should be rate-limited by min_attempt_interval_seconds."""
        app_config.min_attempt_interval_seconds = 60
        orchestrator = TestOrchestrator(config=app_config, progress_emitter=progress_emitter)

        with patch("src.orchestrator.ConversationRunner") as MockRunner, patch(
            "src.orchestrator.asyncio.sleep", new_callable=AsyncMock
        ) as mock_sleep, patch(
            "src.orchestrator.time.monotonic", side_effect=[0.0, 10.0, 60.0]
        ):
            mock_runner_instance = MockRunner.return_value
            mock_runner_instance.run_attempt = AsyncMock(
                side_effect=[
                    make_attempt_result(1, True),
                    make_attempt_result(2, True),
                ]
            )
            await orchestrator.run_suite(simple_suite)

        mock_sleep.assert_awaited_once_with(50.0)

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

        with patch("src.orchestrator.ConversationRunner") as MockRunner:
            mock_runner_instance = MockRunner.return_value
            mock_runner_instance.run_attempt = AsyncMock(
                return_value=make_attempt_result(1, True)
            )
            with patch("src.orchestrator.JudgeLLMClient") as MockJudge:
                await orchestrator.run_suite(simple_suite)

                MockJudge.assert_called_once_with(
                    base_url="http://localhost:11434",
                    model="llama3",
                    timeout=30,
                )
                MockRunner.assert_called_once()
                call_kwargs = MockRunner.call_args[1]
                assert call_kwargs["web_msg_config"]["region"] == "us-east-1"
                assert call_kwargs["web_msg_config"]["deployment_id"] == "deploy-123"
                assert call_kwargs["web_msg_config"]["timeout"] == 30
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
