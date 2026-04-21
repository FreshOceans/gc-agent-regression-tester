"""Unit tests for ConversationRunner using mocked JudgeLLMClient and WebMessagingClient."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.conversation_runner import ConversationRunner, GreetingGateTimeoutError
from src.judge_llm import JudgeLLMError
from src.models import (
    AttemptResult,
    ContinueDecision,
    GoalEvaluation,
    JourneyValidationConfig,
    JourneyValidationResult,
    Message,
    MessageRole,
    TestScenario,
)
from src.web_messaging_client import WebMessagingError


@pytest.fixture
def scenario():
    return TestScenario(
        name="Test Booking",
        persona="A busy professional",
        goal="Book a meeting for next Tuesday at 2pm",
        attempts=3,
    )


@pytest.fixture
def web_msg_config():
    return {
        "region": "mypurecloud.com",
        "deployment_id": "test-deployment-123",
        "timeout": 30,
    }


@pytest.fixture
def mock_judge():
    judge = MagicMock()
    return judge


@pytest.fixture
def runner(mock_judge, web_msg_config):
    return ConversationRunner(judge=mock_judge, web_msg_config=web_msg_config, max_turns=20)


class TestConversationRunnerInit:
    def test_stores_judge_client(self, mock_judge, web_msg_config):
        runner = ConversationRunner(judge=mock_judge, web_msg_config=web_msg_config)
        assert runner.judge is mock_judge

    def test_stores_web_msg_config(self, mock_judge, web_msg_config):
        runner = ConversationRunner(judge=mock_judge, web_msg_config=web_msg_config)
        assert runner.web_msg_config == web_msg_config

    def test_default_max_turns(self, mock_judge, web_msg_config):
        runner = ConversationRunner(judge=mock_judge, web_msg_config=web_msg_config)
        assert runner.max_turns == 20

    def test_custom_max_turns(self, mock_judge, web_msg_config):
        runner = ConversationRunner(judge=mock_judge, web_msg_config=web_msg_config, max_turns=10)
        assert runner.max_turns == 10


class TestRunAttemptSuccess:
    @pytest.mark.asyncio
    async def test_successful_conversation(self, runner, mock_judge, scenario):
        """Test a successful conversation where the goal is achieved."""
        # Judge says continue once, then stop
        mock_judge.should_continue.side_effect = [
            ContinueDecision(should_continue=True, goal_achieved=None),
            ContinueDecision(should_continue=False, goal_achieved=True),
        ]
        mock_judge.generate_user_message.return_value = "I'd like to book a meeting"
        mock_judge.evaluate_goal.return_value = GoalEvaluation(
            success=True, explanation="Meeting was booked successfully"
        )

        with patch("src.conversation_runner.WebMessagingClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.connect = AsyncMock()
            mock_client.wait_for_welcome = AsyncMock(return_value="Hello! How can I help?")
            mock_client.send_message = AsyncMock()
            mock_client.receive_response = AsyncMock(return_value="Sure, I can book that for you.")
            mock_client.disconnect = AsyncMock()
            MockClient.return_value = mock_client

            result = await runner.run_attempt(scenario, attempt_number=1)

        assert result.success is True
        assert result.attempt_number == 1
        assert result.explanation == "Meeting was booked successfully"
        assert result.error is None
        assert len(result.conversation) == 3  # welcome + user + agent
        assert result.conversation[0].role == MessageRole.AGENT
        assert result.conversation[0].content == "Hello! How can I help?"
        assert result.conversation[1].role == MessageRole.USER
        assert result.conversation[2].role == MessageRole.AGENT

    @pytest.mark.asyncio
    async def test_creates_new_client_per_attempt(self, runner, mock_judge, scenario):
        """Test that a new WebMessagingClient is created for each attempt (test isolation)."""
        mock_judge.generate_user_message.return_value = "Hello"
        mock_judge.evaluate_goal.return_value = GoalEvaluation(
            success=True, explanation="Done"
        )

        with patch("src.conversation_runner.WebMessagingClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.connect = AsyncMock()
            mock_client.wait_for_welcome = AsyncMock(return_value="Welcome!")
            mock_client.send_message = AsyncMock()
            mock_client.receive_response = AsyncMock(return_value="Agent reply")
            mock_client.disconnect = AsyncMock()
            MockClient.return_value = mock_client

            await runner.run_attempt(scenario, attempt_number=1)
            await runner.run_attempt(scenario, attempt_number=2)

        assert MockClient.call_count == 2

    @pytest.mark.asyncio
    async def test_disconnect_called_on_success(self, runner, mock_judge, scenario):
        """Test that disconnect is always called even on success."""
        mock_judge.generate_user_message.return_value = "Hello"
        mock_judge.evaluate_goal.return_value = GoalEvaluation(
            success=True, explanation="Done"
        )

        with patch("src.conversation_runner.WebMessagingClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.connect = AsyncMock()
            mock_client.wait_for_welcome = AsyncMock(return_value="Hi")
            mock_client.send_message = AsyncMock()
            mock_client.receive_response = AsyncMock(return_value="Agent reply")
            mock_client.disconnect = AsyncMock()
            MockClient.return_value = mock_client

            await runner.run_attempt(scenario, attempt_number=1)

        mock_client.disconnect.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_language_selection_pre_step_runs_before_main_intent_turn(
        self, mock_judge, web_msg_config
    ):
        scenario = TestScenario(
            name="Flight Cancel ES",
            persona="Traveler",
            goal="Cancel booking",
            first_message="Quiero cancelar mi reserva",
            expected_intent="flight_cancel",
            language_selection_message="espanol",
            attempts=1,
        )
        localized_config = {
            **web_msg_config,
            "expected_greeting": "Hi, I'm Ava, WestJet's virtual assistant. How may I help you today?",
        }
        runner = ConversationRunner(
            judge=mock_judge,
            web_msg_config=localized_config,
            max_turns=1,
        )

        with patch("src.conversation_runner.WebMessagingClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.connect = AsyncMock()
            mock_client.wait_for_welcome = AsyncMock(
                return_value="Welcome to WestJet support."
            )
            mock_client.send_join = AsyncMock()
            mock_client.send_message = AsyncMock()
            mock_client.receive_response = AsyncMock(
                side_effect=[
                    "Perfecto, continuemos en espanol.",
                    "Hi, I'm Ava, WestJet's virtual assistant. How may I help you today?",
                    "detected_intent: flight_cancel",
                ]
            )
            mock_client.disconnect = AsyncMock()
            MockClient.return_value = mock_client

            result = await runner.run_attempt(scenario, attempt_number=1)

        assert result.success is True
        sent_messages = [call.args[0] for call in mock_client.send_message.await_args_list]
        assert sent_messages == ["espanol", "Quiero cancelar mi reserva"]
        main_user_index = next(
            idx
            for idx, msg in enumerate(result.conversation)
            if msg.role == MessageRole.USER and msg.content == "Quiero cancelar mi reserva"
        )
        greeting_index = next(
            idx
            for idx, msg in enumerate(result.conversation)
            if msg.role == MessageRole.AGENT
            and "Hi, I'm Ava, WestJet's virtual assistant. How may I help you today?" in msg.content
        )
        assert greeting_index < main_user_index
        assert any(
            "language selection message" in str(step.get("message", "")).lower()
            for step in result.step_log
            if isinstance(step, dict)
        )

    @pytest.mark.asyncio
    async def test_language_selection_pre_step_does_not_consume_turn_budget(
        self, mock_judge, web_msg_config
    ):
        scenario = TestScenario(
            name="One Turn Max",
            persona="Traveler",
            goal="Get answer",
            first_message="I need help",
            language_selection_message="francais",
            attempts=1,
        )
        localized_config = {
            **web_msg_config,
            "expected_greeting": "Hi, I'm Ava, WestJet's virtual assistant. How may I help you today?",
        }
        runner = ConversationRunner(
            judge=mock_judge,
            web_msg_config=localized_config,
            max_turns=1,
        )
        mock_judge.evaluate_goal.return_value = GoalEvaluation(
            success=False,
            explanation="Need more turns",
        )

        with patch("src.conversation_runner.WebMessagingClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.connect = AsyncMock()
            mock_client.wait_for_welcome = AsyncMock(
                return_value="Hi, I'm Ava, WestJet's virtual assistant. How may I help you today?"
            )
            mock_client.send_join = AsyncMock()
            mock_client.send_message = AsyncMock()
            mock_client.receive_response = AsyncMock(
                side_effect=[
                    "Tres bien, poursuivons en francais.",
                    "How can I help today?",
                ]
            )
            mock_client.disconnect = AsyncMock()
            MockClient.return_value = mock_client

            result = await runner.run_attempt(scenario, attempt_number=1)

        assert result.success is False
        user_messages = [msg.content for msg in result.conversation if msg.role == MessageRole.USER]
        assert user_messages == ["francais", "I need help"]
        assert len([msg for msg in result.conversation if msg.role == MessageRole.AGENT]) >= 3

    @pytest.mark.asyncio
    async def test_language_selection_pre_step_times_out_when_greeting_never_arrives(
        self, mock_judge, web_msg_config
    ):
        scenario = TestScenario(
            name="Flight Cancel ES Timeout",
            persona="Traveler",
            goal="Cancel booking",
            first_message="Quiero cancelar mi reserva",
            expected_intent="flight_cancel",
            language_selection_message="espanol",
            attempts=1,
        )
        localized_config = {
            **web_msg_config,
            "expected_greeting": "Hi, I'm Ava, WestJet's virtual assistant. How may I help you today?",
        }
        runner = ConversationRunner(
            judge=mock_judge,
            web_msg_config=localized_config,
            max_turns=1,
        )

        with patch("src.conversation_runner.WebMessagingClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.connect = AsyncMock()
            mock_client.wait_for_welcome = AsyncMock(
                return_value="Welcome to WestJet support."
            )
            mock_client.send_join = AsyncMock()
            mock_client.send_message = AsyncMock()
            mock_client.receive_response = AsyncMock(
                side_effect=[
                    "Perfecto, continuemos en espanol.",
                    TimeoutError("Timed out waiting for greeting"),
                ]
            )
            mock_client.disconnect = AsyncMock()
            MockClient.return_value = mock_client

            result = await runner.run_attempt(scenario, attempt_number=1)

        assert result.success is False
        assert result.timed_out is True
        assert "waiting for expected greeting" in result.explanation.lower()
        assert result.timeout_diagnostics is not None
        assert result.timeout_diagnostics.timeout_class == "greeting_gate"
        assert result.timeout_diagnostics.language_pre_step_active is True
        assert result.timeout_diagnostics.greeting_wait_timeout_seconds == pytest.approx(13.0)
        sent_messages = [call.args[0] for call in mock_client.send_message.await_args_list]
        assert sent_messages == ["espanol"]
        assert all(
            not (msg.role == MessageRole.USER and msg.content == "Quiero cancelar mi reserva")
            for msg in result.conversation
        )

    @pytest.mark.asyncio
    async def test_language_selection_terminal_error_before_greeting_fails_fast(
        self, mock_judge, web_msg_config
    ):
        scenario = TestScenario(
            name="Flight Cancel EN Terminal Error",
            persona="Traveler",
            goal="Cancel booking",
            first_message="I want to cancel my booking",
            expected_intent="flight_cancel",
            language_selection_message="english",
            attempts=1,
        )
        localized_config = {
            **web_msg_config,
            "expected_greeting": "Hi, I'm Ava, WestJet's virtual assistant. How may I help you today?",
        }
        runner = ConversationRunner(
            judge=mock_judge,
            web_msg_config=localized_config,
            max_turns=1,
        )

        with patch("src.conversation_runner.WebMessagingClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.connect = AsyncMock()
            mock_client.wait_for_welcome = AsyncMock(
                return_value="What is your language preference?"
            )
            mock_client.send_join = AsyncMock()
            mock_client.send_message = AsyncMock()
            mock_client.receive_response = AsyncMock(
                side_effect=[
                    "Sorry, an error occurred. One moment, please, while I put you through to someone who can help.",
                ]
            )
            mock_client.disconnect = AsyncMock()
            MockClient.return_value = mock_client

            result = await runner.run_attempt(scenario, attempt_number=1)

        assert result.success is False
        assert result.timed_out is False
        assert result.skipped is False
        assert result.error is not None
        assert "upstream_agent_error_before_greeting" in result.error
        assert result.failure_diagnostics is not None
        assert result.failure_diagnostics.failure_class == "upstream_agent_error_before_greeting"
        assert result.failure_diagnostics.gate_step == (
            "Waiting for expected greeting before sending first user message"
        )
        assert result.failure_diagnostics.matched_pattern_id == "en_error_handoff"
        assert result.timeout_diagnostics is None
        sent_messages = [call.args[0] for call in mock_client.send_message.await_args_list]
        assert sent_messages == ["english"]
        assert all(
            not (msg.role == MessageRole.USER and msg.content == "I want to cancel my booking")
            for msg in result.conversation
        )

    @pytest.mark.asyncio
    async def test_localized_french_greeting_variant_passes_heuristic(
        self, mock_judge, web_msg_config
    ):
        scenario = TestScenario(
            name="Flight Cancel FR-CA",
            persona="Voyageur",
            goal="Annuler une reservation",
            first_message="Je veux annuler ma reservation",
            expected_intent="flight_cancel",
            language_selection_message="francais",
            attempts=1,
        )
        localized_config = {
            **web_msg_config,
            "language": "fr-CA",
            "expected_greeting": "Hi, I'm Ava, WestJet's virtual assistant. How may I help you today?",
        }
        runner = ConversationRunner(
            judge=mock_judge,
            web_msg_config=localized_config,
            max_turns=1,
        )

        with patch("src.conversation_runner.WebMessagingClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.connect = AsyncMock()
            mock_client.wait_for_welcome = AsyncMock(
                return_value="What is your language preference?"
            )
            mock_client.send_join = AsyncMock()
            mock_client.send_message = AsyncMock()
            mock_client.receive_response = AsyncMock(
                side_effect=[
                    "fr-ca",
                    "Bonjour, je suis Ava, l'assistant virtuel de WestJet. En quoi puis-je vous aider aujourd'hui ?",
                    "detected_intent: flight_cancel",
                ]
            )
            mock_client.disconnect = AsyncMock()
            MockClient.return_value = mock_client

            result = await runner.run_attempt(scenario, attempt_number=1)

        assert result.success is True
        sent_messages = [call.args[0] for call in mock_client.send_message.await_args_list]
        assert sent_messages == ["francais", "Je veux annuler ma reservation"]
        greeting_index = next(
            idx
            for idx, msg in enumerate(result.conversation)
            if msg.role == MessageRole.AGENT and "En quoi puis-je vous aider" in msg.content
        )
        main_user_index = next(
            idx
            for idx, msg in enumerate(result.conversation)
            if msg.role == MessageRole.USER and msg.content == "Je veux annuler ma reservation"
        )
        assert greeting_index < main_user_index

    @pytest.mark.asyncio
    async def test_greeting_gate_applies_buffer_after_language_pre_step(
        self, mock_judge, web_msg_config
    ):
        localized_config = {
            **web_msg_config,
            "expected_greeting": "Hi, I'm Ava, WestJet's virtual assistant. How may I help you today?",
            "greeting_wait_timeout_seconds": 8,
            "localized_greeting_wait_buffer_seconds": 5,
            "timeout": 30,
        }
        runner = ConversationRunner(
            judge=mock_judge,
            web_msg_config=localized_config,
            max_turns=1,
        )
        runner._await_step = AsyncMock(side_effect=TimeoutError("Timed out waiting for greeting"))
        client = AsyncMock()
        client.receive_response = AsyncMock(return_value="unused")
        conversation = [
            Message(
                role=MessageRole.AGENT,
                content="What is your language preference?",
                timestamp=runner._now_utc(),
            )
        ]

        with patch("src.conversation_runner.time.monotonic", return_value=100.0):
            with pytest.raises(GreetingGateTimeoutError):
                await runner._ensure_expected_greeting_before_main_utterance(
                    client=client,
                    conversation=conversation,
                    language_pre_step_active=True,
                )

        timeout_override = runner._await_step.await_args.kwargs["timeout_override_seconds"]
        assert timeout_override == pytest.approx(13.0)

    @pytest.mark.asyncio
    async def test_greeting_gate_uses_base_wait_without_language_pre_step(
        self, mock_judge, web_msg_config
    ):
        localized_config = {
            **web_msg_config,
            "expected_greeting": "Hi, I'm Ava, WestJet's virtual assistant. How may I help you today?",
            "greeting_wait_timeout_seconds": 8,
            "localized_greeting_wait_buffer_seconds": 5,
            "timeout": 30,
        }
        runner = ConversationRunner(
            judge=mock_judge,
            web_msg_config=localized_config,
            max_turns=1,
        )
        runner._await_step = AsyncMock(side_effect=TimeoutError("Timed out waiting for greeting"))
        client = AsyncMock()
        client.receive_response = AsyncMock(return_value="unused")
        conversation = [
            Message(
                role=MessageRole.AGENT,
                content="What is your language preference?",
                timestamp=runner._now_utc(),
            )
        ]

        with patch("src.conversation_runner.time.monotonic", return_value=100.0):
            with pytest.raises(GreetingGateTimeoutError):
                await runner._ensure_expected_greeting_before_main_utterance(
                    client=client,
                    conversation=conversation,
                    language_pre_step_active=False,
                )

        timeout_override = runner._await_step.await_args.kwargs["timeout_override_seconds"]
        assert timeout_override == pytest.approx(8.0)

    @pytest.mark.asyncio
    async def test_split_conversation_and_evaluation_results_languages(
        self, mock_judge, web_msg_config
    ):
        scenario = TestScenario(
            name="Localized Evaluation",
            persona="Traveler",
            goal="Get flight status",
            attempts=1,
        )
        localized_config = {
            **web_msg_config,
            "language": "fr-CA",
            "evaluation_results_language": "es",
        }
        runner = ConversationRunner(judge=mock_judge, web_msg_config=localized_config, max_turns=1)
        mock_judge.generate_user_message.return_value = "Bonjour, je veux le statut de mon vol."
        mock_judge.evaluate_goal.return_value = GoalEvaluation(
            success=True,
            explanation="Objetivo cumplido",
        )

        with patch("src.conversation_runner.WebMessagingClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.connect = AsyncMock()
            mock_client.wait_for_welcome = AsyncMock(
                return_value="Hi, I'm Ava, WestJet's virtual assistant. How may I help you today?"
            )
            mock_client.send_join = AsyncMock()
            mock_client.send_message = AsyncMock()
            mock_client.receive_response = AsyncMock(return_value="Voici le statut de votre vol.")
            mock_client.disconnect = AsyncMock()
            MockClient.return_value = mock_client

            result = await runner.run_attempt(scenario, attempt_number=1)

        assert result.success is True
        assert mock_judge.generate_user_message.call_args.kwargs["language_code"] == "fr-CA"
        assert mock_judge.evaluate_goal.call_args.kwargs["language_code"] == "es"


class TestRunAttemptMaxTurns:
    @pytest.mark.asyncio
    async def test_enforces_max_turns(self, mock_judge, web_msg_config, scenario):
        """Test that conversation stops at max_turns even if judge says continue."""
        runner = ConversationRunner(judge=mock_judge, web_msg_config=web_msg_config, max_turns=3)

        # Judge always says continue
        mock_judge.should_continue.return_value = ContinueDecision(
            should_continue=True, goal_achieved=None
        )
        mock_judge.generate_user_message.return_value = "Next message"
        mock_judge.evaluate_goal.return_value = GoalEvaluation(
            success=False, explanation="Max turns reached without achieving goal"
        )

        with patch("src.conversation_runner.WebMessagingClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.connect = AsyncMock()
            mock_client.wait_for_welcome = AsyncMock(return_value="Welcome!")
            mock_client.send_message = AsyncMock()
            mock_client.receive_response = AsyncMock(return_value="Agent reply")
            mock_client.disconnect = AsyncMock()
            MockClient.return_value = mock_client

            result = await runner.run_attempt(scenario, attempt_number=1)

        # 1 welcome + 3 user + 3 agent = 7 messages
        assert len(result.conversation) == 7
        # Count user-agent pairs (turns)
        user_messages = [m for m in result.conversation if m.role == MessageRole.USER]
        assert len(user_messages) == 3


class TestRunAttemptErrors:
    @pytest.mark.asyncio
    async def test_timeout_error_on_welcome(self, runner, mock_judge, scenario):
        """Test that TimeoutError during welcome is handled gracefully."""
        with patch("src.conversation_runner.WebMessagingClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.connect = AsyncMock()
            mock_client.wait_for_welcome = AsyncMock(
                side_effect=TimeoutError("Timed out waiting for welcome message after 30s")
            )
            mock_client.disconnect = AsyncMock()
            MockClient.return_value = mock_client

            result = await runner.run_attempt(scenario, attempt_number=1)

        assert result.success is False
        assert result.attempt_number == 1
        assert "timeout" in result.explanation.lower()
        assert result.error is not None
        mock_client.disconnect.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_timeout_error_on_response(self, runner, mock_judge, scenario):
        """Test that TimeoutError during agent response is handled gracefully."""
        mock_judge.should_continue.return_value = ContinueDecision(
            should_continue=True, goal_achieved=None
        )
        mock_judge.generate_user_message.return_value = "Hello"

        with patch("src.conversation_runner.WebMessagingClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.connect = AsyncMock()
            mock_client.wait_for_welcome = AsyncMock(return_value="Welcome!")
            mock_client.send_message = AsyncMock()
            mock_client.receive_response = AsyncMock(
                side_effect=TimeoutError("Timed out waiting for agent response after 30s")
            )
            mock_client.disconnect = AsyncMock()
            MockClient.return_value = mock_client

            result = await runner.run_attempt(scenario, attempt_number=1)

        assert result.success is False
        assert "timeout" in result.explanation.lower()
        assert result.error is not None
        assert result.timeout_diagnostics is not None
        assert result.timeout_diagnostics.timeout_class == "response_timeout"
        assert result.timeout_diagnostics.step_name == "Waiting for agent response"
        mock_client.disconnect.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_web_messaging_error_on_connect(self, runner, mock_judge, scenario):
        """Test that WebMessagingError during connect is handled gracefully."""
        with patch("src.conversation_runner.WebMessagingClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.connect = AsyncMock(
                side_effect=WebMessagingError(
                    "Failed to connect: deployment_id=test-deployment-123, region=mypurecloud.com"
                )
            )
            mock_client.disconnect = AsyncMock()
            MockClient.return_value = mock_client

            result = await runner.run_attempt(scenario, attempt_number=1)

        assert result.success is False
        assert "web messaging" in result.explanation.lower()
        assert "deployment_id=test-deployment-123" in result.error
        assert "region=mypurecloud.com" in result.error
        mock_client.disconnect.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_judge_llm_error(self, runner, mock_judge, scenario):
        """Test that JudgeLLMError is handled gracefully."""
        mock_judge.generate_user_message.side_effect = JudgeLLMError(
            "Failed to parse ContinueDecision from LLM response"
        )

        with patch("src.conversation_runner.WebMessagingClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.connect = AsyncMock()
            mock_client.wait_for_welcome = AsyncMock(return_value="Welcome!")
            mock_client.disconnect = AsyncMock()
            MockClient.return_value = mock_client

            result = await runner.run_attempt(scenario, attempt_number=1)

        assert result.success is False
        assert "judge llm" in result.explanation.lower()
        assert result.error is not None
        mock_client.disconnect.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_judge_llm_timeout_error_marked_as_timeout(self, runner, mock_judge, scenario):
        """Judge timeout errors should be counted as timed out attempts."""
        mock_judge.generate_user_message.side_effect = JudgeLLMError(
            "Failed to call Ollama chat API: Read timed out."
        )

        with patch("src.conversation_runner.WebMessagingClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.connect = AsyncMock()
            mock_client.send_join = AsyncMock()
            mock_client.wait_for_welcome = AsyncMock(return_value="Welcome!")
            mock_client.disconnect = AsyncMock()
            MockClient.return_value = mock_client

            result = await runner.run_attempt(scenario, attempt_number=1)

        assert result.success is False
        assert result.timed_out is True
        assert result.skipped is False
        assert "timeout" in result.explanation.lower()
        assert result.timeout_diagnostics is not None
        assert result.timeout_diagnostics.timeout_class == "judge_timeout"
        mock_client.disconnect.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_step_log_includes_judge_evaluation_statuses(self, mock_judge, web_msg_config):
        """Step log should show Judge LLM evaluation phases for debugging."""
        runner = ConversationRunner(judge=mock_judge, web_msg_config=web_msg_config, max_turns=1)
        scenario = TestScenario(
            name="knowledge - Smoke 01",
            persona="Traveler",
            goal="Answer pet policy question correctly",
            first_message="what are the rules for pets to fly",
            expected_intent="knowledge",
            attempts=1,
        )
        mock_judge.evaluate_goal.side_effect = [
            JudgeLLMError("Mid-turn eval timed out"),
            JudgeLLMError("Final eval timed out"),
        ]

        with patch("src.conversation_runner.WebMessagingClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.connect = AsyncMock()
            mock_client.send_join = AsyncMock()
            mock_client.wait_for_welcome = AsyncMock(
                return_value="Hi, I'm Ava, WestJet's virtual assistant. How may I help you today?"
            )
            mock_client.send_message = AsyncMock()
            mock_client.receive_response = AsyncMock(
                return_value=(
                    "Pet travel has specific restrictions by destination. "
                    "Would you like to know how to book this online?"
                )
            )
            mock_client.disconnect = AsyncMock()
            MockClient.return_value = mock_client

            result = await runner.run_attempt(scenario, attempt_number=1)

        assert result.success is False
        assert result.timed_out is True
        assert "timeout" in result.explanation.lower()
        step_messages = [entry["message"] for entry in result.step_log]
        assert "Evaluating goal with Judge LLM (mid-conversation)" in step_messages
        assert "Running final goal evaluation with Judge LLM" in step_messages
        assert any(m.startswith("Judge LLM error:") for m in step_messages)
        mock_client.disconnect.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_disconnect_called_on_error(self, runner, mock_judge, scenario):
        """Test that disconnect is always called in the finally block."""
        with patch("src.conversation_runner.WebMessagingClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.connect = AsyncMock(side_effect=WebMessagingError("Connection failed"))
            mock_client.disconnect = AsyncMock()
            MockClient.return_value = mock_client

            await runner.run_attempt(scenario, attempt_number=1)

        mock_client.disconnect.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_step_timeout_skips_attempt(self, mock_judge, scenario):
        """When a single step runs too long, attempt should be marked skipped."""
        web_msg_config = {
            "region": "mypurecloud.com",
            "deployment_id": "test-deployment-123",
            "timeout": 90,
            "step_skip_timeout_seconds": 1,
        }
        runner = ConversationRunner(judge=mock_judge, web_msg_config=web_msg_config, max_turns=20)

        def slow_generate(*args, **kwargs):
            time.sleep(1.2)
            return "hello"

        mock_judge.generate_user_message.side_effect = slow_generate

        with patch("src.conversation_runner.WebMessagingClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.connect = AsyncMock()
            mock_client.send_join = AsyncMock()
            mock_client.wait_for_welcome = AsyncMock(return_value="Welcome!")
            mock_client.disconnect = AsyncMock()
            MockClient.return_value = mock_client

            result = await runner.run_attempt(scenario, attempt_number=1)

        assert result.success is False
        assert result.skipped is True
        assert result.timed_out is False
        assert "Attempt skipped because a step exceeded the time limit" in result.explanation
        assert "Generating user message with Judge LLM" in result.error
        assert result.timeout_diagnostics is not None
        assert result.timeout_diagnostics.timeout_class == "step_timeout"
        assert result.timeout_diagnostics.step_name == "Generating user message with Judge LLM"
        mock_client.disconnect.assert_awaited_once()


class TestRunAttemptConversationHistory:
    @pytest.mark.asyncio
    async def test_welcome_message_added_to_history(self, runner, mock_judge, scenario):
        """Test that the welcome message is added as an AGENT message."""
        mock_judge.generate_user_message.return_value = "Hello"
        mock_judge.evaluate_goal.return_value = GoalEvaluation(
            success=True, explanation="Done"
        )

        with patch("src.conversation_runner.WebMessagingClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.connect = AsyncMock()
            mock_client.wait_for_welcome = AsyncMock(return_value="Welcome to support!")
            mock_client.send_message = AsyncMock()
            mock_client.receive_response = AsyncMock(return_value="Agent reply")
            mock_client.disconnect = AsyncMock()
            MockClient.return_value = mock_client

            result = await runner.run_attempt(scenario, attempt_number=1)

        assert result.conversation[0].role == MessageRole.AGENT
        assert result.conversation[0].content == "Welcome to support!"
        assert result.conversation[0].timestamp is not None

    @pytest.mark.asyncio
    async def test_multi_turn_conversation_history(self, runner, mock_judge, scenario):
        """Test that full conversation history is built correctly over multiple turns."""
        mock_judge.evaluate_goal.side_effect = [
            GoalEvaluation(success=False, explanation="Keep going"),
            GoalEvaluation(success=True, explanation="Goal achieved"),
        ]
        mock_judge.generate_user_message.side_effect = ["First msg", "Second msg"]

        with patch("src.conversation_runner.WebMessagingClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.connect = AsyncMock()
            mock_client.wait_for_welcome = AsyncMock(return_value="Hello!")
            mock_client.send_message = AsyncMock()
            mock_client.receive_response = AsyncMock(
                side_effect=["Reply 1", "Reply 2"]
            )
            mock_client.disconnect = AsyncMock()
            MockClient.return_value = mock_client

            result = await runner.run_attempt(scenario, attempt_number=1)

        expected_pairs = [
            (MessageRole.AGENT, "Hello!"),
            (MessageRole.USER, "First msg"),
            (MessageRole.AGENT, "Reply 1"),
            (MessageRole.USER, "Second msg"),
            (MessageRole.AGENT, "Reply 2"),
        ]
        actual_pairs = [(m.role, m.content) for m in result.conversation]
        assert actual_pairs == expected_pairs
        assert all(m.timestamp is not None for m in result.conversation)

    @pytest.mark.asyncio
    async def test_judge_receives_full_history(self, runner, mock_judge, scenario):
        """Test that the judge LLM receives the full conversation history."""
        mock_judge.should_continue.side_effect = [
            ContinueDecision(should_continue=True),
            ContinueDecision(should_continue=False, goal_achieved=True),
        ]
        mock_judge.generate_user_message.return_value = "User message"
        mock_judge.evaluate_goal.return_value = GoalEvaluation(
            success=True, explanation="Done"
        )

        with patch("src.conversation_runner.WebMessagingClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.connect = AsyncMock()
            mock_client.wait_for_welcome = AsyncMock(return_value="Welcome!")
            mock_client.send_message = AsyncMock()
            mock_client.receive_response = AsyncMock(return_value="Agent reply")
            mock_client.disconnect = AsyncMock()
            MockClient.return_value = mock_client

            await runner.run_attempt(scenario, attempt_number=1)

        # Verify evaluate_goal was called with full history
        eval_call = mock_judge.evaluate_goal.call_args
        history = eval_call[1]["conversation_history"] if eval_call[1] else eval_call[0][2]
        assert len(history) == 3  # welcome + user + agent

    @pytest.mark.asyncio
    async def test_waits_for_expected_greeting_before_first_user(self, mock_judge, scenario):
        """Runner should not send first user message until expected greeting appears."""
        web_msg_config = {
            "region": "mypurecloud.com",
            "deployment_id": "test-deployment-123",
            "timeout": 30,
            "expected_greeting": "Hi, I'm Ava, WestJet's virtual assistant. How may I help you today?",
        }
        runner = ConversationRunner(judge=mock_judge, web_msg_config=web_msg_config, max_turns=20)
        mock_judge.evaluate_goal.return_value = GoalEvaluation(
            success=True, explanation="Done"
        )
        mock_judge.generate_user_message.return_value = "I want to cancel my booking"

        with patch("src.conversation_runner.WebMessagingClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.connect = AsyncMock()
            mock_client.send_join = AsyncMock()
            mock_client.wait_for_welcome = AsyncMock(
                return_value="Presence events are not supported in this configuration"
            )
            mock_client.receive_response = AsyncMock(
                side_effect=[
                    "Hi, I'm Ava, WestJet's virtual assistant. How may I help you today?",
                    "Sure, I can help with that.",
                ]
            )
            mock_client.send_message = AsyncMock()
            mock_client.disconnect = AsyncMock()
            MockClient.return_value = mock_client

            result = await runner.run_attempt(scenario, attempt_number=1)

        assert result.success is True
        # A bootstrap user message may be sent first to elicit greeting when
        # presence events are unsupported. The scenario utterance must still
        # occur only after greeting appears.
        assert result.conversation[0].role == MessageRole.AGENT
        assert result.conversation[1].role == MessageRole.USER
        assert result.conversation[1].content == "Hi"
        assert result.conversation[2].role == MessageRole.AGENT
        assert "Hi, I'm Ava, WestJet's virtual assistant. How may I help you today?" in result.conversation[2].content
        assert result.conversation[3].role == MessageRole.USER
        assert result.conversation[3].content == "I want to cancel my booking"

    @pytest.mark.asyncio
    async def test_attempt_includes_turn_timing(self, runner, mock_judge, scenario):
        """Attempt result should include timing metadata."""
        mock_judge.generate_user_message.return_value = "Hello"
        mock_judge.evaluate_goal.return_value = GoalEvaluation(
            success=True, explanation="Done"
        )

        with patch("src.conversation_runner.WebMessagingClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.connect = AsyncMock()
            mock_client.send_join = AsyncMock()
            mock_client.wait_for_welcome = AsyncMock(return_value="Hello! How can I help?")
            mock_client.send_message = AsyncMock()
            mock_client.receive_response = AsyncMock(return_value="Agent reply")
            mock_client.disconnect = AsyncMock()
            MockClient.return_value = mock_client

            result = await runner.run_attempt(scenario, attempt_number=1)

        assert result.started_at is not None
        assert result.completed_at is not None
        assert result.duration_seconds is not None
        assert result.duration_seconds >= 0
        assert len(result.turn_durations_seconds) == 1
        assert result.turn_durations_seconds[0] >= 0


class TestExpectedIntentMode:
    @pytest.mark.asyncio
    async def test_knowledge_expected_intent_uses_goal_evaluation(
        self, mock_judge, web_msg_config
    ):
        scenario = TestScenario(
            name="knowledge - Smoke 01",
            persona="Traveler",
            goal="Answer pet policy question correctly",
            first_message="what are the rules for pets to fly",
            expected_intent="knowledge",
            attempts=1,
        )
        mock_judge.evaluate_goal.return_value = GoalEvaluation(
            success=True,
            explanation="Agent answered the knowledge question accurately.",
        )
        runner = ConversationRunner(judge=mock_judge, web_msg_config=web_msg_config, max_turns=20)

        with patch("src.conversation_runner.WebMessagingClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.connect = AsyncMock()
            mock_client.send_join = AsyncMock()
            mock_client.wait_for_welcome = AsyncMock(
                return_value="Hi, I'm Ava, WestJet's virtual assistant. How may I help you today?"
            )
            mock_client.send_message = AsyncMock()
            mock_client.receive_response = AsyncMock(
                return_value=(
                    "Pet travel has specific restrictions by destination. "
                    "Would you like to know how to book this online?"
                )
            )
            mock_client.disconnect = AsyncMock()
            MockClient.return_value = mock_client

            result = await runner.run_attempt(scenario, attempt_number=1)

        assert result.success is True
        assert result.detected_intent is None
        assert result.explanation == "Agent answered the knowledge question accurately."
        mock_judge.evaluate_goal.assert_called_once()

    @pytest.mark.asyncio
    async def test_knowledge_mode_uses_effective_timeout_overrides(
        self, mock_judge, web_msg_config
    ):
        scenario = TestScenario(
            name="knowledge - Timeout Override",
            persona="Traveler",
            goal="Answer a policy question",
            first_message="What are the pet rules?",
            expected_intent="knowledge",
            attempts=1,
        )
        knowledge_cfg = dict(web_msg_config)
        knowledge_cfg["timeout"] = 90
        knowledge_cfg["step_skip_timeout_seconds"] = 90
        knowledge_cfg["knowledge_mode_timeout_seconds"] = 120
        runner = ConversationRunner(
            judge=mock_judge,
            web_msg_config=knowledge_cfg,
            max_turns=20,
        )
        mock_judge.evaluate_goal.return_value = GoalEvaluation(
            success=True,
            explanation="Answered correctly.",
        )
        status_messages: list[str] = []

        with patch("src.conversation_runner.WebMessagingClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.connect = AsyncMock()
            mock_client.send_join = AsyncMock()
            mock_client.wait_for_welcome = AsyncMock(
                return_value="Hi, I'm Ava, WestJet's virtual assistant. How may I help you today?"
            )
            mock_client.send_message = AsyncMock()
            mock_client.receive_response = AsyncMock(
                return_value="Pets are allowed depending on route."
            )
            mock_client.disconnect = AsyncMock()
            MockClient.return_value = mock_client

            await runner.run_attempt(
                scenario,
                attempt_number=1,
                status_callback=status_messages.append,
            )

        assert MockClient.call_args.kwargs["timeout"] == 120
        assert any(
            "Knowledge-mode timeout override active:" in message
            for message in status_messages
        )

    @pytest.mark.asyncio
    async def test_non_knowledge_mode_keeps_base_timeout(
        self, mock_judge, web_msg_config
    ):
        scenario = TestScenario(
            name="intent - flight_cancel",
            persona="Traveler",
            goal="Classify cancellation intent",
            first_message="I want to cancel my booking",
            expected_intent="flight_cancel",
            attempts=1,
        )
        non_knowledge_cfg = dict(web_msg_config)
        non_knowledge_cfg["timeout"] = 90
        non_knowledge_cfg["step_skip_timeout_seconds"] = 90
        non_knowledge_cfg["knowledge_mode_timeout_seconds"] = 120
        runner = ConversationRunner(
            judge=mock_judge,
            web_msg_config=non_knowledge_cfg,
            max_turns=20,
        )
        status_messages: list[str] = []

        with patch("src.conversation_runner.WebMessagingClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.connect = AsyncMock()
            mock_client.send_join = AsyncMock()
            mock_client.wait_for_welcome = AsyncMock(
                return_value="Hi, I'm Ava, WestJet's virtual assistant. How may I help you today?"
            )
            mock_client.send_message = AsyncMock()
            mock_client.receive_response = AsyncMock(return_value="INTENT=flight_cancel")
            mock_client.disconnect = AsyncMock()
            MockClient.return_value = mock_client

            await runner.run_attempt(
                scenario,
                attempt_number=1,
                status_callback=status_messages.append,
            )

        assert MockClient.call_args.kwargs["timeout"] == 90
        assert not any(
            "Knowledge-mode timeout override active:" in message
            for message in status_messages
        )

    @pytest.mark.asyncio
    async def test_knowledge_timeout_diagnostics_use_effective_windows(
        self, mock_judge, web_msg_config
    ):
        scenario = TestScenario(
            name="knowledge - timeout diagnostics",
            persona="Traveler",
            goal="Get policy answer",
            first_message="Can pets fly?",
            expected_intent="knowledge",
            attempts=1,
        )
        knowledge_cfg = dict(web_msg_config)
        knowledge_cfg["timeout"] = 90
        knowledge_cfg["step_skip_timeout_seconds"] = 90
        knowledge_cfg["knowledge_mode_timeout_seconds"] = 120
        runner = ConversationRunner(
            judge=mock_judge,
            web_msg_config=knowledge_cfg,
            max_turns=20,
        )

        with patch("src.conversation_runner.WebMessagingClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.connect = AsyncMock()
            mock_client.send_join = AsyncMock()
            mock_client.wait_for_welcome = AsyncMock(
                return_value="Hi, I'm Ava, WestJet's virtual assistant. How may I help you today?"
            )
            mock_client.send_message = AsyncMock()
            mock_client.receive_response = AsyncMock(
                side_effect=TimeoutError("Timed out waiting for agent response after 120s")
            )
            mock_client.disconnect = AsyncMock()
            MockClient.return_value = mock_client

            result = await runner.run_attempt(scenario, attempt_number=1)

        assert result.success is False
        assert result.timed_out is True
        assert result.timeout_diagnostics is not None
        assert result.timeout_diagnostics.timeout_class == "response_timeout"
        assert result.timeout_diagnostics.configured_timeout_seconds == pytest.approx(120.0)
        assert result.timeout_diagnostics.step_skip_timeout_seconds == pytest.approx(120.0)

    @pytest.mark.asyncio
    async def test_expected_intent_success(self, mock_judge, web_msg_config):
        scenario = TestScenario(
            name="Intent Classification",
            persona="Traveler",
            goal="Classify the request",
            first_message="I want to cancel my booking",
            expected_intent="flight_cancel",
            attempts=1,
        )
        runner = ConversationRunner(judge=mock_judge, web_msg_config=web_msg_config, max_turns=20)

        with patch("src.conversation_runner.WebMessagingClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.connect = AsyncMock()
            mock_client.send_join = AsyncMock()
            mock_client.wait_for_welcome = AsyncMock(
                return_value="Hi, I'm Ava, WestJet's virtual assistant. How may I help you today?"
            )
            mock_client.send_message = AsyncMock()
            mock_client.receive_response = AsyncMock(return_value="INTENT=flight_cancel")
            mock_client.disconnect = AsyncMock()
            MockClient.return_value = mock_client

            result = await runner.run_attempt(scenario, attempt_number=1)

        assert result.success is True
        assert result.detected_intent == "flight_cancel"
        assert "Intent matched expected value" in result.explanation
        mock_judge.evaluate_goal.assert_not_called()

    @pytest.mark.asyncio
    async def test_expected_intent_mismatch(self, mock_judge, web_msg_config):
        scenario = TestScenario(
            name="Intent Classification",
            persona="Traveler",
            goal="Classify the request",
            first_message="I want to cancel my booking",
            expected_intent="flight_cancel",
            attempts=1,
        )
        runner = ConversationRunner(judge=mock_judge, web_msg_config=web_msg_config, max_turns=20)

        with patch("src.conversation_runner.WebMessagingClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.connect = AsyncMock()
            mock_client.send_join = AsyncMock()
            mock_client.wait_for_welcome = AsyncMock(
                return_value="Hi, I'm Ava, WestJet's virtual assistant. How may I help you today?"
            )
            mock_client.send_message = AsyncMock()
            mock_client.receive_response = AsyncMock(return_value='{"intent":"flight_change"}')
            mock_client.disconnect = AsyncMock()
            MockClient.return_value = mock_client

            result = await runner.run_attempt(scenario, attempt_number=1)

        assert result.success is False
        assert result.detected_intent == "flight_change"
        assert "expected 'flight_cancel' but got 'flight_change'" in result.explanation
        mock_judge.evaluate_goal.assert_not_called()

    @pytest.mark.asyncio
    async def test_expected_intent_not_found(self, mock_judge, web_msg_config):
        scenario = TestScenario(
            name="Intent Classification",
            persona="Traveler",
            goal="Classify the request",
            first_message="I want to cancel my booking",
            expected_intent="flight_cancel",
            attempts=1,
        )
        runner = ConversationRunner(judge=mock_judge, web_msg_config=web_msg_config, max_turns=20)

        with patch("src.conversation_runner.WebMessagingClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.connect = AsyncMock()
            mock_client.send_join = AsyncMock()
            mock_client.wait_for_welcome = AsyncMock(
                return_value="Hi, I'm Ava, WestJet's virtual assistant. How may I help you today?"
            )
            mock_client.send_message = AsyncMock()
            mock_client.receive_response = AsyncMock(return_value="Thanks, working on it.")
            mock_client.disconnect = AsyncMock()
            MockClient.return_value = mock_client

            result = await runner.run_attempt(scenario, attempt_number=1)

        assert result.success is False
        assert result.detected_intent is None
        assert "Expected intent 'flight_cancel' was not found" in result.explanation
        mock_judge.evaluate_goal.assert_not_called()

    @pytest.mark.asyncio
    async def test_expected_intent_no_bot_response(self, mock_judge, web_msg_config):
        scenario = TestScenario(
            name="Intent Classification",
            persona="Traveler",
            goal="Classify the request",
            first_message="I want to cancel my booking",
            expected_intent="flight_cancel",
            attempts=1,
        )
        runner = ConversationRunner(judge=mock_judge, web_msg_config=web_msg_config, max_turns=20)

        with patch("src.conversation_runner.WebMessagingClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.connect = AsyncMock()
            mock_client.send_join = AsyncMock()
            mock_client.wait_for_welcome = AsyncMock(
                return_value="Hi, I'm Ava, WestJet's virtual assistant. How may I help you today?"
            )
            mock_client.send_message = AsyncMock()
            mock_client.receive_response = AsyncMock(
                side_effect=TimeoutError("Timed out waiting for agent response after 30s")
            )
            mock_client.disconnect = AsyncMock()
            MockClient.return_value = mock_client

            result = await runner.run_attempt(scenario, attempt_number=1)

        assert result.success is False
        assert result.timed_out is False
        assert "did not return a test-mode intent response" in result.explanation
        mock_judge.evaluate_goal.assert_not_called()

    @pytest.mark.asyncio
    async def test_expected_intent_timeout_uses_conversations_api_fallback(
        self, mock_judge, web_msg_config
    ):
        scenario = TestScenario(
            name="Intent Classification",
            persona="Traveler",
            goal="Classify the request",
            first_message="I want to cancel my booking",
            expected_intent="flight_cancel",
            attempts=1,
        )
        config_with_fallback = dict(
            web_msg_config,
            gc_client_id="client-id",
            gc_client_secret="client-secret",
            intent_attribute_name="detected_intent",
        )
        runner = ConversationRunner(
            judge=mock_judge,
            web_msg_config=config_with_fallback,
            max_turns=20,
        )

        with (
            patch("src.conversation_runner.WebMessagingClient") as MockClient,
            patch("src.conversation_runner.GenesysConversationsClient") as MockConversationsClient,
        ):
            mock_client = AsyncMock()
            mock_client.connect = AsyncMock()
            mock_client.send_join = AsyncMock()
            mock_client.wait_for_welcome = AsyncMock(
                return_value="Hi, I'm Ava, WestJet's virtual assistant. How may I help you today?"
            )
            mock_client.send_message = AsyncMock()
            mock_client.receive_response = AsyncMock(
                side_effect=TimeoutError("Timed out waiting for agent response after 30s")
            )
            mock_client.disconnect = AsyncMock()
            mock_client.conversation_id = "11111111-2222-4333-8444-555555555555"
            mock_client.participant_id = "participant-456"
            MockClient.return_value = mock_client

            mock_conversations_client = MagicMock()
            mock_conversations_client.get_participant_attribute.return_value = "flight_cancel"
            MockConversationsClient.return_value = mock_conversations_client

            result = await runner.run_attempt(scenario, attempt_number=1)

        assert result.success is True
        assert result.timed_out is False
        assert result.detected_intent == "flight_cancel"
        assert "Conversations API participant attribute" in result.explanation
        mock_judge.evaluate_goal.assert_not_called()
        mock_conversations_client.get_participant_attribute.assert_called_once()

    @pytest.mark.asyncio
    async def test_expected_intent_non_intent_text_uses_conversations_api_fallback(
        self, mock_judge, web_msg_config
    ):
        scenario = TestScenario(
            name="Intent Classification",
            persona="Traveler",
            goal="Classify the request",
            first_message="I want to cancel my booking",
            expected_intent="flight_cancel",
            attempts=1,
        )
        config_with_fallback = dict(
            web_msg_config,
            gc_client_id="client-id",
            gc_client_secret="client-secret",
            intent_attribute_name="detected_intent",
        )
        runner = ConversationRunner(
            judge=mock_judge,
            web_msg_config=config_with_fallback,
            max_turns=20,
        )

        with (
            patch("src.conversation_runner.WebMessagingClient") as MockClient,
            patch("src.conversation_runner.GenesysConversationsClient") as MockConversationsClient,
        ):
            mock_client = AsyncMock()
            mock_client.connect = AsyncMock()
            mock_client.send_join = AsyncMock()
            mock_client.wait_for_welcome = AsyncMock(
                return_value="Hi, I'm Ava, WestJet's virtual assistant. How may I help you today?"
            )
            mock_client.send_message = AsyncMock()
            mock_client.receive_response = AsyncMock(return_value="Thanks, working on it.")
            mock_client.disconnect = AsyncMock()
            mock_client.conversation_id = "11111111-2222-4333-8444-555555555555"
            mock_client.participant_id = "66666666-7777-4888-9999-aaaaaaaaaaaa"
            mock_client.get_conversation_id_candidates = MagicMock(return_value=[])
            MockClient.return_value = mock_client

            mock_conversations_client = MagicMock()
            mock_conversations_client.get_participant_attribute.return_value = "flight_cancel"
            MockConversationsClient.return_value = mock_conversations_client

            result = await runner.run_attempt(scenario, attempt_number=1)

        assert result.success is True
        assert result.detected_intent == "flight_cancel"
        assert "Conversations API participant attribute" in result.explanation
        mock_judge.evaluate_goal.assert_not_called()
        mock_conversations_client.get_participant_attribute.assert_called_once_with(
            conversation_id="11111111-2222-4333-8444-555555555555",
            attribute_name="detected_intent",
            participant_id="66666666-7777-4888-9999-aaaaaaaaaaaa",
            retries=3,
            retry_delay_seconds=1.0,
        )

    @pytest.mark.asyncio
    async def test_expected_intent_uses_explicit_transcript_ids_for_fallback(
        self, mock_judge, web_msg_config
    ):
        scenario = TestScenario(
            name="Intent Classification",
            persona="Traveler",
            goal="Classify the request",
            first_message="I want to cancel my booking",
            expected_intent="flight_cancel",
            attempts=1,
        )
        config_with_fallback = dict(
            web_msg_config,
            gc_client_id="client-id",
            gc_client_secret="client-secret",
            intent_attribute_name="detected_intent",
        )
        runner = ConversationRunner(
            judge=mock_judge,
            web_msg_config=config_with_fallback,
            max_turns=20,
        )

        with (
            patch("src.conversation_runner.WebMessagingClient") as MockClient,
            patch("src.conversation_runner.GenesysConversationsClient") as MockConversationsClient,
        ):
            mock_client = AsyncMock()
            mock_client.connect = AsyncMock()
            mock_client.send_join = AsyncMock()
            mock_client.wait_for_welcome = AsyncMock(
                return_value="Hi, I'm Ava, WestJet's virtual assistant. How may I help you today?"
            )
            mock_client.send_message = AsyncMock()
            mock_client.receive_response = AsyncMock(
                return_value=(
                    '"conversation_id":"11111111-2222-4333-8444-555555555555"\n'
                    '"participant_id":"66666666-7777-4888-9999-aaaaaaaaaaaa"'
                )
            )
            mock_client.disconnect = AsyncMock()
            mock_client.conversation_id = None
            mock_client.participant_id = None
            mock_client.get_conversation_id_candidates = MagicMock(return_value=[])
            MockClient.return_value = mock_client

            mock_conversations_client = MagicMock()
            mock_conversations_client.get_participant_attribute.return_value = "flight_cancel"
            MockConversationsClient.return_value = mock_conversations_client

            result = await runner.run_attempt(scenario, attempt_number=1)

        assert result.success is True
        assert result.detected_intent == "flight_cancel"
        mock_conversations_client.get_participant_attribute.assert_called_once_with(
            conversation_id="11111111-2222-4333-8444-555555555555",
            attribute_name="detected_intent",
            participant_id="66666666-7777-4888-9999-aaaaaaaaaaaa",
            retries=3,
            retry_delay_seconds=1.0,
        )

    @pytest.mark.asyncio
    async def test_expected_intent_timeout_uses_explicit_transcript_ids_for_fallback(
        self, mock_judge, web_msg_config
    ):
        scenario = TestScenario(
            name="Intent Classification",
            persona="Traveler",
            goal="Classify the request",
            first_message="I want to cancel my booking",
            expected_intent="flight_cancel",
            attempts=1,
        )
        config_with_fallback = dict(
            web_msg_config,
            gc_client_id="client-id",
            gc_client_secret="client-secret",
            intent_attribute_name="detected_intent",
        )
        runner = ConversationRunner(
            judge=mock_judge,
            web_msg_config=config_with_fallback,
            max_turns=20,
        )

        with (
            patch("src.conversation_runner.WebMessagingClient") as MockClient,
            patch("src.conversation_runner.GenesysConversationsClient") as MockConversationsClient,
        ):
            mock_client = AsyncMock()
            mock_client.connect = AsyncMock()
            mock_client.send_join = AsyncMock()
            mock_client.wait_for_welcome = AsyncMock(
                return_value=(
                    "Hi, I'm Ava, WestJet's virtual assistant. How may I help you today?\n"
                    "conversation_id: 11111111-2222-4333-8444-555555555555\n"
                    "participant_id: 66666666-7777-4888-9999-aaaaaaaaaaaa"
                )
            )
            mock_client.send_message = AsyncMock()
            mock_client.receive_response = AsyncMock(
                side_effect=TimeoutError("Timed out waiting for agent response after 30s")
            )
            mock_client.disconnect = AsyncMock()
            mock_client.conversation_id = None
            mock_client.participant_id = None
            mock_client.get_conversation_id_candidates = MagicMock(return_value=[])
            MockClient.return_value = mock_client

            mock_conversations_client = MagicMock()
            mock_conversations_client.get_participant_attribute.return_value = "flight_cancel"
            MockConversationsClient.return_value = mock_conversations_client

            result = await runner.run_attempt(scenario, attempt_number=1)

        assert result.success is True
        assert result.detected_intent == "flight_cancel"
        assert "Conversations API participant attribute" in result.explanation
        mock_conversations_client.get_participant_attribute.assert_called_once_with(
            conversation_id="11111111-2222-4333-8444-555555555555",
            attribute_name="detected_intent",
            participant_id="66666666-7777-4888-9999-aaaaaaaaaaaa",
            retries=3,
            retry_delay_seconds=1.0,
        )

    @pytest.mark.asyncio
    async def test_expected_intent_reads_follow_up_agent_message_before_ending(
        self, mock_judge, web_msg_config
    ):
        scenario = TestScenario(
            name="Intent Classification",
            persona="Traveler",
            goal="Classify the request",
            first_message="I want to cancel my booking",
            expected_intent="flight_cancel",
            attempts=1,
        )
        runner = ConversationRunner(judge=mock_judge, web_msg_config=web_msg_config, max_turns=20)

        with (
            patch("src.conversation_runner.WebMessagingClient") as MockClient,
            patch("src.conversation_runner.GenesysConversationsClient") as MockConversationsClient,
        ):
            mock_client = AsyncMock()
            mock_client.connect = AsyncMock()
            mock_client.send_join = AsyncMock()
            mock_client.wait_for_welcome = AsyncMock(
                return_value="Hi, I'm Ava, WestJet's virtual assistant. How may I help you today?"
            )
            mock_client.send_message = AsyncMock()
            mock_client.receive_response = AsyncMock(
                side_effect=[
                    "conversation_id: 11111111-2222-4333-8444-555555555555",
                    "detected_intent: flight_cancel",
                    TimeoutError("No more follow-up messages"),
                ]
            )
            mock_client.disconnect = AsyncMock()
            mock_client.conversation_id = None
            mock_client.participant_id = None
            mock_client.get_conversation_id_candidates = MagicMock(return_value=[])
            MockClient.return_value = mock_client

            mock_conversations_client = MagicMock()
            MockConversationsClient.return_value = mock_conversations_client

            result = await runner.run_attempt(scenario, attempt_number=1)

        assert result.success is True
        assert result.detected_intent == "flight_cancel"
        assert "Intent matched expected value 'flight_cancel'." in result.explanation
        # Should not need API fallback when a follow-up agent message includes detected_intent.
        mock_conversations_client.get_participant_attribute.assert_not_called()

    @pytest.mark.asyncio
    async def test_expected_intent_sends_knowledge_closure_message_when_needed(
        self, mock_judge, web_msg_config
    ):
        scenario = TestScenario(
            name="Knowledge Intent Classification",
            persona="Traveler",
            goal="Classify the request",
            first_message="I want to cancel my booking",
            expected_intent="flight_cancel",
            attempts=1,
        )
        runner = ConversationRunner(judge=mock_judge, web_msg_config=web_msg_config, max_turns=20)

        with (
            patch("src.conversation_runner.WebMessagingClient") as MockClient,
            patch("src.conversation_runner.GenesysConversationsClient") as MockConversationsClient,
        ):
            mock_client = AsyncMock()
            mock_client.connect = AsyncMock()
            mock_client.send_join = AsyncMock()
            mock_client.wait_for_welcome = AsyncMock(
                return_value="Hi, I'm Ava, WestJet's virtual assistant. How may I help you today?"
            )
            mock_client.send_message = AsyncMock()
            mock_client.receive_response = AsyncMock(
                side_effect=[
                    "conversation_id: 11111111-2222-4333-8444-555555555555",
                    TimeoutError("no immediate follow-up"),
                    "detected_intent: flight_cancel",
                    TimeoutError("no more messages"),
                ]
            )
            mock_client.disconnect = AsyncMock()
            mock_client.conversation_id = None
            mock_client.participant_id = None
            mock_client.get_conversation_id_candidates = MagicMock(return_value=[])
            MockClient.return_value = mock_client

            mock_conversations_client = MagicMock()
            MockConversationsClient.return_value = mock_conversations_client

            result = await runner.run_attempt(scenario, attempt_number=1)

        assert result.success is True
        assert result.detected_intent == "flight_cancel"
        sent_messages = [call.args[0] for call in mock_client.send_message.call_args_list]
        assert sent_messages == [
            "I want to cancel my booking",
            "no, thank you that is all",
        ]
        mock_conversations_client.get_participant_attribute.assert_not_called()

    @pytest.mark.asyncio
    async def test_expected_intent_uses_localized_knowledge_closure_message(
        self, mock_judge, web_msg_config
    ):
        scenario = TestScenario(
            name="Knowledge Intent Classification",
            persona="Viajero",
            goal="Clasificar solicitud",
            first_message="Quiero cancelar mi reserva",
            expected_intent="flight_cancel",
            attempts=1,
        )
        localized_config = dict(web_msg_config)
        localized_config["language"] = "es"
        runner = ConversationRunner(
            judge=mock_judge,
            web_msg_config=localized_config,
            max_turns=20,
        )

        with (
            patch("src.conversation_runner.WebMessagingClient") as MockClient,
            patch("src.conversation_runner.GenesysConversationsClient") as MockConversationsClient,
        ):
            mock_client = AsyncMock()
            mock_client.connect = AsyncMock()
            mock_client.send_join = AsyncMock()
            mock_client.wait_for_welcome = AsyncMock(
                return_value="Hola, soy Ava, la asistente virtual de WestJet. Como puedo ayudarte hoy?"
            )
            mock_client.send_message = AsyncMock()
            mock_client.receive_response = AsyncMock(
                side_effect=[
                    "conversation_id: 11111111-2222-4333-8444-555555555555",
                    TimeoutError("no immediate follow-up"),
                    "detected_intent: flight_cancel",
                    TimeoutError("no more messages"),
                ]
            )
            mock_client.disconnect = AsyncMock()
            mock_client.conversation_id = None
            mock_client.participant_id = None
            mock_client.get_conversation_id_candidates = MagicMock(return_value=[])
            MockClient.return_value = mock_client

            mock_conversations_client = MagicMock()
            MockConversationsClient.return_value = mock_conversations_client

            result = await runner.run_attempt(scenario, attempt_number=1)

        assert result.success is True
        sent_messages = [call.args[0] for call in mock_client.send_message.call_args_list]
        assert sent_messages == [
            "Quiero cancelar mi reserva",
            "no, gracias, eso es todo",
        ]

    @pytest.mark.asyncio
    async def test_flight_priority_change_yes_maps_to_within_72_hours_intent(
        self, mock_judge, web_msg_config
    ):
        scenario = TestScenario(
            name="Flight Priority Change",
            persona="Traveler",
            goal="Classify the request",
            first_message="I need to change my booking",
            expected_intent="flight_priority_change",
            attempts=1,
        )
        runner = ConversationRunner(judge=mock_judge, web_msg_config=web_msg_config, max_turns=20)

        with (
            patch("src.conversation_runner.WebMessagingClient") as MockClient,
            patch("src.conversation_runner.random.choice", return_value="yes"),
        ):
            mock_client = AsyncMock()
            mock_client.connect = AsyncMock()
            mock_client.send_join = AsyncMock()
            mock_client.wait_for_welcome = AsyncMock(
                return_value="Hi, I'm Ava, WestJet's virtual assistant. How may I help you today?"
            )
            mock_client.send_message = AsyncMock()
            mock_client.receive_response = AsyncMock(
                side_effect=[
                    "Do you want to continue with the change?",
                    TimeoutError("no immediate follow-up"),
                    "detected_intent: flight_change_priority_within_72_hours",
                ]
            )
            mock_client.disconnect = AsyncMock()
            MockClient.return_value = mock_client

            result = await runner.run_attempt(scenario, attempt_number=1)

        assert result.success is True
        assert result.detected_intent == "flight_change_priority_within_72_hours"
        sent_messages = [call.args[0] for call in mock_client.send_message.call_args_list]
        assert sent_messages == ["I need to change my booking", "yes"]

    @pytest.mark.asyncio
    async def test_flight_priority_change_no_maps_to_later_than_72_hours_intent(
        self, mock_judge, web_msg_config
    ):
        scenario = TestScenario(
            name="Flight Priority Change",
            persona="Traveler",
            goal="Classify the request",
            first_message="I need to change my booking",
            expected_intent="flight_priority_change",
            attempts=1,
        )
        runner = ConversationRunner(judge=mock_judge, web_msg_config=web_msg_config, max_turns=20)

        with (
            patch("src.conversation_runner.WebMessagingClient") as MockClient,
            patch("src.conversation_runner.random.choice", return_value="no"),
        ):
            mock_client = AsyncMock()
            mock_client.connect = AsyncMock()
            mock_client.send_join = AsyncMock()
            mock_client.wait_for_welcome = AsyncMock(
                return_value="Hi, I'm Ava, WestJet's virtual assistant. How may I help you today?"
            )
            mock_client.send_message = AsyncMock()
            mock_client.receive_response = AsyncMock(
                side_effect=[
                    "Is your flight within 72 hours?",
                    TimeoutError("no immediate follow-up"),
                    "detected_intent: flight_change_later_than_72_hours",
                ]
            )
            mock_client.disconnect = AsyncMock()
            MockClient.return_value = mock_client

            result = await runner.run_attempt(scenario, attempt_number=1)

        assert result.success is True
        assert result.detected_intent == "flight_change_later_than_72_hours"
        sent_messages = [call.args[0] for call in mock_client.send_message.call_args_list]
        assert sent_messages == ["I need to change my booking", "no"]

    @pytest.mark.asyncio
    async def test_vacation_flight_and_hotel_uses_deterministic_follow_up_answer(
        self, mock_judge, web_msg_config
    ):
        scenario = TestScenario(
            name="Vacation Inquiry",
            persona="Traveler",
            goal="Classify the request",
            first_message="Can you help price out a vacation for me",
            expected_intent="vacation_flight_and_hotel",
            attempts=1,
        )
        runner = ConversationRunner(judge=mock_judge, web_msg_config=web_msg_config, max_turns=20)

        with patch("src.conversation_runner.WebMessagingClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.connect = AsyncMock()
            mock_client.send_join = AsyncMock()
            mock_client.wait_for_welcome = AsyncMock(
                return_value="Hi, I'm Ava, WestJet's virtual assistant. How may I help you today?"
            )
            mock_client.send_message = AsyncMock()
            mock_client.receive_response = AsyncMock(
                side_effect=[
                    "Are you looking for flight only or flight and hotel?",
                    TimeoutError("no immediate follow-up"),
                    "detected_intent: vacation_flight_and_hotel",
                ]
            )
            mock_client.disconnect = AsyncMock()
            MockClient.return_value = mock_client

            result = await runner.run_attempt(scenario, attempt_number=1)

        assert result.success is True
        assert result.detected_intent == "vacation_flight_and_hotel"
        sent_messages = [call.args[0] for call in mock_client.send_message.call_args_list]
        assert sent_messages == [
            "Can you help price out a vacation for me",
            "flight and hotel",
        ]

    @pytest.mark.asyncio
    async def test_vacation_inquiry_flight_only_uses_deterministic_follow_up_answer(
        self, mock_judge, web_msg_config
    ):
        scenario = TestScenario(
            name="Vacation Inquiry",
            persona="Traveler",
            goal="Classify the request",
            first_message="I need help with my vacation package",
            expected_intent="vacation_inquiry_flight_only",
            attempts=1,
        )
        runner = ConversationRunner(judge=mock_judge, web_msg_config=web_msg_config, max_turns=20)

        with patch("src.conversation_runner.WebMessagingClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.connect = AsyncMock()
            mock_client.send_join = AsyncMock()
            mock_client.wait_for_welcome = AsyncMock(
                return_value="Hi, I'm Ava, WestJet's virtual assistant. How may I help you today?"
            )
            mock_client.send_message = AsyncMock()
            mock_client.receive_response = AsyncMock(
                side_effect=[
                    "Are you looking for flight only or flight and hotel?",
                    TimeoutError("no immediate follow-up"),
                    "detected_intent: vacation_inquiry_flight_only",
                ]
            )
            mock_client.disconnect = AsyncMock()
            MockClient.return_value = mock_client

            result = await runner.run_attempt(scenario, attempt_number=1)

        assert result.success is True
        assert result.detected_intent == "vacation_inquiry_flight_only"
        sent_messages = [call.args[0] for call in mock_client.send_message.call_args_list]
        assert sent_messages == [
            "I need help with my vacation package",
            "flight only",
        ]

    @pytest.mark.asyncio
    async def test_vacation_inquiry_legacy_expected_intent_maps_from_follow_up_answer(
        self, mock_judge, web_msg_config
    ):
        scenario = TestScenario(
            name="Vacation Inquiry",
            persona="Traveler",
            goal="Classify the request",
            first_message="Can you help price out a vacation for me",
            expected_intent="vacation_inquiry",
            attempts=1,
        )
        runner = ConversationRunner(judge=mock_judge, web_msg_config=web_msg_config, max_turns=20)

        with (
            patch("src.conversation_runner.WebMessagingClient") as MockClient,
            patch("src.conversation_runner.random.choice", return_value="flight and hotel"),
        ):
            mock_client = AsyncMock()
            mock_client.connect = AsyncMock()
            mock_client.send_join = AsyncMock()
            mock_client.wait_for_welcome = AsyncMock(
                return_value="Hi, I'm Ava, WestJet's virtual assistant. How may I help you today?"
            )
            mock_client.send_message = AsyncMock()
            mock_client.receive_response = AsyncMock(
                side_effect=[
                    "Are you looking for flight only or flight and hotel?",
                    TimeoutError("no immediate follow-up"),
                    "detected_intent: vacation_flight_and_hotel",
                ]
            )
            mock_client.disconnect = AsyncMock()
            MockClient.return_value = mock_client

            result = await runner.run_attempt(scenario, attempt_number=1)

        assert result.success is True
        assert result.detected_intent == "vacation_flight_and_hotel"
        sent_messages = [call.args[0] for call in mock_client.send_message.call_args_list]
        assert sent_messages == [
            "Can you help price out a vacation for me",
            "flight and hotel",
        ]

    @pytest.mark.asyncio
    async def test_speak_to_agent_uses_default_follow_up_confirmation(
        self, mock_judge, web_msg_config
    ):
        scenario = TestScenario(
            name="Speak To Agent",
            persona="Traveler",
            goal="Escalate to a live agent",
            first_message="I need to speak with someone",
            expected_intent="speak_to_agent",
            attempts=1,
        )
        runner = ConversationRunner(judge=mock_judge, web_msg_config=web_msg_config, max_turns=20)

        with patch("src.conversation_runner.WebMessagingClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.connect = AsyncMock()
            mock_client.send_join = AsyncMock()
            mock_client.wait_for_welcome = AsyncMock(
                return_value="Hi, I'm Ava, WestJet's virtual assistant. How may I help you today?"
            )
            mock_client.send_message = AsyncMock()
            mock_client.receive_response = AsyncMock(
                side_effect=[
                    (
                        "I can help answer your questions or provide information, "
                        "but if you'd like to speak with a live agent, I can escalate "
                        "your request. Would you like me to do that?"
                    ),
                    TimeoutError("no immediate follow-up"),
                    "detected_intent: speak_to_agent",
                ]
            )
            mock_client.disconnect = AsyncMock()
            MockClient.return_value = mock_client

            result = await runner.run_attempt(scenario, attempt_number=1)

        assert result.success is True
        assert result.detected_intent == "speak_to_agent"
        sent_messages = [call.args[0] for call in mock_client.send_message.call_args_list]
        assert sent_messages == [
            "I need to speak with someone",
            "Yes, connect me to a live agent",
        ]

    @pytest.mark.asyncio
    async def test_speak_to_agent_uses_scenario_follow_up_override(
        self, mock_judge, web_msg_config
    ):
        scenario = TestScenario(
            name="Speak To Agent",
            persona="Traveler",
            goal="Escalate to a live agent",
            first_message="Can I talk to support?",
            expected_intent="speak_to_agent",
            intent_follow_up_user_message="Yes, transfer me now",
            attempts=1,
        )
        runner = ConversationRunner(judge=mock_judge, web_msg_config=web_msg_config, max_turns=20)

        with patch("src.conversation_runner.WebMessagingClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.connect = AsyncMock()
            mock_client.send_join = AsyncMock()
            mock_client.wait_for_welcome = AsyncMock(
                return_value="Hi, I'm Ava, WestJet's virtual assistant. How may I help you today?"
            )
            mock_client.send_message = AsyncMock()
            mock_client.receive_response = AsyncMock(
                side_effect=[
                    "Would you like me to escalate your request to a live agent?",
                    TimeoutError("no immediate follow-up"),
                    "detected_intent: speak_to_agent",
                ]
            )
            mock_client.disconnect = AsyncMock()
            MockClient.return_value = mock_client

            result = await runner.run_attempt(scenario, attempt_number=1)

        assert result.success is True
        assert result.detected_intent == "speak_to_agent"
        sent_messages = [call.args[0] for call in mock_client.send_message.call_args_list]
        assert sent_messages == [
            "Can I talk to support?",
            "Yes, transfer me now",
        ]

    @pytest.mark.asyncio
    async def test_speak_to_agent_uses_french_default_follow_up(
        self, mock_judge, web_msg_config
    ):
        scenario = TestScenario(
            name="Parler a un agent",
            persona="Voyageur",
            goal="Escalade vers un agent",
            first_message="Je veux parler a un agent",
            expected_intent="speak_to_agent",
            attempts=1,
        )
        localized_config = dict(web_msg_config)
        localized_config["language"] = "fr-CA"
        runner = ConversationRunner(
            judge=mock_judge,
            web_msg_config=localized_config,
            max_turns=20,
        )

        with patch("src.conversation_runner.WebMessagingClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.connect = AsyncMock()
            mock_client.send_join = AsyncMock()
            mock_client.wait_for_welcome = AsyncMock(
                return_value="Bonjour, je suis Ava, l'assistante virtuelle de WestJet. Comment puis-je vous aider aujourd'hui?"
            )
            mock_client.send_message = AsyncMock()
            mock_client.receive_response = AsyncMock(
                side_effect=[
                    "Voulez-vous que je fasse l'escalade vers un agent?",
                    TimeoutError("no immediate follow-up"),
                    "detected_intent: speak_to_agent",
                ]
            )
            mock_client.disconnect = AsyncMock()
            MockClient.return_value = mock_client

            result = await runner.run_attempt(scenario, attempt_number=1)

        assert result.success is True
        sent_messages = [call.args[0] for call in mock_client.send_message.call_args_list]
        assert sent_messages == [
            "Je veux parler a un agent",
            "Oui, connectez-moi a un agent en direct",
        ]

    @pytest.mark.asyncio
    async def test_flight_priority_change_french_yes_maps_to_within_72_hours(
        self, mock_judge, web_msg_config
    ):
        scenario = TestScenario(
            name="Priorite changement vol",
            persona="Voyageur",
            goal="Classer l'intention",
            first_message="Je dois changer mon vol",
            expected_intent="flight_priority_change",
            attempts=1,
        )
        localized_config = dict(web_msg_config)
        localized_config["language"] = "fr"
        runner = ConversationRunner(
            judge=mock_judge,
            web_msg_config=localized_config,
            max_turns=20,
        )

        with (
            patch("src.conversation_runner.WebMessagingClient") as MockClient,
            patch("src.conversation_runner.random.choice", return_value="oui"),
        ):
            mock_client = AsyncMock()
            mock_client.connect = AsyncMock()
            mock_client.send_join = AsyncMock()
            mock_client.wait_for_welcome = AsyncMock(
                return_value="Bonjour, je suis Ava, l'assistante virtuelle de WestJet. Comment puis-je vous aider aujourd'hui?"
            )
            mock_client.send_message = AsyncMock()
            mock_client.receive_response = AsyncMock(
                side_effect=[
                    "Votre vol est-il dans les 72 prochaines heures?",
                    TimeoutError("no immediate follow-up"),
                    "detected_intent: flight_change_priority_within_72_hours",
                ]
            )
            mock_client.disconnect = AsyncMock()
            MockClient.return_value = mock_client

            result = await runner.run_attempt(scenario, attempt_number=1)

        assert result.success is True
        sent_messages = [call.args[0] for call in mock_client.send_message.call_args_list]
        assert sent_messages == ["Je dois changer mon vol", "oui"]

    @pytest.mark.asyncio
    async def test_custom_follow_up_utterance_requires_strict_expected_intent_match(
        self, mock_judge, web_msg_config
    ):
        scenario = TestScenario(
            name="Custom Branch",
            persona="Traveler",
            goal="Classify as flight status after second-turn correction",
            first_message="I need to speak to an agent",
            expected_intent="flight_status",
            intent_follow_up_user_message="Actually check my flight status instead",
            attempts=1,
        )
        runner = ConversationRunner(judge=mock_judge, web_msg_config=web_msg_config, max_turns=20)

        with patch("src.conversation_runner.WebMessagingClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.connect = AsyncMock()
            mock_client.send_join = AsyncMock()
            mock_client.wait_for_welcome = AsyncMock(
                return_value="Hi, I'm Ava, WestJet's virtual assistant. How may I help you today?"
            )
            mock_client.send_message = AsyncMock()
            mock_client.receive_response = AsyncMock(
                side_effect=[
                    "Would you like me to escalate this to a live agent?",
                    TimeoutError("no immediate follow-up"),
                    "detected_intent: flight_status",
                ]
            )
            mock_client.disconnect = AsyncMock()
            MockClient.return_value = mock_client

            result = await runner.run_attempt(scenario, attempt_number=1)

        assert result.success is True
        assert result.detected_intent == "flight_status"
        sent_messages = [call.args[0] for call in mock_client.send_message.call_args_list]
        assert sent_messages == [
            "I need to speak to an agent",
            "Actually check my flight status instead",
        ]

    @pytest.mark.asyncio
    async def test_attempt_includes_debug_frames_when_client_captures_them(
        self, mock_judge, web_msg_config
    ):
        scenario = TestScenario(
            name="Intent Classification",
            persona="Traveler",
            goal="Classify the request",
            first_message="I want to cancel my booking",
            expected_intent="flight_cancel",
            attempts=1,
        )
        runner = ConversationRunner(judge=mock_judge, web_msg_config=web_msg_config, max_turns=20)

        with patch("src.conversation_runner.WebMessagingClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.connect = AsyncMock()
            mock_client.send_join = AsyncMock()
            mock_client.wait_for_welcome = AsyncMock(
                return_value="Hi, I'm Ava, WestJet's virtual assistant. How may I help you today?"
            )
            mock_client.send_message = AsyncMock()
            mock_client.receive_response = AsyncMock(return_value="INTENT=flight_cancel")
            mock_client.disconnect = AsyncMock()
            mock_client.get_debug_frames = MagicMock(return_value=[
                {"stage": "connect", "type": "SessionResponse"}
            ])
            MockClient.return_value = mock_client

            result = await runner.run_attempt(scenario, attempt_number=1)

        assert result.success is True
        assert len(result.debug_frames) == 1
        assert result.debug_frames[0]["type"] == "SessionResponse"


class TestToolValidation:
    @pytest.mark.asyncio
    async def test_missing_tool_signal_fails_attempt(self, mock_judge, web_msg_config):
        scenario = TestScenario(
            name="Tool Validation Missing Signal",
            persona="Traveler",
            goal="Complete flow",
            first_message="I need to change my flight",
            attempts=1,
            tool_validation={
                "loose_rule": {"tool": "flight_lookup"},
            },
        )
        runner = ConversationRunner(judge=mock_judge, web_msg_config=web_msg_config, max_turns=1)
        mock_judge.evaluate_goal.return_value = GoalEvaluation(
            success=True,
            explanation="Handled successfully.",
        )

        with patch("src.conversation_runner.WebMessagingClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.connect = AsyncMock()
            mock_client.send_join = AsyncMock()
            mock_client.wait_for_welcome = AsyncMock(
                return_value="Hi, I'm Ava, WestJet's virtual assistant. How may I help you today?"
            )
            mock_client.send_message = AsyncMock()
            mock_client.receive_response = AsyncMock(return_value="Sure, I can help with that.")
            mock_client.disconnect = AsyncMock()
            MockClient.return_value = mock_client

            result = await runner.run_attempt(scenario, attempt_number=1)

        assert result.success is False
        assert result.error == "missing_tool_signal"
        assert result.tool_validation_result is not None
        assert result.tool_validation_result.loose_pass is False
        assert result.tool_validation_result.missing_signal is True

    @pytest.mark.asyncio
    async def test_marker_fallback_satisfies_loose_validation(self, mock_judge, web_msg_config):
        scenario = TestScenario(
            name="Tool Marker Pass",
            persona="Traveler",
            goal="Complete flow",
            first_message="I need to change my flight",
            attempts=1,
            tool_validation={
                "loose_rule": {
                    "all": [
                        {"tool": "flight_lookup"},
                        {"tool": "flight_change_priority"},
                    ]
                }
            },
        )
        runner = ConversationRunner(judge=mock_judge, web_msg_config=web_msg_config, max_turns=1)
        mock_judge.evaluate_goal.return_value = GoalEvaluation(
            success=True,
            explanation="Handled successfully.",
        )

        with patch("src.conversation_runner.WebMessagingClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.connect = AsyncMock()
            mock_client.send_join = AsyncMock()
            mock_client.wait_for_welcome = AsyncMock(
                return_value="Hi, I'm Ava, WestJet's virtual assistant. How may I help you today?"
            )
            mock_client.send_message = AsyncMock()
            mock_client.receive_response = AsyncMock(
                return_value=(
                    'tool_event: {"tool":"flight_lookup","status":"success"}\n'
                    'tool_event: {"tool":"flight_change_priority","status":"success"}'
                )
            )
            mock_client.disconnect = AsyncMock()
            MockClient.return_value = mock_client

            result = await runner.run_attempt(scenario, attempt_number=1)

        assert result.success is True
        assert result.tool_validation_result is not None
        assert result.tool_validation_result.loose_pass is True
        assert len(result.tool_events) == 2

    @pytest.mark.asyncio
    async def test_strict_order_failure_is_diagnostic_only(self, mock_judge, web_msg_config):
        scenario = TestScenario(
            name="Tool Strict Diagnostic",
            persona="Traveler",
            goal="Complete flow",
            first_message="I need to change my flight",
            attempts=1,
            tool_validation={
                "loose_rule": {
                    "all": [
                        {"tool": "flight_lookup"},
                        {"tool": "flight_change_priority"},
                    ]
                },
                "strict_rule": {
                    "in_order": [
                        {"tool": "flight_lookup"},
                        {"tool": "flight_change_priority"},
                    ]
                },
            },
        )
        runner = ConversationRunner(judge=mock_judge, web_msg_config=web_msg_config, max_turns=1)
        mock_judge.evaluate_goal.return_value = GoalEvaluation(
            success=True,
            explanation="Handled successfully.",
        )

        with patch("src.conversation_runner.WebMessagingClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.connect = AsyncMock()
            mock_client.send_join = AsyncMock()
            mock_client.wait_for_welcome = AsyncMock(
                return_value="Hi, I'm Ava, WestJet's virtual assistant. How may I help you today?"
            )
            mock_client.send_message = AsyncMock()
            mock_client.receive_response = AsyncMock(
                return_value=(
                    'tool_event: {"tool":"flight_change_priority","status":"success"}\n'
                    'tool_event: {"tool":"flight_lookup","status":"success"}'
                )
            )
            mock_client.disconnect = AsyncMock()
            MockClient.return_value = mock_client

            result = await runner.run_attempt(scenario, attempt_number=1)

        assert result.success is True
        assert result.tool_validation_result is not None
        assert result.tool_validation_result.loose_pass is True
        assert result.tool_validation_result.strict_pass is False
        assert result.tool_validation_result.order_violations


class TestJourneyMode:
    @pytest.mark.asyncio
    async def test_journey_mode_uses_metadata_containment_and_passes(self, mock_judge):
        scenario = TestScenario(
            name="Journey Scenario",
            persona="Traveler",
            goal="Handle full journey",
            first_message="I need to cancel my booking",
            attempts=1,
            journey_category="flight_cancel",
            journey_validation=JourneyValidationConfig(
                require_containment=True,
                require_fulfillment=True,
            ),
        )
        web_msg_config = {
            "region": "mypurecloud.com",
            "deployment_id": "test-deployment-123",
            "timeout": 30,
            "harness_mode": "journey",
            "gc_client_id": "client-id",
            "gc_client_secret": "client-secret",
            "primary_categories": [
                {
                    "name": "flight_cancel",
                    "keywords": ["cancel"],
                    "rubric": "Route through cancellation flow.",
                }
            ],
        }
        runner = ConversationRunner(
            judge=mock_judge,
            web_msg_config=web_msg_config,
            max_turns=1,
        )
        mock_judge.evaluate_goal.return_value = GoalEvaluation(
            success=True,
            explanation="Goal reached",
        )
        mock_judge.evaluate_journey.return_value = JourneyValidationResult(
            category_match=True,
            fulfilled=True,
            path_correct=True,
            contained=True,
            explanation="Journey looks correct",
        )

        with patch("src.conversation_runner.WebMessagingClient") as MockClient:
            with patch("src.conversation_runner.GenesysConversationsClient") as MockGenesys:
                mock_client = AsyncMock()
                mock_client.connect = AsyncMock()
                mock_client.send_join = AsyncMock()
                mock_client.wait_for_welcome = AsyncMock(return_value="Welcome!")
                mock_client.send_message = AsyncMock()
                mock_client.receive_response = AsyncMock(return_value="I can help.")
                mock_client.disconnect = AsyncMock()
                mock_client.conversation_id = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
                mock_client.get_conversation_id_candidates = MagicMock(
                    return_value=["aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"]
                )
                mock_client.participant_id = None
                MockClient.return_value = mock_client

                mock_genesys = MagicMock()
                mock_genesys.get_conversation_payload.return_value = {
                    "participants": [{"purpose": "customer"}]
                }
                MockGenesys.return_value = mock_genesys

                result = await runner.run_attempt(scenario, attempt_number=1)

        assert result.success is True
        assert result.journey_validation_result is not None
        assert result.journey_validation_result.containment_source == "metadata"
        assert result.journey_validation_result.contained is True

    @pytest.mark.asyncio
    async def test_journey_mode_fails_when_containment_unknown(self, mock_judge):
        scenario = TestScenario(
            name="Journey Scenario",
            persona="Traveler",
            goal="Handle full journey",
            first_message="I need to cancel my booking",
            attempts=1,
            journey_category="flight_cancel",
            journey_validation=JourneyValidationConfig(
                require_containment=True,
                require_fulfillment=True,
            ),
        )
        web_msg_config = {
            "region": "mypurecloud.com",
            "deployment_id": "test-deployment-123",
            "timeout": 30,
            "harness_mode": "journey",
        }
        runner = ConversationRunner(
            judge=mock_judge,
            web_msg_config=web_msg_config,
            max_turns=1,
        )
        mock_judge.evaluate_goal.return_value = GoalEvaluation(
            success=True,
            explanation="Goal reached",
        )
        mock_judge.infer_containment.return_value = {
            "contained": None,
            "confidence": 0.2,
            "explanation": "Not enough evidence",
        }
        mock_judge.evaluate_journey.return_value = JourneyValidationResult(
            category_match=True,
            fulfilled=True,
            path_correct=True,
            contained=None,
            explanation="Journey mostly correct",
        )

        with patch("src.conversation_runner.WebMessagingClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.connect = AsyncMock()
            mock_client.send_join = AsyncMock()
            mock_client.wait_for_welcome = AsyncMock(return_value="Welcome!")
            mock_client.send_message = AsyncMock()
            mock_client.receive_response = AsyncMock(return_value="I can help.")
            mock_client.disconnect = AsyncMock()
            mock_client.conversation_id = None
            mock_client.get_conversation_id_candidates = MagicMock(return_value=[])
            mock_client.participant_id = None
            MockClient.return_value = mock_client

            result = await runner.run_attempt(scenario, attempt_number=1)

        assert result.success is False
        assert result.error == "journey_containment_unknown"
        assert result.journey_validation_result is not None
        assert "containment_unknown" in result.journey_validation_result.failure_reasons
