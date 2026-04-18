"""Unit tests for ConversationRunner using mocked JudgeLLMClient and WebMessagingClient."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.conversation_runner import ConversationRunner
from src.judge_llm import JudgeLLMError
from src.models import (
    AttemptResult,
    ContinueDecision,
    GoalEvaluation,
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
    async def test_disconnect_called_on_error(self, runner, mock_judge, scenario):
        """Test that disconnect is always called in the finally block."""
        with patch("src.conversation_runner.WebMessagingClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.connect = AsyncMock(side_effect=WebMessagingError("Connection failed"))
            mock_client.disconnect = AsyncMock()
            MockClient.return_value = mock_client

            await runner.run_attempt(scenario, attempt_number=1)

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
            mock_client.participant_id = "participant-456"
            MockClient.return_value = mock_client

            mock_conversations_client = MagicMock()
            mock_conversations_client.get_participant_attribute.return_value = "flight_cancel"
            MockConversationsClient.return_value = mock_conversations_client

            result = await runner.run_attempt(scenario, attempt_number=1)

        assert result.success is True
        assert result.detected_intent == "flight_cancel"
        assert "Conversations API participant attribute" in result.explanation
        mock_judge.evaluate_goal.assert_not_called()

    @pytest.mark.asyncio
    async def test_expected_intent_uses_judge_to_infer_conversation_id_for_fallback(
        self, mock_judge, web_msg_config
    ):
        scenario = TestScenario(
            name="Intent Classification",
            persona="Traveler",
            goal="Classify the request",
            first_message="I want to cancel my booking",
            expected_intent="flight_cancel",
            judge_capture_conversation_id=True,
            attempts=1,
        )
        config_with_fallback = dict(
            web_msg_config,
            gc_client_id="client-id",
            gc_client_secret="client-secret",
            intent_attribute_name="detected_intent",
            judge_capture_conversation_id=True,
        )
        runner = ConversationRunner(
            judge=mock_judge,
            web_msg_config=config_with_fallback,
            max_turns=20,
        )
        mock_judge.extract_conversation_id.return_value = "11111111-2222-4333-8444-555555555555"

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
            mock_client.conversation_id = None
            mock_client.participant_id = "participant-456"
            MockClient.return_value = mock_client

            mock_conversations_client = MagicMock()
            mock_conversations_client.get_participant_attribute.return_value = "flight_cancel"
            MockConversationsClient.return_value = mock_conversations_client

            result = await runner.run_attempt(scenario, attempt_number=1)

        assert result.success is True
        assert result.detected_intent == "flight_cancel"
        assert "Conversations API participant attribute" in result.explanation
        mock_judge.extract_conversation_id.assert_called()
        mock_conversations_client.get_participant_attribute.assert_called_once()

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
