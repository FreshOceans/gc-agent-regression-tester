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

        assert result.conversation[0] == Message(
            role=MessageRole.AGENT, content="Welcome to support!"
        )

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

        expected = [
            Message(role=MessageRole.AGENT, content="Hello!"),
            Message(role=MessageRole.USER, content="First msg"),
            Message(role=MessageRole.AGENT, content="Reply 1"),
            Message(role=MessageRole.USER, content="Second msg"),
            Message(role=MessageRole.AGENT, content="Reply 2"),
        ]
        assert result.conversation == expected

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
