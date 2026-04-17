"""Unit tests for the Judge LLM Client."""

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.judge_llm import JudgeLLMClient, JudgeLLMError
from src.models import ContinueDecision, GoalEvaluation, Message, MessageRole


@pytest.fixture
def client():
    """Create a JudgeLLMClient instance for testing."""
    return JudgeLLMClient(
        base_url="http://localhost:11434", model="llama3", timeout=30
    )


@pytest.fixture
def sample_conversation():
    """Create a sample conversation history."""
    return [
        Message(role=MessageRole.AGENT, content="Hello! How can I help you today?"),
        Message(role=MessageRole.USER, content="I'd like to book a meeting."),
        Message(
            role=MessageRole.AGENT,
            content="Sure! When would you like to schedule it?",
        ),
    ]


class TestInit:
    def test_stores_base_url(self):
        client = JudgeLLMClient("http://localhost:11434", "llama3")
        assert client.base_url == "http://localhost:11434"

    def test_strips_trailing_slash(self):
        client = JudgeLLMClient("http://localhost:11434/", "llama3")
        assert client.base_url == "http://localhost:11434"

    def test_stores_model(self):
        client = JudgeLLMClient("http://localhost:11434", "llama3")
        assert client.model == "llama3"

    def test_default_timeout(self):
        client = JudgeLLMClient("http://localhost:11434", "llama3")
        assert client.timeout == 120

    def test_custom_timeout(self):
        client = JudgeLLMClient("http://localhost:11434", "llama3", timeout=60)
        assert client.timeout == 60


class TestVerifyConnection:
    @patch("src.judge_llm.requests.get")
    def test_success_when_model_available(self, mock_get, client):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "models": [{"name": "llama3:latest"}, {"name": "mistral:latest"}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        # Should not raise
        client.verify_connection()

    @patch("src.judge_llm.requests.get")
    def test_success_with_exact_name_match(self, mock_get, client):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "models": [{"name": "llama3"}, {"name": "mistral:latest"}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        client.verify_connection()

    @patch("src.judge_llm.requests.get")
    def test_raises_when_ollama_unreachable(self, mock_get, client):
        mock_get.side_effect = requests.ConnectionError("Connection refused")

        with pytest.raises(JudgeLLMError) as exc_info:
            client.verify_connection()

        error_msg = str(exc_info.value)
        assert "localhost:11434" in error_msg
        assert "llama3" in error_msg

    @patch("src.judge_llm.requests.get")
    def test_raises_when_model_not_found(self, mock_get, client):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "models": [{"name": "mistral:latest"}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        with pytest.raises(JudgeLLMError) as exc_info:
            client.verify_connection()

        error_msg = str(exc_info.value)
        assert "llama3" in error_msg
        assert "localhost:11434" in error_msg

    @patch("src.judge_llm.requests.get")
    def test_error_contains_both_url_and_model(self, mock_get):
        custom_client = JudgeLLMClient(
            "http://myserver:9999", "custom-model", timeout=10
        )
        mock_get.side_effect = requests.ConnectionError("Connection refused")

        with pytest.raises(JudgeLLMError) as exc_info:
            custom_client.verify_connection()

        error_msg = str(exc_info.value)
        assert "myserver:9999" in error_msg
        assert "custom-model" in error_msg


class TestGenerateUserMessage:
    @patch("src.judge_llm.requests.post")
    def test_returns_generated_message(self, mock_post, client):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {"content": "I'd like to book a meeting for Tuesday at 2pm."}
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        history = [
            Message(role=MessageRole.AGENT, content="Hello! How can I help you?")
        ]
        result = client.generate_user_message(
            persona="A busy executive",
            goal="Book a meeting for Tuesday at 2pm",
            conversation_history=history,
        )

        assert result == "I'd like to book a meeting for Tuesday at 2pm."

    @patch("src.judge_llm.requests.post")
    def test_initial_prompt_contains_persona_goal_and_welcome(
        self, mock_post, client
    ):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {"content": "Hi, I need help."}
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        history = [
            Message(role=MessageRole.AGENT, content="Welcome to our service!")
        ]
        client.generate_user_message(
            persona="A frustrated customer",
            goal="Get a refund for order #123",
            conversation_history=history,
        )

        # Verify the request payload
        call_args = mock_post.call_args
        payload = call_args[1]["json"] if "json" in call_args[1] else call_args[0][1]
        messages = payload["messages"]

        # System prompt should contain persona and goal
        system_content = messages[0]["content"]
        assert "A frustrated customer" in system_content
        assert "Get a refund for order #123" in system_content

        # Conversation history should contain the welcome message
        all_content = " ".join(m["content"] for m in messages)
        assert "Welcome to our service!" in all_content

    @patch("src.judge_llm.requests.post")
    def test_subsequent_prompt_includes_full_history(self, mock_post, client, sample_conversation):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {"content": "Next Tuesday at 2pm please."}
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        client.generate_user_message(
            persona="A busy executive",
            goal="Book a meeting",
            conversation_history=sample_conversation,
        )

        call_args = mock_post.call_args
        payload = call_args[1]["json"] if "json" in call_args[1] else call_args[0][1]
        messages = payload["messages"]

        all_content = " ".join(m["content"] for m in messages)
        assert "Hello! How can I help you today?" in all_content
        assert "I'd like to book a meeting." in all_content
        assert "Sure! When would you like to schedule it?" in all_content

    @patch("src.judge_llm.requests.post")
    def test_raises_on_connection_error(self, mock_post, client):
        mock_post.side_effect = requests.ConnectionError("Connection refused")

        history = [
            Message(role=MessageRole.AGENT, content="Hello!")
        ]
        with pytest.raises(JudgeLLMError):
            client.generate_user_message("persona", "goal", history)

    @patch("src.judge_llm.requests.post")
    def test_raises_on_empty_response(self, mock_post, client):
        mock_response = MagicMock()
        mock_response.json.return_value = {"message": {"content": ""}}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        history = [
            Message(role=MessageRole.AGENT, content="Hello!")
        ]
        with pytest.raises(JudgeLLMError):
            client.generate_user_message("persona", "goal", history)


class TestShouldContinue:
    @patch("src.judge_llm.requests.post")
    def test_returns_continue_decision(self, mock_post, client, sample_conversation):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {
                "content": json.dumps(
                    {"should_continue": True, "goal_achieved": None, "explanation": "Still working on it"}
                )
            }
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        result = client.should_continue("persona", "goal", sample_conversation)

        assert isinstance(result, ContinueDecision)
        assert result.should_continue is True
        assert result.goal_achieved is None

    @patch("src.judge_llm.requests.post")
    def test_goal_achieved(self, mock_post, client, sample_conversation):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {
                "content": json.dumps(
                    {"should_continue": False, "goal_achieved": True, "explanation": "Meeting booked"}
                )
            }
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        result = client.should_continue("persona", "goal", sample_conversation)

        assert result.should_continue is False
        assert result.goal_achieved is True
        assert result.explanation == "Meeting booked"

    @patch("src.judge_llm.requests.post")
    def test_raises_on_unparseable_json(self, mock_post, client, sample_conversation):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {"content": "This is not JSON at all"}
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        with pytest.raises(JudgeLLMError) as exc_info:
            client.should_continue("persona", "goal", sample_conversation)

        assert "ContinueDecision" in str(exc_info.value)

    @patch("src.judge_llm.requests.post")
    def test_handles_markdown_code_fences(self, mock_post, client, sample_conversation):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {
                "content": '```json\n{"should_continue": false, "goal_achieved": true, "explanation": "Done"}\n```'
            }
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        result = client.should_continue("persona", "goal", sample_conversation)

        assert result.should_continue is False
        assert result.goal_achieved is True


class TestEvaluateGoal:
    @patch("src.judge_llm.requests.post")
    def test_returns_goal_evaluation_success(self, mock_post, client, sample_conversation):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {
                "content": json.dumps(
                    {"success": True, "explanation": "The meeting was successfully booked."}
                )
            }
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        result = client.evaluate_goal("persona", "Book a meeting", sample_conversation)

        assert isinstance(result, GoalEvaluation)
        assert result.success is True
        assert result.explanation == "The meeting was successfully booked."

    @patch("src.judge_llm.requests.post")
    def test_returns_goal_evaluation_failure(self, mock_post, client, sample_conversation):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {
                "content": json.dumps(
                    {"success": False, "explanation": "The agent could not find available slots."}
                )
            }
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        result = client.evaluate_goal("persona", "Book a meeting", sample_conversation)

        assert result.success is False
        assert "could not find" in result.explanation

    @patch("src.judge_llm.requests.post")
    def test_raises_on_unparseable_response(self, mock_post, client, sample_conversation):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {"content": "I think the goal was achieved but I'm not sure"}
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        with pytest.raises(JudgeLLMError) as exc_info:
            client.evaluate_goal("persona", "goal", sample_conversation)

        assert "GoalEvaluation" in str(exc_info.value)

    @patch("src.judge_llm.requests.post")
    def test_raises_on_missing_required_field(self, mock_post, client, sample_conversation):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {
                "content": json.dumps({"success": True})  # Missing 'explanation'
            }
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        with pytest.raises(JudgeLLMError):
            client.evaluate_goal("persona", "goal", sample_conversation)

    @patch("src.judge_llm.requests.post")
    def test_includes_full_conversation_in_prompt(self, mock_post, client, sample_conversation):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {
                "content": json.dumps({"success": True, "explanation": "Goal achieved"})
            }
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        client.evaluate_goal("A busy exec", "Book a meeting", sample_conversation)

        call_args = mock_post.call_args
        payload = call_args[1]["json"] if "json" in call_args[1] else call_args[0][1]
        messages = payload["messages"]

        all_content = " ".join(m["content"] for m in messages)
        assert "Hello! How can I help you today?" in all_content
        assert "I'd like to book a meeting." in all_content
        assert "Sure! When would you like to schedule it?" in all_content
        assert "A busy exec" in all_content
        assert "Book a meeting" in all_content


# --- Property-Based Tests ---

from hypothesis import given, settings
from hypothesis import strategies as st
from unittest.mock import patch as mock_patch


class TestPropertyOllamaConnectionError:
    """Property 4: Ollama connection error includes URL and model name.

    **Validates: Requirements 2.3**
    """

    @given(
        base_url=st.from_regex(r"https?://[a-z][a-z0-9]{0,10}(:[0-9]{1,5})?", fullmatch=True),
        model_name=st.text(
            alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="-_"),
            min_size=1,
            max_size=20,
        ),
    )
    @settings(max_examples=100)
    def test_connection_error_contains_url_and_model(self, base_url, model_name):
        """For any base URL and model name, when Ollama is unreachable,
        the raised JudgeLLMError message should contain both the base URL and model name."""
        client = JudgeLLMClient(base_url=base_url, model=model_name, timeout=5)

        with mock_patch("src.judge_llm.requests.get") as mock_get:
            mock_get.side_effect = requests.ConnectionError("Connection refused")

            with pytest.raises(JudgeLLMError) as exc_info:
                client.verify_connection()

            error_msg = str(exc_info.value)
            # The error message must contain the base URL (stripped of trailing slash)
            assert base_url.rstrip("/") in error_msg, (
                f"Error message should contain base URL '{base_url.rstrip('/')}', got: {error_msg}"
            )
            # The error message must contain the model name
            assert model_name in error_msg, (
                f"Error message should contain model name '{model_name}', got: {error_msg}"
            )

    @given(
        base_url=st.from_regex(r"https?://[a-z][a-z0-9]{0,10}(:[0-9]{1,5})?", fullmatch=True),
        model_name=st.text(
            alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="-_"),
            min_size=1,
            max_size=20,
        ),
    )
    @settings(max_examples=100)
    def test_model_not_found_error_contains_url_and_model(self, base_url, model_name):
        """For any base URL and model name, when the model is unavailable,
        the raised JudgeLLMError message should contain both the base URL and model name."""
        client = JudgeLLMClient(base_url=base_url, model=model_name, timeout=5)

        with mock_patch("src.judge_llm.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "models": [{"name": "some-other-model:latest"}]
            }
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            with pytest.raises(JudgeLLMError) as exc_info:
                client.verify_connection()

            error_msg = str(exc_info.value)
            # The error message must contain the base URL (stripped of trailing slash)
            assert base_url.rstrip("/") in error_msg, (
                f"Error message should contain base URL '{base_url.rstrip('/')}', got: {error_msg}"
            )
            # The error message must contain the model name
            assert model_name in error_msg, (
                f"Error message should contain model name '{model_name}', got: {error_msg}"
            )
