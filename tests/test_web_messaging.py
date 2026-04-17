"""Unit tests for the Web Messaging Client."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.web_messaging_client import WebMessagingClient, WebMessagingError


class TestWebMessagingClientInit:
    """Tests for WebMessagingClient initialization."""

    def test_init_stores_region_and_deployment_id(self):
        client = WebMessagingClient(region="mypurecloud.com", deployment_id="abc-123")
        assert client.region == "mypurecloud.com"
        assert client.deployment_id == "abc-123"

    def test_init_default_timeout(self):
        client = WebMessagingClient(region="mypurecloud.com", deployment_id="abc-123")
        assert client.timeout == 30

    def test_init_custom_timeout(self):
        client = WebMessagingClient(
            region="mypurecloud.com", deployment_id="abc-123", timeout=60
        )
        assert client.timeout == 60

    def test_ws_url_format(self):
        client = WebMessagingClient(
            region="mypurecloud.de", deployment_id="deploy-456"
        )
        expected = "wss://webmessaging.mypurecloud.de/v1?deploymentId=deploy-456"
        assert client.ws_url == expected


class TestWebMessagingClientConnect:
    """Tests for WebMessagingClient.connect()."""

    @pytest.mark.asyncio
    async def test_connect_success(self):
        client = WebMessagingClient(region="mypurecloud.com", deployment_id="dep-1")

        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(
            return_value=json.dumps({"type": "SessionResponse", "body": {}})
        )

        with patch("src.web_messaging_client.websockets.connect", new_callable=AsyncMock) as mock_connect:
            mock_connect.return_value = mock_ws
            await client.connect()

        assert client._ws is mock_ws
        assert client._token is not None
        # Verify configureSession was sent
        mock_ws.send.assert_called_once()
        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["action"] == "configureSession"
        assert sent_data["deploymentId"] == "dep-1"

    @pytest.mark.asyncio
    async def test_connect_failure_includes_deployment_id_and_region(self):
        client = WebMessagingClient(
            region="mypurecloud.com", deployment_id="dep-fail"
        )

        with patch("src.web_messaging_client.websockets.connect", new_callable=AsyncMock) as mock_connect:
            mock_connect.side_effect = ConnectionRefusedError("Connection refused")

            with pytest.raises(WebMessagingError) as exc_info:
                await client.connect()

            error_msg = str(exc_info.value)
            assert "dep-fail" in error_msg
            assert "mypurecloud.com" in error_msg

    @pytest.mark.asyncio
    async def test_connect_session_timeout_includes_ids(self):
        client = WebMessagingClient(
            region="mypurecloud.de", deployment_id="dep-timeout", timeout=1
        )

        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(side_effect=asyncio.TimeoutError())

        with patch("src.web_messaging_client.websockets.connect", new_callable=AsyncMock) as mock_connect:
            mock_connect.return_value = mock_ws

            with pytest.raises(WebMessagingError) as exc_info:
                await client.connect()

            error_msg = str(exc_info.value)
            assert "dep-timeout" in error_msg
            assert "mypurecloud.de" in error_msg


class TestWebMessagingClientWaitForWelcome:
    """Tests for WebMessagingClient.wait_for_welcome()."""

    @pytest.mark.asyncio
    async def test_wait_for_welcome_returns_text(self):
        client = WebMessagingClient(region="mypurecloud.com", deployment_id="dep-1")
        client._ws = AsyncMock()
        client._token = "test-token"

        welcome_msg = json.dumps({
            "type": "message",
            "class": "StructuredMessage",
            "body": {"text": "Hello! How can I help you?"},
        })
        client._ws.recv = AsyncMock(return_value=welcome_msg)

        result = await client.wait_for_welcome()
        assert result == "Hello! How can I help you?"

    @pytest.mark.asyncio
    async def test_wait_for_welcome_timeout_raises(self):
        client = WebMessagingClient(
            region="mypurecloud.com", deployment_id="dep-1", timeout=1
        )
        client._ws = AsyncMock()
        client._token = "test-token"
        client._ws.recv = AsyncMock(side_effect=asyncio.TimeoutError())

        with pytest.raises(TimeoutError) as exc_info:
            await client.wait_for_welcome()

        assert "welcome message" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_wait_for_welcome_not_connected_raises(self):
        client = WebMessagingClient(region="mypurecloud.com", deployment_id="dep-1")

        with pytest.raises(WebMessagingError) as exc_info:
            await client.wait_for_welcome()

        assert "dep-1" in str(exc_info.value)
        assert "mypurecloud.com" in str(exc_info.value)


class TestWebMessagingClientSendMessage:
    """Tests for WebMessagingClient.send_message()."""

    @pytest.mark.asyncio
    async def test_send_message_correct_format(self):
        client = WebMessagingClient(region="mypurecloud.com", deployment_id="dep-1")
        client._ws = AsyncMock()
        client._token = "test-token"

        await client.send_message("I need help booking a meeting")

        client._ws.send.assert_called_once()
        sent_data = json.loads(client._ws.send.call_args[0][0])
        assert sent_data["action"] == "onMessage"
        assert sent_data["token"] == "test-token"
        assert sent_data["message"]["type"] == "Text"
        assert sent_data["message"]["text"] == "I need help booking a meeting"

    @pytest.mark.asyncio
    async def test_send_message_not_connected_raises(self):
        client = WebMessagingClient(region="mypurecloud.com", deployment_id="dep-1")

        with pytest.raises(WebMessagingError) as exc_info:
            await client.send_message("hello")

        assert "dep-1" in str(exc_info.value)
        assert "mypurecloud.com" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_send_message_ws_error_includes_ids(self):
        client = WebMessagingClient(
            region="mypurecloud.com", deployment_id="dep-err"
        )
        client._ws = AsyncMock()
        client._token = "test-token"
        client._ws.send = AsyncMock(side_effect=Exception("WebSocket closed"))

        with pytest.raises(WebMessagingError) as exc_info:
            await client.send_message("test")

        error_msg = str(exc_info.value)
        assert "dep-err" in error_msg
        assert "mypurecloud.com" in error_msg


class TestWebMessagingClientReceiveResponse:
    """Tests for WebMessagingClient.receive_response()."""

    @pytest.mark.asyncio
    async def test_receive_response_structured_message(self):
        client = WebMessagingClient(region="mypurecloud.com", deployment_id="dep-1")
        client._ws = AsyncMock()
        client._token = "test-token"

        response_msg = json.dumps({
            "type": "message",
            "class": "StructuredMessage",
            "body": {"text": "Sure, I can help with that."},
        })
        client._ws.recv = AsyncMock(return_value=response_msg)

        result = await client.receive_response()
        assert result == "Sure, I can help with that."

    @pytest.mark.asyncio
    async def test_receive_response_simple_body(self):
        client = WebMessagingClient(region="mypurecloud.com", deployment_id="dep-1")
        client._ws = AsyncMock()
        client._token = "test-token"

        response_msg = json.dumps({
            "type": "message",
            "body": "Simple text response",
        })
        client._ws.recv = AsyncMock(return_value=response_msg)

        result = await client.receive_response()
        assert result == "Simple text response"

    @pytest.mark.asyncio
    async def test_receive_response_timeout_raises(self):
        client = WebMessagingClient(
            region="mypurecloud.com", deployment_id="dep-1", timeout=1
        )
        client._ws = AsyncMock()
        client._token = "test-token"
        client._ws.recv = AsyncMock(side_effect=asyncio.TimeoutError())

        with pytest.raises(TimeoutError) as exc_info:
            await client.receive_response()

        assert "agent response" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_receive_response_not_connected_raises(self):
        client = WebMessagingClient(region="mypurecloud.com", deployment_id="dep-1")

        with pytest.raises(WebMessagingError) as exc_info:
            await client.receive_response()

        assert "dep-1" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_receive_response_skips_non_text_messages(self):
        client = WebMessagingClient(region="mypurecloud.com", deployment_id="dep-1")
        client._ws = AsyncMock()
        client._token = "test-token"

        # First message is a typing indicator (no text), second is the actual response
        typing_msg = json.dumps({"type": "typing", "body": {}})
        text_msg = json.dumps({
            "type": "message",
            "class": "StructuredMessage",
            "body": {"text": "Here is your answer."},
        })
        client._ws.recv = AsyncMock(side_effect=[typing_msg, text_msg])

        result = await client.receive_response()
        assert result == "Here is your answer."


class TestWebMessagingClientDisconnect:
    """Tests for WebMessagingClient.disconnect()."""

    @pytest.mark.asyncio
    async def test_disconnect_closes_ws(self):
        client = WebMessagingClient(region="mypurecloud.com", deployment_id="dep-1")
        mock_ws = AsyncMock()
        client._ws = mock_ws
        client._token = "test-token"

        await client.disconnect()

        mock_ws.close.assert_called_once()
        assert client._ws is None
        assert client._token is None

    @pytest.mark.asyncio
    async def test_disconnect_when_not_connected(self):
        client = WebMessagingClient(region="mypurecloud.com", deployment_id="dep-1")
        # Should not raise
        await client.disconnect()

    @pytest.mark.asyncio
    async def test_disconnect_handles_close_error(self):
        client = WebMessagingClient(region="mypurecloud.com", deployment_id="dep-1")
        client._ws = AsyncMock()
        client._ws.close = AsyncMock(side_effect=Exception("Already closed"))
        client._token = "test-token"

        # Should not raise even if close fails
        await client.disconnect()
        assert client._ws is None


class TestWebMessagingErrorMessages:
    """Tests verifying error messages include deployment ID and region."""

    @pytest.mark.asyncio
    async def test_all_connection_errors_include_both_ids(self):
        """Verify that connection errors always include deployment_id and region."""
        region = "usw2.pure.cloud"
        deployment_id = "unique-deploy-id-789"
        client = WebMessagingClient(region=region, deployment_id=deployment_id)

        with patch("src.web_messaging_client.websockets.connect", new_callable=AsyncMock) as mock_connect:
            mock_connect.side_effect = OSError("Network unreachable")

            with pytest.raises(WebMessagingError) as exc_info:
                await client.connect()

            error_msg = str(exc_info.value)
            assert deployment_id in error_msg
            assert region in error_msg
