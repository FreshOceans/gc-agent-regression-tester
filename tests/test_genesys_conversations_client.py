"""Unit tests for GenesysConversationsClient."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from src.genesys_conversations_client import (
    GenesysConversationsClient,
    GenesysConversationsError,
)


def _mock_response(payload, status_code=200):
    response = MagicMock()
    response.json.return_value = payload
    response.raise_for_status = MagicMock()
    if status_code >= 400:
        response.raise_for_status.side_effect = Exception("http error")
    return response


class TestGenesysConversationsClient:
    def test_get_participant_attribute_filters_by_participant_id(self):
        client = GenesysConversationsClient(
            region="usw2.pure.cloud",
            client_id="client-id",
            client_secret="client-secret",
        )

        token_response = _mock_response({"access_token": "token", "expires_in": 300})
        conversation_response = _mock_response(
            {
                "participants": [
                    {"id": "other", "attributes": {"detected_intent": "flight_change"}},
                    {"id": "target", "attributes": {"detected_intent": "flight_cancel"}},
                ]
            }
        )

        with (
            patch("src.genesys_conversations_client.requests.post", return_value=token_response),
            patch("src.genesys_conversations_client.requests.get", return_value=conversation_response),
        ):
            intent = client.get_participant_attribute(
                conversation_id="conversation-1",
                participant_id="target",
                attribute_name="detected_intent",
                retries=1,
            )

        assert intent == "flight_cancel"

    def test_get_participant_attribute_supports_dict_values(self):
        client = GenesysConversationsClient(
            region="usw2.pure.cloud",
            client_id="client-id",
            client_secret="client-secret",
        )

        token_response = _mock_response({"access_token": "token", "expires_in": 300})
        conversation_response = _mock_response(
            {
                "participants": [
                    {
                        "id": "participant-1",
                        "attributes": {"detected_intent": {"intent": "flight_cancel"}},
                    }
                ]
            }
        )

        with (
            patch("src.genesys_conversations_client.requests.post", return_value=token_response),
            patch("src.genesys_conversations_client.requests.get", return_value=conversation_response),
        ):
            intent = client.get_participant_attribute(
                conversation_id="conversation-1",
                attribute_name="detected_intent",
                retries=1,
            )

        assert intent == "flight_cancel"

    def test_oauth_failure_raises_genesys_conversations_error(self):
        client = GenesysConversationsClient(
            region="usw2.pure.cloud",
            client_id="client-id",
            client_secret="client-secret",
        )

        with patch(
            "src.genesys_conversations_client.requests.post",
            side_effect=requests.RequestException("network down"),
        ):
            with pytest.raises(GenesysConversationsError):
                client.get_participant_attribute(
                    conversation_id="conversation-1",
                    attribute_name="detected_intent",
                    retries=1,
                )
