"""Unit tests for GenesysTranscriptImportClient."""

from unittest.mock import MagicMock, patch

from src.genesys_transcript_import_client import GenesysTranscriptImportClient


def _mock_response(payload, status_code=200):
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = payload
    response.raise_for_status = MagicMock()
    if status_code >= 400:
        response.raise_for_status.side_effect = Exception("http error")
    return response


def test_query_conversation_ids_returns_descending_and_capped():
    client = GenesysTranscriptImportClient(
        region="usw2.pure.cloud",
        client_id="client-id",
        client_secret="client-secret",
        retries=1,
    )
    token_response = _mock_response({"access_token": "token", "expires_in": 300})
    query_response = _mock_response(
        {
            "conversations": [
                {
                    "conversationId": "11111111-2222-4333-8444-555555555555",
                    "conversationEnd": "2026-04-19T01:00:00Z",
                },
                {
                    "conversationId": "66666666-7777-4888-9999-aaaaaaaaaaaa",
                    "conversationEnd": "2026-04-19T02:00:00Z",
                },
            ]
        }
    )
    with (
        patch("src.genesys_transcript_import_client.requests.post", return_value=token_response),
        patch("src.genesys_transcript_import_client.requests.request", return_value=query_response),
    ):
        rows = client.query_conversation_ids(
            filter_payload={},
            interval="2026-04-18T02:00:00Z/2026-04-19T02:00:00Z",
            max_results=1,
        )
    assert len(rows) == 1
    assert rows[0]["conversation_id"] == "66666666-7777-4888-9999-aaaaaaaaaaaa"


def test_import_transcripts_by_ids_returns_fetched_failed_and_skipped():
    client = GenesysTranscriptImportClient(
        region="usw2.pure.cloud",
        client_id="client-id",
        client_secret="client-secret",
        retries=1,
    )

    def _fake_fetch(conversation_id):
        if conversation_id == "bad":
            raise RuntimeError("fetch failed")
        if conversation_id == "empty":
            return {"id": conversation_id, "participants": []}
        return {
            "id": conversation_id,
            "participants": [
                {
                    "id": "p1",
                    "purpose": "customer",
                    "messages": [{"messageText": "hello", "timestamp": "2026-04-19T01:00:00Z"}],
                }
            ],
        }

    with patch.object(client, "fetch_conversation_payload", side_effect=_fake_fetch):
        outcomes = client.import_transcripts_by_ids(
            [
                "11111111-2222-4333-8444-555555555555",
                "empty",
                "bad",
            ]
        )

    assert len(outcomes["fetched"]) == 1
    assert len(outcomes["skipped"]) == 1
    assert len(outcomes["failed"]) == 1


def test_normalize_conversation_payload_extracts_messages():
    client = GenesysTranscriptImportClient(
        region="usw2.pure.cloud",
        client_id="client-id",
        client_secret="client-secret",
        retries=1,
    )
    payload = {
        "id": "11111111-2222-4333-8444-555555555555",
        "participants": [
            {
                "id": "cust-1",
                "purpose": "customer",
                "messages": [
                    {
                        "messageText": "I need help",
                        "timestamp": "2026-04-19T01:00:00Z",
                    }
                ],
            },
            {
                "id": "agent-1",
                "purpose": "agent",
                "messages": [
                    {
                        "messageText": "Sure, I can help",
                        "timestamp": "2026-04-19T01:00:01Z",
                    }
                ],
            },
        ],
    }

    normalized = client.normalize_conversation_payload(payload)
    assert normalized["conversation_id"] == "11111111-2222-4333-8444-555555555555"
    assert len(normalized["participants"]) == 2
    assert len(normalized["messages"]) == 2
    assert normalized["messages"][0]["role"] == "customer"
    assert normalized["messages"][1]["role"] == "agent"

