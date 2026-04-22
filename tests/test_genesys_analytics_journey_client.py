"""Unit tests for the analytics reporting-turns client."""

import pytest
import requests

from src.genesys_analytics_journey_client import (
    GenesysAnalyticsJourneyClient,
    GenesysAnalyticsJourneyError,
)
from src.models import ANALYTICS_AUTH_MODE_MANUAL_BEARER


def test_extract_rows_prefers_known_list_keys():
    payload = {
        "results": [
            {"conversationId": "11111111-1111-1111-1111-111111111111"},
            {"conversationId": "22222222-2222-2222-2222-222222222222"},
        ]
    }
    rows = GenesysAnalyticsJourneyClient.extract_rows(payload)
    assert len(rows) == 2
    assert rows[0]["conversationId"].startswith("1111")


def test_extract_rows_recursive_fallback_when_lists_missing():
    payload = {
        "outer": {
            "inner": [
                {"conversation": {"conversationId": "33333333-3333-3333-3333-333333333333"}},
                {"not": "a-row"},
            ]
        }
    }
    rows = GenesysAnalyticsJourneyClient.extract_rows(payload)
    assert len(rows) >= 1


def test_fetch_conversation_units_paginates_and_dedupes(monkeypatch):
    client = GenesysAnalyticsJourneyClient(
        region="usw2.pure.cloud",
        client_id="client-id",
        client_secret="client-secret",
    )

    calls = []

    def _fake_fetch_page(
        *,
        bot_flow_id,
        interval,
        page_size,
        page_number,
        divisions,
        language_filter,
        extra_params,
        next_uri=None,
        observer=None,
        stop_requested=None,
    ):
        calls.append(page_number)
        if page_number == 1:
            return {
                "results": [
                    {"conversationId": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"},
                    {"conversationId": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"},
                ]
            }
        if page_number == 2:
            return {
                "results": [
                    {"conversationId": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"},
                    {"conversationId": "cccccccc-cccc-cccc-cccc-cccccccccccc"},
                ]
            }
        return {"results": []}

    monkeypatch.setattr(client, "fetch_reporting_turns_page", _fake_fetch_page)
    observer_events: list[dict] = []

    result = client.fetch_conversation_units(
        bot_flow_id="flow-id",
        interval="2026-04-19T00:00:00.000Z/2026-04-20T00:00:00.000Z",
        page_size=2,
        max_conversations=3,
        observer=lambda payload: observer_events.append(payload),
    )

    assert result["page_count"] == 2
    assert calls == [1, 2]
    ids = [entry["conversation_id"] for entry in result["conversations"]]
    assert ids == [
        "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
        "cccccccc-cccc-cccc-cccc-cccccccccccc",
    ]
    assert any(event.get("event") == "page_fetch_started" for event in observer_events)
    assert any(
        event.get("event") == "page_fetch_completed"
        and event.get("page_number") == 1
        and event.get("rows_count") == 2
        for event in observer_events
    )


def test_request_json_emits_retry_observer_events(monkeypatch):
    client = GenesysAnalyticsJourneyClient(
        region="usw2.pure.cloud",
        client_id="client-id",
        client_secret="client-secret",
        retries=3,
        retry_delay_seconds=0,
    )
    monkeypatch.setattr(client, "_get_access_token", lambda: "token")

    class _Response:
        def __init__(self, status_code: int, payload: dict):
            self.status_code = status_code
            self._payload = payload

        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.HTTPError(f"{self.status_code} boom")

        def json(self):
            return self._payload

    responses = iter(
        [
            _Response(429, {}),
            _Response(200, {"results": []}),
        ]
    )
    monkeypatch.setattr(
        requests,
        "request",
        lambda *args, **kwargs: next(responses),
    )

    observer_events: list[dict] = []
    payload = client._request_json(
        method="GET",
        path="/api/v2/analytics/botflows/flow/divisions/reportingturns",
        observer=lambda event: observer_events.append(event),
        request_context={"page_number": 1},
    )

    assert payload == {"results": []}
    assert any(
        event.get("event") == "request_retry"
        and event.get("status_code") == 429
        for event in observer_events
    )
    assert any(
        event.get("event") == "request_attempt_succeeded"
        and event.get("attempt") == 2
        for event in observer_events
    )


def test_fetch_reporting_turns_page_builds_get_request(monkeypatch):
    client = GenesysAnalyticsJourneyClient(
        region="usw2.pure.cloud",
        client_id="client-id",
        client_secret="client-secret",
        page_size_cap=100,
    )

    captured = {}

    def _fake_request_json(
        *,
        method,
        path,
        json_payload=None,
        params=None,
        observer=None,
        request_context=None,
        stop_requested=None,
    ):
        captured["method"] = method
        captured["path"] = path
        captured["json_payload"] = json_payload
        captured["params"] = params
        captured["request_context"] = request_context
        return {"conversations": []}

    monkeypatch.setattr(client, "_request_json", _fake_request_json)
    payload = client.fetch_reporting_turns_page(
        bot_flow_id="flow-id",
        interval="2026-04-19T00:00:00.000Z/2026-04-20T00:00:00.000Z",
        page_size=250,
        page_number=2,
        divisions=["division-1", "division-2"],
        extra_params={"filter": {"type": "and", "predicates": []}},
    )

    assert payload == {"conversations": []}
    assert captured["method"] == "GET"
    assert (
        captured["path"]
        == "/api/v2/analytics/botflows/flow-id/divisions/reportingturns"
    )
    assert captured["params"] is not None
    assert captured["request_context"] == {"page_number": 2}
    assert captured["json_payload"] is None
    assert captured["params"]["interval"].startswith("2026-04-19T00:00:00.000Z")
    assert captured["params"]["pageSize"] == 100
    assert captured["params"]["pageNumber"] == 2
    assert captured["params"]["divisions"] == "division-1,division-2"
    # Unsupported opaque keys should not be forwarded.
    assert "filter" not in captured["params"]


def test_manual_bearer_mode_uses_provided_token_without_oauth(monkeypatch):
    client = GenesysAnalyticsJourneyClient(
        region="usw2.pure.cloud",
        client_id="unused",
        client_secret="unused",
        auth_mode=ANALYTICS_AUTH_MODE_MANUAL_BEARER,
        manual_bearer_token="manual-token",
    )

    def _oauth_call(*args, **kwargs):
        raise AssertionError("OAuth token endpoint should not be called in manual_bearer mode")

    class _Response:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"conversations": []}

    captured_headers = {}

    def _request(method, url, headers=None, params=None, json=None, timeout=None):
        captured_headers.update(headers or {})
        return _Response()

    monkeypatch.setattr(requests, "post", _oauth_call)
    monkeypatch.setattr(requests, "request", _request)

    payload = client._request_json(
        method="GET",
        path="/api/v2/analytics/botflows/flow/divisions/reportingturns",
        params={"interval": "2026-04-19T00:00:00.000Z/2026-04-20T00:00:00.000Z"},
    )

    assert payload == {"conversations": []}
    assert captured_headers.get("Authorization") == "Bearer manual-token"


def test_request_json_honors_stop_requested_before_request(monkeypatch):
    client = GenesysAnalyticsJourneyClient(
        region="usw2.pure.cloud",
        client_id="unused",
        client_secret="unused",
        auth_mode=ANALYTICS_AUTH_MODE_MANUAL_BEARER,
        manual_bearer_token="manual-token",
    )

    with pytest.raises(GenesysAnalyticsJourneyError, match="interrupted by stop request"):
        client._request_json(
            method="GET",
            path="/api/v2/analytics/botflows/flow/divisions/reportingturns",
            stop_requested=lambda: True,
        )


def test_filter_conversation_ids_by_language_excludes_missing_and_mismatched():
    rows_by_conversation = {
        "conv-en": [
            {"conversation": {"id": "conv-en"}, "language": "en"},
        ],
        "conv-unknown": [
            {"conversation": {"id": "conv-unknown"}, "userInput": "hello"},
        ],
        "conv-fr": [
            {"conversation": {"id": "conv-fr"}, "language": "fr"},
        ],
    }

    selected_ids, stats = GenesysAnalyticsJourneyClient.filter_conversation_ids_by_language(
        rows_by_conversation,
        ["conv-en", "conv-unknown", "conv-fr"],
        "en",
    )

    assert selected_ids == ["conv-en"]
    assert stats["language_filter"] == "en"
    assert stats["eligible_conversations"] == 1
    assert stats["selected_conversations"] == 1
    assert stats["excluded_missing_language_conversations"] == 1
    assert stats["excluded_mismatched_conversations"] == 1


def test_filter_conversation_ids_by_language_excludes_conflicting_conversation():
    rows_by_conversation = {
        "conv-mixed": [
            {"conversation": {"id": "conv-mixed"}, "language": "en"},
            {"conversation": {"id": "conv-mixed"}, "language": "fr"},
        ],
        "conv-en": [
            {"conversation": {"id": "conv-en"}, "language": "en-US"},
        ],
    }

    selected_ids, stats = GenesysAnalyticsJourneyClient.filter_conversation_ids_by_language(
        rows_by_conversation,
        ["conv-mixed", "conv-en"],
        "en",
    )

    assert selected_ids == ["conv-en"]
    assert stats["eligible_conversations"] == 1
    assert stats["selected_conversations"] == 1
    assert stats["excluded_mismatched_conversations"] == 1
    assert stats["excluded_missing_language_conversations"] == 0


def test_filter_rows_by_language_uses_conversation_level_eligibility():
    rows = [
        {
            "conversation": {"id": "11111111-1111-1111-1111-111111111111"},
            "language": "en",
            "userInput": "hello",
        },
        {
            "conversation": {"id": "11111111-1111-1111-1111-111111111111"},
            "botPrompts": ["hi"],
        },
        {
            "conversation": {"id": "22222222-2222-2222-2222-222222222222"},
            "language": "fr",
            "userInput": "bonjour",
        },
        {
            "conversation": {"id": "33333333-3333-3333-3333-333333333333"},
            "userInput": "no metadata",
        },
    ]

    filtered_rows, stats = GenesysAnalyticsJourneyClient.filter_rows_by_language(rows, "en")

    assert [row["conversation"]["id"] for row in filtered_rows] == [
        "11111111-1111-1111-1111-111111111111",
        "11111111-1111-1111-1111-111111111111",
    ]
    assert stats["eligible_conversations"] == 1
    assert stats["selected_conversations"] == 1
    assert stats["excluded_missing_language_conversations"] == 1
    assert stats["excluded_mismatched_conversations"] == 1
