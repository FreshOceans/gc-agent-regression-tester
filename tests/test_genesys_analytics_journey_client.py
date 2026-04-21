"""Unit tests for the analytics reporting-turns client."""

import requests

from src.genesys_analytics_journey_client import GenesysAnalyticsJourneyClient
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
        observer=None,
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


def test_fetch_reporting_turns_page_posts_details_query_payload(monkeypatch):
    client = GenesysAnalyticsJourneyClient(
        region="usw2.pure.cloud",
        client_id="client-id",
        client_secret="client-secret",
        page_size_cap=100,
    )

    captured = {}

    def _fake_request_json(*, method, path, json_payload=None, params=None, observer=None, request_context=None):
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
        extra_params={"filter": {"type": "and", "predicates": []}},
    )

    assert payload == {"conversations": []}
    assert captured["method"] == "POST"
    assert captured["path"] == "/api/v2/analytics/conversations/details/query"
    assert captured["params"] is None
    assert captured["request_context"] == {"page_number": 2}
    assert captured["json_payload"]["interval"].startswith("2026-04-19T00:00:00.000Z")
    assert captured["json_payload"]["order"] == "asc"
    assert captured["json_payload"]["orderBy"] == "conversationStart"
    assert captured["json_payload"]["paging"]["pageSize"] == 100
    assert captured["json_payload"]["paging"]["pageNumber"] == 2
    assert captured["json_payload"]["filter"] == {"type": "and", "predicates": []}


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
        method="POST",
        path="/api/v2/analytics/conversations/details/query",
        json_payload={"interval": "2026-04-19T00:00:00.000Z/2026-04-20T00:00:00.000Z"},
    )

    assert payload == {"conversations": []}
    assert captured_headers.get("Authorization") == "Bearer manual-token"
