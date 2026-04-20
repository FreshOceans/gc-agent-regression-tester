"""Unit tests for the analytics reporting-turns client."""

from src.genesys_analytics_journey_client import GenesysAnalyticsJourneyClient


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

    def _fake_fetch_page(*, bot_flow_id, interval, page_size, page_number, divisions, language_filter, extra_params):
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

    result = client.fetch_conversation_units(
        bot_flow_id="flow-id",
        interval="2026-04-19T00:00:00.000Z/2026-04-20T00:00:00.000Z",
        page_size=2,
        max_conversations=3,
    )

    assert result["page_count"] == 2
    assert calls == [1, 2]
    ids = [entry["conversation_id"] for entry in result["conversations"]]
    assert ids == [
        "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
        "cccccccc-cccc-cccc-cccc-cccccccccccc",
    ]
