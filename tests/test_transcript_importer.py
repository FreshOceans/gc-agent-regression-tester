"""Unit tests for transcript conversation-ID import helpers."""

import pytest

from src.transcript_importer import (
    build_last_24h_interval,
    build_transcript_seeder_payload,
    dedupe_and_cap_conversation_ids,
    parse_conversation_ids_from_file,
    parse_conversation_ids_from_paste,
    parse_filter_json,
)


def test_parse_conversation_ids_from_paste_extracts_uuid_values():
    text = """
    Here are IDs:
    11111111-2222-4333-8444-555555555555
    random text
    66666666-7777-4888-9999-aaaaaaaaaaaa
    """
    ids = parse_conversation_ids_from_paste(text)
    assert ids == [
        "11111111-2222-4333-8444-555555555555",
        "66666666-7777-4888-9999-aaaaaaaaaaaa",
    ]


def test_parse_conversation_ids_from_file_csv_header_supported():
    content = (
        "conversation_id,notes\n"
        "11111111-2222-4333-8444-555555555555,first\n"
        "66666666-7777-4888-9999-aaaaaaaaaaaa,second\n"
    )
    ids = parse_conversation_ids_from_file(content=content, filename="ids.csv")
    assert ids == [
        "11111111-2222-4333-8444-555555555555",
        "66666666-7777-4888-9999-aaaaaaaaaaaa",
    ]


def test_dedupe_and_cap_conversation_ids_preserves_order():
    ids = [
        "11111111-2222-4333-8444-555555555555",
        "11111111-2222-4333-8444-555555555555",
        "66666666-7777-4888-9999-aaaaaaaaaaaa",
    ]
    assert dedupe_and_cap_conversation_ids(ids, max_ids=1) == [
        "11111111-2222-4333-8444-555555555555"
    ]
    assert dedupe_and_cap_conversation_ids(ids, max_ids=5) == [
        "11111111-2222-4333-8444-555555555555",
        "66666666-7777-4888-9999-aaaaaaaaaaaa",
    ]


def test_parse_filter_json_requires_object():
    assert parse_filter_json("{}") == {}
    with pytest.raises(ValueError):
        parse_filter_json("[]")


def test_build_last_24h_interval_contains_separator():
    interval = build_last_24h_interval()
    assert "/" in interval


def test_build_transcript_seeder_payload_maps_roles():
    transcripts = [
        {
            "conversation_id": "11111111-2222-4333-8444-555555555555",
            "messages": [
                {"role": "customer", "text": "I want to cancel", "timestamp": "2026-04-19T01:00:00Z"},
                {"role": "agent", "text": "Sure", "timestamp": "2026-04-19T01:00:01Z"},
                {"role": "system", "text": "typing", "timestamp": "2026-04-19T01:00:02Z"},
            ],
        }
    ]
    payload = build_transcript_seeder_payload(transcripts)
    assert payload["conversations"][0]["conversation_id"] == "11111111-2222-4333-8444-555555555555"
    assert payload["conversations"][0]["messages"][0]["speaker"] == "customer"
    assert payload["conversations"][0]["messages"][1]["speaker"] == "agent"
    assert payload["conversations"][0]["messages"][2]["speaker"] == "system"

