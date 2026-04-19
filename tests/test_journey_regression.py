"""Unit tests for journey-mode category resolution and transcript extraction."""

from src.journey_regression import (
    extract_journey_seed_candidates,
    infer_containment_from_payload_metadata,
    resolve_category_with_strategy,
    resolve_primary_categories,
)


def test_extract_journey_seed_candidates_from_conversations_messages():
    payload = {
        "conversations": [
            {
                "conversationId": "123",
                "participants": [{"purpose": "customer"}],
                "messages": [
                    {"speaker": "agent", "text": "Welcome"},
                    {"speaker": "customer", "text": "I need to cancel my booking"},
                ],
            }
        ]
    }

    candidates = extract_journey_seed_candidates(payload)
    assert len(candidates) == 1
    assert candidates[0]["first_customer_message"] == "I need to cancel my booking"
    assert candidates[0]["metadata_contained"] is True


def test_extract_journey_seed_candidates_from_transcript_phrases():
    payload = {
        "transcripts": [
            {
                "phrases": [
                    {"participantPurpose": "agent", "text": "Hello"},
                    {"participantPurpose": "customer", "text": "What are baggage fees?"},
                ]
            }
        ]
    }

    candidates = extract_journey_seed_candidates(payload)
    assert len(candidates) == 1
    assert candidates[0]["first_customer_message"] == "What are baggage fees?"


def test_resolve_category_with_rules_first_and_llm_fallback():
    categories = resolve_primary_categories()
    direct = resolve_category_with_strategy(
        "I need to cancel my booking",
        categories=categories,
        strategy="rules_first",
        llm_classifier=None,
    )
    assert direct["category"] == "flight_cancel"
    assert direct["source"] == "rules"

    llm = resolve_category_with_strategy(
        "This does not match rules",
        categories=categories,
        strategy="rules_first",
        llm_classifier=lambda message, cats: {
            "category": "flight_status",
            "confidence": 0.8,
            "explanation": "classifier",
        },
    )
    assert llm["category"] == "flight_status"
    assert llm["source"] == "llm"


def test_infer_containment_from_payload_metadata_detects_handoff():
    payload = {
        "participants": [
            {"purpose": "customer"},
            {"purpose": "agent", "userId": "agent-123"},
        ]
    }
    assert infer_containment_from_payload_metadata(payload) is False
