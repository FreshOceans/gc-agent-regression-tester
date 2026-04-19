"""Unit tests for transcript seeding helpers."""

import json

import pytest

from src.transcript_seeder import (
    TranscriptSeedError,
    seed_test_suite_from_transcript,
    seed_test_suite_from_transcript_with_diagnostics,
)


class TestSeedTestSuiteFromTranscript:
    def test_seeds_from_structured_json_with_explicit_intent(self):
        content = json.dumps(
            {
                "messages": [
                    {
                        "speaker": "Customer",
                        "text": "I want to cancel my booking",
                        "detected_intent": "flight_cancel",
                    },
                    {
                        "speaker": "Agent",
                        "text": "Sure, I can help with that.",
                    },
                ]
            }
        )
        suite = seed_test_suite_from_transcript(
            content,
            format_hint="json",
            suite_name="Seeded JSON Suite",
        )

        assert suite.name == "Seeded JSON Suite"
        assert len(suite.scenarios) == 1
        scenario = suite.scenarios[0]
        assert scenario.first_message == "I want to cancel my booking"
        assert scenario.expected_intent == "flight_cancel"
        assert scenario.attempts == 1

    def test_structured_json_nested_payload_fields_are_supported(self):
        content = json.dumps(
            {
                "conversation": {
                    "events": [
                        {
                            "participant": {"role": "customer"},
                            "payload": {
                                "message": {
                                    "body": "I need help with my flight status"
                                }
                            },
                            "nlu": {"intentName": "flight_status"},
                        },
                        {
                            "participant": {"role": "assistant"},
                            "payload": {"message": {"body": "Sure, one moment."}},
                        },
                    ]
                }
            }
        )

        suite = seed_test_suite_from_transcript(content, format_hint="json")

        assert len(suite.scenarios) == 1
        assert suite.scenarios[0].first_message == "I need help with my flight status"
        assert suite.scenarios[0].expected_intent == "flight_status"

    def test_guideline_pricing_is_behavior_mode(self):
        content = json.dumps(
            {
                "messages": [
                    {
                        "speaker": "Customer",
                        "text": "How much are baggage fees",
                        "detected_intent": "guideline",
                    }
                ]
            }
        )
        suite = seed_test_suite_from_transcript(content, format_hint="json")

        scenario = suite.scenarios[0]
        assert scenario.expected_intent is None
        assert "does not provide specific baggage fee" in scenario.goal

    def test_vacation_hotel_text_infers_branch_intent(self):
        content = """
        Customer: I booked a vacation package with flight and hotel
        Agent: I can help with that.
        """
        suite = seed_test_suite_from_transcript(content, format_hint="text")

        assert len(suite.scenarios) == 1
        assert suite.scenarios[0].expected_intent == "vacation_flight_and_hotel"

    def test_csv_header_based_transcript_is_supported(self):
        content = """speaker,message,detected_intent
customer,I want to cancel my booking,flight_cancel
agent,Hello there,
"""

        suite = seed_test_suite_from_transcript(content, format_hint="csv")

        assert len(suite.scenarios) == 1
        assert suite.scenarios[0].first_message == "I want to cancel my booking"
        assert suite.scenarios[0].expected_intent == "flight_cancel"

    def test_tsv_fallback_column_mode_is_supported(self):
        content = """customer\tI need help with my booking\tflight_change
agent\tSure\t
"""

        suite = seed_test_suite_from_transcript(content, format_hint="tsv")

        assert len(suite.scenarios) == 1
        assert suite.scenarios[0].first_message == "I need help with my booking"
        assert suite.scenarios[0].expected_intent == "flight_change"

    def test_dedup_and_noise_filtering_are_applied(self):
        content = """
        Customer: conversation_id: 123
        Customer: I want to cancel my booking!!!
        Customer: i want to cancel my booking
        Agent: Hi, I'm Ava, WestJet's virtual assistant. How may I help you today?
        """

        suite, diagnostics = seed_test_suite_from_transcript_with_diagnostics(
            content,
            format_hint="text",
        )

        assert len(suite.scenarios) == 1
        assert suite.scenarios[0].first_message == "I want to cancel my booking!!!"
        assert diagnostics.skipped_messages >= 1
        assert diagnostics.skipped_duplicate >= 1

    def test_max_scenarios_limit_is_respected(self):
        content = """
        Customer: First question
        Customer: Second question
        Customer: Third question
        """
        suite = seed_test_suite_from_transcript(
            content,
            format_hint="text",
            max_scenarios=2,
        )
        assert len(suite.scenarios) == 2

    def test_multi_conversation_payload_order_and_limit_are_deterministic(self):
        content = json.dumps(
            {
                "conversations": [
                    {
                        "messages": [
                            {"speaker": "customer", "text": "First request"},
                            {"speaker": "agent", "text": "Response"},
                        ]
                    },
                    {
                        "messages": [
                            {"speaker": "customer", "text": "Second request"},
                            {"speaker": "customer", "text": "Third request"},
                        ]
                    },
                ]
            }
        )

        suite = seed_test_suite_from_transcript(
            content,
            format_hint="json",
            max_scenarios=2,
        )

        assert [scenario.first_message for scenario in suite.scenarios] == [
            "First request",
            "Second request",
        ]

    def test_raises_when_no_customer_utterance_found(self):
        content = """
        Agent: Hello there
        Assistant: How can I help?
        """
        with pytest.raises(TranscriptSeedError, match="No customer/user utterances"):
            seed_test_suite_from_transcript(content, format_hint="text")
