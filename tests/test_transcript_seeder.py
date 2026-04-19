"""Unit tests for transcript seeding helpers."""

import json

import pytest

from src.transcript_seeder import TranscriptSeedError, seed_test_suite_from_transcript


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

    def test_raises_when_no_customer_utterance_found(self):
        content = """
        Agent: Hello there
        Assistant: How can I help?
        """
        with pytest.raises(TranscriptSeedError, match="No customer/user utterances"):
            seed_test_suite_from_transcript(content, format_hint="text")

