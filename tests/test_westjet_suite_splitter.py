"""Tests for splitting the English WestJet suite into smaller per-intent files."""

from pathlib import Path

from src.config_loader import load_test_suite
from src.models import TestScenario, TestSuite
from src.westjet_suite_splitter import (
    derive_group_key,
    split_westjet_suite_by_intent,
    write_split_westjet_suites,
)


def _source_suite() -> TestSuite:
    return TestSuite(
        name="WestJet Travel Agent Intent Regression Suite",
        language="en",
        scenarios=[
            TestScenario(
                name="flight_change - Utterance 01",
                persona="Traveler",
                goal="Change a flight",
                first_message="I need to change my booking",
                expected_intent="flight_change",
                attempts=1,
            ),
            TestScenario(
                name="flight_change - Utterance 02",
                persona="Traveler",
                goal="Change a flight",
                first_message="Can I move my flight to tomorrow?",
                expected_intent="flight_change",
                attempts=1,
            ),
            TestScenario(
                name="speak_to_agent - Utterance 01",
                persona="Traveler",
                goal="Speak to an agent",
                first_message="I need a live agent",
                expected_intent="speak_to_agent",
                attempts=1,
            ),
            TestScenario(
                name="guideline - Utterance 01",
                persona="Traveler",
                goal="Ask about bag pricing",
                first_message="How much are baggage fees",
                attempts=1,
            ),
        ],
    )


def test_derive_group_key_prefers_expected_intent():
    scenario = TestScenario(
        name="flight_change - Utterance 01",
        persona="Traveler",
        goal="Change a flight",
        first_message="I need to change my booking",
        expected_intent="flight_change",
        attempts=1,
    )
    assert derive_group_key(scenario) == "flight_change"


def test_derive_group_key_falls_back_to_name_prefix():
    scenario = TestScenario(
        name="guideline - Utterance 01",
        persona="Traveler",
        goal="Ask about bag pricing",
        first_message="How much are baggage fees",
        attempts=1,
    )
    assert derive_group_key(scenario) == "guideline"


def test_split_westjet_suite_by_intent_preserves_group_order_and_counts():
    split_suites = split_westjet_suite_by_intent(_source_suite())

    assert list(split_suites.keys()) == [
        "flight_change",
        "speak_to_agent",
        "guideline",
    ]
    assert len(split_suites["flight_change"].scenarios) == 2
    assert len(split_suites["speak_to_agent"].scenarios) == 1
    assert len(split_suites["guideline"].scenarios) == 1
    assert split_suites["flight_change"].scenarios[0].name == "flight_change - Utterance 01"
    assert split_suites["flight_change"].scenarios[1].name == "flight_change - Utterance 02"


def test_write_split_westjet_suites_round_trips(tmp_path: Path):
    source_path = tmp_path / "westjet_test_suite.yaml"
    source_path.write_text(
        "\n".join(
            [
                "name: WestJet Travel Agent Intent Regression Suite",
                "language: en",
                "scenarios:",
                "  - name: flight_change - Utterance 01",
                "    persona: Traveler",
                "    goal: Change a flight",
                "    first_message: I need to change my booking",
                "    expected_intent: flight_change",
                "    attempts: 1",
                "  - name: speak_to_agent - Utterance 01",
                "    persona: Traveler",
                "    goal: Speak to an agent",
                "    first_message: I need a live agent",
                "    expected_intent: speak_to_agent",
                "    attempts: 1",
                "  - name: guideline - Utterance 01",
                "    persona: Traveler",
                "    goal: Ask about bag pricing",
                "    first_message: How much are baggage fees",
                "    attempts: 1",
            ]
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "westjet_by_intent"
    written_paths = write_split_westjet_suites(source_path, output_dir)

    assert set(written_paths.keys()) == {"flight_change", "speak_to_agent", "guideline"}
    reloaded = load_test_suite(str(written_paths["flight_change"]))
    assert reloaded.name == "WestJet Travel Agent Intent Regression Suite - flight_change"
    assert len(reloaded.scenarios) == 1
    assert reloaded.scenarios[0].expected_intent == "flight_change"
