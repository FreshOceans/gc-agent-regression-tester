"""Tests for the deterministic WestJet scripted suite generator."""

from pathlib import Path

from src.config_loader import load_test_suite
from src.models import TestScenario, TestSuite
from src.westjet_scripted_suite import (
    build_westjet_7_turn_suite,
    write_westjet_7_turn_suite,
)


def _source_suite() -> TestSuite:
    scenarios = []
    for index in range(4):
        scenarios.append(
            TestScenario(
                name=f"Speak {index}",
                persona="Traveler",
                goal="Speak to an agent",
                first_message=f"Speak utterance {index}",
                expected_intent="speak_to_agent",
            )
        )
        scenarios.append(
            TestScenario(
                name=f"Knowledge {index}",
                persona="Traveler",
                goal="Ask a knowledge question",
                first_message=f"Knowledge utterance {index}",
                expected_intent="knowledge",
            )
        )
        scenarios.append(
            TestScenario(
                name=f"Priority {index}",
                persona="Traveler",
                goal="Ask about flight priority",
                first_message=f"Priority utterance {index}",
                expected_intent="flight_priority_change",
            )
        )
    scenarios.extend(
        [
            TestScenario(
                name="Vacation flight only",
                persona="Traveler",
                goal="Vacation branch",
                first_message="Vacation flight only utterance",
                expected_intent="vacation_inquiry_flight_only",
            ),
            TestScenario(
                name="Vacation flight hotel A",
                persona="Traveler",
                goal="Vacation branch",
                first_message="Vacation flight and hotel utterance A",
                expected_intent="vacation_flight_and_hotel",
            ),
            TestScenario(
                name="Vacation flight hotel B",
                persona="Traveler",
                goal="Vacation branch",
                first_message="Vacation flight and hotel utterance B",
                expected_intent="vacation_flight_and_hotel",
            ),
        ]
    )
    return TestSuite(name="Source Suite", language="en", scenarios=scenarios)


def test_build_westjet_7_turn_suite_has_expected_shape():
    suite = build_westjet_7_turn_suite(_source_suite())

    assert suite.name == "WestJet 7-Turn Mixed Flow Regression Suite"
    assert suite.language == "en"
    assert len(suite.scenarios) == 30
    assert all(scenario.attempts == 1 for scenario in suite.scenarios)
    assert all(
        scenario.language_selection_message == "english"
        for scenario in suite.scenarios
    )
    assert all(scenario.first_message == "Hi how are you" for scenario in suite.scenarios)
    assert all(
        scenario.scripted_user_turns is not None
        and len(scenario.scripted_user_turns) == 6
        for scenario in suite.scenarios
    )
    assert all(scenario.expected_intent is None for scenario in suite.scenarios)
    assert all(
        scenario.scripted_final_expected_intent is None for scenario in suite.scenarios
    )
    assert all(
        scenario.scripted_user_turns[0] == "I have a question"
        and scenario.scripted_user_turns[1] == "Can you help me with my question"
        and scenario.scripted_user_turns[2] == "I need help with an issue"
        for scenario in suite.scenarios
    )

    flight_priority = [
        scenario for scenario in suite.scenarios if scenario.scripted_user_turns[5] in {"yes", "no"}
    ]
    vacation = [
        scenario
        for scenario in suite.scenarios
        if scenario.scripted_user_turns[5] in {"flight only", "flight and hotel"}
    ]
    assert len(flight_priority) == 15
    assert len(vacation) == 15

    assert (
        sum(
            scenario.scripted_user_turns[5] == "yes"
            for scenario in flight_priority
        )
        == 8
    )
    assert (
        sum(
            scenario.scripted_user_turns[5] == "no"
            for scenario in flight_priority
        )
        == 7
    )
    assert (
        sum(
            scenario.scripted_user_turns[5] == "flight and hotel"
            for scenario in vacation
        )
        == 8
    )
    assert (
        sum(
            scenario.scripted_user_turns[5] == "flight only"
            for scenario in vacation
        )
        == 7
    )


def test_write_westjet_7_turn_suite_round_trips_through_loader(tmp_path: Path):
    source_path = tmp_path / "source.yaml"
    output_path = tmp_path / "generated.yaml"
    source_path.write_text(
        "\n".join(
            [
                "name: Source Suite",
                "language: en",
                "scenarios:",
                "  - name: Speak 1",
                "    persona: Traveler",
                "    goal: Speak to an agent",
                "    first_message: Speak utterance 1",
                "    expected_intent: speak_to_agent",
                "  - name: Speak 2",
                "    persona: Traveler",
                "    goal: Speak to an agent",
                "    first_message: Speak utterance 2",
                "    expected_intent: speak_to_agent",
                "  - name: Speak 3",
                "    persona: Traveler",
                "    goal: Speak to an agent",
                "    first_message: Speak utterance 3",
                "    expected_intent: speak_to_agent",
                "  - name: Speak 4",
                "    persona: Traveler",
                "    goal: Speak to an agent",
                "    first_message: Speak utterance 4",
                "    expected_intent: speak_to_agent",
                "  - name: Knowledge 1",
                "    persona: Traveler",
                "    goal: Knowledge",
                "    first_message: Knowledge utterance 1",
                "    expected_intent: knowledge",
                "  - name: Knowledge 2",
                "    persona: Traveler",
                "    goal: Knowledge",
                "    first_message: Knowledge utterance 2",
                "    expected_intent: knowledge",
                "  - name: Knowledge 3",
                "    persona: Traveler",
                "    goal: Knowledge",
                "    first_message: Knowledge utterance 3",
                "    expected_intent: knowledge",
                "  - name: Knowledge 4",
                "    persona: Traveler",
                "    goal: Knowledge",
                "    first_message: Knowledge utterance 4",
                "    expected_intent: knowledge",
                "  - name: Priority 1",
                "    persona: Traveler",
                "    goal: Priority",
                "    first_message: Priority utterance 1",
                "    expected_intent: flight_priority_change",
                "  - name: Priority 2",
                "    persona: Traveler",
                "    goal: Priority",
                "    first_message: Priority utterance 2",
                "    expected_intent: flight_priority_change",
                "  - name: Priority 3",
                "    persona: Traveler",
                "    goal: Priority",
                "    first_message: Priority utterance 3",
                "    expected_intent: flight_priority_change",
                "  - name: Priority 4",
                "    persona: Traveler",
                "    goal: Priority",
                "    first_message: Priority utterance 4",
                "    expected_intent: flight_priority_change",
                "  - name: Vacation Only",
                "    persona: Traveler",
                "    goal: Vacation",
                "    first_message: Vacation flight only utterance",
                "    expected_intent: vacation_inquiry_flight_only",
                "  - name: Vacation Hotel A",
                "    persona: Traveler",
                "    goal: Vacation",
                "    first_message: Vacation flight and hotel utterance A",
                "    expected_intent: vacation_flight_and_hotel",
                "  - name: Vacation Hotel B",
                "    persona: Traveler",
                "    goal: Vacation",
                "    first_message: Vacation flight and hotel utterance B",
                "    expected_intent: vacation_flight_and_hotel",
            ]
        ),
        encoding="utf-8",
    )

    generated = write_westjet_7_turn_suite(source_path, output_path)
    reloaded = load_test_suite(str(output_path))

    assert generated.name == reloaded.name
    assert len(reloaded.scenarios) == 30
    assert reloaded.scenarios[0].scripted_user_turns is not None
    assert reloaded.scenarios[0].scripted_user_turns[1] == "Can you help me with my question"
    assert reloaded.scenarios[0].scripted_final_expected_intent is None
