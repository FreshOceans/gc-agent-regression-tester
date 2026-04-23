"""Deterministic generators for synthetic WestJet scripted suites."""

from __future__ import annotations

from pathlib import Path
import random

import yaml

from .config_loader import load_test_suite
from .models import TestScenario, TestSuite

DEFAULT_SEED = 42
DEFAULT_SOURCE_SUITE_PATH = Path("local_suites/westjet_test_suite.yaml")
DEFAULT_OUTPUT_SUITE_PATH = Path("local_suites/westjet_test_suite_en_7_turn_30.yaml")

_SHARED_PERSONA = (
    "Traveler in the WestJet agentic flow who drifts across topics before "
    "resolving a final issue."
)
_SHARED_GOAL = (
    "Complete the full seven-turn scripted agentic flow after language "
    "selection without the conversation erroring out, stalling, or ending "
    "before the seventh user turn."
)


def _messages_for_expected_intents(
    suite: TestSuite,
    *,
    expected_intents: set[str],
) -> list[str]:
    messages: list[str] = []
    for scenario in suite.scenarios:
        if scenario.expected_intent not in expected_intents:
            continue
        first_message = str(scenario.first_message or "").strip()
        if first_message:
            messages.append(first_message)
    if not messages:
        joined = ", ".join(sorted(expected_intents))
        raise ValueError(f"No first_message pool found for intents: {joined}")
    return messages


def _shuffled_cycle(pool: list[str], *, rng: random.Random, count: int) -> list[str]:
    shuffled = list(pool)
    rng.shuffle(shuffled)
    return [shuffled[index % len(shuffled)] for index in range(count)]


def build_westjet_7_turn_suite(
    source_suite: TestSuite,
    *,
    seed: int = DEFAULT_SEED,
) -> TestSuite:
    """Build the deterministic 30-scenario scripted English WestJet suite."""

    rng = random.Random(seed)

    knowledge_pool = _messages_for_expected_intents(
        source_suite,
        expected_intents={"knowledge"},
    )
    flight_priority_pool = _messages_for_expected_intents(
        source_suite,
        expected_intents={"flight_priority_change"},
    )
    vacation_pool = _messages_for_expected_intents(
        source_suite,
        expected_intents={
            "vacation_inquiry_flight_only",
            "vacation_flight_and_hotel",
        },
    )

    knowledge_turns = _shuffled_cycle(knowledge_pool, rng=rng, count=30)
    flight_branch_turns = _shuffled_cycle(flight_priority_pool, rng=rng, count=15)
    vacation_branch_turns = _shuffled_cycle(vacation_pool, rng=rng, count=15)

    flight_resolutions = ["yes"] * 8 + ["no"] * 7
    vacation_resolutions = ["flight and hotel"] * 8 + ["flight only"] * 7
    rng.shuffle(flight_resolutions)
    rng.shuffle(vacation_resolutions)

    scenarios: list[TestScenario] = []
    for index in range(15):
        resolution = flight_resolutions[index]
        scenarios.append(
            TestScenario(
                name=f"7-turn mixed flow - Scenario {index + 1:02d}",
                persona=_SHARED_PERSONA,
                goal=_SHARED_GOAL,
                attempts=1,
                language_selection_message="english",
                first_message="Hi how are you",
                scripted_user_turns=[
                    "I have a question",
                    "Can you help me with my question",
                    "I need help with an issue",
                    knowledge_turns[index],
                    flight_branch_turns[index],
                    resolution,
                ],
            )
        )

    for index in range(15):
        absolute_index = index + 15
        resolution = vacation_resolutions[index]
        scenarios.append(
            TestScenario(
                name=f"7-turn mixed flow - Scenario {absolute_index + 1:02d}",
                persona=_SHARED_PERSONA,
                goal=_SHARED_GOAL,
                attempts=1,
                language_selection_message="english",
                first_message="Hi how are you",
                scripted_user_turns=[
                    "I have a question",
                    "Can you help me with my question",
                    "I need help with an issue",
                    knowledge_turns[absolute_index],
                    vacation_branch_turns[index],
                    resolution,
                ],
            )
        )

    return TestSuite(
        name="WestJet 7-Turn Mixed Flow Regression Suite",
        language="en",
        scenarios=scenarios,
    )


def write_westjet_7_turn_suite(
    source_path: str | Path = DEFAULT_SOURCE_SUITE_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_SUITE_PATH,
    *,
    seed: int = DEFAULT_SEED,
) -> TestSuite:
    """Generate and write the deterministic scripted suite to disk."""

    source_suite = load_test_suite(str(source_path))
    generated_suite = build_westjet_7_turn_suite(source_suite, seed=seed)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(
            generated_suite.model_dump(exclude_none=True),
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return generated_suite
