"""Tests for Gemma-powered Suite Builder generation."""

from pathlib import Path

import pytest

from src.config_loader import load_test_suite_from_string
from src.suite_builder import (
    build_suite_builder_description_request,
    build_suite_builder_request,
    distribute_scenarios,
    generate_suite_with_gemma,
    infer_intents_from_description,
    parse_bulk_intents,
    save_generated_suite_yaml,
)


def _chat_response(model, messages):
    del model, messages
    return (
        '{"scenarios": ['
        '{"first_message": "I need to cancel my trip", "scripted_user_turns": []},'
        '{"first_message": "Cancel my booking please", "scripted_user_turns": []}'
        ']}'
    )


def test_parse_bulk_intents_accepts_yaml_object_with_intents():
    raw = parse_bulk_intents(
        """
        intents:
          - id: flight_cancel
            description: User wants to cancel a booking
            examples:
              - cancel my booking
            avoid:
              - flight status
        """
    )

    assert raw[0]["id"] == "flight_cancel"
    assert raw[0]["description"] == "User wants to cancel a booking"


def test_request_validation_rejects_duplicate_intents():
    with pytest.raises(ValueError, match="Duplicate intent id"):
        build_suite_builder_request(
            suite_name="Generated",
            model="gemma4:e4b",
            language="en",
            scenario_count="2",
            attempts="1",
            user_turn_length="1",
            include_language_selection=False,
            intents=[
                {"id": "flight_cancel", "description": "Cancel"},
                {"id": "flight_cancel", "description": "Cancel duplicate"},
            ],
        )


def test_distribute_scenarios_evenly_with_remainder_in_input_order():
    assert distribute_scenarios(8, 3) == [3, 3, 2]


def test_description_request_rejects_blank_suite_description():
    with pytest.raises(ValueError, match="Suite description is required"):
        build_suite_builder_description_request(
            suite_name="Generated",
            model="gemma4:e4b",
            language="en",
            scenario_count="5",
            attempts="1",
            user_turn_length="1",
            include_language_selection=False,
            suite_description=" ",
            inferred_intent_count="3",
        )


def test_description_request_rejects_intent_count_above_scenarios():
    with pytest.raises(ValueError, match="cannot exceed total scenarios"):
        build_suite_builder_description_request(
            suite_name="Generated",
            model="gemma4:e4b",
            language="en",
            scenario_count="2",
            attempts="1",
            user_turn_length="1",
            include_language_selection=False,
            suite_description="Build a support suite.",
            inferred_intent_count="3",
        )


def test_description_request_rejects_intent_count_out_of_bounds():
    with pytest.raises(ValueError, match="between 1 and 50"):
        build_suite_builder_description_request(
            suite_name="Generated",
            model="gemma4:e4b",
            language="en",
            scenario_count="100",
            attempts="1",
            user_turn_length="1",
            include_language_selection=False,
            suite_description="Build a support suite.",
            inferred_intent_count="51",
        )


def test_infer_intents_from_description_parses_valid_plan():
    request = build_suite_builder_description_request(
        suite_name="Generated Support Suite",
        model="gemma4:e4b",
        language="en",
        scenario_count="4",
        attempts="1",
        user_turn_length="1",
        include_language_selection=False,
        suite_description="Cover cancellations and baggage help.",
        inferred_intent_count="2",
    )

    result = infer_intents_from_description(
        request,
        ollama_base_url="http://localhost:11434",
        chat_callable=lambda model, messages: (
            '{"intents": ['
            '{"id": "flight cancel", "description": "User wants to cancel", "examples": ["cancel my flight"], "avoid": ["flight status"]},'
            '{"id": "baggage_help", "description": "User needs baggage support", "examples": ["lost bag"], "avoid": []}'
            ']}'
        ),
    )

    assert [intent.id for intent in result.intents] == ["flight_cancel", "baggage_help"]
    assert result.intents[0].description == "User wants to cancel"
    assert result.intents[0].examples == ["cancel my flight"]
    assert result.diagnostics["mode"] == "description_only"
    assert result.diagnostics["retry_used"] is False


def test_infer_intents_from_description_retries_on_malformed_output():
    calls = []

    def chat_response(model, messages):
        del model
        calls.append(messages)
        if len(calls) == 1:
            return '{"bad": []}'
        return (
            '{"intents": ['
            '{"id": "refund status", "description": "User asks about refund status", "examples": ["where is my refund"], "avoid": []}'
            ']}'
        )

    request = build_suite_builder_description_request(
        suite_name="Generated Support Suite",
        model="gemma4:e4b",
        language="es",
        scenario_count="1",
        attempts="1",
        user_turn_length="2",
        include_language_selection=True,
        suite_description="Cover refund status.",
        inferred_intent_count="1",
    )

    result = infer_intents_from_description(
        request,
        ollama_base_url="http://localhost:11434",
        chat_callable=chat_response,
    )

    assert len(calls) == 2
    assert result.intents[0].id == "refund_status"
    assert result.diagnostics["retry_used"] is True


def test_infer_intents_from_description_errors_after_retry():
    request = build_suite_builder_description_request(
        suite_name="Generated Support Suite",
        model="gemma4:e4b",
        language="en",
        scenario_count="2",
        attempts="1",
        user_turn_length="1",
        include_language_selection=False,
        suite_description="Cover two support intents.",
        inferred_intent_count="2",
    )

    with pytest.raises(ValueError, match="Could not infer a valid intent plan"):
        infer_intents_from_description(
            request,
            ollama_base_url="http://localhost:11434",
            chat_callable=lambda model, messages: '{"intents": []}',
        )


def test_infer_intents_from_description_rejects_duplicate_ids():
    request = build_suite_builder_description_request(
        suite_name="Generated Support Suite",
        model="gemma4:e4b",
        language="fr-CA",
        scenario_count="2",
        attempts="1",
        user_turn_length="1",
        include_language_selection=False,
        suite_description="Cover cancellations.",
        inferred_intent_count="2",
    )

    with pytest.raises(ValueError, match="Could not infer a valid intent plan"):
        infer_intents_from_description(
            request,
            ollama_base_url="http://localhost:11434",
            chat_callable=lambda model, messages: (
                '{"intents": ['
                '{"id": "Annulation Vol", "description": "Cancel", "examples": ["annuler"], "avoid": []},'
                '{"id": "annulation_vol", "description": "Cancel duplicate", "examples": ["annuler"], "avoid": []}'
                ']}'
            ),
        )


def test_generate_single_turn_suite_uses_expected_intent():
    request = build_suite_builder_request(
        suite_name="Generated Suite",
        model="gemma4:e4b",
        language="en",
        scenario_count="2",
        attempts="3",
        user_turn_length="1",
        include_language_selection=False,
        intents=[{"id": "flight_cancel", "description": "User wants to cancel"}],
    )

    result = generate_suite_with_gemma(
        request,
        ollama_base_url="http://localhost:11434",
        chat_callable=_chat_response,
    )

    assert len(result.suite.scenarios) == 2
    assert result.suite.scenarios[0].expected_intent == "flight_cancel"
    assert result.suite.scenarios[0].scripted_final_expected_intent is None
    assert result.suite.scenarios[0].attempts == 3
    reloaded = load_test_suite_from_string(result.suite_yaml, "yaml")
    assert reloaded.scenarios[1].expected_intent == "flight_cancel"


def test_generate_multi_turn_suite_uses_final_intent_only():
    def chat_response(model, messages):
        del model, messages
        return (
            '{"scenarios": ['
            '{"first_message": "I need help", "scripted_user_turns": ["with my booking", "yes"]}'
            ']}'
        )

    request = build_suite_builder_request(
        suite_name="Generated Scripted Suite",
        model="gemma4:31b",
        language="fr-CA",
        scenario_count="1",
        attempts="1",
        user_turn_length="3",
        include_language_selection=True,
        intents=[{"id": "flight_change_priority_within_72_hours", "description": "Change soon"}],
    )

    result = generate_suite_with_gemma(
        request,
        ollama_base_url="http://localhost:11434",
        chat_callable=chat_response,
    )
    scenario = result.suite.scenarios[0]

    assert scenario.expected_intent is None
    assert scenario.scripted_final_expected_intent == "flight_change_priority_within_72_hours"
    assert scenario.scripted_user_turns == ["with my booking", "yes"]
    assert scenario.language_selection_message == "francais"


def test_short_model_output_is_filled_with_warning():
    def short_response(model, messages):
        del model, messages
        return '{"scenarios": []}'

    request = build_suite_builder_request(
        suite_name="Generated Suite",
        model="gemma4:e4b",
        language="es",
        scenario_count="1",
        attempts="1",
        user_turn_length="1",
        include_language_selection=False,
        intents=[{"id": "flight_cancel", "description": "User wants to cancel"}],
    )

    result = generate_suite_with_gemma(
        request,
        ollama_base_url="http://localhost:11434",
        chat_callable=short_response,
    )

    assert len(result.suite.scenarios) == 1
    assert result.suite.scenarios[0].expected_intent == "flight_cancel"
    assert result.warnings


def test_save_generated_suite_yaml_avoids_overwrite(tmp_path: Path):
    request = build_suite_builder_request(
        suite_name="Generated Suite",
        model="gemma4:e4b",
        language="en",
        scenario_count="1",
        attempts="1",
        user_turn_length="1",
        include_language_selection=False,
        intents=[{"id": "flight_cancel", "description": "User wants to cancel"}],
    )
    result = generate_suite_with_gemma(
        request,
        ollama_base_url="http://localhost:11434",
        chat_callable=lambda model, messages: '{"scenarios":[{"first_message":"cancel", "scripted_user_turns":[]}]}',
    )

    first = save_generated_suite_yaml(result.suite_yaml, output_dir=tmp_path)
    second = save_generated_suite_yaml(result.suite_yaml, output_dir=tmp_path)

    assert first.name == "generated_suite.yaml"
    assert second.name == "generated_suite_2.yaml"
