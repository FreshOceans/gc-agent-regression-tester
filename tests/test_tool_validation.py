"""Unit tests for tool event parsing and validation rules."""

from datetime import datetime, timezone
from typing import Optional

from src.models import Message, MessageRole, ToolRuleExpression, ToolValidationConfig
from src.tool_validation import (
    dedupe_tool_events,
    evaluate_tool_validation,
    parse_tool_events_from_attribute_map,
    parse_tool_events_from_markers,
)


def _tool_config(loose_rule: dict, strict_rule: Optional[dict] = None) -> ToolValidationConfig:
    payload = {"loose_rule": loose_rule}
    if strict_rule is not None:
        payload["strict_rule"] = strict_rule
    return ToolValidationConfig.model_validate(payload)


def test_parse_tool_events_from_attribute_map_json_payload():
    attributes = {
        "rth_tool_events": [
            {"tool": "flight_lookup", "status": "success"},
            {"tool": "flight_change_priority", "status": "success"},
        ]
    }
    events = parse_tool_events_from_attribute_map(
        attributes,
        attribute_keys=["rth_tool_events", "tool_events"],
    )
    assert [event.name for event in events] == [
        "flight_lookup",
        "flight_change_priority",
    ]
    assert events[0].source == "participant_attribute"


def test_parse_tool_events_from_markers_json_contract():
    conversation = [
        Message(
            role=MessageRole.AGENT,
            content='tool_event: {"tool":"flight_lookup","status":"success"}',
            timestamp=datetime(2026, 4, 19, 0, 0, tzinfo=timezone.utc),
        ),
        Message(
            role=MessageRole.AGENT,
            content='tool_event: {"tool":"flight_change_priority","status":"success"}',
            timestamp=datetime(2026, 4, 19, 0, 1, tzinfo=timezone.utc),
        ),
    ]
    events = parse_tool_events_from_markers(conversation, marker_prefixes=["tool_event:"])
    assert [event.name for event in events] == [
        "flight_lookup",
        "flight_change_priority",
    ]
    assert events[0].source == "response_marker"


def test_dedupe_tool_events_collapses_duplicate_signature():
    attributes = {
        "rth_tool_events": [
            {"tool": "flight_lookup", "status": "success"},
            {"tool": "flight_lookup", "status": "success"},
        ]
    }
    events = parse_tool_events_from_attribute_map(
        attributes,
        attribute_keys=["rth_tool_events"],
    )
    deduped = dedupe_tool_events(events)
    assert len(events) == 2
    assert len(deduped) == 1


def test_evaluate_tool_validation_loose_and_strict():
    config = _tool_config(
        loose_rule={
            "all": [
                {"tool": "flight_lookup"},
                {
                    "any": [
                        {"tool": "flight_change_priority"},
                        {"tool": "flight_change_standard"},
                    ]
                },
            ]
        },
        strict_rule={
            "in_order": [
                {"tool": "flight_lookup"},
                {
                    "any": [
                        {"tool": "flight_change_priority"},
                        {"tool": "flight_change_standard"},
                    ]
                },
            ]
        },
    )
    events = parse_tool_events_from_attribute_map(
        {
            "tool_events": [
                {"tool": "flight_lookup", "status": "success"},
                {"tool": "flight_change_priority", "status": "success"},
            ]
        },
        attribute_keys=["tool_events"],
    )

    result = evaluate_tool_validation(config, events)
    assert result.loose_pass is True
    assert result.strict_pass is True
    assert result.missing_signal is False


def test_evaluate_tool_validation_missing_signal_is_hard_fail():
    config = _tool_config(loose_rule={"tool": "flight_lookup"})
    result = evaluate_tool_validation(config, [])
    assert result.loose_pass is False
    assert result.missing_signal is True
    assert "No tool events captured" in result.loose_fail_reasons[0]


def test_evaluate_tool_validation_detects_order_violation():
    config = _tool_config(
        loose_rule={
            "all": [{"tool": "flight_lookup"}, {"tool": "flight_change_priority"}]
        },
        strict_rule={
            "in_order": [
                {"tool": "flight_lookup"},
                {"tool": "flight_change_priority"},
            ]
        },
    )
    events = parse_tool_events_from_attribute_map(
        {
            "rth_tool_events": [
                {"tool": "flight_change_priority", "status": "success"},
                {"tool": "flight_lookup", "status": "success"},
            ]
        },
        attribute_keys=["rth_tool_events"],
    )
    result = evaluate_tool_validation(config, events)

    assert result.loose_pass is True
    assert result.strict_pass is False
    assert result.order_violations


def test_tool_rule_expression_rejects_multi_operator_leaf():
    payload = {"tool": "flight_lookup", "any": [{"tool": "flight_change_priority"}]}
    try:
        ToolRuleExpression.model_validate(payload)
        raised = False
    except Exception:
        raised = True
    assert raised is True
