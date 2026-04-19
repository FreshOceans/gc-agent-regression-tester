"""Tool event parsing and validation helpers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from .models import (
    Message,
    MessageRole,
    ToolEvent,
    ToolRuleExpression,
    ToolValidationConfig,
    ToolValidationResult,
)


_EVENT_ID_KEYS = ("event_id", "eventId", "id", "sequence", "seq", "index")
_TOOL_NAME_KEYS = ("tool", "tool_name", "name", "data_action", "action")
_STATUS_KEYS = ("status", "result", "outcome")
_TIMESTAMP_KEYS = ("timestamp", "ts", "time", "executed_at", "created_at")


@dataclass
class RuleEvalResult:
    """Internal result payload for rule expression evaluation."""

    passed: bool
    reasons: list[str]
    matched_indices: list[int]
    matched_tools: set[str]
    missing_tools: set[str]
    order_violations: list[str]


def parse_tool_events_from_attribute_map(
    attributes: dict[str, Any],
    attribute_keys: list[str],
    *,
    source: str = "participant_attribute",
) -> list[ToolEvent]:
    """Extract normalized tool events from participant attributes."""
    if not isinstance(attributes, dict):
        return []
    if not attribute_keys:
        return []

    normalized_attributes = {
        str(key).strip().lower(): value
        for key, value in attributes.items()
        if str(key).strip()
    }
    events: list[ToolEvent] = []
    for key in attribute_keys:
        raw_value = normalized_attributes.get(key.strip().lower())
        if raw_value is None:
            continue
        events.extend(_parse_tool_event_payload(raw_value, source=source))
    return events


def parse_tool_events_from_markers(
    conversation: list[Message],
    marker_prefixes: list[str],
) -> list[ToolEvent]:
    """Extract normalized tool events from explicit marker messages."""
    prefixes = [prefix.strip().lower() for prefix in marker_prefixes if prefix.strip()]
    if not prefixes:
        return []

    events: list[ToolEvent] = []
    for msg in conversation:
        if msg.role != MessageRole.AGENT:
            continue
        text = (msg.content or "").strip()
        if not text:
            continue
        lines = text.splitlines() or [text]
        for raw_line in lines:
            line = raw_line.strip()
            lower = line.lower()
            for prefix in prefixes:
                index = lower.find(prefix)
                if index == -1:
                    continue
                payload_text = line[index + len(prefix) :].strip()
                if not payload_text:
                    continue
                parsed_payload = _parse_marker_payload(payload_text)
                if parsed_payload is None:
                    continue
                events.extend(
                    _parse_tool_event_payload(
                        parsed_payload,
                        source="response_marker",
                        default_timestamp=msg.timestamp,
                    )
                )
    return events


def dedupe_tool_events(events: list[ToolEvent]) -> list[ToolEvent]:
    """Deduplicate tool events deterministically while preserving order."""
    deduped: list[ToolEvent] = []
    seen: set[tuple[str, str, str, str]] = set()
    for event in events:
        timestamp_key = (
            event.timestamp.astimezone(timezone.utc).isoformat()
            if event.timestamp is not None
            else ""
        )
        event_id_key = _extract_event_id(event.raw_payload) if event.raw_payload else ""
        key = (
            event.name.strip().lower(),
            (event.status or "").strip().lower(),
            timestamp_key,
            event_id_key,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(event)
    return deduped


def evaluate_tool_validation(
    config: ToolValidationConfig,
    events: list[ToolEvent],
) -> ToolValidationResult:
    """Evaluate loose and strict tool validation outcomes."""
    if not events:
        strict_pass = False if config.strict_rule is not None else None
        strict_reasons = (
            ["No tool events captured, strict rule could not be evaluated."]
            if config.strict_rule is not None
            else []
        )
        return ToolValidationResult(
            loose_pass=False,
            strict_pass=strict_pass,
            missing_signal=True,
            loose_fail_reasons=[
                "No tool events captured from participant attributes or response markers."
            ],
            strict_fail_reasons=strict_reasons,
            missing_tools=_collect_tools_from_rule(config.loose_rule),
            matched_tools=[],
        )

    indexed_events = list(enumerate(events))

    loose_result = _evaluate_rule(
        config.loose_rule,
        indexed_events=indexed_events,
        min_index=0,
    )
    strict_result: Optional[RuleEvalResult] = None
    if config.strict_rule is not None:
        strict_result = _evaluate_rule(
            config.strict_rule,
            indexed_events=indexed_events,
            min_index=0,
        )

    order_violations = []
    if strict_result is not None and strict_result.order_violations:
        order_violations = strict_result.order_violations

    matched_tools = sorted(loose_result.matched_tools)
    if strict_result is not None:
        matched_tools = sorted(set(matched_tools).union(strict_result.matched_tools))

    missing_tools = sorted(loose_result.missing_tools)
    if strict_result is not None:
        missing_tools = sorted(set(missing_tools).union(strict_result.missing_tools))

    return ToolValidationResult(
        loose_pass=loose_result.passed,
        strict_pass=(strict_result.passed if strict_result is not None else None),
        missing_signal=False,
        loose_fail_reasons=loose_result.reasons,
        strict_fail_reasons=(strict_result.reasons if strict_result is not None else []),
        missing_tools=missing_tools,
        order_violations=order_violations,
        matched_tools=matched_tools,
    )


def _parse_marker_payload(payload_text: str) -> Optional[Any]:
    payload = payload_text.strip()
    if not payload:
        return None

    # Strict JSON marker (recommended): tool_event: {"tool":"x","status":"ok"}
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        pass

    # Allow a wrapped JSON snippet with trailing punctuation.
    json_match = re.search(r"(\{.*\})", payload)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            return None
    return None


def _parse_tool_event_payload(
    payload: Any,
    *,
    source: str,
    default_timestamp: Optional[datetime] = None,
) -> list[ToolEvent]:
    if payload is None:
        return []

    if isinstance(payload, str):
        text = payload.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            # For deterministic behavior, non-JSON strings are ignored.
            return []
        return _parse_tool_event_payload(
            parsed,
            source=source,
            default_timestamp=default_timestamp,
        )

    if isinstance(payload, list):
        events: list[ToolEvent] = []
        for item in payload:
            events.extend(
                _parse_tool_event_payload(
                    item,
                    source=source,
                    default_timestamp=default_timestamp,
                )
            )
        return events

    if isinstance(payload, dict):
        nested_events = payload.get("events")
        if isinstance(nested_events, list):
            events: list[ToolEvent] = []
            for item in nested_events:
                events.extend(
                    _parse_tool_event_payload(
                        item,
                        source=source,
                        default_timestamp=default_timestamp,
                    )
                )
            return events

        event = _normalize_tool_event(
            payload,
            source=source,
            default_timestamp=default_timestamp,
        )
        return [event] if event is not None else []

    return []


def _normalize_tool_event(
    payload: dict[str, Any],
    *,
    source: str,
    default_timestamp: Optional[datetime],
) -> Optional[ToolEvent]:
    tool_name = None
    for key in _TOOL_NAME_KEYS:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            tool_name = value.strip().lower()
            break
    if not tool_name:
        return None

    status_value = None
    for key in _STATUS_KEYS:
        value = payload.get(key)
        if value is None:
            continue
        normalized = str(value).strip().lower()
        if normalized:
            status_value = normalized
            break

    timestamp_value = default_timestamp
    for key in _TIMESTAMP_KEYS:
        parsed = _parse_iso_timestamp(payload.get(key))
        if parsed is not None:
            timestamp_value = parsed
            break

    raw_payload = {
        str(key): payload.get(key)
        for key in payload.keys()
    }

    return ToolEvent(
        name=tool_name,
        status=status_value,
        timestamp=timestamp_value,
        source=source,
        raw_payload=raw_payload,
    )


def _parse_iso_timestamp(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    if not isinstance(value, str):
        return None
    raw = value.strip()
    if not raw:
        return None
    normalized = raw.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _extract_event_id(raw_payload: Optional[dict[str, Any]]) -> str:
    if not isinstance(raw_payload, dict):
        return ""
    for key in _EVENT_ID_KEYS:
        value = raw_payload.get(key)
        if value is None:
            continue
        normalized = str(value).strip()
        if normalized:
            return normalized
    return ""


def _evaluate_rule(
    expression: ToolRuleExpression,
    *,
    indexed_events: list[tuple[int, ToolEvent]],
    min_index: int,
) -> RuleEvalResult:
    if expression.tool is not None:
        return _evaluate_leaf(expression, indexed_events=indexed_events, min_index=min_index)
    if expression.all is not None:
        return _evaluate_all(expression, indexed_events=indexed_events, min_index=min_index)
    if expression.any is not None:
        return _evaluate_any(expression, indexed_events=indexed_events, min_index=min_index)
    if expression.not_rule is not None:
        return _evaluate_not(expression, indexed_events=indexed_events, min_index=min_index)
    if expression.in_order is not None:
        return _evaluate_in_order(expression, indexed_events=indexed_events, min_index=min_index)
    return RuleEvalResult(
        passed=False,
        reasons=["Invalid empty tool rule expression."],
        matched_indices=[],
        matched_tools=set(),
        missing_tools=set(),
        order_violations=[],
    )


def _evaluate_leaf(
    expression: ToolRuleExpression,
    *,
    indexed_events: list[tuple[int, ToolEvent]],
    min_index: int,
) -> RuleEvalResult:
    required_tool = expression.tool or ""
    allowed_statuses = {
        status.strip().lower()
        for status in (expression.status_in or [])
        if status.strip()
    }
    matches: list[int] = []

    for index, event in indexed_events:
        if index < min_index:
            continue
        if event.name != required_tool:
            continue
        event_status = (event.status or "").strip().lower()
        if allowed_statuses and event_status not in allowed_statuses:
            continue
        matches.append(index)

    if len(matches) >= expression.min_count:
        return RuleEvalResult(
            passed=True,
            reasons=[],
            matched_indices=matches[: expression.min_count],
            matched_tools={required_tool},
            missing_tools=set(),
            order_violations=[],
        )

    status_note = ""
    if allowed_statuses:
        status_note = f" with status in {sorted(allowed_statuses)}"
    reason = (
        f"Missing tool '{required_tool}'{status_note}: "
        f"required {expression.min_count}, found {len(matches)}."
    )
    return RuleEvalResult(
        passed=False,
        reasons=[reason],
        matched_indices=matches,
        matched_tools=set(),
        missing_tools={required_tool},
        order_violations=[],
    )


def _evaluate_all(
    expression: ToolRuleExpression,
    *,
    indexed_events: list[tuple[int, ToolEvent]],
    min_index: int,
) -> RuleEvalResult:
    reasons: list[str] = []
    matched_indices: list[int] = []
    matched_tools: set[str] = set()
    missing_tools: set[str] = set()
    order_violations: list[str] = []
    passed = True

    for child in expression.all or []:
        result = _evaluate_rule(child, indexed_events=indexed_events, min_index=min_index)
        if not result.passed:
            passed = False
        reasons.extend(result.reasons)
        matched_indices.extend(result.matched_indices)
        matched_tools.update(result.matched_tools)
        missing_tools.update(result.missing_tools)
        order_violations.extend(result.order_violations)

    return RuleEvalResult(
        passed=passed,
        reasons=reasons,
        matched_indices=sorted(set(matched_indices)),
        matched_tools=matched_tools,
        missing_tools=missing_tools,
        order_violations=order_violations,
    )


def _evaluate_any(
    expression: ToolRuleExpression,
    *,
    indexed_events: list[tuple[int, ToolEvent]],
    min_index: int,
) -> RuleEvalResult:
    failed_reasons: list[str] = []
    candidates: list[RuleEvalResult] = []

    for child in expression.any or []:
        result = _evaluate_rule(child, indexed_events=indexed_events, min_index=min_index)
        if result.passed:
            candidates.append(result)
        else:
            failed_reasons.extend(result.reasons)

    if candidates:
        candidates.sort(
            key=lambda result: (
                min(result.matched_indices) if result.matched_indices else 10**9,
                len(result.reasons),
            )
        )
        return candidates[0]

    return RuleEvalResult(
        passed=False,
        reasons=(
            ["No `any` branch matched expected tool behavior."]
            + failed_reasons
        ),
        matched_indices=[],
        matched_tools=set(),
        missing_tools=_collect_tools_from_any(expression),
        order_violations=[],
    )


def _evaluate_not(
    expression: ToolRuleExpression,
    *,
    indexed_events: list[tuple[int, ToolEvent]],
    min_index: int,
) -> RuleEvalResult:
    if expression.not_rule is None:
        return RuleEvalResult(
            passed=False,
            reasons=["Invalid `not` expression."],
            matched_indices=[],
            matched_tools=set(),
            missing_tools=set(),
            order_violations=[],
        )
    child = _evaluate_rule(expression.not_rule, indexed_events=indexed_events, min_index=min_index)
    if child.passed:
        return RuleEvalResult(
            passed=False,
            reasons=["`not` rule was violated by observed tool events."],
            matched_indices=[],
            matched_tools=set(),
            missing_tools=set(),
            order_violations=[],
        )
    return RuleEvalResult(
        passed=True,
        reasons=[],
        matched_indices=[],
        matched_tools=set(),
        missing_tools=set(),
        order_violations=[],
    )


def _evaluate_in_order(
    expression: ToolRuleExpression,
    *,
    indexed_events: list[tuple[int, ToolEvent]],
    min_index: int,
) -> RuleEvalResult:
    cursor = min_index
    matched_indices: list[int] = []
    matched_tools: set[str] = set()
    missing_tools: set[str] = set()
    reasons: list[str] = []
    order_violations: list[str] = []

    for position, child in enumerate(expression.in_order or [], start=1):
        result = _evaluate_rule(child, indexed_events=indexed_events, min_index=cursor)
        if not result.passed:
            reason = f"in_order step {position} failed."
            reasons.append(reason)
            reasons.extend(result.reasons)
            missing_tools.update(result.missing_tools)
            order_violations.append(reason)
            order_violations.extend(result.order_violations)
            return RuleEvalResult(
                passed=False,
                reasons=reasons,
                matched_indices=matched_indices,
                matched_tools=matched_tools,
                missing_tools=missing_tools,
                order_violations=order_violations,
            )

        if result.matched_indices:
            earliest = min(result.matched_indices)
            latest = max(result.matched_indices)
            if earliest < cursor:
                violation = (
                    f"in_order step {position} matched before the previous step completed."
                )
                return RuleEvalResult(
                    passed=False,
                    reasons=[violation],
                    matched_indices=matched_indices,
                    matched_tools=matched_tools,
                    missing_tools=missing_tools,
                    order_violations=[violation],
                )
            cursor = latest + 1
            matched_indices.extend(result.matched_indices)

        matched_tools.update(result.matched_tools)
        missing_tools.update(result.missing_tools)
        order_violations.extend(result.order_violations)

    return RuleEvalResult(
        passed=True,
        reasons=[],
        matched_indices=sorted(set(matched_indices)),
        matched_tools=matched_tools,
        missing_tools=missing_tools,
        order_violations=order_violations,
    )


def _collect_tools_from_rule(expression: ToolRuleExpression) -> list[str]:
    tools: set[str] = set()
    _walk_tools(expression, tools)
    return sorted(tools)


def _collect_tools_from_any(expression: ToolRuleExpression) -> set[str]:
    tools: set[str] = set()
    for child in expression.any or []:
        _walk_tools(child, tools)
    return tools


def _walk_tools(expression: ToolRuleExpression, tools: set[str]) -> None:
    if expression.tool:
        tools.add(expression.tool)
    for child in expression.all or []:
        _walk_tools(child, tools)
    for child in expression.any or []:
        _walk_tools(child, tools)
    if expression.not_rule is not None:
        _walk_tools(expression.not_rule, tools)
    for child in expression.in_order or []:
        _walk_tools(child, tools)

