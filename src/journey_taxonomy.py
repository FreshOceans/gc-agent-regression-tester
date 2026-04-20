"""Deterministic journey taxonomy classifier and rollup helpers (Phase 12)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .models import AttemptResult, TestReport

TOTAL_CALLS_LABEL = "Total Calls"

TAXONOMY_LABELS: list[str] = [
    TOTAL_CALLS_LABEL,
    "Successful Intent Capture - Successful Guest Authentication - Successful Transfer to Live Agent",
    "Successful Intent Capture - Guest Authentication Required - Successful Transfer to Live Agent",
    "Incorrect Intent Capture - Successful Guest Authentication - Successful Transfer to Live Agent",
    "Incorrect Intent Capture - Guest Authentication Not Required - Successful Transfer to Live Agent",
    "Successful Intent Capture - Successful Call Containment",
    "Agent Request - Successful Transfer To Agent",
    "Guest Hung Up",
    "Successful Intent Capture - Successful Authentication - Guest Disconnects Before Live Agent",
    "Test Call By Genesys",
    "No Guest Response",
    "Incorrect Intent Capture - Guest Hung Up",
    "Flow Issue - Guest Hung Up",
    "Caller Unintelligible",
    "Wrong Number/Marketing",
    "Successful Intent Capture - Guest Hung Up During Authentication",
    "Successful Intent Capture - Guest Hung Up During WestJetRewards",
]

_JOURNEY_VIEW_DEFINITIONS: dict[str, dict] = {
    "overview": {
        "label": "Overview",
        "labels": TAXONOMY_LABELS[1:],
    },
    "live_agent_transfer": {
        "label": "Live Agent Transfer",
        "labels": [
            TAXONOMY_LABELS[1],
            TAXONOMY_LABELS[2],
            TAXONOMY_LABELS[3],
            TAXONOMY_LABELS[4],
            TAXONOMY_LABELS[6],
            TAXONOMY_LABELS[8],
        ],
    },
    "containment": {
        "label": "Containment",
        "labels": [TAXONOMY_LABELS[5]],
    },
    "hangup_disconnect": {
        "label": "Hangup/Disconnect",
        "labels": [
            TAXONOMY_LABELS[7],
            TAXONOMY_LABELS[8],
            TAXONOMY_LABELS[11],
            TAXONOMY_LABELS[12],
            TAXONOMY_LABELS[15],
            TAXONOMY_LABELS[16],
        ],
    },
    "flow_noise_issues": {
        "label": "Flow/Noise Issues",
        "labels": [
            TAXONOMY_LABELS[9],
            TAXONOMY_LABELS[10],
            TAXONOMY_LABELS[12],
            TAXONOMY_LABELS[13],
            TAXONOMY_LABELS[14],
        ],
    },
}


def normalize_journey_view(value: Optional[str]) -> str:
    raw = str(value or "overview").strip().lower()
    return raw if raw in _JOURNEY_VIEW_DEFINITIONS else "overview"


def journey_view_definitions() -> dict[str, dict]:
    return _JOURNEY_VIEW_DEFINITIONS


def load_taxonomy_overrides(
    *,
    overrides_json: Optional[str],
    overrides_file: Optional[str],
) -> dict[str, str]:
    """Load optional keyword->label override mapping."""
    payload: dict = {}
    raw_json = str(overrides_json or "").strip()
    if raw_json:
        try:
            decoded = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "journey taxonomy override JSON is invalid"
            ) from exc
        if not isinstance(decoded, dict):
            raise ValueError("journey taxonomy override JSON must be an object")
        payload.update(decoded)

    file_path = str(overrides_file or "").strip()
    if file_path:
        path = Path(file_path)
        if not path.exists():
            raise ValueError(
                f"journey taxonomy override file not found: {file_path}"
            )
        try:
            decoded = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                "journey taxonomy override file must contain valid JSON"
            ) from exc
        if not isinstance(decoded, dict):
            raise ValueError("journey taxonomy override file must contain a JSON object")
        payload.update(decoded)

    normalized: dict[str, str] = {}
    valid_labels = set(TAXONOMY_LABELS)
    for keyword, label in payload.items():
        key = str(keyword or "").strip().lower()
        target = str(label or "").strip()
        if not key or target not in valid_labels:
            continue
        normalized[key] = target
    return normalized


def classify_attempt_taxonomy(
    *,
    expected_intent: Optional[str],
    attempt: AttemptResult,
    overrides: Optional[dict[str, str]] = None,
) -> str:
    """Map one attempt to the fixed journey taxonomy labels."""
    corpus = _build_corpus(attempt)
    override_map = overrides if isinstance(overrides, dict) else {}
    for keyword, label in override_map.items():
        if keyword and keyword in corpus:
            return label

    if "test call by genesys" in corpus:
        return TAXONOMY_LABELS[9]
    if "wrong number" in corpus or "marketing" in corpus:
        return TAXONOMY_LABELS[14]
    if "caller unintelligible" in corpus or "unintelligible" in corpus:
        return TAXONOMY_LABELS[13]

    user_messages = [
        msg
        for msg in attempt.conversation
        if getattr(msg, "role", None) is not None
        and getattr(msg.role, "value", "") == "user"
    ]
    if attempt.timed_out and len(user_messages) <= 1:
        return TAXONOMY_LABELS[10]

    hung_up = any(
        token in corpus
        for token in [
            "hung up",
            "guest hung up",
            "disconnects before",
            "conversation ended by user",
            "disconnect",
        ]
    )
    flow_issue = "flow issue" in corpus
    auth_success = any(
        token in corpus
        for token in [
            "successful authentication",
            "authentication successful",
            "guest authenticated",
            "auth success",
        ]
    )
    auth_required = any(
        token in corpus
        for token in [
            "authentication required",
            "auth required",
            "verify identity",
            "westjet rewards id",
        ]
    )
    transfer_success = any(
        token in corpus
        for token in [
            "transfer to live agent",
            "transfer to agent",
            "live agent",
            "escalate your request",
            "speak with a live agent",
        ]
    )

    expected_norm = str(expected_intent or "").strip().lower()
    detected_norm = str(attempt.detected_intent or "").strip().lower()
    intent_known = bool(expected_norm)
    intent_correct = bool(intent_known and detected_norm and detected_norm == expected_norm)
    intent_incorrect = bool(intent_known and detected_norm and detected_norm != expected_norm)

    if intent_correct and hung_up and ("authentication" in corpus or auth_required):
        return TAXONOMY_LABELS[15]
    if intent_correct and hung_up and "westjetrewards" in corpus:
        return TAXONOMY_LABELS[16]
    if flow_issue and hung_up:
        return TAXONOMY_LABELS[12]
    if intent_incorrect and hung_up:
        return TAXONOMY_LABELS[11]
    if intent_correct and auth_success and hung_up and not transfer_success:
        return TAXONOMY_LABELS[8]
    if hung_up:
        return TAXONOMY_LABELS[7]

    if expected_norm in {"speak_to_agent", "agent_request"} and transfer_success:
        return TAXONOMY_LABELS[6]

    if transfer_success and intent_correct and auth_success:
        return TAXONOMY_LABELS[1]
    if transfer_success and intent_correct and auth_required and not auth_success:
        return TAXONOMY_LABELS[2]
    if transfer_success and intent_incorrect and auth_success:
        return TAXONOMY_LABELS[3]
    if transfer_success and intent_incorrect and not auth_required:
        return TAXONOMY_LABELS[4]

    journey = attempt.journey_validation_result
    if journey is not None and journey.contained is True and attempt.success:
        return TAXONOMY_LABELS[5]

    return TAXONOMY_LABELS[10] if attempt.timed_out else TAXONOMY_LABELS[12]


def build_journey_taxonomy_rollups(
    report: TestReport,
    *,
    baseline_report: Optional[TestReport] = None,
    overrides: Optional[dict[str, str]] = None,
    active_view: str = "overview",
) -> dict:
    """Compute fixed-label rollups and view slices for the journey dashboard."""
    active = normalize_journey_view(active_view)
    counts = {label: 0 for label in TAXONOMY_LABELS[1:]}
    per_attempt: list[dict] = []

    for scenario in report.scenario_results:
        expected = scenario.expected_intent
        for attempt in scenario.attempt_results:
            label = classify_attempt_taxonomy(
                expected_intent=expected,
                attempt=attempt,
                overrides=overrides,
            )
            if label in counts:
                counts[label] += 1
            attempt.journey_taxonomy_label = label
            per_attempt.append(
                {
                    "scenario_name": scenario.scenario_name,
                    "attempt_number": attempt.attempt_number,
                    "label": label,
                    "success": attempt.success,
                }
            )

    baseline_counts: dict[str, int] = {label: 0 for label in TAXONOMY_LABELS[1:]}
    if baseline_report is not None:
        for scenario in baseline_report.scenario_results:
            expected = scenario.expected_intent
            for attempt in scenario.attempt_results:
                label = classify_attempt_taxonomy(
                    expected_intent=expected,
                    attempt=attempt,
                    overrides=overrides,
                )
                if label in baseline_counts:
                    baseline_counts[label] += 1

    total_calls = report.overall_attempts
    baseline_total = baseline_report.overall_attempts if baseline_report is not None else 0

    rows = [
        {
            "label": TOTAL_CALLS_LABEL,
            "count": total_calls,
            "rate": 1.0 if total_calls > 0 else 0.0,
            "delta": (total_calls - baseline_total) if baseline_report is not None else None,
        }
    ]
    for label in TAXONOMY_LABELS[1:]:
        count = counts.get(label, 0)
        baseline_count = baseline_counts.get(label, 0)
        rows.append(
            {
                "label": label,
                "count": count,
                "rate": (float(count) / float(total_calls)) if total_calls > 0 else 0.0,
                "delta": (count - baseline_count) if baseline_report is not None else None,
            }
        )

    view_items = []
    for key, definition in _JOURNEY_VIEW_DEFINITIONS.items():
        labels = definition.get("labels") or []
        view_rows = [row for row in rows if row["label"] in labels]
        view_total = sum(int(row["count"]) for row in view_rows)
        baseline_view_total = None
        if baseline_report is not None:
            baseline_view_total = sum(
                int(baseline_counts.get(label, 0))
                for label in labels
            )
        view_items.append(
            {
                "key": key,
                "label": str(definition.get("label") or key),
                "rows": view_rows,
                "total": view_total,
                "delta": (
                    view_total - baseline_view_total
                    if baseline_view_total is not None
                    else None
                ),
            }
        )

    return {
        "active_view": active,
        "labels": rows,
        "views": view_items,
        "total_calls": total_calls,
        "attempt_taxonomy": per_attempt,
    }


def _build_corpus(attempt: AttemptResult) -> str:
    parts: list[str] = []
    if attempt.error:
        parts.append(str(attempt.error))
    if attempt.explanation:
        parts.append(str(attempt.explanation))
    if attempt.detected_intent:
        parts.append(str(attempt.detected_intent))
    if attempt.journey_validation_result is not None:
        jr = attempt.journey_validation_result
        parts.extend(jr.failure_reasons)
        if jr.explanation:
            parts.append(str(jr.explanation))
        if jr.actual_category:
            parts.append(str(jr.actual_category))
        if jr.expected_category:
            parts.append(str(jr.expected_category))
        if jr.containment_source:
            parts.append(str(jr.containment_source))
    for event in attempt.tool_events:
        parts.append(str(event.name or ""))
        if event.status:
            parts.append(str(event.status))
    for message in attempt.conversation:
        content = getattr(message, "content", None)
        if content:
            parts.append(str(content))
    return " ".join(parts).strip().lower()
