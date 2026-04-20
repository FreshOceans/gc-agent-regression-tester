"""Unit tests for analytics journey runner helpers."""

from datetime import datetime, timezone

from src.analytics_journey_runner import (
    evaluate_gate,
    infer_auth_evidence,
    infer_transfer_evidence,
    load_analytics_policy_map,
    resolve_policy_for_category,
)
from src.models import Message, MessageRole


def test_load_analytics_policy_map_merges_defaults_with_overrides(tmp_path):
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        '{"flight_cancel":{"auth_behavior":"required","transfer_behavior":"forbidden"}}',
        encoding="utf-8",
    )

    policy = load_analytics_policy_map(policy_json="", policy_file=str(policy_path))

    assert "default" in policy
    assert policy["flight_cancel"]["auth_behavior"] == "required"
    assert policy["flight_cancel"]["transfer_behavior"] == "forbidden"


def test_resolve_policy_for_category_falls_back_to_default():
    policy = {
        "default": {"auth_behavior": "optional", "transfer_behavior": "optional"},
        "speak_to_agent": {"auth_behavior": "optional", "transfer_behavior": "required"},
    }

    key, resolved = resolve_policy_for_category("unknown_intent", policy)
    assert key == "default"
    assert resolved["transfer_behavior"] == "optional"


def test_evaluate_gate_required_and_optional_behaviors():
    assert evaluate_gate(expected_behavior="required", observed=True) == (True, False)
    assert evaluate_gate(expected_behavior="required", observed=False) == (False, False)
    assert evaluate_gate(expected_behavior="required", observed=None) == (None, True)
    assert evaluate_gate(expected_behavior="optional", observed=None) == (True, False)
    assert evaluate_gate(expected_behavior="forbidden", observed=False) == (True, False)


def test_infer_auth_evidence_from_transcript_tokens():
    messages = [
        Message(
            role=MessageRole.AGENT,
            content="Authentication successful. You're now verified.",
            timestamp=datetime.now(timezone.utc),
        )
    ]

    observed, notes = infer_auth_evidence(messages, raw_payload=None)
    assert observed is True
    assert any("transcript" in note for note in notes)


def test_infer_transfer_evidence_uses_containment_hint_first():
    messages = [Message(role=MessageRole.AGENT, content="Let's continue here")]

    observed_true, notes_true = infer_transfer_evidence(
        messages,
        raw_payload=None,
        contained_hint=False,
    )
    observed_false, notes_false = infer_transfer_evidence(
        messages,
        raw_payload=None,
        contained_hint=True,
    )

    assert observed_true is True
    assert observed_false is False
    assert notes_true
    assert notes_false
