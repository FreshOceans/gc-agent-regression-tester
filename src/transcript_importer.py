"""Helpers for transcript import by conversation ID."""

from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timedelta, timezone
from io import StringIO
from typing import Any

import yaml

_CONVERSATION_ID_REGEX = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b"
)


def _normalize_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (value or "").strip().lower())


def _extract_ids_from_string(value: str) -> list[str]:
    if not value:
        return []
    return [match.group(0).lower() for match in _CONVERSATION_ID_REGEX.finditer(value)]


def dedupe_and_cap_conversation_ids(
    conversation_ids: list[str],
    *,
    max_ids: int,
) -> list[str]:
    """Deduplicate IDs preserving order and cap to max_ids."""
    if max_ids < 1:
        return []
    seen: set[str] = set()
    deduped: list[str] = []
    for raw in conversation_ids:
        normalized = (raw or "").strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
        if len(deduped) >= max_ids:
            break
    return deduped


def parse_conversation_ids_from_paste(raw_text: str) -> list[str]:
    """Parse conversation IDs from freeform pasted text."""
    return _extract_ids_from_string(raw_text or "")


def parse_conversation_ids_from_file(
    *,
    content: str,
    filename: str,
) -> list[str]:
    """Parse conversation IDs from uploaded file content."""
    lower_name = (filename or "").lower()
    if lower_name.endswith(".json"):
        return _parse_ids_from_json(content)
    if lower_name.endswith((".yaml", ".yml")):
        return _parse_ids_from_yaml(content)
    if lower_name.endswith(".csv"):
        return _parse_ids_from_delimited(content, delimiter=",")
    if lower_name.endswith(".tsv"):
        return _parse_ids_from_delimited(content, delimiter="\t")
    return _extract_ids_from_string(content or "")


def _parse_ids_from_json(content: str) -> list[str]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return _extract_ids_from_string(content)
    ids: list[str] = []
    _walk_payload_for_ids(payload, ids)
    return ids


def _parse_ids_from_yaml(content: str) -> list[str]:
    try:
        payload = yaml.safe_load(content)
    except yaml.YAMLError:
        return _extract_ids_from_string(content)
    ids: list[str] = []
    _walk_payload_for_ids(payload, ids)
    return ids


def _walk_payload_for_ids(node: Any, out: list[str]) -> None:
    if isinstance(node, str):
        out.extend(_extract_ids_from_string(node))
        return
    if isinstance(node, list):
        for item in node:
            _walk_payload_for_ids(item, out)
        return
    if not isinstance(node, dict):
        return

    for key, value in node.items():
        key_norm = _normalize_header(str(key))
        if key_norm in {"conversationid", "id"}:
            if isinstance(value, str):
                out.extend(_extract_ids_from_string(value))
            else:
                _walk_payload_for_ids(value, out)
            continue
        _walk_payload_for_ids(value, out)


def _parse_ids_from_delimited(content: str, *, delimiter: str) -> list[str]:
    rows = list(csv.reader(StringIO(content), delimiter=delimiter))
    if not rows:
        return []

    header = [_normalize_header(value) for value in rows[0]]
    id_col_idx = None
    for idx, token in enumerate(header):
        if token in {"conversationid", "id"}:
            id_col_idx = idx
            break

    data_rows = rows[1:] if id_col_idx is not None else rows
    ids: list[str] = []
    for row in data_rows:
        if not row:
            continue
        if id_col_idx is not None and id_col_idx < len(row):
            ids.extend(_extract_ids_from_string(row[id_col_idx]))
            continue
        for cell in row:
            ids.extend(_extract_ids_from_string(cell))
    return ids


def parse_filter_json(raw_filter: str) -> dict[str, Any]:
    """Parse custom filter JSON safely."""
    text = (raw_filter or "").strip()
    if not text:
        return {}
    loaded = json.loads(text)
    if not isinstance(loaded, dict):
        raise ValueError("Custom filter JSON must be an object")
    return loaded


def build_last_24h_interval(*, now_utc: datetime | None = None) -> str:
    """Build a Genesys interval string for the previous 24 hours."""
    now = now_utc or datetime.now(timezone.utc)
    start = now - timedelta(hours=24)
    return f"{start.isoformat()}/{now.isoformat()}"


def build_transcript_seeder_payload(transcripts: list[dict[str, Any]]) -> dict[str, Any]:
    """Convert normalized transcript records into seeder-compatible payload."""
    conversations: list[dict[str, Any]] = []
    for transcript in transcripts:
        conversation_id = str(transcript.get("conversation_id", "")).strip()
        messages_payload: list[dict[str, Any]] = []
        for msg in transcript.get("messages", []):
            if not isinstance(msg, dict):
                continue
            text = str(msg.get("text", "")).strip()
            if not text:
                continue
            role = str(msg.get("role", "")).strip().lower()
            if role in {"customer", "user", "traveler", "guest"}:
                speaker = "customer"
            elif role in {"agent", "assistant", "bot"}:
                speaker = "agent"
            else:
                speaker = role or "system"
            messages_payload.append(
                {
                    "speaker": speaker,
                    "text": text,
                    "timestamp": msg.get("timestamp"),
                    "participant_id": msg.get("participant_id"),
                }
            )
        conversations.append(
            {
                "conversation_id": conversation_id,
                "messages": messages_payload,
                "participants": transcript.get("participants", []),
                "start_time": transcript.get("start_time"),
                "end_time": transcript.get("end_time"),
            }
        )

    return {"conversations": conversations}
