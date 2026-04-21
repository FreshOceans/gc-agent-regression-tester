"""Genesys Cloud transcript import client for conversation-ID workflows."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Optional

import requests

from .models import (
    ANALYTICS_AUTH_MODE_CLIENT_CREDENTIALS,
    normalize_analytics_auth_mode,
)


class GenesysTranscriptImportError(Exception):
    """Raised when transcript import operations fail."""


class GenesysTranscriptImportClient:
    """Client for querying conversation IDs and fetching message transcripts."""

    def __init__(
        self,
        *,
        region: str,
        client_id: str,
        client_secret: str,
        auth_mode: str = ANALYTICS_AUTH_MODE_CLIENT_CREDENTIALS,
        manual_bearer_token: Optional[str] = None,
        timeout: int = 30,
        retries: int = 3,
        retry_delay_seconds: float = 1.0,
    ):
        self.region = region.strip()
        self.client_id = client_id.strip()
        self.client_secret = client_secret.strip()
        self.auth_mode = normalize_analytics_auth_mode(auth_mode)
        self.manual_bearer_token = str(manual_bearer_token or "").strip()
        self.timeout = timeout
        self.retries = max(1, retries)
        self.retry_delay_seconds = max(0.0, retry_delay_seconds)
        self._access_token: Optional[str] = None
        self._token_expiry_monotonic = 0.0

    @property
    def _oauth_url(self) -> str:
        return f"https://login.{self.region}/oauth/token"

    @property
    def _api_base_url(self) -> str:
        return f"https://api.{self.region}"

    def _get_access_token(self) -> str:
        if self.auth_mode != ANALYTICS_AUTH_MODE_CLIENT_CREDENTIALS:
            if not self.manual_bearer_token:
                raise GenesysTranscriptImportError(
                    "Manual bearer auth mode requires a bearer token"
                )
            return self.manual_bearer_token

        now = time.monotonic()
        if self._access_token and now < self._token_expiry_monotonic:
            return self._access_token

        try:
            response = requests.post(
                self._oauth_url,
                data={"grant_type": "client_credentials"},
                auth=(self.client_id, self.client_secret),
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json",
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as e:
            raise GenesysTranscriptImportError(
                f"OAuth token request failed for region '{self.region}': {e}"
            ) from e
        except ValueError as e:
            raise GenesysTranscriptImportError(
                f"Invalid OAuth token response for region '{self.region}': {e}"
            ) from e

        token = payload.get("access_token")
        expires_in = payload.get("expires_in", 300)
        if not isinstance(token, str) or not token.strip():
            raise GenesysTranscriptImportError("OAuth token response missing access_token")

        safe_ttl = max(30, int(expires_in) - 30)
        self._access_token = token
        self._token_expiry_monotonic = time.monotonic() + safe_ttl
        return token

    def _request_json(
        self,
        *,
        method: str,
        path: str,
        json_payload: Optional[dict[str, Any]] = None,
        params: Optional[dict[str, Any]] = None,
        stop_requested=None,
    ) -> dict[str, Any]:
        if self._is_stop_requested(stop_requested):
            raise GenesysTranscriptImportError("Request interrupted by stop request")
        token = self._get_access_token()
        url = f"{self._api_base_url}{path}"
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }
        if method.strip().upper() in {"POST", "PUT", "PATCH"}:
            headers["Content-Type"] = "application/json"

        last_error: Optional[Exception] = None
        for attempt in range(1, self.retries + 1):
            if self._is_stop_requested(stop_requested):
                raise GenesysTranscriptImportError(
                    f"Request interrupted by stop request for {path}"
                )
            try:
                response = requests.request(
                    method,
                    url,
                    headers=headers,
                    json=json_payload,
                    params=params,
                    timeout=self.timeout,
                )
                if response.status_code in {429, 500, 502, 503, 504} and attempt < self.retries:
                    if not self._sleep_with_stop_support(
                        self.retry_delay_seconds * attempt,
                        stop_requested,
                    ):
                        raise GenesysTranscriptImportError(
                            f"Request interrupted by stop request for {path}"
                        )
                    continue
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise GenesysTranscriptImportError(
                        f"Unexpected API payload type for {path}"
                    )
                return payload
            except requests.RequestException as e:
                last_error = e
                if attempt < self.retries:
                    if not self._sleep_with_stop_support(
                        self.retry_delay_seconds * attempt,
                        stop_requested,
                    ):
                        raise GenesysTranscriptImportError(
                            f"Request interrupted by stop request for {path}"
                        )
                    continue
                break
            except ValueError as e:
                last_error = e
                break

        raise GenesysTranscriptImportError(
            f"Request failed for {path}: {last_error}"
        )

    def fetch_conversation_payload(
        self,
        conversation_id: str,
        *,
        stop_requested=None,
    ) -> dict[str, Any]:
        """Fetch raw conversation payload by ID."""
        cid = conversation_id.strip()
        if not cid:
            raise GenesysTranscriptImportError("Conversation ID is required")
        return self._request_json(
            method="GET",
            path=f"/api/v2/conversations/messages/{cid}",
            stop_requested=stop_requested,
        )

    def fetch_conversation_transcript(self, conversation_id: str) -> dict[str, Any]:
        """Fetch and normalize a conversation transcript by ID."""
        payload = self.fetch_conversation_payload(conversation_id)
        return self.normalize_conversation_payload(payload, conversation_id=conversation_id)

    def query_conversation_ids(
        self,
        *,
        filter_payload: Optional[dict[str, Any]],
        interval: str,
        page_size: int = 100,
        max_results: int = 50,
        stop_requested=None,
    ) -> list[dict[str, Optional[str]]]:
        """Query conversation IDs via analytics details query."""
        if not interval.strip():
            raise GenesysTranscriptImportError("Query interval is required")
        if max_results < 1:
            return []

        page_number = 1
        results: list[dict[str, Optional[str]]] = []
        while len(results) < max_results:
            if self._is_stop_requested(stop_requested):
                raise GenesysTranscriptImportError(
                    "Conversation ID query interrupted by stop request"
                )
            request_payload = dict(filter_payload or {})
            request_payload["interval"] = interval
            request_payload["order"] = "desc"
            request_payload["orderBy"] = "conversationEnd"
            request_payload["paging"] = {
                "pageSize": max(1, min(page_size, 100)),
                "pageNumber": page_number,
            }

            payload = self._request_json(
                method="POST",
                path="/api/v2/analytics/conversations/details/query",
                json_payload=request_payload,
                stop_requested=stop_requested,
            )
            conversations = payload.get("conversations", [])
            if not isinstance(conversations, list) or not conversations:
                break

            for convo in conversations:
                if not isinstance(convo, dict):
                    continue
                cid = str(convo.get("conversationId") or convo.get("id") or "").strip()
                if not cid:
                    continue
                end_ts = (
                    convo.get("conversationEnd")
                    or convo.get("conversationEndTime")
                    or convo.get("conversationEndDate")
                    or convo.get("endTime")
                )
                results.append(
                    {
                        "conversation_id": cid,
                        "conversation_end": str(end_ts) if end_ts else None,
                    }
                )
            if len(results) >= max_results:
                break
            if len(conversations) < request_payload["paging"]["pageSize"]:
                break
            page_number += 1

        results.sort(
            key=lambda row: self._parse_timestamp(row.get("conversation_end")),
            reverse=True,
        )
        return results[:max_results]

    def import_transcripts_by_ids(
        self,
        conversation_ids: list[str],
        *,
        stop_requested=None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Fetch transcripts by IDs returning fetched/failed/skipped outcomes."""
        outcomes: dict[str, list[dict[str, Any]]] = {
            "fetched": [],
            "failed": [],
            "skipped": [],
        }
        for cid in conversation_ids:
            if self._is_stop_requested(stop_requested):
                outcomes["skipped"].append(
                    {
                        "conversation_id": "",
                        "reason": "Import interrupted by stop request",
                    }
                )
                break
            conversation_id = str(cid or "").strip()
            if not conversation_id:
                outcomes["skipped"].append(
                    {"conversation_id": conversation_id, "reason": "Empty conversation ID"}
                )
                continue
            try:
                raw_payload = self.fetch_conversation_payload(
                    conversation_id,
                    stop_requested=stop_requested,
                )
                normalized = self.normalize_conversation_payload(
                    raw_payload, conversation_id=conversation_id
                )
            except Exception as e:
                outcomes["failed"].append(
                    {"conversation_id": conversation_id, "reason": str(e)}
                )
                continue

            if not normalized.get("messages"):
                outcomes["skipped"].append(
                    {
                        "conversation_id": conversation_id,
                        "reason": "No transcript messages found in conversation payload",
                    }
                )
                continue

            outcomes["fetched"].append(
                {
                    "conversation_id": conversation_id,
                    "transcript": normalized,
                    "raw_payload": raw_payload,
                }
            )
        return outcomes

    @staticmethod
    def _is_stop_requested(stop_requested) -> bool:
        return bool(callable(stop_requested) and stop_requested())

    @staticmethod
    def _sleep_with_stop_support(total_seconds: float, stop_requested) -> bool:
        remaining = max(0.0, float(total_seconds))
        while remaining > 0:
            if callable(stop_requested) and stop_requested():
                return False
            sleep_for = min(0.1, remaining)
            time.sleep(sleep_for)
            remaining -= sleep_for
        return not (callable(stop_requested) and stop_requested())

    def normalize_conversation_payload(
        self,
        payload: dict[str, Any],
        *,
        conversation_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Normalize raw conversation payload into canonical transcript shape."""
        if not isinstance(payload, dict):
            raise GenesysTranscriptImportError("Conversation payload must be an object")

        normalized_conversation_id = (
            str(conversation_id or payload.get("id") or "").strip()
        )
        participants_payload = payload.get("participants", [])
        participants: list[dict[str, Optional[str]]] = []
        message_rows: list[tuple[float, int, dict[str, Any]]] = []
        order_index = 0

        if isinstance(participants_payload, list):
            for participant in participants_payload:
                if not isinstance(participant, dict):
                    continue
                participant_id = str(participant.get("id") or "").strip() or None
                purpose = str(participant.get("purpose") or "").strip() or None
                name = (
                    str(
                        participant.get("name")
                        or participant.get("participantName")
                        or participant.get("address")
                        or ""
                    ).strip()
                    or None
                )
                participants.append(
                    {
                        "participant_id": participant_id,
                        "purpose": purpose,
                        "name": name,
                    }
                )
                role = self._role_from_purpose(purpose)
                extracted = self._extract_participant_messages(participant, role=role)
                for msg in extracted:
                    parsed = self._parse_timestamp(msg.get("timestamp"))
                    ts_sort = parsed.timestamp() if parsed is not None else float("-inf")
                    message_rows.append((ts_sort, order_index, msg))
                    order_index += 1

        # Fallback for top-level messages list.
        top_level_messages = payload.get("messages")
        if isinstance(top_level_messages, list):
            for message in top_level_messages:
                if not isinstance(message, dict):
                    continue
                text = self._extract_text_from_message(message)
                if not text:
                    continue
                role = self._role_from_purpose(
                    str(message.get("purpose") or message.get("role") or "").strip()
                )
                timestamp = self._extract_timestamp_from_message(message)
                parsed = self._parse_timestamp(timestamp)
                ts_sort = parsed.timestamp() if parsed is not None else float("-inf")
                message_rows.append(
                    (
                        ts_sort,
                        order_index,
                        {
                            "role": role,
                            "text": text,
                            "timestamp": timestamp,
                            "participant_id": str(message.get("participantId") or "").strip() or None,
                        },
                    )
                )
                order_index += 1

        message_rows.sort(key=lambda row: (row[0], row[1]))
        ordered_messages = [row[2] for row in message_rows]

        return {
            "conversation_id": normalized_conversation_id,
            "start_time": (
                payload.get("startTime")
                or payload.get("conversationStart")
                or payload.get("conversationStartTime")
            ),
            "end_time": (
                payload.get("endTime")
                or payload.get("conversationEnd")
                or payload.get("conversationEndTime")
            ),
            "participants": participants,
            "messages": ordered_messages,
        }

    def _extract_participant_messages(
        self,
        participant: dict[str, Any],
        *,
        role: str,
    ) -> list[dict[str, Any]]:
        extracted: list[dict[str, Any]] = []
        participant_id = str(participant.get("id") or "").strip() or None

        def add_message(node: dict[str, Any]) -> None:
            text = self._extract_text_from_message(node)
            if not text:
                return
            extracted.append(
                {
                    "role": role,
                    "text": text,
                    "timestamp": self._extract_timestamp_from_message(node),
                    "participant_id": participant_id,
                }
            )

        for message in participant.get("messages", []) if isinstance(participant.get("messages"), list) else []:
            if isinstance(message, dict):
                add_message(message)

        for container_key in (
            "chats",
            "emails",
            "sms",
            "callbacks",
            "calls",
            "coBrowseSessions",
            "videos",
        ):
            container_list = participant.get(container_key)
            if not isinstance(container_list, list):
                continue
            for container in container_list:
                if not isinstance(container, dict):
                    continue
                communications = container.get("communications")
                if isinstance(communications, list):
                    for communication in communications:
                        if isinstance(communication, dict):
                            add_message(communication)
                nested_messages = container.get("messages")
                if isinstance(nested_messages, list):
                    for nested in nested_messages:
                        if isinstance(nested, dict):
                            add_message(nested)
                if any(
                    key in container
                    for key in ("messageText", "textBody", "body", "text", "content")
                ):
                    add_message(container)

        return extracted

    def _extract_text_from_message(self, message: dict[str, Any]) -> Optional[str]:
        if not isinstance(message, dict):
            return None
        for key in (
            "messageText",
            "textBody",
            "body",
            "text",
            "content",
            "subject",
            "name",
        ):
            value = message.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _extract_timestamp_from_message(self, message: dict[str, Any]) -> Optional[str]:
        if not isinstance(message, dict):
            return None
        for key in (
            "timestamp",
            "time",
            "eventTime",
            "connectedTime",
            "startTime",
            "endTime",
            "recording",
        ):
            value = message.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _role_from_purpose(self, purpose: Optional[str]) -> str:
        normalized = str(purpose or "").strip().lower()
        if normalized in {"customer", "external", "user", "client", "consumer"}:
            return "customer"
        if normalized in {"agent", "assistant", "acd", "bot", "workflow", "system"}:
            return "agent"
        return "system"

    def _parse_timestamp(self, value: Optional[str]) -> datetime:
        text = str(value or "").strip()
        if not text:
            return datetime.fromtimestamp(0, tz=timezone.utc)
        normalized = text
        if normalized.endswith("Z"):
            normalized = normalized[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return datetime.fromtimestamp(0, tz=timezone.utc)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
