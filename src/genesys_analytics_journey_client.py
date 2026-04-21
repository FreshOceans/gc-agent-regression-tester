"""Genesys Analytics details-query client for analytics journey regression."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Callable, Optional

import requests

from .models import (
    ANALYTICS_AUTH_MODE_CLIENT_CREDENTIALS,
    normalize_analytics_auth_mode,
)


class GenesysAnalyticsJourneyError(Exception):
    """Raised when Analytics journey ingestion fails."""


class GenesysAnalyticsJourneyClient:
    """Client for conversations/details/query and conversation ID extraction."""

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
        page_size_cap: int = 100,
    ):
        self.region = region.strip()
        self.client_id = client_id.strip()
        self.client_secret = client_secret.strip()
        self.auth_mode = normalize_analytics_auth_mode(auth_mode)
        self.manual_bearer_token = str(manual_bearer_token or "").strip()
        self.timeout = timeout
        self.retries = max(1, retries)
        self.retry_delay_seconds = max(0.0, retry_delay_seconds)
        self.page_size_cap = max(1, int(page_size_cap))
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
                raise GenesysAnalyticsJourneyError(
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
            raise GenesysAnalyticsJourneyError(
                f"OAuth token request failed for region '{self.region}': {e}"
            ) from e
        except ValueError as e:
            raise GenesysAnalyticsJourneyError(
                f"Invalid OAuth token response for region '{self.region}': {e}"
            ) from e

        token = payload.get("access_token")
        expires_in = payload.get("expires_in", 300)
        if not isinstance(token, str) or not token.strip():
            raise GenesysAnalyticsJourneyError(
                "OAuth token response missing access_token"
            )

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
        observer: Optional[Callable[[dict[str, Any]], None]] = None,
        request_context: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        token = self._get_access_token()
        url = f"{self._api_base_url}{path}"
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }
        if method.strip().upper() in {"POST", "PUT", "PATCH"}:
            headers["Content-Type"] = "application/json"
        context = dict(request_context or {})

        last_error: Optional[Exception] = None
        for attempt in range(1, self.retries + 1):
            request_started_at = time.monotonic()
            self._emit_observer(
                observer,
                {
                    "event": "request_attempt_started",
                    "method": method,
                    "path": path,
                    "attempt": attempt,
                    "max_attempts": self.retries,
                    **context,
                },
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
                if (
                    response.status_code in {429, 500, 502, 503, 504}
                    and attempt < self.retries
                ):
                    backoff_seconds = self.retry_delay_seconds * attempt
                    self._emit_observer(
                        observer,
                        {
                            "event": "request_retry",
                            "method": method,
                            "path": path,
                            "attempt": attempt,
                            "max_attempts": self.retries,
                            "status_code": response.status_code,
                            "backoff_seconds": backoff_seconds,
                            "duration_ms": round(
                                (time.monotonic() - request_started_at) * 1000.0,
                                2,
                            ),
                            **context,
                        },
                    )
                    time.sleep(backoff_seconds)
                    continue
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise GenesysAnalyticsJourneyError(
                        f"Unexpected API payload type for {path}"
                    )
                self._emit_observer(
                    observer,
                    {
                        "event": "request_attempt_succeeded",
                        "method": method,
                        "path": path,
                        "attempt": attempt,
                        "max_attempts": self.retries,
                        "status_code": response.status_code,
                        "duration_ms": round(
                            (time.monotonic() - request_started_at) * 1000.0,
                            2,
                        ),
                        **context,
                    },
                )
                return payload
            except requests.RequestException as e:
                last_error = e
                if attempt < self.retries:
                    backoff_seconds = self.retry_delay_seconds * attempt
                    self._emit_observer(
                        observer,
                        {
                            "event": "request_retry",
                            "method": method,
                            "path": path,
                            "attempt": attempt,
                            "max_attempts": self.retries,
                            "error_type": type(e).__name__,
                            "error": str(e),
                            "backoff_seconds": backoff_seconds,
                            "duration_ms": round(
                                (time.monotonic() - request_started_at) * 1000.0,
                                2,
                            ),
                            **context,
                        },
                    )
                    time.sleep(backoff_seconds)
                    continue
                break
            except ValueError as e:
                last_error = e
                self._emit_observer(
                    observer,
                    {
                        "event": "request_payload_error",
                        "method": method,
                        "path": path,
                        "attempt": attempt,
                        "max_attempts": self.retries,
                        "error_type": type(e).__name__,
                        "error": str(e),
                        "duration_ms": round(
                            (time.monotonic() - request_started_at) * 1000.0,
                            2,
                        ),
                        **context,
                    },
                )
                break

        self._emit_observer(
            observer,
            {
                "event": "request_failed",
                "method": method,
                "path": path,
                "error_type": type(last_error).__name__ if last_error else "unknown",
                "error": str(last_error) if last_error else None,
                **context,
            },
        )
        raise GenesysAnalyticsJourneyError(f"Request failed for {path}: {last_error}")

    def fetch_reporting_turns_page(
        self,
        *,
        bot_flow_id: str,
        interval: str,
        page_size: int,
        page_number: int,
        divisions: Optional[list[str]] = None,
        language_filter: Optional[str] = None,
        extra_params: Optional[dict[str, Any]] = None,
        observer: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> dict[str, Any]:
        normalized_interval = str(interval or "").strip()
        if not normalized_interval:
            raise GenesysAnalyticsJourneyError("Analytics interval is required")

        request_payload: dict[str, Any] = {
            "interval": normalized_interval,
            "order": "asc",
            "orderBy": "conversationStart",
            "paging": {
                "pageSize": max(1, min(int(page_size), self.page_size_cap)),
                "pageNumber": max(1, int(page_number)),
            },
        }
        # Division and language scoping for details/query is tenant-specific.
        # Operators can provide exact predicates via Advanced Raw Filter JSON.
        _ = divisions
        _ = language_filter

        for key, value in (extra_params or {}).items():
            key_name = str(key or "").strip()
            if not key_name:
                continue
            if key_name in {"interval", "paging"}:
                continue
            request_payload[key_name] = value

        return self._request_json(
            method="POST",
            path="/api/v2/analytics/conversations/details/query",
            json_payload=request_payload,
            observer=observer,
            request_context={"page_number": max(1, int(page_number))},
        )

    def fetch_conversation_units(
        self,
        *,
        bot_flow_id: str,
        interval: str,
        page_size: int,
        max_conversations: int,
        divisions: Optional[list[str]] = None,
        language_filter: Optional[str] = None,
        extra_params: Optional[dict[str, Any]] = None,
        observer: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> dict[str, Any]:
        """Fetch details-query pages and return deduped conversation units."""
        deduped_ids: list[str] = []
        seen_ids: set[str] = set()
        rows_by_conversation: dict[str, list[dict[str, Any]]] = defaultdict(list)
        page_payloads: list[dict[str, Any]] = []

        max_items = max(1, int(max_conversations))
        page_num = 1
        while len(deduped_ids) < max_items:
            page_started_at = time.monotonic()
            self._emit_observer(
                observer,
                {
                    "event": "page_fetch_started",
                    "page_number": page_num,
                    "max_conversations": max_items,
                    "current_unique_conversations": len(deduped_ids),
                },
            )
            payload = self.fetch_reporting_turns_page(
                bot_flow_id=bot_flow_id,
                interval=interval,
                page_size=page_size,
                page_number=page_num,
                divisions=divisions,
                language_filter=language_filter,
                extra_params=extra_params,
                observer=observer,
            )
            page_payloads.append(payload)
            page_rows = self.extract_rows(payload)
            ids_before = len(deduped_ids)

            for row in page_rows:
                conversation_id = self.extract_conversation_id(row)
                if not conversation_id:
                    continue
                rows_by_conversation[conversation_id].append(row)
                if conversation_id in seen_ids:
                    continue
                seen_ids.add(conversation_id)
                deduped_ids.append(conversation_id)
                if len(deduped_ids) >= max_items:
                    break
            self._emit_observer(
                observer,
                {
                    "event": "page_fetch_completed",
                    "page_number": page_num,
                    "rows_count": len(page_rows),
                    "new_unique_conversations": max(0, len(deduped_ids) - ids_before),
                    "total_unique_conversations": len(deduped_ids),
                    "duration_ms": round(
                        (time.monotonic() - page_started_at) * 1000.0,
                        2,
                    ),
                },
            )

            if len(deduped_ids) >= max_items:
                break

            # Stop when a page yields no rows or no new IDs.
            if not page_rows or ids_before == len(deduped_ids):
                break

            page_num += 1

        units = [
            {
                "conversation_id": conversation_id,
                "rows": rows_by_conversation.get(conversation_id, []),
            }
            for conversation_id in deduped_ids
        ]
        return {
            "conversations": units,
            "page_payloads": page_payloads,
            "page_count": len(page_payloads),
        }

    @staticmethod
    def _emit_observer(
        observer: Optional[Callable[[dict[str, Any]], None]],
        payload: dict[str, Any],
    ) -> None:
        if observer is None:
            return
        try:
            observer(dict(payload))
        except Exception:
            # Diagnostics hooks must never affect ingestion behavior.
            return

    @classmethod
    def extract_rows(cls, payload: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract likely row dicts from an analytics details-query payload."""
        rows: list[dict[str, Any]] = []
        for key in (
            "results",
            "entities",
            "reportingTurns",
            "turns",
            "conversations",
            "data",
            "items",
        ):
            value = payload.get(key)
            if isinstance(value, list):
                rows.extend(item for item in value if isinstance(item, dict))
        if rows:
            return rows

        rows = [
            item
            for item in cls._iter_dict_nodes(payload)
            if cls.extract_conversation_id(item)
        ]
        return rows

    @classmethod
    def extract_conversation_id(cls, row: Any) -> Optional[str]:
        if not isinstance(row, dict):
            return None
        for key in (
            "conversationId",
            "conversation_id",
            "conversationID",
            "id",
        ):
            value = row.get(key)
            normalized = cls._normalize_conversation_id(value)
            if normalized:
                return normalized
        # Nested conversation envelope fallback.
        nested = row.get("conversation")
        if isinstance(nested, dict):
            return cls.extract_conversation_id(nested)
        return None

    @staticmethod
    def _normalize_conversation_id(value: Any) -> Optional[str]:
        normalized = str(value or "").strip().lower()
        if not normalized:
            return None
        # Keep deterministic and conservative: require UUID-like shape.
        if len(normalized) >= 8 and normalized.count("-") >= 1:
            return normalized
        return None

    @classmethod
    def _iter_dict_nodes(cls, payload: Any):
        if isinstance(payload, dict):
            yield payload
            for value in payload.values():
                yield from cls._iter_dict_nodes(value)
            return
        if isinstance(payload, list):
            for item in payload:
                yield from cls._iter_dict_nodes(item)
