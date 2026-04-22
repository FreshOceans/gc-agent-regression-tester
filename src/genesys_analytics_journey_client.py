"""Genesys Bot Reporting Turns client for analytics journey regression."""

from __future__ import annotations

import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import requests

from .models import (
    ANALYTICS_AUTH_MODE_CLIENT_CREDENTIALS,
    normalize_analytics_auth_mode,
)

# Keep advanced query passthrough intentionally narrow/safe.
_ALLOWED_EXTRA_QUERY_KEYS = {
    "pagenumber": "pageNumber",
    "pagesize": "pageSize",
}


class GenesysAnalyticsJourneyError(Exception):
    """Raised when Analytics journey ingestion fails."""

    def __init__(self, message: str, *, metadata: Optional[dict[str, Any]] = None):
        super().__init__(message)
        self.metadata: dict[str, Any] = dict(metadata or {})


class GenesysAnalyticsJourneyClient:
    """Client for Bot Reporting Turns ingestion and conversation grouping."""

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

    @staticmethod
    def _truncate_text(value: Any, *, max_chars: int = 600) -> str:
        raw = str(value or "")
        if len(raw) <= max_chars:
            return raw
        return raw[:max_chars] + "...<truncated>"

    @classmethod
    def _build_http_error_metadata(
        cls,
        *,
        response: Optional[requests.Response],
        method: str,
        path: str,
        url: str,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "method": str(method or "").upper(),
            "path": str(path or ""),
            "url": str(url or ""),
        }
        if response is None:
            return metadata

        metadata["status_code"] = int(getattr(response, "status_code", 0) or 0)
        headers = getattr(response, "headers", {}) or {}
        lower_headers = {str(key).lower(): str(value) for key, value in headers.items()}
        correlation_id = (
            lower_headers.get("inin-correlation-id")
            or lower_headers.get("x-correlation-id")
            or lower_headers.get("x-request-id")
            or lower_headers.get("request-id")
            or lower_headers.get("trace-id")
            or lower_headers.get("traceparent")
        )
        if correlation_id:
            metadata["correlation_id"] = correlation_id
        content_type = lower_headers.get("content-type")
        if content_type:
            metadata["content_type"] = content_type
        retry_after = lower_headers.get("retry-after")
        if retry_after:
            metadata["retry_after"] = retry_after

        body_text: Optional[str] = None
        try:
            body_text = response.text
        except Exception:
            body_text = None
        if body_text:
            metadata["response_body_excerpt"] = cls._truncate_text(body_text)
        return metadata

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
            metadata = self._build_http_error_metadata(
                response=getattr(e, "response", None),
                method="POST",
                path="/oauth/token",
                url=self._oauth_url,
            )
            raise GenesysAnalyticsJourneyError(
                f"OAuth token request failed for region '{self.region}': {e}",
                metadata=metadata,
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
        stop_requested: Optional[Callable[[], bool]] = None,
    ) -> dict[str, Any]:
        if self._is_stop_requested(stop_requested):
            raise GenesysAnalyticsJourneyError("Request interrupted by stop request")
        token = self._get_access_token()
        if path.startswith("http://") or path.startswith("https://"):
            url = path
        else:
            normalized_path = path if path.startswith("/") else f"/{path}"
            url = f"{self._api_base_url}{normalized_path}"
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }
        if method.strip().upper() in {"POST", "PUT", "PATCH"}:
            headers["Content-Type"] = "application/json"
        context = dict(request_context or {})

        last_error: Optional[Exception] = None
        last_error_metadata: dict[str, Any] = {}
        for attempt in range(1, self.retries + 1):
            if self._is_stop_requested(stop_requested):
                self._emit_observer(
                    observer,
                    {
                        "event": "request_stopped",
                        "method": method,
                        "path": path,
                        "attempt": attempt,
                        "max_attempts": self.retries,
                        **context,
                    },
                )
                raise GenesysAnalyticsJourneyError(
                    f"Request interrupted by stop request for {path}"
                )
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
                    if not self._sleep_with_stop_support(
                        backoff_seconds,
                        stop_requested,
                    ):
                        raise GenesysAnalyticsJourneyError(
                            f"Request interrupted by stop request for {path}"
                        )
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
                response_obj = getattr(e, "response", None)
                last_error_metadata = self._build_http_error_metadata(
                    response=response_obj,
                    method=method,
                    path=path,
                    url=url,
                )
                last_error_metadata["attempt"] = int(attempt)
                last_error_metadata["max_attempts"] = int(self.retries)
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
                    if not self._sleep_with_stop_support(
                        backoff_seconds,
                        stop_requested,
                    ):
                        raise GenesysAnalyticsJourneyError(
                            f"Request interrupted by stop request for {path}"
                        )
                    continue
                break
            except ValueError as e:
                last_error = e
                last_error_metadata = {
                    "method": str(method or "").upper(),
                    "path": str(path or ""),
                    "url": str(url or ""),
                    "attempt": int(attempt),
                    "max_attempts": int(self.retries),
                }
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
        raise GenesysAnalyticsJourneyError(
            f"Request failed for {path}: {last_error}",
            metadata=last_error_metadata,
        )

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
        next_uri: Optional[str] = None,
        observer: Optional[Callable[[dict[str, Any]], None]] = None,
        stop_requested: Optional[Callable[[], bool]] = None,
    ) -> dict[str, Any]:
        normalized_interval = str(interval or "").strip()
        if not normalized_interval:
            raise GenesysAnalyticsJourneyError("Analytics interval is required")
        normalized_flow_id = str(bot_flow_id or "").strip()
        if not normalized_flow_id:
            raise GenesysAnalyticsJourneyError("Bot Flow ID is required")

        _ = language_filter  # language filtering is applied deterministically client-side.

        if next_uri:
            return self._request_json(
                method="GET",
                path=next_uri,
                observer=observer,
                request_context={"page_number": max(1, int(page_number))},
                stop_requested=stop_requested,
            )

        params: dict[str, Any] = {
            "interval": normalized_interval,
            "pageSize": max(1, min(int(page_size), self.page_size_cap)),
            "pageNumber": max(1, int(page_number)),
        }
        if divisions:
            normalized_divisions = [
                str(token).strip()
                for token in divisions
                if str(token).strip()
            ]
            if normalized_divisions:
                params["divisions"] = ",".join(normalized_divisions)

        sanitized_extra, _ignored = self.sanitize_extra_query_params(extra_params)
        for key, value in sanitized_extra.items():
            params[key] = value

        return self._request_json(
            method="GET",
            path=(
                "/api/v2/analytics/botflows/"
                f"{normalized_flow_id}/divisions/reportingturns"
            ),
            params=params,
            observer=observer,
            request_context={"page_number": max(1, int(params.get("pageNumber", 1)))},
            stop_requested=stop_requested,
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
        stop_requested: Optional[Callable[[], bool]] = None,
    ) -> dict[str, Any]:
        """Fetch reporting-turn pages and return conversation-grouped turn units."""
        conversation_ids_in_order: list[str] = []
        seen_ids: set[str] = set()
        rows_by_conversation: dict[str, list[dict[str, Any]]] = defaultdict(list)
        page_payloads: list[dict[str, Any]] = []

        max_items = max(1, int(max_conversations))
        safe_page_size = max(1, min(int(page_size), self.page_size_cap))
        sanitized_extra, ignored_keys = self.sanitize_extra_query_params(extra_params)
        page_num = max(1, int(sanitized_extra.get("pageNumber", 1)))
        if "pageSize" in sanitized_extra:
            safe_page_size = max(1, min(int(sanitized_extra["pageSize"]), self.page_size_cap))

        next_uri: Optional[str] = None
        seen_next_uris: set[str] = set()

        while True:
            if self._is_stop_requested(stop_requested):
                raise GenesysAnalyticsJourneyError(
                    "Analytics ingestion interrupted by stop request"
                )
            page_started_at = time.monotonic()
            self._emit_observer(
                observer,
                {
                        "event": "page_fetch_started",
                        "page_number": page_num,
                        "max_conversations": max_items,
                        "current_unique_conversations": len(conversation_ids_in_order),
                    },
                )
            payload = self.fetch_reporting_turns_page(
                bot_flow_id=bot_flow_id,
                interval=interval,
                page_size=safe_page_size,
                page_number=page_num,
                divisions=divisions,
                language_filter=language_filter,
                extra_params=sanitized_extra,
                next_uri=next_uri,
                observer=observer,
                stop_requested=stop_requested,
            )
            page_payloads.append(payload)
            page_rows = self.extract_rows(payload)
            ids_before = len(conversation_ids_in_order)

            for row in page_rows:
                conversation_id = self.extract_conversation_id(row)
                if not conversation_id:
                    continue
                rows_by_conversation[conversation_id].append(row)
                if conversation_id in seen_ids:
                    continue
                seen_ids.add(conversation_id)
                conversation_ids_in_order.append(conversation_id)

            selected_ids, language_filter_stats = self.filter_conversation_ids_by_language(
                rows_by_conversation,
                conversation_ids_in_order,
                language_filter,
                limit=max_items,
            )

            self._emit_observer(
                observer,
                {
                    "event": "page_fetch_completed",
                    "page_number": page_num,
                    "rows_count": len(page_rows),
                    "new_unique_conversations": max(
                        0,
                        len(conversation_ids_in_order) - ids_before,
                    ),
                    "total_unique_conversations": len(conversation_ids_in_order),
                    "eligible_unique_conversations": int(
                        language_filter_stats.get("eligible_conversations", 0)
                    ),
                    "selected_unique_conversations": int(
                        language_filter_stats.get("selected_conversations", 0)
                    ),
                    "duration_ms": round(
                        (time.monotonic() - page_started_at) * 1000.0,
                        2,
                    ),
                },
            )

            if len(selected_ids) >= max_items:
                break

            candidate_next_uri = self.extract_next_uri(payload)
            if candidate_next_uri and candidate_next_uri not in seen_next_uris:
                seen_next_uris.add(candidate_next_uri)
                next_uri = candidate_next_uri
                page_num += 1
                continue
            next_uri = None

            if not page_rows or ids_before == len(conversation_ids_in_order):
                break
            if len(page_rows) < safe_page_size:
                break
            page_num += 1

        selected_ids, language_filter_stats = self.filter_conversation_ids_by_language(
            rows_by_conversation,
            conversation_ids_in_order,
            language_filter,
            limit=max_items,
        )
        units = []
        for conversation_id in selected_ids:
            sorted_rows = sorted(
                rows_by_conversation.get(conversation_id, []),
                key=self._row_sort_key,
            )
            units.append(
                {
                    "conversation_id": conversation_id,
                    "rows": sorted_rows,
                }
            )

        return {
            "conversations": units,
            "page_payloads": page_payloads,
            "page_count": len(page_payloads),
            "ignored_query_params": ignored_keys,
            "applied_query_params": sorted(sanitized_extra.keys()),
            "language_filter_stats": language_filter_stats,
        }

    @staticmethod
    def sanitize_extra_query_params(
        extra_params: Optional[dict[str, Any]],
    ) -> tuple[dict[str, Any], list[str]]:
        if not isinstance(extra_params, dict):
            return {}, []
        sanitized: dict[str, Any] = {}
        ignored: list[str] = []
        for raw_key, raw_value in extra_params.items():
            key = str(raw_key or "").strip()
            if not key:
                continue
            normalized = key.lower()
            target = _ALLOWED_EXTRA_QUERY_KEYS.get(normalized)
            if not target:
                ignored.append(key)
                continue
            if target in {"pageNumber", "pageSize"}:
                try:
                    parsed = int(raw_value)
                except (TypeError, ValueError):
                    ignored.append(key)
                    continue
                if parsed < 1:
                    ignored.append(key)
                    continue
                sanitized[target] = parsed
                continue
            if isinstance(raw_value, (str, int, float, bool)):
                sanitized[target] = raw_value
            else:
                ignored.append(key)
        return sanitized, sorted(set(ignored))

    @staticmethod
    def extract_next_uri(payload: dict[str, Any]) -> Optional[str]:
        direct = payload.get("nextUri")
        if isinstance(direct, str) and direct.strip():
            return direct.strip()
        links = payload.get("links")
        if isinstance(links, dict):
            next_link = links.get("next")
            if isinstance(next_link, dict):
                href = next_link.get("href")
                if isinstance(href, str) and href.strip():
                    return href.strip()
            if isinstance(next_link, str) and next_link.strip():
                return next_link.strip()
        next_obj = payload.get("next")
        if isinstance(next_obj, dict):
            uri = next_obj.get("uri")
            if isinstance(uri, str) and uri.strip():
                return uri.strip()
        return None

    @staticmethod
    def row_matches_language(row: dict[str, Any], language_filter: Optional[str]) -> bool:
        classification = GenesysAnalyticsJourneyClient.classify_row_language(
            row,
            language_filter,
        )
        return classification["status"] == "match"

    @staticmethod
    def _normalize_language_tag(value: Any) -> str:
        return str(value or "").strip().lower().replace("_", "-")

    @classmethod
    def _language_candidate_matches_target(
        cls,
        candidate: str,
        target: str,
    ) -> bool:
        normalized_candidate = cls._normalize_language_tag(candidate)
        normalized_target = cls._normalize_language_tag(target)
        if not normalized_candidate or not normalized_target:
            return False
        return (
            normalized_candidate == normalized_target
            or normalized_candidate.startswith(f"{normalized_target}-")
            or normalized_target.startswith(f"{normalized_candidate}-")
        )

    @classmethod
    def extract_language_candidates(cls, row: dict[str, Any]) -> set[str]:
        candidates: set[str] = set()
        for key in (
            "language",
            "locale",
            "startingLanguage",
            "endingLanguage",
            "flowLanguage",
        ):
            value = row.get(key)
            normalized = cls._normalize_language_tag(value)
            if normalized:
                candidates.add(normalized)

        nested_conversation = row.get("conversation")
        if isinstance(nested_conversation, dict):
            for key in ("language", "locale", "startingLanguage", "endingLanguage"):
                value = nested_conversation.get(key)
                normalized = cls._normalize_language_tag(value)
                if normalized:
                    candidates.add(normalized)
        return candidates

    @classmethod
    def classify_row_language(
        cls,
        row: dict[str, Any],
        language_filter: Optional[str],
    ) -> dict[str, Any]:
        target = cls._normalize_language_tag(language_filter)
        candidates = cls.extract_language_candidates(row)
        if not target:
            return {
                "status": "match",
                "candidates": sorted(candidates),
                "matched_candidates": sorted(candidates),
            }
        if not candidates:
            return {
                "status": "unknown",
                "candidates": [],
                "matched_candidates": [],
            }
        matched = sorted(
            candidate
            for candidate in candidates
            if cls._language_candidate_matches_target(candidate, target)
        )
        if matched:
            return {
                "status": "match",
                "candidates": sorted(candidates),
                "matched_candidates": matched,
            }
        return {
            "status": "mismatch",
            "candidates": sorted(candidates),
            "matched_candidates": [],
        }

    @classmethod
    def summarize_conversation_language(
        cls,
        rows: list[dict[str, Any]],
        language_filter: Optional[str],
    ) -> dict[str, Any]:
        target = cls._normalize_language_tag(language_filter)
        summary = {
            "language_filter": target or None,
            "explicit_match_rows": 0,
            "explicit_mismatch_rows": 0,
            "unknown_rows": 0,
            "matched_candidates": set(),
            "mismatched_candidates": set(),
        }
        if not target:
            summary["eligible"] = True
            return summary

        for row in rows:
            classification = cls.classify_row_language(row, target)
            status = classification["status"]
            candidates = set(classification.get("candidates") or [])
            matched_candidates = set(classification.get("matched_candidates") or [])
            if status == "match":
                summary["explicit_match_rows"] += 1
                summary["matched_candidates"].update(matched_candidates or candidates)
            elif status == "mismatch":
                summary["explicit_mismatch_rows"] += 1
                summary["mismatched_candidates"].update(candidates)
            else:
                summary["unknown_rows"] += 1

        summary["eligible"] = (
            summary["explicit_match_rows"] > 0
            and summary["explicit_mismatch_rows"] == 0
        )
        return summary

    @classmethod
    def filter_conversation_ids_by_language(
        cls,
        rows_by_conversation: dict[str, list[dict[str, Any]]],
        conversation_ids_in_order: list[str],
        language_filter: Optional[str],
        *,
        limit: Optional[int] = None,
    ) -> tuple[list[str], dict[str, Any]]:
        target = cls._normalize_language_tag(language_filter)
        stats: dict[str, Any] = {
            "language_filter": target or None,
            "eligible_conversations": 0,
            "selected_conversations": 0,
            "excluded_missing_language_conversations": 0,
            "excluded_mismatched_conversations": 0,
        }
        if not target:
            selected = list(conversation_ids_in_order[:limit] if limit is not None else conversation_ids_in_order)
            stats["eligible_conversations"] = len(conversation_ids_in_order)
            stats["selected_conversations"] = len(selected)
            return selected, stats

        eligible_ids: list[str] = []
        for conversation_id in conversation_ids_in_order:
            rows = rows_by_conversation.get(conversation_id, [])
            summary = cls.summarize_conversation_language(rows, target)
            if summary["eligible"]:
                stats["eligible_conversations"] += 1
                eligible_ids.append(conversation_id)
                continue
            if summary["explicit_mismatch_rows"] > 0:
                stats["excluded_mismatched_conversations"] += 1
            else:
                stats["excluded_missing_language_conversations"] += 1

        selected = list(eligible_ids[:limit] if limit is not None else eligible_ids)
        stats["selected_conversations"] = len(selected)
        return selected, stats

    @classmethod
    def filter_rows_by_language(
        cls,
        rows: list[dict[str, Any]],
        language_filter: Optional[str],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        rows_by_conversation: dict[str, list[dict[str, Any]]] = defaultdict(list)
        conversation_ids_in_order: list[str] = []
        seen_ids: set[str] = set()

        for index, row in enumerate(rows):
            conversation_id = cls.extract_conversation_id(row) or f"__row__{index}"
            rows_by_conversation[conversation_id].append(row)
            if conversation_id in seen_ids:
                continue
            seen_ids.add(conversation_id)
            conversation_ids_in_order.append(conversation_id)

        selected_ids, stats = cls.filter_conversation_ids_by_language(
            rows_by_conversation,
            conversation_ids_in_order,
            language_filter,
        )
        filtered_rows: list[dict[str, Any]] = []
        for conversation_id in selected_ids:
            filtered_rows.extend(rows_by_conversation.get(conversation_id, []))
        return filtered_rows, stats

    @classmethod
    def extract_rows(cls, payload: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract likely row dicts from a Bot Reporting Turns payload."""
        rows: list[dict[str, Any]] = []
        for key in (
            "entities",
            "reportingTurns",
            "turns",
            "results",
            "data",
            "items",
        ):
            value = payload.get(key)
            if isinstance(value, list):
                rows.extend(item for item in value if isinstance(item, dict))
        if rows:
            return rows

        return [
            item
            for item in cls._iter_dict_nodes(payload)
            if cls.extract_conversation_id(item)
        ]

    @classmethod
    def extract_conversation_id(cls, row: Any) -> Optional[str]:
        if not isinstance(row, dict):
            return None
        for key in (
            "conversationId",
            "conversation_id",
            "conversationID",
            "conversation.id",
            "id",
        ):
            value = row.get(key)
            normalized = cls._normalize_conversation_id(value)
            if normalized:
                return normalized

        nested = row.get("conversation")
        if isinstance(nested, dict):
            nested_value = nested.get("id") or nested.get("conversationId")
            normalized = cls._normalize_conversation_id(nested_value)
            if normalized:
                return normalized
        return None

    @staticmethod
    def _normalize_conversation_id(value: Any) -> Optional[str]:
        normalized = str(value or "").strip().lower()
        if not normalized:
            return None
        if len(normalized) >= 8 and normalized.count("-") >= 1:
            return normalized
        return None

    @classmethod
    def _row_sort_key(cls, row: dict[str, Any]) -> tuple[float, str]:
        for key in (
            "dateCreated",
            "dateCompleted",
            "timestamp",
            "time",
        ):
            parsed = cls._parse_timestamp(row.get(key))
            if parsed is not None:
                return parsed.timestamp(), key
        return 0.0, ""

    @staticmethod
    def _parse_timestamp(value: Any) -> Optional[datetime]:
        if value is None:
            return None
        raw = str(value).strip()
        if not raw:
            return None
        try:
            if raw.endswith("Z"):
                raw = raw[:-1] + "+00:00"
            parsed = datetime.fromisoformat(raw)
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed
        except Exception:
            return None

    @staticmethod
    def _is_stop_requested(stop_requested: Optional[Callable[[], bool]]) -> bool:
        return bool(callable(stop_requested) and stop_requested())

    @staticmethod
    def _sleep_with_stop_support(
        total_seconds: float,
        stop_requested: Optional[Callable[[], bool]],
    ) -> bool:
        remaining = max(0.0, float(total_seconds))
        while remaining > 0:
            if callable(stop_requested) and stop_requested():
                return False
            sleep_for = min(0.1, remaining)
            time.sleep(sleep_for)
            remaining -= sleep_for
        return not (callable(stop_requested) and stop_requested())

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
    def _iter_dict_nodes(cls, payload: Any):
        if isinstance(payload, dict):
            yield payload
            for value in payload.values():
                yield from cls._iter_dict_nodes(value)
            return
        if isinstance(payload, list):
            for item in payload:
                yield from cls._iter_dict_nodes(item)
