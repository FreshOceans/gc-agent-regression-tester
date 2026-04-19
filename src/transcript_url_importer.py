"""Transcript URL import helpers for transcript seeding workflows."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Optional
from urllib.parse import urlparse, urlunparse

import requests


class TranscriptUrlImportError(ValueError):
    """Raised when transcript URL import cannot be completed safely."""


@dataclass
class TranscriptUrlFetchResult:
    """Result of fetching a transcript payload by URL."""

    source_url: str
    resolved_url: str
    payload: Any
    fetch_chain: list[str]
    followed_wrapper_url: bool


def redact_url_for_display(url: str) -> str:
    """Redact query/fragments from URL for safe UI/status display."""
    parsed = urlparse(str(url or "").strip())
    if not parsed.scheme or not parsed.netloc:
        return str(url or "").strip()
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path or "/", "", "", ""))


def normalize_allowlist_domains(domains: Iterable[str]) -> list[str]:
    """Normalize allowlist domain tokens for suffix matching."""
    normalized: list[str] = []
    for raw in domains:
        token = str(raw or "").strip().lower()
        if not token:
            continue
        if "://" in token:
            parsed = urlparse(token)
            token = (parsed.hostname or "").strip().lower()
        if token.startswith("*."):
            token = token[2:]
        if ":" in token:
            token = token.split(":", 1)[0].strip()
        token = token.lstrip(".")
        if token and token not in normalized:
            normalized.append(token)
    return normalized


def is_url_allowed(url: str, allowlist_domains: Iterable[str]) -> bool:
    """Return True when URL is HTTPS and host matches allowlist."""
    parsed = urlparse(str(url or "").strip())
    if parsed.scheme.lower() != "https":
        return False
    host = (parsed.hostname or "").strip().lower()
    if not host:
        return False

    allowlist = normalize_allowlist_domains(allowlist_domains)
    for token in allowlist:
        if host == token or host.endswith(f".{token}"):
            return True
    return False


class TranscriptUrlImportService:
    """Fetch transcript JSON safely from user-provided URLs."""

    def __init__(
        self,
        *,
        allowlist_domains: Iterable[str],
        timeout_seconds: int = 30,
        max_bytes: int = 5_000_000,
    ):
        self.allowlist_domains = normalize_allowlist_domains(allowlist_domains)
        if not self.allowlist_domains:
            raise TranscriptUrlImportError("Transcript URL allowlist cannot be empty.")
        self.timeout_seconds = max(1, int(timeout_seconds))
        self.max_bytes = max(1024, int(max_bytes))

    def fetch_transcript_json(self, source_url: str) -> TranscriptUrlFetchResult:
        """Fetch transcript JSON from URL, following one wrapper URL when present."""
        normalized_source = self._validate_and_normalize_url(source_url)
        first_payload = self._fetch_json_once(normalized_source)
        fetch_chain = [normalized_source]
        resolved_url = normalized_source
        followed_wrapper = False

        pointer = self._extract_wrapper_url(first_payload)
        if pointer:
            normalized_pointer = self._validate_and_normalize_url(pointer)
            if normalized_pointer != normalized_source:
                first_payload = self._fetch_json_once(normalized_pointer)
                fetch_chain.append(normalized_pointer)
                resolved_url = normalized_pointer
                followed_wrapper = True

        return TranscriptUrlFetchResult(
            source_url=normalized_source,
            resolved_url=resolved_url,
            payload=first_payload,
            fetch_chain=fetch_chain,
            followed_wrapper_url=followed_wrapper,
        )

    def _validate_and_normalize_url(self, url: str) -> str:
        normalized = str(url or "").strip()
        parsed = urlparse(normalized)
        if parsed.scheme.lower() != "https":
            raise TranscriptUrlImportError("Transcript URL must use HTTPS.")
        if not parsed.netloc:
            raise TranscriptUrlImportError("Transcript URL host is missing.")
        if not is_url_allowed(normalized, self.allowlist_domains):
            raise TranscriptUrlImportError(
                "Transcript URL host is not allowed by the configured allowlist."
            )
        return normalized

    def _fetch_json_once(self, url: str) -> Any:
        try:
            response = requests.get(url, timeout=self.timeout_seconds, stream=True)
            response.raise_for_status()
        except requests.RequestException as e:
            raise TranscriptUrlImportError(f"Could not fetch transcript URL: {e}") from e

        content = self._read_limited_content(response)
        try:
            text = content.decode(response.encoding or "utf-8")
        except UnicodeDecodeError:
            text = content.decode("utf-8", errors="replace")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as e:
            raise TranscriptUrlImportError(
                f"Transcript URL did not return valid JSON: {e}"
            ) from e
        if not isinstance(payload, (dict, list)):
            raise TranscriptUrlImportError(
                "Transcript URL JSON payload must be an object or array."
            )
        return payload

    def _read_limited_content(self, response: requests.Response) -> bytes:
        chunks: list[bytes] = []
        total = 0
        for chunk in response.iter_content(chunk_size=65536):
            if not chunk:
                continue
            total += len(chunk)
            if total > self.max_bytes:
                raise TranscriptUrlImportError(
                    f"Transcript payload exceeded max size limit ({self.max_bytes} bytes)."
                )
            chunks.append(chunk)
        return b"".join(chunks)

    def _extract_wrapper_url(self, payload: Any) -> Optional[str]:
        if self._looks_like_transcript_payload(payload):
            return None
        return self._find_candidate_url(payload)

    def _looks_like_transcript_payload(self, payload: Any) -> bool:
        if isinstance(payload, list):
            return True
        if not isinstance(payload, dict):
            return False
        marker_keys = {
            "messages",
            "participants",
            "conversations",
            "conversation_id",
            "conversationid",
            "transcript",
            "transcripts",
        }
        normalized_keys = {str(key or "").strip().lower() for key in payload}
        return any(marker in normalized_keys for marker in marker_keys)

    def _find_candidate_url(self, node: Any, depth: int = 0) -> Optional[str]:
        if depth > 4:
            return None
        if isinstance(node, str):
            candidate = node.strip()
            if candidate.lower().startswith("https://"):
                return candidate
            return None
        if isinstance(node, list):
            for item in node:
                candidate = self._find_candidate_url(item, depth + 1)
                if candidate:
                    return candidate
            return None
        if not isinstance(node, dict):
            return None

        prioritized_keys = (
            "url",
            "transcripturl",
            "transcript_url",
            "downloadurl",
            "download_url",
            "s3url",
            "s3_url",
            "href",
            "uri",
            "link",
        )

        for key in prioritized_keys:
            value = node.get(key)
            candidate = self._find_candidate_url(value, depth + 1)
            if candidate:
                return candidate

        for key, value in node.items():
            key_text = str(key or "").strip().lower()
            if "url" in key_text or key_text.endswith("href") or key_text.endswith("uri"):
                candidate = self._find_candidate_url(value, depth + 1)
                if candidate:
                    return candidate

        for value in node.values():
            candidate = self._find_candidate_url(value, depth + 1)
            if candidate:
                return candidate
        return None
