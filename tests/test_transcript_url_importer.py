"""Unit tests for transcript URL import helpers."""

from __future__ import annotations

import json

import pytest

from src.transcript_url_importer import (
    TranscriptUrlImportError,
    TranscriptUrlImportService,
    is_url_allowed,
    redact_url_for_display,
)


class _FakeResponse:
    def __init__(self, payload, *, status_code: int = 200, encoding: str = "utf-8"):
        self._payload = payload
        self.status_code = status_code
        self.encoding = encoding
        self._body = json.dumps(payload).encode("utf-8")

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size: int = 65536):
        for index in range(0, len(self._body), chunk_size):
            yield self._body[index : index + chunk_size]


def test_redact_url_for_display_strips_query_and_fragment():
    url = "https://api-downloads.cac1.pure.cloud/path/file.json?token=abc#frag"
    assert redact_url_for_display(url) == "https://api-downloads.cac1.pure.cloud/path/file.json"


def test_is_url_allowed_matches_suffix_tokens():
    assert is_url_allowed(
        "https://api-downloads.cac1.pure.cloud/path",
        ["pure.cloud"],
    )
    assert is_url_allowed(
        "https://apps.mypurecloud.com/path",
        ["mypurecloud.com"],
    )
    assert not is_url_allowed(
        "https://example.org/path",
        ["pure.cloud", "mypurecloud.com"],
    )


def test_fetch_transcript_json_direct_success(monkeypatch):
    calls = []

    def _fake_get(url, timeout, stream):
        calls.append((url, timeout, stream))
        return _FakeResponse(
            {"conversations": [{"messages": [{"speaker": "customer", "text": "hello"}]}]}
        )

    monkeypatch.setattr("src.transcript_url_importer.requests.get", _fake_get)
    service = TranscriptUrlImportService(
        allowlist_domains=["pure.cloud"],
        timeout_seconds=10,
        max_bytes=10000,
    )

    result = service.fetch_transcript_json(
        "https://api-downloads.cac1.pure.cloud/test.json?sig=abc"
    )

    assert result.source_url.startswith("https://api-downloads.cac1.pure.cloud/")
    assert result.resolved_url == result.source_url
    assert result.followed_wrapper_url is False
    assert isinstance(result.payload, dict)
    assert len(calls) == 1


def test_fetch_transcript_json_follows_one_wrapper_url(monkeypatch):
    requested = []

    def _fake_get(url, timeout, stream):
        requested.append(url)
        if "wrapper" in url:
            return _FakeResponse(
                {
                    "transcriptsS3UrlResponse": "https://api-downloads.cac1.pure.cloud/transcript.json",
                }
            )
        return _FakeResponse(
            {"conversations": [{"messages": [{"speaker": "customer", "text": "hello"}]}]}
        )

    monkeypatch.setattr("src.transcript_url_importer.requests.get", _fake_get)
    service = TranscriptUrlImportService(
        allowlist_domains=["pure.cloud"],
        timeout_seconds=10,
        max_bytes=10000,
    )

    result = service.fetch_transcript_json(
        "https://api-downloads.cac1.pure.cloud/wrapper.json"
    )

    assert result.followed_wrapper_url is True
    assert len(result.fetch_chain) == 2
    assert result.resolved_url.endswith("/transcript.json")
    assert requested[0].endswith("/wrapper.json")
    assert requested[1].endswith("/transcript.json")


def test_fetch_transcript_json_rejects_disallowed_host():
    service = TranscriptUrlImportService(
        allowlist_domains=["pure.cloud"],
        timeout_seconds=10,
        max_bytes=10000,
    )
    with pytest.raises(TranscriptUrlImportError):
        service.fetch_transcript_json("https://example.org/transcript.json")


def test_fetch_transcript_json_rejects_payload_over_max(monkeypatch):
    def _fake_get(url, timeout, stream):
        return _FakeResponse({"conversations": [{"messages": [{"text": "x" * 4000}]}]})

    monkeypatch.setattr("src.transcript_url_importer.requests.get", _fake_get)
    service = TranscriptUrlImportService(
        allowlist_domains=["pure.cloud"],
        timeout_seconds=10,
        max_bytes=1024,
    )
    with pytest.raises(TranscriptUrlImportError):
        service.fetch_transcript_json("https://api-downloads.cac1.pure.cloud/transcript.json")
