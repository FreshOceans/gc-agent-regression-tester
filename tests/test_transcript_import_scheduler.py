"""Unit tests for transcript import scheduler."""

from datetime import datetime, timedelta, timezone

from src.transcript_import_scheduler import (
    TranscriptImportScheduler,
    compute_next_daily_run_utc,
)


def test_compute_next_daily_run_utc_returns_future_time():
    now = datetime(2026, 4, 19, 1, 0, tzinfo=timezone.utc)
    next_run = compute_next_daily_run_utc(
        time_hhmm="02:00",
        timezone_name="UTC",
        now_utc=now,
    )
    assert next_run > now
    assert next_run.hour == 2


def test_scheduler_runs_job_when_due():
    calls = []
    settings = {
        "enabled": True,
        "time_hhmm": "02:00",
        "timezone_name": "UTC",
        "max_ids": 10,
        "filter_json": "{}",
        "language_code": "en",
    }
    scheduler = TranscriptImportScheduler(
        settings_getter=lambda: settings,
        run_job=lambda payload: calls.append(payload.copy()),
        poll_interval_seconds=60,
    )
    scheduler._last_settings_signature = (
        True,
        "02:00",
        "UTC",
        10,
        "{}",
        "en",
    )
    scheduler._next_run_utc = datetime.now(timezone.utc) - timedelta(seconds=1)
    scheduler._run_pending_once()
    assert len(calls) == 1
