"""Tests for persistent Model Warm Up scheduling."""

import json
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from src.model_warmup_scheduler import (
    ModelWarmupScheduleStore,
    compute_next_model_warmup_run_utc,
    validate_schedule_timezone_name,
)


def _as_local(value, timezone_name: str):
    return value.astimezone(ZoneInfo(timezone_name))


def test_hourly_schedule_next_run_uses_selected_minute():
    next_run = compute_next_model_warmup_run_utc(
        {
            "cadence": "hourly",
            "timezone_name": "UTC",
            "minute": 30,
        },
        now_utc=datetime(2026, 4, 27, 10, 15, tzinfo=timezone.utc),
    )

    assert next_run == datetime(2026, 4, 27, 10, 30, tzinfo=timezone.utc)


def test_daily_schedule_rolls_to_next_day_when_time_passed():
    next_run = compute_next_model_warmup_run_utc(
        {
            "cadence": "daily",
            "timezone_name": "UTC",
            "time_hhmm": "09:00",
        },
        now_utc=datetime(2026, 4, 27, 10, 15, tzinfo=timezone.utc),
    )

    assert next_run == datetime(2026, 4, 28, 9, 0, tzinfo=timezone.utc)


def test_weekly_schedule_uses_selected_weekday_and_time():
    timezone_name = "America/New_York"
    next_run = compute_next_model_warmup_run_utc(
        {
            "cadence": "weekly",
            "timezone_name": timezone_name,
            "weekday": 2,
            "time_hhmm": "08:30",
        },
        now_utc=datetime(2026, 4, 27, 13, 0, tzinfo=timezone.utc),
    )

    local = _as_local(next_run, timezone_name)
    assert local.weekday() == 2
    assert local.hour == 8
    assert local.minute == 30


def test_monthly_schedule_clamps_to_last_day_of_short_month():
    timezone_name = "America/New_York"
    next_run = compute_next_model_warmup_run_utc(
        {
            "cadence": "monthly",
            "timezone_name": timezone_name,
            "day_of_month": 31,
            "time_hhmm": "10:00",
        },
        now_utc=datetime(2026, 2, 1, 12, 0, tzinfo=timezone.utc),
    )

    local = _as_local(next_run, timezone_name)
    assert local.year == 2026
    assert local.month == 2
    assert local.day == 28
    assert local.hour == 10
    assert local.minute == 0


def test_daily_schedule_respects_future_start_date():
    next_run = compute_next_model_warmup_run_utc(
        {
            "cadence": "daily",
            "timezone_name": "UTC",
            "time_hhmm": "09:00",
            "start_date": "2026-04-29",
            "end_date": "2026-05-02",
        },
        now_utc=datetime(2026, 4, 27, 10, 15, tzinfo=timezone.utc),
    )

    assert next_run == datetime(2026, 4, 29, 9, 0, tzinfo=timezone.utc)


def test_weekly_schedule_stays_inside_inclusive_date_range():
    timezone_name = "America/New_York"
    next_run = compute_next_model_warmup_run_utc(
        {
            "cadence": "weekly",
            "timezone_name": timezone_name,
            "weekday": 2,
            "time_hhmm": "08:30",
            "start_date": "2026-04-28",
            "end_date": "2026-04-30",
        },
        now_utc=datetime(2026, 4, 27, 13, 0, tzinfo=timezone.utc),
    )

    local = _as_local(next_run, timezone_name)
    assert local.date().isoformat() == "2026-04-29"


def test_schedule_returns_none_after_end_date():
    next_run = compute_next_model_warmup_run_utc(
        {
            "cadence": "hourly",
            "timezone_name": "UTC",
            "minute": 30,
            "start_date": "2026-04-20",
            "end_date": "2026-04-21",
        },
        now_utc=datetime(2026, 4, 22, 10, 15, tzinfo=timezone.utc),
    )

    assert next_run is None


def test_invalid_timezone_is_rejected():
    with pytest.raises(ValueError):
        validate_schedule_timezone_name("Not/AZone")


def test_schedule_store_save_load_disable(tmp_path):
    store = ModelWarmupScheduleStore(history_dir=str(tmp_path))

    saved = store.save_schedule(
        {
            "cadence": "daily",
            "timezone_name": "UTC",
            "time_hhmm": "02:00",
            "start_date": "2099-04-27",
            "end_date": "2099-04-30",
            "run_request": {
                "deployment_id": "deploy-123",
                "region": "usw2.pure.cloud",
                "attempt_count": 10,
            },
        }
    )

    assert saved["enabled"] is True
    assert saved["schedule_id"]
    assert saved["next_run_utc"]
    assert saved["start_date"] == "2099-04-27"
    assert saved["end_date"] == "2099-04-30"
    assert saved["date_range_status"] == "pending"
    assert saved["last_status"]["status"] == "scheduled"
    assert saved["scheduled_warmups"][0]["status"] == "scheduled"
    assert saved["scheduled_warmups"][0]["start_date"] == "2099-04-27"
    assert store.load()["run_request"]["deployment_id"] == "deploy-123"

    disabled = store.disable()
    assert disabled["enabled"] is False
    assert disabled["next_run_utc"] is None
    assert disabled["canceled_at_utc"]
    assert disabled["last_status"]["status"] == "canceled"
    assert disabled["scheduled_warmups"][0]["status"] == "canceled"


def test_schedule_store_load_marks_expired_date_range_completed(tmp_path):
    store = ModelWarmupScheduleStore(history_dir=str(tmp_path))
    store.path.write_text(
        json.dumps(
            {
                "enabled": True,
                "schedule_id": "schedule-123",
                "cadence": "daily",
                "timezone_name": "UTC",
                "time_hhmm": "02:00",
                "start_date": "2026-04-01",
                "end_date": "2026-04-02",
                "run_request": {"deployment_id": "dep", "region": "usw2.pure.cloud"},
            }
        ),
        encoding="utf-8",
    )

    loaded = store.load()

    assert loaded["enabled"] is False
    assert loaded["next_run_utc"] is None
    assert loaded["date_range_status"] == "completed"
    assert loaded["completed_at_utc"]
    assert loaded["last_status"]["status"] == "completed"
    assert loaded["last_status"]["reason"] == "date_range_completed"
