"""Persistent scheduler for Model Warm Up automation."""

from __future__ import annotations

import calendar
import json
import threading
import uuid
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from zoneinfo import ZoneInfo

MODEL_WARMUP_SCHEDULE_CADENCES = {"hourly", "daily", "weekly", "monthly"}
MODEL_WARMUP_SCHEDULE_FILE = "model_warmup_schedule.json"


def normalize_model_warmup_schedule_cadence(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized not in MODEL_WARMUP_SCHEDULE_CADENCES:
        raise ValueError("Model Warm Up schedule cadence must be hourly, daily, weekly, or monthly.")
    return normalized


def parse_schedule_hhmm(value: str) -> tuple[int, int]:
    raw = str(value or "").strip()
    parts = raw.split(":")
    if len(parts) != 2:
        raise ValueError("Model Warm Up schedule time must use HH:MM format.")
    try:
        hour = int(parts[0])
        minute = int(parts[1])
    except ValueError as exc:
        raise ValueError("Model Warm Up schedule time must use HH:MM format.") from exc
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        raise ValueError("Model Warm Up schedule time must use HH:MM format.")
    return hour, minute


def normalize_schedule_minute(value: Any) -> int:
    try:
        minute = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Hourly Model Warm Up minute must be between 0 and 59.") from exc
    if minute < 0 or minute > 59:
        raise ValueError("Hourly Model Warm Up minute must be between 0 and 59.")
    return minute


def normalize_schedule_weekday(value: Any) -> int:
    try:
        weekday = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Weekly Model Warm Up weekday must be between 0 and 6.") from exc
    if weekday < 0 or weekday > 6:
        raise ValueError("Weekly Model Warm Up weekday must be between 0 and 6.")
    return weekday


def normalize_schedule_month_day(value: Any) -> int:
    try:
        day = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Monthly Model Warm Up day must be between 1 and 31.") from exc
    if day < 1 or day > 31:
        raise ValueError("Monthly Model Warm Up day must be between 1 and 31.")
    return day


def resolve_schedule_timezone(timezone_name: Optional[str]):
    raw = str(timezone_name or "").strip()
    if not raw:
        return timezone.utc
    try:
        return ZoneInfo(raw)
    except Exception:
        return timezone.utc


def validate_schedule_timezone_name(timezone_name: str) -> str:
    normalized = str(timezone_name or "").strip()
    if not normalized:
        return "UTC"
    try:
        ZoneInfo(normalized)
    except Exception as exc:
        raise ValueError(f"Invalid Model Warm Up schedule timezone: {normalized}") from exc
    return normalized


def _last_day_of_month(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]


def _add_month(year: int, month: int) -> tuple[int, int]:
    if month == 12:
        return year + 1, 1
    return year, month + 1


def _monthly_candidate(
    *,
    year: int,
    month: int,
    requested_day: int,
    hour: int,
    minute: int,
    tzinfo,
) -> datetime:
    day = min(requested_day, _last_day_of_month(year, month))
    return datetime.combine(
        datetime(year, month, day).date(),
        time(hour=hour, minute=minute),
        tzinfo=tzinfo,
    )


def compute_next_model_warmup_run_utc(
    settings: dict[str, Any],
    *,
    now_utc: Optional[datetime] = None,
) -> datetime:
    """Compute the next future UTC fire time for a Model Warm Up schedule."""

    now = now_utc or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    now = now.astimezone(timezone.utc)
    tzinfo = resolve_schedule_timezone(settings.get("timezone_name"))
    local_now = now.astimezone(tzinfo)
    cadence = normalize_model_warmup_schedule_cadence(str(settings.get("cadence") or "daily"))

    if cadence == "hourly":
        minute = normalize_schedule_minute(settings.get("minute", 0))
        candidate = local_now.replace(minute=minute, second=0, microsecond=0)
        if candidate <= local_now:
            candidate += timedelta(hours=1)
        return candidate.astimezone(timezone.utc)

    hour, minute = parse_schedule_hhmm(str(settings.get("time_hhmm") or "02:00"))
    if cadence == "daily":
        candidate = datetime.combine(local_now.date(), time(hour=hour, minute=minute), tzinfo=tzinfo)
        if candidate <= local_now:
            candidate += timedelta(days=1)
        return candidate.astimezone(timezone.utc)

    if cadence == "weekly":
        weekday = normalize_schedule_weekday(settings.get("weekday", 0))
        days_ahead = (weekday - local_now.weekday()) % 7
        candidate_date = local_now.date() + timedelta(days=days_ahead)
        candidate = datetime.combine(candidate_date, time(hour=hour, minute=minute), tzinfo=tzinfo)
        if candidate <= local_now:
            candidate += timedelta(days=7)
        return candidate.astimezone(timezone.utc)

    requested_day = normalize_schedule_month_day(settings.get("day_of_month", 1))
    year = local_now.year
    month = local_now.month
    candidate = _monthly_candidate(
        year=year,
        month=month,
        requested_day=requested_day,
        hour=hour,
        minute=minute,
        tzinfo=tzinfo,
    )
    if candidate <= local_now:
        year, month = _add_month(year, month)
        candidate = _monthly_candidate(
            year=year,
            month=month,
            requested_day=requested_day,
            hour=hour,
            minute=minute,
            tzinfo=tzinfo,
        )
    return candidate.astimezone(timezone.utc)


def model_warmup_schedule_label(settings: dict[str, Any]) -> str:
    cadence = str(settings.get("cadence") or "").strip().lower()
    timezone_name = str(settings.get("timezone_name") or "UTC")
    if cadence == "hourly":
        return f"Hourly at minute {int(settings.get('minute', 0)):02d} ({timezone_name})"
    if cadence == "daily":
        return f"Daily at {settings.get('time_hhmm', '02:00')} ({timezone_name})"
    if cadence == "weekly":
        weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        weekday = normalize_schedule_weekday(settings.get("weekday", 0))
        return f"Weekly on {weekdays[weekday]} at {settings.get('time_hhmm', '02:00')} ({timezone_name})"
    if cadence == "monthly":
        return f"Monthly on day {int(settings.get('day_of_month', 1))} at {settings.get('time_hhmm', '02:00')} ({timezone_name})"
    return "Disabled"


class ModelWarmupScheduleStore:
    """Persist the single Model Warm Up schedule and latest scheduler status."""

    def __init__(self, *, history_dir: str):
        self.path = Path(history_dir) / MODEL_WARMUP_SCHEDULE_FILE
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"enabled": False}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"enabled": False}
        return self._with_view(payload) if isinstance(payload, dict) else {"enabled": False}

    def save_schedule(self, settings: dict[str, Any]) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        existing = self.load()
        payload = dict(settings)
        payload["enabled"] = True
        payload["schedule_id"] = str(existing.get("schedule_id") or uuid.uuid4().hex)
        payload["created_at_utc"] = str(existing.get("created_at_utc") or now.isoformat())
        payload["updated_at_utc"] = now.isoformat()
        payload["schedule_label"] = model_warmup_schedule_label(payload)
        payload["next_run_utc"] = compute_next_model_warmup_run_utc(payload, now_utc=now).isoformat()
        payload.pop("canceled_at_utc", None)
        payload["last_status"] = {
            "status": "scheduled",
            "reason": "schedule_saved",
            "schedule_id": payload["schedule_id"],
            "schedule_label": payload["schedule_label"],
            "next_run_utc": payload["next_run_utc"],
            "recorded_at_utc": now.isoformat(),
        }
        return self._write(payload)

    def disable(self) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        payload = self.load()
        payload["enabled"] = False
        payload["updated_at_utc"] = now.isoformat()
        payload["canceled_at_utc"] = now.isoformat()
        payload["next_run_utc"] = None
        if payload.get("schedule_id"):
            payload["last_status"] = {
                "status": "canceled",
                "reason": "user_canceled",
                "schedule_id": payload.get("schedule_id"),
                "schedule_label": payload.get("schedule_label"),
                "canceled_at_utc": now.isoformat(),
                "recorded_at_utc": now.isoformat(),
            }
        return self._write(payload)

    def update_next_run(self, next_run_utc: datetime) -> dict[str, Any]:
        payload = self.load()
        payload["next_run_utc"] = next_run_utc.astimezone(timezone.utc).isoformat()
        return self._write(payload)

    def record_status(self, status: dict[str, Any]) -> dict[str, Any]:
        payload = self.load()
        status_payload = dict(status)
        status_payload["recorded_at_utc"] = datetime.now(timezone.utc).isoformat()
        payload["last_status"] = status_payload
        return self._write(payload)

    def _write(self, payload: dict[str, Any]) -> dict[str, Any]:
        payload_to_write = self._without_view(payload)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.path.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(payload_to_write, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp_path.replace(self.path)
        return self._with_view(payload_to_write)

    def _without_view(self, payload: dict[str, Any]) -> dict[str, Any]:
        clean_payload = dict(payload)
        clean_payload.pop("scheduled_warmups", None)
        return clean_payload

    def _with_view(self, payload: dict[str, Any]) -> dict[str, Any]:
        view_payload = self._without_view(payload)
        schedule_id = view_payload.get("schedule_id")
        if not schedule_id:
            view_payload["scheduled_warmups"] = []
            return view_payload

        last_status = view_payload.get("last_status")
        status = "scheduled" if bool(view_payload.get("enabled")) else "canceled"
        if not bool(view_payload.get("enabled")) and isinstance(last_status, dict):
            status = str(last_status.get("status") or status)

        view_payload["scheduled_warmups"] = [
            {
                "schedule_id": schedule_id,
                "enabled": bool(view_payload.get("enabled")),
                "status": status,
                "cadence": view_payload.get("cadence"),
                "schedule_label": view_payload.get("schedule_label"),
                "timezone_name": view_payload.get("timezone_name"),
                "next_run_utc": view_payload.get("next_run_utc"),
                "canceled_at_utc": view_payload.get("canceled_at_utc"),
                "updated_at_utc": view_payload.get("updated_at_utc"),
                "last_status": last_status,
                "run_request": view_payload.get("run_request") or {},
            }
        ]
        return view_payload


class ModelWarmupScheduler:
    """Daemon scheduler that starts Model Warm Up runs when a schedule is due."""

    def __init__(
        self,
        *,
        settings_getter: Callable[[], dict[str, Any]],
        run_job: Callable[[dict[str, Any], datetime], None],
        next_run_updater: Optional[Callable[[datetime], None]] = None,
        poll_interval_seconds: float = 20.0,
    ):
        self.settings_getter = settings_getter
        self.run_job = run_job
        self.next_run_updater = next_run_updater
        self.poll_interval_seconds = max(1.0, float(poll_interval_seconds))
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_signature: Optional[tuple] = None
        self._next_run_utc: Optional[datetime] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._run_pending_once()
            except Exception:
                # Scheduler safety: never let one exception kill the daemon loop.
                pass
            self._stop_event.wait(self.poll_interval_seconds)

    def _signature(self, settings: dict[str, Any]) -> tuple:
        return (
            bool(settings.get("enabled")),
            str(settings.get("schedule_id") or ""),
            str(settings.get("cadence") or ""),
            str(settings.get("timezone_name") or ""),
            str(settings.get("minute") or ""),
            str(settings.get("time_hhmm") or ""),
            str(settings.get("weekday") or ""),
            str(settings.get("day_of_month") or ""),
            json.dumps(settings.get("run_request") or {}, sort_keys=True),
        )

    def _run_pending_once(self) -> None:
        settings = self.settings_getter() or {}
        if not bool(settings.get("enabled")):
            self._last_signature = None
            self._next_run_utc = None
            return

        signature = self._signature(settings)
        now = datetime.now(timezone.utc)
        if signature != self._last_signature or self._next_run_utc is None:
            self._next_run_utc = compute_next_model_warmup_run_utc(settings, now_utc=now)
            self._last_signature = signature
            if self.next_run_updater is not None:
                self.next_run_updater(self._next_run_utc)

        if self._next_run_utc and now >= self._next_run_utc:
            due_at = self._next_run_utc
            self.run_job(settings, due_at)
            self._next_run_utc = compute_next_model_warmup_run_utc(
                settings,
                now_utc=now + timedelta(seconds=1),
            )
            if self.next_run_updater is not None:
                self.next_run_updater(self._next_run_utc)
