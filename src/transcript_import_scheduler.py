"""Built-in daily scheduler for transcript import automation."""

from __future__ import annotations

import threading
from datetime import datetime, time, timedelta, timezone
from typing import Any, Callable, Optional

from zoneinfo import ZoneInfo


def compute_next_daily_run_utc(
    *,
    time_hhmm: str,
    timezone_name: Optional[str],
    now_utc: Optional[datetime] = None,
) -> datetime:
    """Compute next UTC datetime for a daily local-time schedule."""
    now = now_utc or datetime.now(timezone.utc)
    tz = _resolve_timezone(timezone_name)
    local_now = now.astimezone(tz)

    hour, minute = _parse_hhmm(time_hhmm)
    today_target = datetime.combine(
        local_now.date(),
        time(hour=hour, minute=minute),
        tzinfo=tz,
    )
    if today_target <= local_now:
        today_target = today_target + timedelta(days=1)
    return today_target.astimezone(timezone.utc)


def _parse_hhmm(value: str) -> tuple[int, int]:
    raw = (value or "").strip()
    parts = raw.split(":")
    if len(parts) != 2:
        return 2, 0
    try:
        hour = int(parts[0])
        minute = int(parts[1])
    except ValueError:
        return 2, 0
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        return 2, 0
    return hour, minute


def _resolve_timezone(timezone_name: Optional[str]):
    raw = str(timezone_name or "").strip()
    if not raw:
        return datetime.now().astimezone().tzinfo or timezone.utc
    try:
        return ZoneInfo(raw)
    except Exception:
        return timezone.utc


class TranscriptImportScheduler:
    """Simple daemon scheduler that executes one daily callback."""

    def __init__(
        self,
        *,
        settings_getter: Callable[[], dict[str, Any]],
        run_job: Callable[[dict[str, Any]], None],
        poll_interval_seconds: float = 20.0,
    ):
        self.settings_getter = settings_getter
        self.run_job = run_job
        self.poll_interval_seconds = max(5.0, float(poll_interval_seconds))
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_settings_signature: Optional[tuple] = None
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

    def _run_pending_once(self) -> None:
        settings = self.settings_getter() or {}
        enabled = bool(settings.get("enabled"))
        if not enabled:
            self._last_settings_signature = None
            self._next_run_utc = None
            return

        signature = (
            bool(settings.get("enabled")),
            str(settings.get("time_hhmm") or ""),
            str(settings.get("timezone_name") or ""),
            int(settings.get("max_ids") or 0),
            str(settings.get("filter_json") or ""),
            str(settings.get("language_code") or "en"),
        )
        now_utc = datetime.now(timezone.utc)
        if signature != self._last_settings_signature or self._next_run_utc is None:
            self._next_run_utc = compute_next_daily_run_utc(
                time_hhmm=str(settings.get("time_hhmm") or "02:00"),
                timezone_name=str(settings.get("timezone_name") or ""),
                now_utc=now_utc,
            )
            self._last_settings_signature = signature

        if self._next_run_utc and now_utc >= self._next_run_utc:
            self.run_job(settings)
            self._next_run_utc = compute_next_daily_run_utc(
                time_hhmm=str(settings.get("time_hhmm") or "02:00"),
                timezone_name=str(settings.get("timezone_name") or ""),
                now_utc=now_utc + timedelta(seconds=1),
            )
