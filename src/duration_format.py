"""Human-friendly duration formatting helpers."""

from __future__ import annotations

import math
from typing import Optional


def format_duration(seconds: Optional[float], seconds_precision: int = 1) -> str:
    """Format seconds using adaptive units.

    Rules:
    - < 120 seconds: keep seconds (with configurable precision)
    - >= 120 seconds and < 3600 seconds: minutes + seconds
    - >= 3600 seconds: hours + minutes + seconds
    """
    value = _coerce_number(seconds)
    if value is None:
        return "n/a"

    value = max(0.0, value)
    precision = max(0, int(seconds_precision))

    if value < 120.0:
        return f"{value:.{precision}f}s"

    rounded_total_seconds = int(round(value))
    hours = rounded_total_seconds // 3600
    minutes = (rounded_total_seconds % 3600) // 60
    remaining_seconds = rounded_total_seconds % 60

    if hours > 0:
        return f"{hours}h {minutes}m {remaining_seconds}s"
    return f"{minutes}m {remaining_seconds}s"


def format_duration_delta(delta_seconds: Optional[float], seconds_precision: int = 1) -> str:
    """Format a duration delta with sign and adaptive units."""
    value = _coerce_number(delta_seconds)
    if value is None:
        return "n/a"
    if value == 0:
        return format_duration(0.0, seconds_precision=seconds_precision)

    sign = "+" if value > 0 else "-"
    return f"{sign}{format_duration(abs(value), seconds_precision=seconds_precision)}"


def _coerce_number(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number
