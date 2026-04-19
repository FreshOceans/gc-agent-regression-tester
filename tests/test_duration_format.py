"""Tests for adaptive duration formatting."""

from src.duration_format import format_duration, format_duration_delta


def test_format_duration_below_threshold_uses_seconds():
    assert format_duration(119.4, seconds_precision=1) == "119.4s"


def test_format_duration_at_threshold_uses_minutes_seconds():
    assert format_duration(120.0, seconds_precision=2) == "2m 0s"


def test_format_duration_above_threshold_uses_minutes_seconds():
    assert format_duration(121.6, seconds_precision=3) == "2m 2s"


def test_format_duration_hour_scale_uses_hours_minutes_seconds():
    assert format_duration(3661, seconds_precision=1) == "1h 1m 1s"


def test_format_duration_delta_includes_sign_and_units():
    assert format_duration_delta(130.2, seconds_precision=2) == "+2m 10s"
    assert format_duration_delta(-45.25, seconds_precision=2) == "-45.25s"
    assert format_duration_delta(0.0, seconds_precision=2) == "0.00s"
