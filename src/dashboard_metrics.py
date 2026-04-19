"""Dashboard metric aggregation and run comparison helpers."""

from __future__ import annotations

import math
from statistics import mean, median
from typing import Optional

from .models import AttemptResult, TestReport


def build_dashboard_metrics(
    report: TestReport,
    *,
    baseline_report: Optional[TestReport] = None,
    trend_entries: Optional[list[dict]] = None,
    current_run_id: Optional[str] = None,
) -> dict:
    """Build dashboard-ready metrics and compare data for a report."""
    current_summary = _summarize_report(report)
    compare = None
    if baseline_report is not None:
        baseline_summary = _summarize_report(baseline_report)
        compare = _build_compare(current_summary, baseline_summary, baseline_report)

    trend = _build_trend_series(
        trend_entries=trend_entries or [],
        current_run_id=current_run_id,
    )

    return {
        "kpis": current_summary["kpis"],
        "duration": current_summary["duration"],
        "outcome_mix": current_summary["outcome_mix"],
        "scenario_health": current_summary["scenario_health"],
        "top_regressions": current_summary["top_regressions"],
        "compare": compare,
        "trend": trend,
    }


def _summarize_report(report: TestReport) -> dict:
    durations = _all_attempt_durations(report)
    avg_duration = mean(durations) if durations else 0.0
    median_duration = median(durations) if durations else 0.0
    p95_duration = _percentile(durations, 95.0) if durations else 0.0

    attempts = report.overall_attempts or 0
    failure_rate = _safe_rate(report.overall_failures, attempts)
    timeout_rate = _safe_rate(report.overall_timeouts, attempts)
    skipped_rate = _safe_rate(report.overall_skipped, attempts)

    outcome_mix = [
        _outcome_slice("Success", report.overall_successes, attempts),
        _outcome_slice("Failure", report.overall_failures, attempts),
        _outcome_slice("Timeout", report.overall_timeouts, attempts),
        _outcome_slice("Skipped", report.overall_skipped, attempts),
    ]

    scenario_health = sorted(
        [
            {
                "name": scenario.scenario_name,
                "attempts": scenario.attempts,
                "success_rate": scenario.success_rate,
                "failures": scenario.failures,
                "timeouts": scenario.timeouts,
                "skipped": scenario.skipped,
                "is_regression": scenario.is_regression,
            }
            for scenario in report.scenario_results
        ],
        key=lambda row: (
            row["success_rate"],
            row["failures"] + row["timeouts"] + row["skipped"],
            row["name"].lower(),
        ),
    )

    top_regressions = [
        row
        for row in scenario_health
        if row["is_regression"] or (row["failures"] + row["timeouts"] + row["skipped"]) > 0
    ][:5]

    return {
        "kpis": {
            "attempts": attempts,
            "successes": report.overall_successes,
            "failures": report.overall_failures,
            "timeouts": report.overall_timeouts,
            "skipped": report.overall_skipped,
            "success_rate": report.overall_success_rate,
        },
        "duration": {
            "average_seconds": avg_duration,
            "median_seconds": median_duration,
            "p95_seconds": p95_duration,
        },
        "rates": {
            "failure_rate": failure_rate,
            "timeout_rate": timeout_rate,
            "skipped_rate": skipped_rate,
        },
        "outcome_mix": outcome_mix,
        "scenario_health": scenario_health,
        "top_regressions": top_regressions,
    }


def _build_compare(
    current: dict,
    baseline: dict,
    baseline_report: TestReport,
) -> dict:
    return {
        "baseline_suite_name": baseline_report.suite_name,
        "baseline_timestamp": baseline_report.timestamp.isoformat(),
        "deltas": {
            "success_rate": _delta_metric(
                current["kpis"]["success_rate"],
                baseline["kpis"]["success_rate"],
            ),
            "failure_rate": _delta_metric(
                current["rates"]["failure_rate"],
                baseline["rates"]["failure_rate"],
            ),
            "timeout_rate": _delta_metric(
                current["rates"]["timeout_rate"],
                baseline["rates"]["timeout_rate"],
            ),
            "skipped_rate": _delta_metric(
                current["rates"]["skipped_rate"],
                baseline["rates"]["skipped_rate"],
            ),
            "avg_duration_seconds": _delta_metric(
                current["duration"]["average_seconds"],
                baseline["duration"]["average_seconds"],
            ),
            "median_duration_seconds": _delta_metric(
                current["duration"]["median_seconds"],
                baseline["duration"]["median_seconds"],
            ),
            "p95_duration_seconds": _delta_metric(
                current["duration"]["p95_seconds"],
                baseline["duration"]["p95_seconds"],
            ),
        },
    }


def _build_trend_series(trend_entries: list[dict], current_run_id: Optional[str]) -> list[dict]:
    points = []
    seen = set()
    for entry in trend_entries:
        run_id = str(entry.get("run_id", "")).strip()
        timestamp = str(entry.get("timestamp", "")).strip()
        if not run_id or not timestamp:
            continue
        key = (run_id, timestamp)
        if key in seen:
            continue
        seen.add(key)
        attempts = int(entry.get("overall_attempts") or 0)
        points.append(
            {
                "run_id": run_id,
                "timestamp": timestamp,
                "attempts": attempts,
                "success_rate": float(entry.get("overall_success_rate") or 0.0),
                "failures": int(entry.get("overall_failures") or 0),
                "timeouts": int(entry.get("overall_timeouts") or 0),
                "skipped": int(entry.get("overall_skipped") or 0),
                "duration_seconds": float(entry.get("duration_seconds") or 0.0),
                "is_current": bool(current_run_id and run_id == current_run_id),
            }
        )

    points.sort(key=lambda p: p["timestamp"])
    return points[-10:]


def _all_attempt_durations(report: TestReport) -> list[float]:
    durations = []
    for scenario in report.scenario_results:
        for attempt in scenario.attempt_results:
            if _has_duration(attempt):
                durations.append(float(attempt.duration_seconds))
    return durations


def _has_duration(attempt: AttemptResult) -> bool:
    return attempt.duration_seconds is not None and attempt.duration_seconds >= 0


def _outcome_slice(label: str, count: int, total: int) -> dict:
    return {
        "label": label,
        "count": count,
        "percentage": _safe_rate(count, total),
    }


def _safe_rate(value: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return float(value) / float(total)


def _delta_metric(current: float, baseline: float) -> dict:
    delta = current - baseline
    return {
        "current": current,
        "baseline": baseline,
        "delta": delta,
        "direction": "up" if delta > 0 else ("down" if delta < 0 else "flat"),
    }


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]

    sorted_values = sorted(values)
    rank = (percentile / 100.0) * (len(sorted_values) - 1)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return sorted_values[lower]
    weight = rank - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight
