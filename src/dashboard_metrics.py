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
    baseline_summary: Optional[dict] = None,
    trend_entries: Optional[list[dict]] = None,
    current_run_id: Optional[str] = None,
) -> dict:
    """Build dashboard-ready metrics and compare data for a report."""
    current_summary = _summarize_report(report)
    compare = None

    baseline_compact = None
    if baseline_report is not None:
        baseline_compact = {
            "suite_name": baseline_report.suite_name,
            "timestamp": baseline_report.timestamp.isoformat(),
            "storage_type": "full_json",
            "summary": _summarize_report(baseline_report),
        }
    elif baseline_summary is not None:
        baseline_compact = baseline_summary

    if baseline_compact is not None and isinstance(baseline_compact.get("summary"), dict):
        compare = _build_compare(
            current_summary,
            baseline_compact["summary"],
            baseline_compact,
        )

    history_rows = trend_entries or []
    trend = _build_trend_series(
        trend_entries=history_rows,
        current_run_id=current_run_id,
    )
    flakiness = _build_flakiness(
        trend_entries=history_rows,
        current_suite_name=report.suite_name,
    )

    return {
        "kpis": current_summary["kpis"],
        "tool_effectiveness": current_summary["tool_effectiveness"],
        "journey_effectiveness": current_summary["journey_effectiveness"],
        "duration": current_summary["duration"],
        "outcome_mix": current_summary["outcome_mix"],
        "scenario_health": current_summary["scenario_health"],
        "scenario_tool_health": current_summary["scenario_tool_health"],
        "top_regressions": current_summary["top_regressions"],
        "compare": compare,
        "trend": trend,
        "flakiness": flakiness,
    }


def summarize_entry_for_compare(entry: Optional[dict]) -> Optional[dict]:
    """Convert a run-history entry into compare-ready baseline payload."""
    if not isinstance(entry, dict):
        return None

    attempts = int(entry.get("overall_attempts") or 0)
    failures = int(entry.get("overall_failures") or 0)
    timeouts = int(entry.get("overall_timeouts") or 0)
    skipped = int(entry.get("overall_skipped") or 0)

    summary = {
        "kpis": {
            "attempts": attempts,
            "successes": int(entry.get("overall_successes") or 0),
            "failures": failures,
            "timeouts": timeouts,
            "skipped": skipped,
            "success_rate": float(entry.get("overall_success_rate") or 0.0),
        },
        "tool_effectiveness": {
            "validated_attempts": int(entry.get("overall_tool_validated_attempts") or 0),
            "loose_passes": int(entry.get("overall_tool_loose_passes") or 0),
            "strict_passes": int(entry.get("overall_tool_strict_passes") or 0),
            "missing_signal_count": int(
                entry.get("overall_tool_missing_signal_count") or 0
            ),
            "order_mismatch_count": int(
                entry.get("overall_tool_order_mismatch_count") or 0
            ),
            "loose_pass_rate": float(entry.get("overall_tool_loose_pass_rate") or 0.0),
            "strict_pass_rate": float(entry.get("overall_tool_strict_pass_rate") or 0.0),
        },
        "journey_effectiveness": {
            "validated_attempts": int(entry.get("overall_journey_validated_attempts") or 0),
            "passes": int(entry.get("overall_journey_passes") or 0),
            "contained_passes": int(
                entry.get("overall_journey_contained_passes") or 0
            ),
            "fulfillment_passes": int(
                entry.get("overall_journey_fulfillment_passes") or 0
            ),
            "path_passes": int(entry.get("overall_journey_path_passes") or 0),
            "category_match_passes": int(
                entry.get("overall_journey_category_match_passes") or 0
            ),
        },
        "duration": {
            "average_seconds": float(entry.get("duration_seconds") or 0.0),
            "median_seconds": float(entry.get("duration_seconds") or 0.0),
            "p95_seconds": float(entry.get("duration_seconds") or 0.0),
        },
        "rates": {
            "failure_rate": _safe_rate(failures, attempts),
            "timeout_rate": _safe_rate(timeouts, attempts),
            "skipped_rate": _safe_rate(skipped, attempts),
        },
        "outcome_mix": [],
        "scenario_health": [
            {
                "name": str(s.get("name", "")),
                "attempts": int(s.get("attempts", 0) or 0),
                "success_rate": float(s.get("success_rate", 0.0) or 0.0),
                "failures": int(s.get("failures", 0) or 0),
                "timeouts": int(s.get("timeouts", 0) or 0),
                "skipped": int(s.get("skipped", 0) or 0),
                "is_regression": bool(s.get("is_regression", False)),
                "tool_validated_attempts": int(s.get("tool_validated_attempts", 0) or 0),
                "tool_loose_passes": int(s.get("tool_loose_passes", 0) or 0),
                "tool_strict_passes": int(s.get("tool_strict_passes", 0) or 0),
                "tool_missing_signal_count": int(
                    s.get("tool_missing_signal_count", 0) or 0
                ),
                "tool_order_mismatch_count": int(
                    s.get("tool_order_mismatch_count", 0) or 0
                ),
                "tool_loose_pass_rate": float(s.get("tool_loose_pass_rate", 0.0) or 0.0),
                "tool_strict_pass_rate": float(
                    s.get("tool_strict_pass_rate", 0.0) or 0.0
                ),
                "journey_validated_attempts": int(
                    s.get("journey_validated_attempts", 0) or 0
                ),
                "journey_passes": int(s.get("journey_passes", 0) or 0),
                "journey_contained_passes": int(
                    s.get("journey_contained_passes", 0) or 0
                ),
                "journey_fulfillment_passes": int(
                    s.get("journey_fulfillment_passes", 0) or 0
                ),
                "journey_path_passes": int(s.get("journey_path_passes", 0) or 0),
                "journey_category_match_passes": int(
                    s.get("journey_category_match_passes", 0) or 0
                ),
            }
            for s in (entry.get("scenario_summaries") or [])
            if isinstance(s, dict)
        ],
        "scenario_tool_health": [],
        "top_regressions": [],
    }

    return {
        "suite_name": str(entry.get("suite_name", "")),
        "timestamp": str(entry.get("timestamp", "")),
        "storage_type": str(entry.get("storage_type") or "full_json"),
        "summary": summary,
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
    tool_validated_attempts = int(report.overall_tool_validated_attempts or 0)
    tool_loose_passes = int(report.overall_tool_loose_passes or 0)
    tool_strict_passes = int(report.overall_tool_strict_passes or 0)
    tool_missing_signal_count = int(report.overall_tool_missing_signal_count or 0)
    tool_order_mismatch_count = int(report.overall_tool_order_mismatch_count or 0)
    tool_loose_pass_rate = (
        float(report.overall_tool_loose_pass_rate)
        if tool_validated_attempts > 0
        else 0.0
    )
    tool_strict_pass_rate = (
        float(report.overall_tool_strict_pass_rate)
        if tool_validated_attempts > 0
        else 0.0
    )
    journey_validated_attempts = int(report.overall_journey_validated_attempts or 0)
    journey_passes = int(report.overall_journey_passes or 0)
    journey_contained_passes = int(report.overall_journey_contained_passes or 0)
    journey_fulfillment_passes = int(report.overall_journey_fulfillment_passes or 0)
    journey_path_passes = int(report.overall_journey_path_passes or 0)
    journey_category_match_passes = int(
        report.overall_journey_category_match_passes or 0
    )

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
                "tool_validated_attempts": scenario.tool_validated_attempts,
                "tool_loose_passes": scenario.tool_loose_passes,
                "tool_strict_passes": scenario.tool_strict_passes,
                "tool_missing_signal_count": scenario.tool_missing_signal_count,
                "tool_order_mismatch_count": scenario.tool_order_mismatch_count,
                "tool_loose_pass_rate": scenario.tool_loose_pass_rate,
                "tool_strict_pass_rate": scenario.tool_strict_pass_rate,
                "journey_validated_attempts": scenario.journey_validated_attempts,
                "journey_passes": scenario.journey_passes,
                "journey_contained_passes": scenario.journey_contained_passes,
                "journey_fulfillment_passes": scenario.journey_fulfillment_passes,
                "journey_path_passes": scenario.journey_path_passes,
                "journey_category_match_passes": scenario.journey_category_match_passes,
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
    scenario_tool_health = sorted(
        [
            row
            for row in scenario_health
            if int(row.get("tool_validated_attempts", 0) or 0) > 0
        ],
        key=lambda row: (
            row["tool_loose_pass_rate"],
            row["tool_missing_signal_count"],
            row["tool_order_mismatch_count"],
            row["name"].lower(),
        ),
    )

    return {
        "kpis": {
            "attempts": attempts,
            "successes": report.overall_successes,
            "failures": report.overall_failures,
            "timeouts": report.overall_timeouts,
            "skipped": report.overall_skipped,
            "success_rate": report.overall_success_rate,
        },
        "tool_effectiveness": {
            "validated_attempts": tool_validated_attempts,
            "loose_passes": tool_loose_passes,
            "strict_passes": tool_strict_passes,
            "missing_signal_count": tool_missing_signal_count,
            "order_mismatch_count": tool_order_mismatch_count,
            "loose_pass_rate": tool_loose_pass_rate,
            "strict_pass_rate": tool_strict_pass_rate,
        },
        "journey_effectiveness": {
            "validated_attempts": journey_validated_attempts,
            "passes": journey_passes,
            "contained_passes": journey_contained_passes,
            "fulfillment_passes": journey_fulfillment_passes,
            "path_passes": journey_path_passes,
            "category_match_passes": journey_category_match_passes,
            "pass_rate": _safe_rate(journey_passes, journey_validated_attempts),
            "contained_rate": _safe_rate(
                journey_contained_passes,
                journey_validated_attempts,
            ),
            "fulfillment_rate": _safe_rate(
                journey_fulfillment_passes,
                journey_validated_attempts,
            ),
            "path_rate": _safe_rate(
                journey_path_passes,
                journey_validated_attempts,
            ),
            "category_match_rate": _safe_rate(
                journey_category_match_passes,
                journey_validated_attempts,
            ),
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
        "scenario_tool_health": scenario_tool_health,
        "top_regressions": top_regressions,
    }


def _build_compare(
    current: dict,
    baseline: dict,
    baseline_meta: dict,
) -> dict:
    return {
        "baseline_suite_name": baseline_meta.get("suite_name", ""),
        "baseline_timestamp": baseline_meta.get("timestamp", ""),
        "baseline_storage_type": baseline_meta.get("storage_type", "full_json"),
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
            "tool_loose_pass_rate": _delta_metric(
                current.get("tool_effectiveness", {}).get("loose_pass_rate", 0.0),
                baseline.get("tool_effectiveness", {}).get("loose_pass_rate", 0.0),
            ),
            "tool_strict_pass_rate": _delta_metric(
                current.get("tool_effectiveness", {}).get("strict_pass_rate", 0.0),
                baseline.get("tool_effectiveness", {}).get("strict_pass_rate", 0.0),
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


def _build_flakiness(trend_entries: list[dict], *, current_suite_name: str) -> dict:
    """Build stability metrics from recent same-suite run history."""
    rows = [
        entry
        for entry in trend_entries
        if str(entry.get("suite_name", "")).strip().lower() == current_suite_name.strip().lower()
    ]
    if not rows:
        return {
            "evaluated_runs": 0,
            "scenarios_evaluated": 0,
            "unstable_scenarios": [],
        }

    rows.sort(key=lambda item: str(item.get("timestamp", "")))
    scenario_series: dict[str, list[dict]] = {}

    for row in rows:
        scenario_summaries = row.get("scenario_summaries")
        if not isinstance(scenario_summaries, list):
            continue
        for scenario in scenario_summaries:
            if not isinstance(scenario, dict):
                continue
            name = str(scenario.get("name", "")).strip()
            if not name:
                continue
            success_rate = float(scenario.get("success_rate", 0.0) or 0.0)
            scenario_series.setdefault(name, []).append(
                {
                    "success_rate": success_rate,
                    "is_pass": success_rate >= 0.8,
                }
            )

    unstable = []
    for name, series in scenario_series.items():
        if len(series) < 2:
            continue

        flips = 0
        failures = 0
        success_values = [float(point["success_rate"]) for point in series]

        for index, point in enumerate(series):
            if not point["is_pass"]:
                failures += 1
            if index > 0 and point["is_pass"] != series[index - 1]["is_pass"]:
                flips += 1

        transitions = max(1, len(series) - 1)
        flip_rate = flips / transitions
        failure_rate = failures / len(series)
        volatility = max(success_values) - min(success_values)
        instability_score = (flip_rate * 0.5) + (failure_rate * 0.3) + (volatility * 0.2)

        reason = "High success-rate volatility"
        if flip_rate >= 0.5:
            reason = "Frequent pass/fail flips"
        elif failure_rate >= 0.5:
            reason = "Fails often across runs"

        unstable.append(
            {
                "name": name,
                "flip_rate": flip_rate,
                "failure_occurrence_rate": failure_rate,
                "volatility": volatility,
                "instability_score": instability_score,
                "reason": reason,
                "runs": len(series),
            }
        )

    unstable.sort(
        key=lambda row: (
            row["instability_score"],
            row["failure_occurrence_rate"],
            row["flip_rate"],
            row["name"].lower(),
        ),
        reverse=True,
    )

    return {
        "evaluated_runs": len(rows),
        "scenarios_evaluated": len(scenario_series),
        "unstable_scenarios": unstable[:8],
    }


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
