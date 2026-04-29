"""Helpers for compact run-level performance diagnostics."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Optional

from .models import PerformanceDiagnostics, PerformanceStageSummary, TestReport


def _percentile(sorted_values: list[float], rank: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * rank
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * weight


def _summary(stage: str, values_ms: Iterable[float]) -> Optional[PerformanceStageSummary]:
    values = [max(0.0, float(value)) for value in values_ms if value is not None]
    if not values:
        return None
    ordered = sorted(values)
    total = sum(ordered)
    return PerformanceStageSummary(
        stage=stage,
        count=len(ordered),
        total_ms=round(total, 3),
        average_ms=round(total / len(ordered), 3),
        p50_ms=round(_percentile(ordered, 0.50), 3),
        p95_ms=round(_percentile(ordered, 0.95), 3),
        p99_ms=round(_percentile(ordered, 0.99), 3),
        max_ms=round(max(ordered), 3),
    )


def build_performance_diagnostics(
    report: TestReport,
    *,
    run_type: str,
    planned_attempts: int,
    worker_count: Optional[int] = None,
    pacing_seconds: Optional[float] = None,
    queue_wait_ms: Optional[list[float]] = None,
    notes: Optional[list[str]] = None,
) -> PerformanceDiagnostics:
    """Build an additive timing snapshot without changing report semantics."""

    stage_values: dict[str, list[float]] = defaultdict(list)
    judge_values: dict[str, list[float]] = defaultdict(list)

    if queue_wait_ms:
        stage_values["attempt_queue_wait"].extend(queue_wait_ms)

    for scenario in report.scenario_results:
        for attempt in scenario.attempt_results:
            if attempt.duration_seconds is not None:
                stage_values["attempt_total"].append(float(attempt.duration_seconds) * 1000)
            for stage, duration_ms in attempt.warmup_stage_durations_ms.items():
                stage_values[f"web_messaging_{stage}"].append(float(duration_ms))
            for entry in attempt.step_log:
                duration_ms = entry.get("duration_ms")
                if duration_ms is None:
                    continue
                stage = str(entry.get("stage") or entry.get("step") or "step").strip()
                if stage:
                    stage_values[stage].append(float(duration_ms))
            for judge_diag in attempt.judge_diagnostics:
                judge_values[judge_diag.operation_name].append(
                    float(judge_diag.duration_ms)
                )

    stage_summaries = [
        summary
        for stage, values in sorted(stage_values.items())
        if (summary := _summary(stage, values)) is not None
    ]
    judge_summaries = [
        summary
        for operation, values in sorted(judge_values.items())
        if (summary := _summary(operation, values)) is not None
    ]
    slowest = sorted(stage_summaries, key=lambda item: item.p95_ms, reverse=True)[:5]
    attempts_per_second = (
        report.overall_attempts / report.duration_seconds
        if report.duration_seconds > 0
        else 0.0
    )
    timeout_error_count = report.overall_timeouts + report.overall_failures
    timeout_error_rate = (
        timeout_error_count / report.overall_attempts
        if report.overall_attempts > 0
        else 0.0
    )
    adaptive_summary = None
    if report.adaptive_attempt_pacing_enabled:
        adaptive_summary = {
            "enabled": report.adaptive_attempt_pacing_enabled,
            "base_interval_seconds": report.adaptive_attempt_pacing_base_interval_seconds,
            "final_interval_seconds": report.adaptive_attempt_pacing_final_interval_seconds,
            "adjustment_count": report.adaptive_attempt_pacing_adjustment_count,
            "adjustments": [
                adjustment.model_dump(mode="json")
                for adjustment in report.adaptive_attempt_pacing_adjustments
            ],
        }

    return PerformanceDiagnostics(
        run_type=run_type,
        planned_attempts=max(0, int(planned_attempts)),
        completed_attempts=report.overall_attempts,
        duration_seconds=round(float(report.duration_seconds), 3),
        attempts_per_second=round(attempts_per_second, 4),
        worker_count=worker_count,
        pacing_seconds=pacing_seconds,
        timeout_error_rate=round(timeout_error_rate, 4),
        timeout_count=report.overall_timeouts,
        failure_count=report.overall_failures,
        skipped_count=report.overall_skipped,
        stage_summaries=stage_summaries,
        judge_operation_summaries=judge_summaries,
        slowest_stages=slowest,
        adaptive_pacing_summary=adaptive_summary,
        notes=[note for note in (notes or []) if str(note).strip()],
    )
