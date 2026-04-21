"""Report generator for the GC Agent Regression Tester.

Aggregates scenario results into a complete TestReport and provides
CSV, JSON, JUnit XML, transcript ZIP, and bundled ZIP export functionality.
"""

import csv
import io
import json
import re
import zipfile
from datetime import datetime, timezone
from typing import Optional
from xml.etree import ElementTree as ET

from .models import ScenarioResult, TestReport, TestSuite


def build_report(
    suite: TestSuite,
    scenario_results: list[ScenarioResult],
    duration: float,
) -> TestReport:
    """Aggregate scenario results into a complete TestReport with overall stats.

    Args:
        suite: The TestSuite that was executed.
        scenario_results: List of ScenarioResult from running each scenario.
        duration: Total execution duration in seconds.

    Returns:
        A TestReport with aggregated statistics.
    """
    overall_attempts = sum(r.attempts for r in scenario_results)
    overall_successes = sum(r.successes for r in scenario_results)
    overall_failures = sum(r.failures for r in scenario_results)
    overall_timeouts = sum(r.timeouts for r in scenario_results)
    overall_skipped = sum(r.skipped for r in scenario_results)
    overall_success_rate = (
        overall_successes / overall_attempts if overall_attempts > 0 else 0.0
    )
    overall_tool_validated_attempts = sum(
        r.tool_validated_attempts for r in scenario_results
    )
    overall_tool_loose_passes = sum(r.tool_loose_passes for r in scenario_results)
    overall_tool_strict_passes = sum(r.tool_strict_passes for r in scenario_results)
    overall_tool_missing_signal_count = sum(
        r.tool_missing_signal_count for r in scenario_results
    )
    overall_tool_order_mismatch_count = sum(
        r.tool_order_mismatch_count for r in scenario_results
    )
    overall_tool_loose_pass_rate = (
        overall_tool_loose_passes / overall_tool_validated_attempts
        if overall_tool_validated_attempts > 0
        else 0.0
    )
    overall_tool_strict_pass_rate = (
        overall_tool_strict_passes / overall_tool_validated_attempts
        if overall_tool_validated_attempts > 0
        else 0.0
    )
    overall_journey_validated_attempts = sum(
        r.journey_validated_attempts for r in scenario_results
    )
    overall_journey_passes = sum(r.journey_passes for r in scenario_results)
    overall_journey_contained_passes = sum(
        r.journey_contained_passes for r in scenario_results
    )
    overall_journey_fulfillment_passes = sum(
        r.journey_fulfillment_passes for r in scenario_results
    )
    overall_journey_path_passes = sum(
        r.journey_path_passes for r in scenario_results
    )
    overall_journey_category_match_passes = sum(
        r.journey_category_match_passes for r in scenario_results
    )
    overall_judging_scored_attempts = sum(
        r.judging_scored_attempts for r in scenario_results
    )
    overall_judging_threshold_passes = sum(
        r.judging_threshold_passes for r in scenario_results
    )
    overall_judging_threshold_failures = sum(
        r.judging_threshold_failures for r in scenario_results
    )
    weighted_score_total = sum(
        r.judging_average_score * r.judging_scored_attempts
        for r in scenario_results
    )
    overall_judging_average_score = (
        weighted_score_total / overall_judging_scored_attempts
        if overall_judging_scored_attempts > 0
        else 0.0
    )
    overall_analytics_evaluated_attempts = sum(
        r.analytics_evaluated_attempts for r in scenario_results
    )
    overall_analytics_gate_passes = sum(
        r.analytics_gate_passes for r in scenario_results
    )
    overall_analytics_skipped_unknown = sum(
        r.analytics_skipped_unknown for r in scenario_results
    )
    has_regressions = any(r.is_regression for r in scenario_results)

    return TestReport(
        suite_name=suite.name,
        timestamp=datetime.now(timezone.utc),
        duration_seconds=duration,
        scenario_results=scenario_results,
        overall_attempts=overall_attempts,
        overall_successes=overall_successes,
        overall_failures=overall_failures,
        overall_timeouts=overall_timeouts,
        overall_skipped=overall_skipped,
        overall_success_rate=overall_success_rate,
        overall_tool_validated_attempts=overall_tool_validated_attempts,
        overall_tool_loose_passes=overall_tool_loose_passes,
        overall_tool_strict_passes=overall_tool_strict_passes,
        overall_tool_missing_signal_count=overall_tool_missing_signal_count,
        overall_tool_order_mismatch_count=overall_tool_order_mismatch_count,
        overall_tool_loose_pass_rate=overall_tool_loose_pass_rate,
        overall_tool_strict_pass_rate=overall_tool_strict_pass_rate,
        overall_journey_validated_attempts=overall_journey_validated_attempts,
        overall_journey_passes=overall_journey_passes,
        overall_journey_contained_passes=overall_journey_contained_passes,
        overall_journey_fulfillment_passes=overall_journey_fulfillment_passes,
        overall_journey_path_passes=overall_journey_path_passes,
        overall_journey_category_match_passes=overall_journey_category_match_passes,
        overall_judging_scored_attempts=overall_judging_scored_attempts,
        overall_judging_threshold_passes=overall_judging_threshold_passes,
        overall_judging_threshold_failures=overall_judging_threshold_failures,
        overall_judging_average_score=overall_judging_average_score,
        overall_analytics_evaluated_attempts=overall_analytics_evaluated_attempts,
        overall_analytics_gate_passes=overall_analytics_gate_passes,
        overall_analytics_skipped_unknown=overall_analytics_skipped_unknown,
        has_regressions=has_regressions,
        regression_threshold=0.8,
    )


def export_csv(report: TestReport) -> str:
    """Export TestReport as CSV string.

    Columns include core outcomes plus tool-validation aggregates:
    scenario_name, attempts, successes, failures, timeouts, skipped, success_rate,
    tool_validated_attempts, tool_loose_passes, tool_loose_pass_rate,
    tool_strict_passes, tool_strict_pass_rate, tool_missing_signal_count,
    tool_order_mismatch_count, is_regression.
    Includes a summary row at the end with overall stats.

    Args:
        report: The TestReport to export.

    Returns:
        A CSV-formatted string.
    """
    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow([
        "scenario_name",
        "attempts",
        "successes",
        "failures",
        "timeouts",
        "skipped",
        "success_rate",
        "tool_validated_attempts",
        "tool_loose_passes",
        "tool_loose_pass_rate",
        "tool_strict_passes",
        "tool_strict_pass_rate",
        "tool_missing_signal_count",
        "tool_order_mismatch_count",
        "journey_validated_attempts",
        "journey_passes",
        "journey_contained_passes",
        "journey_fulfillment_passes",
        "journey_path_passes",
        "journey_category_match_passes",
        "judging_scored_attempts",
        "judging_threshold_passes",
        "judging_threshold_failures",
        "judging_average_score",
        "analytics_evaluated_attempts",
        "analytics_gate_passes",
        "analytics_skipped_unknown",
        "is_regression",
    ])

    # Scenario rows
    for result in report.scenario_results:
        writer.writerow([
            result.scenario_name,
            result.attempts,
            result.successes,
            result.failures,
            result.timeouts,
            result.skipped,
            result.success_rate,
            result.tool_validated_attempts,
            result.tool_loose_passes,
            result.tool_loose_pass_rate,
            result.tool_strict_passes,
            result.tool_strict_pass_rate,
            result.tool_missing_signal_count,
            result.tool_order_mismatch_count,
            result.journey_validated_attempts,
            result.journey_passes,
            result.journey_contained_passes,
            result.journey_fulfillment_passes,
            result.journey_path_passes,
            result.journey_category_match_passes,
            result.judging_scored_attempts,
            result.judging_threshold_passes,
            result.judging_threshold_failures,
            result.judging_average_score,
            result.analytics_evaluated_attempts,
            result.analytics_gate_passes,
            result.analytics_skipped_unknown,
            result.is_regression,
        ])

    # Summary row
    writer.writerow([
        "OVERALL",
        report.overall_attempts,
        report.overall_successes,
        report.overall_failures,
        report.overall_timeouts,
        report.overall_skipped,
        report.overall_success_rate,
        report.overall_tool_validated_attempts,
        report.overall_tool_loose_passes,
        report.overall_tool_loose_pass_rate,
        report.overall_tool_strict_passes,
        report.overall_tool_strict_pass_rate,
        report.overall_tool_missing_signal_count,
        report.overall_tool_order_mismatch_count,
        report.overall_journey_validated_attempts,
        report.overall_journey_passes,
        report.overall_journey_contained_passes,
        report.overall_journey_fulfillment_passes,
        report.overall_journey_path_passes,
        report.overall_journey_category_match_passes,
        report.overall_judging_scored_attempts,
        report.overall_judging_threshold_passes,
        report.overall_judging_threshold_failures,
        report.overall_judging_average_score,
        report.overall_analytics_evaluated_attempts,
        report.overall_analytics_gate_passes,
        report.overall_analytics_skipped_unknown,
        report.has_regressions,
    ])

    # Journey taxonomy rollups
    if report.journey_taxonomy_rollups:
        writer.writerow([])
        writer.writerow(["journey_taxonomy_label", "count", "rate", "delta"])
        for row in report.journey_taxonomy_rollups:
            writer.writerow([
                row.label,
                row.count,
                row.rate,
                row.delta if row.delta is not None else "",
            ])

    return output.getvalue()


def export_json(report: TestReport) -> str:
    """Export TestReport as valid JSON string using Pydantic's model_dump.

    Args:
        report: The TestReport to export.

    Returns:
        A JSON-formatted string.
    """
    return json.dumps(report.model_dump(mode="json"), indent=2)


def _build_attempt_transcript(
    report: TestReport,
    scenario_name: str,
    attempt_number: int,
    success: bool,
    skipped: bool,
    explanation: str,
    error: Optional[str],
    detected_intent: Optional[str],
    started_at: Optional[datetime],
    completed_at: Optional[datetime],
    duration_seconds: Optional[float],
    turn_durations_seconds: list[float],
    step_log: list[dict],
    debug_frames: list[dict],
    timeout_diagnostics: Optional[dict],
    failure_diagnostics: Optional[dict],
    journey_taxonomy_label: Optional[str],
    judging_mechanics_result: Optional[dict],
    tool_events: list[dict],
    tool_validation_result: Optional[dict],
    journey_validation_result: Optional[dict],
    analytics_journey_result: Optional[dict],
    conversation: list,
) -> str:
    """Render one attempt transcript as human-readable text."""
    result_label = "SUCCESS" if success else ("SKIPPED" if skipped else "FAILURE")

    lines = [
        f"Suite: {report.suite_name}",
        f"Scenario: {scenario_name}",
        f"Attempt: {attempt_number}",
        f"Result: {result_label}",
    ]
    if started_at is not None:
        lines.append(f"Started At (UTC): {started_at.isoformat()}")
    if completed_at is not None:
        lines.append(f"Completed At (UTC): {completed_at.isoformat()}")
    if duration_seconds is not None:
        lines.append(f"Attempt Duration (s): {duration_seconds:.3f}")
    if turn_durations_seconds:
        formatted_turns = ", ".join(f"{d:.3f}" for d in turn_durations_seconds)
        lines.append(f"Turn Durations (s): {formatted_turns}")
    if detected_intent:
        lines.append(f"Detected Intent: {detected_intent}")
    if journey_taxonomy_label:
        lines.append(f"Journey Taxonomy Label: {journey_taxonomy_label}")
    if error:
        lines.append(f"Error: {error}")
    if step_log:
        lines.append("Step Log:")
        lines.append(json.dumps(step_log, indent=2))
    if debug_frames:
        lines.append("Debug Frames:")
        lines.append(json.dumps(debug_frames, indent=2))
    if timeout_diagnostics:
        lines.append("Timeout Diagnostics:")
        lines.append(json.dumps(timeout_diagnostics, indent=2))
    if failure_diagnostics:
        lines.append("Failure Diagnostics:")
        lines.append(json.dumps(failure_diagnostics, indent=2))
    if tool_events:
        lines.append("Tool Events:")
        lines.append(json.dumps(tool_events, indent=2))
    if tool_validation_result:
        lines.append("Tool Validation Result:")
        lines.append(json.dumps(tool_validation_result, indent=2))
    if journey_validation_result:
        lines.append("Journey Validation Result:")
        lines.append(json.dumps(journey_validation_result, indent=2))
    if analytics_journey_result:
        lines.append("Analytics Journey Result:")
        lines.append(json.dumps(analytics_journey_result, indent=2))
    if judging_mechanics_result:
        lines.append("Judging Mechanics Result:")
        lines.append(json.dumps(judging_mechanics_result, indent=2))
    lines.extend([
        "",
        "Judge Explanation:",
        explanation,
        "",
        "Conversation:",
    ])

    for msg in conversation:
        role = msg.role.value.upper()
        if getattr(msg, "timestamp", None) is not None:
            lines.append(f"[{msg.timestamp.isoformat()}] {role}: {msg.content}")
        else:
            lines.append(f"{role}: {msg.content}")

    lines.append("")
    return "\n".join(lines)


def _slugify(value: str) -> str:
    """Create a filesystem-safe slug for scenario folder names."""
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "scenario"


def export_junit_xml(report: TestReport) -> str:
    """Export TestReport as a JUnit XML string.

    Each scenario is emitted as a testsuite, and each attempt as a testcase.
    Failed attempts include a failure element with the judge explanation.
    """
    testsuites = ET.Element(
        "testsuites",
        {
            "name": report.suite_name,
            "tests": str(report.overall_attempts),
            "failures": str(report.overall_failures + report.overall_timeouts),
            "time": f"{report.duration_seconds:.3f}",
        },
    )

    for scenario in report.scenario_results:
        scenario_suite = ET.SubElement(
            testsuites,
            "testsuite",
            {
                "name": scenario.scenario_name,
                "tests": str(scenario.attempts),
                "failures": str(scenario.failures + scenario.timeouts),
            },
        )

        for attempt in scenario.attempt_results:
            testcase = ET.SubElement(
                scenario_suite,
                "testcase",
                {
                    "classname": scenario.scenario_name,
                    "name": f"attempt_{attempt.attempt_number}",
                },
            )

            if not attempt.success:
                message = attempt.error or "Goal not achieved"
                if (
                    attempt.tool_validation_result is not None
                    and attempt.tool_validation_result.loose_pass is False
                ):
                    detail = (
                        "; ".join(attempt.tool_validation_result.loose_fail_reasons[:2])
                        or "loose tool rule failed"
                    )
                    message = f"Tool validation failed: {detail}"
                failure = ET.SubElement(
                    testcase,
                    "failure",
                    {"message": message},
                )
                failure.text = attempt.explanation

            system_out = ET.SubElement(testcase, "system-out")
            system_out.text = _build_attempt_transcript(
                report=report,
                scenario_name=scenario.scenario_name,
                attempt_number=attempt.attempt_number,
                success=attempt.success,
                skipped=attempt.skipped,
                explanation=attempt.explanation,
                error=attempt.error,
                detected_intent=attempt.detected_intent,
                started_at=attempt.started_at,
                completed_at=attempt.completed_at,
                duration_seconds=attempt.duration_seconds,
                turn_durations_seconds=attempt.turn_durations_seconds,
                step_log=attempt.step_log,
                debug_frames=attempt.debug_frames,
                timeout_diagnostics=(
                    attempt.timeout_diagnostics.model_dump(mode="json")
                    if attempt.timeout_diagnostics is not None
                    else None
                ),
                failure_diagnostics=(
                    attempt.failure_diagnostics.model_dump(mode="json")
                    if attempt.failure_diagnostics is not None
                    else None
                ),
                journey_taxonomy_label=attempt.journey_taxonomy_label,
                judging_mechanics_result=(
                    attempt.judging_mechanics_result.model_dump(mode="json")
                    if attempt.judging_mechanics_result is not None
                    else None
                ),
                tool_events=[
                    event.model_dump(mode="json")
                    for event in attempt.tool_events
                ],
                tool_validation_result=(
                    attempt.tool_validation_result.model_dump(mode="json")
                    if attempt.tool_validation_result is not None
                    else None
                ),
                conversation=attempt.conversation,
                journey_validation_result=(
                    attempt.journey_validation_result.model_dump(mode="json")
                    if attempt.journey_validation_result is not None
                    else None
                ),
                analytics_journey_result=(
                    attempt.analytics_journey_result.model_dump(mode="json")
                    if attempt.analytics_journey_result is not None
                    else None
                ),
            )

    return ET.tostring(testsuites, encoding="unicode")


def export_transcripts_zip(report: TestReport) -> bytes:
    """Export per-attempt transcripts as a ZIP archive.

    Returns:
        A bytes payload suitable for an application/zip response.
    """
    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for filename, transcript in _iter_attempt_transcript_entries(report):
            zf.writestr(filename, transcript)

    return output.getvalue()


def _iter_attempt_transcript_entries(
    report: TestReport, base_dir: str = ""
) -> list[tuple[str, str]]:
    """Build transcript file paths and content for every scenario attempt."""
    entries: list[tuple[str, str]] = []
    normalized_base_dir = base_dir.strip("/")

    for scenario in report.scenario_results:
        scenario_slug = _slugify(scenario.scenario_name)
        for attempt in scenario.attempt_results:
            filename = f"{scenario_slug}/attempt-{attempt.attempt_number:02d}.txt"
            if normalized_base_dir:
                filename = f"{normalized_base_dir}/{filename}"
            transcript = _build_attempt_transcript(
                report=report,
                scenario_name=scenario.scenario_name,
                attempt_number=attempt.attempt_number,
                success=attempt.success,
                skipped=attempt.skipped,
                explanation=attempt.explanation,
                error=attempt.error,
                detected_intent=attempt.detected_intent,
                started_at=attempt.started_at,
                completed_at=attempt.completed_at,
                duration_seconds=attempt.duration_seconds,
                turn_durations_seconds=attempt.turn_durations_seconds,
                step_log=attempt.step_log,
                debug_frames=attempt.debug_frames,
                timeout_diagnostics=(
                    attempt.timeout_diagnostics.model_dump(mode="json")
                    if attempt.timeout_diagnostics is not None
                    else None
                ),
                failure_diagnostics=(
                    attempt.failure_diagnostics.model_dump(mode="json")
                    if attempt.failure_diagnostics is not None
                    else None
                ),
                journey_taxonomy_label=attempt.journey_taxonomy_label,
                judging_mechanics_result=(
                    attempt.judging_mechanics_result.model_dump(mode="json")
                    if attempt.judging_mechanics_result is not None
                    else None
                ),
                tool_events=[
                    event.model_dump(mode="json")
                    for event in attempt.tool_events
                ],
                tool_validation_result=(
                    attempt.tool_validation_result.model_dump(mode="json")
                    if attempt.tool_validation_result is not None
                    else None
                ),
                journey_validation_result=(
                    attempt.journey_validation_result.model_dump(mode="json")
                    if attempt.journey_validation_result is not None
                    else None
                ),
                analytics_journey_result=(
                    attempt.analytics_journey_result.model_dump(mode="json")
                    if attempt.analytics_journey_result is not None
                    else None
                ),
                conversation=attempt.conversation,
            )
            entries.append((filename, transcript))

    return entries


def export_report_bundle_zip(report: TestReport) -> bytes:
    """Export all report formats and transcripts inside one ZIP archive."""
    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("report.json", export_json(report))
        zf.writestr("report.csv", export_csv(report))
        zf.writestr("report.junit.xml", export_junit_xml(report))
        for filename, transcript in _iter_attempt_transcript_entries(
            report, base_dir="transcripts"
        ):
            zf.writestr(filename, transcript)

    return output.getvalue()
