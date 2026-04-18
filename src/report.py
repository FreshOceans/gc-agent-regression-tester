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
    overall_success_rate = (
        overall_successes / overall_attempts if overall_attempts > 0 else 0.0
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
        overall_success_rate=overall_success_rate,
        has_regressions=has_regressions,
        regression_threshold=0.8,
    )


def export_csv(report: TestReport) -> str:
    """Export TestReport as CSV string.

    Columns: scenario_name, attempts, successes, failures, success_rate, is_regression.
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
        "success_rate",
        "is_regression",
    ])

    # Scenario rows
    for result in report.scenario_results:
        writer.writerow([
            result.scenario_name,
            result.attempts,
            result.successes,
            result.failures,
            result.success_rate,
            result.is_regression,
        ])

    # Summary row
    writer.writerow([
        "OVERALL",
        report.overall_attempts,
        report.overall_successes,
        report.overall_failures,
        report.overall_success_rate,
        report.has_regressions,
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
    explanation: str,
    error: Optional[str],
    started_at: Optional[datetime],
    completed_at: Optional[datetime],
    duration_seconds: Optional[float],
    turn_durations_seconds: list[float],
    conversation: list,
) -> str:
    """Render one attempt transcript as human-readable text."""
    lines = [
        f"Suite: {report.suite_name}",
        f"Scenario: {scenario_name}",
        f"Attempt: {attempt_number}",
        f"Result: {'SUCCESS' if success else 'FAILURE'}",
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
    if error:
        lines.append(f"Error: {error}")
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
            "failures": str(report.overall_failures),
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
                "failures": str(scenario.failures),
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
                explanation=attempt.explanation,
                error=attempt.error,
                started_at=attempt.started_at,
                completed_at=attempt.completed_at,
                duration_seconds=attempt.duration_seconds,
                turn_durations_seconds=attempt.turn_durations_seconds,
                conversation=attempt.conversation,
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
                explanation=attempt.explanation,
                error=attempt.error,
                started_at=attempt.started_at,
                completed_at=attempt.completed_at,
                duration_seconds=attempt.duration_seconds,
                turn_durations_seconds=attempt.turn_durations_seconds,
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
