"""Report generator for the GC Agent Regression Tester.

Aggregates scenario results into a complete TestReport and provides
CSV and JSON export functionality.
"""

import csv
import io
import json
from datetime import datetime, timezone

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
