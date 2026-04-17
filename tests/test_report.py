"""Unit tests for the Report Generator module."""

import csv
import io
import json
from datetime import datetime, timezone

import pytest

from src.models import (
    AttemptResult,
    Message,
    MessageRole,
    ScenarioResult,
    TestReport,
    TestScenario,
    TestSuite,
)
from src.report import build_report, export_csv, export_json


# --- Fixtures ---


@pytest.fixture
def sample_suite():
    """A simple test suite with two scenarios."""
    return TestSuite(
        name="Sample Suite",
        scenarios=[
            TestScenario(name="Scenario A", persona="User A", goal="Goal A", attempts=3),
            TestScenario(name="Scenario B", persona="User B", goal="Goal B", attempts=2),
        ],
    )


@pytest.fixture
def sample_attempt_results():
    """Helper to create attempt results."""
    def _make(successes: int, total: int) -> list[AttemptResult]:
        results = []
        for i in range(total):
            results.append(
                AttemptResult(
                    attempt_number=i + 1,
                    success=i < successes,
                    conversation=[
                        Message(role=MessageRole.AGENT, content="Hello"),
                        Message(role=MessageRole.USER, content="Hi"),
                    ],
                    explanation="Goal achieved" if i < successes else "Goal not achieved",
                )
            )
        return results
    return _make


@pytest.fixture
def sample_scenario_results(sample_attempt_results):
    """Two scenario results: one passing, one failing."""
    return [
        ScenarioResult(
            scenario_name="Scenario A",
            attempts=3,
            successes=3,
            failures=0,
            success_rate=1.0,
            is_regression=False,
            attempt_results=sample_attempt_results(3, 3),
        ),
        ScenarioResult(
            scenario_name="Scenario B",
            attempts=2,
            successes=1,
            failures=1,
            success_rate=0.5,
            is_regression=True,
            attempt_results=sample_attempt_results(1, 2),
        ),
    ]


# --- build_report tests ---


class TestBuildReport:
    """Tests for the build_report function."""

    def test_aggregates_overall_attempts(self, sample_suite, sample_scenario_results):
        report = build_report(sample_suite, sample_scenario_results, duration=10.5)
        assert report.overall_attempts == 5  # 3 + 2

    def test_aggregates_overall_successes(self, sample_suite, sample_scenario_results):
        report = build_report(sample_suite, sample_scenario_results, duration=10.5)
        assert report.overall_successes == 4  # 3 + 1

    def test_aggregates_overall_failures(self, sample_suite, sample_scenario_results):
        report = build_report(sample_suite, sample_scenario_results, duration=10.5)
        assert report.overall_failures == 1  # 0 + 1

    def test_computes_overall_success_rate(self, sample_suite, sample_scenario_results):
        report = build_report(sample_suite, sample_scenario_results, duration=10.5)
        assert report.overall_success_rate == pytest.approx(4 / 5)

    def test_detects_regressions(self, sample_suite, sample_scenario_results):
        report = build_report(sample_suite, sample_scenario_results, duration=10.5)
        assert report.has_regressions is True

    def test_no_regressions_when_all_pass(self, sample_suite, sample_attempt_results):
        results = [
            ScenarioResult(
                scenario_name="Scenario A",
                attempts=3,
                successes=3,
                failures=0,
                success_rate=1.0,
                is_regression=False,
                attempt_results=sample_attempt_results(3, 3),
            ),
            ScenarioResult(
                scenario_name="Scenario B",
                attempts=2,
                successes=2,
                failures=0,
                success_rate=1.0,
                is_regression=False,
                attempt_results=sample_attempt_results(2, 2),
            ),
        ]
        report = build_report(sample_suite, results, duration=5.0)
        assert report.has_regressions is False

    def test_sets_suite_name(self, sample_suite, sample_scenario_results):
        report = build_report(sample_suite, sample_scenario_results, duration=1.0)
        assert report.suite_name == "Sample Suite"

    def test_sets_duration(self, sample_suite, sample_scenario_results):
        report = build_report(sample_suite, sample_scenario_results, duration=42.5)
        assert report.duration_seconds == 42.5

    def test_sets_timestamp(self, sample_suite, sample_scenario_results):
        report = build_report(sample_suite, sample_scenario_results, duration=1.0)
        assert isinstance(report.timestamp, datetime)

    def test_includes_scenario_results(self, sample_suite, sample_scenario_results):
        report = build_report(sample_suite, sample_scenario_results, duration=1.0)
        assert len(report.scenario_results) == 2
        assert report.scenario_results[0].scenario_name == "Scenario A"
        assert report.scenario_results[1].scenario_name == "Scenario B"

    def test_empty_scenario_results(self, sample_suite):
        report = build_report(sample_suite, [], duration=0.0)
        assert report.overall_attempts == 0
        assert report.overall_successes == 0
        assert report.overall_failures == 0
        assert report.overall_success_rate == 0.0
        assert report.has_regressions is False


# --- export_csv tests ---


class TestExportCsv:
    """Tests for the export_csv function."""

    def test_produces_valid_csv(self, sample_suite, sample_scenario_results):
        report = build_report(sample_suite, sample_scenario_results, duration=10.0)
        csv_str = export_csv(report)
        reader = csv.reader(io.StringIO(csv_str))
        rows = list(reader)
        # Header + 2 scenarios + 1 summary = 4 rows
        assert len(rows) == 4

    def test_header_row(self, sample_suite, sample_scenario_results):
        report = build_report(sample_suite, sample_scenario_results, duration=10.0)
        csv_str = export_csv(report)
        reader = csv.reader(io.StringIO(csv_str))
        header = next(reader)
        assert header == [
            "scenario_name",
            "attempts",
            "successes",
            "failures",
            "success_rate",
            "is_regression",
        ]

    def test_scenario_rows(self, sample_suite, sample_scenario_results):
        report = build_report(sample_suite, sample_scenario_results, duration=10.0)
        csv_str = export_csv(report)
        reader = csv.reader(io.StringIO(csv_str))
        next(reader)  # skip header
        row_a = next(reader)
        assert row_a[0] == "Scenario A"
        assert row_a[1] == "3"
        assert row_a[2] == "3"
        assert row_a[3] == "0"
        assert float(row_a[4]) == pytest.approx(1.0)
        assert row_a[5] == "False"

    def test_summary_row(self, sample_suite, sample_scenario_results):
        report = build_report(sample_suite, sample_scenario_results, duration=10.0)
        csv_str = export_csv(report)
        reader = csv.reader(io.StringIO(csv_str))
        rows = list(reader)
        summary = rows[-1]
        assert summary[0] == "OVERALL"
        assert summary[1] == "5"
        assert summary[2] == "4"
        assert summary[3] == "1"
        assert float(summary[4]) == pytest.approx(0.8)
        assert summary[5] == "True"

    def test_single_scenario(self, sample_attempt_results):
        suite = TestSuite(
            name="Single",
            scenarios=[TestScenario(name="Only", persona="P", goal="G", attempts=1)],
        )
        results = [
            ScenarioResult(
                scenario_name="Only",
                attempts=1,
                successes=1,
                failures=0,
                success_rate=1.0,
                is_regression=False,
                attempt_results=sample_attempt_results(1, 1),
            )
        ]
        report = build_report(suite, results, duration=1.0)
        csv_str = export_csv(report)
        reader = csv.reader(io.StringIO(csv_str))
        rows = list(reader)
        assert len(rows) == 3  # header + 1 scenario + summary


# --- export_json tests ---


class TestExportJson:
    """Tests for the export_json function."""

    def test_produces_valid_json(self, sample_suite, sample_scenario_results):
        report = build_report(sample_suite, sample_scenario_results, duration=10.0)
        json_str = export_json(report)
        data = json.loads(json_str)
        assert isinstance(data, dict)

    def test_contains_suite_name(self, sample_suite, sample_scenario_results):
        report = build_report(sample_suite, sample_scenario_results, duration=10.0)
        json_str = export_json(report)
        data = json.loads(json_str)
        assert data["suite_name"] == "Sample Suite"

    def test_contains_overall_stats(self, sample_suite, sample_scenario_results):
        report = build_report(sample_suite, sample_scenario_results, duration=10.0)
        json_str = export_json(report)
        data = json.loads(json_str)
        assert data["overall_attempts"] == 5
        assert data["overall_successes"] == 4
        assert data["overall_failures"] == 1
        assert data["overall_success_rate"] == pytest.approx(0.8)
        assert data["has_regressions"] is True

    def test_contains_scenario_results(self, sample_suite, sample_scenario_results):
        report = build_report(sample_suite, sample_scenario_results, duration=10.0)
        json_str = export_json(report)
        data = json.loads(json_str)
        assert len(data["scenario_results"]) == 2
        assert data["scenario_results"][0]["scenario_name"] == "Scenario A"
        assert data["scenario_results"][1]["scenario_name"] == "Scenario B"

    def test_contains_timestamp(self, sample_suite, sample_scenario_results):
        report = build_report(sample_suite, sample_scenario_results, duration=10.0)
        json_str = export_json(report)
        data = json.loads(json_str)
        assert "timestamp" in data

    def test_contains_duration(self, sample_suite, sample_scenario_results):
        report = build_report(sample_suite, sample_scenario_results, duration=42.0)
        json_str = export_json(report)
        data = json.loads(json_str)
        assert data["duration_seconds"] == 42.0

    def test_round_trip_via_pydantic(self, sample_suite, sample_scenario_results):
        """JSON export can be loaded back into a TestReport."""
        report = build_report(sample_suite, sample_scenario_results, duration=10.0)
        json_str = export_json(report)
        data = json.loads(json_str)
        restored = TestReport(**data)
        assert restored.suite_name == report.suite_name
        assert restored.overall_attempts == report.overall_attempts
        assert restored.overall_successes == report.overall_successes
        assert restored.overall_failures == report.overall_failures
        assert restored.has_regressions == report.has_regressions
        assert len(restored.scenario_results) == len(report.scenario_results)
