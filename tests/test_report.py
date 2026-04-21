"""Unit tests for the Report Generator module."""

import csv
import io
import json
import zipfile
from datetime import datetime, timezone
from xml.etree import ElementTree as ET

import pytest

from src.models import (
    AttemptResult,
    Message,
    MessageRole,
    ScenarioResult,
    TestReport,
    TestScenario,
    TestSuite,
    TimeoutDiagnostics,
    ToolEvent,
    ToolValidationResult,
)
from src.report import (
    build_report,
    export_csv,
    export_json,
    export_junit_xml,
    export_report_bundle_zip,
    export_transcripts_zip,
)


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

    def test_aggregates_overall_timeouts(self, sample_suite, sample_scenario_results):
        sample_scenario_results[1].timeouts = 1
        report = build_report(sample_suite, sample_scenario_results, duration=10.5)
        assert report.overall_timeouts == 1

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
        assert row_a[4] == "0"
        assert row_a[5] == "0"
        assert float(row_a[6]) == pytest.approx(1.0)
        assert row_a[7] == "0"
        assert row_a[8] == "0"
        assert float(row_a[9]) == pytest.approx(0.0)
        assert row_a[10] == "0"
        assert float(row_a[11]) == pytest.approx(0.0)
        assert row_a[12] == "0"
        assert row_a[13] == "0"
        assert row_a[14] == "0"
        assert row_a[15] == "0"
        assert row_a[16] == "0"
        assert row_a[17] == "0"
        assert row_a[18] == "0"
        assert row_a[19] == "0"
        assert row_a[20] == "0"
        assert row_a[21] == "0"
        assert row_a[22] == "0"
        assert float(row_a[23]) == pytest.approx(0.0)
        assert row_a[24] == "0"
        assert row_a[25] == "0"
        assert row_a[26] == "0"
        assert row_a[27] == "False"

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
        assert summary[4] == "0"
        assert summary[5] == "0"
        assert float(summary[6]) == pytest.approx(0.8)
        assert summary[7] == "0"
        assert summary[8] == "0"
        assert float(summary[9]) == pytest.approx(0.0)
        assert summary[10] == "0"
        assert float(summary[11]) == pytest.approx(0.0)
        assert summary[12] == "0"
        assert summary[13] == "0"
        assert summary[14] == "0"
        assert summary[15] == "0"
        assert summary[16] == "0"
        assert summary[17] == "0"
        assert summary[18] == "0"
        assert summary[19] == "0"
        assert summary[20] == "0"
        assert summary[21] == "0"
        assert summary[22] == "0"
        assert float(summary[23]) == pytest.approx(0.0)
        assert summary[24] == "0"
        assert summary[25] == "0"
        assert summary[26] == "0"
        assert summary[27] == "True"

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

    def test_includes_timeout_diagnostics_when_available(self):
        attempt = AttemptResult(
            attempt_number=1,
            success=False,
            timed_out=True,
            conversation=[Message(role=MessageRole.AGENT, content="Welcome")],
            explanation="Attempt failed due to timeout",
            error="Timed out waiting for agent response after 30s",
            timeout_diagnostics=TimeoutDiagnostics(
                timeout_class="response_timeout",
                step_name="Waiting for agent response",
                step_timeout_seconds=30.0,
            ),
        )
        scenario = ScenarioResult(
            scenario_name="Scenario Timeout",
            attempts=1,
            successes=0,
            failures=1,
            success_rate=0.0,
            is_regression=True,
            attempt_results=[attempt],
        )
        report = TestReport(
            suite_name="Timeout Suite",
            timestamp=datetime.now(timezone.utc),
            duration_seconds=1.0,
            scenario_results=[scenario],
            overall_attempts=1,
            overall_successes=0,
            overall_failures=1,
            overall_timeouts=1,
            overall_success_rate=0.0,
            has_regressions=True,
            regression_threshold=0.8,
        )
        payload = json.loads(export_json(report))
        timeout_payload = payload["scenario_results"][0]["attempt_results"][0]["timeout_diagnostics"]
        assert timeout_payload["timeout_class"] == "response_timeout"
        assert timeout_payload["step_name"] == "Waiting for agent response"


class TestExportJUnitXml:
    """Tests for the export_junit_xml function."""

    def test_produces_valid_xml(self, sample_suite, sample_scenario_results):
        report = build_report(sample_suite, sample_scenario_results, duration=10.0)
        xml_str = export_junit_xml(report)
        root = ET.fromstring(xml_str)

        assert root.tag == "testsuites"
        assert root.attrib["name"] == "Sample Suite"
        assert root.attrib["tests"] == "5"
        assert root.attrib["failures"] == "1"

    def test_includes_scenario_test_suites(self, sample_suite, sample_scenario_results):
        report = build_report(sample_suite, sample_scenario_results, duration=10.0)
        root = ET.fromstring(export_junit_xml(report))
        suites = root.findall("testsuite")

        assert len(suites) == 2
        assert suites[0].attrib["name"] == "Scenario A"
        assert suites[0].attrib["tests"] == "3"
        assert suites[1].attrib["name"] == "Scenario B"
        assert suites[1].attrib["failures"] == "1"

    def test_failed_attempt_contains_failure_node(self, sample_suite, sample_scenario_results):
        report = build_report(sample_suite, sample_scenario_results, duration=10.0)
        root = ET.fromstring(export_junit_xml(report))
        failure_nodes = root.findall(".//failure")

        assert len(failure_nodes) == 1
        assert failure_nodes[0].text == "Goal not achieved"

    def test_attempt_transcript_embedded_in_system_out(self, sample_suite, sample_scenario_results):
        report = build_report(sample_suite, sample_scenario_results, duration=10.0)
        root = ET.fromstring(export_junit_xml(report))
        outputs = root.findall(".//system-out")

        assert len(outputs) == 5
        assert "Scenario: Scenario A" in outputs[0].text
        assert "AGENT: Hello" in outputs[0].text
        assert "USER: Hi" in outputs[0].text

    def test_junit_failure_message_includes_tool_validation_detail(self):
        attempt = AttemptResult(
            attempt_number=1,
            success=False,
            conversation=[Message(role=MessageRole.USER, content="hello")],
            explanation="Goal achieved but tool validation failed",
            tool_events=[ToolEvent(name="flight_lookup", status="success", source="response_marker")],
            tool_validation_result=ToolValidationResult(
                loose_pass=False,
                strict_pass=False,
                missing_signal=False,
                loose_fail_reasons=["Missing tool 'flight_change_priority': required 1, found 0."],
                strict_fail_reasons=["in_order step 2 failed."],
                missing_tools=["flight_change_priority"],
                order_violations=["in_order step 2 failed."],
                matched_tools=["flight_lookup"],
            ),
        )
        scenario = ScenarioResult(
            scenario_name="Scenario A",
            attempts=1,
            successes=0,
            failures=1,
            success_rate=0.0,
            is_regression=True,
            attempt_results=[attempt],
        )
        report = TestReport(
            suite_name="Tool Suite",
            timestamp=datetime.now(timezone.utc),
            duration_seconds=1.0,
            scenario_results=[scenario],
            overall_attempts=1,
            overall_successes=0,
            overall_failures=1,
            overall_success_rate=0.0,
            has_regressions=True,
            regression_threshold=0.8,
        )

        root = ET.fromstring(export_junit_xml(report))
        failure = root.find(".//failure")
        assert failure is not None
        assert "Tool validation failed" in failure.attrib["message"]
        system_out = root.find(".//system-out")
        assert system_out is not None
        assert "Tool Events:" in system_out.text
        assert "Tool Validation Result:" in system_out.text

    def test_junit_system_out_includes_timeout_diagnostics(self):
        attempt = AttemptResult(
            attempt_number=1,
            success=False,
            timed_out=True,
            conversation=[Message(role=MessageRole.AGENT, content="Welcome")],
            explanation="Attempt failed due to timeout",
            error="Timed out waiting for agent response after 30s",
            timeout_diagnostics=TimeoutDiagnostics(
                timeout_class="response_timeout",
                step_name="Waiting for agent response",
                step_timeout_seconds=30.0,
                configured_timeout_seconds=30.0,
                conversation_total_messages=1,
                conversation_agent_messages=1,
                conversation_user_messages=0,
            ),
        )
        scenario = ScenarioResult(
            scenario_name="Scenario Timeout",
            attempts=1,
            successes=0,
            failures=1,
            success_rate=0.0,
            is_regression=True,
            attempt_results=[attempt],
        )
        report = TestReport(
            suite_name="Timeout Suite",
            timestamp=datetime.now(timezone.utc),
            duration_seconds=1.0,
            scenario_results=[scenario],
            overall_attempts=1,
            overall_successes=0,
            overall_failures=1,
            overall_timeouts=1,
            overall_success_rate=0.0,
            has_regressions=True,
            regression_threshold=0.8,
        )
        root = ET.fromstring(export_junit_xml(report))
        system_out = root.find(".//system-out")
        assert system_out is not None
        assert "Timeout Diagnostics:" in system_out.text
        assert '"timeout_class": "response_timeout"' in system_out.text


class TestExportTranscriptsZip:
    """Tests for the export_transcripts_zip function."""

    def test_exports_all_attempt_transcripts(self, sample_suite, sample_scenario_results):
        report = build_report(sample_suite, sample_scenario_results, duration=10.0)
        zip_bytes = export_transcripts_zip(report)

        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
            names = sorted(zf.namelist())

        assert names == [
            "scenario-a/attempt-01.txt",
            "scenario-a/attempt-02.txt",
            "scenario-a/attempt-03.txt",
            "scenario-b/attempt-01.txt",
            "scenario-b/attempt-02.txt",
        ]

    def test_transcript_content_includes_metadata_and_messages(self, sample_suite, sample_scenario_results):
        report = build_report(sample_suite, sample_scenario_results, duration=10.0)
        zip_bytes = export_transcripts_zip(report)

        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
            text = zf.read("scenario-b/attempt-02.txt").decode("utf-8")

        assert "Suite: Sample Suite" in text
        assert "Scenario: Scenario B" in text
        assert "Attempt: 2" in text
        assert "Result: FAILURE" in text
        assert "Judge Explanation:" in text
        assert "Goal not achieved" in text
        assert "AGENT: Hello" in text

    def test_transcript_content_includes_timeout_diagnostics_when_available(self):
        attempt = AttemptResult(
            attempt_number=1,
            success=False,
            timed_out=True,
            conversation=[Message(role=MessageRole.AGENT, content="Welcome")],
            explanation="Attempt failed due to timeout",
            error="Timed out waiting for agent response after 30s",
            timeout_diagnostics=TimeoutDiagnostics(
                timeout_class="response_timeout",
                step_name="Waiting for agent response",
                step_timeout_seconds=30.0,
                configured_timeout_seconds=30.0,
                conversation_total_messages=1,
                conversation_agent_messages=1,
                conversation_user_messages=0,
            ),
        )
        scenario = ScenarioResult(
            scenario_name="Scenario Timeout",
            attempts=1,
            successes=0,
            failures=1,
            success_rate=0.0,
            is_regression=True,
            attempt_results=[attempt],
        )
        report = TestReport(
            suite_name="Timeout Suite",
            timestamp=datetime.now(timezone.utc),
            duration_seconds=1.0,
            scenario_results=[scenario],
            overall_attempts=1,
            overall_successes=0,
            overall_failures=1,
            overall_timeouts=1,
            overall_success_rate=0.0,
            has_regressions=True,
            regression_threshold=0.8,
        )

        zip_bytes = export_transcripts_zip(report)
        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
            text = zf.read("scenario-timeout/attempt-01.txt").decode("utf-8")

        assert "Timeout Diagnostics:" in text
        assert '"timeout_class": "response_timeout"' in text


class TestExportReportBundleZip:
    """Tests for the export_report_bundle_zip function."""

    def test_bundle_includes_all_formats_and_transcripts(self, sample_suite, sample_scenario_results):
        report = build_report(sample_suite, sample_scenario_results, duration=10.0)
        zip_bytes = export_report_bundle_zip(report)

        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
            names = sorted(zf.namelist())

        assert names == [
            "report.csv",
            "report.json",
            "report.junit.xml",
            "transcripts/scenario-a/attempt-01.txt",
            "transcripts/scenario-a/attempt-02.txt",
            "transcripts/scenario-a/attempt-03.txt",
            "transcripts/scenario-b/attempt-01.txt",
            "transcripts/scenario-b/attempt-02.txt",
        ]

    def test_bundle_content_is_parseable(self, sample_suite, sample_scenario_results):
        report = build_report(sample_suite, sample_scenario_results, duration=10.0)
        zip_bytes = export_report_bundle_zip(report)

        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
            json_data = json.loads(zf.read("report.json").decode("utf-8"))
            csv_text = zf.read("report.csv").decode("utf-8")
            xml_root = ET.fromstring(zf.read("report.junit.xml").decode("utf-8"))
            transcript = zf.read("transcripts/scenario-b/attempt-02.txt").decode("utf-8")

        assert json_data["suite_name"] == "Sample Suite"
        csv_rows = list(csv.reader(io.StringIO(csv_text)))
        overall_row = csv_rows[-1]
        assert overall_row[0] == "OVERALL"
        assert overall_row[1] == "5"
        assert overall_row[2] == "4"
        assert overall_row[3] == "1"
        assert overall_row[24] == "0"
        assert overall_row[25] == "0"
        assert overall_row[26] == "0"
        assert overall_row[27] == "True"
        assert xml_root.tag == "testsuites"
        assert transcript.startswith("Suite: Sample Suite")

    def test_bundle_includes_timeout_diagnostics_when_available(self):
        attempt = AttemptResult(
            attempt_number=1,
            success=False,
            timed_out=True,
            conversation=[Message(role=MessageRole.AGENT, content="Welcome")],
            explanation="Attempt failed due to timeout",
            error="Timed out waiting for agent response after 30s",
            timeout_diagnostics=TimeoutDiagnostics(
                timeout_class="response_timeout",
                step_name="Waiting for agent response",
                step_timeout_seconds=30.0,
            ),
        )
        scenario = ScenarioResult(
            scenario_name="Scenario Timeout",
            attempts=1,
            successes=0,
            failures=1,
            success_rate=0.0,
            is_regression=True,
            attempt_results=[attempt],
        )
        report = TestReport(
            suite_name="Timeout Suite",
            timestamp=datetime.now(timezone.utc),
            duration_seconds=1.0,
            scenario_results=[scenario],
            overall_attempts=1,
            overall_successes=0,
            overall_failures=1,
            overall_timeouts=1,
            overall_success_rate=0.0,
            has_regressions=True,
            regression_threshold=0.8,
        )
        zip_bytes = export_report_bundle_zip(report)
        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
            json_payload = json.loads(zf.read("report.json").decode("utf-8"))
            transcript = zf.read("transcripts/scenario-timeout/attempt-01.txt").decode("utf-8")

        timeout_payload = json_payload["scenario_results"][0]["attempt_results"][0]["timeout_diagnostics"]
        assert timeout_payload["timeout_class"] == "response_timeout"
        assert "Timeout Diagnostics:" in transcript
