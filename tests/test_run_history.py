"""Tests for local run history persistence."""

from datetime import datetime, timezone
from src.models import AttemptResult, Message, MessageRole, ScenarioResult, TestReport
from src.run_history import RunHistoryStore


def _build_report(
    *,
    suite_name: str,
    timestamp: datetime,
    success_rate: float = 1.0,
    failures: int = 0,
    timeouts: int = 0,
    skipped: int = 0,
) -> TestReport:
    attempts = 1
    successes = 1 if success_rate >= 1.0 else 0
    attempt = AttemptResult(
        attempt_number=1,
        success=bool(successes),
        conversation=[Message(role=MessageRole.USER, content="hello")],
        explanation="ok",
        duration_seconds=1.0,
    )
    scenario = ScenarioResult(
        scenario_name="Scenario A",
        attempts=attempts,
        successes=successes,
        failures=failures,
        timeouts=timeouts,
        skipped=skipped,
        success_rate=success_rate,
        is_regression=success_rate < 0.8,
        attempt_results=[attempt],
    )
    return TestReport(
        suite_name=suite_name,
        timestamp=timestamp,
        duration_seconds=1.0,
        scenario_results=[scenario],
        overall_attempts=attempts,
        overall_successes=successes,
        overall_failures=failures,
        overall_timeouts=timeouts,
        overall_skipped=skipped,
        overall_success_rate=success_rate,
        has_regressions=success_rate < 0.8,
        regression_threshold=0.8,
    )


def test_save_and_load_report(tmp_path):
    store = RunHistoryStore(str(tmp_path / "history"), max_runs=50)
    report = _build_report(
        suite_name="Suite A",
        timestamp=datetime(2026, 4, 18, 12, 0, tzinfo=timezone.utc),
    )

    entry = store.save_report(report)
    loaded = store.load_report_from_entry(entry)

    assert loaded is not None
    assert loaded.suite_name == "Suite A"
    assert len(store.list_entries()) == 1


def test_get_previous_same_suite(tmp_path):
    store = RunHistoryStore(str(tmp_path / "history"), max_runs=50)
    first = store.save_report(
        _build_report(
            suite_name="Suite A",
            timestamp=datetime(2026, 4, 18, 12, 0, tzinfo=timezone.utc),
            success_rate=0.5,
        )
    )
    second = store.save_report(
        _build_report(
            suite_name="Suite A",
            timestamp=datetime(2026, 4, 18, 13, 0, tzinfo=timezone.utc),
            success_rate=1.0,
        )
    )

    prev = store.get_previous_same_suite("Suite A", exclude_run_id=second["run_id"])
    assert prev is not None
    assert prev["run_id"] == first["run_id"]


def test_retention_prunes_old_runs(tmp_path):
    store = RunHistoryStore(str(tmp_path / "history"), max_runs=2)
    for hour in range(3):
        store.save_report(
            _build_report(
                suite_name="Suite A",
                timestamp=datetime(2026, 4, 18, 12 + hour, 0, tzinfo=timezone.utc),
            )
        )

    entries = store.list_entries()
    assert len(entries) == 2

    # Verify only files for retained runs remain.
    runs_dir = tmp_path / "history" / "runs"
    run_files = list(runs_dir.glob("*"))
    assert len(run_files) == 2


def test_compaction_transitions_to_full_gzip_and_summary_only(tmp_path):
    store = RunHistoryStore(
        str(tmp_path / "history"),
        max_runs=5,
        full_json_runs=2,
        gzip_runs=2,
    )
    for hour in range(5):
        store.save_report(
            _build_report(
                suite_name="Suite A",
                timestamp=datetime(2026, 4, 18, 12 + hour, 0, tzinfo=timezone.utc),
            )
        )

    entries = store.list_entries()
    assert len(entries) == 5
    assert [entry.get("storage_type") for entry in entries] == [
        "full_json",
        "full_json",
        "gz_json",
        "gz_json",
        "summary_only",
    ]

    newest_two = entries[:2]
    middle_two = entries[2:4]
    oldest = entries[4]

    for entry in newest_two:
        assert isinstance(entry.get("report_file"), str)
        assert str(entry["report_file"]).endswith(".json")
    for entry in middle_two:
        assert isinstance(entry.get("report_file"), str)
        assert str(entry["report_file"]).endswith(".json.gz")
    assert oldest.get("report_file") is None

    runs_dir = tmp_path / "history" / "runs"
    assert len(list(runs_dir.glob("*.json"))) == 2
    assert len(list(runs_dir.glob("*.json.gz"))) == 2


def test_load_report_from_gzip_entry(tmp_path):
    store = RunHistoryStore(
        str(tmp_path / "history"),
        max_runs=5,
        full_json_runs=0,
        gzip_runs=5,
    )
    entry = store.save_report(
        _build_report(
            suite_name="Suite A",
            timestamp=datetime(2026, 4, 18, 12, 0, tzinfo=timezone.utc),
        )
    )
    assert entry.get("storage_type") == "gz_json"
    assert str(entry.get("report_file", "")).endswith(".json.gz")

    loaded = store.load_report_from_entry(entry)
    assert loaded is not None
    assert loaded.suite_name == "Suite A"


def test_summary_only_entry_returns_no_report(tmp_path):
    store = RunHistoryStore(
        str(tmp_path / "history"),
        max_runs=3,
        full_json_runs=0,
        gzip_runs=0,
    )
    entry = store.save_report(
        _build_report(
            suite_name="Suite A",
            timestamp=datetime(2026, 4, 18, 12, 0, tzinfo=timezone.utc),
        )
    )
    assert entry.get("storage_type") == "summary_only"
    assert entry.get("report_file") is None
    assert store.load_report_from_entry(entry) is None
