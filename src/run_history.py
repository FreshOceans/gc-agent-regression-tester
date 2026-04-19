"""Local run history persistence for dashboard trends and comparisons."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .models import TestReport


class RunHistoryStore:
    """Persist completed reports and retrieve historical baselines."""

    def __init__(self, history_dir: str, max_runs: int = 50):
        self.history_dir = Path(history_dir)
        self.runs_dir = self.history_dir / "runs"
        self.index_path = self.history_dir / "index.json"
        self.max_runs = max(1, int(max_runs))

    def save_report(self, report: TestReport) -> dict:
        """Persist report JSON and return index metadata entry."""
        self._ensure_dirs()
        run_id = self._new_run_id()
        report_rel_path = f"runs/{run_id}.json"
        report_path = self.history_dir / report_rel_path

        self._atomic_write_json(report_path, report.model_dump(mode="json"))

        entry = {
            "run_id": run_id,
            "suite_name": report.suite_name,
            "timestamp": report.timestamp.isoformat(),
            "report_file": report_rel_path,
            "overall_attempts": report.overall_attempts,
            "overall_successes": report.overall_successes,
            "overall_failures": report.overall_failures,
            "overall_timeouts": report.overall_timeouts,
            "overall_skipped": report.overall_skipped,
            "overall_success_rate": report.overall_success_rate,
            "duration_seconds": report.duration_seconds,
            "has_regressions": report.has_regressions,
        }

        index = self._load_index()
        entries = [entry] + index["runs"]
        kept_entries = entries[: self.max_runs]
        pruned_entries = entries[self.max_runs :]

        for pruned in pruned_entries:
            rel_path = pruned.get("report_file")
            if isinstance(rel_path, str) and rel_path.strip():
                with_path = self.history_dir / rel_path
                try:
                    with_path.unlink(missing_ok=True)
                except OSError:
                    pass

        self._atomic_write_json(self.index_path, {"runs": kept_entries})
        return entry

    def list_entries(
        self,
        *,
        suite_name: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[dict]:
        """List history entries (newest first)."""
        entries = self._load_index()["runs"]
        if suite_name:
            target = suite_name.strip().lower()
            entries = [
                e
                for e in entries
                if isinstance(e.get("suite_name"), str)
                and e["suite_name"].strip().lower() == target
            ]
        if limit is not None:
            entries = entries[: max(0, int(limit))]
        return entries

    def get_previous_same_suite(
        self,
        suite_name: str,
        *,
        exclude_run_id: Optional[str] = None,
    ) -> Optional[dict]:
        """Return the newest prior entry for the same suite."""
        for entry in self.list_entries(suite_name=suite_name):
            run_id = entry.get("run_id")
            if exclude_run_id and run_id == exclude_run_id:
                continue
            return entry
        return None

    def load_report_from_entry(self, entry: dict) -> Optional[TestReport]:
        """Load TestReport for an entry, or None if missing/invalid."""
        rel_path = entry.get("report_file")
        if not isinstance(rel_path, str) or not rel_path.strip():
            return None
        report_path = self.history_dir / rel_path
        if not report_path.exists():
            return None

        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        try:
            return TestReport.model_validate(payload)
        except Exception:
            return None

    def _ensure_dirs(self) -> None:
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def _load_index(self) -> dict:
        if not self.index_path.exists():
            return {"runs": []}
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"runs": []}
        runs = payload.get("runs")
        if not isinstance(runs, list):
            return {"runs": []}
        return {"runs": [entry for entry in runs if isinstance(entry, dict)]}

    def _atomic_write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temp_path, path)

    def _new_run_id(self) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        return f"{timestamp}-{uuid.uuid4().hex[:8]}"
