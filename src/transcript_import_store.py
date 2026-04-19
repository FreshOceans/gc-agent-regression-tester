"""Persistence helpers for transcript import artifacts and status."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


class TranscriptImportStore:
    """Store transcript import artifacts on local disk."""

    def __init__(self, import_dir: str):
        self.import_dir = Path(import_dir)
        self.import_dir.mkdir(parents=True, exist_ok=True)
        self.latest_status_path = self.import_dir / "latest_status.json"

    def _new_run_id(self) -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        suffix = uuid.uuid4().hex[:8]
        return f"{stamp}-{suffix}"

    def save_run(
        self,
        *,
        manifest: dict[str, Any],
        transcripts_by_id: dict[str, dict[str, Any]],
        suite_yaml: Optional[str],
    ) -> dict[str, Any]:
        """Persist one transcript import run with artifacts."""
        run_id = str(manifest.get("run_id") or "").strip() or self._new_run_id()
        run_dir = self.import_dir / run_id
        transcripts_dir = run_dir / "transcripts"
        transcripts_dir.mkdir(parents=True, exist_ok=True)

        stored_manifest = dict(manifest)
        stored_manifest["run_id"] = run_id
        stored_manifest["saved_at_utc"] = datetime.now(timezone.utc).isoformat()
        stored_manifest["run_dir"] = str(run_dir)
        stored_manifest["transcript_count"] = len(transcripts_by_id)

        for conversation_id, payload in transcripts_by_id.items():
            safe_name = self._safe_filename(conversation_id or "unknown")
            out_path = transcripts_dir / f"{safe_name}.json"
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)

        if suite_yaml is not None:
            suite_path = run_dir / "seeded_suite.yaml"
            suite_path.write_text(suite_yaml, encoding="utf-8")
            stored_manifest["seeded_suite_path"] = str(suite_path)

        manifest_path = run_dir / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(stored_manifest, f, indent=2, ensure_ascii=False)
        stored_manifest["manifest_path"] = str(manifest_path)

        latest_status = {
            "run_id": run_id,
            "saved_at_utc": stored_manifest["saved_at_utc"],
            "status": stored_manifest.get("status", "completed"),
            "mode": stored_manifest.get("mode"),
            "requested_ids": int(stored_manifest.get("requested_ids", 0) or 0),
            "selected_ids": int(stored_manifest.get("selected_ids", 0) or 0),
            "fetched_ids": int(stored_manifest.get("fetched_ids", 0) or 0),
            "failed_ids": int(stored_manifest.get("failed_ids", 0) or 0),
            "skipped_ids": int(stored_manifest.get("skipped_ids", 0) or 0),
            "scenarios_generated": int(
                stored_manifest.get("scenarios_generated", 0) or 0
            ),
            "manifest_path": str(manifest_path),
        }
        with self.latest_status_path.open("w", encoding="utf-8") as f:
            json.dump(latest_status, f, indent=2, ensure_ascii=False)
        return stored_manifest

    def load_latest_status(self) -> Optional[dict[str, Any]]:
        """Load latest import status if available."""
        if not self.latest_status_path.exists():
            return None
        try:
            with self.latest_status_path.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
        except Exception:
            return None
        return loaded if isinstance(loaded, dict) else None

    def load_manifest(self, run_id: str) -> Optional[dict[str, Any]]:
        """Load manifest for a specific run id."""
        normalized = self._safe_filename(run_id)
        if not normalized:
            return None
        manifest_path = self.import_dir / normalized / "manifest.json"
        if not manifest_path.exists():
            return None
        try:
            with manifest_path.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
        except Exception:
            return None
        return loaded if isinstance(loaded, dict) else None

    def _safe_filename(self, value: str) -> str:
        return "".join(ch for ch in str(value) if ch.isalnum() or ch in {"_", "-", "."})[:120]
