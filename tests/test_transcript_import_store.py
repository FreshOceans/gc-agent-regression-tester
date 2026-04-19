"""Unit tests for transcript import artifact store."""

from src.transcript_import_store import TranscriptImportStore


def test_save_run_persists_manifest_transcripts_and_latest_status(tmp_path):
    store = TranscriptImportStore(str(tmp_path / "imports"))
    manifest = {
        "status": "completed",
        "mode": "ids_paste",
        "requested_ids": 2,
        "selected_ids": 2,
        "fetched_ids": 1,
        "failed_ids": 1,
        "skipped_ids": 0,
        "scenarios_generated": 1,
        "failures": [{"conversation_id": "bad", "reason": "not found"}],
    }
    transcripts = {
        "11111111-2222-4333-8444-555555555555": {"id": "11111111-2222-4333-8444-555555555555"}
    }
    stored = store.save_run(
        manifest=manifest,
        transcripts_by_id=transcripts,
        suite_yaml="name: seeded",
    )
    assert stored["run_id"]
    assert store.load_latest_status() is not None
    loaded_manifest = store.load_manifest(stored["run_id"])
    assert loaded_manifest is not None
    assert loaded_manifest["fetched_ids"] == 1
    assert loaded_manifest["mode"] == "ids_paste"

