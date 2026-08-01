from __future__ import annotations

import zipfile

import pytest

from archivetrust.htr.training.checkpoint_index import (
    CheckpointEntry,
    append_checkpoint_entry,
    best_validation_checkpoint,
    latest_resumable_checkpoint,
    load_index,
    verify_checkpoint,
)


def _entry(**overrides) -> CheckpointEntry:
    base = dict(
        checkpoint_id="ckpt_1", run_id="run_1", session_id="session_1", source_checkpoint="parent",
        epoch=1, global_step=100, cumulative_training_seconds=60.0, session_training_seconds=60.0,
        checkpoint_dir="/some/dir", model_file_hash="abc", model_state_present=True,
        optimizer_state_present=True, scheduler_state_present=True, sampler_state_present=False,
        configuration_hash="cfg1", training_manifest_hash="tm1", validation_manifest_hash="vm1",
        validation_metrics={"val_CER_metric": 0.5}, created_at="2026-08-01T00:00:00Z",
        verification_status="verified", resumable=True, checkpoint_kind="latest",
    )
    base.update(overrides)
    return CheckpointEntry(**base)


def test_load_index_on_missing_file_returns_empty(tmp_path):
    assert load_index(tmp_path / "nope.json") == []


def test_append_and_load_round_trips(tmp_path):
    path = tmp_path / "index.json"
    append_checkpoint_entry(path, _entry())
    entries = load_index(path)
    assert len(entries) == 1
    assert entries[0].checkpoint_id == "ckpt_1"


def test_append_refuses_duplicate_checkpoint_id(tmp_path):
    path = tmp_path / "index.json"
    append_checkpoint_entry(path, _entry())
    with pytest.raises(ValueError):
        append_checkpoint_entry(path, _entry())


def test_index_is_append_only_never_overwrites_earlier_entries(tmp_path):
    path = tmp_path / "index.json"
    append_checkpoint_entry(path, _entry(checkpoint_id="ckpt_1", epoch=1))
    append_checkpoint_entry(path, _entry(checkpoint_id="ckpt_2", epoch=2))
    entries = load_index(path)
    assert len(entries) == 2
    assert {e.checkpoint_id for e in entries} == {"ckpt_1", "ckpt_2"}


def test_latest_resumable_checkpoint_ignores_unresumable_entries(tmp_path):
    path = tmp_path / "index.json"
    append_checkpoint_entry(path, _entry(checkpoint_id="ckpt_1", epoch=1, resumable=True))
    append_checkpoint_entry(path, _entry(checkpoint_id="ckpt_2", epoch=2, resumable=False))
    latest = latest_resumable_checkpoint(path)
    assert latest.checkpoint_id == "ckpt_1"


def test_latest_resumable_checkpoint_picks_highest_epoch(tmp_path):
    path = tmp_path / "index.json"
    append_checkpoint_entry(path, _entry(checkpoint_id="ckpt_1", epoch=1, resumable=True))
    append_checkpoint_entry(path, _entry(checkpoint_id="ckpt_2", epoch=5, resumable=True))
    append_checkpoint_entry(path, _entry(checkpoint_id="ckpt_3", epoch=3, resumable=True))
    latest = latest_resumable_checkpoint(path)
    assert latest.checkpoint_id == "ckpt_2"


def test_never_determines_newest_by_filename_ordering(tmp_path):
    """A checkpoint_dir name that would sort *last* alphabetically must still win if its recorded
    epoch is highest -- proves the index, not a directory listing, is authoritative."""
    path = tmp_path / "index.json"
    append_checkpoint_entry(path, _entry(checkpoint_id="ckpt_1", epoch=9, checkpoint_dir="/a_first_alphabetically", resumable=True))
    append_checkpoint_entry(path, _entry(checkpoint_id="ckpt_2", epoch=1, checkpoint_dir="/z_last_alphabetically", resumable=True))
    latest = latest_resumable_checkpoint(path)
    assert latest.checkpoint_dir == "/a_first_alphabetically"


def test_best_validation_checkpoint_picks_lowest_cer(tmp_path):
    path = tmp_path / "index.json"
    append_checkpoint_entry(path, _entry(checkpoint_id="ckpt_1", validation_metrics={"val_CER_metric": 0.6}, resumable=True))
    append_checkpoint_entry(path, _entry(checkpoint_id="ckpt_2", validation_metrics={"val_CER_metric": 0.3}, resumable=True))
    best = best_validation_checkpoint(path)
    assert best.checkpoint_id == "ckpt_2"


def test_verify_checkpoint_reads_a_real_keras_zip(tmp_path):
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    with zipfile.ZipFile(ckpt_dir / "model.keras", "w") as zf:
        zf.writestr("config.json", "{}")
    ok, digest, extra = verify_checkpoint(ckpt_dir)
    assert ok is True
    assert digest is not None and len(digest) == 64


def test_verify_checkpoint_rejects_missing_keras_file(tmp_path):
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    ok, digest, extra = verify_checkpoint(ckpt_dir)
    assert ok is False
    assert digest is None


def test_verify_checkpoint_rejects_corrupt_zip(tmp_path):
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    (ckpt_dir / "model.keras").write_bytes(b"not a real zip file")
    ok, digest, extra = verify_checkpoint(ckpt_dir)
    assert ok is False


def test_verify_checkpoint_rejects_empty_file(tmp_path):
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    (ckpt_dir / "model.keras").write_bytes(b"")
    ok, digest, extra = verify_checkpoint(ckpt_dir)
    assert ok is False
