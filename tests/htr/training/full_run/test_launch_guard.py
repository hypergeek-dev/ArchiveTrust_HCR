from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from archivetrust.htr.training.full_run.corpus_sharding import build_full_corpus_shards
from archivetrust.htr.training.full_run.launch_guard import LaunchGuardRejected, enforce_launch_guard, evaluate_launch_guard
from archivetrust.htr.training.full_run.launch_manifest import build_launch_manifest
from archivetrust.htr.training.full_run.run_state import (
    STATUS_RUNNING,
    create_initial_run_state,
    mark_completed,
    mark_failed,
    mark_running,
    mark_stopped,
)


def _make_manifest(path, line_ids):
    table = pa.table({
        "line_id": pa.array(line_ids, type=pa.string()), "collection": pa.array(["c"] * len(line_ids)),
        "source_parquet_file": pa.array(["s"] * len(line_ids)), "row_index": pa.array(range(len(line_ids)), type=pa.int64()),
        "image_content_hash": pa.array(["h"] * len(line_ids)), "image_width": pa.array([10] * len(line_ids), type=pa.int64()),
        "image_height": pa.array([10] * len(line_ids), type=pa.int64()), "transcription_length": pa.array([1] * len(line_ids), type=pa.int64()),
        "valid": pa.array([True] * len(line_ids)),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return path


@pytest.fixture()
def prepared_setup(tmp_path):
    inventory_path = tmp_path / "inventory.parquet"
    _make_manifest(inventory_path, [f"line{i}" for i in range(20)])
    val_manifest_path = _make_manifest(tmp_path / "val.parquet", ["v1", "v2"])

    shards_dir = tmp_path / "run" / "shards"
    summary = build_full_corpus_shards(
        inventory_path=inventory_path, output_dir=shards_dir,
        exclude_manifest_paths=(val_manifest_path,), shard_line_count=5, seed=1, target_shard_count=4,
    )

    run_dir = tmp_path / "run"
    run_state = create_initial_run_state(run_id="r1", configuration_hash="h1", dataset_hash="dh1")

    manifest = build_launch_manifest(
        run_id="r1", original_loghi_base_checkpoint_path=tmp_path / "base", base_checkpoint_hash="bh",
        known_pilot_run_dirs=(), training_manifest_dir=shards_dir, training_manifest_hash=summary.line_id_set_hash,
        validation_manifest_path=val_manifest_path, validation_manifest_hash="vh", dataset_hash="dh1",
        train_line_count=summary.usable_line_count, validation_line_count=2, excluded_test_line_count=0,
        code_commit_hash="c1", container_image_name="img:latest", container_image_digest=None,
        batch_size=16, optimizer="adam", scheduler="constant", learning_rate=0.0001, random_seed=42,
        max_epochs=100, max_epochs_basis="test", early_stopping_patience=5, early_stopping_basis="test",
        wall_clock_policy="test", shard_count=len(summary.shards), checkpoint_policy="test",
        dashboard_config="test", exact_launch_command="python -m x start --confirm-full-corpus-run",
    )

    # `build_launch_manifest` records the real, current repository dirty-state -- pin it to False
    # here so tests about *other* guard conditions get a deterministic clean baseline; the dedicated
    # dirty-repository test below overrides it explicitly.
    manifest = manifest.model_copy(update={"repository_dirty": False})

    return {
        "run_dir": run_dir, "run_state": run_state, "launch_manifest": manifest,
        "current_dataset_hash": "dh1", "current_training_manifest_hash": summary.line_id_set_hash,
        "preflight_passed": True, "preflight_age_seconds": 10.0, "max_preflight_age_seconds": 3600.0,
        "min_free_disk_gb": 0.0, "container_image_tag": "img:latest", "container_image_digest": None,
        "check_docker": False,
    }


def test_passes_with_a_clean_matching_setup(prepared_setup):
    problems = evaluate_launch_guard(confirmed=True, **prepared_setup)
    assert problems == []


def test_missing_confirmation_is_rejected(prepared_setup):
    problems = evaluate_launch_guard(confirmed=False, **prepared_setup)
    assert any("confirmation" in p.lower() for p in problems)


def test_enforce_raises_with_all_problems_listed(prepared_setup):
    prepared_setup["confirmed"] = False
    with pytest.raises(LaunchGuardRejected) as exc_info:
        enforce_launch_guard(**prepared_setup)
    assert "confirmation" in str(exc_info.value).lower()


def test_pilot_checkpoint_as_parent_is_rejected(prepared_setup):
    prepared_setup["launch_manifest"] = prepared_setup["launch_manifest"].model_copy(
        update={"pilot_checkpoint_used_as_parent": True}
    )
    problems = evaluate_launch_guard(confirmed=True, **prepared_setup)
    assert any("pilot checkpoint" in p.lower() for p in problems)


def test_changed_dataset_hash_is_rejected(prepared_setup):
    prepared_setup["current_dataset_hash"] = "different-hash"
    problems = evaluate_launch_guard(confirmed=True, **prepared_setup)
    assert any("dataset hash changed" in p.lower() for p in problems)


def test_changed_training_manifest_hash_is_rejected(prepared_setup):
    prepared_setup["current_training_manifest_hash"] = "different-hash"
    problems = evaluate_launch_guard(confirmed=True, **prepared_setup)
    assert any("training manifest hash changed" in p.lower() for p in problems)


def test_missing_launch_manifest_is_rejected(prepared_setup):
    prepared_setup["launch_manifest"] = None
    problems = evaluate_launch_guard(confirmed=True, **prepared_setup)
    assert any("no launch_manifest.json" in p.lower() for p in problems)


def test_missing_run_state_is_rejected(prepared_setup):
    prepared_setup["run_state"] = None
    problems = evaluate_launch_guard(confirmed=True, **prepared_setup)
    assert any("no run_state.json" in p.lower() for p in problems)


def test_already_running_run_state_is_rejected(prepared_setup):
    prepared_setup["run_state"] = mark_running(prepared_setup["run_state"])
    problems = evaluate_launch_guard(confirmed=True, **prepared_setup)
    assert any("already exists" in p.lower() and "running" in p.lower() for p in problems)


def test_no_preflight_recorded_is_rejected(prepared_setup):
    prepared_setup["preflight_passed"] = False
    problems = evaluate_launch_guard(confirmed=True, **prepared_setup)
    assert any("no passing preflight" in p.lower() for p in problems)


def test_stale_preflight_is_rejected(prepared_setup):
    prepared_setup["preflight_age_seconds"] = 999999.0
    problems = evaluate_launch_guard(confirmed=True, **prepared_setup)
    assert any("older than the allowed" in p.lower() for p in problems)


def test_insufficient_disk_space_is_rejected(prepared_setup):
    prepared_setup["min_free_disk_gb"] = 1e12
    problems = evaluate_launch_guard(confirmed=True, **prepared_setup)
    assert any("below the required" in p.lower() for p in problems)


def test_invalid_output_directory_is_rejected(prepared_setup, tmp_path):
    prepared_setup["run_dir"] = tmp_path / "does-not-exist"
    problems = evaluate_launch_guard(confirmed=True, **prepared_setup)
    assert any("is not valid" in p.lower() for p in problems)


def test_dirty_repository_is_rejected_unless_allowed(prepared_setup):
    prepared_setup["launch_manifest"] = prepared_setup["launch_manifest"].model_copy(update={"repository_dirty": True})
    problems = evaluate_launch_guard(confirmed=True, **prepared_setup)
    assert any("uncommitted changes" in p.lower() for p in problems)

    problems_allowed = evaluate_launch_guard(confirmed=True, allow_dirty_repository=True, **prepared_setup)
    assert not any("uncommitted changes" in p.lower() for p in problems_allowed)


def test_resuming_a_completed_run_is_rejected(prepared_setup):
    """Real audit finding: `resume` previously had no status guard at all (only `start` checked
    for STATUS_PREPARED) -- a completed run could be silently re-entered and its optimizer state
    mutated again. `is_resume=True` must reject STATUS_COMPLETED."""
    prepared_setup["run_state"] = mark_completed(prepared_setup["run_state"], stop_reason="max_full_run_epochs_reached")
    problems = evaluate_launch_guard(confirmed=True, is_resume=True, **prepared_setup)
    assert any("completed" in p.lower() and "terminal" in p.lower() for p in problems)


def test_resuming_a_failed_run_is_rejected(prepared_setup):
    prepared_setup["run_state"] = mark_failed(prepared_setup["run_state"], stop_reason="epoch_failed", failure_detail="x")
    problems = evaluate_launch_guard(confirmed=True, is_resume=True, **prepared_setup)
    assert any("failed" in p.lower() and "terminal" in p.lower() for p in problems)


def test_starting_fresh_is_not_penalized_by_the_terminal_status_check(prepared_setup):
    """The terminal-status rejection only applies to `resume` (`is_resume=True`) -- a fresh `start`
    against a STATUS_PREPARED run must be unaffected."""
    problems = evaluate_launch_guard(confirmed=True, is_resume=False, **prepared_setup)
    assert problems == []


def test_resuming_a_stopped_run_is_allowed(prepared_setup):
    """STOPPED (a graceful interruption, not a terminal completion/failure) must remain resumable."""
    prepared_setup["run_state"] = mark_stopped(prepared_setup["run_state"], stop_reason="stop_requested")
    problems = evaluate_launch_guard(confirmed=True, is_resume=True, **prepared_setup)
    assert problems == []


def test_code_revision_drift_is_rejected(prepared_setup):
    """Real audit finding: a run actually prepared in this repository recorded `code_commit_hash`
    at one commit, and a genuine bug-fix commit landed afterward with nothing to catch the drift --
    the guard now compares the live `git rev-parse HEAD` against what was frozen at `prepare` time."""
    problems = evaluate_launch_guard(confirmed=True, current_code_revision="deadbeef", **prepared_setup)
    assert any("code commit changed" in p.lower() for p in problems)


def test_code_revision_drift_can_be_explicitly_overridden(prepared_setup):
    problems = evaluate_launch_guard(
        confirmed=True, current_code_revision="deadbeef", allow_code_revision_drift=True, **prepared_setup
    )
    assert not any("code commit changed" in p.lower() for p in problems)


def test_matching_code_revision_is_not_flagged(prepared_setup):
    problems = evaluate_launch_guard(confirmed=True, current_code_revision="c1", **prepared_setup)
    assert not any("code commit changed" in p.lower() for p in problems)


def test_train_val_overlap_is_rejected(prepared_setup, tmp_path):
    # Rebuild the launch manifest to point validation at a manifest that DOES overlap a real shard.
    from archivetrust.htr.training.full_run.corpus_sharding import load_sharding_summary

    summary = load_sharding_summary(prepared_setup["run_dir"] / "shards")
    overlapping_line = pq.read_table(summary.shards[0].manifest_path, columns=["line_id"]).column("line_id")[0].as_py()
    overlapping_val = _make_manifest(tmp_path / "overlap_val.parquet", [overlapping_line])
    prepared_setup["launch_manifest"] = prepared_setup["launch_manifest"].model_copy(
        update={"validation_manifest_path": str(overlapping_val)}
    )
    problems = evaluate_launch_guard(confirmed=True, **prepared_setup)
    assert any("overlap" in p.lower() for p in problems)
