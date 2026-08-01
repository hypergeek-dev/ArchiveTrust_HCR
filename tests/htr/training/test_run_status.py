from __future__ import annotations

import json
import time

import pytest

from archivetrust.htr.training.run_status import compute_run_status
from archivetrust.htr.training.training_session import TrainingSessionState


def _write_identity(run_dir, **overrides):
    run_state_dir = run_dir / "run-state"
    run_state_dir.mkdir(parents=True, exist_ok=True)
    identity = {
        "method_id": "loghi_swedish_finetuned_v1",
        "parent_method_id": "loghi",
        "parent_checkpoint": "generic-2023-02-15@abc123",
        "training_dataset": "riksarkivet_swedish_lion_libre_training_data",
        "training_phase": "pilot_10k",
        "run_id": "loghi_training_run_test1",
        "created_at": "2026-08-01T00:00:00Z",
    }
    identity.update(overrides)
    (run_state_dir / "training_identity.json").write_text(
        json.dumps({"identity": identity, "configuration_hash": "h1", "configuration": {}}), encoding="utf-8"
    )


def _write_session_state(run_dir, **overrides):
    run_state_dir = run_dir / "run-state"
    run_state_dir.mkdir(parents=True, exist_ok=True)
    base = dict(run_id="loghi_training_run_test1", configuration_hash="h1", random_seed=1)
    base.update(overrides)
    state = TrainingSessionState(**base)
    (run_state_dir / "session_state.json").write_text(state.model_dump_json(indent=2), encoding="utf-8")


def _write_telemetry_status(run_dir, *, last_sample_at, container_alive):
    telemetry_dir = run_dir / "run-state" / "telemetry"
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    (telemetry_dir / "status.json").write_text(
        json.dumps({
            "epoch": 1, "last_sample_at": last_sample_at, "container_alive": container_alive,
            "gpu": {}, "system": {},
        }),
        encoding="utf-8",
    )


def _now_utc_string(offset_seconds: float = 0.0) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + offset_seconds))


def test_raises_when_no_identity_present(tmp_path):
    run_dir = tmp_path / "some-run"
    run_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        compute_run_status(run_dir)


def test_initializing_when_no_session_has_ever_run(tmp_path):
    run_dir = tmp_path / "run1"
    _write_identity(run_dir)
    status = compute_run_status(run_dir)
    assert status.status == "initializing"
    assert status.cumulative_epoch == 0
    assert status.run_id == "loghi_training_run_test1"


def test_running_when_telemetry_heartbeat_is_fresh_and_container_alive(tmp_path):
    run_dir = tmp_path / "run1"
    _write_identity(run_dir)
    _write_session_state(run_dir, cumulative_epoch=2)
    _write_telemetry_status(run_dir, last_sample_at=_now_utc_string(), container_alive=True)
    status = compute_run_status(run_dir)
    assert status.status == "running"


def test_stopping_when_stop_sentinel_present_while_still_alive(tmp_path):
    run_dir = tmp_path / "run1"
    _write_identity(run_dir)
    _write_session_state(run_dir, cumulative_epoch=2)
    _write_telemetry_status(run_dir, last_sample_at=_now_utc_string(), container_alive=True)
    (run_dir / "run-state" / "STOP_REQUESTED").write_text("", encoding="utf-8")
    status = compute_run_status(run_dir)
    assert status.status == "stopping"


def test_completed_when_a_clean_terminal_stop_reason_is_recorded(tmp_path):
    run_dir = tmp_path / "run1"
    _write_identity(run_dir)
    _write_session_state(
        run_dir, cumulative_epoch=3, last_stop_reason="target_epochs_reached",
        last_stopped_mid_epoch=False, last_stop_boundary="after_validation_at_epoch_boundary",
        last_session_ended_at=_now_utc_string(),
    )
    status = compute_run_status(run_dir)
    assert status.status == "completed"
    assert status.stop_reason == "target_epochs_reached"
    assert status.completion_time is not None


def test_failed_when_the_stop_reason_is_epoch_failed(tmp_path):
    run_dir = tmp_path / "run1"
    _write_identity(run_dir)
    _write_session_state(
        run_dir, cumulative_epoch=1, last_stop_reason="epoch_failed",
        last_stopped_mid_epoch=True, last_stop_boundary="epoch_failed_before_completion",
        last_session_ended_at=_now_utc_string(),
    )
    status = compute_run_status(run_dir)
    assert status.status == "failed"
    assert status.stopped_mid_epoch is True


def test_interrupted_when_heartbeat_is_stale_and_no_terminal_reason_was_ever_recorded(tmp_path):
    """The real 'process was killed mid-run' signal: a heartbeat exists, has gone stale, and no
    session ever got the chance to record a clean `last_stop_reason`."""
    run_dir = tmp_path / "run1"
    _write_identity(run_dir)
    _write_session_state(run_dir, cumulative_epoch=2)  # no last_stop_reason ever set
    _write_telemetry_status(run_dir, last_sample_at=_now_utc_string(offset_seconds=-600), container_alive=True)
    status = compute_run_status(run_dir)
    assert status.status == "interrupted"


def test_interrupted_when_heartbeat_is_stale_even_after_a_prior_completion(tmp_path):
    """A heartbeat that has gone stale with no new terminal reason since it started is still
    'interrupted', even if an *earlier* session completed cleanly -- the most recent activity is
    what a live dashboard needs to reflect."""
    run_dir = tmp_path / "run1"
    _write_identity(run_dir)
    _write_session_state(run_dir, cumulative_epoch=2, last_session_ended_at=None)
    _write_telemetry_status(run_dir, last_sample_at=_now_utc_string(offset_seconds=-600), container_alive=True)
    status = compute_run_status(run_dir)
    assert status.status == "interrupted"


def test_not_interrupted_when_no_heartbeat_was_ever_recorded_and_nothing_is_terminal(tmp_path):
    """A session_state.json can exist with cumulative_epoch=0 immediately after
    `run_training_session` writes the configured-budget fields, before the first epoch's sampler
    ever ticks -- this is still 'initializing', not a fabricated 'interrupted'."""
    run_dir = tmp_path / "run1"
    _write_identity(run_dir)
    _write_session_state(run_dir, cumulative_epoch=0, configured_wall_clock_hours=3.0)
    status = compute_run_status(run_dir)
    assert status.status == "initializing"


def test_running_inferred_from_progress_when_no_heartbeat_was_ever_recorded(tmp_path):
    """A real gap this module's own real-data sanity check caught: a session launched before
    `telemetry_sampler.py` was wired in (or run without `run_state_dir` configured) has real
    progress recorded but no `telemetry/status.json` at all -- must not be misclassified as
    'initializing' just because there is no heartbeat to check."""
    run_dir = tmp_path / "run1"
    _write_identity(run_dir)
    _write_session_state(run_dir, cumulative_epoch=5)  # no stop_reason, no telemetry file at all
    status = compute_run_status(run_dir)
    assert status.status == "running"


def test_resumable_reflects_the_real_checkpoint_index(tmp_path):
    from archivetrust.htr.training.checkpoint_index import CheckpointEntry, append_checkpoint_entry

    run_dir = tmp_path / "run1"
    _write_identity(run_dir)
    _write_session_state(run_dir, cumulative_epoch=1)
    index_path = run_dir / "run-state" / "checkpoint_index.json"
    append_checkpoint_entry(
        index_path,
        CheckpointEntry(
            checkpoint_id="ckpt_1", run_id="r1", session_id="s1", source_checkpoint="parent",
            epoch=1, global_step=0, cumulative_training_seconds=1.0, session_training_seconds=1.0,
            checkpoint_dir="/some/dir", model_file_hash="abc", model_state_present=True,
            optimizer_state_present=True, scheduler_state_present=True, sampler_state_present=False,
            configuration_hash="h1", training_manifest_hash="", validation_manifest_hash="",
            validation_metrics={"val_CER_metric": 0.3}, created_at="2026-08-01T00:00:00Z",
            verification_status="verified", resumable=True, checkpoint_kind="latest",
        ),
    )
    status = compute_run_status(run_dir)
    assert status.resumable is True
    assert status.latest_checkpoint_path == "/some/dir"


def test_not_resumable_when_no_verified_checkpoint_exists(tmp_path):
    run_dir = tmp_path / "run1"
    _write_identity(run_dir)
    _write_session_state(run_dir, cumulative_epoch=0)
    status = compute_run_status(run_dir)
    assert status.resumable is False
    assert status.latest_checkpoint_path is None


def test_run_profile_maps_pilot_10k_training_phase_to_pilot(tmp_path):
    run_dir = tmp_path / "run1"
    _write_identity(run_dir, training_phase="pilot_10k")
    status = compute_run_status(run_dir)
    assert status.run_profile == "pilot"


def test_run_profile_is_unknown_for_an_unrecognized_training_phase(tmp_path):
    run_dir = tmp_path / "run1"
    _write_identity(run_dir, training_phase="some_future_phase")
    status = compute_run_status(run_dir)
    assert status.run_profile == "unknown"


def test_configured_budget_fields_flow_through_to_the_safety_cap_panel(tmp_path):
    run_dir = tmp_path / "run1"
    _write_identity(run_dir)
    _write_session_state(
        run_dir, cumulative_epoch=1, configured_wall_clock_hours=2.0,
        configured_max_epochs_this_call=None, configured_early_stopping_patience=5,
    )
    status = compute_run_status(run_dir)
    assert status.wall_clock_safety_cap_hours == 2.0
    assert status.early_stopping_patience == 5


def test_best_epoch_comes_from_the_best_validation_checkpoint_entry(tmp_path):
    from archivetrust.htr.training.checkpoint_index import CheckpointEntry, append_checkpoint_entry

    run_dir = tmp_path / "run1"
    _write_identity(run_dir)
    _write_session_state(run_dir, cumulative_epoch=3, epochs_since_improvement=2)
    index_path = run_dir / "run-state" / "checkpoint_index.json"
    append_checkpoint_entry(
        index_path,
        CheckpointEntry(
            checkpoint_id="ckpt_best", run_id="r1", session_id="s1", source_checkpoint="parent",
            epoch=2, global_step=0, cumulative_training_seconds=1.0, session_training_seconds=1.0,
            checkpoint_dir="/best/dir", model_file_hash="abc", model_state_present=True,
            optimizer_state_present=True, scheduler_state_present=True, sampler_state_present=False,
            configuration_hash="h1", training_manifest_hash="", validation_manifest_hash="",
            validation_metrics={"val_CER_metric": 0.2}, created_at="2026-08-01T00:00:00Z",
            verification_status="verified", resumable=True, checkpoint_kind="best_val",
        ),
    )
    status = compute_run_status(run_dir)
    assert status.best_epoch == 2
    assert status.epochs_since_improvement == 2


def test_start_time_comes_from_the_identitys_real_created_at(tmp_path):
    run_dir = tmp_path / "run1"
    _write_identity(run_dir, created_at="2026-08-01T06:34:21Z")
    status = compute_run_status(run_dir)
    assert status.start_time == "2026-08-01T06:34:21Z"
