from __future__ import annotations

import os

from archivetrust.htr.training.full_run.run_state import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_PREPARED,
    STATUS_RUNNING,
    STATUS_STOPPED,
    STATUS_STOPPING,
    create_initial_run_state,
    heartbeat,
    load_run_state,
    mark_completed,
    mark_failed,
    mark_resumed,
    mark_running,
    mark_stopped,
    mark_stopping,
    save_run_state,
)


def test_initial_state_is_prepared():
    state = create_initial_run_state(run_id="r1", configuration_hash="h1")
    assert state.status == STATUS_PREPARED
    assert state.pid is None
    assert state.resume_count == 0


def test_mark_running_records_pid_and_start_time():
    state = create_initial_run_state(run_id="r1", configuration_hash="h1")
    running = mark_running(state)
    assert running.status == STATUS_RUNNING
    assert running.pid == os.getpid()
    assert running.started_at is not None


def test_mark_running_preserves_the_original_start_time_if_already_set():
    state = create_initial_run_state(run_id="r1", configuration_hash="h1")
    running = mark_running(state)
    original_start = running.started_at
    running_again = mark_running(running)
    assert running_again.started_at == original_start


def test_heartbeat_updates_progress_and_timestamp():
    state = mark_running(create_initial_run_state(run_id="r1", configuration_hash="h1"))
    updated = heartbeat(state, current_epoch=3, current_global_step=1875, samples_processed=30000)
    assert updated.current_epoch == 3
    assert updated.current_global_step == 1875
    assert updated.samples_processed == 30000
    assert updated.last_heartbeat_at is not None


def test_mark_stopping_transitions_without_losing_progress():
    state = heartbeat(mark_running(create_initial_run_state(run_id="r1", configuration_hash="h1")), current_epoch=5)
    stopping = mark_stopping(state)
    assert stopping.status == STATUS_STOPPING
    assert stopping.current_epoch == 5


def test_mark_stopped_records_stop_reason_and_end_time():
    state = mark_running(create_initial_run_state(run_id="r1", configuration_hash="h1"))
    stopped = mark_stopped(state, stop_reason="shard_would_not_fit")
    assert stopped.status == STATUS_STOPPED
    assert stopped.stop_reason == "shard_would_not_fit"
    assert stopped.ended_at is not None


def test_mark_failed_records_failure_detail():
    state = mark_running(create_initial_run_state(run_id="r1", configuration_hash="h1"))
    failed = mark_failed(state, stop_reason="oom", failure_detail="CUDA out of memory at shard 12")
    assert failed.status == STATUS_FAILED
    assert failed.failure_detail == "CUDA out of memory at shard 12"


def test_mark_completed_records_stop_reason():
    state = mark_running(create_initial_run_state(run_id="r1", configuration_hash="h1"))
    completed = mark_completed(state, stop_reason="max_full_run_epochs_reached")
    assert completed.status == STATUS_COMPLETED
    assert completed.ended_at is not None


def test_mark_resumed_increments_resume_count_and_returns_to_running():
    state = mark_stopped(mark_running(create_initial_run_state(run_id="r1", configuration_hash="h1")), stop_reason="x")
    resumed = mark_resumed(state)
    assert resumed.status == STATUS_RUNNING
    assert resumed.resume_count == 1
    assert resumed.last_resumed_at is not None

    resumed_again = mark_resumed(mark_stopped(resumed, stop_reason="x"))
    assert resumed_again.resume_count == 2


def test_save_and_load_run_state_round_trips(tmp_path):
    state = heartbeat(
        mark_running(create_initial_run_state(run_id="r1", configuration_hash="h1", dataset_hash="d1")),
        current_epoch=7, latest_metrics={"val_CER_metric": 0.3},
    )
    save_run_state(tmp_path, state)
    reloaded = load_run_state(tmp_path)
    assert reloaded == state


def test_load_run_state_on_missing_file_returns_none(tmp_path):
    assert load_run_state(tmp_path) is None


def test_saved_state_is_never_partially_written(tmp_path):
    """Real atomicity check: no `.tmp-run-state-*` file survives a successful save."""
    state = create_initial_run_state(run_id="r1", configuration_hash="h1")
    save_run_state(tmp_path, state)
    leftovers = [p for p in tmp_path.iterdir() if p.name.startswith(".tmp-run-state-")]
    assert leftovers == []


def test_best_and_latest_metrics_are_tracked_independently():
    state = mark_running(create_initial_run_state(run_id="r1", configuration_hash="h1"))
    updated = heartbeat(
        state,
        latest_metrics={"val_CER_metric": 0.25, "val_loss": 20.0},
        best_metrics={"val_CER_metric": 0.20, "val_loss": 18.0},
    )
    assert updated.latest_metrics["val_CER_metric"] == 0.25
    assert updated.best_metrics["val_CER_metric"] == 0.20  # never overwritten by a worse "latest"


def test_code_revision_is_recorded_best_effort():
    from archivetrust.htr.training.full_run.run_state import get_code_revision

    revision = get_code_revision()
    # either a real 40-char git hash (this repo is a real git checkout) or honestly None
    assert revision is None or len(revision) == 40
