from __future__ import annotations

from archivetrust.htr.training.checkpoint_index import CheckpointEntry
from archivetrust.htr.training.run_health import compute_health
from archivetrust.htr.training.run_status import RunStatus


def _status(**overrides) -> RunStatus:
    base = dict(
        run_id="r1", run_profile="pilot", status="running", dataset_name="d", dataset_path=None,
        training_line_count=9999, validation_line_count=1000, total_valid_corpus_line_count=563933,
        base_model="generic-2023-02-15@abc", output_dir="/out", device="nvidia-gpu:0", batch_size=16,
        max_epochs=None, early_stopping_patience=5, wall_clock_safety_cap_hours=3.0,
        start_time="2026-08-01T00:00:00Z", last_update_time="2026-08-01T00:00:00Z",
        completion_time=None, stop_reason=None, stopped_mid_epoch=None, stop_boundary=None,
        resumable=True, latest_checkpoint_path="/ckpt/latest", best_checkpoint_path=None,
        cumulative_epoch=3, global_step=0, best_val_cer=0.2, best_epoch=2,
        epochs_since_improvement=0, session_count=1, cumulative_training_seconds=1000.0,
    )
    base.update(overrides)
    return RunStatus(**base)


def _entry(**overrides) -> CheckpointEntry:
    base = dict(
        checkpoint_id="ckpt_1", run_id="r1", session_id="s1", source_checkpoint="parent",
        epoch=3, global_step=0, cumulative_training_seconds=1.0, session_training_seconds=1.0,
        checkpoint_dir="/some/dir", model_file_hash="abc", model_state_present=True,
        optimizer_state_present=True, scheduler_state_present=True, sampler_state_present=False,
        configuration_hash="h1", training_manifest_hash="", validation_manifest_hash="",
        validation_metrics={"val_CER_metric": 0.2}, created_at="2026-08-01T00:00:00Z",
        verification_status="verified", resumable=True, checkpoint_kind="latest",
    )
    base.update(overrides)
    return CheckpointEntry(**base)


def test_healthy_baseline_with_no_signals_of_a_problem(tmp_path):
    best_dir = tmp_path / "best_checkpoint"
    best_dir.mkdir()
    status = _status(best_checkpoint_path=str(best_dir))
    assessment = compute_health(status=status, validation_history=({"epoch": 1}, {"epoch": 2}, {"epoch": 3}))
    assert assessment.level == "healthy"
    assert assessment.findings == ()


def test_failed_status_short_circuits_to_a_single_finding():
    status = _status(status="failed", stop_reason="epoch_failed", stopped_mid_epoch=True)
    assessment = compute_health(status=status)
    assert assessment.level == "failed"
    assert len(assessment.findings) == 1
    assert assessment.findings[0].reason == "run_failed"


def test_interrupted_status_short_circuits_to_stalled():
    status = _status(status="interrupted")
    assessment = compute_health(status=status)
    assert assessment.level == "stalled"
    assert assessment.findings[0].reason == "interrupted"


def test_stale_heartbeat_triggers_a_warning(tmp_path):
    status = _status(status="running")
    assessment = compute_health(status=status, seconds_since_last_heartbeat=45.0)
    assert assessment.level == "warning"
    assert any(f.reason == "stale_heartbeat" for f in assessment.findings)


def test_fresh_heartbeat_does_not_trigger_a_warning():
    status = _status(status="running")
    assessment = compute_health(status=status, seconds_since_last_heartbeat=3.0)
    assert not any(f.reason == "stale_heartbeat" for f in assessment.findings)


def test_missing_validation_records_triggers_a_warning():
    status = _status(cumulative_epoch=3)
    assessment = compute_health(status=status, validation_history=({"epoch": 1}, {"epoch": 2}))  # only 2 of 3
    assert any(f.reason == "validation_missing" for f in assessment.findings)


def test_checkpoint_save_failure_triggers_a_warning():
    status = _status()
    entries = (_entry(epoch=1, verification_status="verified"), _entry(epoch=3, checkpoint_id="ckpt_2", verification_status="verification_failed"))
    assessment = compute_health(status=status, checkpoint_entries=entries)
    assert any(f.reason == "checkpoint_save_failed" for f in assessment.findings)


def test_low_disk_space_triggers_a_warning():
    status = _status()
    samples = ({"gpu": {}, "system": {"disk_free_gb": 2.0}},)
    assessment = compute_health(status=status, recent_gpu_samples=samples)
    assert any(f.reason == "low_disk_space" for f in assessment.findings)


def test_ample_disk_space_does_not_trigger_a_warning():
    status = _status()
    samples = ({"gpu": {}, "system": {"disk_free_gb": 200.0}},)
    assessment = compute_health(status=status, recent_gpu_samples=samples)
    assert not any(f.reason == "low_disk_space" for f in assessment.findings)


def test_gpu_telemetry_disappearing_triggers_a_warning():
    status = _status()
    samples = (
        {"gpu": {"name": "RTX 3070"}, "system": {}},
        {"gpu": {"name": "RTX 3070"}, "system": {}},
        {"gpu": {"name": None}, "system": {}},
    )
    assessment = compute_health(status=status, recent_gpu_samples=samples)
    assert any(f.reason == "gpu_telemetry_disappeared" for f in assessment.findings)


def test_gpu_telemetry_never_present_is_not_flagged_as_disappeared():
    """No GPU ever detected (e.g. this dev sandbox) is a different, honest state from "was present,
    now gone" -- must not be conflated."""
    status = _status()
    samples = ({"gpu": {"name": None}, "system": {}}, {"gpu": {"name": None}, "system": {}})
    assessment = compute_health(status=status, recent_gpu_samples=samples)
    assert not any(f.reason == "gpu_telemetry_disappeared" for f in assessment.findings)


def test_cap_warning_is_surfaced_verbatim_when_provided():
    status = _status()
    assessment = compute_health(status=status, cap_warning="WARNING: cap too short")
    finding = next(f for f in assessment.findings if f.reason == "cap_shorter_than_epoch")
    assert finding.message == "WARNING: cap too short"


def test_no_cap_warning_when_none_provided():
    status = _status()
    assessment = compute_health(status=status, cap_warning=None)
    assert not any(f.reason == "cap_shorter_than_epoch" for f in assessment.findings)


def test_val_cer_not_improving_for_multiple_epochs_triggers_a_warning():
    status = _status(epochs_since_improvement=3)
    assessment = compute_health(status=status)
    assert any(f.reason == "val_cer_not_improving" for f in assessment.findings)


def test_val_cer_still_improving_recently_does_not_trigger_a_warning():
    status = _status(epochs_since_improvement=0)
    assessment = compute_health(status=status)
    assert not any(f.reason == "val_cer_not_improving" for f in assessment.findings)


def test_mid_epoch_stop_triggers_a_warning(tmp_path):
    best_dir = tmp_path / "best_checkpoint"
    best_dir.mkdir()
    status = _status(status="completed", stopped_mid_epoch=True, best_checkpoint_path=str(best_dir))
    assessment = compute_health(status=status)
    assert any(f.reason == "stopped_mid_epoch" for f in assessment.findings)


def test_missing_best_checkpoint_after_progress_triggers_a_warning():
    status = _status(cumulative_epoch=3, best_checkpoint_path=None)
    assessment = compute_health(status=status)
    assert any(f.reason == "best_checkpoint_missing" for f in assessment.findings)


def test_no_best_checkpoint_expected_before_any_epoch_completes():
    status = _status(cumulative_epoch=0, best_checkpoint_path=None)
    assessment = compute_health(status=status)
    assert not any(f.reason == "best_checkpoint_missing" for f in assessment.findings)


def test_best_checkpoint_path_recorded_but_missing_on_disk_triggers_a_warning(tmp_path):
    status = _status(best_checkpoint_path=str(tmp_path / "does_not_exist"))
    assessment = compute_health(status=status)
    assert any(f.reason == "best_checkpoint_missing" for f in assessment.findings)


def test_best_checkpoint_path_that_genuinely_exists_does_not_trigger_a_warning(tmp_path):
    real_dir = tmp_path / "real_checkpoint"
    real_dir.mkdir()
    status = _status(best_checkpoint_path=str(real_dir))
    assessment = compute_health(status=status)
    assert not any(f.reason == "best_checkpoint_missing" for f in assessment.findings)


def test_completed_status_with_no_findings_is_reported_as_completed_not_healthy(tmp_path):
    best_dir = tmp_path / "best_checkpoint"
    best_dir.mkdir()
    status = _status(status="completed", stop_reason="target_epochs_reached", best_checkpoint_path=str(best_dir))
    assessment = compute_health(status=status, validation_history=({"epoch": 1}, {"epoch": 2}, {"epoch": 3}))
    assert assessment.level == "completed"


def test_multiple_simultaneous_findings_are_all_reported_not_just_the_first():
    status = _status(cumulative_epoch=3, best_checkpoint_path=None, epochs_since_improvement=5)
    assessment = compute_health(status=status)
    reasons = {f.reason for f in assessment.findings}
    assert "best_checkpoint_missing" in reasons
    assert "val_cer_not_improving" in reasons
    assert len(assessment.findings) >= 2
