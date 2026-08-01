from __future__ import annotations

from archivetrust.htr.training.full_run.monitoring_config import FullRunMonitoringConfig
from archivetrust.htr.training.full_run.monitoring_state import (
    COMPLETED,
    FAILED,
    IMPROVING,
    LIKELY_OVERFITTING,
    PLATEAU_CANDIDATE,
    WARM_UP,
    classify_monitoring_state,
)
from archivetrust.htr.training.full_run.run_state import create_initial_run_state, heartbeat, mark_completed, mark_failed, mark_running


def _config(**overrides):
    base = dict(
        pilot_run_id="r1", derived_at="2026-08-01T00:00:00Z", shard_line_count=100, steps_per_shard=10,
        min_exposure_steps=50, min_exposure_basis="test", meaningful_improvement_threshold=0.01,
        recommended_patience=5, patience_basis="test", max_full_run_epochs=1000,
        warning_thresholds={"loss_explosion_multiplier": 10.0, "loss_explosion_absolute_reference": None,
                             "stall_timeout_seconds": None, "check_nan_or_inf": True},
        assumptions_and_method="test",
    )
    base.update(overrides)
    return FullRunMonitoringConfig(**base)


def _state(**overrides):
    state = create_initial_run_state(run_id="r1", configuration_hash="h1")
    state = mark_running(state)
    return heartbeat(state, **overrides)


def test_failed_status_always_wins():
    state = mark_failed(_state(current_epoch=10), stop_reason="epoch_failed", failure_detail="x")
    assert classify_monitoring_state(run_state=state, monitoring_config=_config()) == FAILED


def test_completed_status_always_wins():
    state = mark_completed(_state(current_epoch=10), stop_reason="max_full_run_epochs_reached")
    assert classify_monitoring_state(run_state=state, monitoring_config=_config()) == COMPLETED


def test_warm_up_before_minimum_exposure():
    # current_epoch=2, steps_per_shard=10 -> 20 steps < min_exposure_steps=50
    state = _state(current_epoch=2)
    assert classify_monitoring_state(run_state=state, monitoring_config=_config()) == WARM_UP


def test_improving_when_this_shard_just_set_a_new_best():
    # epochs_since_improvement defaults to 0 -- the most recent shard was the new best
    state = _state(current_epoch=6, latest_metrics={"val_cer": 0.20}, best_metrics={"val_cer": 0.20})
    assert classify_monitoring_state(run_state=state, monitoring_config=_config()) == IMPROVING


def test_plateau_candidate_when_epochs_since_improvement_is_positive():
    state = _state(
        current_epoch=6, latest_metrics={"val_cer": 0.205}, best_metrics={"val_cer": 0.20},
        epochs_since_improvement=2,
    )
    assert classify_monitoring_state(run_state=state, monitoring_config=_config()) == PLATEAU_CANDIDATE


def test_improving_when_a_genuinely_better_result_resets_the_streak():
    state = _state(
        current_epoch=6, latest_metrics={"val_cer": 0.15}, best_metrics={"val_cer": 0.15},
        epochs_since_improvement=0,
    )
    assert classify_monitoring_state(run_state=state, monitoring_config=_config()) == IMPROVING


def test_likely_overfitting_when_val_loss_far_exceeds_train_loss():
    state = _state(
        current_epoch=6,
        latest_metrics={"val_cer": 0.15, "train_loss": 10.0, "val_loss": 20.0},
        best_metrics={"val_cer": 0.15},
    )
    assert classify_monitoring_state(run_state=state, monitoring_config=_config()) == LIKELY_OVERFITTING


def test_not_overfitting_when_loss_gap_is_modest():
    state = _state(
        current_epoch=6,
        latest_metrics={"val_cer": 0.15, "train_loss": 10.0, "val_loss": 11.0},
        best_metrics={"val_cer": 0.15},
    )
    assert classify_monitoring_state(run_state=state, monitoring_config=_config()) != LIKELY_OVERFITTING


def test_never_declares_overfitting_from_training_loss_alone():
    """The brief's rule extended to overfitting detection -- with no val_loss recorded, overfitting
    must never be inferred from train_loss alone."""
    state = _state(
        current_epoch=6,
        latest_metrics={"val_cer": 0.15, "train_loss": 5.0}, best_metrics={"val_cer": 0.15},
    )
    assert classify_monitoring_state(run_state=state, monitoring_config=_config()) != LIKELY_OVERFITTING


def test_no_min_exposure_configured_skips_warm_up_entirely():
    state = _state(current_epoch=0, latest_metrics={"val_cer": 0.5}, best_metrics={"val_cer": 0.5})
    config = _config(min_exposure_steps=None)
    assert classify_monitoring_state(run_state=state, monitoring_config=config) != WARM_UP
