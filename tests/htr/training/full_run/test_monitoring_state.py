from __future__ import annotations

from archivetrust.htr.training.full_run.monitoring_config import FullRunMonitoringConfig
from archivetrust.htr.training.full_run.monitoring_state import (
    COMPLETED,
    DEGRADING,
    FAILED,
    HEALTHY,
    IMPROVING,
    INITIALIZING,
    INTERRUPTED,
    PLATEAU_CANDIDATE,
    PLATEAU_CONFIRMED,
    SLOWING,
    STALLED,
    classify_monitoring_state,
)
from archivetrust.htr.training.full_run.run_state import (
    create_initial_run_state,
    heartbeat,
    mark_completed,
    mark_failed,
    mark_stopped,
    mark_running,
)


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


def test_interrupted_when_running_but_heartbeat_stale():
    state = _state(current_epoch=10, last_heartbeat_at="2020-01-01T00:00:00Z")
    assert classify_monitoring_state(run_state=state, monitoring_config=_config(), now=1_700_000_000.0) == INTERRUPTED


def test_stalled_when_stop_reason_is_stalled_data_loading():
    state = mark_stopped(_state(current_epoch=10), stop_reason="stalled_data_loading")
    # STOPPED is a terminal status but stop_reason still drives STALLED classification when checked
    # before a heartbeat-staleness short-circuit would otherwise apply.
    assert classify_monitoring_state(run_state=state, monitoring_config=_config(), now=state_now(state)) == STALLED


def state_now(state):
    import calendar
    import time

    if state.last_heartbeat_at is None:
        return time.time()
    return calendar.timegm(time.strptime(state.last_heartbeat_at, "%Y-%m-%dT%H:%M:%SZ"))


def test_initializing_before_minimum_exposure():
    # current_epoch=2, steps_per_shard=10 -> 20 steps < min_exposure_steps=50
    state = _state(current_epoch=2)
    assert classify_monitoring_state(run_state=state, monitoring_config=_config(), now=state_now(state)) == INITIALIZING


def test_no_min_exposure_configured_skips_initializing_once_a_shard_completed():
    state = _state(current_epoch=1, latest_metrics={"val_cer": 0.5}, best_metrics={"val_cer": 0.5})
    config = _config(min_exposure_steps=None)
    assert classify_monitoring_state(run_state=state, monitoring_config=config, now=state_now(state)) != INITIALIZING


def test_degrading_when_val_loss_far_exceeds_train_loss():
    state = _state(
        current_epoch=6,
        latest_metrics={"val_cer": 0.15, "train_loss": 10.0, "val_loss": 20.0},
        best_metrics={"val_cer": 0.15},
    )
    assert classify_monitoring_state(run_state=state, monitoring_config=_config(), now=state_now(state)) == DEGRADING


def test_not_degrading_when_loss_gap_is_modest_and_no_worsening_streak():
    state = _state(
        current_epoch=6,
        latest_metrics={"val_cer": 0.15, "train_loss": 10.0, "val_loss": 11.0},
        best_metrics={"val_cer": 0.15},
    )
    assert classify_monitoring_state(run_state=state, monitoring_config=_config(), now=state_now(state)) != DEGRADING


def test_never_declares_degrading_from_training_loss_alone():
    """With no val_loss recorded, degrading must never be inferred from train_loss alone."""
    state = _state(
        current_epoch=6,
        latest_metrics={"val_cer": 0.15, "train_loss": 5.0}, best_metrics={"val_cer": 0.15},
    )
    assert classify_monitoring_state(run_state=state, monitoring_config=_config(), now=state_now(state)) != DEGRADING


def test_plateau_confirmed_when_epochs_since_improvement_meets_patience():
    state = _state(
        current_epoch=10, latest_metrics={"val_cer": 0.205}, best_metrics={"val_cer": 0.20},
        epochs_since_improvement=5,
    )
    assert classify_monitoring_state(run_state=state, monitoring_config=_config(), now=state_now(state)) == PLATEAU_CONFIRMED


def test_plateau_candidate_when_epochs_since_improvement_is_positive_but_below_patience():
    state = _state(
        current_epoch=6, latest_metrics={"val_cer": 0.205}, best_metrics={"val_cer": 0.20},
        epochs_since_improvement=2,
    )
    assert classify_monitoring_state(run_state=state, monitoring_config=_config(), now=state_now(state)) == PLATEAU_CANDIDATE


def test_degrading_beats_plateau_candidate_on_a_real_worsening_streak():
    state = _state(
        current_epoch=6, latest_metrics={"val_cer": 0.30}, best_metrics={"val_cer": 0.20},
        epochs_since_improvement=2,
    )
    validation_history = (
        {"epoch": 4, "val_cer": 0.25}, {"epoch": 5, "val_cer": 0.27}, {"epoch": 6, "val_cer": 0.30},
    )
    assert classify_monitoring_state(
        run_state=state, monitoring_config=_config(), validation_history=validation_history, now=state_now(state)
    ) == DEGRADING


def test_healthy_with_a_single_post_warmup_datapoint():
    state = _state(current_epoch=1, latest_metrics={"val_cer": 0.20}, best_metrics={"val_cer": 0.20})
    config = _config(min_exposure_steps=None)
    validation_history = ({"epoch": 1, "val_cer": 0.20},)
    assert classify_monitoring_state(
        run_state=state, monitoring_config=config, validation_history=validation_history, now=state_now(state)
    ) == HEALTHY


def test_improving_when_deltas_show_a_steady_rate():
    state = _state(current_epoch=6, latest_metrics={"val_cer": 0.15}, best_metrics={"val_cer": 0.15})
    validation_history = tuple({"epoch": e, "val_cer": 0.30 - e * 0.02} for e in range(1, 7))
    assert classify_monitoring_state(
        run_state=state, monitoring_config=_config(), validation_history=validation_history, now=state_now(state)
    ) == IMPROVING


def test_slowing_when_latest_delta_is_much_smaller_than_the_rolling_mean():
    state = _state(current_epoch=6, latest_metrics={"val_cer": 0.1499}, best_metrics={"val_cer": 0.1499})
    validation_history = (
        {"epoch": 1, "val_cer": 0.30}, {"epoch": 2, "val_cer": 0.25}, {"epoch": 3, "val_cer": 0.20},
        {"epoch": 4, "val_cer": 0.16}, {"epoch": 5, "val_cer": 0.150}, {"epoch": 6, "val_cer": 0.1499},
    )
    assert classify_monitoring_state(
        run_state=state, monitoring_config=_config(), validation_history=validation_history, now=state_now(state)
    ) == SLOWING
