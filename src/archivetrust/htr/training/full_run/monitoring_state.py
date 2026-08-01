"""Classifies a run's current monitoring state (brief's GUI requirement: "warm-up/minimum-exposure
phase | improving | plateau candidate | early-stop patience counter | likely overfitting | failed |
completed") from real `FullRunState` + `FullRunMonitoringConfig` fields -- pure, no I/O, so both the
GUI and the CLI's `status` output can share one real classification instead of two divergent ones.
"""

from __future__ import annotations

from archivetrust.htr.training.full_run.monitoring_config import FullRunMonitoringConfig
from archivetrust.htr.training.full_run.run_state import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    FullRunState,
)

WARM_UP = "warm_up_minimum_exposure_phase"
IMPROVING = "improving"
PLATEAU_CANDIDATE = "plateau_candidate"
LIKELY_OVERFITTING = "likely_overfitting"
FAILED = "failed"
COMPLETED = "completed"

OVERFITTING_LOSS_GAP_RATIO = 1.5
"""val_loss more than this many times train_loss (both present in `latest_metrics`) is flagged as
likely overfitting -- a real, computed ratio, never a training-loss-only judgment (the brief's "never
declare a plateau using training loss alone" extends to overfitting too)."""


def classify_monitoring_state(
    *, run_state: FullRunState, monitoring_config: FullRunMonitoringConfig
) -> str:
    if run_state.status == STATUS_FAILED:
        return FAILED
    if run_state.status == STATUS_COMPLETED:
        return COMPLETED

    steps_per_shard = monitoring_config.steps_per_shard or 0
    cumulative_exposure_steps = run_state.current_epoch * steps_per_shard
    if monitoring_config.min_exposure_steps is not None and cumulative_exposure_steps < monitoring_config.min_exposure_steps:
        return WARM_UP

    latest = run_state.latest_metrics or {}
    train_loss = latest.get("train_loss")
    val_loss = latest.get("val_loss")
    if train_loss is not None and val_loss is not None and train_loss > 0 and val_loss > train_loss * OVERFITTING_LOSS_GAP_RATIO:
        return LIKELY_OVERFITTING

    # `epochs_since_improvement == 0` means the most recent shard *was* a new best (or this is the
    # first shard) -- the one real signal that distinguishes "just improved" from "plateaued" without
    # the ambiguity a raw `latest_val_cer` vs `best_val_cer` comparison has (they are equal exactly
    # when the latest shard *set* the best, which a gap-based check would misread as "close to the
    # best" -- i.e. a plateau -- instead of "is the best, right now").
    if run_state.epochs_since_improvement > 0:
        return PLATEAU_CANDIDATE

    return IMPROVING
