"""Wraps `training_session.run_training_session` (unmodified) with full-run-specific bookkeeping:
chains through corpus shards instead of one fixed `train_list.txt`, gates early stopping behind the
derived minimum exposure, and keeps `run_state.json` (the WP6 audit file) current.

**One call to `run_training_session` per shard, always `max_epochs_this_call=1`.** This is the whole
trick that lets a full-corpus "epoch" mean "one shard" without touching `training_session.py` at all:
the orchestrator's own outer loop decides which shard's manifest becomes `train_list_path` for the
next call, based on `TrainingSessionState.cumulative_epoch` (= shards already consumed) after each
reload -- exactly the same "a new process reads persisted state" pattern `training_session.py`'s own
`prove_full_state_resume` already proves works.

**Never stops before minimum exposure.** `early_stopping_patience` is passed to
`run_training_session` as `None` (disabled) until `cumulative_exposure_steps >=
monitoring_config.min_exposure_steps` -- computed here as `shards_completed * steps_per_shard`, the
only real proxy available (`TrainingSessionState.global_step` is not actually populated by the
underlying engine; `providers/loghi`'s own optimizer step lives inside the container process).
"""

from __future__ import annotations

import math
import re
import time
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, ConfigDict

from archivetrust.htr.training.checkpoint_index import verify_checkpoint
from archivetrust.htr.training.full_run.corpus_sharding import ShardInfo
from archivetrust.htr.training.full_run.monitoring_config import FullRunMonitoringConfig
from archivetrust.htr.training.full_run.run_state import (
    FullRunState,
    heartbeat,
    load_run_state,
    mark_completed,
    mark_failed,
    mark_running,
    mark_stopped,
    mark_stopping,
    save_run_state,
)
from archivetrust.htr.training.training_session import (
    EpochRunner,
    load_session_state,
    run_training_session,
)

_OOM_SIGNATURES = ("OutOfMemoryError", "CUDA_ERROR_OUT_OF_MEMORY", "ResourceExhaustedError")


class FullRunSessionSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    stop_reason: str
    """`"max_full_run_epochs_reached" | "no_val_cer_improvement" | "epoch_failed" |
    "stop_requested" | "time_budget_reached" | "nan_or_inf_detected" | "likely_oom" |
    "stalled_data_loading"`."""
    shards_completed_this_call: int
    cumulative_shards_completed: int
    completed: bool
    """`True` only when `stop_reason` represents a genuine, verified completion -- gates
    `run_state.py::mark_completed` (never set on a failure or a mid-run stop)."""


def _has_nan_or_inf(*values: float | None) -> bool:
    return any(v is not None and (math.isnan(v) or math.isinf(v)) for v in values)


def _oom_signature_in(text: str) -> str | None:
    for signature in _OOM_SIGNATURES:
        if signature in text:
            return signature
    return None


def run_full_corpus_session(
    *,
    run_state_dir: str | Path,
    checkpoint_index_path: str | Path,
    epoch_runner: EpochRunner,
    shards: tuple[ShardInfo, ...],
    validation_list_path: str,
    parent_checkpoint_dir: str,
    run_id: str,
    configuration_hash: str,
    random_seed: int,
    monitoring_config: FullRunMonitoringConfig,
    max_wall_clock_seconds: float,
    stop_requested: Callable[[], bool],
) -> FullRunSessionSummary:
    run_state_dir = Path(run_state_dir)

    full_run_state = load_run_state(run_state_dir)
    if full_run_state is None:
        raise RuntimeError(f"No run_state.json under {run_state_dir} -- call `prepare` before `start`.")
    full_run_state = mark_running(full_run_state)
    save_run_state(run_state_dir, full_run_state)

    session_start = time.monotonic()
    shards_completed_this_call = 0
    stop_reason = "max_full_run_epochs_reached"
    completed = False

    while True:
        state = load_session_state(run_state_dir)
        cumulative_shards = state.cumulative_epoch if state else 0

        if cumulative_shards >= len(shards):
            stop_reason = "max_full_run_epochs_reached"
            completed = True
            break
        if stop_requested():
            stop_reason = "stop_requested"
            full_run_state = mark_stopping(full_run_state)
            save_run_state(run_state_dir, full_run_state)
            break

        remaining_budget = max_wall_clock_seconds - (time.monotonic() - session_start)
        if remaining_budget <= 0:
            stop_reason = "time_budget_reached"
            break

        cumulative_exposure_steps = cumulative_shards * (monitoring_config.steps_per_shard or 0)
        min_exposure_reached = (
            monitoring_config.min_exposure_steps is None
            or cumulative_exposure_steps >= monitoring_config.min_exposure_steps
        )
        patience = monitoring_config.recommended_patience if min_exposure_reached else None

        current_shard = shards[cumulative_shards]

        summary = run_training_session(
            run_state_dir=run_state_dir,
            checkpoint_index_path=checkpoint_index_path,
            epoch_runner=epoch_runner,
            train_list_path=current_shard.manifest_path,
            validation_list_path=validation_list_path,
            parent_checkpoint_dir=parent_checkpoint_dir,
            run_id=run_id,
            configuration_hash=configuration_hash,
            random_seed=random_seed,
            max_wall_clock_seconds=remaining_budget,
            stop_requested=stop_requested,
            max_epochs_this_call=1,
            early_stopping_patience=patience,
        )

        if summary.epochs_completed_this_session == 0:
            # The inner call stopped before even attempting this shard (time budget or a stop
            # request evaluated at its own loop top) -- reflect that as this call's real reason.
            stop_reason = summary.stop_reason
            break

        shards_completed_this_call += 1
        last_result = summary.epoch_results[-1]

        latest_metrics = {
            "train_cer": last_result.train_cer, "val_cer": last_result.val_cer,
            "train_wer": last_result.train_wer, "val_wer": last_result.val_wer,
            "train_loss": last_result.train_loss, "val_loss": last_result.val_loss,
        }
        reloaded_state = load_session_state(run_state_dir)
        best_metrics = {"val_cer": reloaded_state.best_val_cer} if reloaded_state else {}

        full_run_state = heartbeat(
            full_run_state,
            current_epoch=reloaded_state.cumulative_epoch if reloaded_state else cumulative_shards + 1,
            samples_processed=(reloaded_state.cumulative_epoch if reloaded_state else 0) * current_shard.line_count,
            latest_metrics=latest_metrics,
            best_metrics=best_metrics,
            epochs_since_improvement=reloaded_state.epochs_since_improvement if reloaded_state else 0,
            latest_checkpoint=summary.latest_checkpoint_dir,
            best_checkpoint=summary.best_checkpoint_dir,
        )
        save_run_state(run_state_dir, full_run_state)

        if not last_result.ok:
            stop_reason = "epoch_failed"
            break
        if _has_nan_or_inf(last_result.train_loss, last_result.val_loss, last_result.val_cer):
            stop_reason = "nan_or_inf_detected"
            break
        oom_signature = _oom_signature_in(last_result.stderr_tail or "")
        if oom_signature:
            stop_reason = "likely_oom"
            break
        stall_timeout = monitoring_config.warning_thresholds.get("stall_timeout_seconds")
        if stall_timeout and last_result.duration_seconds > stall_timeout:
            stop_reason = "stalled_data_loading"
            break
        if last_result.val_cer is None:
            stop_reason = "missing_validation_result"
            break

        if min_exposure_reached and summary.stop_reason == "no_val_cer_improvement":
            stop_reason = "no_val_cer_improvement"
            completed = True
            break

    if completed:
        final_state = load_session_state(run_state_dir)
        checkpoint_ok = (
            final_state is not None and final_state.latest_checkpoint_dir is not None
            and verify_checkpoint(final_state.latest_checkpoint_dir)[0]
        )
        if not checkpoint_ok:
            completed = False
            stop_reason = "final_checkpoint_verification_failed"

    if completed:
        full_run_state = mark_completed(full_run_state, stop_reason=stop_reason)
    elif stop_reason in ("epoch_failed", "nan_or_inf_detected", "likely_oom", "final_checkpoint_verification_failed"):
        full_run_state = mark_failed(full_run_state, stop_reason=stop_reason, failure_detail=stop_reason)
    else:
        full_run_state = mark_stopped(full_run_state, stop_reason=stop_reason)
    save_run_state(run_state_dir, full_run_state)

    final_reloaded = load_session_state(run_state_dir)
    return FullRunSessionSummary(
        stop_reason=stop_reason,
        shards_completed_this_call=shards_completed_this_call,
        cumulative_shards_completed=final_reloaded.cumulative_epoch if final_reloaded else 0,
        completed=completed,
    )
