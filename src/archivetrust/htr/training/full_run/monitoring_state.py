"""Classifies a run's current monitoring state from real `FullRunState` + `FullRunMonitoringConfig`
(+ optionally the real per-shard `validation_history` from `session_state.json`) fields -- pure, no
I/O, so the GUI, CLI `status` output, and the dashboard share one real classification instead of
divergent ones.

**Deterministic priority order** (checked top to bottom, first match wins -- documented here because
it *is* the specification, not an implementation detail):

1. `FAILED` -- `run_state.status == "failed"`.
2. `COMPLETED` -- `run_state.status == "completed"`.
3. `INTERRUPTED` -- status is `"running"`/`"stopping"` but the telemetry heartbeat has gone stale
   (no update within `stale_telemetry_threshold_seconds`) -- the process died without a chance to
   record a terminal status. Mirrors `htr/training/run_status.py`'s pilot-dashboard logic exactly.
4. `STALLED` -- the run stopped with `stop_reason == "stalled_data_loading"` (real, orchestrator-
   detected: a shard took longer than the configured stall timeout).
5. `INITIALIZING` -- no shards completed yet, or fewer cumulative steps than `min_exposure_steps`
   (the brief's "never declare a plateau/regression before minimum exposure" rule, extended to every
   later state too -- nothing past this point is evaluated before minimum exposure).
6. `DEGRADING` -- either (a) `val_loss > train_loss * overfitting_ratio` (real, requires *both*
   values -- never inferred from training loss alone), or (b) the latest val_CER is worse than the
   running best by more than `meaningful_improvement_threshold` for at least
   `consecutive_worsening_threshold` shards running (a real regression, not mere non-improvement).
7. `PLATEAU_CONFIRMED` -- `epochs_since_improvement >= recommended_patience` (the exact condition
   that would trigger early stopping).
8. `PLATEAU_CANDIDATE` -- `0 < epochs_since_improvement < recommended_patience`.
9. `SLOWING` -- enough `validation_history` exists to compute a rolling mean |delta| over
   `rolling_validation_window` shards, and the latest delta's magnitude is less than half that
   rolling mean (real deceleration, not a guess).
10. `HEALTHY` -- improving, but not enough history yet to characterize a rate (e.g. only one
    post-warm-up data point).
11. `IMPROVING` -- the default once a real rate of improvement is established and nothing above
    matched.
"""

from __future__ import annotations

from archivetrust.htr.training.full_run.monitoring_config import FullRunMonitoringConfig
from archivetrust.htr.training.full_run.run_state import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_RUNNING,
    STATUS_STOPPING,
    FullRunState,
)

INITIALIZING = "INITIALIZING"
HEALTHY = "HEALTHY"
IMPROVING = "IMPROVING"
SLOWING = "SLOWING"
PLATEAU_CANDIDATE = "PLATEAU_CANDIDATE"
PLATEAU_CONFIRMED = "PLATEAU_CONFIRMED"
DEGRADING = "DEGRADING"
STALLED = "STALLED"
FAILED = "FAILED"
COMPLETED = "COMPLETED"
INTERRUPTED = "INTERRUPTED"

OVERFITTING_LOSS_GAP_RATIO = 1.5
"""val_loss more than this many times train_loss (both present in `latest_metrics`) counts toward
DEGRADING -- a real, computed ratio, never a training-loss-only judgment."""
SLOWING_RATIO = 0.5
"""The latest delta's magnitude below `rolling_mean * this ratio` counts as SLOWING."""


def _is_heartbeat_stale(last_heartbeat_at: str | None, *, now: float, threshold_seconds: float) -> bool:
    if last_heartbeat_at is None:
        return True
    import calendar
    import time

    try:
        parsed = calendar.timegm(time.strptime(last_heartbeat_at, "%Y-%m-%dT%H:%M:%SZ"))
    except ValueError:
        return True
    return (now - parsed) > threshold_seconds


def classify_monitoring_state(
    *,
    run_state: FullRunState,
    monitoring_config: FullRunMonitoringConfig,
    validation_history: tuple[dict, ...] = (),
    now: float | None = None,
) -> str:
    import time as _time

    now = now if now is not None else _time.time()

    if run_state.status == STATUS_FAILED:
        return FAILED
    if run_state.status == STATUS_COMPLETED:
        return COMPLETED
    if run_state.status in (STATUS_RUNNING, STATUS_STOPPING) and _is_heartbeat_stale(
        run_state.last_heartbeat_at, now=now, threshold_seconds=monitoring_config.stale_telemetry_threshold_seconds
    ):
        return INTERRUPTED
    if run_state.stop_reason == "stalled_data_loading":
        return STALLED

    steps_per_shard = monitoring_config.steps_per_shard or 0
    cumulative_exposure_steps = run_state.current_epoch * steps_per_shard
    if run_state.current_epoch == 0 or (
        monitoring_config.min_exposure_steps is not None and cumulative_exposure_steps < monitoring_config.min_exposure_steps
    ):
        return INITIALIZING

    latest = run_state.latest_metrics or {}
    train_loss = latest.get("train_loss")
    val_loss = latest.get("val_loss")
    if train_loss is not None and val_loss is not None and train_loss > 0 and val_loss > train_loss * OVERFITTING_LOSS_GAP_RATIO:
        return DEGRADING

    latest_val_cer = latest.get("val_cer")
    best_val_cer = (run_state.best_metrics or {}).get("val_cer")
    threshold = monitoring_config.meaningful_improvement_threshold or 0.0
    if (
        latest_val_cer is not None and best_val_cer is not None
        and latest_val_cer > best_val_cer + threshold
        and _consecutive_worsening_count(validation_history, best_val_cer=best_val_cer, threshold=threshold)
        >= monitoring_config.consecutive_worsening_threshold
    ):
        return DEGRADING

    if run_state.epochs_since_improvement >= monitoring_config.recommended_patience:
        return PLATEAU_CONFIRMED
    if run_state.epochs_since_improvement > 0:
        return PLATEAU_CANDIDATE

    window = monitoring_config.rolling_validation_window
    deltas = _val_cer_deltas(validation_history)
    if len(deltas) >= 2:
        rolling = deltas[-window:]
        rolling_mean = sum(abs(d) for d in rolling) / len(rolling)
        latest_delta = abs(deltas[-1])
        if rolling_mean > 0 and latest_delta < rolling_mean * SLOWING_RATIO:
            return SLOWING
        return IMPROVING

    return HEALTHY


def _val_cer_deltas(validation_history: tuple[dict, ...]) -> list[float]:
    vals = [h["val_cer"] for h in validation_history if h.get("val_cer") is not None]
    return [vals[i] - vals[i - 1] for i in range(1, len(vals))]


def _consecutive_worsening_count(validation_history: tuple[dict, ...], *, best_val_cer: float, threshold: float) -> int:
    """Consecutive trailing shards whose val_CER was worse than `best_val_cer` by more than
    `threshold` -- real evidence from the actual history, not merely "the current shard is worse.\""""
    count = 0
    for entry in reversed(validation_history):
        val_cer = entry.get("val_cer")
        if val_cer is None:
            break
        if val_cer > best_val_cer + threshold:
            count += 1
        else:
            break
    return count
