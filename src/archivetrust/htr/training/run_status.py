"""Computes `RunStatus` -- the dashboard's "Required run metadata" (brief §"Required run metadata")
-- **entirely from existing real durable files**, plus the new `telemetry/status.json` liveness
snapshot `telemetry_sampler.py` writes. No new "run.json" source of truth is invented: identity comes
from `training_identity.json`, progress/checkpoints/launch-budget from `session_state.json`, resumable
checkpoint paths from `checkpoint_index.json`, dataset size from `pilot_split_summary.json` and the
real source inventory.

**Honest limitation, disclosed rather than guessed around.** This project's epoch-granularity scope
decision (`run_profile.py`'s module docstring) means "currently training" and "currently validating"
are not distinguishable from outside the blocking container call -- both collapse into `"running"`
here. A future per-batch-streaming phase (already scoped out, see `container_epoch_runner.py`) would
be able to split them.
"""

from __future__ import annotations

import calendar
import json
import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from archivetrust.htr.training.checkpoint_index import (
    best_validation_checkpoint,
    latest_resumable_checkpoint,
)
from archivetrust.htr.training.pilot_split import load_pilot_split_summary
from archivetrust.htr.training.training_session import load_session_state
from archivetrust.providers.loghi.pinned_versions import CURRENT_PINNED_VERSIONS

STALE_THRESHOLD_SECONDS = 60.0
"""A `telemetry/status.json` older than this (12x the sampler's own default 5s tick) is treated as
"nothing is currently updating it" -- either the epoch genuinely finished (and a terminal
`last_stop_reason` should already be recorded) or the process died without recording one, which is
exactly what makes `"interrupted"` a real, evidence-based classification rather than a guess."""

_PROFILE_BY_TRAINING_PHASE = {"pilot_10k": "pilot", "full_corpus": "full_corpus"}
"""`training_identity.py::TRAINING_PHASE` / `TRAINING_PHASE_FULL_CORPUS` are the only two real phases
this identity mechanism has ever produced -- unrecognized phases map to `"unknown"` rather than
guessing a profile name that was never actually configured."""


class RunStatus(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    run_profile: str
    status: str
    """`"initializing" | "running" | "stopping" | "completed" | "failed" | "interrupted"` -- see this
    module's docstring for why `"validating"` is not distinguished from `"running"` this phase."""
    dataset_name: str
    dataset_path: str | None
    training_line_count: int | None
    validation_line_count: int | None
    total_valid_corpus_line_count: int | None
    base_model: str
    output_dir: str
    device: str | None
    batch_size: int | None
    max_epochs: int | None
    early_stopping_patience: int | None
    wall_clock_safety_cap_hours: float | None
    start_time: str | None
    last_update_time: str | None
    completion_time: str | None
    stop_reason: str | None
    stopped_mid_epoch: bool | None
    stop_boundary: str | None
    resumable: bool
    latest_checkpoint_path: str | None
    best_checkpoint_path: str | None
    cumulative_epoch: int
    global_step: int
    best_val_cer: float | None
    best_epoch: int | None
    epochs_since_improvement: int
    session_count: int
    cumulative_training_seconds: float


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _is_stale(timestamp: str | None, *, now: float) -> bool:
    if timestamp is None:
        return True
    try:
        # `calendar.timegm`, not `time.mktime` -- the parsed struct is UTC (the "Z" suffix), and
        # `mktime` would incorrectly reinterpret it as local time (off by the local UTC offset, and
        # by an extra hour whenever DST is active).
        parsed = calendar.timegm(time.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ"))
    except ValueError:
        return True
    return (now - parsed) > STALE_THRESHOLD_SECONDS


def compute_run_status(run_dir: str | Path) -> RunStatus:
    """`run_dir` is a run's root (e.g. `training/loghi-swedish-v1/`), containing `run-state/`,
    `manifests/`, `reports/`, `prepared-data/`. Raises `FileNotFoundError` if `training_identity.json`
    is missing -- callers (the ViewModel's run-discovery) only ever call this for a directory they
    already confirmed has one."""
    run_dir = Path(run_dir)
    run_state_dir = run_dir / "run-state"
    identity_path = run_state_dir / "training_identity.json"
    if not identity_path.exists():
        raise FileNotFoundError(f"No training_identity.json under {run_state_dir} -- not a training run directory.")
    identity_payload = json.loads(identity_path.read_text(encoding="utf-8"))
    identity = identity_payload["identity"]

    state = load_session_state(run_state_dir)
    telemetry_status = _read_json(run_state_dir / "telemetry" / "status.json")
    stop_sentinel_present = (run_state_dir / "STOP_REQUESTED").exists()

    now = time.time()
    telemetry_fresh = telemetry_status is not None and not _is_stale(telemetry_status.get("last_sample_at"), now=now)
    container_alive = bool(telemetry_status.get("container_alive")) if telemetry_status else False

    if state is None:
        status = "initializing"
    elif telemetry_fresh and container_alive:
        status = "stopping" if stop_sentinel_present else "running"
    elif state.last_stop_reason is None:
        # A session was configured (state exists) but has neither a fresh liveness heartbeat nor any
        # recorded terminal reason.
        if telemetry_status is not None:
            # A heartbeat existed at some point but is not fresh now -- the process died mid-run.
            status = "interrupted"
        elif state.cumulative_epoch > 0:
            # No heartbeat was ever recorded for this session at all (e.g. a session launched before
            # `telemetry_sampler.py` was wired in, or run without `run_state_dir` configured), but
            # real progress exists and nothing recorded a stop -- the best available inference from
            # last-known progress is that it is still running, not a live liveness confirmation.
            status = "running"
        else:
            status = "initializing"
    elif telemetry_status is not None and not telemetry_fresh and state.last_session_ended_at is None:
        # A heartbeat exists, has gone stale, and no session ever recorded a clean end -- the process
        # was killed mid-run without a chance to write `last_stop_reason`.
        status = "interrupted"
    elif state.last_stop_reason == "epoch_failed":
        status = "failed"
    else:
        status = "completed"

    split_summary_path = run_dir / "manifests" / "pilot_split_summary.json"
    split_summary = load_pilot_split_summary(run_dir / "manifests") if split_summary_path.exists() else None

    prep_report = _read_json(run_dir / "prepared-data" / "preparation_report.json")
    memory_probe = _read_json(run_dir / "reports" / "memory-probe.json")

    total_valid_corpus_line_count = None
    inventory_path = run_dir / "source-inventory" / "full-inventory.parquet"
    if inventory_path.exists():
        import pyarrow.parquet as pq

        table = pq.read_table(inventory_path, columns=["valid"])
        total_valid_corpus_line_count = sum(table.column("valid").to_pylist())

    checkpoint_index_path = run_state_dir / "checkpoint_index.json"
    latest = latest_resumable_checkpoint(checkpoint_index_path)
    best = best_validation_checkpoint(checkpoint_index_path)

    last_update_candidates = [
        c for c in (
            identity.get("created_at"),
            state.last_session_ended_at if state else None,
            telemetry_status.get("last_sample_at") if telemetry_status else None,
        ) if c is not None
    ]

    return RunStatus(
        run_id=identity["run_id"],
        run_profile=_PROFILE_BY_TRAINING_PHASE.get(identity.get("training_phase"), "unknown"),
        status=status,
        dataset_name=identity.get("training_dataset", "unknown"),
        dataset_path=prep_report.get("source_inventory_path") if prep_report else None,
        training_line_count=split_summary.actual_train if split_summary else None,
        validation_line_count=split_summary.actual_val if split_summary else None,
        total_valid_corpus_line_count=total_valid_corpus_line_count,
        base_model=identity.get("parent_checkpoint", "unknown"),
        output_dir=str(run_state_dir),
        device=f"nvidia-gpu:{CURRENT_PINNED_VERSIONS.gpu_selection}" if CURRENT_PINNED_VERSIONS.gpu_selection else None,
        batch_size=memory_probe.get("chosen_batch_size") if memory_probe else None,
        max_epochs=state.configured_max_epochs_this_call if state else None,
        early_stopping_patience=state.configured_early_stopping_patience if state else None,
        wall_clock_safety_cap_hours=state.configured_wall_clock_hours if state else None,
        start_time=identity.get("created_at"),
        last_update_time=max(last_update_candidates) if last_update_candidates else None,
        completion_time=state.last_session_ended_at if state and status in ("completed", "failed") else None,
        stop_reason=state.last_stop_reason if state else None,
        stopped_mid_epoch=state.last_stopped_mid_epoch if state else None,
        stop_boundary=state.last_stop_boundary if state else None,
        resumable=latest is not None,
        latest_checkpoint_path=latest.checkpoint_dir if latest else None,
        best_checkpoint_path=best.checkpoint_dir if best else None,
        cumulative_epoch=state.cumulative_epoch if state else 0,
        global_step=state.global_step if state else 0,
        best_val_cer=state.best_val_cer if state else None,
        best_epoch=best.epoch if best else None,
        epochs_since_improvement=state.epochs_since_improvement if state else 0,
        session_count=state.session_count if state else 0,
        cumulative_training_seconds=state.cumulative_training_seconds if state else 0.0,
    )
