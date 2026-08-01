"""`FullRunState` (`run_state.json`) -- the full-corpus run's audit/evidence file (brief §6). Written
atomically (write-temp-then-`os.replace`, the same discipline `checkpoint_index.py`/
`telemetry_sampler.py` already use) so a reader never sees a torn write. Every transition function
below returns a new, updated `FullRunState` -- the model itself stays frozen, matching every other
state object in this project (`TrainingSessionState`, `CheckpointEntry`).
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict

STATUS_PREPARED = "prepared"
STATUS_RUNNING = "running"
STATUS_STOPPING = "stopping"
STATUS_STOPPED = "stopped"
STATUS_FAILED = "failed"
STATUS_COMPLETED = "completed"


class FullRunState(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    status: str
    """`"prepared" | "running" | "stopping" | "stopped" | "failed" | "completed"`."""
    pid: int | None = None
    started_at: str | None = None
    last_resumed_at: str | None = None
    ended_at: str | None = None
    last_heartbeat_at: str | None = None
    current_epoch: int = 0
    """The current shard index -- one full-run "epoch" = one shard (`corpus_sharding.py`)."""
    current_global_step: int = 0
    samples_processed: int = 0
    latest_metrics: dict = {}
    best_metrics: dict = {}
    epochs_since_improvement: int = 0
    """Echoed from `TrainingSessionState.epochs_since_improvement` after each shard -- the GUI's
    "early-stop patience counter" needs this, and `run_state.json` is the one audit file meant to
    carry it without a reader having to separately open `session_state.json`."""
    latest_checkpoint: str | None = None
    best_checkpoint: str | None = None
    resume_count: int = 0
    stop_reason: str | None = None
    failure_detail: str | None = None
    configuration_hash: str
    dataset_hash: str | None = None
    code_revision: str | None = None


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def get_code_revision() -> str | None:
    """`git rev-parse HEAD`, best-effort -- `None` (never a fabricated revision) when git is
    unavailable or this isn't a git checkout."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5.0, check=True,
        )
        return completed.stdout.strip() or None
    except (subprocess.SubprocessError, OSError):
        return None


def _run_state_path(run_state_dir: str | Path) -> Path:
    return Path(run_state_dir) / "run_state.json"


def create_initial_run_state(
    *, run_id: str, configuration_hash: str, dataset_hash: str | None = None
) -> FullRunState:
    return FullRunState(
        run_id=run_id,
        status=STATUS_PREPARED,
        configuration_hash=configuration_hash,
        dataset_hash=dataset_hash,
        code_revision=get_code_revision(),
    )


def load_run_state(run_state_dir: str | Path) -> FullRunState | None:
    path = _run_state_path(run_state_dir)
    if not path.exists():
        return None
    return FullRunState.model_validate_json(path.read_text(encoding="utf-8"))


def save_run_state(run_state_dir: str | Path, state: FullRunState) -> None:
    run_state_dir = Path(run_state_dir)
    run_state_dir.mkdir(parents=True, exist_ok=True)
    path = _run_state_path(run_state_dir)
    fd, tmp_path = tempfile.mkstemp(dir=str(run_state_dir), prefix=".tmp-run-state-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(state.model_dump_json(indent=2))
        os.replace(tmp_path, str(path))
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def mark_running(state: FullRunState) -> FullRunState:
    now = _now()
    return state.model_copy(update={
        "status": STATUS_RUNNING, "pid": os.getpid(), "last_heartbeat_at": now,
        "started_at": state.started_at or now,
    })


def heartbeat(state: FullRunState, **progress) -> FullRunState:
    """Updates `last_heartbeat_at` (and `pid`, in case a resumed process has a new one) plus any
    real progress fields passed as keyword overrides -- called after every shard, never inferred
    from console output."""
    return state.model_copy(update={"last_heartbeat_at": _now(), "pid": os.getpid(), **progress})


def mark_stopping(state: FullRunState) -> FullRunState:
    return state.model_copy(update={"status": STATUS_STOPPING})


def mark_stopped(state: FullRunState, *, stop_reason: str) -> FullRunState:
    return state.model_copy(update={"status": STATUS_STOPPED, "stop_reason": stop_reason, "ended_at": _now()})


def mark_failed(state: FullRunState, *, stop_reason: str, failure_detail: str) -> FullRunState:
    return state.model_copy(update={
        "status": STATUS_FAILED, "stop_reason": stop_reason, "failure_detail": failure_detail, "ended_at": _now(),
    })


def mark_completed(state: FullRunState, *, stop_reason: str) -> FullRunState:
    """Only ever called after final validation + `verify_checkpoint` success -- the brief's "never
    mark a run complete unless final validation and checkpoint verification succeed," enforced by
    `orchestrator.py`'s caller, not by this function alone (which just records the transition)."""
    return state.model_copy(update={"status": STATUS_COMPLETED, "stop_reason": stop_reason, "ended_at": _now()})


def mark_resumed(state: FullRunState) -> FullRunState:
    return state.model_copy(update={
        "status": STATUS_RUNNING, "resume_count": state.resume_count + 1, "last_resumed_at": _now(),
        "pid": os.getpid(),
    })
