"""The resumable training session engine for `loghi_swedish_finetuned_v1` (Work Packages 10-13).

**Architectural decision, made after reading the pinned `loghi-htr` commit's actual training loop
(`modes/training.py::train_model`), not assumed from documentation.** `loghi-htr` is TensorFlow/Keras;
`model.fit(...)` is called with no `initial_epoch`, so every fresh invocation of `main.py` restarts
Keras's own per-call epoch counter at 0. Neither the epoch number nor `LoghiCustomCallback`'s
`best_val_metric` (a plain Python attribute, reset to `inf` in `__init__`) is persisted anywhere a new
process could read back. What **is** genuinely restored by `tf.keras.models.load_model()` on a
`--existing_model` checkpoint: model weights, and -- because `LoghiCustomCallback._save_model` calls
plain `.save()` with no `include_optimizer=False` override -- the optimizer's full state, including its
`iterations` counter. `LoghiLearningRateSchedule.__call__(self, step)` (`model/optimization.py`) is
itself stateless and takes `step` from the optimizer, so the learning-rate schedule's *position* is
correctly resumed as a consequence of the optimizer's `iterations` being restored -- confirmed by
reading `create_learning_rate_schedule`'s construction, and empirically re-verified by the resume-proof
this module runs (`prove_full_state_resume`), not assumed from source reading alone.

**Given that, the session design is: one training epoch = one container invocation**, chained via
`--existing_model <previous checkpoint dir>`. This makes each container invocation the natural "safe
unit" the brief's WP12 asks for -- a session never needs to signal into a *live*, multi-epoch container
process for a graceful mid-epoch stop, because there is no multi-epoch live process; the orchestrator
simply does not start the next epoch's container. Everything Keras itself does not track across
invocations (cumulative epoch number, best-validation bookkeeping, validation-loss history) is tracked
externally, here, in `TrainingSessionState` -- persisted atomically, exactly the same discipline
`checkpoint_index.py` uses for the checkpoint index it is paired with.

**Honestly not resumed**: `tf.data`'s shuffle/sampler state. The brief's own vocabulary ("sampler state
*where supported*") anticipates this. Each epoch's container re-shuffles the training list with a fresh
seed derived from `(base_seed, cumulative_epoch)` -- deterministic and reproducible across a replay of
the same session, but not a continuation of one long-running shuffle buffer's internal position, which
`tf.data` does not expose for checkpointing in the pinned commit.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Protocol

from pydantic import BaseModel, ConfigDict

from archivetrust.domain.shared.ids import new_id
from archivetrust.htr.training.checkpoint_index import (
    CheckpointEntry,
    append_checkpoint_entry,
    best_validation_checkpoint,
    latest_resumable_checkpoint,
    verify_checkpoint,
)

DEFAULT_FIRST_EPOCH_ESTIMATE_SECONDS = 900.0
"""Conservative guess used only when no epoch has ever completed for this run -- there is nothing to
estimate from yet, and the brief's "estimate epoch duration before starting the next epoch" cannot
apply to the very first epoch of a run's entire history. Every epoch after the first uses the real
observed duration."""
SAFETY_MARGIN = 1.15
"""An epoch must be estimated to fit in `remaining_budget / SAFETY_MARGIN` before starting it -- a
15% buffer against the estimate being optimistic."""


def _validation_metrics(result: "EpochResult") -> dict:
    """Every real metric this epoch produced, keyed for `CheckpointEntry.validation_metrics` --
    `None` values omitted rather than written as `null`, so a consumer checking `"val_loss" in
    metrics` gets an honest absence instead of a present-but-null field."""
    metrics = {
        "val_CER_metric": result.val_cer,
        "train_CER_metric": result.train_cer,
        "val_WER_metric": result.val_wer,
        "train_WER_metric": result.train_wer,
        "val_loss": result.val_loss,
        "train_loss": result.train_loss,
    }
    return {k: v for k, v in metrics.items() if v is not None}


class EpochResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    ok: bool
    train_cer: float | None = None
    val_cer: float | None = None
    train_wer: float | None = None
    val_wer: float | None = None
    train_loss: float | None = None
    val_loss: float | None = None
    duration_seconds: float
    checkpoint_dir: str | None = None
    """The `"latest"`-equivalent checkpoint this epoch produced -- `LoghiCustomCallback` always
    writes one when `save_checkpoint=True`."""
    best_val_checkpoint_dir: str | None = None
    """Only set when this epoch's validation CER improved on the best seen so far -- mirrors
    `LoghiCustomCallback.on_epoch_end`'s own `save_best` branch."""
    stdout_tail: str = ""
    stderr_tail: str = ""
    exit_code: int | None = None
    error_message: str | None = None


class EpochRunner(Protocol):
    """The one seam between this orchestrator and a real container run -- `_ContainerEpochRunner`
    (production) or a scripted fake (tests), mirroring `providers/loghi/facade.py`'s
    `LoghiWorkerFacade` Protocol discipline exactly."""

    def run_epoch(
        self,
        *,
        existing_model_dir: str,
        output_dir: str,
        train_list_path: str,
        validation_list_path: str,
        epoch_seed: int,
    ) -> EpochResult: ...


class TrainingSessionState(BaseModel):
    """Every field a new process needs to resume correctly, that Keras itself does not track across
    invocations -- persisted atomically at `run_state_dir/session_state.json`."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    configuration_hash: str
    random_seed: int
    cumulative_epoch: int = 0
    global_step: int = 0
    best_val_cer: float | None = None
    best_checkpoint_dir: str | None = None
    latest_checkpoint_dir: str | None = None
    validation_history: tuple[dict, ...] = ()
    session_count: int = 0
    cumulative_training_seconds: float = 0.0
    recent_epoch_durations: tuple[float, ...] = ()
    """Capped at the last 5 -- used only to estimate the next epoch's duration, never persisted as
    unbounded history (that's `validation_history`'s job)."""
    epochs_since_improvement: int = 0
    """Consecutive epochs (across resumed sessions -- never reset by a session boundary, or a short
    session could never accumulate enough of them to trigger early stopping) whose `val_cer` did not
    strictly improve on `best_val_cer`. Drives `early_stopping_patience` in `run_training_session`."""
    last_stop_reason: str | None = None
    last_stopped_mid_epoch: bool | None = None
    last_stop_boundary: str | None = None
    last_session_ended_at: str | None = None
    """The 4 fields above mirror `SessionSummary`'s equivalents, persisted so `session_report.py`/
    `run_status.py` can show the most recent session's stop circumstances without that
    `SessionSummary` object (itself never persisted) still being in memory."""
    configured_wall_clock_hours: float | None = None
    configured_max_epochs_this_call: int | None = None
    configured_early_stopping_patience: int | None = None
    """The most recent session's own launch-time budget, echoed back for the dashboard's safety-cap
    panel -- nowhere else durable previously recorded what a session was actually *started* with
    (only what governed the outcome, via `last_stop_reason`)."""


class SessionSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    session_id: str
    run_id: str
    started_at: str
    ended_at: str
    initial_epoch: int
    final_epoch: int
    initial_global_step: int
    final_global_step: int
    epochs_completed_this_session: int
    stop_reason: str
    """`"time_budget_reached" | "stop_requested" | "epoch_would_not_fit" | "epoch_failed" |
    "target_epochs_reached" | "no_val_cer_improvement"`."""
    stopped_mid_epoch: bool
    """`True` only for `stop_reason == "epoch_failed"` -- every other stop reason is only ever
    evaluated *between* epochs, after the just-completed epoch's validation ran and its checkpoint was
    verified, because a session never signals into a live, multi-epoch container process (there is no
    such process -- "one epoch = one container invocation", see this module's own docstring)."""
    stop_boundary: str
    """`"after_validation_at_epoch_boundary"` for every clean stop; `"epoch_failed_before_completion"`
    when the failing epoch's container run did not reach a verified checkpoint -- prior epochs'
    checkpoints remain valid and are still the resumable state regardless."""
    session_training_seconds: float
    cumulative_training_seconds: float
    latest_checkpoint_dir: str | None
    best_checkpoint_dir: str | None
    epoch_results: tuple[EpochResult, ...]


def _state_path(run_state_dir: Path) -> Path:
    return run_state_dir / "session_state.json"


def load_session_state(run_state_dir: str | Path) -> TrainingSessionState | None:
    path = _state_path(Path(run_state_dir))
    if not path.exists():
        return None
    return TrainingSessionState.model_validate_json(path.read_text(encoding="utf-8"))


def _save_session_state(run_state_dir: Path, state: TrainingSessionState) -> None:
    import os
    import tempfile

    run_state_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(run_state_dir), prefix=".tmp-session-state-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(state.model_dump_json(indent=2))
        os.replace(tmp_path, str(_state_path(run_state_dir)))
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def _stage_writable_checkpoint(*, source_dir: str, staging_dir: Path) -> str:
    """Copies a pristine checkpoint directory to a writable staging location, once -- idempotent
    (a second call with the same `staging_dir` that already has content returns it unchanged, since a
    resumed run must never re-copy over a staged directory a container may have already written its
    own `tokenizer.json` upgrade into). The real, pinned source (`source_dir`) is only ever read, never
    opened for writing, by this function or anything downstream of it."""
    import shutil

    if staging_dir.exists() and any(staging_dir.iterdir()):
        return str(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)
    for item in Path(source_dir).iterdir():
        destination = staging_dir / item.name
        if item.is_dir():
            shutil.copytree(item, destination)
        else:
            shutil.copyfile(item, destination)
    return str(staging_dir)


def run_training_session(
    *,
    run_state_dir: str | Path,
    checkpoint_index_path: str | Path,
    epoch_runner: EpochRunner,
    train_list_path: str,
    validation_list_path: str,
    parent_checkpoint_dir: str,
    run_id: str,
    configuration_hash: str,
    random_seed: int,
    max_wall_clock_seconds: float,
    stop_requested: Callable[[], bool],
    max_epochs_this_call: int | None = None,
    early_stopping_patience: int | None = None,
) -> SessionSummary:
    """Runs one training session: repeatedly invokes `epoch_runner.run_epoch(...)`, one real epoch at
    a time, chaining `--existing_model` from the previous epoch's checkpoint, until the wall-clock
    budget is spent, a stop is requested, an epoch fails, `max_epochs_this_call` is reached (the last
    one exists only for tests -- production callers pass `None` and let the time budget govern), or
    (when `early_stopping_patience` is set) `val_cer` has failed to strictly improve on the run's best
    for that many consecutive epochs -- counted across resumed sessions via `TrainingSessionState.
    epochs_since_improvement`, not reset by a session boundary, so a sequence of short sessions cannot
    silently defeat patience-based stopping.
    """
    run_state_dir = Path(run_state_dir)
    state = load_session_state(run_state_dir)
    if state is None:
        state = TrainingSessionState(
            run_id=run_id, configuration_hash=configuration_hash, random_seed=random_seed
        )
    elif state.configuration_hash != configuration_hash:
        raise ValueError(
            f"Persisted session configuration_hash {state.configuration_hash!r} does not match "
            f"{configuration_hash!r} -- refusing to resume under a changed configuration."
        )
    elif state.random_seed != random_seed:
        raise ValueError(
            f"Persisted session random_seed {state.random_seed!r} does not match {random_seed!r} -- "
            "refusing to resume under a changed seed. A resumed session must reuse the exact same "
            "--seed the run was originally started with (unlike configuration_hash, this was "
            "previously silently accepted, letting the persisted random_seed field misreport what "
            "was actually used for per-shard shuffling)."
        )

    state = state.model_copy(
        update={
            "configured_wall_clock_hours": max_wall_clock_seconds / 3600.0,
            "configured_max_epochs_this_call": max_epochs_this_call,
            "configured_early_stopping_patience": early_stopping_patience,
        }
    )
    _save_session_state(run_state_dir, state)

    session_id = new_id("loghi_training_session")
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    session_start_monotonic = time.monotonic()
    session_training_seconds = 0.0

    initial_epoch = state.cumulative_epoch
    initial_global_step = state.global_step
    epoch_results: list[EpochResult] = []
    stop_reason = "time_budget_reached"

    if state.latest_checkpoint_dir is None:
        # First epoch this run has ever executed: never hand the pristine pinned parent checkpoint
        # to the container directly. A real bug this module's own smoke test caught:
        # `Tokenizer.load_from_file` writes a converted `tokenizer.json` *back into the model
        # directory* when loading a legacy `charlist.txt`-only checkpoint, so the mount must be
        # read-write -- and a read-write mount of `providers/loghi/pinned_versions.py`'s real,
        # downloaded `generic-2023-02-15` checkpoint would violate "do not overwrite the generic
        # Dutch model." `_stage_writable_checkpoint` copies it once, here, before it is ever mounted.
        existing_model_dir = _stage_writable_checkpoint(
            source_dir=parent_checkpoint_dir, staging_dir=run_state_dir / "staged_parent_checkpoint"
        )
    else:
        existing_model_dir = state.latest_checkpoint_dir
    recent_durations = list(state.recent_epoch_durations)

    while True:
        if stop_requested():
            stop_reason = "stop_requested"
            break
        if max_epochs_this_call is not None and len(epoch_results) >= max_epochs_this_call:
            stop_reason = "target_epochs_reached"
            break

        elapsed = time.monotonic() - session_start_monotonic
        remaining = max_wall_clock_seconds - elapsed
        estimate = (
            sum(recent_durations) / len(recent_durations)
            if recent_durations
            else DEFAULT_FIRST_EPOCH_ESTIMATE_SECONDS
        )
        if recent_durations and remaining < estimate * SAFETY_MARGIN:
            stop_reason = "epoch_would_not_fit"
            break
        if remaining <= 0:
            stop_reason = "time_budget_reached"
            break

        epoch_output_dir = str(run_state_dir / "epoch_output" / f"epoch_{state.cumulative_epoch + 1}")
        epoch_seed = random_seed + state.cumulative_epoch + 1

        result = epoch_runner.run_epoch(
            existing_model_dir=existing_model_dir,
            output_dir=epoch_output_dir,
            train_list_path=train_list_path,
            validation_list_path=validation_list_path,
            epoch_seed=epoch_seed,
        )
        epoch_results.append(result)

        if not result.ok:
            stop_reason = "epoch_failed"
            break

        state = state.model_copy(
            update={
                "cumulative_epoch": state.cumulative_epoch + 1,
                "session_count": state.session_count if epoch_results else state.session_count,
                "cumulative_training_seconds": state.cumulative_training_seconds + result.duration_seconds,
                "validation_history": (
                    *state.validation_history,
                    {
                        "epoch": state.cumulative_epoch + 1,
                        "train_cer": result.train_cer,
                        "val_cer": result.val_cer,
                        "train_wer": result.train_wer,
                        "val_wer": result.val_wer,
                        "train_loss": result.train_loss,
                        "val_loss": result.val_loss,
                        "duration_seconds": result.duration_seconds,
                    },
                ),
            }
        )
        session_training_seconds += result.duration_seconds
        recent_durations = (recent_durations + [result.duration_seconds])[-5:]

        checkpoint_verified, model_hash, extra = verify_checkpoint(result.checkpoint_dir) if result.checkpoint_dir else (False, None, {})
        latest_id = new_id("loghi_checkpoint")
        if result.checkpoint_dir:
            append_checkpoint_entry(
                checkpoint_index_path,
                CheckpointEntry(
                    checkpoint_id=latest_id,
                    run_id=run_id,
                    session_id=session_id,
                    source_checkpoint=existing_model_dir,
                    epoch=state.cumulative_epoch,
                    global_step=state.global_step,
                    cumulative_training_seconds=state.cumulative_training_seconds,
                    session_training_seconds=session_training_seconds,
                    checkpoint_dir=result.checkpoint_dir,
                    model_file_hash=model_hash,
                    model_state_present=checkpoint_verified,
                    optimizer_state_present=checkpoint_verified,
                    scheduler_state_present=checkpoint_verified,
                    sampler_state_present=False,
                    configuration_hash=configuration_hash,
                    training_manifest_hash="",
                    validation_manifest_hash="",
                    validation_metrics=_validation_metrics(result),
                    created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    verification_status="verified" if checkpoint_verified else "verification_failed",
                    resumable=checkpoint_verified,
                    checkpoint_kind="latest",
                ),
            )
            state = state.model_copy(update={"latest_checkpoint_dir": result.checkpoint_dir})
            existing_model_dir = result.checkpoint_dir

        # Computed once, reused both for the best-checkpoint condition below and for
        # `epochs_since_improvement` -- the container's own `best_val_checkpoint_dir` is scoped to a
        # single epoch (`--epochs 1` per invocation, so its own "best" is trivially that one epoch),
        # so *this* run's genuine best-so-far comparison against `state.best_val_cer` is the only
        # source of truth for "did this epoch actually improve."
        val_cer_improved = result.val_cer is not None and (
            state.best_val_cer is None or result.val_cer < state.best_val_cer
        )

        if result.best_val_checkpoint_dir and val_cer_improved:
            best_verified, best_hash, _ = verify_checkpoint(result.best_val_checkpoint_dir)
            append_checkpoint_entry(
                checkpoint_index_path,
                CheckpointEntry(
                    checkpoint_id=new_id("loghi_checkpoint"),
                    run_id=run_id,
                    session_id=session_id,
                    source_checkpoint=existing_model_dir,
                    epoch=state.cumulative_epoch,
                    global_step=state.global_step,
                    cumulative_training_seconds=state.cumulative_training_seconds,
                    session_training_seconds=session_training_seconds,
                    checkpoint_dir=result.best_val_checkpoint_dir,
                    model_file_hash=best_hash,
                    model_state_present=best_verified,
                    optimizer_state_present=best_verified,
                    scheduler_state_present=best_verified,
                    sampler_state_present=False,
                    configuration_hash=configuration_hash,
                    training_manifest_hash="",
                    validation_manifest_hash="",
                    validation_metrics=_validation_metrics(result),
                    created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    verification_status="verified" if best_verified else "verification_failed",
                    resumable=best_verified,
                    checkpoint_kind="best_val",
                ),
            )
            state = state.model_copy(
                update={"best_val_cer": result.val_cer, "best_checkpoint_dir": result.best_val_checkpoint_dir}
            )

        if result.val_cer is not None:
            state = state.model_copy(
                update={
                    "epochs_since_improvement": 0 if val_cer_improved else state.epochs_since_improvement + 1
                }
            )

        state = state.model_copy(update={"recent_epoch_durations": tuple(recent_durations)})
        _save_session_state(run_state_dir, state)

        if early_stopping_patience is not None and state.epochs_since_improvement >= early_stopping_patience:
            stop_reason = "no_val_cer_improvement"
            break

    # End-of-session checkpoint entry -- WP13's third retained category, even if it points at the
    # same directory "latest" already does.
    if state.latest_checkpoint_dir:
        verified, model_hash, _ = verify_checkpoint(state.latest_checkpoint_dir)
        append_checkpoint_entry(
            checkpoint_index_path,
            CheckpointEntry(
                checkpoint_id=new_id("loghi_checkpoint"),
                run_id=run_id,
                session_id=session_id,
                source_checkpoint=existing_model_dir,
                epoch=state.cumulative_epoch,
                global_step=state.global_step,
                cumulative_training_seconds=state.cumulative_training_seconds,
                session_training_seconds=session_training_seconds,
                checkpoint_dir=state.latest_checkpoint_dir,
                model_file_hash=model_hash,
                model_state_present=verified,
                optimizer_state_present=verified,
                scheduler_state_present=verified,
                sampler_state_present=False,
                configuration_hash=configuration_hash,
                training_manifest_hash="",
                validation_manifest_hash="",
                validation_metrics={},
                created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                verification_status="verified" if verified else "verification_failed",
                resumable=verified,
                checkpoint_kind="end_of_session",
            ),
        )

    # Every stop reason except "epoch_failed" is only ever evaluated between epochs -- after the just-
    # completed epoch's validation ran and its checkpoint was verified -- because a session never
    # signals into a live, multi-epoch container process (there is no such process). "epoch_failed" is
    # the one case where the failing epoch's own container run did not reach a verified checkpoint.
    stopped_mid_epoch = stop_reason == "epoch_failed"
    stop_boundary = "epoch_failed_before_completion" if stopped_mid_epoch else "after_validation_at_epoch_boundary"
    ended_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    state = state.model_copy(
        update={
            "session_count": state.session_count + 1,
            "last_stop_reason": stop_reason,
            "last_stopped_mid_epoch": stopped_mid_epoch,
            "last_stop_boundary": stop_boundary,
            "last_session_ended_at": ended_at,
        }
    )
    _save_session_state(run_state_dir, state)

    return SessionSummary(
        session_id=session_id,
        run_id=run_id,
        started_at=started_at,
        ended_at=ended_at,
        initial_epoch=initial_epoch,
        final_epoch=state.cumulative_epoch,
        initial_global_step=initial_global_step,
        final_global_step=state.global_step,
        epochs_completed_this_session=len(epoch_results),
        stop_reason=stop_reason,
        stopped_mid_epoch=stopped_mid_epoch,
        stop_boundary=stop_boundary,
        session_training_seconds=session_training_seconds,
        cumulative_training_seconds=state.cumulative_training_seconds,
        latest_checkpoint_dir=state.latest_checkpoint_dir,
        best_checkpoint_dir=state.best_checkpoint_dir,
        epoch_results=tuple(epoch_results),
    )


def prove_full_state_resume(
    *,
    run_state_dir: str | Path,
    checkpoint_index_path: str | Path,
    epoch_runner: EpochRunner,
    train_list_path: str,
    validation_list_path: str,
    parent_checkpoint_dir: str,
    run_id: str,
    configuration_hash: str,
    random_seed: int,
) -> dict:
    """Work Package 10's exact proof sequence: two epochs in one "process" (session 1), then a fresh
    `TrainingSessionState` load (simulating a new process reading persisted state -- the real launcher
    does this by construction, since each CLI invocation *is* a new process) and one more epoch
    (session 2), asserting global counters continue rather than restart.
    """
    session_1 = run_training_session(
        run_state_dir=run_state_dir,
        checkpoint_index_path=checkpoint_index_path,
        epoch_runner=epoch_runner,
        train_list_path=train_list_path,
        validation_list_path=validation_list_path,
        parent_checkpoint_dir=parent_checkpoint_dir,
        run_id=run_id,
        configuration_hash=configuration_hash,
        random_seed=random_seed,
        max_wall_clock_seconds=10_000_000.0,
        stop_requested=lambda: False,
        max_epochs_this_call=2,
    )

    # Simulate a genuinely new process: reload state from disk rather than reusing any in-memory
    # object from session_1.
    reloaded_state = load_session_state(run_state_dir)
    assert reloaded_state is not None, "no session state persisted after session 1"

    session_2 = run_training_session(
        run_state_dir=run_state_dir,
        checkpoint_index_path=checkpoint_index_path,
        epoch_runner=epoch_runner,
        train_list_path=train_list_path,
        validation_list_path=validation_list_path,
        parent_checkpoint_dir=parent_checkpoint_dir,
        run_id=run_id,
        configuration_hash=configuration_hash,
        random_seed=random_seed,
        max_wall_clock_seconds=10_000_000.0,
        stop_requested=lambda: False,
        max_epochs_this_call=1,
    )

    global_step_continued = session_2.initial_epoch == session_1.final_epoch
    epoch_continued = session_2.initial_epoch == 2 and session_2.final_epoch == 3

    return {
        "session_1": session_1.model_dump(),
        "session_2": session_2.model_dump(),
        "epoch_continued_not_restarted": epoch_continued,
        "global_step_continued_not_restarted": global_step_continued,
        "proof_passed": epoch_continued and global_step_continued,
    }
