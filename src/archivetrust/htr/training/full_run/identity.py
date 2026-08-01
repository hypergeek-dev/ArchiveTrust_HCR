"""Full-corpus run identity -- reuses `training_identity.py::create_or_load_identity` exactly
(passing `TRAINING_PHASE_FULL_CORPUS`), adding only the two extra structural safety rules a *fresh*
full-corpus run needs that a resumed pilot session never had to enforce: never silently reuse an
existing run directory, and never accept a parent checkpoint path that is actually a pilot run's own
output.

**Why rejecting a pilot-path parent checkpoint is enough to guarantee a fresh optimizer/scheduler.**
`training_session.py::_stage_writable_checkpoint` copies `parent_checkpoint_dir` once, before any
epoch runs, into `run_state_dir/staged_parent_checkpoint/` -- a brand-new `run_state_dir` has no prior
`session_state.json`, so the very first epoch always starts from that staged copy. As long as
`parent_checkpoint_dir` is never a pilot epoch's own checkpoint output, the pristine pinned checkpoint
is what gets staged -- and (confirmed by reading `model/management.py::load_model_from_directory`
directly) loading *that* pristine, old-format checkpoint goes through the `_convert_old_model_to_new`
weights-only fallback path, which never touches optimizer state. Fresh optimizer/scheduler state is
therefore a structural consequence of this one check, not separate engineering.
"""

from __future__ import annotations

import time
from pathlib import Path

from archivetrust.htr.training.training_identity import (
    TRAINING_PHASE_FULL_CORPUS,
    TrainingConfiguration,
    TrainingIdentity,
    create_or_load_identity,
)


class PilotCheckpointRejected(RuntimeError):
    """Raised when a full-run's `parent_checkpoint`/`parent_checkpoint_dir` resolves to (or names)
    a pilot run's own directory -- the brief's "never treat the pilot's final checkpoint as the
    full-run starting model" and "prevent accidental loading of pilot checkpoints" rules, enforced
    structurally rather than by convention."""


class RunDirectoryAlreadyExists(RuntimeError):
    """Raised when `prepare` is asked to create a full run at a `run_state_dir` that already has a
    `training_identity.json` -- the brief's "never overwrite a run directory" rule."""


def generate_run_name(*, base_name: str | None = None) -> str:
    """An explicit `base_name` is used verbatim; otherwise a unique, timestamped identifier is
    generated -- "the reset operation must require an explicit full-run name or generate a unique
    timestamped run identifier," never silently defaulting to something that could collide."""
    if base_name:
        return base_name
    return "full-corpus-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def assert_parent_checkpoint_is_not_a_pilot_path(
    parent_checkpoint_dir: str | Path, *, known_pilot_run_dirs: tuple[str | Path, ...]
) -> None:
    resolved = Path(parent_checkpoint_dir).resolve()
    for pilot_dir in known_pilot_run_dirs:
        pilot_resolved = Path(pilot_dir).resolve()
        if resolved == pilot_resolved or resolved.is_relative_to(pilot_resolved):
            raise PilotCheckpointRejected(
                f"Refusing to use {parent_checkpoint_dir!r} as the full run's parent checkpoint -- it "
                f"is under the known pilot run directory {pilot_dir!r}. The full run must start from "
                "the original pristine pinned base checkpoint, never from any pilot output."
            )


def create_full_run_identity(
    *,
    run_state_dir: str | Path,
    parent_checkpoint: str,
    parent_checkpoint_dir: str | Path,
    configuration: TrainingConfiguration,
    known_pilot_run_dirs: tuple[str | Path, ...] = (),
) -> tuple[TrainingIdentity, str]:
    """The full-run counterpart of `training_identity.create_or_load_identity` -- refuses to reuse
    an existing run directory (a full run is always freshly prepared, never silently resumed by this
    function; `resume` is a separate, explicit CLI verb) and refuses a pilot-path parent checkpoint.
    """
    assert_parent_checkpoint_is_not_a_pilot_path(parent_checkpoint_dir, known_pilot_run_dirs=known_pilot_run_dirs)

    run_state_dir = Path(run_state_dir)
    if (run_state_dir / "training_identity.json").exists():
        raise RunDirectoryAlreadyExists(
            f"{run_state_dir} already has a training_identity.json -- `prepare` never overwrites an "
            "existing run directory. Choose a new run name, or use `resume` to continue this one."
        )

    return create_or_load_identity(
        run_state_dir=run_state_dir,
        parent_checkpoint=parent_checkpoint,
        configuration=configuration,
        training_phase=TRAINING_PHASE_FULL_CORPUS,
    )
