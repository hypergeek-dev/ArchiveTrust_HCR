"""Correct epoch/shard progress accounting for the full-corpus run.

**The terminology this module exists to fix.** The orchestrator's internal unit of work is a *shard*
-- one container invocation over ~9,999 lines. `TrainingSessionState.cumulative_epoch` counts those
shards, and because the pilot's single 9,999-line pass genuinely was its whole training set, the
codebase inherited the habit of calling one shard "an epoch". At full-corpus scale that is simply
wrong: one pass over the 562,123-line corpus takes **57 shards**, so completing one shard is 1/57th
of an epoch, not an epoch.

Reporting a shard as an epoch overstates progress by 57x and invites false conclusions -- a single
shard's validation result is not evidence about epoch-level convergence or overfitting.

`shards_per_epoch` is derived from the real shard plan (`sharding_summary.json`), never assumed: it is
the number of shards in lap 0, which by construction covers every usable line exactly once. Laps and
epochs are the same partition viewed from two sides -- `corpus_sharding.py` builds a lap as one
complete pass, so lap boundaries *are* epoch boundaries.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class EpochPosition(BaseModel):
    """Where a run actually is, in units that mean what they say."""

    model_config = ConfigDict(frozen=True)

    global_shards_completed: int
    """Total shard executions finished across the whole run -- the raw counter the orchestrator
    increments. Formerly (and misleadingly) surfaced as 'cumulative_epoch'."""
    total_shards_planned: int
    shards_per_epoch: int
    """Shards in one complete pass over the corpus (57 for the real plan), derived from lap 0."""
    epochs_completed: int
    """Full passes over the corpus finished. Only increments when a lap's final shard completes."""
    shards_completed_in_current_epoch: int
    """0..shards_per_epoch-1 -- progress within the epoch currently in flight."""
    epoch_progress: float
    """`shards_completed_in_current_epoch / shards_per_epoch`, 0.0-1.0."""
    total_epochs_planned: int
    is_epoch_boundary: bool
    """True when the last completed shard finished a full pass, i.e. an end-of-epoch checkpoint and a
    full validation gate are due."""

    def summary_line(self) -> str:
        return (
            f"epoch {self.epochs_completed + (0 if self.is_epoch_boundary else 1)}"
            f" | shards_completed_in_current_epoch: {self.shards_completed_in_current_epoch}"
            f"/{self.shards_per_epoch}"
            f" | epochs_completed: {self.epochs_completed}"
            f" | epoch_progress: {self.epoch_progress:.4f}"
            f" | global_shards_completed: {self.global_shards_completed}/{self.total_shards_planned}"
        )


def shards_per_epoch_from_plan(shards) -> int:
    """Number of shards in one full corpus pass, read from the real plan rather than assumed.

    Counts lap-0 shards: `corpus_sharding.build_full_corpus_shards` emits lap 0 as exactly one
    complete, duplicate-free pass over every usable line, so its length is the epoch size. Falls back
    to the total shard count when the plan carries no lap information at all (a degenerate single-lap
    plan), which keeps the accounting honest instead of dividing by zero."""
    lap0 = sum(1 for s in shards if getattr(s, "lap", 0) == 0)
    return lap0 or len(shards) or 1


def compute_epoch_position(*, global_shards_completed: int, shards, total_shards_planned: int | None = None) -> EpochPosition:
    per_epoch = shards_per_epoch_from_plan(shards)
    total = total_shards_planned if total_shards_planned is not None else len(shards)
    completed_epochs = global_shards_completed // per_epoch
    within = global_shards_completed % per_epoch
    return EpochPosition(
        global_shards_completed=global_shards_completed,
        total_shards_planned=total,
        shards_per_epoch=per_epoch,
        epochs_completed=completed_epochs,
        shards_completed_in_current_epoch=within,
        epoch_progress=within / per_epoch,
        total_epochs_planned=max(1, -(-total // per_epoch)),
        # A boundary is "the shard just completed was the last of a lap": within wraps to 0 while at
        # least one shard has run. Deliberately not `within == per_epoch - 1`, which would fire one
        # shard early and promote an end_of_epoch checkpoint before the epoch was actually finished.
        is_epoch_boundary=global_shards_completed > 0 and within == 0,
    )
