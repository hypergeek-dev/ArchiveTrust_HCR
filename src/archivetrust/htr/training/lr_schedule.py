"""Pure port of `LoghiLearningRateSchedule` (`.loghi-upstream/loghi-htr/src/model/optimization.py`,
pinned commit `providers/loghi/pinned_versions.py::CURRENT_PINNED_VERSIONS.loghi_htr_commit`) --
lets the dashboard compute "learning rate at global_step N" from the real, recorded schedule
parameters each epoch's own `config.json` already contains (`args.learning_rate.*`), without ever
scraping the container's live stdout for it.

**Why this is safe to reconstruct exactly, not approximate.** `LoghiLearningRateSchedule.__call__`
is a pure, stateless function of `step` (`model/optimization.py:68`) -- it holds no hidden state of
its own; everything it needs is `(initial_learning_rate, decay_rate, decay_steps, warmup_ratio,
total_steps, decay_per_epoch, linear_decay)`, all of which are written into `config.json` verbatim
(confirmed against a real completed epoch's `config.json`: `learning_rate.{decay_per_epoch: false,
decay_rate: 0.99, decay_steps: -1, learning_rate: 0.0001, linear_decay: false, warmup_ratio: 0.0}`).

**The one real subtlety**: `create_learning_rate_schedule` (`model/optimization.py:199`) resolves
`decay_steps == -1` to that invocation's own `train_batches` count, and sets
`total_steps = epochs * train_batches + 1` -- and because "one epoch = one container invocation"
(`--epochs 1` always), a *fresh* `LoghiLearningRateSchedule` instance is constructed every epoch,
with `total_steps`/`decay_steps` computed from *that epoch's* batch count, not the whole multi-epoch
run's. What *does* carry over across epochs is the optimizer's own `iterations` counter (restored
from the saved checkpoint, per `training_session.py`'s own module docstring), so `step` keeps
counting up across epochs even though each epoch's schedule object is freshly constructed.
`resolve_schedule_parameters` below takes that into account explicitly rather than silently assuming
`decay_steps` is a fixed run-wide constant.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict


class LrScheduleParameters(BaseModel):
    """Mirrors `config.json`'s real `args.learning_rate` block plus the one value that block does
    not itself carry: this epoch's real batch count (needed to resolve `decay_steps == -1`)."""

    model_config = ConfigDict(frozen=True)

    initial_learning_rate: float
    decay_rate: float
    decay_steps: int
    """`-1` as recorded in a real `config.json` means "use `train_batches_this_epoch"" -- already
    resolved to a concrete positive value by `resolve_schedule_parameters`, never left as `-1` here."""
    warmup_ratio: float
    total_steps: int
    decay_per_epoch: bool
    linear_decay: bool


def resolve_schedule_parameters(
    *,
    learning_rate: float,
    decay_rate: float,
    decay_steps: int,
    warmup_ratio: float,
    train_batches_this_epoch: int,
    epochs_this_invocation: int = 1,
) -> LrScheduleParameters:
    """Reproduces `create_learning_rate_schedule`'s own resolution exactly: `decay_steps == -1`
    becomes `train_batches_this_epoch`; `total_steps = epochs_this_invocation * train_batches_this_epoch
    + 1` (always `epochs_this_invocation=1` for this project's one-epoch-per-invocation design, kept
    as a parameter rather than hardcoded so this function stays a faithful, general port)."""
    resolved_decay_steps = train_batches_this_epoch if decay_steps == -1 else decay_steps
    return LrScheduleParameters(
        initial_learning_rate=learning_rate,
        decay_rate=decay_rate,
        decay_steps=resolved_decay_steps,
        warmup_ratio=warmup_ratio,
        total_steps=epochs_this_invocation * train_batches_this_epoch + 1,
        decay_per_epoch=False,
        linear_decay=False,
    )


def learning_rate_at(step: int, params: LrScheduleParameters) -> float:
    """Faithful port of `LoghiLearningRateSchedule.__call__` (`model/optimization.py:68-128`) --
    same branch structure, same formulas, evaluated in plain Python instead of `tf.cond`/`tf.Tensor`."""
    warmup_steps = params.warmup_ratio * params.total_steps

    if step < warmup_steps:
        if warmup_steps == 0:
            return params.initial_learning_rate
        return params.initial_learning_rate * (step / warmup_steps)

    if params.linear_decay:
        if params.decay_per_epoch:
            epoch = math.floor(step / params.decay_steps)
            total_epochs = math.floor((params.total_steps - warmup_steps) / params.decay_steps)
            proportion_completed = epoch / total_epochs if total_epochs else 0.0
        else:
            denominator = params.total_steps - warmup_steps
            proportion_completed = (step - warmup_steps) / denominator if denominator else 0.0
        return max(params.initial_learning_rate * (1 - proportion_completed), 0.0)

    # Exponential decay (the pinned default: `linear_decay=False`).
    if params.decay_per_epoch:
        exponent = math.floor(step / params.decay_steps)
    else:
        exponent = (step - warmup_steps) / params.decay_steps
    return params.initial_learning_rate * (params.decay_rate**exponent)
