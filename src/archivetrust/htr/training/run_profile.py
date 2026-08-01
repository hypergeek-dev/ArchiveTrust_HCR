"""Named training-session run profiles.

The launcher originally had one implicit profile ("the pilot split, ~5h") baked into its defaults.
Once a much larger full-corpus run becomes a real possibility, an implicit default is actively
dangerous: silently applying the pilot's ~5h/~9-minutes-per-epoch assumptions to a run over the full
563,933-line corpus (~7.2h/epoch, extrapolated) would terminate before completing even one epoch,
before validation or early stopping can operate at all. This module makes the profile, and its
duration assumptions, an explicit, named choice instead.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

PILOT_TRAIN_LINE_COUNT = 9999
PILOT_VAL_LINE_COUNT = 1000
PILOT_MEASURED_EPOCH_SECONDS = 547.7
"""Real, measured: one epoch over the 9999-train/1000-val pilot split, batch_size=16, on this host
(2026-08-01) -- not a guess. See `training/loghi-swedish-v1/run-state/session_state.json`'s first
`validation_history` entry."""

FULL_CORPUS_LINE_COUNT = 563_933
"""`swedish_dataset_inventory.py`'s own real count of valid lines across all 11 collections --
the pilot split (`PILOT_TRAIN_LINE_COUNT` + `PILOT_VAL_LINE_COUNT`) is a deliberately bounded ~2%
subset of this, not the whole dataset."""
FULL_CORPUS_ESTIMATED_EPOCH_SECONDS = 25_818.0
"""Extrapolated, NOT measured: fit a fixed-overhead-plus-per-line-cost line through the two real
timings available (the 25-line smoke probe and the 10,999-line pilot epoch), then scaled to
`FULL_CORPUS_LINE_COUNT`. ~7.17h/epoch. Every surface that shows this number must label it estimated
-- the same "do not interpret X as Y" discipline `session_report.py` already applies to training loss
extends here to "do not interpret an extrapolated duration as a measured one."""
FULL_CORPUS_DEFAULT_EPOCH_CAP = 3
FULL_CORPUS_DEFAULT_BUDGET_HOURS = 27.0
"""A deliberately conservative, explicitly-chosen starting budget for a first full-corpus run --
not derived from `FULL_CORPUS_ESTIMATED_EPOCH_SECONDS * FULL_CORPUS_DEFAULT_EPOCH_CAP` (~21.5h), which
would understate the real margin an unmeasured, non-linear-risk extrapolation deserves. The two
numbers are shown side by side wherever this default surfaces, never silently reconciled."""


class RunProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    dataset_description: str
    estimated_seconds_per_epoch: float
    is_estimated: bool
    """`True` for extrapolated timings (`full_corpus`) -- `False` only for a real, measured value
    (`pilot`)."""
    default_wall_clock_hours: float | None
    """`None` means this profile refuses a silent numeric default -- the caller must choose an
    explicit budget; a blank prompt response must be rejected, not defaulted."""
    default_patience: int | None
    default_epoch_cap: int | None = None
    suggested_budget_hours: float | None = None
    """A human-chosen suggested starting point shown as guidance text, distinct from
    `default_wall_clock_hours` (which this profile may deliberately leave `None`)."""
    data_prepared: bool = True
    """Whether this profile's train/val list files actually exist on disk yet -- `full_corpus`'s is
    `False` until a full-corpus inventory/split/data-prep pass has actually been run, which this
    module does not itself trigger."""


def pilot_profile() -> RunProfile:
    return RunProfile(
        name="pilot",
        dataset_description=f"pilot split ({PILOT_TRAIN_LINE_COUNT} train + {PILOT_VAL_LINE_COUNT} val lines)",
        estimated_seconds_per_epoch=PILOT_MEASURED_EPOCH_SECONDS,
        is_estimated=False,
        default_wall_clock_hours=3.0,
        default_patience=5,
    )


def full_corpus_profile(*, data_prepared: bool) -> RunProfile:
    return RunProfile(
        name="full_corpus",
        dataset_description=f"full valid corpus ({FULL_CORPUS_LINE_COUNT} lines)",
        estimated_seconds_per_epoch=FULL_CORPUS_ESTIMATED_EPOCH_SECONDS,
        is_estimated=True,
        default_wall_clock_hours=None,
        default_patience=None,
        default_epoch_cap=FULL_CORPUS_DEFAULT_EPOCH_CAP,
        suggested_budget_hours=FULL_CORPUS_DEFAULT_BUDGET_HOURS,
        data_prepared=data_prepared,
    )


def cap_shorter_than_one_epoch_warning(*, profile: RunProfile, wall_clock_hours: float) -> str | None:
    """`None` when the cap is safely long enough for at least one epoch; otherwise a ready-to-print
    warning naming the shortfall -- never silently allowed through."""
    estimated_hours = profile.estimated_seconds_per_epoch / 3600.0
    if wall_clock_hours >= estimated_hours:
        return None
    basis = "an estimated (extrapolated, not measured)" if profile.is_estimated else "the measured"
    return (
        f"WARNING: the selected wall-clock cap ({wall_clock_hours:.2f}h) is shorter than {basis} "
        f"single-epoch duration for the {profile.name!r} profile (~{estimated_hours:.2f}h). This "
        "session may stop before completing even one full epoch, before validation or early "
        "stopping can run at all."
    )
