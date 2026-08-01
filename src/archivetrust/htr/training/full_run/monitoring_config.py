"""Translates a `PilotAnalysis` (real, observed pilot behavior) into `FullRunMonitoringConfig` --
the full-corpus run's actual stopping/checkpointing/warning policy. Every threshold here is expressed
in **steps or samples**, never a raw epoch count copied from the pilot (`docs/methods/
loghi-full-corpus-training.md` explains why: a full-corpus "epoch" -- one shard, see
`corpus_sharding.py` -- covers a materially different number of training examples than a pilot epoch).

Every derived value is a plain field on this frozen model -- the CLI's `prepare` step and the GUI's
"load monitoring config" panel both show and allow overriding each one before a run is prepared,
satisfying "make all derived monitoring values visible and editable before launch."
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from archivetrust.htr.training.full_run.pilot_analysis import PilotAnalysis

DEFAULT_MAX_FULL_RUN_EPOCHS = 500
"""A generous, explicit ceiling used only when the caller does not supply
`total_valid_corpus_line_count` to compute a "one full lap around the corpus" figure from."""
DEFAULT_PATIENCE_FLOOR = 3
DEFAULT_PATIENCE_MARGIN = 2
"""`recommended_patience = max(DEFAULT_PATIENCE_FLOOR, max_near_flat_streak + DEFAULT_PATIENCE_MARGIN)`
-- real evidence (the pilot's own longest near-flat streak before recovering) plus a margin, not a
fixed default pulled from nowhere."""
FALLBACK_PATIENCE = 5
"""Used only when the pilot recorded no usable `max_near_flat_streak` at all (e.g. too few epochs) --
matches the patience this project's own pilot sessions were actually run with, disclosed as a
fallback, not presented as derived."""
DEFAULT_LOSS_EXPLOSION_MULTIPLIER = 10.0
"""A shard's train_loss more than this many times the pilot's own highest observed train_loss is
flagged as a likely loss explosion."""
STALL_TIMEOUT_SAFETY_MARGIN = 3.0
"""A shard taking longer than `mean_shard_duration * this margin` is flagged as a possible stalled
data loader -- 3x a real observed mean, not a guessed fixed number of minutes."""


class FullRunMonitoringConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    pilot_run_id: str
    derived_at: str

    shard_line_count: int
    """One full-run "epoch" = one shard of this many training lines -- defaults to the pilot's own
    train line count for direct step-for-step comparability, operator-editable."""
    steps_per_shard: int | None
    """`ceil(shard_line_count / batch_size)` -- `None` only if the pilot's batch size could not be
    determined (`preflight` refuses to proceed in that case)."""

    min_exposure_steps: int | None
    min_exposure_basis: str
    """Which pilot epoch (`estimated_plateau_epoch`, or a disclosed fallback) `min_exposure_steps` was
    computed from, and why -- e.g. `"estimated_plateau_epoch=20 (real, observed)"` or
    `"best_pilot_epoch=22 (plateau not available, using best observed epoch instead)"`."""

    meaningful_improvement_threshold: float | None
    recommended_patience: int
    patience_basis: str

    checkpoint_frequency_shards: int = 1
    validation_frequency_shards: int = 1
    """Both `1` by construction -- every shard boundary is a container invocation boundary, and every
    invocation always validates and checkpoints (`training_session.py`'s existing per-epoch behavior),
    so "frequency" here is never coarser than one shard."""

    max_full_run_epochs: int
    """A safety ceiling on shard count, not a target -- training stops earlier via patience/manual
    stop in the normal case."""

    warning_thresholds: dict
    assumptions_and_method: str


def derive_monitoring_config(
    pilot_analysis: PilotAnalysis,
    *,
    shard_line_count: int | None = None,
    total_valid_corpus_line_count: int | None = None,
    patience_override: int | None = None,
    max_full_run_epochs_override: int | None = None,
) -> FullRunMonitoringConfig:
    resolved_shard_line_count = shard_line_count or pilot_analysis.train_line_count or (
        pilot_analysis.steps_per_pilot_epoch * pilot_analysis.batch_size
        if pilot_analysis.steps_per_pilot_epoch and pilot_analysis.batch_size
        else 9999
    )
    steps_per_shard = (
        -(-resolved_shard_line_count // pilot_analysis.batch_size) if pilot_analysis.batch_size else None
    )

    plateau_epoch = pilot_analysis.estimated_plateau_epoch
    if plateau_epoch is not None and pilot_analysis.steps_per_pilot_epoch:
        min_exposure_steps = pilot_analysis.steps_per_pilot_epoch * plateau_epoch
        min_exposure_basis = f"estimated_plateau_epoch={plateau_epoch} (real, observed or extrapolated -- see pilot_analysis.plateau_is_extrapolated)"
    elif pilot_analysis.best_pilot_epoch is not None and pilot_analysis.steps_per_pilot_epoch:
        min_exposure_steps = pilot_analysis.steps_per_pilot_epoch * pilot_analysis.best_pilot_epoch
        min_exposure_basis = f"best_pilot_epoch={pilot_analysis.best_pilot_epoch} (no plateau estimate available, using best observed epoch instead)"
    else:
        min_exposure_steps = None
        min_exposure_basis = "Could not be computed -- pilot's steps_per_pilot_epoch is unknown (missing batch_size or train line count)."

    if patience_override is not None:
        recommended_patience = patience_override
        patience_basis = "Explicitly overridden by the operator."
    elif pilot_analysis.max_near_flat_streak > 0:
        recommended_patience = max(DEFAULT_PATIENCE_FLOOR, pilot_analysis.max_near_flat_streak + DEFAULT_PATIENCE_MARGIN)
        patience_basis = (
            f"max(floor={DEFAULT_PATIENCE_FLOOR}, pilot's longest observed near-flat streak "
            f"({pilot_analysis.max_near_flat_streak}) + margin={DEFAULT_PATIENCE_MARGIN})."
        )
    else:
        recommended_patience = FALLBACK_PATIENCE
        patience_basis = f"Pilot recorded no usable near-flat streak -- falling back to {FALLBACK_PATIENCE} (this project's own pilot session default)."

    if max_full_run_epochs_override is not None:
        max_full_run_epochs = max_full_run_epochs_override
    elif total_valid_corpus_line_count:
        max_full_run_epochs = -(-total_valid_corpus_line_count // resolved_shard_line_count) * 3
    else:
        max_full_run_epochs = DEFAULT_MAX_FULL_RUN_EPOCHS

    max_train_loss = max(
        (o.train_loss for o in pilot_analysis.epoch_observations if o.train_loss is not None), default=None
    )
    mean_epoch_duration = pilot_analysis.runtime_stability.get("epoch_duration_mean_seconds")
    duration_scale = resolved_shard_line_count / (pilot_analysis.steps_per_pilot_epoch * pilot_analysis.batch_size) \
        if pilot_analysis.steps_per_pilot_epoch and pilot_analysis.batch_size else 1.0
    stall_timeout_seconds = (
        mean_epoch_duration * duration_scale * STALL_TIMEOUT_SAFETY_MARGIN if mean_epoch_duration else None
    )

    warning_thresholds = {
        "loss_explosion_multiplier": DEFAULT_LOSS_EXPLOSION_MULTIPLIER,
        "loss_explosion_absolute_reference": max_train_loss,
        "stall_timeout_seconds": stall_timeout_seconds,
        "check_nan_or_inf": True,
    }

    assumptions = (
        f"shard_line_count={resolved_shard_line_count} "
        f"({'explicit' if shard_line_count else 'defaulted to pilot train line count'}). "
        f"min_exposure_steps basis: {min_exposure_basis}. patience basis: {patience_basis}. "
        f"max_full_run_epochs: {'explicit override' if max_full_run_epochs_override is not None else (f'3x one lap of {total_valid_corpus_line_count} lines' if total_valid_corpus_line_count else 'default ceiling, no corpus size supplied')}."
    )

    return FullRunMonitoringConfig(
        pilot_run_id=pilot_analysis.pilot_run_id,
        derived_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        shard_line_count=resolved_shard_line_count,
        steps_per_shard=steps_per_shard,
        min_exposure_steps=min_exposure_steps,
        min_exposure_basis=min_exposure_basis,
        meaningful_improvement_threshold=pilot_analysis.meaningful_improvement_threshold,
        recommended_patience=recommended_patience,
        patience_basis=patience_basis,
        max_full_run_epochs=max_full_run_epochs,
        warning_thresholds=warning_thresholds,
        assumptions_and_method=assumptions,
    )


def write_monitoring_config(config: FullRunMonitoringConfig, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(output_path.parent), prefix=".tmp-monitoring-config-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(config.model_dump_json(indent=2))
        os.replace(tmp_path, str(output_path))
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def load_monitoring_config(path: str | Path) -> FullRunMonitoringConfig:
    return FullRunMonitoringConfig.model_validate_json(Path(path).read_text(encoding="utf-8"))
