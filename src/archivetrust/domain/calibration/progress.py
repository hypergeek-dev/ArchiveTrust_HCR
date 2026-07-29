"""Calibration progress: per-pattern review count vs. statistically required count, with dynamic
sample sizing (Milestone 11, Phases 4 and 5).

Phase 5's rule, made concrete: `compute_calibration_progress` uses the worst-case `p=0.5`
assumption only when a pattern has zero completed calibration reviews. The moment even one review
exists, required-n is recomputed from the empirical proportion instead -- required review counts
shrink as real evidence accumulates, they never stay pinned to the worst case out of inertia.

Maturity categories are exactly the three statistical thresholds computed here (`required_n` at
+/-10%, +/-5%, +/-2%) -- not separately invented labels, per
`CONFIDENCE_CALIBRATION_SUFFICIENCY_REPORT.md` Sec.3, which this module supersedes with a
reusable, importable implementation instead of a one-off script.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict

from archivetrust.domain.calibration.pattern import CalibrationPattern
from archivetrust.domain.calibration.statistics import required_sample_size, wilson_ci

MARGINS: tuple[float, float, float] = (0.10, 0.05, 0.02)


class CalibrationMaturity(str, Enum):
    """Where a pattern sits on the evidence ladder (Sec.3 of the sufficiency report, now
    computed live instead of narrated by hand)."""

    NO_EVIDENCE = "no_evidence"
    EARLY_EVIDENCE = "early_evidence"
    EMERGING_CALIBRATION = "emerging_calibration"
    MODERATE_CONFIDENCE = "moderate_confidence"
    STATISTICALLY_MATURE = "statistically_mature"


class CalibrationProgress(BaseModel):
    """One pattern's calibration state: how much evidence exists, what it implies about
    correctness, and how much more evidence the statistics say is still needed.

    `reviews_completed`/`corrections_recorded` must come only from `ReviewIntent.CALIBRATION`
    corrections (unbiased, statistically-justified sampling) -- never from operational-review
    corrections, which are a deterministic, confidence-triggered subsample and would bias any
    estimate of a pattern's true population correctness. See `review/sampling/` for how intent is
    tracked and filtered before results reach this function.
    """

    model_config = ConfigDict(frozen=True)

    pattern: str
    population: int
    reviews_completed: int
    corrections_recorded: int
    """Reviews whose action was not a plain ACCEPT -- the AI's output needed correction."""

    empirical_correctness: float | None
    wilson_95_ci: tuple[float, float] | None
    used_worst_case_p: bool
    """True iff `p=0.5` (no evidence yet) was used for `required_n`; False once real evidence
    exists and the empirical proportion is used instead (Phase 5)."""
    required_n: dict[str, int]
    """Keyed `margin_pm10pct` / `margin_pm5pct` / `margin_pm2pct`."""
    maturity: CalibrationMaturity
    progress_pct_of_loosest_requirement: float | None


def compute_calibration_progress(
    pattern: CalibrationPattern,
    *,
    reviews_completed: int,
    corrections_recorded: int,
) -> CalibrationProgress:
    if corrections_recorded > reviews_completed:
        raise ValueError("corrections_recorded cannot exceed reviews_completed")

    used_worst_case_p = reviews_completed == 0
    p = 0.5 if used_worst_case_p else (reviews_completed - corrections_recorded) / reviews_completed

    required_n = {
        f"margin_pm{int(m * 100)}pct": required_sample_size(m, pattern.population, p=p)
        for m in MARGINS
    }

    empirical_correctness: float | None = None
    ci: tuple[float, float] | None = None
    if reviews_completed > 0:
        empirical_correctness = round(1 - corrections_recorded / reviews_completed, 4)
        raw_ci = wilson_ci(reviews_completed - corrections_recorded, reviews_completed)
        ci = (round(raw_ci[0], 4), round(raw_ci[1], 4)) if raw_ci else None

    maturity = _maturity_for(reviews_completed, required_n)

    loosest = required_n["margin_pm10pct"]
    progress_pct = round(100 * reviews_completed / loosest, 1) if loosest else None

    return CalibrationProgress(
        pattern=pattern.name,
        population=pattern.population,
        reviews_completed=reviews_completed,
        corrections_recorded=corrections_recorded,
        empirical_correctness=empirical_correctness,
        wilson_95_ci=ci,
        used_worst_case_p=used_worst_case_p,
        required_n=required_n,
        maturity=maturity,
        progress_pct_of_loosest_requirement=progress_pct,
    )


def _maturity_for(reviews_completed: int, required_n: dict[str, int]) -> CalibrationMaturity:
    if reviews_completed <= 0:
        return CalibrationMaturity.NO_EVIDENCE
    if reviews_completed >= required_n["margin_pm2pct"]:
        return CalibrationMaturity.STATISTICALLY_MATURE
    if reviews_completed >= required_n["margin_pm5pct"]:
        return CalibrationMaturity.MODERATE_CONFIDENCE
    if reviews_completed >= required_n["margin_pm10pct"]:
        return CalibrationMaturity.EMERGING_CALIBRATION
    return CalibrationMaturity.EARLY_EVIDENCE
