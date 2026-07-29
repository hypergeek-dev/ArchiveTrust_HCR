"""Expected information gain from reviewing one more item of a pattern (Milestone 11, Phase 3).

The objective, per the milestone spec, is measurable statistical value -- not AI scoring. The
value function here is the expected reduction in posterior variance of a pattern's true
correctness rate under a Beta(1, 1) (uninformative) prior updated by observed reviews: a real,
well-defined Bayesian quantity, not a heuristic. It is combined with corpus reach (population
share) and sampling cost (required-n) into one ranking number, generalizing the informal
Impact/Cost ranking already used in `CONFIDENCE_CALIBRATION_SUFFICIENCY_REPORT.md` Sec.6 into a
reusable function.
"""

from __future__ import annotations

from archivetrust.domain.calibration.progress import CalibrationProgress


def _beta_variance(alpha: float, beta: float) -> float:
    return (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))


def expected_variance_reduction(reviews_completed: int, corrections_recorded: int) -> float:
    """Expected drop in posterior variance of the pattern's true correctness rate from one more
    calibration review, under a Beta(1, 1) prior updated by `reviews_completed` observations
    (`corrections_recorded` of which were failures/overrides).

    Every pattern at `reviews_completed=0` starts from the same Beta(1,1) posterior, so this value
    is identical across patterns with zero evidence -- the differentiator between them is corpus
    reach and sampling cost, applied in `expected_information_gain`, not this term alone.
    """
    successes = reviews_completed - corrections_recorded
    failures = corrections_recorded
    alpha, beta = 1 + successes, 1 + failures
    current_variance = _beta_variance(alpha, beta)
    p_success = alpha / (alpha + beta)
    variance_if_success = _beta_variance(alpha + 1, beta)
    variance_if_failure = _beta_variance(alpha, beta + 1)
    expected_next_variance = p_success * variance_if_success + (1 - p_success) * variance_if_failure
    return max(current_variance - expected_next_variance, 0.0)


def expected_information_gain(progress: CalibrationProgress, *, corpus_share: float) -> float:
    """One pattern's ranking value: expected variance reduction per review, scaled by how much of
    the corpus this pattern governs, divided by the statistically required sampling cost to reach
    a loose (+/-10%) estimate. Higher is a better use of the next available calibration review.
    """
    variance_reduction = expected_variance_reduction(
        progress.reviews_completed, progress.corrections_recorded
    )
    cost = progress.required_n["margin_pm10pct"] or 1
    return variance_reduction * corpus_share / cost
