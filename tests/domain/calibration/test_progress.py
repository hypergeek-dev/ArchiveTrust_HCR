from __future__ import annotations

import pytest

from archivetrust.domain.calibration.pattern import CalibrationPattern
from archivetrust.domain.calibration.progress import (
    CalibrationMaturity,
    compute_calibration_progress,
)

PATTERN = CalibrationPattern(name="contested/paragraph", population=4866, queued_for_review=4866)


def test_zero_reviews_uses_worst_case_p_and_is_no_evidence():
    progress = compute_calibration_progress(PATTERN, reviews_completed=0, corrections_recorded=0)
    assert progress.used_worst_case_p is True
    assert progress.maturity == CalibrationMaturity.NO_EVIDENCE
    assert progress.empirical_correctness is None
    assert progress.wilson_95_ci is None
    # Matches the known worst-case figures from the sufficiency report.
    assert progress.required_n["margin_pm10pct"] == 95


def test_required_n_shrinks_once_real_evidence_exists():
    """Phase 5's core acceptance criterion: required-n must never stay pinned to the worst case
    once real evidence exists -- it must be recomputed from the empirical proportion."""
    baseline = compute_calibration_progress(PATTERN, reviews_completed=0, corrections_recorded=0)
    # Strong, lopsided evidence (97.5% correct) -- far from the worst-case p=0.5 assumption.
    with_evidence = compute_calibration_progress(
        PATTERN, reviews_completed=400, corrections_recorded=10
    )
    assert with_evidence.used_worst_case_p is False
    for margin in ("margin_pm10pct", "margin_pm5pct", "margin_pm2pct"):
        assert with_evidence.required_n[margin] < baseline.required_n[margin]


def test_maturity_ladder_progresses_with_more_evidence():
    # p=0.8 (not extreme) so required-n stays large enough to actually observe every rung.
    small = compute_calibration_progress(PATTERN, reviews_completed=10, corrections_recorded=2)
    assert small.maturity == CalibrationMaturity.EARLY_EVIDENCE

    at_loosest_threshold = compute_calibration_progress(
        PATTERN,
        reviews_completed=small.required_n["margin_pm10pct"],
        corrections_recorded=round(0.2 * small.required_n["margin_pm10pct"]),
    )
    assert at_loosest_threshold.maturity in (
        CalibrationMaturity.EMERGING_CALIBRATION,
        CalibrationMaturity.MODERATE_CONFIDENCE,
        CalibrationMaturity.STATISTICALLY_MATURE,
    )

    mature = compute_calibration_progress(PATTERN, reviews_completed=5000, corrections_recorded=1000)
    assert mature.maturity == CalibrationMaturity.STATISTICALLY_MATURE


def test_empirical_correctness_and_ci_computed_when_evidence_exists():
    progress = compute_calibration_progress(PATTERN, reviews_completed=100, corrections_recorded=5)
    assert progress.empirical_correctness == 0.95
    assert progress.wilson_95_ci is not None
    lo, hi = progress.wilson_95_ci
    assert lo < 0.95 < hi


def test_corrections_cannot_exceed_reviews():
    with pytest.raises(ValueError):
        compute_calibration_progress(PATTERN, reviews_completed=5, corrections_recorded=6)
