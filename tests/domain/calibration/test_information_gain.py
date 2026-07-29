from __future__ import annotations

from archivetrust.domain.calibration.information_gain import (
    expected_information_gain,
    expected_variance_reduction,
)
from archivetrust.domain.calibration.pattern import CalibrationPattern
from archivetrust.domain.calibration.progress import compute_calibration_progress


def test_variance_reduction_is_positive_at_zero_evidence():
    assert expected_variance_reduction(0, 0) > 0


def test_variance_reduction_shrinks_as_evidence_accumulates():
    early = expected_variance_reduction(2, 0)
    late = expected_variance_reduction(500, 5)
    assert late < early


def test_larger_pattern_scores_higher_at_equal_evidence():
    small = CalibrationPattern(name="a", population=100, queued_for_review=0)
    large = CalibrationPattern(name="b", population=100_000, queued_for_review=0)
    small_progress = compute_calibration_progress(small, reviews_completed=0, corrections_recorded=0)
    large_progress = compute_calibration_progress(large, reviews_completed=0, corrections_recorded=0)

    small_gain = expected_information_gain(small_progress, corpus_share=0.001)
    large_gain = expected_information_gain(large_progress, corpus_share=0.5)
    assert large_gain > small_gain


def test_mature_pattern_scores_lower_than_unreviewed_pattern_of_same_size():
    pattern = CalibrationPattern(name="p", population=10_000, queued_for_review=0)
    fresh = compute_calibration_progress(pattern, reviews_completed=0, corrections_recorded=0)
    mature = compute_calibration_progress(pattern, reviews_completed=3000, corrections_recorded=30)

    fresh_gain = expected_information_gain(fresh, corpus_share=0.1)
    mature_gain = expected_information_gain(mature, corpus_share=0.1)
    assert mature_gain < fresh_gain
