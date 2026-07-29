from __future__ import annotations

from archivetrust.domain.calibration.coverage import BlindSpotReason, find_blind_spots
from archivetrust.domain.calibration.pattern import CalibrationPattern
from archivetrust.domain.calibration.progress import compute_calibration_progress


def test_never_queued_pattern_flagged_as_blind_spot():
    pattern = CalibrationPattern(name="uncorroborated_single_source/paragraph", population=1000, queued_for_review=0)
    progress = {pattern.name: compute_calibration_progress(pattern, reviews_completed=0, corrections_recorded=0)}

    blind_spots = find_blind_spots((pattern,), progress, total_population=1000)

    assert len(blind_spots) == 1
    assert blind_spots[0].reason == BlindSpotReason.NEVER_QUEUED_OPERATIONALLY
    assert blind_spots[0].recommended_sampling_strategy == "stratified_sampling"


def test_queued_but_uncalibrated_pattern_flagged_differently():
    pattern = CalibrationPattern(name="contested/paragraph", population=500, queued_for_review=500)
    progress = {pattern.name: compute_calibration_progress(pattern, reviews_completed=0, corrections_recorded=0)}

    blind_spots = find_blind_spots((pattern,), progress, total_population=500)

    assert blind_spots[0].reason == BlindSpotReason.NO_CALIBRATION_EVIDENCE
    assert blind_spots[0].recommended_sampling_strategy == "random_sampling"


def test_pattern_with_evidence_is_not_a_blind_spot():
    pattern = CalibrationPattern(name="p", population=500, queued_for_review=500)
    progress = {pattern.name: compute_calibration_progress(pattern, reviews_completed=200, corrections_recorded=5)}

    assert find_blind_spots((pattern,), progress, total_population=500) == ()


def test_blind_spots_sorted_by_corpus_share_descending():
    big = CalibrationPattern(name="big", population=900, queued_for_review=0)
    small = CalibrationPattern(name="small", population=100, queued_for_review=0)
    progress = {
        p.name: compute_calibration_progress(p, reviews_completed=0, corrections_recorded=0)
        for p in (small, big)
    }

    blind_spots = find_blind_spots((small, big), progress, total_population=1000)

    assert [b.pattern for b in blind_spots] == ["big", "small"]
