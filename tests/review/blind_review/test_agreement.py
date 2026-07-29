"""Agreement classification thresholds (task brief item 4), using real Swedish 17th-century
ground-truth text (`tests/fixtures/htr/trolldomskommissionen_sample_line.txt`) with constructed
reviewer transcriptions for the Agreed/Minor/Material/Requires-adjudication cases -- there is no
real second reviewer in this repo, so these are constructed-for-testing reviewer inputs (labeled as
such here), built on the real ground-truth line.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from archivetrust.review.blind_review.agreement import (
    AgreementPolicy,
    BenchmarkStatus,
    compute_agreement,
)
from archivetrust.review.htr_models import ReviewSubmission

FIXTURE_PATH = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "htr"
    / "trolldomskommissionen_sample_line.txt"
)
GROUND_TRUTH_LINE = FIXTURE_PATH.read_text(encoding="utf-8").strip()

# Constructed-for-testing reviewer transcriptions of the real ground-truth line above -- not real
# second-reviewer data (this repo has no real second reviewer). Each is a plausible, hand-plausible
# reading a careful transcriber of 17th-century Swedish court-record handwriting could produce.
REVIEWER_A_EXACT = GROUND_TRUTH_LINE
REVIEWER_B_EXACT_MATCH = GROUND_TRUTH_LINE
REVIEWER_B_MINOR = GROUND_TRUTH_LINE.replace("waritt", "warit")  # one dropped doubled consonant
REVIEWER_B_MATERIAL = (
    GROUND_TRUTH_LINE.replace("gånger", "ganger").replace("waritt", "warit")
)  # a dropped diacritic plus the same doubled-consonant slip
REVIEWER_B_SEVERE = "Han nekade heelt och holdet till alt thet som war honom."  # a different line


def _submission_pair(value_a: str, value_b: str) -> tuple[ReviewSubmission, ReviewSubmission]:
    submission_a = ReviewSubmission.create(
        assignment_id="review_assignment_a",
        reviewer_ref="reviewer-alice",
        submitted_value=value_a,
        submitted_at="2026-01-01T09:00:00Z",
    )
    submission_b = ReviewSubmission.create(
        assignment_id="review_assignment_b",
        reviewer_ref="reviewer-bob",
        submitted_value=value_b,
        submitted_at="2026-01-01T10:00:00Z",
    )
    return submission_a, submission_b


def test_identical_transcriptions_are_classified_agreed():
    submission_a, submission_b = _submission_pair(REVIEWER_A_EXACT, REVIEWER_B_EXACT_MATCH)
    assessment = compute_agreement(
        target_ref="ground_truth_annotation_1",
        submission_a=submission_a,
        submission_b=submission_b,
        computed_at="2026-01-01T11:00:00Z",
    )
    assert assessment.status == BenchmarkStatus.AGREED
    assert assessment.agreement_result.agrees is True
    assert assessment.recognition_metrics.character_error_rate_normalized == 0.0
    assert assessment.disagreement_locations == ()


def test_small_realistic_difference_is_classified_minor_disagreement():
    submission_a, submission_b = _submission_pair(REVIEWER_A_EXACT, REVIEWER_B_MINOR)
    assessment = compute_agreement(
        target_ref="ground_truth_annotation_1",
        submission_a=submission_a,
        submission_b=submission_b,
        computed_at="2026-01-01T11:00:00Z",
    )
    cer = assessment.recognition_metrics.character_error_rate_normalized
    assert 0.0 < cer <= 0.02
    assert assessment.status == BenchmarkStatus.MINOR_DISAGREEMENT
    assert assessment.agreement_result.agrees is True  # minor disagreement is still "agrees"
    assert len(assessment.disagreement_locations) >= 1


def test_substantial_difference_is_classified_material_disagreement():
    submission_a, submission_b = _submission_pair(REVIEWER_A_EXACT, REVIEWER_B_MATERIAL)
    assessment = compute_agreement(
        target_ref="ground_truth_annotation_1",
        submission_a=submission_a,
        submission_b=submission_b,
        computed_at="2026-01-01T11:00:00Z",
    )
    cer = assessment.recognition_metrics.character_error_rate_normalized
    assert 0.02 < cer <= 0.15
    assert assessment.status == BenchmarkStatus.MATERIAL_DISAGREEMENT
    assert assessment.agreement_result.agrees is False
    assert len(assessment.disagreement_locations) >= 1


def test_severe_difference_requires_adjudication():
    submission_a, submission_b = _submission_pair(REVIEWER_A_EXACT, REVIEWER_B_SEVERE)
    assessment = compute_agreement(
        target_ref="ground_truth_annotation_1",
        submission_a=submission_a,
        submission_b=submission_b,
        computed_at="2026-01-01T11:00:00Z",
    )
    cer = assessment.recognition_metrics.character_error_rate_normalized
    assert cer > 0.15
    assert assessment.status == BenchmarkStatus.REQUIRES_ADJUDICATION
    assert assessment.agreement_result.agrees is False


def test_illegibility_mismatch_always_requires_adjudication():
    submission_a = ReviewSubmission.create(
        assignment_id="review_assignment_a",
        reviewer_ref="reviewer-alice",
        submitted_value=GROUND_TRUTH_LINE,
        submitted_at="2026-01-01T09:00:00Z",
    )
    submission_b = ReviewSubmission.create(
        assignment_id="review_assignment_b",
        reviewer_ref="reviewer-bob",
        submitted_value=None,
        illegible=True,
        submitted_at="2026-01-01T10:00:00Z",
    )
    assessment = compute_agreement(
        target_ref="ground_truth_annotation_1",
        submission_a=submission_a,
        submission_b=submission_b,
        computed_at="2026-01-01T11:00:00Z",
    )
    assert assessment.status == BenchmarkStatus.REQUIRES_ADJUDICATION
    assert assessment.recognition_metrics is None


def test_both_illegible_is_agreed():
    submission_a = ReviewSubmission.create(
        assignment_id="review_assignment_a",
        reviewer_ref="reviewer-alice",
        submitted_value=None,
        illegible=True,
        submitted_at="2026-01-01T09:00:00Z",
    )
    submission_b = ReviewSubmission.create(
        assignment_id="review_assignment_b",
        reviewer_ref="reviewer-bob",
        submitted_value=None,
        illegible=True,
        submitted_at="2026-01-01T10:00:00Z",
    )
    assessment = compute_agreement(
        target_ref="ground_truth_annotation_1",
        submission_a=submission_a,
        submission_b=submission_b,
        computed_at="2026-01-01T11:00:00Z",
    )
    assert assessment.status == BenchmarkStatus.AGREED
    assert assessment.agreement_result.agrees is True


def test_disagreement_locations_identify_the_actual_differing_words():
    """Disagreement locations are real spans, not a placeholder -- for the material-disagreement
    fixture, the located spans must contain the actual words that differ."""
    submission_a, submission_b = _submission_pair(REVIEWER_A_EXACT, REVIEWER_B_MATERIAL)
    assessment = compute_agreement(
        target_ref="ground_truth_annotation_1",
        submission_a=submission_a,
        submission_b=submission_b,
        computed_at="2026-01-01T11:00:00Z",
    )
    a_texts = {span.reviewer_a_text for span in assessment.disagreement_locations}
    b_texts = {span.reviewer_b_text for span in assessment.disagreement_locations}
    assert any("gånger" in text or "waritt" in text for text in a_texts)
    assert any("ganger" in text or "warit" in text for text in b_texts)


def test_agreement_policy_rejects_inverted_thresholds():
    with pytest.raises(ValueError):
        AgreementPolicy(minor_disagreement_cer_max=0.5, material_disagreement_cer_max=0.1)


def test_compute_agreement_rejects_two_submissions_from_the_same_assignment():
    submission_a, _ = _submission_pair(REVIEWER_A_EXACT, REVIEWER_B_MINOR)
    with pytest.raises(ValueError):
        compute_agreement(
            target_ref="ground_truth_annotation_1",
            submission_a=submission_a,
            submission_b=submission_a,
            computed_at="2026-01-01T11:00:00Z",
        )
