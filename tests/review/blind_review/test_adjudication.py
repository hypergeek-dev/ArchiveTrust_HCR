"""Adjudication creation (task brief item 5): adjudicator identity, adjudicated final result,
required non-empty reasoning; reviewer A's, reviewer B's, and the adjudicated result all preserved
separately."""

from __future__ import annotations

import pytest

from archivetrust.review.blind_review.adjudication import adjudicate
from archivetrust.review.blind_review.agreement import BenchmarkStatus, compute_agreement
from archivetrust.review.blind_review.store import BlindReviewStore
from archivetrust.review.htr_models import ReviewSubmission

GROUND_TRUTH_LINE = "bekiendt. Säger och deth hon Minnes hoon Tuå gånger waritt"
REVIEWER_B_SEVERE = "Han nekade heelt och holdet till alt thet som war honom."


def _requires_adjudication_assessment():
    submission_a = ReviewSubmission.create(
        assignment_id="review_assignment_a",
        reviewer_ref="reviewer-alice",
        submitted_value=GROUND_TRUTH_LINE,
        submitted_at="2026-01-01T09:00:00Z",
    )
    submission_b = ReviewSubmission.create(
        assignment_id="review_assignment_b",
        reviewer_ref="reviewer-bob",
        submitted_value=REVIEWER_B_SEVERE,
        submitted_at="2026-01-01T10:00:00Z",
    )
    assessment = compute_agreement(
        target_ref="ground_truth_annotation_1",
        submission_a=submission_a,
        submission_b=submission_b,
        computed_at="2026-01-01T11:00:00Z",
    )
    assert assessment.status == BenchmarkStatus.REQUIRES_ADJUDICATION
    return submission_a, submission_b, assessment


def test_adjudication_resolves_and_records_adjudicator_identity_and_result():
    store = BlindReviewStore()
    submission_a, submission_b, assessment = _requires_adjudication_assessment()

    adjudication = adjudicate(
        store=store,
        assessment=assessment,
        adjudicator_ref="adjudicator-carl",
        resolved_value=submission_a.submitted_value,
        rationale=(
            "Reviewer B appears to have transcribed a different line entirely; reviewer A's "
            "reading matches the visible ink strokes in the source image."
        ),
        adjudicated_at="2026-01-01T12:00:00Z",
    )

    assert adjudication.adjudicator_ref == "adjudicator-carl"
    assert adjudication.resolved_value == submission_a.submitted_value
    assert adjudication.rationale.startswith("Reviewer B appears")
    assert store.adjudication_for(assessment.agreement_result.agreement_result_id) is adjudication

    # All three results remain independently readable -- nothing was overwritten.
    assert submission_a.submitted_value == GROUND_TRUTH_LINE
    assert submission_b.submitted_value == REVIEWER_B_SEVERE
    assert adjudication.resolved_value == GROUND_TRUTH_LINE
    assert assessment.agreement_result.submission_a_id == submission_a.submission_id
    assert assessment.agreement_result.submission_b_id == submission_b.submission_id


def test_adjudication_rejects_empty_rationale():
    store = BlindReviewStore()
    _submission_a, _submission_b, assessment = _requires_adjudication_assessment()
    with pytest.raises(ValueError, match="rationale"):
        adjudicate(
            store=store,
            assessment=assessment,
            adjudicator_ref="adjudicator-carl",
            resolved_value=GROUND_TRUTH_LINE,
            rationale="   ",
            adjudicated_at="2026-01-01T12:00:00Z",
        )


def test_adjudication_rejects_missing_rationale_argument_type():
    store = BlindReviewStore()
    _submission_a, _submission_b, assessment = _requires_adjudication_assessment()
    with pytest.raises(ValueError, match="rationale"):
        adjudicate(
            store=store,
            assessment=assessment,
            adjudicator_ref="adjudicator-carl",
            resolved_value=GROUND_TRUTH_LINE,
            rationale="",
            adjudicated_at="2026-01-01T12:00:00Z",
        )


def test_adjudication_rejects_items_that_do_not_require_it():
    store = BlindReviewStore()
    submission_a = ReviewSubmission.create(
        assignment_id="review_assignment_a",
        reviewer_ref="reviewer-alice",
        submitted_value=GROUND_TRUTH_LINE,
        submitted_at="2026-01-01T09:00:00Z",
    )
    submission_b = ReviewSubmission.create(
        assignment_id="review_assignment_b",
        reviewer_ref="reviewer-bob",
        submitted_value=GROUND_TRUTH_LINE,
        submitted_at="2026-01-01T10:00:00Z",
    )
    agreed_assessment = compute_agreement(
        target_ref="ground_truth_annotation_1",
        submission_a=submission_a,
        submission_b=submission_b,
        computed_at="2026-01-01T11:00:00Z",
    )
    assert agreed_assessment.status == BenchmarkStatus.AGREED
    with pytest.raises(ValueError, match="not 'requires_adjudication'"):
        adjudicate(
            store=store,
            assessment=agreed_assessment,
            adjudicator_ref="adjudicator-carl",
            resolved_value=GROUND_TRUTH_LINE,
            rationale="Not actually needed.",
            adjudicated_at="2026-01-01T12:00:00Z",
        )
