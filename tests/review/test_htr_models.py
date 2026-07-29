from __future__ import annotations

import pytest
from pydantic import ValidationError

from archivetrust.review.htr_models import (
    Adjudication,
    AgreementResult,
    ReviewAssignment,
    ReviewAssignmentRole,
    ReviewSubmission,
)


def test_review_assignment_round_trips():
    assignment = ReviewAssignment.create(
        target_ref="ground_truth_annotation_1",
        role=ReviewAssignmentRole.REVIEWER_A,
        reviewer_ref="reviewer-1",
        convention_id="convention_1",
        convention_version=1,
        assigned_at="2026-01-01T00:00:00Z",
    )
    restored = ReviewAssignment.model_validate(assignment.model_dump())
    assert restored == assignment


def test_review_submission_illegible_requires_no_value():
    with pytest.raises(ValidationError):
        ReviewSubmission(
            submission_id="review_submission_x",
            assignment_id="review_assignment_1",
            reviewer_ref="reviewer-1",
            submitted_value="not none",
            illegible=True,
            submitted_at="2026-01-01T00:00:00Z",
        )


def test_review_submission_non_illegible_requires_value():
    with pytest.raises(ValidationError):
        ReviewSubmission(
            submission_id="review_submission_x",
            assignment_id="review_assignment_1",
            reviewer_ref="reviewer-1",
            submitted_value=None,
            illegible=False,
            submitted_at="2026-01-01T00:00:00Z",
        )


def test_review_submission_valid_construction():
    submission = ReviewSubmission.create(
        assignment_id="review_assignment_1",
        reviewer_ref="reviewer-1",
        submitted_value="hej",
        submitted_at="2026-01-01T00:00:00Z",
    )
    assert submission.submitted_value == "hej"


def test_agreement_result_rejects_out_of_range_similarity():
    with pytest.raises(ValidationError):
        AgreementResult(
            agreement_result_id="agreement_result_x",
            target_ref="ground_truth_annotation_1",
            submission_a_id="review_submission_a",
            submission_b_id="review_submission_b",
            agrees=False,
            similarity_score=1.5,
            computed_at="2026-01-01T00:00:00Z",
        )


def test_adjudication_round_trips():
    adjudication = Adjudication.create(
        agreement_result_id="agreement_result_1",
        adjudicator_ref="adjudicator-1",
        resolved_value="hej",
        rationale="reviewer B misread the abbreviation",
        adjudicated_at="2026-01-01T00:00:00Z",
    )
    restored = Adjudication.model_validate(adjudication.model_dump())
    assert restored == adjudication
