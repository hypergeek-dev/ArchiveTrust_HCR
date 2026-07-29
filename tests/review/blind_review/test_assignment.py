from __future__ import annotations

import pytest

from archivetrust.review.blind_review.assignment import create_blind_review_pair
from archivetrust.review.htr_models import ReviewAssignmentRole, ReviewAssignmentStatus


def test_create_blind_review_pair_produces_two_independent_open_assignments():
    assignment_a, assignment_b = create_blind_review_pair(
        target_ref="ground_truth_annotation_1",
        reviewer_a_ref="reviewer-alice",
        reviewer_b_ref="reviewer-bob",
        convention_id="convention_1",
        convention_version=1,
        assigned_at="2026-01-01T00:00:00Z",
    )
    assert assignment_a.role == ReviewAssignmentRole.REVIEWER_A
    assert assignment_b.role == ReviewAssignmentRole.REVIEWER_B
    assert assignment_a.reviewer_ref == "reviewer-alice"
    assert assignment_b.reviewer_ref == "reviewer-bob"
    assert assignment_a.target_ref == assignment_b.target_ref == "ground_truth_annotation_1"
    assert assignment_a.assignment_id != assignment_b.assignment_id
    assert assignment_a.status == assignment_b.status == ReviewAssignmentStatus.OPEN


def test_create_blind_review_pair_rejects_same_reviewer_for_both_roles():
    with pytest.raises(ValueError, match="two distinct reviewers"):
        create_blind_review_pair(
            target_ref="ground_truth_annotation_1",
            reviewer_a_ref="reviewer-alice",
            reviewer_b_ref="reviewer-alice",
            convention_id="convention_1",
            convention_version=1,
            assigned_at="2026-01-01T00:00:00Z",
        )
