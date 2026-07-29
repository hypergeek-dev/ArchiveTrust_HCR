"""Blind dual-review assignment (`docs/htr-domain-design.md` §1: "GroundTruthItem ->
ReviewAssignment (reviewer A, reviewer B -- blind)").

Creates the two independent `ReviewAssignment`s a target needs. "Independent" here means: two
distinct reviewer identities, two distinct `assignment_id`s, no shared mutable state -- the actual
blind *isolation* (reviewer A cannot see reviewer B's submission) is enforced by `store.py`, not by
anything representable on `ReviewAssignment` itself (per that model's own docstring).
"""

from __future__ import annotations

from archivetrust.review.htr_models import ReviewAssignment, ReviewAssignmentRole


def create_blind_review_pair(
    *,
    target_ref: str,
    reviewer_a_ref: str,
    reviewer_b_ref: str,
    convention_id: str,
    convention_version: int,
    assigned_at: str,
) -> tuple[ReviewAssignment, ReviewAssignment]:
    """Creates two independent `ReviewAssignment`s (reviewer A, reviewer B) for the same
    `target_ref` (a `GroundTruthAnnotation.annotation_id` or a not-yet-annotated target key, per
    `ReviewAssignment.target_ref`'s own docstring).

    Rejects assigning the same reviewer to both roles -- a "blind dual review" by one person is not
    a blind dual review at all, and this is the one invariant expressible at assignment time, before
    either reviewer has touched the target.
    """
    if reviewer_a_ref == reviewer_b_ref:
        raise ValueError(
            "Blind dual review requires two distinct reviewers "
            f"(reviewer_a_ref == reviewer_b_ref == {reviewer_a_ref!r})"
        )
    assignment_a = ReviewAssignment.create(
        target_ref=target_ref,
        role=ReviewAssignmentRole.REVIEWER_A,
        reviewer_ref=reviewer_a_ref,
        convention_id=convention_id,
        convention_version=convention_version,
        assigned_at=assigned_at,
    )
    assignment_b = ReviewAssignment.create(
        target_ref=target_ref,
        role=ReviewAssignmentRole.REVIEWER_B,
        reviewer_ref=reviewer_b_ref,
        convention_id=convention_id,
        convention_version=convention_version,
        assigned_at=assigned_at,
    )
    return assignment_a, assignment_b
