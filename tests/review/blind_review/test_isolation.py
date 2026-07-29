"""Proves blind isolation is mechanically enforced (task brief item 2), not merely a convention."""

from __future__ import annotations

import pytest

from archivetrust.review.blind_review.assignment import create_blind_review_pair
from archivetrust.review.blind_review.store import BlindIsolationError, BlindReviewStore
from archivetrust.review.htr_models import ReviewSubmission


def _paired_store(target_ref: str = "ground_truth_annotation_1") -> tuple[BlindReviewStore, tuple]:
    store = BlindReviewStore()
    pair = create_blind_review_pair(
        target_ref=target_ref,
        reviewer_a_ref="reviewer-alice",
        reviewer_b_ref="reviewer-bob",
        convention_id="convention_1",
        convention_version=1,
        assigned_at="2026-01-01T00:00:00Z",
    )
    store.register_blind_pair(*pair)
    return store, pair


def test_reviewer_b_cannot_see_reviewer_a_before_b_submits():
    """The core proof: A submits first. B attempts to read A's result before submitting their own
    -- this must be mechanically blocked (raises BlindIsolationError), not merely documented as
    forbidden. Once B submits, both results become visible to both reviewers."""
    store, (assignment_a, assignment_b) = _paired_store()

    submission_a = ReviewSubmission.create(
        assignment_id=assignment_a.assignment_id,
        reviewer_ref="reviewer-alice",
        submitted_value="bekiendt. Säger och deth hon Minnes hoon Tuå gånger waritt",
        submitted_at="2026-01-01T09:00:00Z",
    )
    store.submit_submission(submission_a)

    # B has NOT submitted yet -- B attempting to peek at A's submission must be refused.
    with pytest.raises(BlindIsolationError):
        store.submission_for_other_reviewer(
            target_ref="ground_truth_annotation_1", requesting_reviewer_ref="reviewer-bob"
        )

    # A, having already submitted, attempting to look at B's (not-yet-existing) submission is
    # refused for the same reason -- B hasn't submitted, regardless of A's own status.
    with pytest.raises(BlindIsolationError):
        store.submission_for_other_reviewer(
            target_ref="ground_truth_annotation_1", requesting_reviewer_ref="reviewer-alice"
        )

    # A partial pair must never be returned by the listing-side query either.
    assert store.both_submissions_if_complete("ground_truth_annotation_1") is None

    # Now B submits.
    submission_b = ReviewSubmission.create(
        assignment_id=assignment_b.assignment_id,
        reviewer_ref="reviewer-bob",
        submitted_value="bekiendt. Säger och deth hon Minnes hoon Tuå gånger waritt",
        submitted_at="2026-01-01T10:00:00Z",
    )
    store.submit_submission(submission_b)

    # Both are now finalized: each reviewer can see the OTHER's submission.
    seen_by_bob = store.submission_for_other_reviewer(
        target_ref="ground_truth_annotation_1", requesting_reviewer_ref="reviewer-bob"
    )
    assert seen_by_bob.submission_id == submission_a.submission_id

    seen_by_alice = store.submission_for_other_reviewer(
        target_ref="ground_truth_annotation_1", requesting_reviewer_ref="reviewer-alice"
    )
    assert seen_by_alice.submission_id == submission_b.submission_id

    # And the listing-side query now returns the complete pair.
    complete = store.both_submissions_if_complete("ground_truth_annotation_1")
    assert complete is not None
    assert {s.submission_id for s in complete} == {
        submission_a.submission_id,
        submission_b.submission_id,
    }


def test_reviewer_a_cannot_see_reviewer_b_before_a_submits():
    """The symmetric case: B submits first, A has not -- A must be blocked from reading B's
    submission."""
    store, (assignment_a, assignment_b) = _paired_store()

    submission_b = ReviewSubmission.create(
        assignment_id=assignment_b.assignment_id,
        reviewer_ref="reviewer-bob",
        submitted_value="bekiendt. Säger och deth hon Minnes hoon Tuå gånger waritt",
        submitted_at="2026-01-01T09:00:00Z",
    )
    store.submit_submission(submission_b)

    with pytest.raises(BlindIsolationError):
        store.submission_for_other_reviewer(
            target_ref="ground_truth_annotation_1", requesting_reviewer_ref="reviewer-alice"
        )


def test_unassigned_reviewer_cannot_query_isolation_state():
    store, _ = _paired_store()
    with pytest.raises(KeyError):
        store.submission_for_other_reviewer(
            target_ref="ground_truth_annotation_1", requesting_reviewer_ref="reviewer-mallory"
        )


def test_own_submission_is_always_readable_regardless_of_isolation():
    store, (assignment_a, _assignment_b) = _paired_store()
    submission_a = ReviewSubmission.create(
        assignment_id=assignment_a.assignment_id,
        reviewer_ref="reviewer-alice",
        submitted_value="bekiendt. Säger och deth hon Minnes hoon Tuå gånger waritt",
        submitted_at="2026-01-01T09:00:00Z",
    )
    store.submit_submission(submission_a)
    assert store.own_submission(assignment_a.assignment_id).submission_id == submission_a.submission_id
