"""Proves a finalized ReviewSubmission cannot be mutated/replaced (task brief item 3), and that the
same locking discipline applies to Adjudication and ExclusionRecord recording."""

from __future__ import annotations

import pytest

from archivetrust.review.blind_review.assignment import create_blind_review_pair
from archivetrust.review.blind_review.exclusion import exclude_from_benchmark
from archivetrust.review.blind_review.store import AlreadyFinalizedError, BlindReviewStore
from archivetrust.review.htr_models import Adjudication, ReviewSubmission


def _paired_store(target_ref: str = "ground_truth_annotation_1"):
    store = BlindReviewStore()
    assignment_a, assignment_b = create_blind_review_pair(
        target_ref=target_ref,
        reviewer_a_ref="reviewer-alice",
        reviewer_b_ref="reviewer-bob",
        convention_id="convention_1",
        convention_version=1,
        assigned_at="2026-01-01T00:00:00Z",
    )
    store.register_blind_pair(assignment_a, assignment_b)
    return store, assignment_a, assignment_b


def test_review_submission_model_itself_is_frozen():
    """The Pydantic-level half of immutability: no field on a constructed ReviewSubmission can be
    reassigned in place -- matches how Evidence/CanonicalObservation are frozen."""
    submission = ReviewSubmission.create(
        assignment_id="review_assignment_1",
        reviewer_ref="reviewer-alice",
        submitted_value="hej",
        submitted_at="2026-01-01T00:00:00Z",
    )
    with pytest.raises(Exception):
        submission.submitted_value = "changed"  # type: ignore[misc]


def test_submitting_twice_for_the_same_assignment_is_rejected():
    """The store-level half: a second, DIFFERENT submission for an assignment that already has a
    finalized one is rejected outright, never silently overwritten."""
    store, assignment_a, _assignment_b = _paired_store()

    first = ReviewSubmission.create(
        assignment_id=assignment_a.assignment_id,
        reviewer_ref="reviewer-alice",
        submitted_value="bekiendt. Säger och deth hon Minnes hoon Tuå gånger waritt",
        submitted_at="2026-01-01T09:00:00Z",
    )
    store.submit_submission(first)

    attempted_correction = ReviewSubmission.create(
        assignment_id=assignment_a.assignment_id,
        reviewer_ref="reviewer-alice",
        submitted_value="a completely different reading",
        submitted_at="2026-01-01T09:05:00Z",
    )
    with pytest.raises(AlreadyFinalizedError):
        store.submit_submission(attempted_correction)

    # The original submission is untouched.
    assert store.own_submission(assignment_a.assignment_id).submitted_value == (
        "bekiendt. Säger och deth hon Minnes hoon Tuå gånger waritt"
    )


def test_submitting_identical_content_twice_is_still_rejected():
    """Locking applies even when the second attempt's content is byte-identical -- the rule is
    "one submission per assignment, ever," not "no conflicting submissions."""
    store, assignment_a, _assignment_b = _paired_store()
    value = "bekiendt. Säger och deth hon Minnes hoon Tuå gånger waritt"
    first = ReviewSubmission.create(
        assignment_id=assignment_a.assignment_id,
        reviewer_ref="reviewer-alice",
        submitted_value=value,
        submitted_at="2026-01-01T09:00:00Z",
    )
    store.submit_submission(first)
    duplicate = ReviewSubmission.create(
        assignment_id=assignment_a.assignment_id,
        reviewer_ref="reviewer-alice",
        submitted_value=value,
        submitted_at="2026-01-01T09:01:00Z",
    )
    with pytest.raises(AlreadyFinalizedError):
        store.submit_submission(duplicate)


def test_recording_a_second_adjudication_for_the_same_agreement_result_is_rejected():
    store, _assignment_a, _assignment_b = _paired_store()
    first = Adjudication.create(
        agreement_result_id="agreement_result_1",
        adjudicator_ref="adjudicator-1",
        resolved_value="hej",
        rationale="Reviewer B misread the abbreviation as a different word.",
        adjudicated_at="2026-01-01T12:00:00Z",
    )
    store.record_adjudication(first)
    second = Adjudication.create(
        agreement_result_id="agreement_result_1",
        adjudicator_ref="adjudicator-2",
        resolved_value="something else",
        rationale="Overruling the first adjudicator.",
        adjudicated_at="2026-01-01T13:00:00Z",
    )
    with pytest.raises(AlreadyFinalizedError):
        store.record_adjudication(second)
    # The original adjudication is untouched.
    assert store.adjudication_for("agreement_result_1").adjudicator_ref == "adjudicator-1"


def test_excluding_the_same_target_twice_is_rejected():
    store, _assignment_a, _assignment_b = _paired_store()
    exclude_from_benchmark(
        store=store,
        target_ref="ground_truth_annotation_1",
        reason="Source image is illegible due to water damage.",
        excluded_by="curator-1",
        excluded_at="2026-01-01T08:00:00Z",
    )
    with pytest.raises(ValueError):
        exclude_from_benchmark(
            store=store,
            target_ref="ground_truth_annotation_1",
            reason="A different, later reason.",
            excluded_by="curator-2",
            excluded_at="2026-01-01T09:00:00Z",
        )
