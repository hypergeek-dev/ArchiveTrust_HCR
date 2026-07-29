from __future__ import annotations

import pytest

from archivetrust.review.blind_review.adjudication import adjudicate
from archivetrust.review.blind_review.agreement import BenchmarkStatus, compute_agreement
from archivetrust.review.blind_review.assignment import create_blind_review_pair
from archivetrust.review.blind_review.exclusion import exclude_from_benchmark
from archivetrust.review.blind_review.outcome import resolve_benchmark_outcome
from archivetrust.review.blind_review.store import BlindReviewStore
from archivetrust.review.htr_models import ReviewSubmission

GROUND_TRUTH_LINE = "bekiendt. Säger och deth hon Minnes hoon Tuå gånger waritt"
REVIEWER_B_SEVERE = "Han nekade heelt och holdet till alt thet som war honom."


def _store_with_pair(target_ref: str = "ground_truth_annotation_1"):
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


def test_resolve_benchmark_outcome_raises_when_review_not_yet_complete():
    store, _assignment_a, _assignment_b = _store_with_pair()
    with pytest.raises(ValueError):
        resolve_benchmark_outcome(store, "ground_truth_annotation_1", computed_at="2026-01-01T12:00:00Z")


def test_agreed_outcome_is_ready_for_benchmark_using_reviewer_a_value():
    store, assignment_a, assignment_b = _store_with_pair()
    store.submit_submission(
        ReviewSubmission.create(
            assignment_id=assignment_a.assignment_id,
            reviewer_ref="reviewer-alice",
            submitted_value=GROUND_TRUTH_LINE,
            submitted_at="2026-01-01T09:00:00Z",
        )
    )
    store.submit_submission(
        ReviewSubmission.create(
            assignment_id=assignment_b.assignment_id,
            reviewer_ref="reviewer-bob",
            submitted_value=GROUND_TRUTH_LINE,
            submitted_at="2026-01-01T10:00:00Z",
        )
    )
    outcome = resolve_benchmark_outcome(
        store, "ground_truth_annotation_1", computed_at="2026-01-01T11:00:00Z"
    )
    assert outcome.status == BenchmarkStatus.AGREED
    assert outcome.ready_for_benchmark is True
    assert outcome.benchmark_value == GROUND_TRUTH_LINE


def test_requires_adjudication_outcome_becomes_ready_only_after_adjudication():
    store, assignment_a, assignment_b = _store_with_pair()
    submission_a = ReviewSubmission.create(
        assignment_id=assignment_a.assignment_id,
        reviewer_ref="reviewer-alice",
        submitted_value=GROUND_TRUTH_LINE,
        submitted_at="2026-01-01T09:00:00Z",
    )
    store.submit_submission(submission_a)
    store.submit_submission(
        ReviewSubmission.create(
            assignment_id=assignment_b.assignment_id,
            reviewer_ref="reviewer-bob",
            submitted_value=REVIEWER_B_SEVERE,
            submitted_at="2026-01-01T10:00:00Z",
        )
    )

    pending = resolve_benchmark_outcome(
        store, "ground_truth_annotation_1", computed_at="2026-01-01T11:00:00Z"
    )
    assert pending.status == BenchmarkStatus.REQUIRES_ADJUDICATION
    assert pending.ready_for_benchmark is False
    assert pending.benchmark_value is None

    adjudicate(
        store=store,
        assessment=pending.assessment,
        adjudicator_ref="adjudicator-carl",
        resolved_value=submission_a.submitted_value,
        rationale="Reviewer A's reading matches the source image; reviewer B misread the line.",
        adjudicated_at="2026-01-01T12:00:00Z",
    )

    resolved = resolve_benchmark_outcome(
        store, "ground_truth_annotation_1", computed_at="2026-01-01T13:00:00Z"
    )
    # The classification itself is unchanged (Article 15: the original disagreement is a recorded
    # fact, not erased by resolving it) -- but it is now ready, with the adjudicated value.
    assert resolved.status == BenchmarkStatus.REQUIRES_ADJUDICATION
    assert resolved.ready_for_benchmark is True
    assert resolved.benchmark_value == GROUND_TRUTH_LINE
    assert resolved.adjudication.adjudicator_ref == "adjudicator-carl"


def test_exclusion_always_overrides_the_computed_classification():
    store, assignment_a, assignment_b = _store_with_pair()
    store.submit_submission(
        ReviewSubmission.create(
            assignment_id=assignment_a.assignment_id,
            reviewer_ref="reviewer-alice",
            submitted_value=GROUND_TRUTH_LINE,
            submitted_at="2026-01-01T09:00:00Z",
        )
    )
    store.submit_submission(
        ReviewSubmission.create(
            assignment_id=assignment_b.assignment_id,
            reviewer_ref="reviewer-bob",
            submitted_value=GROUND_TRUTH_LINE,
            submitted_at="2026-01-01T10:00:00Z",
        )
    )
    exclude_from_benchmark(
        store=store,
        target_ref="ground_truth_annotation_1",
        reason="Duplicate of another catalogued item, discovered after both reviews completed.",
        excluded_by="curator-1",
        excluded_at="2026-01-01T10:30:00Z",
    )
    outcome = resolve_benchmark_outcome(
        store, "ground_truth_annotation_1", computed_at="2026-01-01T11:00:00Z"
    )
    assert outcome.status == BenchmarkStatus.EXCLUDED_FROM_BENCHMARK
    assert outcome.ready_for_benchmark is False
    assert outcome.benchmark_value is None
    # The underlying agreement computation is still preserved/visible, not discarded.
    assert outcome.assessment is not None
    assert outcome.assessment.status == BenchmarkStatus.AGREED


def test_exclusion_before_both_submissions_finalized_still_resolves():
    store, _assignment_a, _assignment_b = _store_with_pair()
    exclude_from_benchmark(
        store=store,
        target_ref="ground_truth_annotation_1",
        reason="Source archive object was withdrawn from the collection.",
        excluded_by="curator-1",
        excluded_at="2026-01-01T08:00:00Z",
    )
    outcome = resolve_benchmark_outcome(
        store, "ground_truth_annotation_1", computed_at="2026-01-01T09:00:00Z"
    )
    assert outcome.status == BenchmarkStatus.EXCLUDED_FROM_BENCHMARK
    assert outcome.assessment is None
