"""Review center ViewModel: blind isolation, disagreement display, adjudication, exclusion."""

from __future__ import annotations

import pytest

from archivetrust.presentation.htr_review_center_viewmodel import (
    CORRECTION_TIME_GAP,
    ReviewCenterViewModel,
)
from archivetrust.review.blind_review.assignment import create_blind_review_pair
from archivetrust.review.blind_review.store import (
    AlreadyFinalizedError,
    BlindIsolationError,
    BlindReviewStore,
)
from archivetrust.review.htr_models import ReviewSubmission

COMPUTED_AT = "2026-07-29T12:00:00Z"


def _pair(store: BlindReviewStore, target_ref: str):
    a, b = create_blind_review_pair(
        target_ref=target_ref,
        reviewer_a_ref="reviewer_a",
        reviewer_b_ref="reviewer_b",
        convention_id="riksarkivet_diplomatic",
        convention_version=1,
        assigned_at="2026-07-29T10:00:00Z",
    )
    store.register_blind_pair(a, b)
    return a, b


def _submit(vm: ReviewCenterViewModel, assignment, value: str | None, *, illegible=False):
    return vm.submit(
        ReviewSubmission.create(
            assignment_id=assignment.assignment_id,
            reviewer_ref=assignment.reviewer_ref,
            submitted_value=value,
            submitted_at="2026-07-29T11:00:00Z",
            illegible=illegible,
        )
    )


def test_reviewer_queue_shows_only_the_requesting_reviewers_own_assignments() -> None:
    """Blind review means reviewer A must not even see that reviewer B's assignment row exists in
    their own queue."""
    store = BlindReviewStore()
    a, b = _pair(store, "line_1")
    vm = ReviewCenterViewModel(store)

    queue_a = vm.reviewer_queue("reviewer_a")
    assert [row.assignment_id for row in queue_a] == [a.assignment_id]
    assert all(row.role == "reviewer_a" for row in queue_a)

    queue_b = vm.reviewer_queue("reviewer_b")
    assert [row.assignment_id for row in queue_b] == [b.assignment_id]


def test_the_other_reviewers_submission_is_refused_until_you_submit_your_own() -> None:
    """The ViewModel delegates to `BlindReviewStore`'s guard and does not soften it."""
    store = BlindReviewStore()
    a, b = _pair(store, "line_1")
    vm = ReviewCenterViewModel(store)
    _submit(vm, b, "waritt i Stockholm")

    with pytest.raises(BlindIsolationError):
        vm.other_reviewer_submission(target_ref="line_1", requesting_reviewer_ref="reviewer_a")

    _submit(vm, a, "waritt i Stockholm")
    other = vm.other_reviewer_submission(
        target_ref="line_1", requesting_reviewer_ref="reviewer_a"
    )
    assert other.reviewer_ref == "reviewer_b"


def test_a_second_submission_for_the_same_assignment_is_locked_out() -> None:
    store = BlindReviewStore()
    a, _b = _pair(store, "line_1")
    vm = ReviewCenterViewModel(store)
    _submit(vm, a, "first reading")

    with pytest.raises(AlreadyFinalizedError):
        _submit(vm, a, "second reading")


def test_a_target_pending_both_submissions_has_no_state_yet_rather_than_an_empty_one() -> None:
    store = BlindReviewStore()
    a, _b = _pair(store, "line_1")
    vm = ReviewCenterViewModel(store)
    _submit(vm, a, "waritt i Stockholm")

    assert vm.target_state("line_1", computed_at=COMPUTED_AT) is None


def test_identical_readings_resolve_as_agreed_and_ready_for_benchmark() -> None:
    store = BlindReviewStore()
    a, b = _pair(store, "line_1")
    vm = ReviewCenterViewModel(store)
    _submit(vm, a, "waritt i Stockholm")
    _submit(vm, b, "waritt i Stockholm")

    state = vm.target_state("line_1", computed_at=COMPUTED_AT)
    assert state is not None
    assert state.status == "agreed"
    assert state.status_label == "Reviewers agreed"
    assert state.ready_for_benchmark is True
    assert state.benchmark_value == "waritt i Stockholm"
    assert state.disagreements == ()
    assert state.requires_adjudication is False


def test_divergent_readings_expose_the_located_disagreement_spans() -> None:
    """A scalar CER cannot answer "which words differed" -- the spans must survive to the UI."""
    store = BlindReviewStore()
    a, b = _pair(store, "line_1")
    vm = ReviewCenterViewModel(store)
    _submit(vm, a, "waritt i Stockholm den tjugonde")
    _submit(vm, b, "warit uti Goeteborg den tjugonde")

    state = vm.target_state("line_1", computed_at=COMPUTED_AT)
    assert state is not None
    assert state.disagreements
    first = state.disagreements[0]
    assert first.op == "replace"
    assert "waritt" in first.reviewer_a_text
    assert "warit" in first.reviewer_b_text
    assert state.character_error_rate_normalized is not None


def test_an_illegibility_mismatch_requires_adjudication() -> None:
    store = BlindReviewStore()
    a, b = _pair(store, "line_1")
    vm = ReviewCenterViewModel(store)
    _submit(vm, a, None, illegible=True)
    _submit(vm, b, "waritt i Stockholm")

    state = vm.target_state("line_1", computed_at=COMPUTED_AT)
    assert state is not None
    assert state.status == "requires_adjudication"
    assert state.requires_adjudication is True
    assert state.ready_for_benchmark is False


def test_adjudication_resolves_the_target_without_erasing_the_original_classification() -> None:
    """Adjudicating does not retroactively make the readings agree -- the divergence stays a fact."""
    store = BlindReviewStore()
    a, b = _pair(store, "line_1")
    vm = ReviewCenterViewModel(store)
    _submit(vm, a, None, illegible=True)
    _submit(vm, b, "waritt i Stockholm")

    vm.adjudicate_target(
        target_ref="line_1",
        adjudicator_ref="adjudicator_1",
        resolved_value="waritt i Stockholm",
        rationale="The ascender is faint but legible under raking light.",
        adjudicated_at=COMPUTED_AT,
    )

    state = vm.target_state("line_1", computed_at=COMPUTED_AT)
    assert state is not None
    assert state.status == "requires_adjudication"  # unchanged
    assert state.adjudicated is True
    assert state.ready_for_benchmark is True
    assert state.benchmark_value == "waritt i Stockholm"
    assert state.adjudicator_ref == "adjudicator_1"
    assert "raking light" in state.adjudication_rationale


def test_adjudicating_an_agreed_target_is_rejected() -> None:
    store = BlindReviewStore()
    a, b = _pair(store, "line_1")
    vm = ReviewCenterViewModel(store)
    _submit(vm, a, "waritt i Stockholm")
    _submit(vm, b, "waritt i Stockholm")

    with pytest.raises(ValueError):
        vm.adjudicate_target(
            target_ref="line_1",
            adjudicator_ref="adjudicator_1",
            resolved_value="waritt i Stockholm",
            rationale="Looks fine.",
            adjudicated_at=COMPUTED_AT,
        )


def test_an_empty_adjudication_rationale_is_rejected() -> None:
    store = BlindReviewStore()
    a, b = _pair(store, "line_1")
    vm = ReviewCenterViewModel(store)
    _submit(vm, a, None, illegible=True)
    _submit(vm, b, "waritt i Stockholm")

    with pytest.raises(ValueError):
        vm.adjudicate_target(
            target_ref="line_1",
            adjudicator_ref="adjudicator_1",
            resolved_value="waritt i Stockholm",
            rationale="   ",
            adjudicated_at=COMPUTED_AT,
        )


def test_exclusion_requires_a_reason_and_surfaces_it_with_who_excluded_it() -> None:
    store = BlindReviewStore()
    _pair(store, "line_1")
    vm = ReviewCenterViewModel(store)

    with pytest.raises(ValueError):
        vm.exclude_target(
            target_ref="line_1", reason="", excluded_by="researcher_1", excluded_at=COMPUTED_AT
        )

    vm.exclude_target(
        target_ref="line_1",
        reason="The source scan is corrupt below the fold.",
        excluded_by="researcher_1",
        excluded_at=COMPUTED_AT,
    )
    state = vm.target_state("line_1", computed_at=COMPUTED_AT)
    assert state is not None
    assert state.excluded is True
    assert state.status == "excluded_from_benchmark"
    assert state.ready_for_benchmark is False
    assert "corrupt" in state.exclusion_reason
    assert state.excluded_by == "researcher_1"


def test_workload_counts_come_from_the_review_queries_module() -> None:
    from archivetrust.review.blind_review.queries import reviewer_workload

    store = BlindReviewStore()
    a, _b = _pair(store, "line_1")
    _pair(store, "line_2")
    vm = ReviewCenterViewModel(store)
    _submit(vm, a, "waritt i Stockholm")

    row = vm.workload("reviewer_a")
    source = reviewer_workload(store, "reviewer_a")
    assert (row.open_count, row.submitted_count, row.total_count) == (
        source.open_count,
        source.submitted_count,
        source.total_count,
    )
    assert row.open_count == 1
    assert row.submitted_count == 1
    assert vm.open_assignment_count("reviewer_a") == 1

    assert {w.reviewer_ref for w in vm.all_workloads()} == {"reviewer_a", "reviewer_b"}


def test_completion_status_delegates_and_counts_pending_separately() -> None:
    store = BlindReviewStore()
    a1, b1 = _pair(store, "line_1")
    _pair(store, "line_2")
    vm = ReviewCenterViewModel(store)
    _submit(vm, a1, "waritt i Stockholm")
    _submit(vm, b1, "waritt i Stockholm")

    status = vm.completion_status(("line_1", "line_2"), computed_at=COMPUTED_AT)
    assert status.total_targets == 2
    assert status.both_submitted_count == 1
    assert status.pending_count == 1
    assert status.excluded_count == 0


def test_pending_disagreements_lists_only_unadjudicated_ones() -> None:
    store = BlindReviewStore()
    a, b = _pair(store, "line_1")
    a2, b2 = _pair(store, "line_2")
    vm = ReviewCenterViewModel(store)
    _submit(vm, a, None, illegible=True)
    _submit(vm, b, "waritt i Stockholm")
    _submit(vm, a2, "till den 23 Januarii")
    _submit(vm, b2, "till den 23 Januarii")

    unresolved = vm.pending_disagreements(("line_1", "line_2"), computed_at=COMPUTED_AT)
    assert [state.target_ref for state in unresolved] == ["line_1"]

    vm.adjudicate_target(
        target_ref="line_1",
        adjudicator_ref="adjudicator_1",
        resolved_value="waritt i Stockholm",
        rationale="Legible on close inspection.",
        adjudicated_at=COMPUTED_AT,
    )
    assert vm.pending_disagreements(("line_1", "line_2"), computed_at=COMPUTED_AT) == ()


def test_elapsed_is_named_as_latency_and_the_correction_time_gap_is_stated() -> None:
    """`ReviewSubmission` has no started_at, so per-item correction time cannot be computed. The
    surface reports wall-clock latency under that name and names the missing field."""
    store = BlindReviewStore()
    a, _b = _pair(store, "line_1")
    vm = ReviewCenterViewModel(store)

    before = vm.reviewer_queue("reviewer_a")[0]
    assert before.elapsed_since_assignment is None
    assert before.submitted_at is None

    _submit(vm, a, "waritt i Stockholm")
    after = vm.reviewer_queue("reviewer_a")[0]
    assert after.status == "submitted"
    assert after.elapsed_since_assignment == "2026-07-29T10:00:00Z -> 2026-07-29T11:00:00Z"

    assert "started_at" in CORRECTION_TIME_GAP
    assert "correction time is not tracked" in CORRECTION_TIME_GAP
    from archivetrust.review.htr_models import ReviewSubmission as Submission

    assert "started_at" not in Submission.model_fields  # the gap is real, not hypothetical
