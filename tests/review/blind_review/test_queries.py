"""Reviewer workload / review completion status queries, and the task brief's item 7 requirement:
Minor disagreement and Material disagreement items are NEVER silently dropped from a listing --
only an explicit ExclusionRecord removes a target from benchmark readiness, and even then it stays
visible."""

from __future__ import annotations

from archivetrust.review.blind_review.agreement import BenchmarkStatus
from archivetrust.review.blind_review.assignment import create_blind_review_pair
from archivetrust.review.blind_review.exclusion import exclude_from_benchmark
from archivetrust.review.blind_review.queries import batch_completion_status, reviewer_workload
from archivetrust.review.blind_review.store import BlindReviewStore
from archivetrust.review.htr_models import ReviewSubmission

GROUND_TRUTH_LINE = "bekiendt. Säger och deth hon Minnes hoon Tuå gånger waritt"
MINOR_VARIANT = GROUND_TRUTH_LINE.replace("waritt", "warit")
MATERIAL_VARIANT = GROUND_TRUTH_LINE.replace("gånger", "ganger").replace("waritt", "warit")
SEVERE_VARIANT = "Han nekade heelt och holdet till alt thet som war honom."


def _assign_and_submit(
    store: BlindReviewStore, *, target_ref: str, value_a: str | None, value_b: str | None
):
    assignment_a, assignment_b = create_blind_review_pair(
        target_ref=target_ref,
        reviewer_a_ref="reviewer-alice",
        reviewer_b_ref="reviewer-bob",
        convention_id="convention_1",
        convention_version=1,
        assigned_at="2026-01-01T00:00:00Z",
    )
    store.register_blind_pair(assignment_a, assignment_b)
    if value_a is not None:
        store.submit_submission(
            ReviewSubmission.create(
                assignment_id=assignment_a.assignment_id,
                reviewer_ref="reviewer-alice",
                submitted_value=value_a,
                submitted_at="2026-01-01T09:00:00Z",
            )
        )
    if value_b is not None:
        store.submit_submission(
            ReviewSubmission.create(
                assignment_id=assignment_b.assignment_id,
                reviewer_ref="reviewer-bob",
                submitted_value=value_b,
                submitted_at="2026-01-01T10:00:00Z",
            )
        )
    return assignment_a, assignment_b


def test_reviewer_workload_counts_open_and_submitted_assignments():
    store = BlindReviewStore()
    # Two targets: alice is reviewer_a on both, so she has two assignments.
    _assign_and_submit(
        store, target_ref="item_1", value_a=GROUND_TRUTH_LINE, value_b=GROUND_TRUTH_LINE
    )
    _assign_and_submit(store, target_ref="item_2", value_a=None, value_b=None)

    workload = reviewer_workload(store, "reviewer-alice")
    assert workload.total_count == 2
    assert workload.submitted_count == 1
    assert workload.open_count == 1
    assert workload.withdrawn_count == 0


def test_reviewer_workload_for_unknown_reviewer_is_all_zero():
    store = BlindReviewStore()
    _assign_and_submit(store, target_ref="item_1", value_a=None, value_b=None)
    workload = reviewer_workload(store, "reviewer-nobody")
    assert workload.total_count == 0


def test_batch_completion_status_counts_both_submitted_adjudicated_and_excluded():
    store = BlindReviewStore()
    _assign_and_submit(
        store, target_ref="item_agreed", value_a=GROUND_TRUTH_LINE, value_b=GROUND_TRUTH_LINE
    )
    _assign_and_submit(
        store, target_ref="item_pending", value_a=GROUND_TRUTH_LINE, value_b=None
    )
    exclude_from_benchmark(
        store=store,
        target_ref="item_excluded",
        reason="Discovered to be a duplicate scan of item_agreed.",
        excluded_by="curator-1",
        excluded_at="2026-01-01T08:00:00Z",
    )

    status = batch_completion_status(
        store,
        ["item_agreed", "item_pending", "item_excluded"],
        computed_at="2026-01-01T12:00:00Z",
    )
    assert status.total_targets == 3
    assert status.both_submitted_count == 1
    assert status.pending_count == 1
    assert status.excluded_count == 1
    assert status.by_status[BenchmarkStatus.AGREED] == 1
    assert status.by_status[BenchmarkStatus.EXCLUDED_FROM_BENCHMARK] == 1
    assert status.target_statuses["item_agreed"] == BenchmarkStatus.AGREED
    assert status.target_statuses["item_excluded"] == BenchmarkStatus.EXCLUDED_FROM_BENCHMARK
    assert "item_pending" not in status.target_statuses  # genuinely pending, not a dropped status


def test_minor_and_material_disagreements_remain_visible_and_are_never_dropped():
    """The task brief's explicit requirement (item 7): a listing/query function must never
    silently discard Minor disagreement or Material disagreement items -- they stay queryable with
    their status, distinct from Excluded, which requires an explicit reason."""
    store = BlindReviewStore()
    _assign_and_submit(
        store, target_ref="item_minor", value_a=GROUND_TRUTH_LINE, value_b=MINOR_VARIANT
    )
    _assign_and_submit(
        store, target_ref="item_material", value_a=GROUND_TRUTH_LINE, value_b=MATERIAL_VARIANT
    )
    _assign_and_submit(
        store, target_ref="item_needs_adjudication", value_a=GROUND_TRUTH_LINE, value_b=SEVERE_VARIANT
    )

    status = batch_completion_status(
        store,
        ["item_minor", "item_material", "item_needs_adjudication"],
        computed_at="2026-01-01T12:00:00Z",
    )

    # None of the three disagreement targets are missing from the per-target listing.
    assert set(status.target_statuses) == {
        "item_minor",
        "item_material",
        "item_needs_adjudication",
    }
    assert status.target_statuses["item_minor"] == BenchmarkStatus.MINOR_DISAGREEMENT
    assert status.target_statuses["item_material"] == BenchmarkStatus.MATERIAL_DISAGREEMENT
    assert (
        status.target_statuses["item_needs_adjudication"] == BenchmarkStatus.REQUIRES_ADJUDICATION
    )
    # And the aggregate counts reflect them too -- not zeroed out.
    assert status.by_status[BenchmarkStatus.MINOR_DISAGREEMENT] == 1
    assert status.by_status[BenchmarkStatus.MATERIAL_DISAGREEMENT] == 1
    assert status.by_status[BenchmarkStatus.REQUIRES_ADJUDICATION] == 1
    # No exclusion happened, so none of them are (mis)counted as excluded.
    assert status.excluded_count == 0
    assert all(
        status.target_statuses[t] != BenchmarkStatus.EXCLUDED_FROM_BENCHMARK
        for t in ("item_minor", "item_material", "item_needs_adjudication")
    )
