"""Reviewer workload / review completion status (task brief's "Review center" list): plain
data-returning query functions backing a future UI. No UI code here, per the brief's instruction.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from archivetrust.review.blind_review.agreement import AgreementPolicy, BenchmarkStatus
from archivetrust.review.blind_review.outcome import resolve_benchmark_outcome
from archivetrust.review.blind_review.store import BlindReviewStore
from archivetrust.review.htr_models import ReviewAssignmentStatus


class ReviewerWorkload(BaseModel):
    """How many assignments one reviewer has open vs. closed (task brief: "how many assignments a
    reviewer has open/closed")."""

    model_config = ConfigDict(frozen=True)

    reviewer_ref: str
    open_count: int
    submitted_count: int
    withdrawn_count: int
    total_count: int


def reviewer_workload(store: BlindReviewStore, reviewer_ref: str) -> ReviewerWorkload:
    assignments = [a for a in store.all_assignments() if a.reviewer_ref == reviewer_ref]
    open_count = sum(1 for a in assignments if a.status is ReviewAssignmentStatus.OPEN)
    submitted_count = sum(1 for a in assignments if a.status is ReviewAssignmentStatus.SUBMITTED)
    withdrawn_count = sum(1 for a in assignments if a.status is ReviewAssignmentStatus.WITHDRAWN)
    return ReviewerWorkload(
        reviewer_ref=reviewer_ref,
        open_count=open_count,
        submitted_count=submitted_count,
        withdrawn_count=withdrawn_count,
        total_count=len(assignments),
    )


class BatchCompletionStatus(BaseModel):
    """How many `GroundTruthItem`s in a batch have both submissions, how many are adjudicated, how
    many are excluded (task brief: "review completion status"). `target_statuses` keeps every
    target's individual classification queryable -- `MINOR_DISAGREEMENT` and `MATERIAL_DISAGREEMENT`
    items are never collapsed out of this listing, only `EXCLUDED_FROM_BENCHMARK` items are marked
    as such, and only via an explicit, reasoned exclusion (task brief item 7)."""

    model_config = ConfigDict(frozen=True)

    total_targets: int
    both_submitted_count: int
    pending_count: int
    """Targets that do not yet have both blind submissions finalized and are not excluded."""
    adjudicated_count: int
    excluded_count: int
    by_status: dict[BenchmarkStatus, int]
    target_statuses: dict[str, BenchmarkStatus]
    """Every target that has a resolvable status (both-submitted or excluded), individually --
    never aggregated away. Targets still `pending_count` (neither both-submitted nor excluded) are
    intentionally absent here: there is no status to report yet, not a dropped one."""


def batch_completion_status(
    store: BlindReviewStore,
    target_refs: Sequence[str],
    *,
    computed_at: str,
    policy: AgreementPolicy | None = None,
) -> BatchCompletionStatus:
    both_submitted = 0
    excluded = 0
    adjudicated = 0
    by_status: dict[BenchmarkStatus, int] = {status: 0 for status in BenchmarkStatus}
    target_statuses: dict[str, BenchmarkStatus] = {}

    for target_ref in target_refs:
        exclusion = store.exclusion_for(target_ref)
        pair = store.both_submissions_if_complete(target_ref)

        if exclusion is not None:
            excluded += 1
            by_status[BenchmarkStatus.EXCLUDED_FROM_BENCHMARK] += 1
            target_statuses[target_ref] = BenchmarkStatus.EXCLUDED_FROM_BENCHMARK
            continue

        if pair is None:
            continue  # genuinely pending: neither both-submitted nor excluded

        both_submitted += 1
        outcome = resolve_benchmark_outcome(
            store, target_ref, computed_at=computed_at, policy=policy
        )
        by_status[outcome.status] += 1
        target_statuses[target_ref] = outcome.status
        if outcome.adjudication is not None:
            adjudicated += 1

    return BatchCompletionStatus(
        total_targets=len(target_refs),
        both_submitted_count=both_submitted,
        pending_count=len(target_refs) - both_submitted - excluded,
        adjudicated_count=adjudicated,
        excluded_count=excluded,
        by_status=by_status,
        target_statuses=target_statuses,
    )
