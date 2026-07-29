"""Review center ViewModel (Stage 11, brief's "User interface" -> review center).

A framework-independent wrapper over `review/blind_review/*`: assignment, submission, disagreement
visualization, adjudication, exclusion-with-reason, reviewer workload, and completion status.

**Blind isolation is not re-implemented here, and not routed around.** `BlindReviewStore` enforces
it mechanically (`submission_for_other_reviewer` raises `BlindIsolationError` until the requesting
reviewer has submitted; `both_submissions_if_complete` returns `None` rather than a partial pair).
This ViewModel calls those methods and lets them refuse -- it never reads the private
`_submissions` dict, and `reviewer_queue` deliberately exposes only the requesting reviewer's own
assignments. A UI built on this cannot leak the other reviewer's reading, because this layer has no
path to it either.

**Correction-time tracking gap (surfaced, not silently omitted).** The brief asks for correction
time per item. `ReviewSubmission` (`review/htr_models.py`) carries `submitted_at` but no
started/opened timestamp, and `ReviewAssignment` carries `assigned_at` -- so the only interval this
codebase can honestly compute today is assignment-to-submission *elapsed* time, which is wall-clock
latency (including nights and weekends), not correction effort. `ReviewItemRow.elapsed_since_assignment`
reports exactly that, under exactly that name, and `CORRECTION_TIME_GAP` below states the missing
field. Reporting wall-clock latency as "correction time" would be the fabrication this codebase's
Article 18 discipline exists to prevent.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from archivetrust.review.blind_review.agreement import (
    AgreementPolicy,
    BenchmarkStatus,
    DisagreementSpan,
)
from archivetrust.review.blind_review.adjudication import adjudicate
from archivetrust.review.blind_review.exclusion import ExclusionRecord, exclude_from_benchmark
from archivetrust.review.blind_review.outcome import resolve_benchmark_outcome
from archivetrust.review.blind_review.queries import (
    BatchCompletionStatus,
    batch_completion_status,
    reviewer_workload,
)
from archivetrust.review.blind_review.store import BlindReviewStore
from archivetrust.review.htr_models import (
    Adjudication,
    ReviewAssignment,
    ReviewAssignmentStatus,
    ReviewSubmission,
)
from archivetrust.presentation.display_names import benchmark_status_label

CORRECTION_TIME_GAP = (
    "Per-item correction time is not tracked: ReviewSubmission records submitted_at but no "
    "started_at/opened_at, so only assignment-to-submission wall-clock latency can be computed. "
    "Adding a started_at field to ReviewSubmission would close this."
)
"""Stated as a constant so a View can show the gap to a researcher rather than leaving an
unexplained blank column, and so a later stage that adds the field has one place to delete."""


class ReviewItemRow(BaseModel):
    """One assignment in a reviewer's own queue. Carries nothing about the other reviewer."""

    model_config = ConfigDict(frozen=True)

    assignment_id: str
    target_ref: str
    role: str
    status: str
    convention_id: str
    convention_version: int
    assigned_at: str
    submitted_at: str | None = None
    submitted_value: str | None = None
    illegible: bool = False
    elapsed_since_assignment: str | None = None
    """Wall-clock latency from `assigned_at` to `submitted_at` as a raw
    "<assigned_at> -> <submitted_at>" pair. Explicitly NOT correction time -- see
    `CORRECTION_TIME_GAP`."""


class DisagreementRow(BaseModel):
    """One located divergence between the two blind readings, for side-by-side rendering. Only
    available once both submissions are finalized -- `BlindReviewStore` refuses otherwise."""

    model_config = ConfigDict(frozen=True)

    op: str
    reviewer_a_word_index: int
    reviewer_b_word_index: int
    reviewer_a_text: str
    reviewer_b_text: str


class TargetReviewState(BaseModel):
    """The resolvable state of one review target."""

    model_config = ConfigDict(frozen=True)

    target_ref: str
    status: str
    status_label: str
    ready_for_benchmark: bool
    benchmark_value: str | None
    similarity_score: float | None = None
    character_error_rate_normalized: float | None = None
    disagreements: tuple[DisagreementRow, ...] = ()
    requires_adjudication: bool = False
    adjudicated: bool = False
    adjudicator_ref: str | None = None
    adjudication_rationale: str | None = None
    excluded: bool = False
    exclusion_reason: str | None = None
    excluded_by: str | None = None


class ReviewerWorkloadRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    reviewer_ref: str
    open_count: int
    submitted_count: int
    withdrawn_count: int
    total_count: int


class ReviewCenterViewModel:
    """Read/act surface over one `BlindReviewStore`.

    Acting methods (`submit`, `adjudicate_target`, `exclude_target`) delegate straight to the
    package's own service functions so every invariant those enforce -- submission locking,
    adjudicate-only-when-required, non-empty rationale, non-empty exclusion reason, no duplicate
    exclusion -- applies unchanged. This ViewModel adds no policy of its own.
    """

    def __init__(
        self,
        store: BlindReviewStore,
        *,
        policy: AgreementPolicy | None = None,
    ) -> None:
        self._store = store
        self._policy = policy

    # -- Reading -------------------------------------------------------------------------------

    def reviewer_queue(self, reviewer_ref: str) -> tuple[ReviewItemRow, ...]:
        """This reviewer's OWN assignments only. Never includes another reviewer's assignment, even
        for a target this reviewer also works on."""
        rows: list[ReviewItemRow] = []
        for assignment in self._store.all_assignments():
            if assignment.reviewer_ref != reviewer_ref:
                continue
            rows.append(self._item_row(assignment))
        return tuple(sorted(rows, key=lambda row: (row.assigned_at, row.assignment_id)))

    def _item_row(self, assignment: ReviewAssignment) -> ReviewItemRow:
        submission = self._store.own_submission(assignment.assignment_id)
        elapsed = None
        if submission is not None:
            elapsed = f"{assignment.assigned_at} -> {submission.submitted_at}"
        return ReviewItemRow(
            assignment_id=assignment.assignment_id,
            target_ref=assignment.target_ref,
            role=assignment.role.value,
            status=assignment.status.value,
            convention_id=assignment.convention_id,
            convention_version=assignment.convention_version,
            assigned_at=assignment.assigned_at,
            submitted_at=submission.submitted_at if submission is not None else None,
            submitted_value=submission.submitted_value if submission is not None else None,
            illegible=submission.illegible if submission is not None else False,
            elapsed_since_assignment=elapsed,
        )

    def open_assignment_count(self, reviewer_ref: str) -> int:
        return sum(
            1
            for assignment in self._store.all_assignments()
            if assignment.reviewer_ref == reviewer_ref
            and assignment.status is ReviewAssignmentStatus.OPEN
        )

    def workload(self, reviewer_ref: str) -> ReviewerWorkloadRow:
        """Delegates to `review/blind_review/queries.py::reviewer_workload` -- the counts are not
        recomputed here."""
        source = reviewer_workload(self._store, reviewer_ref)
        return ReviewerWorkloadRow(
            reviewer_ref=source.reviewer_ref,
            open_count=source.open_count,
            submitted_count=source.submitted_count,
            withdrawn_count=source.withdrawn_count,
            total_count=source.total_count,
        )

    def all_workloads(self) -> tuple[ReviewerWorkloadRow, ...]:
        reviewer_refs = sorted({a.reviewer_ref for a in self._store.all_assignments()})
        return tuple(self.workload(ref) for ref in reviewer_refs)

    def completion_status(
        self, target_refs: Sequence[str], *, computed_at: str
    ) -> BatchCompletionStatus:
        """Delegates to `queries.batch_completion_status`."""
        return batch_completion_status(
            self._store, target_refs, computed_at=computed_at, policy=self._policy
        )

    def target_state(self, target_ref: str, *, computed_at: str) -> TargetReviewState | None:
        """The resolved state of one target, or `None` when it is genuinely still pending (neither
        both-submitted nor excluded) -- a pending target has no state to report yet, which is not
        the same as an empty one."""
        exclusion = self._store.exclusion_for(target_ref)
        if exclusion is None and self._store.both_submissions_if_complete(target_ref) is None:
            return None

        outcome = resolve_benchmark_outcome(
            self._store, target_ref, computed_at=computed_at, policy=self._policy
        )
        assessment = outcome.assessment
        return TargetReviewState(
            target_ref=target_ref,
            status=outcome.status.value,
            status_label=benchmark_status_label(outcome.status),
            ready_for_benchmark=outcome.ready_for_benchmark,
            benchmark_value=outcome.benchmark_value,
            similarity_score=(
                assessment.agreement_result.similarity_score if assessment is not None else None
            ),
            character_error_rate_normalized=(
                assessment.recognition_metrics.character_error_rate_normalized
                if assessment is not None and assessment.recognition_metrics is not None
                else None
            ),
            disagreements=(
                tuple(_disagreement_row(span) for span in assessment.disagreement_locations)
                if assessment is not None
                else ()
            ),
            requires_adjudication=(
                assessment is not None
                and assessment.status is BenchmarkStatus.REQUIRES_ADJUDICATION
                and outcome.adjudication is None
            ),
            adjudicated=outcome.adjudication is not None,
            adjudicator_ref=(
                outcome.adjudication.adjudicator_ref if outcome.adjudication is not None else None
            ),
            adjudication_rationale=(
                outcome.adjudication.rationale if outcome.adjudication is not None else None
            ),
            excluded=outcome.exclusion is not None,
            exclusion_reason=(
                outcome.exclusion.reason if outcome.exclusion is not None else None
            ),
            excluded_by=(
                outcome.exclusion.excluded_by if outcome.exclusion is not None else None
            ),
        )

    def pending_disagreements(
        self, target_refs: Sequence[str], *, computed_at: str
    ) -> tuple[TargetReviewState, ...]:
        """Targets classified `REQUIRES_ADJUDICATION` with no adjudication recorded yet -- the
        research dashboard's "unresolved disagreements" count and the adjudication queue."""
        rows: list[TargetReviewState] = []
        for target_ref in target_refs:
            state = self.target_state(target_ref, computed_at=computed_at)
            if state is not None and state.requires_adjudication:
                rows.append(state)
        return tuple(rows)

    # -- Acting (delegated, never re-implemented) ----------------------------------------------

    def submit(self, submission: ReviewSubmission) -> ReviewSubmission:
        """Delegates to `BlindReviewStore.submit_submission`, whose locking guard rejects a second
        submission for an already-finalized assignment."""
        return self._store.submit_submission(submission)

    def other_reviewer_submission(
        self, *, target_ref: str, requesting_reviewer_ref: str
    ) -> ReviewSubmission:
        """Delegates to the store's blind-isolation guard. Raises `BlindIsolationError` if the
        requesting reviewer has not submitted -- this ViewModel does not catch or soften that."""
        return self._store.submission_for_other_reviewer(
            target_ref=target_ref, requesting_reviewer_ref=requesting_reviewer_ref
        )

    def adjudicate_target(
        self,
        *,
        target_ref: str,
        adjudicator_ref: str,
        resolved_value: str | None,
        rationale: str,
        adjudicated_at: str,
        illegible: bool = False,
    ) -> Adjudication:
        """Delegates to `review/blind_review/adjudication.py::adjudicate`, which rejects
        adjudicating a target that does not require it and rejects an empty rationale."""
        assessment = self._store.get_or_compute_agreement(
            target_ref, computed_at=adjudicated_at, policy=self._policy
        )
        return adjudicate(
            store=self._store,
            assessment=assessment,
            adjudicator_ref=adjudicator_ref,
            resolved_value=resolved_value,
            rationale=rationale,
            adjudicated_at=adjudicated_at,
            illegible=illegible,
        )

    def exclude_target(
        self, *, target_ref: str, reason: str, excluded_by: str, excluded_at: str
    ) -> ExclusionRecord:
        """Delegates to `review/blind_review/exclusion.py::exclude_from_benchmark`, which rejects a
        blank reason and a duplicate exclusion."""
        return exclude_from_benchmark(
            store=self._store,
            target_ref=target_ref,
            reason=reason,
            excluded_by=excluded_by,
            excluded_at=excluded_at,
        )


def _disagreement_row(span: DisagreementSpan) -> DisagreementRow:
    return DisagreementRow(
        op=span.op,
        reviewer_a_word_index=span.reviewer_a_word_index,
        reviewer_b_word_index=span.reviewer_b_word_index,
        reviewer_a_text=span.reviewer_a_text,
        reviewer_b_text=span.reviewer_b_text,
    )
