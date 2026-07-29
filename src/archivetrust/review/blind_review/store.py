"""Blind dual-review submission store: the mechanical enforcement of blind isolation and
submission locking (task brief's "Blind dual-review workflow" -- this is its "hard, important
part").

Two invariants live here, enforced by code a caller cannot route around, not by convention:

1. **Blind isolation.** Reviewer A can never read reviewer B's `ReviewSubmission` for the same
   target (and vice versa) until BOTH have submitted. `submission_for_other_reviewer` is the one
   path through which a caller can read the *other* blind reviewer's result, and it raises
   `BlindIsolationError` unless the requesting reviewer's own submission is already finalized (i.e.
   already recorded here). `both_submissions_if_complete` is the listing-side counterpart: it
   returns `None` -- never a partial pair -- until both are finalized.
2. **Submission locking.** Once a `ReviewSubmission` is recorded against a `ReviewAssignment` here,
   that assignment is finalized: a second `submit_submission` call for the same `assignment_id` is
   rejected with `AlreadyFinalizedError`, never silently overwritten. `ReviewSubmission` itself is
   already a frozen Pydantic model (no field can be mutated in place); this guard closes the
   remaining gap -- replacing a finalized submission with a *different* frozen instance under the
   same assignment -- the same "supersede, never silently overwrite" discipline Constitution
   Article 15 requires of the Canonical layer, applied here by refusing the overwrite outright
   (a blind submission has no legitimate "corrected" successor the way a Canonical Observation
   does; a wrong submission is fixed by adjudication, not by re-submitting).

In-memory by design, mirroring `review/sampling/log.py`'s `InMemorySamplingLogSink` /
`FileSamplingLogSink` split: a durable, file-backed variant can be added the same way, if/when a
deployment needs one. Not required by this phase's scope, which is the workflow logic itself.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from archivetrust.review.blind_review.agreement import (
    AgreementAssessment,
    AgreementPolicy,
    compute_agreement,
)
from archivetrust.review.htr_models import (
    Adjudication,
    ReviewAssignment,
    ReviewAssignmentRole,
    ReviewAssignmentStatus,
    ReviewSubmission,
)

if TYPE_CHECKING:  # pragma: no cover - import cycle avoidance only
    from archivetrust.review.blind_review.exclusion import ExclusionRecord


class UnknownAssignmentError(KeyError):
    """Raised when an operation references a `ReviewAssignment`, target, or reviewer the store has
    never seen (e.g. a target with no registered blind pair, or a reviewer not assigned to it)."""


class AlreadyFinalizedError(ValueError):
    """Raised when a caller attempts to write over something this store already finalized: a
    second `ReviewSubmission` for an assignment that already has one, or a second `Adjudication`
    for an `AgreementResult` that already has one. Nothing here is silently overwritten."""


class BlindIsolationError(PermissionError):
    """Raised when a reviewer attempts to read the other blind reviewer's `ReviewSubmission` for a
    target before their OWN submission is finalized -- the mechanical enforcement of blind review,
    not a convention a caller could bypass by simply not checking."""


class DuplicateExclusionError(ValueError):
    """Raised when a target already has a recorded `ExclusionRecord` (exclusion, like submission,
    is not silently overwritten -- a second exclusion attempt for the same target is rejected)."""


class BlindReviewStore:
    """Holds `ReviewAssignment`s, their `ReviewSubmission`s, recorded `Adjudication`s, and
    `ExclusionRecord`s for a set of blind dual-review targets. The sole gate through which a
    submission becomes visible to the other blind reviewer or to any listing/query function in this
    package.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._assignments: dict[str, ReviewAssignment] = {}
        self._submissions: dict[str, ReviewSubmission] = {}  # assignment_id -> submission
        self._agreements: dict[str, AgreementAssessment] = {}  # target_ref -> assessment
        self._adjudications: dict[str, Adjudication] = {}  # agreement_result_id -> adjudication
        self._exclusions: dict[str, "ExclusionRecord"] = {}  # target_ref -> exclusion

    # -- Assignment registration ---------------------------------------------------------------

    def register_assignment(self, assignment: ReviewAssignment) -> None:
        with self._lock:
            self._assignments[assignment.assignment_id] = assignment

    def register_blind_pair(
        self, assignment_a: ReviewAssignment, assignment_b: ReviewAssignment
    ) -> None:
        """Registers exactly one REVIEWER_A and one REVIEWER_B assignment for the same target --
        the pair `assignment.create_blind_review_pair` produces."""
        if assignment_a.target_ref != assignment_b.target_ref:
            raise ValueError("A blind pair must target the same GroundTruthItem (target_ref)")
        if {assignment_a.role, assignment_b.role} != {
            ReviewAssignmentRole.REVIEWER_A,
            ReviewAssignmentRole.REVIEWER_B,
        }:
            raise ValueError(
                "A blind pair must contain exactly one REVIEWER_A and one REVIEWER_B assignment"
            )
        self.register_assignment(assignment_a)
        self.register_assignment(assignment_b)

    def assignments_for_target(self, target_ref: str) -> tuple[ReviewAssignment, ...]:
        with self._lock:
            return tuple(a for a in self._assignments.values() if a.target_ref == target_ref)

    def all_assignments(self) -> tuple[ReviewAssignment, ...]:
        with self._lock:
            return tuple(self._assignments.values())

    def _blind_pair_for_target(self, target_ref: str) -> tuple[ReviewAssignment, ReviewAssignment]:
        """Caller must hold `self._lock`."""
        pair = [
            a
            for a in self._assignments.values()
            if a.target_ref == target_ref
            and a.role in (ReviewAssignmentRole.REVIEWER_A, ReviewAssignmentRole.REVIEWER_B)
        ]
        if len(pair) != 2:
            raise UnknownAssignmentError(
                f"target {target_ref!r} does not have exactly two blind-reviewer assignments "
                f"registered (found {len(pair)})"
            )
        pair.sort(key=lambda a: a.role.value)  # deterministic: reviewer_a before reviewer_b
        return pair[0], pair[1]

    # -- Submission (locking) ------------------------------------------------------------------

    def submit_submission(self, submission: ReviewSubmission) -> ReviewSubmission:
        """Finalizes one reviewer's submission against their `ReviewAssignment`. Locked: a second
        call for the same `assignment_id` -- whether identical or different content -- is rejected
        with `AlreadyFinalizedError`, never silently accepted as a replacement."""
        with self._lock:
            assignment = self._assignments.get(submission.assignment_id)
            if assignment is None:
                raise UnknownAssignmentError(
                    f"submission references unknown assignment {submission.assignment_id!r}"
                )
            if assignment.reviewer_ref != submission.reviewer_ref:
                raise ValueError(
                    f"submission reviewer_ref {submission.reviewer_ref!r} does not match "
                    f"assignment {submission.assignment_id!r}'s reviewer_ref "
                    f"{assignment.reviewer_ref!r}"
                )
            if submission.assignment_id in self._submissions:
                raise AlreadyFinalizedError(
                    f"assignment {submission.assignment_id!r} already has a finalized submission "
                    "-- a submitted ReviewSubmission cannot be mutated or replaced"
                )
            self._submissions[submission.assignment_id] = submission
            self._assignments[assignment.assignment_id] = assignment.model_copy(
                update={"status": ReviewAssignmentStatus.SUBMITTED}
            )
            return submission

    def is_finalized(self, assignment_id: str) -> bool:
        with self._lock:
            return assignment_id in self._submissions

    def own_submission(self, assignment_id: str) -> ReviewSubmission | None:
        """A reviewer reading their OWN submission back -- always allowed, no isolation concern."""
        with self._lock:
            return self._submissions.get(assignment_id)

    # -- Blind isolation (the hard, mechanical guard) -------------------------------------------

    def submission_for_other_reviewer(
        self, *, target_ref: str, requesting_reviewer_ref: str
    ) -> ReviewSubmission:
        """Returns the OTHER blind reviewer's submission for `target_ref` -- ONLY once the
        requesting reviewer's own submission for this target is already finalized. Raises
        `BlindIsolationError` otherwise. This is the one path through which a submission could leak
        to the other blind reviewer, and it is mechanically refused here, not merely undocumented.
        """
        with self._lock:
            assignment_a, assignment_b = self._blind_pair_for_target(target_ref)
            if requesting_reviewer_ref == assignment_a.reviewer_ref:
                own, other = assignment_a, assignment_b
            elif requesting_reviewer_ref == assignment_b.reviewer_ref:
                own, other = assignment_b, assignment_a
            else:
                raise UnknownAssignmentError(
                    f"{requesting_reviewer_ref!r} is not one of the two blind reviewers assigned "
                    f"to {target_ref!r}"
                )
            if own.assignment_id not in self._submissions:
                raise BlindIsolationError(
                    f"{requesting_reviewer_ref!r} cannot view the other reviewer's submission for "
                    f"{target_ref!r} before submitting their own (blind review isolation)"
                )
            other_submission = self._submissions.get(other.assignment_id)
            if other_submission is None:
                raise BlindIsolationError(
                    f"the other blind reviewer has not yet submitted for {target_ref!r}"
                )
            return other_submission

    def both_submissions_if_complete(
        self, target_ref: str
    ) -> tuple[ReviewSubmission, ReviewSubmission] | None:
        """Both blind submissions for `target_ref`, in (reviewer_a, reviewer_b) order -- returned
        ONLY once both are finalized; `None` otherwise (including when `target_ref` has no
        registered blind pair at all). The listing-side counterpart to
        `submission_for_other_reviewer`'s per-reviewer guard: no caller of this function sees a
        partial pair, regardless of who they are.
        """
        with self._lock:
            try:
                assignment_a, assignment_b = self._blind_pair_for_target(target_ref)
            except UnknownAssignmentError:
                return None
            submission_a = self._submissions.get(assignment_a.assignment_id)
            submission_b = self._submissions.get(assignment_b.assignment_id)
            if submission_a is None or submission_b is None:
                return None
            return submission_a, submission_b

    # -- Agreement computation (cached, so its id stays stable) ---------------------------------

    def get_or_compute_agreement(
        self, target_ref: str, *, computed_at: str, policy: AgreementPolicy | None = None
    ) -> AgreementAssessment:
        """The `AgreementAssessment` (and its wrapped `AgreementResult`) for `target_ref`, computed
        once and cached here. `compute_agreement` itself is a pure function that mints a fresh,
        random `AgreementResult.agreement_result_id` on every call (by design -- it has no store to
        remember an id in); an `Adjudication` recorded against one computation's id must still be
        findable the next time this target's agreement is resolved (e.g. `outcome.
        resolve_benchmark_outcome`, possibly called again after an `Adjudication` is recorded), so
        this method is the one place that computation is actually cached and reused, matching
        `AgreementResult`'s own semantics: "computed from the two submissions" once, not
        re-minted with a new identity on every read.
        """
        with self._lock:
            cached = self._agreements.get(target_ref)
            if cached is not None:
                return cached
            assignment_a, assignment_b = self._blind_pair_for_target(target_ref)
            submission_a = self._submissions.get(assignment_a.assignment_id)
            submission_b = self._submissions.get(assignment_b.assignment_id)
            if submission_a is None or submission_b is None:
                raise UnknownAssignmentError(
                    f"target {target_ref!r} does not yet have both blind submissions finalized"
                )
            assessment = compute_agreement(
                target_ref=target_ref,
                submission_a=submission_a,
                submission_b=submission_b,
                computed_at=computed_at,
                policy=policy,
            )
            self._agreements[target_ref] = assessment
            return assessment

    def cached_agreement_for(self, target_ref: str) -> AgreementAssessment | None:
        with self._lock:
            return self._agreements.get(target_ref)

    # -- Adjudication / exclusion recording ------------------------------------------------------

    def record_adjudication(self, adjudication: Adjudication) -> None:
        with self._lock:
            if adjudication.agreement_result_id in self._adjudications:
                raise AlreadyFinalizedError(
                    f"agreement result {adjudication.agreement_result_id!r} already has a "
                    "recorded adjudication -- never overwritten"
                )
            self._adjudications[adjudication.agreement_result_id] = adjudication

    def adjudication_for(self, agreement_result_id: str) -> Adjudication | None:
        with self._lock:
            return self._adjudications.get(agreement_result_id)

    def record_exclusion(self, exclusion: "ExclusionRecord") -> None:
        with self._lock:
            if exclusion.target_ref in self._exclusions:
                raise DuplicateExclusionError(
                    f"target {exclusion.target_ref!r} already has a recorded exclusion"
                )
            self._exclusions[exclusion.target_ref] = exclusion

    def exclusion_for(self, target_ref: str) -> "ExclusionRecord | None":
        with self._lock:
            return self._exclusions.get(target_ref)
