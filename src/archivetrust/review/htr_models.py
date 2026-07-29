"""Blind double-annotation review models (docs/htr-domain-design.md §1: "TranscriptionConvention
-> GroundTruthItem -> ReviewAssignment (reviewer A, reviewer B -- blind) -> ReviewSubmission ->
AgreementResult -> Adjudication").

Distinct from the pre-existing `review/service.py`/`ReviewAction`/`HumanCorrection` machinery,
which reviews a Canonical Observation the Trust Engine already produced. These models instead
govern the *ground-truth authoring* workflow: two independent, blinded annotators transcribe the
same `GroundTruthAnnotation` target, their submissions are compared for agreement, and a
disagreement is adjudicated by a third reviewer. Added alongside the existing package, matching
its style (frozen Pydantic models, plain-string ids via `new_id`).
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, model_validator

from archivetrust.domain.shared.ids import new_id


class ReviewAssignmentRole(str, Enum):
    """Which of the two blind annotators (or the adjudicator) this assignment is for (§1)."""

    REVIEWER_A = "reviewer_a"
    REVIEWER_B = "reviewer_b"
    ADJUDICATOR = "adjudicator"


class ReviewAssignmentStatus(str, Enum):
    OPEN = "open"
    SUBMITTED = "submitted"
    WITHDRAWN = "withdrawn"


class ReviewAssignment(BaseModel):
    """One reviewer's assignment to independently transcribe/review one `GroundTruthAnnotation`
    target. Blind by construction: a `ReviewAssignment` never carries the other role's submission
    or identity -- that isolation is enforced at the store/service layer (a later migration
    stage), not representable as a field here (there is nothing on this type *to* leak)."""

    model_config = ConfigDict(frozen=True)

    assignment_id: str
    target_ref: str
    """The `GroundTruthAnnotation.annotation_id` (or a not-yet-annotated target key) this
    assignment concerns."""
    role: ReviewAssignmentRole
    reviewer_ref: str
    convention_id: str
    convention_version: int
    status: ReviewAssignmentStatus = ReviewAssignmentStatus.OPEN
    assigned_at: str

    @classmethod
    def create(
        cls,
        *,
        target_ref: str,
        role: ReviewAssignmentRole,
        reviewer_ref: str,
        convention_id: str,
        convention_version: int,
        assigned_at: str,
    ) -> "ReviewAssignment":
        return cls(
            assignment_id=new_id("review_assignment"),
            target_ref=target_ref,
            role=role,
            reviewer_ref=reviewer_ref,
            convention_id=convention_id,
            convention_version=convention_version,
            assigned_at=assigned_at,
        )


class ReviewSubmission(BaseModel):
    """One reviewer's completed submission against a `ReviewAssignment` -- what they actually
    transcribed/decided, kept isolated from the other blind reviewer's submission until both are
    closed (enforcement lives at the service layer, §1 pre-licenses this as a future
    `ReviewSubmission`-write rejection, mirroring `HUMAN_REVIEW_SPECIFICATION.md`'s blind-isolation
    discipline, not yet wired at this stage)."""

    model_config = ConfigDict(frozen=True)

    submission_id: str
    assignment_id: str
    reviewer_ref: str
    submitted_value: str | None
    illegible: bool = False
    notes: str | None = None
    submitted_at: str

    @model_validator(mode="after")
    def _validate(self) -> "ReviewSubmission":
        if self.illegible and self.submitted_value is not None:
            raise ValueError("ReviewSubmission.illegible=True requires submitted_value=None")
        if not self.illegible and self.submitted_value is None:
            raise ValueError("ReviewSubmission.submitted_value is required unless illegible=True")
        return self

    @classmethod
    def create(
        cls,
        *,
        assignment_id: str,
        reviewer_ref: str,
        submitted_value: str | None,
        submitted_at: str,
        illegible: bool = False,
        notes: str | None = None,
    ) -> "ReviewSubmission":
        return cls(
            submission_id=new_id("review_submission"),
            assignment_id=assignment_id,
            reviewer_ref=reviewer_ref,
            submitted_value=submitted_value,
            illegible=illegible,
            notes=notes,
            submitted_at=submitted_at,
        )


class AgreementResult(BaseModel):
    """Computed from the two blind `ReviewSubmission`s for one target (§1). `agrees=False` routes
    to `Adjudication`; `agrees=True` needs none."""

    model_config = ConfigDict(frozen=True)

    agreement_result_id: str
    target_ref: str
    submission_a_id: str
    submission_b_id: str
    agrees: bool
    similarity_score: float | None = None
    computed_at: str

    @model_validator(mode="after")
    def _validate(self) -> "AgreementResult":
        if self.similarity_score is not None and not (0.0 <= self.similarity_score <= 1.0):
            raise ValueError("AgreementResult.similarity_score must be within [0.0, 1.0]")
        return self

    @classmethod
    def create(
        cls,
        *,
        target_ref: str,
        submission_a_id: str,
        submission_b_id: str,
        agrees: bool,
        computed_at: str,
        similarity_score: float | None = None,
    ) -> "AgreementResult":
        return cls(
            agreement_result_id=new_id("agreement_result"),
            target_ref=target_ref,
            submission_a_id=submission_a_id,
            submission_b_id=submission_b_id,
            agrees=agrees,
            similarity_score=similarity_score,
            computed_at=computed_at,
        )


class Adjudication(BaseModel):
    """Resolves a disagreeing `AgreementResult` -- only created when one is required (§1: "only
    when AgreementResult requires it")."""

    model_config = ConfigDict(frozen=True)

    adjudication_id: str
    agreement_result_id: str
    adjudicator_ref: str
    resolved_value: str | None
    illegible: bool = False
    rationale: str
    adjudicated_at: str

    @model_validator(mode="after")
    def _validate(self) -> "Adjudication":
        if self.illegible and self.resolved_value is not None:
            raise ValueError("Adjudication.illegible=True requires resolved_value=None")
        if not self.illegible and self.resolved_value is None:
            raise ValueError("Adjudication.resolved_value is required unless illegible=True")
        return self

    @classmethod
    def create(
        cls,
        *,
        agreement_result_id: str,
        adjudicator_ref: str,
        resolved_value: str | None,
        rationale: str,
        adjudicated_at: str,
        illegible: bool = False,
    ) -> "Adjudication":
        return cls(
            adjudication_id=new_id("adjudication"),
            agreement_result_id=agreement_result_id,
            adjudicator_ref=adjudicator_ref,
            resolved_value=resolved_value,
            illegible=illegible,
            rationale=rationale,
            adjudicated_at=adjudicated_at,
        )
