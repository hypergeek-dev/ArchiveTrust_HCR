"""Independent, blinded evaluation-reference assignment and adjudication workflow."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
import threading

from pydantic import BaseModel, ConfigDict, model_validator

from archivetrust.domain.evidence.models import BoundingBox
from archivetrust.domain.shared.ids import new_id
from archivetrust.evaluation.ground_truth import (
    AdjudicationStatus,
    AnnotatorKind,
    BlindingState,
    EvaluationReferenceStatus,
    FileGroundTruthStore,
    GroundTruthAnnotation,
    ReviewerIndependenceState,
    VerificationStatus,
)


class EvaluationAction(str, Enum):
    ACCEPT = "accept"
    CORRECT = "correct"
    ILLEGIBLE = "illegible"
    REJECT_SEGMENT = "reject_segment"
    BOUNDARY_INCORRECT = "boundary_incorrect"
    CANNOT_DETERMINE = "cannot_determine"


class SamplingStratum(str, Enum):
    PARAGRAPHS = "paragraphs"
    HEADINGS = "headings"
    TABLE_CELLS = "table_cells"
    TABLES = "tables"
    PAGE_REGIONS = "page_regions"
    DEGRADED_TEXT = "degraded_text"
    HANDWRITING = "handwriting"
    READING_ORDER_RELATIONSHIPS = "reading_order_relationships"
    LAYOUT_RELATIONSHIPS = "layout_relationships"
    DATES = "dates"
    NAMES = "names"
    IDENTIFIERS = "identifiers"


class EvaluationAssignment(BaseModel):
    model_config = ConfigDict(frozen=True)

    assignment_id: str
    archive_object_ref: str
    content_hash: str
    page: int
    region_geometry: BoundingBox
    semantic_slot_id: str | None = None
    observation_type: str
    task_type: str
    scope: str
    proposed_value: str | None = None
    annotation_method: str
    campaign_id: str
    sampling_stratum_id: SamplingStratum
    legal_basis: str
    sampling_basis: str
    reviewer_refs: tuple[str, str]
    status: EvaluationReferenceStatus = EvaluationReferenceStatus.ASSIGNED
    revision: int = 1
    recorded_at: str

    @model_validator(mode="after")
    def _independent_reviewers(self) -> "EvaluationAssignment":
        if self.reviewer_refs[0] == self.reviewer_refs[1]:
            raise ValueError("evaluation assignment requires two distinct reviewers")
        return self


class EvaluationTask(BaseModel):
    """Blinded frontend contract; deliberately contains no provider/canonical/confidence fields."""

    model_config = ConfigDict(frozen=True)

    assignment_id: str
    archive_object_ref: str
    page: int
    region_geometry: BoundingBox
    task_type: str
    scope: str
    observation_type: str
    instructions: str
    proposed_value: str | None


class AdjudicationTask(BaseModel):
    """Unblinded only after two independent submissions disagree."""

    model_config = ConfigDict(frozen=True)

    assignment_id: str
    archive_object_ref: str
    page: int
    region_geometry: BoundingBox
    task_type: str
    scope: str
    observation_type: str
    instructions: str
    reviewer_answers: tuple[str | None, str | None]


class AgreementMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    assignments_with_two_reviews: int
    exact_agreements: int
    disagreements: int
    exact_agreement_rate: float | None


class FileEvaluationAssignmentStore:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self._lock = threading.Lock()

    def append(self, assignment: EvaluationAssignment) -> None:
        current = self.by_id(assignment.assignment_id)
        if current is not None and assignment.revision != current.revision + 1:
            raise ValueError("assignment revision must advance exactly once")
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(assignment.model_dump_json())
            handle.write("\n")

    def all_records(self) -> tuple[EvaluationAssignment, ...]:
        with self._lock, self.path.open("r", encoding="utf-8") as handle:
            return tuple(
                EvaluationAssignment.model_validate_json(line)
                for line in handle
                if line.strip()
            )

    def latest(self) -> tuple[EvaluationAssignment, ...]:
        latest: dict[str, EvaluationAssignment] = {}
        order: list[str] = []
        for record in self.all_records():
            if record.assignment_id not in latest:
                order.append(record.assignment_id)
            current = latest.get(record.assignment_id)
            if current is None or record.revision > current.revision:
                latest[record.assignment_id] = record
        return tuple(latest[assignment_id] for assignment_id in order)

    def by_id(self, assignment_id: str) -> EvaluationAssignment | None:
        return next((item for item in self.latest() if item.assignment_id == assignment_id), None)


class EvaluationApprovalService:
    def __init__(
        self,
        *,
        assignments: FileEvaluationAssignmentStore,
        annotations: FileGroundTruthStore,
    ) -> None:
        self.assignments = assignments
        self.annotations = annotations

    def create_assignment(self, **kwargs) -> EvaluationAssignment:
        assignment = EvaluationAssignment(
            assignment_id=new_id("evaluation_assignment"),
            recorded_at=datetime.now(timezone.utc).isoformat(),
            **kwargs,
        )
        self.assignments.append(assignment)
        return assignment

    def next_task(self, reviewer_ref: str) -> EvaluationTask | None:
        submitted = {
            annotation.assignment_id
            for annotation in self.annotations.latest()
            if annotation.annotator == reviewer_ref
        }
        assignment = next(
            (
                item
                for item in self.assignments.latest()
                if reviewer_ref in item.reviewer_refs
                and item.assignment_id not in submitted
                and item.status
                not in (
                    EvaluationReferenceStatus.EXCLUDED,
                    EvaluationReferenceStatus.SUPERSEDED,
                    EvaluationReferenceStatus.VERIFIED,
                    EvaluationReferenceStatus.ADJUDICATED,
                )
            ),
            None,
        )
        if assignment is None:
            return None
        return EvaluationTask(
            assignment_id=assignment.assignment_id,
            archive_object_ref=assignment.archive_object_ref,
            page=assignment.page,
            region_geometry=assignment.region_geometry,
            task_type=assignment.task_type,
            scope=assignment.scope,
            observation_type=assignment.observation_type,
            instructions=(
                "Judge only the highlighted source segment. Do not infer from system identity, "
                "confidence, agreement, or another reviewer's answer."
            ),
            proposed_value=assignment.proposed_value,
        )

    def submit(
        self,
        *,
        assignment_id: str,
        reviewer_ref: str,
        action: EvaluationAction,
        submitted_value: str | None = None,
        uncertain: bool = False,
        annotator_kind: AnnotatorKind = AnnotatorKind.HUMAN,
    ) -> GroundTruthAnnotation:
        assignment = self._assignment_for_reviewer(assignment_id, reviewer_ref)
        if any(
            annotation.assignment_id == assignment_id and annotation.annotator == reviewer_ref
            for annotation in self.annotations.latest()
        ):
            raise ValueError("reviewer already submitted this blinded assignment")
        if action is EvaluationAction.ACCEPT:
            submitted_value = assignment.proposed_value
        if action is EvaluationAction.CORRECT and not submitted_value:
            raise ValueError("correct requires submitted_value")
        illegible = action is EvaluationAction.ILLEGIBLE
        text = None if illegible else submitted_value
        excluded_action = action in (
            EvaluationAction.REJECT_SEGMENT,
            EvaluationAction.BOUNDARY_INCORRECT,
            EvaluationAction.CANNOT_DETERMINE,
        )
        annotation = GroundTruthAnnotation(
            annotation_id=new_id("evaluation_annotation"),
            archive_object_ref=assignment.archive_object_ref,
            content_hash=assignment.content_hash,
            page=assignment.page,
            field=assignment.scope,
            observation_type=assignment.observation_type,
            text=text,
            uncertain=uncertain or action is EvaluationAction.CANNOT_DETERMINE,
            illegible=illegible,
            annotator=reviewer_ref,
            method=assignment.annotation_method,
            source="independent_evaluation_approval",
            created_at=datetime.now(timezone.utc).isoformat(),
            region_geometry=assignment.region_geometry,
            semantic_slot_id=assignment.semantic_slot_id,
            task_type=assignment.task_type,
            scope=assignment.scope,
            proposed_value=assignment.proposed_value,
            submitted_value=submitted_value,
            assignment_id=assignment.assignment_id,
            verification_status=VerificationStatus.UNVERIFIED,
            reference_status=EvaluationReferenceStatus.INDEPENDENTLY_ANNOTATED,
            reviewer_independence_state=ReviewerIndependenceState.INDEPENDENT,
            blinding_state=BlindingState.BLINDED,
            campaign_id=assignment.campaign_id,
            sampling_stratum_id=assignment.sampling_stratum_id,
            legal_basis=assignment.legal_basis,
            sampling_basis=assignment.sampling_basis,
            exclusion_reason=action.value if excluded_action else None,
            annotator_kind=annotator_kind,
            notes=f"evaluation_action:{action.value}",
        )
        self.annotations.append(annotation, expected_content_hash=assignment.content_hash)
        self._advance_after_submission(assignment)
        return annotation

    def next_adjudication(self, adjudicator_ref: str) -> AdjudicationTask | None:
        assignment = next(
            (
                item
                for item in self.assignments.latest()
                if item.status is EvaluationReferenceStatus.AWAITING_ADJUDICATION
                and adjudicator_ref not in item.reviewer_refs
            ),
            None,
        )
        if assignment is None:
            return None
        reviews = self._reviews(assignment.assignment_id)
        return AdjudicationTask(
            assignment_id=assignment.assignment_id,
            archive_object_ref=assignment.archive_object_ref,
            page=assignment.page,
            region_geometry=assignment.region_geometry,
            task_type=assignment.task_type,
            scope=assignment.scope,
            observation_type=assignment.observation_type,
            instructions=(
                "Resolve the two independent readings against the highlighted source segment. "
                "The resulting reference is appended; neither reading is overwritten."
            ),
            reviewer_answers=(reviews[0].submitted_value, reviews[1].submitted_value),
        )

    def adjudicate(
        self,
        *,
        assignment_id: str,
        adjudicator_ref: str,
        submitted_value: str,
        method: str,
    ) -> GroundTruthAnnotation:
        assignment = self._required_assignment(assignment_id)
        reviews = self._reviews(assignment_id)
        if assignment.status not in (
            EvaluationReferenceStatus.DISAGREEMENT,
            EvaluationReferenceStatus.AWAITING_ADJUDICATION,
        ) or len(reviews) != 2:
            raise ValueError("adjudication requires two disagreeing independent reviews")
        if adjudicator_ref in assignment.reviewer_refs:
            raise ValueError("adjudicator must be independent of both assigned reviewers")
        annotation = GroundTruthAnnotation(
            annotation_id=new_id("evaluation_reference"),
            archive_object_ref=assignment.archive_object_ref,
            content_hash=assignment.content_hash,
            page=assignment.page,
            field=assignment.scope,
            observation_type=assignment.observation_type,
            text=submitted_value,
            annotator=adjudicator_ref,
            method=method,
            source="independent_evaluation_adjudication",
            created_at=datetime.now(timezone.utc).isoformat(),
            adjudication_status=AdjudicationStatus.ADJUDICATED,
            region_geometry=assignment.region_geometry,
            semantic_slot_id=assignment.semantic_slot_id,
            task_type=assignment.task_type,
            scope=assignment.scope,
            proposed_value=assignment.proposed_value,
            submitted_value=submitted_value,
            assignment_id=assignment.assignment_id,
            verification_status=VerificationStatus.VERIFIED,
            reference_status=EvaluationReferenceStatus.ADJUDICATED,
            reviewer_independence_state=ReviewerIndependenceState.ADJUDICATOR,
            blinding_state=BlindingState.UNBLINDED,
            campaign_id=assignment.campaign_id,
            sampling_stratum_id=assignment.sampling_stratum_id,
            legal_basis=assignment.legal_basis,
            sampling_basis=assignment.sampling_basis,
            annotator_kind=AnnotatorKind.HUMAN,
            adjudicates=tuple(review.annotation_id for review in reviews),
        )
        self.annotations.append(annotation, expected_content_hash=assignment.content_hash)
        self._revise(assignment, EvaluationReferenceStatus.ADJUDICATED)
        return annotation

    def agreement_metrics(self) -> AgreementMetrics:
        pairs = [self._reviews(item.assignment_id) for item in self.assignments.latest()]
        pairs = [pair for pair in pairs if len(pair) == 2]
        exact = sum(self._answer_key(pair[0]) == self._answer_key(pair[1]) for pair in pairs)
        return AgreementMetrics(
            assignments_with_two_reviews=len(pairs),
            exact_agreements=exact,
            disagreements=len(pairs) - exact,
            exact_agreement_rate=exact / len(pairs) if pairs else None,
        )

    def _advance_after_submission(self, assignment: EvaluationAssignment) -> None:
        reviews = self._reviews(assignment.assignment_id)
        if len(reviews) == 1:
            self._revise(assignment, EvaluationReferenceStatus.AWAITING_SECOND_REVIEW)
        elif len(reviews) == 2:
            agreed = self._answer_key(reviews[0]) == self._answer_key(reviews[1])
            if agreed:
                self._record_agreed_reference(assignment, reviews)
            if agreed:
                self._revise(assignment, EvaluationReferenceStatus.VERIFIED)
            else:
                self._revise(assignment, EvaluationReferenceStatus.DISAGREEMENT)
                self._revise(assignment, EvaluationReferenceStatus.AWAITING_ADJUDICATION)

    def _record_agreed_reference(
        self,
        assignment: EvaluationAssignment,
        reviews: tuple[GroundTruthAnnotation, ...],
    ) -> GroundTruthAnnotation:
        first, second = reviews
        reference = GroundTruthAnnotation(
            annotation_id=new_id("evaluation_reference"),
            archive_object_ref=assignment.archive_object_ref,
            content_hash=assignment.content_hash,
            page=assignment.page,
            field=assignment.scope,
            observation_type=assignment.observation_type,
            text=first.text,
            uncertain=first.uncertain or second.uncertain,
            illegible=first.illegible,
            annotator=f"agreement:{first.annotator}+{second.annotator}",
            method=assignment.annotation_method,
            source="independent_evaluation_agreement",
            created_at=datetime.now(timezone.utc).isoformat(),
            adjudication_status=AdjudicationStatus.ADJUDICATED,
            region_geometry=assignment.region_geometry,
            semantic_slot_id=assignment.semantic_slot_id,
            task_type=assignment.task_type,
            scope=assignment.scope,
            proposed_value=assignment.proposed_value,
            submitted_value=first.submitted_value,
            assignment_id=assignment.assignment_id,
            verification_status=(
                VerificationStatus.EXCLUDED
                if first.exclusion_reason is not None or first.illegible
                else VerificationStatus.VERIFIED
            ),
            reference_status=(
                EvaluationReferenceStatus.EXCLUDED
                if first.exclusion_reason is not None or first.illegible
                else EvaluationReferenceStatus.VERIFIED
            ),
            reviewer_independence_state=ReviewerIndependenceState.INDEPENDENT,
            blinding_state=BlindingState.BLINDED,
            campaign_id=assignment.campaign_id,
            sampling_stratum_id=assignment.sampling_stratum_id,
            legal_basis=assignment.legal_basis,
            sampling_basis=assignment.sampling_basis,
            exclusion_reason=first.exclusion_reason,
            annotator_kind=(
                AnnotatorKind.HUMAN
                if first.annotator_kind is AnnotatorKind.HUMAN
                and second.annotator_kind is AnnotatorKind.HUMAN
                else AnnotatorKind.AI_ASSISTED
            ),
            adjudicates=(first.annotation_id, second.annotation_id),
        )
        self.annotations.append(reference, expected_content_hash=assignment.content_hash)
        return reference

    def _revise(self, assignment: EvaluationAssignment, status: EvaluationReferenceStatus) -> None:
        current = self._required_assignment(assignment.assignment_id)
        self.assignments.append(
            current.model_copy(
                update={
                    "status": status,
                    "revision": current.revision + 1,
                    "recorded_at": datetime.now(timezone.utc).isoformat(),
                }
            )
        )

    def _reviews(self, assignment_id: str) -> tuple[GroundTruthAnnotation, ...]:
        return tuple(
            annotation
            for annotation in self.annotations.latest()
            if annotation.assignment_id == assignment_id
            and annotation.source == "independent_evaluation_approval"
            and annotation.reviewer_independence_state is ReviewerIndependenceState.INDEPENDENT
        )

    @staticmethod
    def _answer_key(annotation: GroundTruthAnnotation) -> tuple:
        return (
            annotation.notes,
            annotation.submitted_value,
            annotation.illegible,
            annotation.exclusion_reason,
        )

    def _assignment_for_reviewer(
        self, assignment_id: str, reviewer_ref: str
    ) -> EvaluationAssignment:
        assignment = self._required_assignment(assignment_id)
        if reviewer_ref not in assignment.reviewer_refs:
            raise PermissionError("reviewer is not assigned to this evaluation chunk")
        return assignment

    def _required_assignment(self, assignment_id: str) -> EvaluationAssignment:
        assignment = self.assignments.by_id(assignment_id)
        if assignment is None:
            raise KeyError(assignment_id)
        return assignment
