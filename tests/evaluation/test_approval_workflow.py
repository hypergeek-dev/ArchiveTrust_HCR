from __future__ import annotations

import pytest

from archivetrust.domain.evidence.models import BoundingBox, Precision
from archivetrust.evaluation.ground_truth import (
    AnnotatorKind,
    EvaluationReferenceStatus,
    FileGroundTruthStore,
    VerificationStatus,
)
from archivetrust.evaluation.workflow import (
    EvaluationAction,
    EvaluationApprovalService,
    FileEvaluationAssignmentStore,
)


def _service(tmp_path) -> EvaluationApprovalService:
    return EvaluationApprovalService(
        assignments=FileEvaluationAssignmentStore(tmp_path / "evaluation" / "assignments.jsonl"),
        annotations=FileGroundTruthStore(tmp_path / "evaluation" / "annotations.jsonl"),
    )


def _assignment(service: EvaluationApprovalService, **overrides):
    fields = dict(
        archive_object_ref="archive_object_1",
        content_hash="sha256:abc",
        page=2,
        region_geometry=BoundingBox(
            x0=10, y0=20, x1=210, y1=80, precision=Precision.PIXEL_ACCURATE
        ),
        semantic_slot_id="slot_1",
        observation_type="paragraph",
        task_type="transcription_approval",
        scope="paragraph",
        proposed_value="KommunfullmÃ¤ktige beslutar",
        annotation_method="visual transcription from source crop",
        campaign_id="campaign_1",
        sampling_stratum_id="degraded_text",
        legal_basis="internal municipal evaluation authorization",
        sampling_basis="campaign protocol section 4 stratified sample",
        reviewer_refs=("reviewer_a", "reviewer_b"),
    )
    fields.update(overrides)
    return service.create_assignment(**fields)


def test_blinded_task_contract_exposes_no_system_identity_or_confidence(tmp_path):
    service = _service(tmp_path)
    assignment = _assignment(service)

    task = service.next_task("reviewer_a")

    assert task.assignment_id == assignment.assignment_id
    assert task.proposed_value == "KommunfullmÃ¤ktige beslutar"
    forbidden = {
        "provider_id",
        "canonical_id",
        "provider_confidence",
        "canonical_confidence",
        "agreement_count",
        "previous_reviews",
        "expected_result",
    }
    assert forbidden.isdisjoint(type(task).model_fields)


def test_two_reviewers_are_blind_and_exact_agreement_creates_new_verified_record(tmp_path):
    service = _service(tmp_path)
    assignment = _assignment(service)

    first = service.submit(
        assignment_id=assignment.assignment_id,
        reviewer_ref="reviewer_a",
        action=EvaluationAction.ACCEPT,
    )
    assert service.assignments.by_id(assignment.assignment_id).status is (
        EvaluationReferenceStatus.AWAITING_SECOND_REVIEW
    )
    # Reviewer B receives only the original blinded task, never reviewer A's answer.
    assert service.next_task("reviewer_b").proposed_value == assignment.proposed_value

    second = service.submit(
        assignment_id=assignment.assignment_id,
        reviewer_ref="reviewer_b",
        action=EvaluationAction.ACCEPT,
    )

    assert service.assignments.by_id(assignment.assignment_id).status is (
        EvaluationReferenceStatus.VERIFIED
    )
    (reference,) = service.annotations.verified_references()
    assert reference.text == assignment.proposed_value
    assert reference.adjudicates == (first.annotation_id, second.annotation_id)
    assert reference.annotation_id not in reference.adjudicates
    assert len(service.annotations.all_records()) == 3


def test_disagreement_requires_independent_third_party_adjudication(tmp_path):
    service = _service(tmp_path)
    assignment = _assignment(service)
    first = service.submit(
        assignment_id=assignment.assignment_id,
        reviewer_ref="reviewer_a",
        action=EvaluationAction.ACCEPT,
    )
    second = service.submit(
        assignment_id=assignment.assignment_id,
        reviewer_ref="reviewer_b",
        action=EvaluationAction.CORRECT,
        submitted_value="Kommunstyrelsen beslutar",
    )
    assert service.assignments.by_id(assignment.assignment_id).status is (
        EvaluationReferenceStatus.AWAITING_ADJUDICATION
    )
    assert service.next_adjudication("adjudicator_c").reviewer_answers == (
        "KommunfullmÃ¤ktige beslutar",
        "Kommunstyrelsen beslutar",
    )
    assert service.annotations.verified_references() == ()
    with pytest.raises(ValueError, match="independent"):
        service.adjudicate(
            assignment_id=assignment.assignment_id,
            adjudicator_ref="reviewer_a",
            submitted_value="KommunfullmÃ¤ktige beslutar",
            method="adjudication from source crop",
        )

    reference = service.adjudicate(
        assignment_id=assignment.assignment_id,
        adjudicator_ref="adjudicator_c",
        submitted_value="KommunfullmÃ¤ktige beslutar",
        method="adjudication from source crop",
    )
    assert reference.verification_status is VerificationStatus.VERIFIED
    assert reference.adjudicates == (first.annotation_id, second.annotation_id)
    assert len(service.annotations.all_records()) == 3


def test_ai_assisted_verified_reference_is_explicit_and_excluded_by_default(tmp_path):
    service = _service(tmp_path)
    assignment = _assignment(service)
    for reviewer in assignment.reviewer_refs:
        service.submit(
            assignment_id=assignment.assignment_id,
            reviewer_ref=reviewer,
            action=EvaluationAction.ACCEPT,
            annotator_kind=(
                AnnotatorKind.AI_ASSISTED if reviewer == "reviewer_a" else AnnotatorKind.HUMAN
            ),
        )

    assert service.annotations.verified_references() == ()
    (explicitly_included,) = service.annotations.verified_references(include_ai=True)
    assert explicitly_included.annotator_kind is AnnotatorKind.AI_ASSISTED


def test_exclusion_actions_preserve_reason_without_fabricating_empty_text(tmp_path):
    service = _service(tmp_path)
    assignment = _assignment(service)
    for reviewer in assignment.reviewer_refs:
        service.submit(
            assignment_id=assignment.assignment_id,
            reviewer_ref=reviewer,
            action=EvaluationAction.BOUNDARY_INCORRECT,
        )

    records = service.annotations.all_records()
    assert records[-1].verification_status is VerificationStatus.EXCLUDED
    assert records[-1].text is None
    assert records[-1].exclusion_reason == "boundary_incorrect"
    assert service.annotations.verified_references() == ()


def test_reviewer_cannot_submit_twice_or_read_unassigned_chunk(tmp_path):
    service = _service(tmp_path)
    assignment = _assignment(service)
    service.submit(
        assignment_id=assignment.assignment_id,
        reviewer_ref="reviewer_a",
        action=EvaluationAction.ACCEPT,
    )
    with pytest.raises(ValueError, match="already submitted"):
        service.submit(
            assignment_id=assignment.assignment_id,
            reviewer_ref="reviewer_a",
            action=EvaluationAction.ACCEPT,
        )
    with pytest.raises(PermissionError, match="not assigned"):
        service.submit(
            assignment_id=assignment.assignment_id,
            reviewer_ref="reviewer_x",
            action=EvaluationAction.ACCEPT,
        )
