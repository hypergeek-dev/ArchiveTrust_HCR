from __future__ import annotations

import pytest

from archivetrust.domain.canonical.observation import CanonicalObservation
from archivetrust.domain.confidence.models import ComparisonClassification, ComparisonConfidence
from archivetrust.domain.evidence.models import Evidence, ProcessingStage
from archivetrust.domain.feedback.engine import apply_human_correction
from archivetrust.domain.feedback.models import CorrectionAction, CorrectionCategory, HumanCorrection
from archivetrust.domain.feedback.policy import FeedbackPolicy
from archivetrust.domain.ontology.base import Observation
from archivetrust.domain.ontology.payloads import HeadingPayload, PagePayload


def _policy() -> FeedbackPolicy:
    return FeedbackPolicy(feedback_policy_version=1)


def _obs(text: str) -> Observation:
    ev = Evidence.create(provider="docling", provider_version="1.0", raw_output=text, processing_stage=ProcessingStage.OCR)
    return Observation.from_evidence(provider_id="docling", provider_version="1.0", payload=HeadingPayload(text=text, level=1), evidence=(ev,))


def _canonical(observation: Observation) -> CanonicalObservation:
    return CanonicalObservation.reconcile(
        semantic_slot_id="slot-1",
        payload=observation.payload,
        contributing_observations=(observation,),
        comparison_confidence=ComparisonConfidence(
            classification=ComparisonClassification.UNCORROBORATED_SINGLE_SOURCE, magnitude=None, basis="n/a"
        ),
        clustering_basis="test",
        reconciliation_basis="single-source acceptance",
        reconciliation_sequence=0,
    )


def test_accept_supersedes_with_unchanged_payload_and_high_confidence():
    observation = _obs("Chapter 1")
    original = _canonical(observation)
    correction = HumanCorrection(
        correction_id="correction-1",
        target_canonical_observation_id=original.canonical_observation_id,
        category=CorrectionCategory.TRANSCRIPTION_ERROR,
        action=CorrectionAction.ACCEPT,
        raw_ai_output="Chapter 1",
    )
    resulting = apply_human_correction(original, correction, (observation,), _policy())

    assert resulting.payload == original.payload
    assert resulting.supersedes == original.canonical_observation_id
    assert resulting.canonical_confidence.value == _policy().accepted_confidence
    assert resulting.human_correction_ref == "correction-1"
    assert original.canonical_confidence is None  # predecessor never mutated (Article 15)


def test_edit_replaces_text_payload_content():
    observation = _obs("Chaptr 1")
    original = _canonical(observation)
    correction = HumanCorrection(
        correction_id="correction-2",
        target_canonical_observation_id=original.canonical_observation_id,
        category=CorrectionCategory.TRANSCRIPTION_ERROR,
        action=CorrectionAction.EDIT,
        raw_ai_output="Chaptr 1",
        raw_corrected_output="Chapter 1",
        rationale="OCR misread",
    )
    resulting = apply_human_correction(original, correction, (observation,), _policy())

    assert resulting.payload.text == "Chapter 1"
    assert resulting.canonical_confidence.value == _policy().edited_confidence
    assert resulting.rationale == "OCR misread"


def test_edit_requires_raw_corrected_output():
    observation = _obs("Chapter 1")
    original = _canonical(observation)
    correction = HumanCorrection(
        correction_id="correction-3",
        target_canonical_observation_id=original.canonical_observation_id,
        category=CorrectionCategory.TRANSCRIPTION_ERROR,
        action=CorrectionAction.EDIT,
        raw_ai_output="Chapter 1",
    )
    with pytest.raises(ValueError):
        apply_human_correction(original, correction, (observation,), _policy())


def test_reject_lowers_confidence_without_changing_payload():
    observation = _obs("Chapter 1")
    original = _canonical(observation)
    correction = HumanCorrection(
        correction_id="correction-4",
        target_canonical_observation_id=original.canonical_observation_id,
        category=CorrectionCategory.SPURIOUS_CONTENT,
        action=CorrectionAction.REJECT,
        raw_ai_output="Chapter 1",
    )
    resulting = apply_human_correction(original, correction, (observation,), _policy())
    assert resulting.payload == original.payload
    assert resulting.canonical_confidence.value == _policy().rejected_confidence


def test_edit_on_non_text_payload_retains_original_payload():
    ev = Evidence.create(provider="docling", provider_version="1.0", raw_output="1", processing_stage=ProcessingStage.OCR)
    observation = Observation.from_evidence(provider_id="docling", provider_version="1.0", payload=PagePayload(page_number=1), evidence=(ev,))
    original = _canonical(observation)
    correction = HumanCorrection(
        correction_id="correction-5",
        target_canonical_observation_id=original.canonical_observation_id,
        category=CorrectionCategory.METADATA_ERROR,
        action=CorrectionAction.EDIT,
        raw_ai_output="1",
        raw_corrected_output="2",
    )
    resulting = apply_human_correction(original, correction, (observation,), _policy())
    assert resulting.payload == original.payload  # PagePayload has no .text field to overwrite


def test_merge_and_split_are_recorded_but_not_structurally_executed():
    observation = _obs("Chapter 1")
    original = _canonical(observation)
    correction = HumanCorrection(
        correction_id="correction-6",
        target_canonical_observation_id=original.canonical_observation_id,
        category=CorrectionCategory.SEGMENTATION_ERROR,
        action=CorrectionAction.MERGE,
        raw_ai_output="Chapter 1",
    )
    resulting = apply_human_correction(original, correction, (observation,), _policy())
    assert "not executed" in resulting.canonical_confidence.derivation
    assert resulting.payload == original.payload


def test_mismatched_target_id_raises():
    observation = _obs("Chapter 1")
    original = _canonical(observation)
    correction = HumanCorrection(
        correction_id="correction-7",
        target_canonical_observation_id="canonical_observation_wrong",
        category=CorrectionCategory.OTHER,
        action=CorrectionAction.ACCEPT,
        raw_ai_output="Chapter 1",
    )
    with pytest.raises(ValueError):
        apply_human_correction(original, correction, (observation,), _policy())


def test_comparison_confidence_is_never_touched_by_a_correction():
    observation = _obs("Chapter 1")
    original = _canonical(observation)
    correction = HumanCorrection(
        correction_id="correction-8",
        target_canonical_observation_id=original.canonical_observation_id,
        category=CorrectionCategory.TRANSCRIPTION_ERROR,
        action=CorrectionAction.EDIT,
        raw_ai_output="Chapter 1",
        raw_corrected_output="Chapter One",
    )
    resulting = apply_human_correction(original, correction, (observation,), _policy())
    assert resulting.comparison_confidence == original.comparison_confidence
