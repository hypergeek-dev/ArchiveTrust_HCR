from __future__ import annotations

import pytest
from pydantic import ValidationError

from archivetrust.domain.evidence.models import Evidence, ProcessingStage
from archivetrust.domain.ontology.base import Observation, RelatedObservationLink
from archivetrust.domain.ontology.payloads import HeadingPayload, ParagraphPayload
from archivetrust.domain.ontology.types import ObservationType
from archivetrust.domain.shared.versioning import CURRENT_ONTOLOGY_VERSION, CURRENT_SCHEMA_VERSION


def test_observation_requires_at_least_one_evidence_record(heading_evidence: Evidence):
    obs = Observation.from_evidence(
        provider_id="docling",
        provider_version="1.0",
        payload=HeadingPayload(text="Chapter 1", level=1),
        evidence=(heading_evidence,),
    )
    assert obs.evidence_ids == (heading_evidence.evidence_id,)


def test_observation_from_evidence_rejects_empty_evidence():
    with pytest.raises(ValueError):
        Observation.from_evidence(
            provider_id="docling",
            provider_version="1.0",
            payload=HeadingPayload(text="x", level=1),
            evidence=(),
        )


def test_observation_type_must_match_payload_type(heading_evidence: Evidence):
    with pytest.raises(ValidationError):
        Observation(
            observation_id="observation_1",
            provider_id="docling",
            provider_version="1.0",
            observation_type=ObservationType.PARAGRAPH,
            payload=HeadingPayload(text="x", level=1),
            evidence_ids=(heading_evidence.evidence_id,),
            provider_confidence=None,
            ontology_version=CURRENT_ONTOLOGY_VERSION,
            schema_version=CURRENT_SCHEMA_VERSION,
        )


def test_provider_confidence_is_never_invented_when_evidence_has_none():
    ev = Evidence.create(
        provider="tesseract", provider_version="5.3", raw_output="x", processing_stage=ProcessingStage.OCR
    )
    obs = Observation.from_evidence(
        provider_id="tesseract",
        provider_version="5.3",
        payload=ParagraphPayload(text="x"),
        evidence=(ev,),
    )
    assert obs.provider_confidence is None


def test_from_evidence_populates_scope_automatically():
    ev = Evidence.create(
        provider="docling", provider_version="1.0", raw_output="x", processing_stage=ProcessingStage.OCR
    )
    obs = Observation.from_evidence(
        provider_id="docling",
        provider_version="1.0",
        payload=ParagraphPayload(text="A short paragraph."),
        evidence=(ev,),
    )
    assert obs.scope is not None
    assert obs.scope.character_count == len("A short paragraph.")
    assert obs.scope.has_bounding_box is False


def test_direct_construction_without_scope_still_works(heading_evidence: Evidence):
    """Backward compatibility: historical telemetry recorded before `scope` existed has no such
    key in its serialized JSON; the field must default to None rather than fail validation."""
    obs = Observation(
        observation_id="observation_1",
        provider_id="docling",
        provider_version="1.0",
        observation_type=ObservationType.HEADING,
        payload=HeadingPayload(text="x", level=1),
        evidence_ids=(heading_evidence.evidence_id,),
        provider_confidence=None,
        ontology_version=CURRENT_ONTOLOGY_VERSION,
        schema_version=CURRENT_SCHEMA_VERSION,
    )
    assert obs.scope is None


def test_provider_confidence_is_read_through_when_evidence_agrees():
    ev1 = Evidence.create(
        provider="tesseract",
        provider_version="5.3",
        raw_output="x",
        processing_stage=ProcessingStage.OCR,
        provider_confidence=0.8,
    )
    ev2 = Evidence.create(
        provider="tesseract",
        provider_version="5.3",
        raw_output="y",
        processing_stage=ProcessingStage.OCR,
        provider_confidence=0.8,
    )
    obs = Observation.from_evidence(
        provider_id="tesseract",
        provider_version="5.3",
        payload=ParagraphPayload(text="xy"),
        evidence=(ev1, ev2),
    )
    assert obs.provider_confidence is not None
    assert obs.provider_confidence.value == 0.8


def test_provider_confidence_left_unset_when_evidence_disagrees():
    ev1 = Evidence.create(
        provider="tesseract",
        provider_version="5.3",
        raw_output="x",
        processing_stage=ProcessingStage.OCR,
        provider_confidence=0.8,
    )
    ev2 = Evidence.create(
        provider="tesseract",
        provider_version="5.3",
        raw_output="y",
        processing_stage=ProcessingStage.OCR,
        provider_confidence=0.3,
    )
    obs = Observation.from_evidence(
        provider_id="tesseract",
        provider_version="5.3",
        payload=ParagraphPayload(text="xy"),
        evidence=(ev1, ev2),
    )
    assert obs.provider_confidence is None


def test_from_evidence_rejects_evidence_from_a_different_provider(heading_evidence: Evidence):
    with pytest.raises(ValueError):
        Observation.from_evidence(
            provider_id="qwen2.5-vl",
            provider_version="2.5",
            payload=HeadingPayload(text="x", level=1),
            evidence=(heading_evidence,),
        )


def test_observation_carries_ontology_and_schema_version(heading_observation: Observation):
    assert heading_observation.ontology_version == CURRENT_ONTOLOGY_VERSION
    assert heading_observation.schema_version == CURRENT_SCHEMA_VERSION


def test_related_observations_carry_a_typed_relation(heading_observation: Observation, paragraph_observation: Observation):
    linked = heading_observation.model_copy(
        update={
            "related_observations": (
                RelatedObservationLink(
                    target_observation_id=paragraph_observation.observation_id,
                    relation_type="precedes",
                ),
            )
        }
    )
    assert linked.related_observations[0].relation_type == "precedes"


def test_round_trip_serialization_preserves_polymorphic_payload(heading_observation: Observation):
    restored = Observation.model_validate(heading_observation.model_dump())
    assert restored == heading_observation
    assert isinstance(restored.payload, HeadingPayload)
    assert restored.payload.level == 1

    restored_json = Observation.model_validate_json(heading_observation.model_dump_json())
    assert restored_json == heading_observation


def test_observation_is_immutable(heading_observation: Observation):
    with pytest.raises(ValidationError):
        heading_observation.provider_id = "someone-else"  # type: ignore[misc]
