from __future__ import annotations

import pytest
from pydantic import ValidationError

from archivetrust.domain.canonical.observation import CanonicalObservation
from archivetrust.domain.confidence.models import ComparisonClassification, ComparisonConfidence
from archivetrust.domain.evidence.models import Evidence, ProcessingStage
from archivetrust.domain.ontology.base import Observation
from archivetrust.domain.ontology.payloads import (
    HeadingPayload,
    ParagraphPayload,
)


def _evidence(provider: str, text: str, provider_version: str = "1.0") -> Evidence:
    return Evidence.create(
        provider=provider,
        provider_version=provider_version,
        raw_output=text,
        processing_stage=ProcessingStage.OCR,
    )


def test_reconcile_requires_at_least_one_contributing_observation():
    with pytest.raises(ValueError):
        CanonicalObservation.reconcile(
            semantic_slot_id="slot-1",
            payload=HeadingPayload(text="Ch 1", level=1),
            contributing_observations=(),
            comparison_confidence=ComparisonConfidence(
                classification=ComparisonClassification.UNCORROBORATED_SINGLE_SOURCE,
                magnitude=None,
                basis="n/a",
            ),
            clustering_basis="n/a",
            reconciliation_basis="n/a",
            reconciliation_sequence=0,
        )


def test_reconcile_rejects_mixed_observation_types():
    heading_obs = Observation.from_evidence(
        provider_id="docling",
        provider_version="1.0",
        payload=HeadingPayload(text="Ch 1", level=1),
        evidence=(_evidence("docling", "Ch 1"),),
    )
    para_obs = Observation.from_evidence(
        provider_id="docling",
        provider_version="1.0",
        payload=ParagraphPayload(text="body"),
        evidence=(_evidence("docling", "body"),),
    )
    with pytest.raises(ValueError):
        CanonicalObservation.reconcile(
            semantic_slot_id="slot-1",
            payload=HeadingPayload(text="Ch 1", level=1),
            contributing_observations=(heading_obs, para_obs),
            comparison_confidence=ComparisonConfidence(
                classification=ComparisonClassification.CORROBORATED, magnitude=0.9, basis="x"
            ),
            clustering_basis="geometric overlap",
            reconciliation_basis="majority agreement",
            reconciliation_sequence=0,
        )


def test_single_source_slot_uses_not_applicable_marker_never_a_low_number(
    heading_observation: Observation, single_source_confidence: ComparisonConfidence
):
    canonical = CanonicalObservation.reconcile(
        semantic_slot_id="slot-1",
        payload=heading_observation.payload,
        contributing_observations=(heading_observation,),
        comparison_confidence=single_source_confidence,
        clustering_basis="sole observation in slot",
        reconciliation_basis="single-source acceptance",
        reconciliation_sequence=0,
    )
    assert canonical.comparison_confidence.magnitude is None
    assert (
        canonical.comparison_confidence.classification
        == ComparisonClassification.UNCORROBORATED_SINGLE_SOURCE
    )


def test_canonical_observation_is_not_complete_without_canonical_confidence(
    heading_observation: Observation, single_source_confidence: ComparisonConfidence
):
    # Milestone 1 defines the shape; canonical_confidence is legitimately None until Milestone 5
    # (MILESTONE1_DOMAIN_MODEL.md S1.3). It must still exist as a field from the start.
    canonical = CanonicalObservation.reconcile(
        semantic_slot_id="slot-1",
        payload=heading_observation.payload,
        contributing_observations=(heading_observation,),
        comparison_confidence=single_source_confidence,
        clustering_basis="sole observation in slot",
        reconciliation_basis="single-source acceptance",
        reconciliation_sequence=0,
    )
    assert canonical.canonical_confidence is None
    assert "canonical_confidence" in CanonicalObservation.model_fields


def test_shared_contribution_one_observation_in_two_canonical_observations():
    # MILESTONE1_DOMAIN_MODEL.md S1.2: a single Observation may legitimately contribute to two
    # Canonical Observations (e.g. relevant to both a Paragraph and a Table cell). Each
    # contribution must be aspect-annotated so reuse is distinguishable from independent
    # corroboration. (Originally demonstrated with `NamedEntityPayload`, deleted in
    # docs/htr-migration-plan.md Stage 5 -- EXECUTED; `HeadingPayload` demonstrates the same
    # sharing/aspect-annotation principle, which is generic to any payload type.)
    paragraph_obs = Observation.from_evidence(
        provider_id="qwen2.5-vl",
        provider_version="2.5",
        payload=ParagraphPayload(text="Born in Denmark in 1990."),
        evidence=(_evidence("qwen2.5-vl", "Born in Denmark in 1990.", provider_version="2.5"),),
    )
    entity_obs = Observation.from_evidence(
        provider_id="qwen2.5-vl",
        provider_version="2.5",
        payload=HeadingPayload(text="Denmark", level=1),
        evidence=(_evidence("qwen2.5-vl", "Denmark", provider_version="2.5"),),
    )
    single_source = ComparisonConfidence(
        classification=ComparisonClassification.UNCORROBORATED_SINGLE_SOURCE,
        magnitude=None,
        basis="n/a",
    )

    canonical_paragraph = CanonicalObservation.reconcile(
        semantic_slot_id="slot-paragraph-1",
        payload=paragraph_obs.payload,
        contributing_observations=(paragraph_obs,),
        comparison_confidence=single_source,
        clustering_basis="sole paragraph observation",
        reconciliation_basis="single-source acceptance",
        reconciliation_sequence=0,
    )
    canonical_entity_in_paragraph = CanonicalObservation.reconcile(
        semantic_slot_id="slot-entity-1",
        payload=entity_obs.payload,
        contributing_observations=(entity_obs,),
        comparison_confidence=single_source,
        clustering_basis="entity co-located with paragraph slot",
        reconciliation_basis="single-source acceptance",
        reconciliation_sequence=0,
        contribution_aspects={entity_obs.observation_id: "entity within paragraph text"},
    )
    canonical_entity_in_table_cell = CanonicalObservation.reconcile(
        semantic_slot_id="slot-entity-2",
        payload=entity_obs.payload,
        contributing_observations=(entity_obs,),
        comparison_confidence=single_source,
        clustering_basis="entity co-located with table cell slot",
        reconciliation_basis="single-source acceptance",
        reconciliation_sequence=0,
        contribution_aspects={entity_obs.observation_id: "entity within table cell text"},
    )

    shared_id = entity_obs.observation_id
    assert canonical_entity_in_paragraph.contributing_observations[0].observation_id == shared_id
    assert canonical_entity_in_table_cell.contributing_observations[0].observation_id == shared_id
    assert (
        canonical_entity_in_paragraph.contributing_observations[0].aspect
        != canonical_entity_in_table_cell.contributing_observations[0].aspect
    )
    assert canonical_paragraph.canonical_observation_id != canonical_entity_in_paragraph.canonical_observation_id


def test_supersede_creates_new_version_linked_to_predecessor_never_mutating_it(
    heading_observation: Observation, single_source_confidence: ComparisonConfidence
):
    original = CanonicalObservation.reconcile(
        semantic_slot_id="slot-1",
        payload=heading_observation.payload,
        contributing_observations=(heading_observation,),
        comparison_confidence=single_source_confidence,
        clustering_basis="sole observation in slot",
        reconciliation_basis="single-source acceptance",
        reconciliation_sequence=0,
    )
    revised_payload = HeadingPayload(text="Chapter One: Origins", level=1)
    revised = original.supersede(
        payload=revised_payload,
        contributing_observations=(heading_observation,),
        comparison_confidence=single_source_confidence,
        clustering_basis="sole observation in slot",
        reconciliation_basis="human correction applied",
        human_correction_ref="correction_1",
    )

    assert revised.semantic_slot_id == original.semantic_slot_id
    assert revised.supersedes == original.canonical_observation_id
    assert revised.reconciliation_sequence == original.reconciliation_sequence + 1
    assert revised.canonical_observation_id != original.canonical_observation_id
    # original is untouched -- Constitution Article 15: supersession, never erasure
    assert original.payload.text == "Chapter 1: Origins"
    assert original.superseded_by is None


def test_canonical_observation_is_immutable(
    heading_observation: Observation, single_source_confidence: ComparisonConfidence
):
    canonical = CanonicalObservation.reconcile(
        semantic_slot_id="slot-1",
        payload=heading_observation.payload,
        contributing_observations=(heading_observation,),
        comparison_confidence=single_source_confidence,
        clustering_basis="sole observation in slot",
        reconciliation_basis="single-source acceptance",
        reconciliation_sequence=0,
    )
    with pytest.raises(ValidationError):
        canonical.reconciliation_sequence = 5  # type: ignore[misc]


def test_round_trip_serialization(
    heading_observation: Observation, single_source_confidence: ComparisonConfidence
):
    canonical = CanonicalObservation.reconcile(
        semantic_slot_id="slot-1",
        payload=heading_observation.payload,
        contributing_observations=(heading_observation,),
        comparison_confidence=single_source_confidence,
        clustering_basis="sole observation in slot",
        reconciliation_basis="single-source acceptance",
        reconciliation_sequence=0,
    )
    restored = CanonicalObservation.model_validate(canonical.model_dump())
    assert restored == canonical
