from __future__ import annotations

import pytest
from pydantic import ValidationError

from archivetrust.domain.canonical.observation import CanonicalGraphEdge, CanonicalObservation
from archivetrust.domain.confidence.models import ComparisonClassification, ComparisonConfidence
from archivetrust.domain.evidence.models import Evidence, ProcessingStage
from archivetrust.domain.graph.reconciled_graph import ReconciledObservationGraph
from archivetrust.domain.ontology.base import Observation
from archivetrust.domain.ontology.payloads import HeadingPayload, ParagraphPayload


def _obs(payload, text: str) -> Observation:
    ev = Evidence.create(
        provider="docling", provider_version="1.0", raw_output=text, processing_stage=ProcessingStage.OCR
    )
    return Observation.from_evidence(provider_id="docling", provider_version="1.0", payload=payload, evidence=(ev,))


def _single_source() -> ComparisonConfidence:
    return ComparisonConfidence(
        classification=ComparisonClassification.UNCORROBORATED_SINGLE_SOURCE, magnitude=None, basis="n/a"
    )


def test_graph_contains_only_canonical_observation_nodes():
    heading = _obs(HeadingPayload(text="Ch 1", level=1), "Ch 1")
    canonical = CanonicalObservation.reconcile(
        semantic_slot_id="slot-1",
        payload=heading.payload,
        contributing_observations=(heading,),
        comparison_confidence=_single_source(),
        clustering_basis="sole observation",
        reconciliation_basis="single-source acceptance",
        reconciliation_sequence=0,
    )
    graph = ReconciledObservationGraph(reconciliation_sequence=0, canonical_observations=(canonical,))
    assert graph.canonical_observations == (canonical,)


def test_graph_rejects_dangling_edges():
    heading = _obs(HeadingPayload(text="Ch 1", level=1), "Ch 1")
    canonical = CanonicalObservation.reconcile(
        semantic_slot_id="slot-1",
        payload=heading.payload,
        contributing_observations=(heading,),
        comparison_confidence=_single_source(),
        clustering_basis="sole observation",
        reconciliation_basis="single-source acceptance",
        reconciliation_sequence=0,
    ).model_copy(
        update={"child_observations": (CanonicalGraphEdge(target_canonical_observation_id="canonical_observation_missing"),)}
    )
    with pytest.raises(ValidationError):
        ReconciledObservationGraph(reconciliation_sequence=0, canonical_observations=(canonical,))


def test_graph_rejects_cycles():
    heading = _obs(HeadingPayload(text="A", level=1), "A")
    para = _obs(ParagraphPayload(text="B"), "B")
    canon_a = CanonicalObservation.reconcile(
        semantic_slot_id="slot-a",
        payload=heading.payload,
        contributing_observations=(heading,),
        comparison_confidence=_single_source(),
        clustering_basis="sole observation",
        reconciliation_basis="single-source acceptance",
        reconciliation_sequence=0,
    )
    canon_b = CanonicalObservation.reconcile(
        semantic_slot_id="slot-b",
        payload=para.payload,
        contributing_observations=(para,),
        comparison_confidence=_single_source(),
        clustering_basis="sole observation",
        reconciliation_basis="single-source acceptance",
        reconciliation_sequence=0,
    )
    canon_a = canon_a.model_copy(
        update={"child_observations": (CanonicalGraphEdge(target_canonical_observation_id=canon_b.canonical_observation_id),)}
    )
    canon_b = canon_b.model_copy(
        update={"child_observations": (CanonicalGraphEdge(target_canonical_observation_id=canon_a.canonical_observation_id),)}
    )
    with pytest.raises(ValidationError):
        ReconciledObservationGraph(
            reconciliation_sequence=0, canonical_observations=(canon_a, canon_b)
        )


def test_roots_returns_nodes_without_a_parent():
    heading = _obs(HeadingPayload(text="Ch 1", level=1), "Ch 1")
    para = _obs(ParagraphPayload(text="body"), "body")
    canon_heading = CanonicalObservation.reconcile(
        semantic_slot_id="slot-heading",
        payload=heading.payload,
        contributing_observations=(heading,),
        comparison_confidence=_single_source(),
        clustering_basis="sole observation",
        reconciliation_basis="single-source acceptance",
        reconciliation_sequence=0,
    )
    canon_para = CanonicalObservation.reconcile(
        semantic_slot_id="slot-para",
        payload=para.payload,
        contributing_observations=(para,),
        comparison_confidence=_single_source(),
        clustering_basis="sole observation",
        reconciliation_basis="single-source acceptance",
        reconciliation_sequence=0,
    )
    canon_heading = canon_heading.model_copy(
        update={"child_observations": (CanonicalGraphEdge(target_canonical_observation_id=canon_para.canonical_observation_id),)}
    )
    canon_para = canon_para.model_copy(
        update={"parent_observations": (CanonicalGraphEdge(target_canonical_observation_id=canon_heading.canonical_observation_id),)}
    )

    graph = ReconciledObservationGraph(
        reconciliation_sequence=0, canonical_observations=(canon_heading, canon_para)
    )
    assert graph.roots() == (canon_heading,)
