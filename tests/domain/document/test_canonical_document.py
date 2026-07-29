from __future__ import annotations

import pytest
from pydantic import ValidationError

from archivetrust.domain.canonical.observation import CanonicalGraphEdge, CanonicalObservation
from archivetrust.domain.confidence.models import ComparisonClassification, ComparisonConfidence
from archivetrust.domain.document.canonical_document import CanonicalDocument
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


def _graph_with_hierarchy() -> ReconciledObservationGraph:
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
    return ReconciledObservationGraph(
        reconciliation_sequence=0, canonical_observations=(canon_heading, canon_para)
    )


def test_assemble_only_references_root_canonical_observations_by_id():
    graph = _graph_with_hierarchy()
    doc = CanonicalDocument.assemble(
        reconciled_graph=graph, archive_object_ref="archive-1", reassembly_trigger="initial_assembly"
    )
    assert doc.contained_observations == (graph.roots()[0].canonical_observation_id,)


def test_assemble_has_no_public_constructor_path_around_the_reconciled_graph():
    # Article 12: the Canonical Document has exactly one legal source. There is no way to
    # construct a valid CanonicalDocument without first having a ReconciledObservationGraph --
    # attempting to hand-construct one with fabricated ids for contained_observations is the
    # closest thing to a bypass, and it is still only ever a reference, never raw provider data.
    with pytest.raises(ValidationError):
        CanonicalDocument(
            logical_document_id="doc-1",
            document_snapshot_id="snapshot-1",
            archive_object_ref="archive-1",
            contained_observations=(),
            ontology_version=1,
            document_version=0,
            reassembly_trigger="initial_assembly",
        )


def test_reassembly_produces_a_new_snapshot_linked_to_predecessor():
    graph = _graph_with_hierarchy()
    first = CanonicalDocument.assemble(
        reconciled_graph=graph, archive_object_ref="archive-1", reassembly_trigger="initial_assembly"
    )
    reassembled_graph = graph.model_copy(update={"reconciliation_sequence": 1})
    second = CanonicalDocument.assemble(
        reconciled_graph=reassembled_graph,
        archive_object_ref="archive-1",
        reassembly_trigger="reassembled_after_supersession",
        logical_document_id=first.logical_document_id,
        supersedes=first.document_snapshot_id,
    )
    assert second.logical_document_id == first.logical_document_id
    assert second.supersedes == first.document_snapshot_id
    assert second.document_snapshot_id != first.document_snapshot_id
    assert first.superseded_by is None  # immutable predecessor, never mutated


def test_canonical_document_is_immutable():
    graph = _graph_with_hierarchy()
    doc = CanonicalDocument.assemble(
        reconciled_graph=graph, archive_object_ref="archive-1", reassembly_trigger="initial_assembly"
    )
    with pytest.raises(ValidationError):
        doc.document_version = 5  # type: ignore[misc]


def test_round_trip_serialization():
    graph = _graph_with_hierarchy()
    doc = CanonicalDocument.assemble(
        reconciled_graph=graph, archive_object_ref="archive-1", reassembly_trigger="initial_assembly"
    )
    restored = CanonicalDocument.model_validate(doc.model_dump())
    assert restored == doc
