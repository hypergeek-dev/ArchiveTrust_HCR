from __future__ import annotations

from archivetrust.domain.canonical.observation import CanonicalObservation
from archivetrust.domain.confidence.engine import apply_confidence_engine
from archivetrust.domain.confidence.policy import ConfidencePolicy
from archivetrust.domain.confidence.models import ComparisonClassification, ComparisonConfidence
from archivetrust.domain.confidence.telemetry import confidence_changed_events
from archivetrust.domain.evidence.models import Evidence, ProcessingStage
from archivetrust.domain.graph.reconciled_graph import ReconciledObservationGraph
from archivetrust.domain.ontology.base import Observation
from archivetrust.domain.ontology.payloads import HeadingPayload
from archivetrust.domain.telemetry.events import ConfidenceLevel


def _policy() -> ConfidencePolicy:
    return ConfidencePolicy(confidence_policy_version=5)


def _obs() -> Observation:
    ev = Evidence.create(provider="docling", provider_version="1.0", raw_output="Chapter 1", processing_stage=ProcessingStage.OCR)
    return Observation.from_evidence(provider_id="docling", provider_version="1.0", payload=HeadingPayload(text="Chapter 1", level=1), evidence=(ev,))


def test_confidence_changed_events_carry_the_confidence_policy_version():
    docling = _obs()
    comparison = ComparisonConfidence(
        classification=ComparisonClassification.UNCORROBORATED_SINGLE_SOURCE, magnitude=None, basis="n/a"
    )
    canonical = CanonicalObservation.reconcile(
        semantic_slot_id="slot-1", payload=docling.payload, contributing_observations=(docling,),
        comparison_confidence=comparison, clustering_basis="test", reconciliation_basis="test", reconciliation_sequence=0,
    )
    graph = ReconciledObservationGraph(reconciliation_sequence=0, canonical_observations=(canonical,))
    completed = apply_confidence_engine(graph, {docling.observation_id: docling}, _policy())

    events = confidence_changed_events(completed.canonical_observations, document_ref="doc-1", policy=_policy())
    assert len(events) == 1
    event = events[0]
    assert event.level == ConfidenceLevel.CANONICAL
    assert event.confidence_policy_version == 5
    assert event.new_value == completed.canonical_observations[0].canonical_confidence.value
    assert event.subject_id == canonical.canonical_observation_id


def test_no_event_emitted_for_incomplete_canonical_observations():
    docling = _obs()
    comparison = ComparisonConfidence(
        classification=ComparisonClassification.UNCORROBORATED_SINGLE_SOURCE, magnitude=None, basis="n/a"
    )
    canonical = CanonicalObservation.reconcile(
        semantic_slot_id="slot-1", payload=docling.payload, contributing_observations=(docling,),
        comparison_confidence=comparison, clustering_basis="test", reconciliation_basis="test", reconciliation_sequence=0,
    )
    # canonical_confidence is still None here -- Milestone 4 output, before Milestone 5 runs.
    events = confidence_changed_events((canonical,), document_ref="doc-1", policy=_policy())
    assert events == ()
