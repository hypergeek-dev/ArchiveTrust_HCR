"""ROADMAP.md Milestone 5 acceptance criterion: "Confidence scores are reproducible from stored
telemetry (replay test)." Runs the full M3(observations)->M4(comparison)->M5(confidence) chain,
emits telemetry for both stages, replays it through the Journal, and checks the replayed
CanonicalObservation's canonical_confidence matches what the engines actually computed.
"""

from __future__ import annotations

from archivetrust.application.journal import Journal
from archivetrust.domain.comparison.capability_matrix import CapabilityMatrix
from archivetrust.domain.comparison.engine import run_comparison_engine
from archivetrust.domain.comparison.policy import ReconciliationPolicy
from archivetrust.domain.comparison.telemetry import comparison_result_to_events
from archivetrust.domain.confidence.engine import apply_confidence_engine
from archivetrust.domain.confidence.policy import ConfidencePolicy
from archivetrust.domain.confidence.telemetry import confidence_changed_events
from archivetrust.domain.evidence.models import BoundingBox, Evidence, Precision, ProcessingStage
from archivetrust.domain.graph.provider_graph import ProviderObservationGraph
from archivetrust.domain.ontology.base import Observation
from archivetrust.domain.ontology.payloads import HeadingPayload
from archivetrust.domain.telemetry.events import EvidenceCreated, ObservationCreated
from archivetrust.infrastructure.storage.telemetry_sink import InMemoryTelemetrySink

DOCUMENT_REF = "archive-object-confidence-1"


def test_confidence_scores_are_reproducible_from_stored_telemetry():
    sink = InMemoryTelemetrySink()

    box_docling = BoundingBox(x0=0, y0=0, x1=100, y1=20, precision=Precision.PIXEL_ACCURATE)
    box_tesseract = BoundingBox(x0=1, y0=1, x1=99, y1=19, precision=Precision.PIXEL_ACCURATE)
    ev_docling = Evidence.create(
        provider="docling", provider_version="1.0", raw_output="Chapter 1", processing_stage=ProcessingStage.OCR,
        bounding_box=box_docling, provider_confidence=0.9,
    )
    ev_tesseract = Evidence.create(
        provider="tesseract", provider_version="1.0", raw_output="Chapter l", processing_stage=ProcessingStage.OCR,
        bounding_box=box_tesseract, provider_confidence=0.6,
    )
    obs_docling = Observation.from_evidence(
        provider_id="docling", provider_version="1.0", payload=HeadingPayload(text="Chapter 1", level=1), evidence=(ev_docling,)
    )
    obs_tesseract = Observation.from_evidence(
        provider_id="tesseract", provider_version="1.0", payload=HeadingPayload(text="Chapter l", level=1), evidence=(ev_tesseract,)
    )

    for evidence, invocation_id in ((ev_docling, "inv-docling"), (ev_tesseract, "inv-tesseract")):
        sink.append(
            EvidenceCreated(event_id=f"event-{evidence.evidence_id}", document_ref=DOCUMENT_REF, invocation_id=invocation_id, evidence=evidence)
        )
    for observation, invocation_id in ((obs_docling, "inv-docling"), (obs_tesseract, "inv-tesseract")):
        sink.append(
            ObservationCreated(
                event_id=f"event-{observation.observation_id}", document_ref=DOCUMENT_REF, invocation_id=invocation_id, observation=observation
            )
        )

    evidence_by_id = {ev_docling.evidence_id: ev_docling, ev_tesseract.evidence_id: ev_tesseract}
    observations_by_id = {obs_docling.observation_id: obs_docling, obs_tesseract.observation_id: obs_tesseract}
    docling_graph = ProviderObservationGraph(provider_id="docling", provider_version="1.0", invocation_id="inv-docling", observations=(obs_docling,))
    tesseract_graph = ProviderObservationGraph(provider_id="tesseract", provider_version="1.0", invocation_id="inv-tesseract", observations=(obs_tesseract,))

    reconciliation_policy = ReconciliationPolicy(policy_version=1)
    capability_matrix = CapabilityMatrix(matrix_version=1, entries=())
    confidence_policy = ConfidencePolicy(confidence_policy_version=1)

    comparison_result = run_comparison_engine((docling_graph, tesseract_graph), evidence_by_id, capability_matrix, reconciliation_policy)
    completed_graph = apply_confidence_engine(comparison_result.reconciled_graph, observations_by_id, confidence_policy)

    for event in comparison_result_to_events(
        comparison_result.__class__(
            reconciled_graph=completed_graph,
            bundles=comparison_result.bundles,
            alignment=comparison_result.alignment,
        ),
        document_ref=DOCUMENT_REF, policy=reconciliation_policy, capability_matrix=capability_matrix,
    ):
        sink.append(event)
    for event in confidence_changed_events(completed_graph.canonical_observations, document_ref=DOCUMENT_REF, policy=confidence_policy):
        sink.append(event)

    original_canonical = completed_graph.canonical_observations[0]

    state = Journal().replay(sink.events_for_document(DOCUMENT_REF))
    replayed_canonical = state.canonical_observation(original_canonical.canonical_observation_id)

    assert replayed_canonical.canonical_confidence == original_canonical.canonical_confidence
    evolution = state.confidence_evolution(original_canonical.canonical_observation_id)
    assert len(evolution) == 1
    assert evolution[0].new_value == original_canonical.canonical_confidence.value
    assert evolution[0].confidence_policy_version == 1
