from __future__ import annotations

from archivetrust.domain.comparison.capability_matrix import CapabilityMatrix
from archivetrust.domain.comparison.engine import run_comparison_engine
from archivetrust.domain.comparison.policy import ReconciliationPolicy
from archivetrust.domain.comparison.telemetry import comparison_result_to_events
from archivetrust.domain.evidence.models import BoundingBox, Evidence, Precision, ProcessingStage
from archivetrust.domain.graph.provider_graph import ProviderObservationGraph
from archivetrust.domain.ontology.base import Observation
from archivetrust.domain.ontology.payloads import HeadingPayload
from archivetrust.domain.telemetry.events import AgreementCalculated, CanonicalDecisionCreated


def test_comparison_result_carries_policy_and_matrix_versions_on_every_event():
    box = BoundingBox(x0=0, y0=0, x1=100, y1=20, precision=Precision.PIXEL_ACCURATE)
    ev = Evidence.create(
        provider="docling", provider_version="1.0", raw_output="Chapter 1", processing_stage=ProcessingStage.OCR, bounding_box=box
    )
    obs = Observation.from_evidence(
        provider_id="docling", provider_version="1.0", payload=HeadingPayload(text="Chapter 1", level=1), evidence=(ev,)
    )
    graph = ProviderObservationGraph(provider_id="docling", provider_version="1.0", invocation_id="inv-1", observations=(obs,))
    policy = ReconciliationPolicy(policy_version=7)
    matrix = CapabilityMatrix(matrix_version=3, entries=())

    result = run_comparison_engine((graph,), {ev.evidence_id: ev}, matrix, policy)
    events = comparison_result_to_events(result, document_ref="doc-1", policy=policy, capability_matrix=matrix)

    assert events
    for event in events:
        assert event.reconciliation_policy_version == 7
        assert event.capability_matrix_version == 3

    decision_events = [e for e in events if isinstance(e, CanonicalDecisionCreated)]
    agreement_events = [e for e in events if isinstance(e, AgreementCalculated)]
    assert len(decision_events) == 1
    assert len(agreement_events) == 1
