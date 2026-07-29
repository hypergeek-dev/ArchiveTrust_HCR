"""Telemetry-stream fixtures for Human Review tests.

Builds a real Trust Engine telemetry stream (`InMemoryTelemetrySink`) for one document, with slots
of configurable agreement classification, canonical confidence, and per-provider bounding-box
geometry — exercising the Review service against the exact event shapes it reads in production.
"""

from __future__ import annotations

from archivetrust.domain.canonical.observation import CanonicalObservation
from archivetrust.domain.document.canonical_document import CanonicalDocument
from archivetrust.domain.confidence.models import (
    CanonicalConfidence,
    ComparisonClassification,
    ComparisonConfidence,
)
from archivetrust.domain.evidence.models import (
    BoundingBox,
    Evidence,
    Precision,
    ProcessingStage,
)
from archivetrust.domain.ontology.base import Observation
from archivetrust.domain.ontology.payloads import HeadingPayload
from archivetrust.domain.ontology.payloads.base import ObservationPayload
from archivetrust.domain.shared.ids import new_id
from archivetrust.domain.telemetry.events import (
    CanonicalDecisionCreated,
    CanonicalDocumentCreated,
    EvidenceCreated,
    ObservationCreated,
    ProviderObservationAttempted,
    stamp_recorded_at,
)
from archivetrust.infrastructure.storage.telemetry_sink import InMemoryTelemetrySink


def _observation(
    provider: str, payload: ObservationPayload, *, with_bbox: bool
) -> tuple[Observation, tuple[Evidence, ...]]:
    bbox = (
        BoundingBox(x0=10.0, y0=20.0, x1=200.0, y1=48.0, precision=Precision.PIXEL_ACCURATE)
        if with_bbox
        else None
    )
    evidence = Evidence.create(
        provider=provider,
        provider_version="1.0",
        raw_output=f"{provider}:{getattr(payload, 'text', payload.observation_type.value)}",
        processing_stage=ProcessingStage.RAW,
        page=1,
        bounding_box=bbox,
        provider_confidence=0.8,
    )
    observation = Observation.from_evidence(
        provider_id=provider,
        provider_version="1.0",
        payload=payload,
        evidence=(evidence,),
    )
    return observation, (evidence,)


def emit_slot(
    sink: InMemoryTelemetrySink,
    *,
    document_ref: str,
    canonical_payload: ObservationPayload,
    provider_payloads: tuple[tuple[str, ObservationPayload], ...],
    classification: ComparisonClassification,
    canonical_confidence: float | None = None,
    with_bbox: bool = True,
    reconciliation_basis_code: str | None = None,
) -> CanonicalObservation:
    """Emit one reconciled slot.

    `provider_payloads` is (provider_id, payload) per contributing provider — pass differing
    payloads to model a CONTESTED slot, one provider for a single-source slot. `classification`
    sets the recorded ComparisonClassification directly, so tests control the triage signal.
    """
    built = tuple(
        _observation(provider, payload, with_bbox=with_bbox)
        for provider, payload in provider_payloads
    )
    observations = tuple(obs for obs, _ in built)
    for (provider, _), (obs, evidence_records) in zip(provider_payloads, built):
        invocation_id = new_id("invocation")
        sink.append(
            ProviderObservationAttempted(
                event_id=new_id("event"),
                document_ref=document_ref,
                provider_id=provider,
                provider_version="1.0",
                invocation_id=invocation_id,
            )
        )
        for evidence in evidence_records:
            sink.append(
                EvidenceCreated(
                    event_id=new_id("event"),
                    document_ref=document_ref,
                    invocation_id=invocation_id,
                    evidence=evidence,
                )
            )
        sink.append(
            ObservationCreated(
                event_id=new_id("event"),
                document_ref=document_ref,
                invocation_id=invocation_id,
                observation=obs,
            )
        )

    if classification == ComparisonClassification.UNCORROBORATED_SINGLE_SOURCE:
        comparison = ComparisonConfidence(
            classification=classification, magnitude=None, basis="one provider attempted this slot"
        )
    else:
        comparison = ComparisonConfidence(
            classification=classification, magnitude=0.5, basis="test-fixture agreement"
        )

    confidence = (
        CanonicalConfidence(value=canonical_confidence, derivation="test fixture")
        if canonical_confidence is not None
        else None
    )

    canonical = CanonicalObservation.reconcile(
        semantic_slot_id=new_id("slot"),
        payload=canonical_payload,
        contributing_observations=observations,
        comparison_confidence=comparison,
        canonical_confidence=confidence,
        clustering_basis="type+geometry",
        reconciliation_basis="test fixture",
        reconciliation_basis_code=reconciliation_basis_code,
        reconciliation_sequence=0,
    )
    sink.append(
        stamp_recorded_at(
            CanonicalDecisionCreated(
                event_id=new_id("event"),
                document_ref=document_ref,
                canonical_observation=canonical,
            )
        )
    )
    return canonical


def emit_document_snapshot(
    sink: InMemoryTelemetrySink,
    *,
    document_ref: str,
    archive_object_ref: str | None = None,
) -> CanonicalDocument:
    """Assemble a test document from every initial canonical decision emitted so far."""

    existing_events = tuple(sink.events_for_document(document_ref))
    decisions = [
        event.canonical_observation
        for event in existing_events
        if isinstance(event, CanonicalDecisionCreated)
    ]
    snapshots = [
        event.canonical_document
        for event in existing_events
        if isinstance(event, CanonicalDocumentCreated)
    ]
    previous = snapshots[-1] if snapshots else None
    document = CanonicalDocument(
        logical_document_id=(previous.logical_document_id if previous else new_id("document")),
        document_snapshot_id=new_id("document_snapshot"),
        archive_object_ref=(
            previous.archive_object_ref if previous else archive_object_ref or document_ref
        ),
        contained_observations=tuple(
            decision.canonical_observation_id for decision in decisions
        ),
        ontology_version=max(decision.ontology_version for decision in decisions),
        document_version=(previous.document_version + 1 if previous else 0),
        reassembly_trigger="review_test_fixture",
        supersedes=(previous.document_snapshot_id if previous else None),
    )
    sink.append(
        stamp_recorded_at(
            CanonicalDocumentCreated(
                event_id=new_id("event"),
                document_ref=document_ref,
                canonical_document=document,
            )
        )
    )
    return document


def heading(text: str, level: int = 1) -> HeadingPayload:
    return HeadingPayload(text=text, level=level)
