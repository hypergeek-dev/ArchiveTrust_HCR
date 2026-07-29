"""Telemetry-stream fixtures for Learning Platform tests.

Builds a real `InMemoryTelemetrySink` populated with Trust Engine telemetry (ObservationCreated,
CanonicalDecisionCreated, HumanCorrectionSubmitted, ProviderObservationAttempted, EvidenceRejected)
so the analyzers are exercised against the exact event shapes they read in production -- never a
hand-rolled stand-in for the stream.
"""

from __future__ import annotations

from archivetrust.domain.canonical.observation import CanonicalObservation
from archivetrust.domain.confidence.models import (
    ComparisonClassification,
    ComparisonConfidence,
)
from archivetrust.domain.evidence.models import Evidence, ProcessingStage
from archivetrust.domain.ontology.base import Observation
from archivetrust.domain.ontology.payloads import HeadingPayload
from archivetrust.domain.ontology.payloads.base import ObservationPayload
from archivetrust.domain.ontology.types import ObservationType
from archivetrust.domain.shared.ids import new_id
from archivetrust.domain.telemetry.events import (
    CanonicalDecisionCreated,
    EvidenceRejected,
    HumanCorrectionSubmitted,
    ObservationCreated,
    ProviderObservationAttempted,
)
from archivetrust.infrastructure.storage.telemetry_sink import InMemoryTelemetrySink


def _observation(provider: str, payload: ObservationPayload) -> Observation:
    evidence = Evidence.create(
        provider=provider,
        provider_version="1.0",
        raw_output=f"{provider}:{getattr(payload, 'text', payload.observation_type.value)}",
        processing_stage=ProcessingStage.RAW,
        page=1,
        provider_confidence=0.8,
    )
    return Observation.from_evidence(
        provider_id=provider,
        provider_version="1.0",
        payload=payload,
        evidence=(evidence,),
    )


def emit_slot(
    sink: InMemoryTelemetrySink,
    *,
    document_ref: str,
    payload: ObservationPayload,
    providers: tuple[str, ...],
    reconciliation_sequence: int = 0,
) -> CanonicalObservation:
    """Emit one reconciled semantic slot: one Observation per provider plus the Canonical
    Observation reconciled from them. Multi-provider slots are CORROBORATED; single-provider slots
    are UNCORROBORATED_SINGLE_SOURCE, matching how the Comparison Engine classifies them.
    """
    observations = tuple(_observation(p, payload) for p in providers)
    for provider, obs in zip(providers, observations):
        sink.append(
            ProviderObservationAttempted(
                event_id=new_id("event"),
                document_ref=document_ref,
                provider_id=provider,
                provider_version="1.0",
                invocation_id=new_id("invocation"),
            )
        )
        sink.append(
            ObservationCreated(
                event_id=new_id("event"),
                document_ref=document_ref,
                invocation_id=new_id("invocation"),
                observation=obs,
            )
        )

    if len(providers) == 1:
        comparison = ComparisonConfidence(
            classification=ComparisonClassification.UNCORROBORATED_SINGLE_SOURCE,
            magnitude=None,
            basis="only one provider attempted this slot",
        )
    else:
        comparison = ComparisonConfidence(
            classification=ComparisonClassification.CORROBORATED,
            magnitude=0.9,
            basis=f"{len(providers)} providers agreed",
        )

    canonical = CanonicalObservation.reconcile(
        semantic_slot_id=new_id("slot"),
        payload=payload,
        contributing_observations=observations,
        comparison_confidence=comparison,
        clustering_basis="type+geometry",
        reconciliation_basis="capability-weighted election",
        reconciliation_sequence=reconciliation_sequence,
    )
    sink.append(
        CanonicalDecisionCreated(
            event_id=new_id("event"),
            document_ref=document_ref,
            canonical_observation=canonical,
        )
    )
    return canonical


def emit_correction(
    sink: InMemoryTelemetrySink,
    *,
    document_ref: str,
    target: CanonicalObservation,
    action: str,
    category: str = "transcription_error",
) -> str:
    correction_id = new_id("correction")
    sink.append(
        HumanCorrectionSubmitted(
            event_id=new_id("event"),
            document_ref=document_ref,
            correction_id=correction_id,
            target_canonical_observation_id=target.canonical_observation_id,
            category=category,
            action=action,
            raw_ai_output=getattr(target.payload, "text", ""),
            raw_corrected_output="corrected",
        )
    )
    return correction_id


def emit_evidence_rejection(
    sink: InMemoryTelemetrySink, *, document_ref: str, provider: str
) -> None:
    sink.append(
        EvidenceRejected(
            event_id=new_id("event"),
            document_ref=document_ref,
            provider_id=provider,
            provider_version="1.0",
            invocation_id=new_id("invocation"),
            processing_stage=ProcessingStage.RAW,
            raw_output="{malformed",
            rejection_reason="invalid JSON",
        )
    )


def heading(text: str, level: int = 1) -> HeadingPayload:
    return HeadingPayload(text=text, level=level)
