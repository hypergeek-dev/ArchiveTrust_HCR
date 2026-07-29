"""Tests for `review.regenerate.regenerate_review_queue` (Review Packet Integrity Audit,
2026-07-14): rebuilding a document's review queue purely from persisted Evidence/Observation
telemetry, re-running Comparison + Confidence, never a provider.
"""

from __future__ import annotations

import pytest

from archivetrust.review.regenerate import RegenerationError, regenerate_review_queue
from archivetrust.domain.canonical.observation import CanonicalObservation
from archivetrust.domain.comparison.capability_matrix_data import production_capability_matrix
from archivetrust.domain.comparison.policy import ReconciliationPolicy
from archivetrust.domain.confidence.models import CanonicalConfidence, ComparisonClassification, ComparisonConfidence
from archivetrust.domain.confidence.policy import ConfidencePolicy
from archivetrust.domain.evidence.models import BoundingBox, Evidence, Precision, ProcessingStage
from archivetrust.domain.ontology.base import Observation
from archivetrust.domain.ontology.payloads import HeadingPayload, ParagraphPayload
from archivetrust.domain.ontology.payloads.base import ObservationPayload
from archivetrust.domain.shared.ids import new_id
from archivetrust.domain.telemetry.events import (
    CanonicalDecisionCreated,
    EvidenceCreated,
    ObservationCreated,
    ProviderInvocationOutcome,
    ProviderObservationAttempted,
)

DOC = "doc_1"
ARCHIVE = "archive_object_1"


def _policy() -> ReconciliationPolicy:
    return ReconciliationPolicy(policy_version=1)


def _confidence_policy() -> ConfidencePolicy:
    return ConfidencePolicy(confidence_policy_version=1)


def _emit_observation(
    events: list,
    *,
    provider: str,
    payload: ObservationPayload,
    bbox: BoundingBox | None,
    page: int = 1,
):
    evidence = Evidence.create(
        provider=provider,
        provider_version="1.0",
        raw_output=getattr(payload, "text", None) or payload.observation_type.value,
        processing_stage=ProcessingStage.OCR,
        page=page,
        bounding_box=bbox,
        provider_confidence=0.8,
    )
    observation = Observation.from_evidence(
        provider_id=provider, provider_version="1.0", payload=payload, evidence=(evidence,)
    )
    invocation_id = new_id("invocation")
    events.append(
        ProviderObservationAttempted(
            event_id=new_id("event"),
            document_ref=DOC,
            provider_id=provider,
            provider_version="1.0",
            invocation_id=invocation_id,
            outcome=ProviderInvocationOutcome.PRODUCED_OBSERVATIONS,
            observation_count=1,
        )
    )
    events.append(
        EvidenceCreated(event_id=new_id("event"), document_ref=DOC, invocation_id=invocation_id, evidence=evidence)
    )
    events.append(
        ObservationCreated(
            event_id=new_id("event"), document_ref=DOC, invocation_id=invocation_id, observation=observation
        )
    )
    return observation


_PIXEL_BOX = BoundingBox(x0=0, y0=0, x1=100, y1=20, precision=Precision.PIXEL_ACCURATE)


def test_regenerates_review_queue_from_evidence_and_observations_only() -> None:
    """No CanonicalDecisionCreated/AgreementCalculated/AlignmentAttempted event is ever supplied
    -- only what a live provider run actually produces (attempts, Evidence, Observations). The
    review queue must still come out correct, proving Comparison+Confidence genuinely re-run
    rather than requiring a prior engine pass's output.
    """
    events: list = []
    _emit_observation(events, provider="docling", payload=HeadingPayload(text="Chapter One", level=1), bbox=_PIXEL_BOX)
    _emit_observation(
        events,
        provider="tesseract_layoutparser",
        payload=HeadingPayload(text="Chapter Two", level=1),
        bbox=BoundingBox(x0=2, y0=1, x1=98, y1=19, precision=Precision.PIXEL_ACCURATE),
    )

    packets = regenerate_review_queue(
        events,
        document_ref=DOC,
        archive_object_ref=ARCHIVE,
        reconciliation_policy=_policy(),
        capability_matrix=production_capability_matrix(),
        confidence_policy=_confidence_policy(),
    )

    assert len(packets) == 1
    packet = packets[0]
    assert packet.agreement.classification == ComparisonClassification.CONTESTED
    values = {c.value for c in packet.candidates}
    assert values == {"Chapter One", "Chapter Two"}


def test_regeneration_discards_a_stale_prior_canonical_decision() -> None:
    """A CanonicalDecisionCreated left over from a previous (e.g. pre-fix) engine run must never
    leak into the regenerated queue -- proves the prior Canonical layer is genuinely discarded,
    not merely appended to.
    """
    events: list = []
    _emit_observation(events, provider="docling", payload=HeadingPayload(text="Chapter 1", level=1), bbox=_PIXEL_BOX)
    _emit_observation(
        events,
        provider="tesseract_layoutparser",
        payload=HeadingPayload(text="Chapter I", level=1),
        bbox=BoundingBox(x0=2, y0=1, x1=98, y1=19, precision=Precision.PIXEL_ACCURATE),
    )

    stale_observation = Observation.from_evidence(
        provider_id="docling",
        provider_version="1.0",
        payload=HeadingPayload(text="A completely unrelated stale reading", level=1),
        evidence=(
            Evidence.create(
                provider="docling",
                provider_version="1.0",
                raw_output="stale",
                processing_stage=ProcessingStage.OCR,
                page=1,
                bounding_box=_PIXEL_BOX,
            ),
        ),
    )
    stale_canonical = CanonicalObservation.reconcile(
        semantic_slot_id="stale-slot",
        payload=stale_observation.payload,
        contributing_observations=(stale_observation,),
        comparison_confidence=ComparisonConfidence(
            classification=ComparisonClassification.UNCORROBORATED_SINGLE_SOURCE,
            magnitude=None,
            basis="stale prior run",
        ),
        canonical_confidence=CanonicalConfidence(value=0.99, derivation="stale prior run"),
        clustering_basis="stale prior run",
        reconciliation_basis="stale prior run",
        reconciliation_sequence=0,
    )
    events.append(
        CanonicalDecisionCreated(event_id=new_id("event"), document_ref=DOC, canonical_observation=stale_canonical)
    )

    packets = regenerate_review_queue(
        events,
        document_ref=DOC,
        archive_object_ref=ARCHIVE,
        reconciliation_policy=_policy(),
        capability_matrix=production_capability_matrix(),
        confidence_policy=_confidence_policy(),
    )

    all_values = {c.value for packet in packets for c in packet.candidates}
    assert "A completely unrelated stale reading" not in all_values
    assert not any(p.semantic_slot_id == "stale-slot" for p in packets)


def test_raises_when_no_evidence_producing_invocation_is_present() -> None:
    events = [
        ProviderObservationAttempted(
            event_id=new_id("event"),
            document_ref=DOC,
            provider_id="docling",
            provider_version="1.0",
            invocation_id=new_id("invocation"),
            outcome=ProviderInvocationOutcome.FAILED,
            observation_count=0,
        )
    ]
    with pytest.raises(RegenerationError):
        regenerate_review_queue(
            events,
            document_ref=DOC,
            archive_object_ref=ARCHIVE,
            reconciliation_policy=_policy(),
            capability_matrix=production_capability_matrix(),
            confidence_policy=_confidence_policy(),
        )


def test_regenerated_queue_reflects_the_fixed_clustering_for_the_real_corpus_shape() -> None:
    """Regression test mirroring the Review Packet Integrity Audit's real finding: a whole-page
    transcript source (no bounding box) alongside several small pixel-accurate fragments from
    another source on the same page. With the fixed clustering, regeneration must not fuse them
    into one contested Review Packet.
    """
    events: list = []
    _emit_observation(
        events, provider="paddleocr-vl", payload=ParagraphPayload(text="a whole page of unrelated text " * 40), bbox=None
    )
    for i in range(3):
        _emit_observation(
            events,
            provider="tesseract_layoutparser",
            payload=ParagraphPayload(text=f"fragment {i}"),
            bbox=BoundingBox(x0=10.0, y0=float(50 * i), x1=200.0, y1=float(50 * i + 20), precision=Precision.PIXEL_ACCURATE),
        )

    packets = regenerate_review_queue(
        events,
        document_ref=DOC,
        archive_object_ref=ARCHIVE,
        reconciliation_policy=_policy(),
        capability_matrix=production_capability_matrix(),
        confidence_policy=_confidence_policy(),
    )

    # No packet mixes the whole-page transcript with a small fragment as competing candidates.
    for packet in packets:
        values = [c.value or "" for c in packet.candidates]
        if len(values) < 2:
            continue
        assert max(len(v) for v in values) / max(1, min(len(v) for v in values)) < 5
