from __future__ import annotations

import pytest

from archivetrust.domain.feedback.models import CorrectionAction
from archivetrust.domain.ontology.payloads import MetadataPayload
from archivetrust.domain.ontology.types import ObservationType
from archivetrust.infrastructure.storage.telemetry_sink import InMemoryTelemetrySink
from archivetrust.learning.analytics.evolution import (
    EvolutionPolicy,
    EvolutionSignalKind,
    detect_correction_hotspots,
    detect_single_source_concepts,
)
from archivetrust.learning.analytics.provider_health import provider_health
from archivetrust.learning.analytics.quality import (
    human_effort_summary,
    observation_type_quality,
)
from archivetrust.learning.review.interaction import ReviewInteraction, ReviewInteractionKind
from archivetrust.learning.review.session import sessions_from_interactions

from tests.learning._helpers import (
    emit_correction,
    emit_evidence_rejection,
    emit_slot,
    heading,
)


def a_metadata_field(value: str) -> MetadataPayload:
    # `NamedEntityPayload` (docs/htr-migration-plan.md Stage 5 -- EXECUTED) was deleted along with
    # `ObservationType.NAMED_ENTITY`; `MetadataPayload` is a distinct-from-`heading` stand-in for
    # this test's real point, which is generic to any observation type: single-source vs.
    # corroborated detection.
    return MetadataPayload(key="subject", value=value)


# -- Quality Center ---------------------------------------------------------------------------


def test_observation_type_quality_tracks_survival() -> None:
    sink = InMemoryTelemetrySink()
    # 4 heading slots, 1 corrected (edit) -> survival 0.75
    canonicals = [
        emit_slot(sink, document_ref="d1", payload=heading(f"H{i}"), providers=("docling", "qwen"))
        for i in range(4)
    ]
    emit_correction(sink, document_ref="d1", target=canonicals[0], action=CorrectionAction.EDIT.value)

    (quality,) = observation_type_quality(sink)
    assert quality.observation_type is ObservationType.HEADING
    assert quality.slot_count == 4
    assert quality.corrected_slot_count == 1
    assert quality.survival_rate == pytest.approx(0.75)


def test_accept_correction_counts_as_survival_not_defect() -> None:
    sink = InMemoryTelemetrySink()
    canonical = emit_slot(
        sink, document_ref="d1", payload=heading("H"), providers=("docling", "qwen")
    )
    emit_correction(sink, document_ref="d1", target=canonical, action=CorrectionAction.ACCEPT.value)

    (quality,) = observation_type_quality(sink)
    assert quality.corrected_slot_count == 0
    assert quality.confirmed_slot_count == 1
    assert quality.survival_rate == pytest.approx(1.0)


def test_human_effort_summary_edit_rate() -> None:
    interactions = (
        ReviewInteraction(
            interaction_id="i1", review_id="a", target_canonical_observation_id="c",
            reviewer_ref="r", kind=ReviewInteractionKind.REVIEW_OPENED, timestamp=0.0,
        ),
        ReviewInteraction(
            interaction_id="i2", review_id="a", target_canonical_observation_id="c",
            reviewer_ref="r", kind=ReviewInteractionKind.VALUE_ACCEPTED, timestamp=1.0,
        ),
        ReviewInteraction(
            interaction_id="i3", review_id="a", target_canonical_observation_id="c",
            reviewer_ref="r", kind=ReviewInteractionKind.REVIEW_COMPLETED, timestamp=2.0,
        ),
        ReviewInteraction(
            interaction_id="i4", review_id="b", target_canonical_observation_id="c",
            reviewer_ref="r", kind=ReviewInteractionKind.REVIEW_OPENED, timestamp=0.0,
        ),
        ReviewInteraction(
            interaction_id="i5", review_id="b", target_canonical_observation_id="c",
            reviewer_ref="r", kind=ReviewInteractionKind.EDIT_STARTED, timestamp=1.0,
        ),
        ReviewInteraction(
            interaction_id="i6", review_id="b", target_canonical_observation_id="c",
            reviewer_ref="r", kind=ReviewInteractionKind.EDIT_COMMITTED, timestamp=4.0,
        ),
        ReviewInteraction(
            interaction_id="i7", review_id="b", target_canonical_observation_id="c",
            reviewer_ref="r", kind=ReviewInteractionKind.REVIEW_COMPLETED, timestamp=5.0,
        ),
    )
    summary = human_effort_summary(sessions_from_interactions(interactions))
    assert summary.session_count == 2
    assert summary.accepted_count == 1
    assert summary.edited_count == 1
    assert summary.edit_rate == pytest.approx(0.5)
    assert summary.total_edit_duration == pytest.approx(3.0)


# -- Provider Health --------------------------------------------------------------------------


def test_provider_health_correction_rate_is_per_provider() -> None:
    sink = InMemoryTelemetrySink()
    c1 = emit_slot(sink, document_ref="d1", payload=heading("A"), providers=("docling", "qwen"))
    emit_slot(sink, document_ref="d1", payload=heading("B"), providers=("docling", "qwen"))
    emit_correction(sink, document_ref="d1", target=c1, action=CorrectionAction.EDIT.value)
    emit_evidence_rejection(sink, document_ref="d1", provider="qwen")

    health = {h.provider_id: h for h in provider_health(sink)}
    # Both providers contributed to 2 slots; 1 slot corrected -> rate 0.5 for each.
    assert health["docling"].contributing_slot_count == 2
    assert health["docling"].correction_rate == pytest.approx(0.5)
    assert health["qwen"].rejection_count == 1
    # Health never blends confidence across providers -- the type carries no confidence field.
    assert not hasattr(health["docling"], "provider_confidence")


# -- Evolution Center -------------------------------------------------------------------------


def test_correction_hotspot_requires_min_slots_and_threshold() -> None:
    sink = InMemoryTelemetrySink()
    canonicals = [
        emit_slot(sink, document_ref="d1", payload=heading(f"H{i}"), providers=("docling", "qwen"))
        for i in range(4)
    ]
    # 2 of 4 corrected = 50% >= 25% threshold, and 4 slots >= min 4.
    emit_correction(sink, document_ref="d1", target=canonicals[0], action=CorrectionAction.EDIT.value)
    emit_correction(sink, document_ref="d1", target=canonicals[1], action=CorrectionAction.REJECT.value)

    candidates = detect_correction_hotspots(sink)
    assert len(candidates) == 1
    assert candidates[0].signal_kind is EvolutionSignalKind.CORRECTION_HOTSPOT
    assert candidates[0].subject == "heading"
    assert candidates[0].magnitude == pytest.approx(0.5)
    assert candidates[0].policy_version == EvolutionPolicy().version
    assert len(candidates[0].supporting_refs) == 2


def test_correction_hotspot_suppressed_below_min_slots() -> None:
    sink = InMemoryTelemetrySink()
    canonicals = [
        emit_slot(sink, document_ref="d1", payload=heading(f"H{i}"), providers=("docling", "qwen"))
        for i in range(3)  # below default min of 4
    ]
    for c in canonicals:
        emit_correction(sink, document_ref="d1", target=c, action=CorrectionAction.EDIT.value)

    assert detect_correction_hotspots(sink) == ()


def test_single_source_concept_detected_and_corroborated_excluded() -> None:
    sink = InMemoryTelemetrySink()
    # Metadata only ever produced by one provider -> single source.
    emit_slot(sink, document_ref="d1", payload=a_metadata_field("Alice"), providers=("qwen",))
    emit_slot(sink, document_ref="d1", payload=a_metadata_field("Bob"), providers=("qwen",))
    # Heading is corroborated by two providers -> must NOT be flagged.
    emit_slot(sink, document_ref="d1", payload=heading("Title"), providers=("docling", "qwen"))

    candidates = detect_single_source_concepts(sink)
    subjects = {c.subject for c in candidates}
    assert subjects == {"metadata"}
    assert candidates[0].signal_kind is EvolutionSignalKind.SINGLE_SOURCE_CONCEPT
    assert len(candidates[0].supporting_refs) == 2


def test_analyzers_are_reproducible() -> None:
    sink = InMemoryTelemetrySink()
    c = emit_slot(sink, document_ref="d1", payload=heading("A"), providers=("docling", "qwen"))
    emit_correction(sink, document_ref="d1", target=c, action=CorrectionAction.EDIT.value)

    assert observation_type_quality(sink) == observation_type_quality(sink)
    assert provider_health(sink) == provider_health(sink)
    assert detect_single_source_concepts(sink) == detect_single_source_concepts(sink)
