from __future__ import annotations

from archivetrust.application.journal import Journal
from archivetrust.domain.confidence.models import ComparisonClassification
from archivetrust.domain.ontology.types import ObservationType
from archivetrust.infrastructure.storage.telemetry_sink import InMemoryTelemetrySink
from archivetrust.review.sampling.intent import ReviewIntent
from archivetrust.review.sampling.queue import build_adaptive_queue
from archivetrust.review.sampling.strategies import SampledItem
from archivetrust.review.triage import triage_review_queue
from tests.review._helpers import emit_slot, heading


def _document_with_slots() -> InMemoryTelemetrySink:
    sink = InMemoryTelemetrySink()
    for i in range(3):
        emit_slot(
            sink,
            document_ref="doc1",
            canonical_payload=heading(f"single {i}"),
            provider_payloads=(("docling", heading(f"single {i}")),),
            classification=ComparisonClassification.UNCORROBORATED_SINGLE_SOURCE,
        )
    emit_slot(
        sink,
        document_ref="doc1",
        canonical_payload=heading("contested"),
        provider_payloads=(
            ("docling", heading("contested")),
            ("tesseract", heading("contested-x")),
        ),
        classification=ComparisonClassification.CONTESTED,
    )
    return sink


def test_operational_only_queue_matches_triage_review_queue_exactly():
    """The milestone's explicit validation requirement: operational review is unchanged when no
    calibration/research sampling is configured."""
    sink = _document_with_slots()
    state = Journal().replay(sink.events_for_document("doc1"))
    direct = triage_review_queue(state)

    operational = [("doc1", item) for item in direct]
    queue = build_adaptive_queue(operational=operational, sampled=())

    assert len(queue) == len(direct)
    for entry, item in zip(queue, direct):
        assert entry.semantic_slot_id == item.semantic_slot_id
        assert entry.canonical_observation_id == item.canonical_observation_id
        assert entry.observation_type == item.observation_type
        assert entry.disclosure_tier == item.disclosure_tier
        assert entry.policy_version == item.policy_version
        assert entry.reason == item.reason
        assert entry.intent == ReviewIntent.OPERATIONAL


def test_sampled_items_appended_after_operational_by_default():
    sink = _document_with_slots()
    state = Journal().replay(sink.events_for_document("doc1"))
    direct = triage_review_queue(state)
    operational = [("doc1", item) for item in direct]

    sampled = (
        SampledItem(
            semantic_slot_id="slot_x",
            canonical_observation_id="canonical_x",
            document_ref="doc1",
            observation_type=direct[0].observation_type,
            pattern="contested/heading",
            intent=ReviewIntent.CALIBRATION,
            strategy_name="random_sampling",
            explanation="test",
        ),
    )
    queue = build_adaptive_queue(operational=operational, sampled=sampled, order_by="intent")

    assert len(queue) == len(direct) + 1
    assert queue[-1].intent == ReviewIntent.CALIBRATION
    assert all(e.intent == ReviewIntent.OPERATIONAL for e in queue[:-1])


def test_order_by_value_ranks_sampled_items_by_score():
    high = SampledItem(
        semantic_slot_id="slot_high",
        canonical_observation_id="c_high",
        document_ref="doc1",
        observation_type=ObservationType.HEADING,
        pattern="p",
        intent=ReviewIntent.CALIBRATION,
        strategy_name="weighted_sampling",
        explanation="high",
        value_score=0.9,
    )
    low = high.model_copy(update={"semantic_slot_id": "slot_low", "value_score": 0.1, "explanation": "low"})

    queue = build_adaptive_queue(operational=(), sampled=(low, high), order_by="value")
    assert [e.semantic_slot_id for e in queue] == ["slot_high", "slot_low"]
