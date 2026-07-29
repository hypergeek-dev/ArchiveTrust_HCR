"""Phase 6 (Constitution Article 30, revised; Article 33) -- `ReviewOutcomeRecorded` for every
terminal decision including SKIP, the not-eligible/withheld-by-policy triage distinction, and the
derived (never-persisted) aging query.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from archivetrust.domain.confidence.models import ComparisonClassification
from archivetrust.domain.telemetry.events import ReviewOutcome, ReviewOutcomeRecorded
from archivetrust.infrastructure.storage.telemetry_sink import InMemoryTelemetrySink
from archivetrust.learning.review.sink import InMemoryReviewInteractionSink
from archivetrust.application.journal import Journal
from archivetrust.review.aging import aged_review_slots
from archivetrust.review.service import ReviewAction, ReviewService
from archivetrust.review.triage import (
    TriageClassification,
    TriagePolicy,
    triage_classification_for,
    triage_review_queue,
)

from tests.review._helpers import emit_document_snapshot, emit_slot, heading

DOC = "doc_1"
ARCHIVE = "archive_object_1"


def _service(sink: InMemoryTelemetrySink) -> ReviewService:
    emit_document_snapshot(sink, document_ref=DOC, archive_object_ref=ARCHIVE)
    return ReviewService(
        telemetry_source=sink, telemetry_sink=sink, interaction_sink=InMemoryReviewInteractionSink()
    )


def test_every_terminal_action_emits_exactly_one_review_outcome_recorded() -> None:
    for action, expected_outcome in (
        (ReviewAction.ACCEPT_PROVIDER, ReviewOutcome.ACCEPTED),
        (ReviewAction.REJECT, ReviewOutcome.REJECTED),
        (ReviewAction.MARK_AMBIGUOUS, ReviewOutcome.RESOLVED),
        (ReviewAction.DIFFERENT_THINGS, ReviewOutcome.DIFFERENT_THINGS),
        (ReviewAction.ILLEGIBLE, ReviewOutcome.ILLEGIBLE),
        (ReviewAction.REQUEST_FURTHER_REVIEW, ReviewOutcome.RESOLVED),
    ):
        sink = InMemoryTelemetrySink()
        emit_slot(
            sink, document_ref=DOC, canonical_payload=heading("X"),
            provider_payloads=(("docling", heading("X")), ("qwen", heading("Y"))),
            classification=ComparisonClassification.CONTESTED,
        )
        service = _service(sink)
        (packet,) = service.open_document(document_ref=DOC, archive_object_ref=ARCHIVE)
        result = service.submit_decision(packet=packet, action=action)

        outcomes = [e for e in result.emitted_events if isinstance(e, ReviewOutcomeRecorded)]
        assert len(outcomes) == 1, f"expected exactly one ReviewOutcomeRecorded for {action}"
        assert outcomes[0].outcome == expected_outcome
        assert outcomes[0].correction_id == result.correction_id
        assert outcomes[0].action == action.value


def test_manual_edit_also_emits_exactly_one_review_outcome_recorded() -> None:
    sink = InMemoryTelemetrySink()
    emit_slot(
        sink, document_ref=DOC, canonical_payload=heading("X"),
        provider_payloads=(("docling", heading("X")), ("qwen", heading("Y"))),
        classification=ComparisonClassification.CONTESTED,
    )
    service = _service(sink)
    (packet,) = service.open_document(document_ref=DOC, archive_object_ref=ARCHIVE)
    result = service.submit_decision(
        packet=packet, action=ReviewAction.MANUAL_EDIT, corrected_output="Corrected"
    )
    outcomes = [e for e in result.emitted_events if isinstance(e, ReviewOutcomeRecorded)]
    assert len(outcomes) == 1
    assert outcomes[0].outcome == ReviewOutcome.CORRECTED


def test_opening_a_document_repeatedly_emits_no_new_telemetry() -> None:
    # Constitution Article 33: triage/packet assembly is a deterministic projection -- re-running
    # it any number of times over an unchanged sink must add zero events.
    sink = InMemoryTelemetrySink()
    emit_slot(
        sink, document_ref=DOC, canonical_payload=heading("X"),
        provider_payloads=(("docling", heading("X")), ("qwen", heading("Y"))),
        classification=ComparisonClassification.CONTESTED,
    )
    service = _service(sink)
    before = len(tuple(sink.all_events()))
    service.open_document(document_ref=DOC, archive_object_ref=ARCHIVE)
    service.open_document(document_ref=DOC, archive_object_ref=ARCHIVE)
    service.open_document(document_ref=DOC, archive_object_ref=ARCHIVE)
    assert len(tuple(sink.all_events())) == before


def test_triage_classification_distinguishes_not_eligible_from_withheld_by_policy() -> None:
    sink = InMemoryTelemetrySink()
    # Corroborated, high confidence: genuinely nothing to review.
    fine_canonical = emit_slot(
        sink, document_ref=DOC, canonical_payload=heading("Fine"),
        provider_payloads=(("docling", heading("Fine")), ("qwen", heading("Fine"))),
        classification=ComparisonClassification.CORROBORATED, canonical_confidence=0.95,
    )
    # Single-source PARAGRAPH: excluded only because PARAGRAPH isn't in
    # DEFAULT_SINGLE_SOURCE_REVIEW_TYPES (the Adversarial Audit's finding) -- withheld by policy.
    from archivetrust.domain.ontology.payloads import ParagraphPayload

    withheld_canonical = emit_slot(
        sink, document_ref=DOC, canonical_payload=ParagraphPayload(text="Solo"),
        provider_payloads=(("paddleocr_vl", ParagraphPayload(text="Solo")),),
        classification=ComparisonClassification.UNCORROBORATED_SINGLE_SOURCE,
    )

    assert triage_classification_for(fine_canonical) == TriageClassification.NOT_ELIGIBLE
    assert triage_classification_for(withheld_canonical) == TriageClassification.WITHHELD_BY_POLICY
    # A more permissive policy would have queued the withheld slot -- proving the distinction.
    permissive = TriagePolicy(single_source_review_types=None)
    assert triage_classification_for(withheld_canonical, permissive) == TriageClassification.QUEUED


def test_resolving_actions_remove_the_slot_from_the_active_queue() -> None:
    """Phase 23 (Review Queue Consistency): before this fix, `comparison_confidence.classification`
    (carried over unchanged by `apply_human_correction`) kept a CONTESTED slot as SOURCES_DISAGREE
    forever -- a resolved packet re-appeared in a freshly rebuilt queue and could be re-reviewed."""
    for action, kwargs in (
        (ReviewAction.ACCEPT_PROVIDER, {}),
        (ReviewAction.MANUAL_EDIT, {"corrected_output": "Corrected"}),
        (ReviewAction.REJECT, {}),
        (ReviewAction.MARK_AMBIGUOUS, {}),
        (ReviewAction.DIFFERENT_THINGS, {}),
        (ReviewAction.REQUEST_FURTHER_REVIEW, {}),
    ):
        sink = InMemoryTelemetrySink()
        emit_slot(
            sink, document_ref=DOC, canonical_payload=heading("X"),
            provider_payloads=(("docling", heading("X")), ("qwen", heading("Y"))),
            classification=ComparisonClassification.CONTESTED,
        )
        service = _service(sink)
        (packet,) = service.open_document(document_ref=DOC, archive_object_ref=ARCHIVE)
        service.submit_decision(packet=packet, action=action, **kwargs)

        state = Journal().replay(sink.events_for_document(DOC))
        assert triage_review_queue(state) == (), f"{action} left the slot in the active queue"


def test_skip_leaves_the_slot_in_the_active_queue() -> None:
    """Skip is deliberately not a resolution (§10.7: "a skip is never an acceptance") -- the
    uncertainty must remain available for a future reviewer, unlike every other decision."""
    sink = InMemoryTelemetrySink()
    emit_slot(
        sink, document_ref=DOC, canonical_payload=heading("X"),
        provider_payloads=(("docling", heading("X")), ("qwen", heading("Y"))),
        classification=ComparisonClassification.CONTESTED,
    )
    service = _service(sink)
    (packet,) = service.open_document(document_ref=DOC, archive_object_ref=ARCHIVE)
    service.submit_decision(packet=packet, action=ReviewAction.SKIP)

    state = Journal().replay(sink.events_for_document(DOC))
    assert len(triage_review_queue(state)) == 1


def test_aged_review_slots_finds_nothing_within_horizon() -> None:
    sink = InMemoryTelemetrySink()
    emit_slot(
        sink, document_ref=DOC, canonical_payload=heading("X"),
        provider_payloads=(("docling", heading("X")), ("qwen", heading("Y"))),
        classification=ComparisonClassification.CONTESTED,
    )
    aged = aged_review_slots(document_ref=DOC, sink=sink, horizon=timedelta(days=7))
    assert aged == ()


def test_aged_review_slots_finds_an_eligible_slot_past_the_horizon() -> None:
    sink = InMemoryTelemetrySink()
    emit_slot(
        sink, document_ref=DOC, canonical_payload=heading("X"),
        provider_payloads=(("docling", heading("X")), ("qwen", heading("Y"))),
        classification=ComparisonClassification.CONTESTED,
    )
    far_future = datetime.now(timezone.utc) + timedelta(days=30)
    aged = aged_review_slots(document_ref=DOC, sink=sink, horizon=timedelta(days=7), now=far_future)
    assert len(aged) == 1


def test_aged_review_slots_excludes_a_slot_with_a_recorded_outcome() -> None:
    sink = InMemoryTelemetrySink()
    emit_slot(
        sink, document_ref=DOC, canonical_payload=heading("X"),
        provider_payloads=(("docling", heading("X")), ("qwen", heading("Y"))),
        classification=ComparisonClassification.CONTESTED,
    )
    service = _service(sink)
    (packet,) = service.open_document(document_ref=DOC, archive_object_ref=ARCHIVE)
    service.submit_decision(packet=packet, action=ReviewAction.SKIP)

    far_future = datetime.now(timezone.utc) + timedelta(days=30)
    aged = aged_review_slots(document_ref=DOC, sink=sink, horizon=timedelta(days=7), now=far_future)
    assert aged == ()
