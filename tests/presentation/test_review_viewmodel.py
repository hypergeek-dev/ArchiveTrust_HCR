from __future__ import annotations

from itertools import count

import pytest

from archivetrust.domain.confidence.models import ComparisonClassification
from archivetrust.domain.telemetry.events import HumanCorrectionApplied
from archivetrust.infrastructure.storage.telemetry_sink import InMemoryTelemetrySink
from archivetrust.learning.review.interaction import ReviewInteractionKind
from archivetrust.learning.review.session import ReviewOutcome, sessions_from_interactions
from archivetrust.learning.review.sink import InMemoryReviewInteractionSink
from archivetrust.presentation.review_viewmodel import ReviewStatus, ReviewViewModel
from archivetrust.review.service import ReviewService

from tests.review._helpers import emit_document_snapshot, emit_slot, heading

DOC = "doc_1"
ARCHIVE = "archive_1"


class _FakeClock:
    def __init__(self) -> None:
        self._t = count(start=0, step=1)

    def __call__(self) -> float:
        return float(next(self._t))


def _id_factory() -> "callable":
    counters: dict[str, count] = {}

    def make(prefix: str) -> str:
        counters.setdefault(prefix, count())
        return f"{prefix}_{next(counters[prefix])}"

    return make


def _build(sink: InMemoryTelemetrySink) -> None:
    # A contested slot (candidates "A"/"B") — queued first.
    emit_slot(
        sink,
        document_ref=DOC,
        canonical_payload=heading("A"),
        provider_payloads=(("docling", heading("A")), ("qwen", heading("B"))),
        classification=ComparisonClassification.CONTESTED,
    )
    # A single-source slot — queued second.
    emit_slot(
        sink,
        document_ref=DOC,
        canonical_payload=heading("Solo"),
        provider_payloads=(("qwen", heading("Solo")),),
        classification=ComparisonClassification.UNCORROBORATED_SINGLE_SOURCE,
    )
    emit_document_snapshot(sink, document_ref=DOC, archive_object_ref=ARCHIVE)


def _viewmodel(sink: InMemoryTelemetrySink, interactions: InMemoryReviewInteractionSink) -> ReviewViewModel:
    service = ReviewService(
        telemetry_source=sink, telemetry_sink=sink, interaction_sink=interactions
    )
    return ReviewViewModel(
        service=service,
        reviewer_ref="reviewer_a",
        document_ref=DOC,
        archive_object_ref=ARCHIVE,
        clock=_FakeClock(),
        id_factory=_id_factory(),
    )


def _kinds(interactions: InMemoryReviewInteractionSink) -> list[ReviewInteractionKind]:
    return [i.kind for i in interactions.all_interactions()]


def test_load_opens_first_uncertainty_and_emits_review_opened() -> None:
    sink = InMemoryTelemetrySink()
    _build(sink)
    interactions = InMemoryReviewInteractionSink()
    vm = _viewmodel(sink, interactions)

    vm.load()

    assert vm.status.value is ReviewStatus.REVIEWING
    assert vm.current_packet.value is not None
    assert vm.current_packet.value.current_value == "A"
    assert vm.queue_total == 2
    assert vm.queue_index == 1
    assert _kinds(interactions) == [ReviewInteractionKind.REVIEW_OPENED]


def test_accept_matching_candidate_confirms_and_advances() -> None:
    sink = InMemoryTelemetrySink()
    _build(sink)
    interactions = InMemoryReviewInteractionSink()
    vm = _viewmodel(sink, interactions)
    vm.load()
    packet = vm.current_packet.value
    matching = next(c for c in packet.candidates if c.value == "A")

    vm.accept(matching.observation_id)

    # Advanced to the second (single-source) uncertainty.
    assert vm.current_packet.value.current_value == "Solo"
    assert vm.queue_index == 2
    # One correction applied, value unchanged (pure ACCEPT).
    applied = [e for e in sink.all_events() if isinstance(e, HumanCorrectionApplied)]
    assert len(applied) == 1
    assert applied[0].resulting_canonical_observation.payload.text == "A"
    assert ReviewInteractionKind.VALUE_ACCEPTED in _kinds(interactions)
    assert ReviewInteractionKind.REVIEW_COMPLETED in _kinds(interactions)


def test_accept_other_candidate_changes_value_via_edit_path() -> None:
    sink = InMemoryTelemetrySink()
    _build(sink)
    vm = _viewmodel(sink, InMemoryReviewInteractionSink())
    vm.load()
    packet = vm.current_packet.value
    other = next(c for c in packet.candidates if c.value == "B")

    vm.accept(other.observation_id)

    applied = [e for e in sink.all_events() if isinstance(e, HumanCorrectionApplied)]
    assert applied[0].resulting_canonical_observation.payload.text == "B"  # value changed


def test_commit_edit_records_edit_span_and_changes_value() -> None:
    sink = InMemoryTelemetrySink()
    _build(sink)
    interactions = InMemoryReviewInteractionSink()
    vm = _viewmodel(sink, interactions)
    vm.load()

    vm.begin_edit()
    vm.commit_edit("Edited")

    applied = [e for e in sink.all_events() if isinstance(e, HumanCorrectionApplied)]
    assert applied[0].resulting_canonical_observation.payload.text == "Edited"
    kinds = _kinds(interactions)
    assert kinds.index(ReviewInteractionKind.EDIT_STARTED) < kinds.index(
        ReviewInteractionKind.EDIT_COMMITTED
    )


def test_mark_different_things_resolves_and_advances() -> None:
    sink = InMemoryTelemetrySink()
    _build(sink)
    interactions = InMemoryReviewInteractionSink()
    vm = _viewmodel(sink, interactions)
    vm.load()

    vm.mark_different_things("candidate readings are both valid in different regions")

    applied = [e for e in sink.all_events() if isinstance(e, HumanCorrectionApplied)]
    assert len(applied) == 1
    assert applied[0].resulting_canonical_observation.payload.text == "A"
    assert vm.current_packet.value.current_value == "Solo"
    assert ReviewInteractionKind.REVIEW_COMPLETED in _kinds(interactions)


def test_skip_produces_no_correction_and_leaves_review_incomplete() -> None:
    sink = InMemoryTelemetrySink()
    _build(sink)
    interactions = InMemoryReviewInteractionSink()
    vm = _viewmodel(sink, interactions)
    vm.load()
    first_review_id = next(iter(interactions.all_interactions())).review_id

    vm.skip()

    # No correction emitted for the skipped slot.
    applied = [e for e in sink.all_events() if isinstance(e, HumanCorrectionApplied)]
    assert applied == []
    # The skipped review has no REVIEW_COMPLETED -> its session is INCOMPLETE (§10.7).
    sessions = {s.review_id: s for s in sessions_from_interactions(tuple(interactions.all_interactions()))}
    assert sessions[first_review_id].outcome is ReviewOutcome.INCOMPLETE
    # But it did advance to the next uncertainty.
    assert vm.current_packet.value.current_value == "Solo"


def test_queue_exhaustion_sets_complete() -> None:
    sink = InMemoryTelemetrySink()
    _build(sink)
    vm = _viewmodel(sink, InMemoryReviewInteractionSink())
    vm.load()
    vm.accept(next(c for c in vm.current_packet.value.candidates if c.value == "A").observation_id)
    # second uncertainty (single source) — accept its only candidate
    vm.accept(vm.current_packet.value.candidates[0].observation_id)

    assert vm.status.value is ReviewStatus.COMPLETE
    assert vm.current_packet.value is None
    assert vm.queue_index == 0


def test_passive_observation_hooks_emit_their_kinds() -> None:
    sink = InMemoryTelemetrySink()
    _build(sink)
    interactions = InMemoryReviewInteractionSink()
    vm = _viewmodel(sink, interactions)
    vm.load()

    vm.observe_zoom()
    vm.observe_pan()
    vm.observe_overlay_toggle()
    vm.observe_navigation()

    kinds = _kinds(interactions)
    for expected in (
        ReviewInteractionKind.ZOOMED,
        ReviewInteractionKind.PANNED,
        ReviewInteractionKind.OVERLAY_TOGGLED,
        ReviewInteractionKind.NAVIGATED,
    ):
        assert expected in kinds


def test_decision_without_loaded_packet_raises() -> None:
    sink = InMemoryTelemetrySink()
    _build(sink)
    vm = _viewmodel(sink, InMemoryReviewInteractionSink())
    with pytest.raises(RuntimeError, match="No uncertainty"):
        vm.skip()
