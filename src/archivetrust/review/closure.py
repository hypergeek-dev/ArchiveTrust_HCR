"""Review-packet lifecycle closure projection and helpers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from pydantic import BaseModel, ConfigDict

from archivetrust.domain.shared.ids import new_id
from archivetrust.domain.telemetry.events import (
    ReviewOutcome,
    ReviewOutcomeRecorded,
    ReviewPacketClosed,
    ReviewPacketClosureKind,
    ReviewPacketCreated,
    ReviewPacketDispatched,
    ReviewPacketOpened,
    TelemetryEvent,
    stamp_recorded_at,
)
from archivetrust.domain.telemetry.sink import TelemetrySink
from archivetrust.review.packet import ReviewPacket


class OpenReviewPacket(BaseModel):
    model_config = ConfigDict(frozen=True)

    packet_id: str
    document_ref: str
    semantic_slot_id: str
    canonical_observation_id: str
    dispatched_at: str | None
    reviewer_ref: str | None = None
    opened_at: str | None = None
    status: str = "dispatched"


class ClosedReviewPacket(BaseModel):
    model_config = ConfigDict(frozen=True)

    packet_id: str
    document_ref: str
    semantic_slot_id: str
    canonical_observation_id: str
    closure_kind: ReviewPacketClosureKind
    dispatched_at: str | None
    closed_at: str | None
    reason: str | None = None
    outcome_id: str
    outcome: str | None = None


class ReviewClosureStatus(BaseModel):
    model_config = ConfigDict(frozen=True)

    dispatched_count: int
    open_count: int
    closed_count: int
    aged_open_count: int
    oldest_open_age_seconds: float | None
    open_packets: tuple[OpenReviewPacket, ...]
    closed_packets: tuple[ClosedReviewPacket, ...]
    created_count: int = 0
    opened_count: int = 0
    legacy_incomplete_count: int = 0


def dispatch_review_packet(
    *,
    sink: TelemetrySink,
    packet: ReviewPacket,
    reviewer_ref: str | None = None,
    review_intent: str | None = None,
) -> ReviewPacketDispatched:
    packet_id = new_id("review_packet")
    created = stamp_recorded_at(
        ReviewPacketCreated(
            event_id=new_id("event"),
            document_ref=packet.document_ref,
            packet_id=packet_id,
            semantic_slot_id=packet.semantic_slot_id,
            canonical_observation_id=packet.canonical_observation_id,
            review_intent=review_intent,
        )
    )
    event = stamp_recorded_at(
        ReviewPacketDispatched(
            event_id=new_id("event"),
            document_ref=packet.document_ref,
            packet_id=packet_id,
            semantic_slot_id=packet.semantic_slot_id,
            canonical_observation_id=packet.canonical_observation_id,
            reviewer_ref=reviewer_ref,
            review_intent=review_intent,
        )
    )
    sink.append(created)
    sink.append(event)
    return event


def open_review_packet(
    *, sink: TelemetrySink, dispatch: ReviewPacketDispatched, reviewer_ref: str | None = None
) -> ReviewPacketOpened:
    event = stamp_recorded_at(
        ReviewPacketOpened(
            event_id=new_id("event"),
            document_ref=dispatch.document_ref,
            packet_id=dispatch.packet_id,
            semantic_slot_id=dispatch.semantic_slot_id,
            canonical_observation_id=dispatch.canonical_observation_id,
            reviewer_ref=reviewer_ref,
        )
    )
    sink.append(event)
    return event


def close_review_packet(
    *,
    sink: TelemetrySink,
    dispatch: ReviewPacketDispatched,
    closure_kind: ReviewPacketClosureKind,
    reason: str | None = None,
    outcome_id: str | None = None,
    correction_id: str | None = None,
) -> ReviewPacketClosed:
    existing = next(
        (
            event
            for event in sink.events_for_document(dispatch.document_ref)
            if isinstance(event, ReviewPacketClosed) and event.packet_id == dispatch.packet_id
        ),
        None,
    )
    if existing is not None:
        return existing
    if outcome_id is None:
        outcome_by_closure = {
            ReviewPacketClosureKind.EXPIRED: ReviewOutcome.EXPIRED,
            ReviewPacketClosureKind.WITHDRAWN: ReviewOutcome.WITHDRAWN,
            ReviewPacketClosureKind.FAILED_DURING_APPLICATION: ReviewOutcome.FAILED_DURING_APPLICATION,
            ReviewPacketClosureKind.DEFERRED: ReviewOutcome.DEFERRED,
            ReviewPacketClosureKind.ACCEPTED: ReviewOutcome.ACCEPTED,
            ReviewPacketClosureKind.CORRECTED: ReviewOutcome.CORRECTED,
            ReviewPacketClosureKind.REJECTED: ReviewOutcome.REJECTED,
            ReviewPacketClosureKind.ILLEGIBLE: ReviewOutcome.ILLEGIBLE,
            ReviewPacketClosureKind.DIFFERENT_THINGS: ReviewOutcome.DIFFERENT_THINGS,
            ReviewPacketClosureKind.RESOLVED: ReviewOutcome.RESOLVED,
        }
        outcome = stamp_recorded_at(
            ReviewOutcomeRecorded(
                event_id=new_id("event"),
                document_ref=dispatch.document_ref,
                outcome_id=new_id("review_outcome"),
                semantic_slot_id=dispatch.semantic_slot_id,
                canonical_observation_id=dispatch.canonical_observation_id,
                action=reason or closure_kind.value,
                outcome=outcome_by_closure[closure_kind],
                correction_id=correction_id,
                reviewer_ref=dispatch.reviewer_ref,
            )
        )
        sink.append(outcome)
        outcome_id = outcome.outcome_id
    event = stamp_recorded_at(
        ReviewPacketClosed(
            event_id=new_id("event"),
            document_ref=dispatch.document_ref,
            packet_id=dispatch.packet_id,
            semantic_slot_id=dispatch.semantic_slot_id,
            canonical_observation_id=dispatch.canonical_observation_id,
            closure_kind=closure_kind,
            reason=reason,
            outcome_id=outcome_id,
            correction_id=correction_id,
        )
    )
    sink.append(event)
    return event


def review_closure_status(
    events: tuple[TelemetryEvent, ...] | list[TelemetryEvent],
    *,
    horizon: timedelta,
    now: datetime | None = None,
) -> ReviewClosureStatus:
    now = now or datetime.now(timezone.utc)
    dispatches: dict[str, ReviewPacketDispatched] = {}
    created: dict[str, ReviewPacketCreated] = {}
    opened: dict[str, ReviewPacketOpened] = {}
    closures: dict[str, ReviewPacketClosed] = {}
    outcomes: dict[str, ReviewOutcomeRecorded] = {}

    for event in events:
        if isinstance(event, ReviewPacketCreated):
            created[event.packet_id] = event
        elif isinstance(event, ReviewPacketDispatched):
            dispatches[event.packet_id] = event
        elif isinstance(event, ReviewPacketOpened):
            opened[event.packet_id] = event
        elif isinstance(event, ReviewPacketClosed) and event.packet_id not in closures:
            # First terminal closure wins. Later duplicate closures remain telemetry, but the
            # lifecycle projection never reopens or re-closes a packet.
            closures[event.packet_id] = event
        elif isinstance(event, ReviewOutcomeRecorded):
            outcomes[event.outcome_id] = event

    open_packets: list[OpenReviewPacket] = []
    closed_packets: list[ClosedReviewPacket] = []
    oldest_open_age: float | None = None
    aged_open_count = 0

    for packet_id, dispatch in sorted(dispatches.items()):
        closure = closures.get(packet_id)
        if closure is not None:
            closed_packets.append(
                ClosedReviewPacket(
                    packet_id=packet_id,
                    document_ref=dispatch.document_ref,
                    semantic_slot_id=dispatch.semantic_slot_id,
                    canonical_observation_id=dispatch.canonical_observation_id,
                    closure_kind=closure.closure_kind,
                    dispatched_at=dispatch.recorded_at,
                    closed_at=closure.recorded_at,
                    reason=closure.reason,
                    outcome_id=closure.outcome_id,
                    outcome=(
                        outcomes[closure.outcome_id].outcome.value
                        if closure.outcome_id in outcomes
                        else None
                    ),
                )
            )
            continue

        opened_event = opened.get(packet_id)
        open_packets.append(
            OpenReviewPacket(
                packet_id=packet_id,
                document_ref=dispatch.document_ref,
                semantic_slot_id=dispatch.semantic_slot_id,
                canonical_observation_id=dispatch.canonical_observation_id,
                dispatched_at=dispatch.recorded_at,
                reviewer_ref=dispatch.reviewer_ref,
                opened_at=opened_event.recorded_at if opened_event is not None else None,
                status="in_progress" if opened_event is not None else "dispatched",
            )
        )
        age = _age_seconds(dispatch.recorded_at, now)
        if age is not None:
            oldest_open_age = age if oldest_open_age is None else max(oldest_open_age, age)
            if age > horizon.total_seconds():
                aged_open_count += 1

    return ReviewClosureStatus(
        dispatched_count=len(dispatches),
        open_count=len(open_packets),
        closed_count=len(closed_packets),
        aged_open_count=aged_open_count,
        oldest_open_age_seconds=oldest_open_age,
        open_packets=tuple(open_packets),
        closed_packets=tuple(closed_packets),
        created_count=len(created),
        opened_count=len(opened),
        legacy_incomplete_count=sum(
            outcome.outcome_id not in {closure.outcome_id for closure in closures.values()}
            for outcome in outcomes.values()
        ),
    )


def _age_seconds(recorded_at: str | None, now: datetime) -> float | None:
    if recorded_at is None:
        return None
    return max(0.0, (now - datetime.fromisoformat(recorded_at)).total_seconds())
