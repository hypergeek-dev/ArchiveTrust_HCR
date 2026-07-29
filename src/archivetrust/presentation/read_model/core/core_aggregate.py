"""Incremental aggregation over the Trust Engine telemetry stream."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from datetime import datetime

from archivetrust.domain.telemetry.events import (
    CanonicalDecisionCreated,
    CanonicalDocumentCreated,
    EvidenceRejected,
    HumanCorrectionApplied,
    HumanCorrectionSubmitted,
    ObservationCreated,
    ProviderInvocationOutcome,
    ProviderObservationAttempted,
    ReviewPacketClosed,
    ReviewPacketDispatched,
    ReviewOutcomeRecorded,
    TelemetryEvent,
)

_ACTIVITY_BUFFER = 64


class CoreAggregate:
    """Core presentation aggregate, maintained incrementally from domain telemetry."""

    def __init__(self) -> None:
        self._cursor = 0
        self.documents: set[str] = set()
        self.documents_with_canonical_snapshot: set[str] = set()
        self.observations_captured = 0
        self.provider_invocations = 0
        self.failures = 0
        self.corrections_applied = 0
        self.review_outcomes_recorded = 0
        self.review_outcomes_by_outcome: dict[str, int] = {}
        self.review_packets_dispatched = 0
        self.review_packets_closed = 0
        self.review_packet_closures_by_kind: dict[str, int] = {}
        self.corrections_submitted = 0
        self.corrections_by_category_action: dict[tuple[str, str], int] = {}
        self.latest_canonicals: dict[tuple[str, str], object] = {}
        self.archive_refs: dict[str, str] = {}
        self.first_seen: dict[str, datetime] = {}
        self.last_seen: dict[str, datetime] = {}
        self.activity: deque[tuple[str, TelemetryEvent]] = deque(maxlen=_ACTIVITY_BUFFER)
        self.provider_counters: dict[tuple[str, str], list[int]] = {}

    @property
    def cursor(self) -> int:
        return self._cursor

    def update(self, events: Sequence[TelemetryEvent] | tuple[TelemetryEvent, ...]) -> None:
        if not isinstance(events, Sequence):
            events = tuple(events)
        if len(events) < self._cursor:
            self._reset()
        for event in events[self._cursor:]:
            self._consume(event)
        self._cursor = len(events)

    def _reset(self) -> None:
        fresh = CoreAggregate()
        self.__dict__.update(fresh.__dict__)

    def _consume(self, event: TelemetryEvent) -> None:
        self.documents.add(event.document_ref)

        if event.recorded_at is not None:
            when = datetime.fromisoformat(event.recorded_at)
            doc = event.document_ref
            if doc not in self.first_seen or when < self.first_seen[doc]:
                self.first_seen[doc] = when
            if doc not in self.last_seen or when > self.last_seen[doc]:
                self.last_seen[doc] = when

        canonical = None
        if isinstance(event, ObservationCreated):
            self.observations_captured += 1
        elif isinstance(event, ProviderObservationAttempted):
            self.provider_invocations += 1
            counters = self.provider_counters.setdefault(
                (event.provider_id, event.provider_version), [0, 0, 0, 0]
            )
            counters[0] += 1
            if event.outcome == ProviderInvocationOutcome.NO_OBSERVATIONS:
                counters[1] += 1
            elif event.outcome == ProviderInvocationOutcome.FAILED:
                counters[2] += 1
            self.activity.append((event.document_ref, event))
        elif isinstance(event, EvidenceRejected):
            self.failures += 1
            self.provider_counters.setdefault(
                (event.provider_id, event.provider_version), [0, 0, 0, 0]
            )[3] += 1
            self.activity.append((event.document_ref, event))
        elif isinstance(event, CanonicalDecisionCreated):
            canonical = event.canonical_observation
            self.activity.append((event.document_ref, event))
        elif isinstance(event, HumanCorrectionApplied):
            self.corrections_applied += 1
            canonical = event.resulting_canonical_observation
            self.activity.append((event.document_ref, event))
        elif isinstance(event, HumanCorrectionSubmitted):
            self.corrections_submitted += 1
            key = (event.category, event.action)
            self.corrections_by_category_action[key] = self.corrections_by_category_action.get(key, 0) + 1
        elif isinstance(event, ReviewOutcomeRecorded):
            self.review_outcomes_recorded += 1
            outcome = event.outcome.value
            self.review_outcomes_by_outcome[outcome] = self.review_outcomes_by_outcome.get(outcome, 0) + 1
        elif isinstance(event, ReviewPacketDispatched):
            self.review_packets_dispatched += 1
        elif isinstance(event, ReviewPacketClosed):
            self.review_packets_closed += 1
            closure_kind = event.closure_kind.value
            self.review_packet_closures_by_kind[closure_kind] = (
                self.review_packet_closures_by_kind.get(closure_kind, 0) + 1
            )
        elif isinstance(event, CanonicalDocumentCreated):
            self.archive_refs[event.document_ref] = event.canonical_document.archive_object_ref
            self.documents_with_canonical_snapshot.add(event.document_ref)
            self.activity.append((event.document_ref, event))

        if canonical is not None:
            key = (event.document_ref, canonical.semantic_slot_id)
            current = self.latest_canonicals.get(key)
            if current is None or canonical.reconciliation_sequence >= current.reconciliation_sequence:  # type: ignore[attr-defined]
                self.latest_canonicals[key] = canonical


TelemetryAggregate = CoreAggregate
