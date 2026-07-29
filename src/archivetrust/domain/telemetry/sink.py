"""TelemetrySink interface (ROADMAP.md S11: "start with an in-process, pluggable interface").

The domain layer defines only the interface (Dependency Inversion, S5.7) -- concrete sinks
(in-memory, file-based, and any future message-broker-backed sink) live in
`infrastructure/storage` and depend inward on this Protocol, never the reverse.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Protocol, runtime_checkable

from archivetrust.domain.telemetry.events import TelemetryEvent


@runtime_checkable
class TelemetrySink(Protocol):
    """Append-only. No update or delete operation exists, by design -- telemetry describes
    knowledge evolution (Constitution Article 16), and knowledge evolution is itself expressed by
    appending new events, never rewriting old ones.
    """

    def append(self, event: TelemetryEvent) -> None:
        """Persists one event. Must preserve insertion order for events sharing a
        `document_ref` -- `events_for_document` relies on it for replay ordering.
        """
        ...

    def events_for_document(self, document_ref: str) -> Iterator[TelemetryEvent]:
        """All events for one document, in the order they were appended."""
        ...

    def all_events(self) -> Iterable[TelemetryEvent]:
        """Every event this sink holds, across all documents, in append order."""
        ...
