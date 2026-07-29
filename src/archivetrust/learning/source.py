"""The Learning Platform's single input seam (ROADMAP_V2.md S5.2, S7).

Everything the Learning Platform knows, it learns by reading the Trust Engine's telemetry stream
-- never by querying live pipeline state, and never by holding a reference it could write through.
`TelemetrySource` is a deliberately *read-only* view over that stream: it exposes only the read
half of `archivetrust.domain.telemetry.sink.TelemetrySink` (`all_events`, `events_for_document`)
and omits `append` entirely, so an analytics component cannot -- even by accident -- write an event
back into the Trust Engine's history.

The existing `InMemoryTelemetrySink` / `FileTelemetrySink` (infrastructure) satisfy this Protocol
structurally, so the Learning Platform reads exactly the same durable stream the Trust Engine
wrote, with no adapter and no copy (ROADMAP_V2.md S13: analytics storage is reconstructible from
this stream, never a separate source of truth).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Protocol, runtime_checkable

from archivetrust.domain.telemetry.events import TelemetryEvent


@runtime_checkable
class TelemetrySource(Protocol):
    """A read-only view over the Trust Engine's knowledge-evolution telemetry.

    Intentionally narrower than `TelemetrySink`: no `append`. The Learning Platform observes; it
    does not emit into the stream it observes (ROADMAP_V2.md Guiding Principle 2).
    """

    def events_for_document(self, document_ref: str) -> Iterator[TelemetryEvent]:
        """Every Trust Engine event for one Archive Object, in append order."""
        ...

    def all_events(self) -> Iterable[TelemetryEvent]:
        """Every Trust Engine event this source holds, across all documents, in append order."""
        ...
