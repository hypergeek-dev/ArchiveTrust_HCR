"""ResearchTelemetrySink interface (Constitution Article 34).

Mirrors `domain.telemetry.sink.TelemetrySink`'s shape exactly, as a deliberately separate Protocol
-- a Research Telemetry event is never appended to, nor retrieved from, a document-scoped
`TelemetrySink`, and the reverse. `events_for_corpus` is the corpus-scoped analogue of
`TelemetrySink.events_for_document`.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Protocol, runtime_checkable

from archivetrust.domain.research.events import ResearchEvent


@runtime_checkable
class ResearchTelemetrySink(Protocol):
    """Append-only, exactly like `TelemetrySink` (Article 16's discipline, applied to this stream
    too): knowledge -- here, self-knowledge -- evolves only by appending, never by rewriting.
    """

    def append(self, event: ResearchEvent) -> None:
        """Persists one event. Must preserve insertion order for events sharing a `corpus_ref`."""
        ...

    def events_for_corpus(self, corpus_ref: str) -> Iterator[ResearchEvent]:
        """Every event recorded for one corpus, in the order they were appended."""
        ...

    def all_events(self) -> Iterable[ResearchEvent]:
        """Every event this sink holds, across all corpora, in append order."""
        ...
