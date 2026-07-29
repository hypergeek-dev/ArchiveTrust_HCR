"""Application access to ArchiveTrust's authoritative current-state projection.

Consumers depend on this service instead of replaying telemetry and choosing their own notion of
"latest".  The domain projection remains pure; this layer owns source access and cache lifetime.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sized
from typing import Protocol

from archivetrust.domain.current_state import CurrentDocumentState, project_current_document
from archivetrust.domain.telemetry.events import TelemetryEvent


class CurrentStateSource(Protocol):
    """Read-only source seam kept below presentation/review/learning layers."""

    def events_for_document(self, document_ref: str) -> Iterator[TelemetryEvent]: ...

    def all_events(self) -> Iterable[TelemetryEvent]: ...


class CurrentStateService:
    """Project current document truth from one append-only telemetry source.

    Cache entries are keyed by the source's append cursor.  Appends to another document may cause
    a harmless re-projection; an append can never leave a cached document stale.  ``invalidate``
    is exposed for recovery/rebuild tools that replace a source behind an application boundary.
    """

    def __init__(self, telemetry_source: CurrentStateSource) -> None:
        self._source = telemetry_source
        self._cache: dict[str, tuple[int | None, CurrentDocumentState]] = {}

    def document(
        self, document_ref: str, *, integrity_status: str = "not_checked"
    ) -> CurrentDocumentState:
        cursor = self._source_cursor()
        cached = self._cache.get(document_ref)
        if (
            cached is not None
            and cursor is not None
            and cached[0] == cursor
            and cached[1].integrity_status == integrity_status
        ):
            return cached[1]

        state = project_current_document(
            self._source.events_for_document(document_ref),
            integrity_status=integrity_status,
        )
        self._cache[document_ref] = (cursor, state)
        return state

    def documents(
        self, *, integrity_status: str = "not_checked"
    ) -> tuple[CurrentDocumentState, ...]:
        """Return all current documents in first-seen telemetry order."""

        document_refs: list[str] = []
        seen: set[str] = set()
        for event in self._source.all_events():
            if event.document_ref not in seen:
                seen.add(event.document_ref)
                document_refs.append(event.document_ref)
        return tuple(
            self.document(document_ref, integrity_status=integrity_status)
            for document_ref in document_refs
        )

    def invalidate(self, document_ref: str | None = None) -> None:
        if document_ref is None:
            self._cache.clear()
        else:
            self._cache.pop(document_ref, None)

    def _source_cursor(self) -> int | None:
        events: Iterable[TelemetryEvent] = self._source.all_events()
        return len(events) if isinstance(events, Sized) else None


__all__ = ["CurrentStateService"]
