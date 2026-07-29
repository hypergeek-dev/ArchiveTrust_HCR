"""Acquisition Telemetry (ROADMAP.md §12.1) — a closed event set describing a Workspace's
operational acquisition history. Deliberately **not** part of `domain.telemetry.events`'s
`TelemetryEventKind`: that enum is closed by design to the Trust Engine's knowledge-evolution events
about document facts (Article 16); these events describe a different concern, one stage upstream of
any document fact existing at all. Same philosophy — immutable, append-only, replayable — as a
separate, independently closed stream.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict

from archivetrust.acquisition.archive_object import ArchiveObject


class AcquisitionEventKind(str, Enum):
    WORKSPACE_CREATED = "WorkspaceCreated"
    WORKSPACE_OPENED = "WorkspaceOpened"
    ACQUISITION_STARTED = "AcquisitionStarted"
    ARCHIVE_OBJECT_DISCOVERED = "ArchiveObjectDiscovered"
    ARCHIVE_OBJECT_REGISTERED = "ArchiveObjectRegistered"
    ACQUISITION_COMPLETED = "AcquisitionCompleted"
    ACQUISITION_FAILED = "AcquisitionFailed"


class AcquisitionEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: AcquisitionEventKind
    event_id: str
    workspace_id: str


class WorkspaceCreated(AcquisitionEvent):
    kind: AcquisitionEventKind = AcquisitionEventKind.WORKSPACE_CREATED

    name: str


class WorkspaceOpened(AcquisitionEvent):
    kind: AcquisitionEventKind = AcquisitionEventKind.WORKSPACE_OPENED


class AcquisitionStarted(AcquisitionEvent):
    kind: AcquisitionEventKind = AcquisitionEventKind.ACQUISITION_STARTED

    source_id: str
    source_kind: str


class ArchiveObjectDiscovered(AcquisitionEvent):
    """Records that a candidate file was found by a source, before hashing/dedup/registration
    decides whether it becomes an Archive Object — so "discovered but a duplicate" is a
    distinguishable, recorded fact, never a silent skip (Constitution Article 18's discipline,
    applied to acquisition).
    """

    kind: AcquisitionEventKind = AcquisitionEventKind.ARCHIVE_OBJECT_DISCOVERED

    source_id: str
    original_filename: str
    content_hash: str


class ArchiveObjectRegistered(AcquisitionEvent):
    kind: AcquisitionEventKind = AcquisitionEventKind.ARCHIVE_OBJECT_REGISTERED

    archive_object: ArchiveObject


class AcquisitionCompleted(AcquisitionEvent):
    kind: AcquisitionEventKind = AcquisitionEventKind.ACQUISITION_COMPLETED

    source_id: str
    discovered_count: int
    registered_count: int


class AcquisitionFailed(AcquisitionEvent):
    kind: AcquisitionEventKind = AcquisitionEventKind.ACQUISITION_FAILED

    source_id: str
    reason: str


EVENT_TYPE_BY_KIND: dict[AcquisitionEventKind, type[AcquisitionEvent]] = {
    event_cls.model_fields["kind"].default: event_cls
    for event_cls in (
        WorkspaceCreated,
        WorkspaceOpened,
        AcquisitionStarted,
        ArchiveObjectDiscovered,
        ArchiveObjectRegistered,
        AcquisitionCompleted,
        AcquisitionFailed,
    )
}


def parse_acquisition_event(data: dict[str, Any]) -> AcquisitionEvent:
    kind = AcquisitionEventKind(data["kind"])
    event_cls = EVENT_TYPE_BY_KIND[kind]
    return event_cls.model_validate(data)


class AcquisitionTelemetrySink:
    """Append-only in-memory sink — the acquisition-stream analogue of
    `infrastructure.storage.telemetry_sink.InMemoryTelemetrySink`, kept in this package since
    Acquisition telemetry is a deliberately separate stream (module docstring).
    """

    def __init__(self) -> None:
        self._events: list[AcquisitionEvent] = []

    def append(self, event: AcquisitionEvent) -> None:
        self._events.append(event)

    def all_events(self) -> tuple[AcquisitionEvent, ...]:
        return tuple(self._events)

    def events_for_workspace(self, workspace_id: str) -> tuple[AcquisitionEvent, ...]:
        return tuple(e for e in self._events if e.workspace_id == workspace_id)


class FileAcquisitionTelemetrySink(AcquisitionTelemetrySink):
    """Durable, append-only JSON-Lines acquisition stream (Operational Hardening milestone) — the
    acquisition-stream analogue of `FileTelemetrySink`, same write-through-mirror design: the file
    is the single durable record, the in-memory list a cache of it loaded once at construction.

    Durability here is what makes two operational facts survive a restart: which content hashes
    are already registered (so a restart never re-registers or re-processes the same corpus —
    the previously documented `AcquisitionManager._known_hashes` gap), and which `storage_path`
    each Archive Object lives at (so the Review Center can open the original document for any
    historical Archive Object, not only ones acquired this session).
    """

    def __init__(self, path) -> None:
        from pathlib import Path
        import json

        super().__init__()
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._path.touch()
        for line in self._path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                self._events.append(parse_acquisition_event(json.loads(line)))

    def append(self, event: AcquisitionEvent) -> None:
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(event.model_dump_json())
            handle.write("\n")
        super().append(event)
