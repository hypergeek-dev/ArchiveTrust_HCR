"""`AcquisitionManager` (ROADMAP.md §5.13.2) — owns the configured Acquisition Sources for one
Workspace, drains them, and aggregates their status for the Acquisition Manager UI.
"""

from __future__ import annotations

from datetime import datetime, timezone

from archivetrust.acquisition.archive_object import ArchiveObject
from archivetrust.acquisition.events import (
    AcquisitionCompleted,
    AcquisitionFailed,
    AcquisitionStarted,
    AcquisitionTelemetrySink,
)
from pathlib import Path

from archivetrust.acquisition.manual_import import ManualImportSource
from archivetrust.acquisition.registration import register_archive_object
from archivetrust.acquisition.source import AcquisitionSource, SourceHealth
from archivetrust.domain.shared.ids import new_id
from archivetrust.workspace.layout import WorkspaceLayout
from archivetrust.security.input_policy import InputSecurityPolicy


class SourceStatus:
    def __init__(self, source: AcquisitionSource, health: SourceHealth) -> None:
        self.source_id = source.source_id
        self.kind = source.kind
        self.enabled = source.enabled
        self.health = health


class AcquisitionManager:
    """`known_hashes` is shared across every source in the Workspace so the same file dropped into
    two different sources is still recognized as one duplicate (Article 25: no Archive Object is
    ever registered twice for the same content within a Workspace).
    """

    def __init__(
        self,
        workspace_id: str,
        layout: WorkspaceLayout,
        sink: AcquisitionTelemetrySink,
        processing_progress: object | None = None,
        security_policy: InputSecurityPolicy | None = None,
    ) -> None:
        """`processing_progress` (Production Incident Recovery, 2026-07-14): without it, `pending`
        starts empty and only grows from *new* imports/scans in this session -- a Workspace reopened
        after a crash mid-run re-registers nothing (every file is already a known content hash) and
        so silently drops every never-processed Archive Object from the queue, indistinguishable
        from a finished run. Passing the durable processing-progress stream lets a reopened
        Workspace resume: any registered Archive Object with no `DocumentProcessingCompleted` record
        yet -- whether never started, or started and never finished (a crash mid-document) -- is
        requeued; only Archive Objects with a recorded completion (successful or failed) are not.
        """
        self._workspace_id = workspace_id
        self._layout = layout
        self._sink = sink
        self._security_policy = security_policy
        self._sources: dict[str, AcquisitionSource] = {}
        existing = self._existing_archive_objects()
        self._known_hashes: set[str] = {obj.content_hash for obj in existing}
        self.pending: list[ArchiveObject] = list(
            self._unprocessed(existing, processing_progress)
        )

    @staticmethod
    def _unprocessed(
        existing: tuple[ArchiveObject, ...], processing_progress: object | None
    ) -> tuple[ArchiveObject, ...]:
        if processing_progress is None:
            return ()
        completed_ids = {
            event.archive_object_id
            for event in processing_progress.events()
            if event.kind.value == "DocumentProcessingCompleted"
            and (
                event.document_state
                in (
                    "completed",
                    "completed_with_review",
                    "provider_partial",
                    "terminal_failure",
                )
                or (event.document_state is None and event.had_failure is not True)
            )
        }
        return tuple(obj for obj in existing if obj.id not in completed_ids)

    def _existing_archive_objects(self) -> tuple[ArchiveObject, ...]:
        """Every Archive Object this Workspace's acquisition stream has ever registered — replayed
        from the sink (Operational Hardening milestone). With a durable sink
        (`FileAcquisitionTelemetrySink`), dedup now survives restarts: the same file re-imported
        after a restart is recorded as a duplicate discovery, never registered or processed twice.
        An in-memory sink replays to nothing, which is exactly the prior behavior.
        """
        from archivetrust.acquisition.events import ArchiveObjectRegistered

        return tuple(
            event.archive_object
            for event in self._sink.events_for_workspace(self._workspace_id)
            if isinstance(event, ArchiveObjectRegistered)
        )

    def archive_object_by_ref(self, archive_object_ref: str) -> ArchiveObject | None:
        """Resolve an Archive Object id back to its full registration record (storage path,
        original filename, hash) from the acquisition stream — how a review surface opens the
        original document for any historical Archive Object, not only this session's.
        """
        from archivetrust.acquisition.events import ArchiveObjectRegistered

        for event in self._sink.events_for_workspace(self._workspace_id):
            if isinstance(event, ArchiveObjectRegistered) and event.archive_object.id == archive_object_ref:
                return event.archive_object
        return None

    def add_source(self, source: AcquisitionSource) -> None:
        self._sources[source.source_id] = source

    def remove_source(self, source_id: str) -> None:
        self._sources.pop(source_id, None)

    def set_enabled(self, source_id: str, enabled: bool) -> None:
        self._sources[source_id].enabled = enabled

    def sources(self) -> tuple[AcquisitionSource, ...]:
        return tuple(self._sources.values())

    def source_status(self) -> tuple[SourceStatus, ...]:
        return tuple(SourceStatus(source, source.health()) for source in self._sources.values())

    def import_files(self, source: ManualImportSource, paths: tuple[Path, ...]) -> tuple[ArchiveObject, ...]:
        """The Manual Import path into `pending` (Operational Completion milestone): a manually
        imported Archive Object used to be registered but never queued for processing, unlike a
        Folder Watch discovery — an honestly-noted gap this closes, so every implemented
        Acquisition Source now feeds the same queue the Process Queue button drains."""
        registered = source.import_files(
            paths, workspace_id=self._workspace_id, layout=self._layout,
            known_hashes=self._known_hashes, sink=self._sink, security_policy=self._security_policy,
        )
        self.pending.extend(registered)
        return registered

    def import_folder(
        self, source: ManualImportSource, folder: Path, *, recursive: bool
    ) -> tuple[ArchiveObject, ...]:
        registered = source.import_folder(
            folder, recursive=recursive, workspace_id=self._workspace_id, layout=self._layout,
            known_hashes=self._known_hashes, sink=self._sink, security_policy=self._security_policy,
        )
        self.pending.extend(registered)
        return registered

    def run_scan_cycle(self) -> tuple[ArchiveObject, ...]:
        """Drains every enabled source once: discover → register. Manual Import sources
        contribute nothing here (they are always operator-triggered via `import_files`/
        `import_folder` above, which register and queue directly)."""
        registered: list[ArchiveObject] = []
        for source in self._sources.values():
            if not source.enabled:
                continue
            self._sink.append(
                AcquisitionStarted(
                    event_id=new_id("event"),
                    workspace_id=self._workspace_id,
                    source_id=source.source_id,
                    source_kind=source.kind.value,
                )
            )
            discovered_count = 0
            try:
                for discovered in source.scan(frozenset(self._known_hashes)):
                    discovered_count += 1
                    archive_object = register_archive_object(
                        discovered,
                        workspace_id=self._workspace_id,
                        layout=self._layout,
                        known_hashes=self._known_hashes,
                        source_id=source.source_id,
                        sink=self._sink,
                        security_policy=self._security_policy,
                    )
                    if archive_object is not None:
                        registered.append(archive_object)
                        self.pending.append(archive_object)
            except Exception as exc:  # noqa: BLE001 -- a source failure is a recorded fact, never a crash
                self._sink.append(
                    AcquisitionFailed(
                        event_id=new_id("event"),
                        workspace_id=self._workspace_id,
                        source_id=source.source_id,
                        reason=str(exc),
                    )
                )
                continue
            self._sink.append(
                AcquisitionCompleted(
                    event_id=new_id("event"),
                    workspace_id=self._workspace_id,
                    source_id=source.source_id,
                    discovered_count=discovered_count,
                    registered_count=len(registered),
                )
            )
        return tuple(registered)
