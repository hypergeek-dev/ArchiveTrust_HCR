"""Acquisition Manager ViewModel (ROADMAP.md §5.13.2) — lists configured Acquisition Sources for
the current Workspace with enabled/status/last_scan/documents_imported/errors/health, plus
add/remove/enable/disable/trigger-scan actions.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from archivetrust.acquisition.archive_object import ArchiveObject
from archivetrust.acquisition.folder_watch import FolderWatchConfig, FolderWatchSource
from archivetrust.acquisition.manager import AcquisitionManager
from archivetrust.acquisition.manual_import import ManualImportSource
from archivetrust.acquisition.source import IMPLEMENTED_SOURCE_KINDS, AcquisitionSourceKind
from archivetrust.workspace.layout import WorkspaceLayout

UNIMPLEMENTED_SOURCE_KINDS: tuple[AcquisitionSourceKind, ...] = tuple(
    kind for kind in AcquisitionSourceKind if kind not in IMPLEMENTED_SOURCE_KINDS
)
"""Network Share, Email Inbox, REST API, SharePoint, ZIP Import, Cloud Storage — real, selectable
entries in the Acquisition Manager UI, honestly labeled "not yet connected" (ROADMAP.md §5.13.2)."""


class SourceRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_id: str
    kind: str
    label: str
    enabled: bool
    status: str
    last_scan_at: str | None
    documents_imported: int
    error_count: int
    last_error: str | None
    implemented: bool


_KIND_LABELS: dict[AcquisitionSourceKind, str] = {
    AcquisitionSourceKind.MANUAL_IMPORT: "Manual Import",
    AcquisitionSourceKind.FOLDER_WATCH: "Folder Watch",
    AcquisitionSourceKind.NETWORK_SHARE: "Network Share",
    AcquisitionSourceKind.EMAIL_INBOX: "Email Inbox",
    AcquisitionSourceKind.REST_API: "REST API",
    AcquisitionSourceKind.SHAREPOINT: "SharePoint",
    AcquisitionSourceKind.ZIP_IMPORT: "ZIP Import",
    AcquisitionSourceKind.CLOUD_STORAGE: "Cloud Storage",
}


class AcquisitionManagerViewModel:
    def __init__(self, *, manager: AcquisitionManager, layout: WorkspaceLayout, workspace_id: str) -> None:
        self._manager = manager
        self._layout = layout
        self._workspace_id = workspace_id

    def configured_sources(self) -> tuple[SourceRow, ...]:
        rows = []
        for status in self._manager.source_status():
            rows.append(
                SourceRow(
                    source_id=status.source_id,
                    kind=status.kind.value,
                    label=_KIND_LABELS.get(status.kind, status.kind.value),
                    enabled=status.enabled,
                    status=status.health.status,
                    last_scan_at=status.health.last_scan_at.isoformat() if status.health.last_scan_at else None,
                    documents_imported=status.health.documents_imported,
                    error_count=status.health.error_count,
                    last_error=status.health.last_error,
                    implemented=True,
                )
            )
        return tuple(rows)

    def not_yet_connected_kinds(self) -> tuple[tuple[str, str], ...]:
        """Kind value + label pairs for the honestly-labeled, not-yet-implemented source kinds."""
        return tuple((kind.value, _KIND_LABELS[kind]) for kind in UNIMPLEMENTED_SOURCE_KINDS)

    def add_manual_import(self, source_id: str = "manual-import") -> ManualImportSource:
        source = ManualImportSource(source_id=source_id)
        self._manager.add_source(source)
        return source

    def add_folder_watch(self, path: str, *, recursive: bool = True, source_id: str = "folder-watch") -> None:
        self._manager.add_source(
            FolderWatchSource(FolderWatchConfig(path=path, recursive=recursive), source_id=source_id)
        )

    def _manual_source(self) -> ManualImportSource | None:
        return next(
            (s for s in self._manager.sources() if isinstance(s, ManualImportSource)), None
        )

    def import_files(self, paths: tuple[str, ...]) -> tuple[ArchiveObject, ...]:
        """Imports one or more operator-selected files through the configured Manual Import
        source (added automatically if none exists yet), queuing each for processing — the
        Acquisition Manager's "Import Files…" action (Operational Completion milestone)."""
        source = self._manual_source() or self.add_manual_import()
        return self._manager.import_files(source, tuple(Path(p) for p in paths))

    def remove_source(self, source_id: str) -> None:
        self._manager.remove_source(source_id)

    def set_enabled(self, source_id: str, enabled: bool) -> None:
        self._manager.set_enabled(source_id, enabled)

    def trigger_scan(self) -> int:
        """Runs one scan cycle across every enabled source; returns the number of newly registered
        Archive Objects."""
        return len(self._manager.run_scan_cycle())
