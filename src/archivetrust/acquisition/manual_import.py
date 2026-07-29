"""Manual Import (ROADMAP.md §5.13.2) — an operator explicitly imports files or a folder (or, in
the desktop client, drags them in). Never modifies or removes the originals; goes through
`register_archive_object` exactly like every other source.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from archivetrust.acquisition.archive_object import ArchiveObject
from archivetrust.acquisition.events import AcquisitionTelemetrySink
from archivetrust.acquisition.registration import register_archive_object
from archivetrust.acquisition.source import AcquisitionSourceKind, DiscoveredFile, SourceHealth
from archivetrust.workspace.layout import WorkspaceLayout
from archivetrust.security.input_policy import InputSecurityPolicy


class ManualImportSource:
    kind = AcquisitionSourceKind.MANUAL_IMPORT

    def __init__(self, source_id: str = "manual-import") -> None:
        self.source_id = source_id
        self.enabled = True
        self._documents_imported = 0
        self._error_count = 0
        self._last_scan_at: datetime | None = None

    def scan(self, known_hashes: frozenset[str]):
        return ()  # Manual Import never discovers on its own — it is always operator-triggered.

    def health(self) -> SourceHealth:
        return SourceHealth(
            status="ok",
            last_scan_at=self._last_scan_at,
            documents_imported=self._documents_imported,
        )

    def import_files(
        self,
        paths: tuple[Path, ...],
        *,
        workspace_id: str,
        layout: WorkspaceLayout,
        known_hashes: set[str],
        sink: AcquisitionTelemetrySink,
        security_policy: InputSecurityPolicy | None = None,
    ) -> tuple[ArchiveObject, ...]:
        registered = []
        for path in paths:
            discovered = DiscoveredFile(
                path=path,
                original_filename=path.name,
                byte_size=path.stat().st_size,
                modified_at=datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc),
            )
            archive_object = register_archive_object(
                discovered,
                workspace_id=workspace_id,
                layout=layout,
                known_hashes=known_hashes,
                source_id=self.source_id,
                sink=sink,
                security_policy=security_policy,
            )
            if archive_object is not None:
                registered.append(archive_object)
        self._documents_imported += len(registered)
        self._last_scan_at = datetime.now(timezone.utc)
        return tuple(registered)

    def import_folder(
        self,
        folder: Path,
        *,
        recursive: bool,
        workspace_id: str,
        layout: WorkspaceLayout,
        known_hashes: set[str],
        sink: AcquisitionTelemetrySink,
        security_policy: InputSecurityPolicy | None = None,
    ) -> tuple[ArchiveObject, ...]:
        pattern = "**/*" if recursive else "*"
        files = tuple(p for p in folder.glob(pattern) if p.is_file())
        return self.import_files(
            files, workspace_id=workspace_id, layout=layout, known_hashes=known_hashes, sink=sink,
            security_policy=security_policy,
        )
