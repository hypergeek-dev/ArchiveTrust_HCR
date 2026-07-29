"""`register_archive_object` — the one place a `DiscoveredFile` becomes an `ArchiveObject`
(ROADMAP.md §5.13.2): hash, dedup, copy into `archive/`, emit telemetry. Every acquisition source
goes through this function; none may register a file, or bypass registration to invoke a provider,
by any other path (Constitution Article 25).
"""

from __future__ import annotations

import hashlib
import mimetypes
import shutil
from datetime import datetime, timezone

from archivetrust.acquisition.archive_object import ArchiveObject
from archivetrust.acquisition.events import (
    AcquisitionTelemetrySink,
    ArchiveObjectDiscovered,
    ArchiveObjectRegistered,
)
from archivetrust.acquisition.source import DiscoveredFile
from archivetrust.domain.shared.ids import new_id
from archivetrust.workspace.layout import WorkspaceLayout
from archivetrust.security.input_policy import InputSecurityPolicy, validate_source_file

_HASH_CHUNK_SIZE = 1024 * 1024


def _content_hash(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def register_archive_object(
    discovered: DiscoveredFile,
    *,
    workspace_id: str,
    layout: WorkspaceLayout,
    known_hashes: set[str],
    source_id: str,
    sink: AcquisitionTelemetrySink,
    security_policy: InputSecurityPolicy | None = None,
) -> ArchiveObject | None:
    """Returns the new `ArchiveObject`, or `None` if `discovered` is a duplicate (already-known
    content hash) — a duplicate is still recorded as `ArchiveObjectDiscovered`, just never
    registered twice.
    """
    original_filename = discovered.original_filename
    if security_policy is not None:
        original_filename = validate_source_file(
            discovered.path, original_filename=original_filename, policy=security_policy
        )
    content_hash = _content_hash(discovered.path)
    sink.append(
        ArchiveObjectDiscovered(
            event_id=new_id("event"),
            workspace_id=workspace_id,
            source_id=source_id,
            original_filename=original_filename,
            content_hash=content_hash,
        )
    )
    if content_hash in known_hashes:
        return None

    archive_object_id = new_id("archive_object")
    storage_name = f"{content_hash[:12]}_{original_filename}"
    destination = layout.archive_dir / storage_name
    layout.archive_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(discovered.path, destination)

    archive_object = ArchiveObject(
        id=archive_object_id,
        workspace_id=workspace_id,
        source_id=source_id,
        original_filename=original_filename,
        content_hash=content_hash,
        byte_size=discovered.byte_size,
        mime_type=mimetypes.guess_type(discovered.original_filename)[0] or "application/octet-stream",
        storage_path=storage_name,
        discovered_at=discovered.modified_at,
        registered_at=datetime.now(timezone.utc),
    )
    known_hashes.add(content_hash)
    sink.append(
        ArchiveObjectRegistered(
            event_id=new_id("event"), workspace_id=workspace_id, archive_object=archive_object
        )
    )
    return archive_object
