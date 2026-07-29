"""`AcquisitionSource` — the shared interface every acquisition mechanism implements (ROADMAP.md
§5.13.2), mirroring the provider-independence discipline Article 20 already imposes on Observation
producers, applied one stage upstream to file producers.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Iterable, Protocol

from pydantic import BaseModel, ConfigDict


class AcquisitionSourceKind(str, Enum):
    MANUAL_IMPORT = "manual_import"
    FOLDER_WATCH = "folder_watch"
    NETWORK_SHARE = "network_share"
    EMAIL_INBOX = "email_inbox"
    REST_API = "rest_api"
    SHAREPOINT = "sharepoint"
    ZIP_IMPORT = "zip_import"
    CLOUD_STORAGE = "cloud_storage"


IMPLEMENTED_SOURCE_KINDS: tuple[AcquisitionSourceKind, ...] = (
    AcquisitionSourceKind.MANUAL_IMPORT,
    AcquisitionSourceKind.FOLDER_WATCH,
)
"""The only kinds this revision implements (ROADMAP.md §5.13.2). Every other kind is a real,
selectable, honestly-labeled "not yet connected" entry in the Acquisition Manager UI — never a fake
integration."""


class DiscoveredFile(BaseModel):
    """One candidate file a source has found, prior to hashing/dedup/registration."""

    model_config = ConfigDict(frozen=True)

    path: Path
    original_filename: str
    byte_size: int
    modified_at: datetime


class SourceHealth(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: str
    """One of "ok", "degraded", "error" — kept as a plain label rather than an enum so a new source
    kind can report a status without a code change here."""
    last_scan_at: datetime | None = None
    documents_imported: int = 0
    error_count: int = 0
    last_error: str | None = None


class AcquisitionSource(Protocol):
    source_id: str
    kind: AcquisitionSourceKind
    enabled: bool

    def scan(self, known_hashes: frozenset[str]) -> Iterable[DiscoveredFile]:
        """Returns candidate files not already known by content hash. Never modifies or removes the
        original file (Article 25).
        """
        ...

    def health(self) -> SourceHealth:
        ...
