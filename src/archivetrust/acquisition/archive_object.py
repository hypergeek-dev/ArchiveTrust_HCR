"""`ArchiveObject` — the registered record of one immutable file inside a Workspace (ROADMAP.md
§5.13.2). Never modified once created; a re-scan of the same file (same content hash) is a
duplicate, not a new or updated Archive Object.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field


class ArchiveObject(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    workspace_id: str
    source_id: str
    original_filename: str
    content_hash: str
    """SHA-256 hex digest of the file's bytes, used for duplicate detection across sources."""
    byte_size: int
    mime_type: str
    storage_path: str
    """Path relative to the owning Workspace's `WorkspaceLayout.archive_dir`."""
    discovered_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    registered_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
