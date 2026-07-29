"""The `Workspace` domain type (ROADMAP.md §5.13.1)."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from archivetrust.runtime.provider_profiles import ProviderProfileName
from archivetrust.governance.policy import WorkspaceGovernance


class WorkspaceStatus(str, Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"


class Workspace(BaseModel):
    """A complete archival project — the highest-level operational unit. Immutable; changes (rename,
    archive, profile change) are made by replacing the stored record via `WorkspaceStore`, never by
    mutating an instance in place.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    description: str = ""
    processing_profile: ProviderProfileName = ProviderProfileName.ARCHIVE
    ontology_version: int = 1
    status: WorkspaceStatus = WorkspaceStatus.ACTIVE
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    governance: WorkspaceGovernance = Field(default_factory=WorkspaceGovernance)
