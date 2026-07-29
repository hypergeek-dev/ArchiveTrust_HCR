"""`WorkspaceStore` (ROADMAP.md §5.13.1): create/open/rename/clone/archive/delete Workspaces, each
persisted as `workspace.json` under its own root — the multi-Workspace analogue of
`ModelRegistry.save_bindings()`'s JSON-under-`config_dir` persistence.
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from archivetrust.workspace.layout import WorkspaceLayout
from archivetrust.workspace.models import Workspace, WorkspaceStatus
from archivetrust.runtime.provider_profiles import ProviderProfileName
from archivetrust.governance.policy import WorkspaceGovernance

_METADATA_FILENAME = "workspace.json"


class WorkspaceNotFoundError(KeyError):
    """Raised when a Workspace id has no corresponding record under the store's root."""


class WorkspaceStore:
    """`app_root` is the directory under which every Workspace gets its own subdirectory, named by
    its id (`app_root/<workspace-id>/...`) — this *is* that Workspace's `WorkspaceLayout.root`.
    """

    def __init__(self, app_root: Path) -> None:
        self._app_root = app_root
        self._app_root.mkdir(parents=True, exist_ok=True)

    def _root_for(self, workspace_id: str) -> Path:
        return self._app_root / workspace_id

    def _metadata_path(self, workspace_id: str) -> Path:
        return self._root_for(workspace_id) / _METADATA_FILENAME

    def _save(self, workspace: Workspace) -> Workspace:
        layout = WorkspaceLayout(root=self._root_for(workspace.id)).ensure()
        layout.config_dir.mkdir(parents=True, exist_ok=True)
        self._metadata_path(workspace.id).write_text(
            workspace.model_dump_json(indent=2), encoding="utf-8"
        )
        return workspace

    def create(
        self,
        name: str,
        description: str = "",
        processing_profile: ProviderProfileName = ProviderProfileName.ARCHIVE,
    ) -> Workspace:
        workspace = Workspace(
            id=uuid.uuid4().hex,
            name=name,
            description=description,
            processing_profile=processing_profile,
        )
        return self._save(workspace)

    def open(self, workspace_id: str) -> Workspace:
        path = self._metadata_path(workspace_id)
        if not path.exists():
            raise WorkspaceNotFoundError(workspace_id)
        return Workspace.model_validate_json(path.read_text(encoding="utf-8"))

    def layout_for(self, workspace_id: str) -> WorkspaceLayout:
        return WorkspaceLayout(root=self._root_for(workspace_id)).ensure()

    def rename(self, workspace_id: str, name: str) -> Workspace:
        workspace = self.open(workspace_id).model_copy(update={"name": name})
        return self._save(workspace)

    def set_processing_profile(
        self, workspace_id: str, processing_profile: ProviderProfileName
    ) -> Workspace:
        workspace = self.open(workspace_id).model_copy(
            update={"processing_profile": processing_profile}
        )
        return self._save(workspace)

    def archive(self, workspace_id: str) -> Workspace:
        workspace = self.open(workspace_id).model_copy(
            update={"status": WorkspaceStatus.ARCHIVED}
        )
        return self._save(workspace)

    def reactivate(self, workspace_id: str) -> Workspace:
        workspace = self.open(workspace_id).model_copy(update={"status": WorkspaceStatus.ACTIVE})
        return self._save(workspace)

    def set_governance(self, workspace_id: str, governance: WorkspaceGovernance) -> Workspace:
        workspace = self.open(workspace_id).model_copy(update={"governance": governance})
        return self._save(workspace)

    def clone(self, workspace_id: str, new_name: str) -> Workspace:
        """Clones configuration only (name, description, processing profile) — never archive
        contents or telemetry history, which belong to the original Workspace alone (Article 25:
        Archive Objects are never duplicated behind the operator's back).
        """
        source = self.open(workspace_id)
        return self.create(
            name=new_name,
            description=source.description,
            processing_profile=source.processing_profile,
        )

    def delete(self, workspace_id: str) -> None:
        root = self._root_for(workspace_id)
        if not root.exists():
            raise WorkspaceNotFoundError(workspace_id)
        self.open(workspace_id).governance.assert_permanent_deletion_allowed()
        shutil.rmtree(root)

    def list(self) -> tuple[Workspace, ...]:
        workspaces = []
        for entry in sorted(self._app_root.iterdir()):
            metadata = entry / _METADATA_FILENAME
            if metadata.exists():
                workspaces.append(Workspace.model_validate_json(metadata.read_text(encoding="utf-8")))
        return tuple(workspaces)
