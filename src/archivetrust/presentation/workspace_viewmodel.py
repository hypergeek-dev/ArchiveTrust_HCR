"""Workspace Manager ViewModel (ROADMAP.md §5.13.1) — list/create/open/rename/clone/archive/delete/
switch. Framework-independent; the `context` dependency is expressed as a local `Protocol` rather
than importing `clients.desktop.composition.AppContext` directly, since `presentation` may never
import `clients` (`tests/review/test_separation.py`).
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, ConfigDict

from archivetrust.admin.identity import (
    AdminAction,
    AdminAuditLog,
    IdentityRole,
    LocalAccount,
    Permission,
    PermissionDenied,
    require_permission,
    require_role,
)
from archivetrust.runtime.provider_profiles import ProviderProfileName
from archivetrust.workspace.models import Workspace
from archivetrust.workspace.store import WorkspaceStore


class SwitchesWorkspace(Protocol):
    def open_workspace(self, workspace_id: str) -> Workspace: ...
    def create_workspace(self, name: str, description: str = "") -> Workspace: ...
    current_workspace: Workspace | None


class WorkspaceRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    description: str
    processing_profile: str
    status: str
    is_current: bool


class WorkspaceDeleteConfirmationError(ValueError):
    """Raised when a destructive workspace action is not confirmed with the workspace name."""


class WorkspaceManagerViewModel:
    def __init__(
        self,
        *,
        store: WorkspaceStore,
        context: SwitchesWorkspace,
        audit_log: AdminAuditLog | None = None,
        actor: LocalAccount | None = None,
        session_id: str | None = None,
    ) -> None:
        self._store = store
        self._context = context
        self._audit_log = audit_log
        self._actor = actor
        self._session_id = session_id

    def _authorize(self, permission: Permission) -> LocalAccount | None:
        if self._actor is not None:
            require_permission(self._actor, permission)
        return self._actor

    def _audit(
        self,
        action: AdminAction,
        workspace: Workspace,
        *,
        prior_state_ref: str | None = None,
        resulting_state_ref: str | None = None,
        reason: str | None = None,
        success: bool = True,
    ) -> None:
        if self._audit_log is not None and self._actor is not None:
            self._audit_log.record(
                actor=self._actor,
                action=action,
                target_ref=workspace.id,
                target_label=workspace.name,
                prior_state_ref=prior_state_ref,
                resulting_state_ref=resulting_state_ref,
                reason=reason,
                success=success,
                session_id=self._session_id,
            )

    def workspaces(self) -> tuple[WorkspaceRow, ...]:
        current = getattr(self._context, "current_workspace", None)
        current_id = current.id if current is not None else None
        return tuple(
            WorkspaceRow(
                id=w.id,
                name=w.name,
                description=w.description,
                processing_profile=w.processing_profile.value,
                status=w.status.value,
                is_current=(w.id == current_id),
            )
            for w in self._store.list()
        )

    def create(
        self, name: str, description: str = "", processing_profile: ProviderProfileName = ProviderProfileName.ARCHIVE
    ) -> Workspace:
        self._authorize(Permission.WORKSPACE_CREATE)
        workspace = self._store.create(name, description=description, processing_profile=processing_profile)
        self._context.open_workspace(workspace.id)
        self._audit(AdminAction.WORKSPACE_CREATED, workspace, resulting_state_ref=workspace.id)
        return workspace

    def open(self, workspace_id: str) -> Workspace:
        return self._context.open_workspace(workspace_id)

    def rename(self, workspace_id: str, name: str) -> Workspace:
        self._authorize(Permission.WORKSPACE_CREATE)
        prior = self._store.open(workspace_id)
        result = self._store.rename(workspace_id, name)
        self._audit(AdminAction.WORKSPACE_RENAMED, result, prior_state_ref=prior.name, resulting_state_ref=result.name)
        return result

    def clone(self, workspace_id: str, new_name: str) -> Workspace:
        self._authorize(Permission.WORKSPACE_CREATE)
        result = self._store.clone(workspace_id, new_name)
        self._audit(AdminAction.WORKSPACE_CREATED, result, prior_state_ref=workspace_id, resulting_state_ref=result.id)
        return result

    def archive(self, workspace_id: str) -> Workspace:
        self._authorize(Permission.WORKSPACE_RETIRE)
        result = self._store.archive(workspace_id)
        self._audit(AdminAction.WORKSPACE_RETIRED, result, prior_state_ref="active", resulting_state_ref="archived")
        return result

    def reactivate(self, workspace_id: str) -> Workspace:
        self._authorize(Permission.WORKSPACE_RETIRE)
        result = self._store.reactivate(workspace_id)
        self._audit(AdminAction.WORKSPACE_REACTIVATED, result, prior_state_ref="archived", resulting_state_ref="active")
        return result

    def delete(self, workspace_id: str) -> None:
        """Deletes the Workspace's files from disk. If it was the currently *open* Workspace, the
        context is immediately switched to another remaining Workspace (or a fresh "Default
        Workspace" is created if none remain) -- mirroring the exact fallback `AppContext.__init__`
        already uses on startup. Without this, the context would keep holding live telemetry/
        acquisition/model-registry state built from paths under the now-deleted directory tree,
        and the next write anywhere (e.g. the background processing queue) would crash with a
        `FileNotFoundError` instead of the delete simply taking effect (2026-07-16 incident: a real
        deleted-while-current Workspace crashed `QueueWorker._record_progress` this way).
        """
        if self._actor is not None:
            raise PermissionDenied("permanent deletion requires delete_confirmed with administrator, exact name, and reason")
        self._delete_unchecked(workspace_id)

    def _delete_unchecked(self, workspace_id: str) -> None:
        current = getattr(self._context, "current_workspace", None)
        was_current = current is not None and current.id == workspace_id
        self._store.delete(workspace_id)
        if not was_current:
            return
        remaining = self._store.list()
        if remaining:
            self._context.open_workspace(remaining[0].id)
        else:
            self._context.create_workspace("Default Workspace")

    def delete_confirmed(
        self,
        workspace_id: str,
        *,
        confirmation: str,
        actor: LocalAccount,
        reason: str,
    ) -> None:
        """A5 destructive-action path: only administrators can permanently delete a Workspace,
        and the confirmation must be the exact Workspace name. Ordinary operators should use
        `archive()`/Retire by default.
        """
        require_role(actor, IdentityRole.ADMINISTRATOR)
        if self._actor is not None and actor.account_id != self._actor.account_id:
            raise PermissionDenied("deletion actor must match the authenticated session")
        workspace = self._store.open(workspace_id)
        if not reason.strip():
            self._audit(
                AdminAction.WORKSPACE_DELETED,
                workspace,
                reason="deletion rejected: reason required",
                success=False,
            )
            raise WorkspaceDeleteConfirmationError("a deletion reason is required")
        if confirmation != workspace.name:
            self._audit(
                AdminAction.WORKSPACE_DELETED,
                workspace,
                reason="deletion rejected: confirmation mismatch",
                success=False,
            )
            raise WorkspaceDeleteConfirmationError("confirmation must exactly match workspace name")
        try:
            self._delete_unchecked(workspace_id)
        except PermissionError as exc:
            self._audit(
                AdminAction.WORKSPACE_DELETED,
                workspace,
                reason=f"deletion rejected: {exc}",
                success=False,
            )
            raise
        if self._audit_log is not None:
            self._audit_log.record(
                actor=actor,
                action=AdminAction.WORKSPACE_DELETED,
                target_ref=workspace_id,
                target_label=workspace.name,
                prior_state_ref=workspace.id,
                resulting_state_ref="deleted",
                reason=reason.strip(),
                session_id=self._session_id,
            )
