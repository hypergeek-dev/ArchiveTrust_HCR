from __future__ import annotations

import pytest

from archivetrust.admin.identity import AdminAction, AdminAuditLog, IdentityRole, LocalAccount, PermissionDenied
from archivetrust.presentation.workspace_viewmodel import WorkspaceManagerViewModel
from archivetrust.presentation.workspace_viewmodel import WorkspaceDeleteConfirmationError
from archivetrust.workspace.store import WorkspaceStore


class _FakeContext:
    def __init__(self, store: WorkspaceStore) -> None:
        self.current_workspace = None
        self._store = store

    def open_workspace(self, workspace_id: str):
        self.current_workspace = self._store.open(workspace_id)
        return self.current_workspace

    def create_workspace(self, name: str, description: str = ""):
        workspace = self._store.create(name, description=description)
        self.current_workspace = workspace
        return workspace


def test_workspaces_lists_every_workspace_and_marks_current(tmp_path) -> None:
    store = WorkspaceStore(app_root=tmp_path / "workspaces")
    a = store.create("Lund Municipality")
    store.create("Historical Church Records")
    context = _FakeContext(store)
    context.open_workspace(a.id)
    vm = WorkspaceManagerViewModel(store=store, context=context)

    rows = vm.workspaces()
    assert {r.name for r in rows} == {"Lund Municipality", "Historical Church Records"}
    assert next(r for r in rows if r.id == a.id).is_current is True


def test_create_opens_the_new_workspace(tmp_path) -> None:
    store = WorkspaceStore(app_root=tmp_path / "workspaces")
    context = _FakeContext(store)
    vm = WorkspaceManagerViewModel(store=store, context=context)

    created = vm.create("Research Dataset")
    assert context.current_workspace is not None
    assert context.current_workspace.id == created.id


def test_rename_clone_archive_delete(tmp_path) -> None:
    store = WorkspaceStore(app_root=tmp_path / "workspaces")
    context = _FakeContext(store)
    vm = WorkspaceManagerViewModel(store=store, context=context)
    workspace = vm.create("Legal Archive")

    renamed = vm.rename(workspace.id, "Legal Archive (Renamed)")
    assert renamed.name == "Legal Archive (Renamed)"

    clone = vm.clone(workspace.id, "Legal Archive Clone")
    assert clone.id != workspace.id

    archived = vm.archive(workspace.id)
    assert archived.status.value == "archived"

    vm.delete(clone.id)
    assert clone.id not in {w.id for w in vm.workspaces()}


def test_deleting_the_current_workspace_switches_to_another_remaining_one(tmp_path) -> None:
    """2026-07-16 incident: deleting the currently-open Workspace used to leave the context holding
    live state (telemetry sinks, etc.) built from paths under the now-deleted directory -- the next
    write anywhere crashed with FileNotFoundError instead of the delete just taking effect."""
    store = WorkspaceStore(app_root=tmp_path / "workspaces")
    context = _FakeContext(store)
    vm = WorkspaceManagerViewModel(store=store, context=context)
    other = vm.create("Kept Workspace")
    current = vm.create("Polluted Research Workspace")  # create() opens it -> now current
    assert context.current_workspace.id == current.id

    vm.delete(current.id)

    assert context.current_workspace is not None
    assert context.current_workspace.id == other.id  # switched to the only remaining workspace
    assert current.id not in {w.id for w in vm.workspaces()}


def test_deleting_the_only_workspace_falls_back_to_a_fresh_default(tmp_path) -> None:
    store = WorkspaceStore(app_root=tmp_path / "workspaces")
    context = _FakeContext(store)
    vm = WorkspaceManagerViewModel(store=store, context=context)
    only = vm.create("Polluted Research Workspace")

    vm.delete(only.id)

    assert context.current_workspace is not None
    assert context.current_workspace.id != only.id
    assert context.current_workspace.name == "Default Workspace"


def test_deleting_a_non_current_workspace_does_not_touch_the_open_one(tmp_path) -> None:
    store = WorkspaceStore(app_root=tmp_path / "workspaces")
    context = _FakeContext(store)
    vm = WorkspaceManagerViewModel(store=store, context=context)
    current = vm.create("Kept Open")
    other = vm.create("Deleted")
    context.open_workspace(current.id)  # switch back so `other` is not the current one

    vm.delete(other.id)

    assert context.current_workspace.id == current.id  # unchanged, no unnecessary switch


def test_confirmed_delete_requires_administrator_exact_name_and_audits(tmp_path) -> None:
    store = WorkspaceStore(app_root=tmp_path / "workspaces")
    context = _FakeContext(store)
    audit_log = AdminAuditLog(tmp_path / "admin-audit.jsonl")
    vm = WorkspaceManagerViewModel(store=store, context=context, audit_log=audit_log)
    workspace = vm.create("Legal Archive")
    admin = LocalAccount(
        display_name="Ada Administrator",
        reviewer_ref="reviewer.ada",
        roles=(IdentityRole.ADMINISTRATOR,),
    )

    with pytest.raises(WorkspaceDeleteConfirmationError):
        vm.delete_confirmed(workspace.id, confirmation="legal archive", actor=admin, reason="retention expired")

    vm.delete_confirmed(workspace.id, confirmation="Legal Archive", actor=admin, reason="retention expired")

    assert workspace.id not in {w.id for w in vm.workspaces()}
    event = audit_log.all_events()[0]
    assert event.actor_account_id == admin.account_id
    assert event.action is AdminAction.WORKSPACE_DELETED
    assert event.target_label == "Legal Archive"


def test_confirmed_delete_rejects_non_administrator(tmp_path) -> None:
    store = WorkspaceStore(app_root=tmp_path / "workspaces")
    context = _FakeContext(store)
    vm = WorkspaceManagerViewModel(store=store, context=context)
    workspace = vm.create("Legal Archive")
    operator = LocalAccount(
        display_name="Olivia Operator",
        reviewer_ref="reviewer.olivia",
        roles=(IdentityRole.OPERATOR,),
    )

    with pytest.raises(PermissionDenied):
        vm.delete_confirmed(workspace.id, confirmation="Legal Archive", actor=operator, reason="requested")

    assert workspace.id in {w.id for w in vm.workspaces()}
