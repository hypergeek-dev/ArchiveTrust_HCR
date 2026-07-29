from __future__ import annotations

import pytest

from archivetrust.governance.policy import GovernanceViolation, WorkspaceGovernance
from archivetrust.runtime.provider_profiles import ProviderProfileName
from archivetrust.workspace.models import WorkspaceStatus
from archivetrust.workspace.store import WorkspaceNotFoundError, WorkspaceStore


def _store(tmp_path):
    return WorkspaceStore(app_root=tmp_path / "workspaces")


def test_create_persists_and_round_trips(tmp_path) -> None:
    store = _store(tmp_path)
    created = store.create("Lund Municipality", description="Church records", processing_profile=ProviderProfileName.RESEARCH)
    reopened = store.open(created.id)
    assert reopened == created
    assert reopened.processing_profile == ProviderProfileName.RESEARCH


def test_open_missing_workspace_raises(tmp_path) -> None:
    store = _store(tmp_path)
    with pytest.raises(WorkspaceNotFoundError):
        store.open("nonexistent")


def test_rename_persists(tmp_path) -> None:
    store = _store(tmp_path)
    created = store.create("Old Name")
    renamed = store.rename(created.id, "New Name")
    assert renamed.name == "New Name"
    assert store.open(created.id).name == "New Name"


def test_archive_and_reactivate(tmp_path) -> None:
    store = _store(tmp_path)
    created = store.create("Historical Archive")
    archived = store.archive(created.id)
    assert archived.status == WorkspaceStatus.ARCHIVED
    reactivated = store.reactivate(created.id)
    assert reactivated.status == WorkspaceStatus.ACTIVE


def test_legal_hold_blocks_permanent_deletion(tmp_path) -> None:
    store = _store(tmp_path)
    workspace = store.create("Records under litigation hold")
    store.set_governance(
        workspace.id,
        WorkspaceGovernance(legal_hold=True, legal_hold_reason="case 2026-17"),
    )

    with pytest.raises(GovernanceViolation, match="legal hold"):
        store.delete(workspace.id)
    assert store.open(workspace.id).name == workspace.name


def test_clone_copies_configuration_not_content(tmp_path) -> None:
    store = _store(tmp_path)
    original = store.create("Original", description="desc", processing_profile=ProviderProfileName.MAXIMUM_QUALITY)
    clone = store.clone(original.id, "Clone")
    assert clone.id != original.id
    assert clone.description == original.description
    assert clone.processing_profile == original.processing_profile


def test_delete_removes_workspace(tmp_path) -> None:
    store = _store(tmp_path)
    created = store.create("Temp")
    store.delete(created.id)
    with pytest.raises(WorkspaceNotFoundError):
        store.open(created.id)


def test_list_returns_every_workspace(tmp_path) -> None:
    store = _store(tmp_path)
    store.create("A")
    store.create("B")
    assert {w.name for w in store.list()} == {"A", "B"}
