from __future__ import annotations

from archivetrust.workspace.layout import WorkspaceLayout


def test_ensure_creates_every_directory(tmp_path) -> None:
    layout = WorkspaceLayout(root=tmp_path / "ws1").ensure()
    for directory in layout.all_dirs():
        assert directory.exists() and directory.is_dir()


def test_layout_adds_workspace_specific_dirs_beyond_deployment_layout(tmp_path) -> None:
    layout = WorkspaceLayout(root=tmp_path / "ws1")
    names = {d.name for d in layout.all_dirs()}
    assert {"input", "archive", "derived", "reports", "config", "models", "cache", "logs", "telemetry", "plugins"} <= names


def test_ensure_is_idempotent(tmp_path) -> None:
    layout = WorkspaceLayout(root=tmp_path / "ws1")
    layout.ensure()
    layout.ensure()  # must not raise
    assert layout.archive_dir.exists()
