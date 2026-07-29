from __future__ import annotations

import pytest

from archivetrust.admin.backup import create_backup, restore_backup, verify_backup


def test_full_deployment_backup_restore_and_hash_verification(tmp_path) -> None:
    deployment = tmp_path / "deployment"
    files = {
        "workspaces/ws1/archive/record.pdf": b"%PDF-1.7 immutable",
        "workspaces/ws1/telemetry/events.jsonl": b'{"event":"one"}\n',
        "workspaces/ws1/evaluation/annotations.jsonl": b'{"annotation":"one"}\n',
        "workspaces/ws1/exports/release.json": b'{"release":"one"}',
        "identity/accounts.json": b"[]",
        "identity/admin-audit.jsonl": b"",
        "config/site.json": b'{"profile":"windows-workstation"}',
    }
    for relative, content in files.items():
        path = deployment / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    backup = tmp_path / "deployment.atbackup"

    created = create_backup(deployment, backup)
    assert verify_backup(backup) == created
    restored = tmp_path / "clean-machine" / "archivetrust_data"
    restored_manifest = restore_backup(backup, restored)

    assert restored_manifest == created
    for relative, content in files.items():
        assert (restored / relative).read_bytes() == content


def test_restore_refuses_nonempty_target(tmp_path) -> None:
    deployment = tmp_path / "deployment"
    deployment.mkdir()
    (deployment / "one.txt").write_text("one", encoding="utf-8")
    backup = tmp_path / "backup.zip"
    create_backup(deployment, backup)
    target = tmp_path / "target"
    target.mkdir()
    (target / "existing.txt").write_text("do not overwrite", encoding="utf-8")

    with pytest.raises(FileExistsError, match="absent or empty"):
        restore_backup(backup, target)
