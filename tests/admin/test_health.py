from __future__ import annotations

import os
import zipfile

from archivetrust.admin.health import create_diagnostic_bundle, deployment_health


def test_health_and_diagnostic_bundle_are_sanitized(tmp_path) -> None:
    root = tmp_path / "deployment"
    workspace = root / "workspaces" / "ws1"
    (workspace / "archive").mkdir(parents=True)
    (workspace / "archive" / "secret.pdf").write_bytes(b"raw document")
    (workspace / "telemetry" / "blobs").mkdir(parents=True)
    (workspace / "telemetry" / "events.jsonl").write_text("", encoding="utf-8")
    (workspace / "config").mkdir()
    (workspace / "workspace.json").write_text('{"id":"ws1"}', encoding="utf-8")

    health = deployment_health(root)
    assert any(check.name == "storage_capacity" for check in health.checks)
    assert any(check.name == "acl_policy" for check in health.checks)
    bundle = tmp_path / "diagnostics.zip"
    create_diagnostic_bundle(root, bundle)

    expected_names = {
        "health.json",
        "environment.json",
        "worker-security.json",
        "configuration-manifests.json",
        "recent-worker-failures.json",
        "CONTENT.txt",
    }
    if os.name == "nt":
        expected_names.add("acl-policy.json")
    with zipfile.ZipFile(bundle) as archive:
        names = set(archive.namelist())
        assert names == expected_names
        combined = b"".join(archive.read(name) for name in names)
        assert b"raw document" not in combined
        assert b"secret.pdf" not in combined
