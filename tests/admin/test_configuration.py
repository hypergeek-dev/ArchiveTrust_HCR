from __future__ import annotations

import pytest

from archivetrust.admin.configuration import (
    ConfigurationDriftError,
    environment_reproducibility_report,
    verify_configuration_lock,
    write_configuration_lock,
)


def test_configuration_lock_detects_drift_and_excludes_its_own_file(tmp_path) -> None:
    root = tmp_path / "workspace"
    (root / "config").mkdir(parents=True)
    (root / "models").mkdir()
    (root / "workspace.json").write_text('{"id":"ws1"}', encoding="utf-8")
    config = root / "config" / "policy.json"
    config.write_text('{"version":1}', encoding="utf-8")

    written = write_configuration_lock(root, workspace_id="ws1")
    assert verify_configuration_lock(root) == written
    assert all(row.relative_path != "config/configuration.lock.json" for row in written.files)

    config.write_text('{"version":2}', encoding="utf-8")
    with pytest.raises(ConfigurationDriftError, match="drift"):
        verify_configuration_lock(root)


def test_environment_report_contains_versions_not_secret_values() -> None:
    report = environment_reproducibility_report()
    assert report["python"]
    assert report["packages"]
    assert report["secrets_included"] is False
