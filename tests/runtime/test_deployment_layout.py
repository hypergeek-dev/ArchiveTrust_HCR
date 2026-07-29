from __future__ import annotations

from archivetrust.runtime.deployment_layout import DeploymentLayout


def test_ensure_creates_all_directories(tmp_path) -> None:
    layout = DeploymentLayout(root=tmp_path).ensure()
    for directory in layout.all_dirs():
        assert directory.is_dir()


def test_layout_names_match_the_documented_deployment_shape(tmp_path) -> None:
    layout = DeploymentLayout(root=tmp_path)
    assert layout.config_dir.name == "config"
    assert layout.models_dir.name == "models"
    assert layout.cache_dir.name == "cache"
    assert layout.logs_dir.name == "logs"
    assert layout.telemetry_dir.name == "telemetry"
    assert layout.plugins_dir.name == "plugins"


def test_ensure_is_idempotent(tmp_path) -> None:
    layout = DeploymentLayout(root=tmp_path)
    layout.ensure()
    layout.ensure()  # must not raise on already-existing directories
    assert layout.models_dir.is_dir()
