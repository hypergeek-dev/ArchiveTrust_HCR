"""Per-Workspace storage layout (ROADMAP.md §5.13.1): `input/ archive/ derived/ telemetry/ logs/
cache/ models/ config/ reports/` beneath one Workspace's own root.

Wraps the existing `runtime.deployment_layout.DeploymentLayout` — reused unmodified for
`config/models/cache/logs/telemetry/plugins` — rather than reimplementing it, and adds the
Workspace-specific directories it doesn't have (`input/ archive/ derived/ reports/`).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from archivetrust.runtime.deployment_layout import DeploymentLayout


class WorkspaceLayout(BaseModel):
    model_config = ConfigDict(frozen=True)

    root: Path

    @property
    def deployment_layout(self) -> DeploymentLayout:
        return DeploymentLayout(root=self.root)

    @property
    def config_dir(self) -> Path:
        return self.deployment_layout.config_dir

    @property
    def models_dir(self) -> Path:
        return self.deployment_layout.models_dir

    @property
    def cache_dir(self) -> Path:
        return self.deployment_layout.cache_dir

    @property
    def logs_dir(self) -> Path:
        return self.deployment_layout.logs_dir

    @property
    def telemetry_dir(self) -> Path:
        return self.deployment_layout.telemetry_dir

    @property
    def plugins_dir(self) -> Path:
        return self.deployment_layout.plugins_dir

    @property
    def input_dir(self) -> Path:
        return self.root / "input"

    @property
    def archive_dir(self) -> Path:
        return self.root / "archive"

    @property
    def derived_dir(self) -> Path:
        return self.root / "derived"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    def all_dirs(self) -> tuple[Path, ...]:
        return self.deployment_layout.all_dirs() + (
            self.input_dir,
            self.archive_dir,
            self.derived_dir,
            self.reports_dir,
        )

    def ensure(self) -> "WorkspaceLayout":
        for directory in self.all_dirs():
            directory.mkdir(parents=True, exist_ok=True)
        return self
