"""External deployment layout (Part 4, Part 14): the executable stays generic; models, config,
cache, logs, telemetry, and plugins live beside it, on disk, and are never compiled in.

```
ArchiveTrust/
    ArchiveTrust.exe
    config/          # provider configuration, provider profiles, prompt templates
    models/          # installed model files (Model Manager, Part 11)
    cache/           # runtime download/activation cache
    logs/
    telemetry/       # durable FileTelemetrySink output (ROADMAP.md S11)
    plugins/         # future third-party provider adapters
```

This is what makes the enterprise story (Part 14) real: an operator upgrades a model, replaces a
provider, or moves models to a different disk by changing files and configuration under this
layout — never by rebuilding the executable. `DeploymentLayout` is the single place that computes
these paths, so nothing else in the codebase hardcodes a directory name.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict


class DeploymentLayout(BaseModel):
    """Resolved, existing (on `ensure()`) directories for one ArchiveTrust installation.

    `root` defaults to the current working directory's `archivetrust_data` in tests/dev, and to the
    executable's own directory in a packaged deployment (the caller decides `root`; this type only
    computes the fixed sub-layout beneath it — see module docstring).
    """

    model_config = ConfigDict(frozen=True)

    root: Path

    @property
    def config_dir(self) -> Path:
        return self.root / "config"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def cache_dir(self) -> Path:
        return self.root / "cache"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def telemetry_dir(self) -> Path:
        return self.root / "telemetry"

    @property
    def plugins_dir(self) -> Path:
        return self.root / "plugins"

    def all_dirs(self) -> tuple[Path, ...]:
        return (
            self.config_dir,
            self.models_dir,
            self.cache_dir,
            self.logs_dir,
            self.telemetry_dir,
            self.plugins_dir,
        )

    def ensure(self) -> "DeploymentLayout":
        """Creates every directory in the layout if missing. Idempotent; safe to call on every
        startup (First Launch, Part 12, and every later launch alike)."""
        for directory in self.all_dirs():
            directory.mkdir(parents=True, exist_ok=True)
        return self
