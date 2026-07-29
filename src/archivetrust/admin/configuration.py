"""Deterministic configuration locking and environment reproducibility reports."""

from __future__ import annotations

import hashlib
import json
import platform
from importlib.metadata import distributions
from pathlib import Path

from pydantic import BaseModel, ConfigDict


LOCK_FILENAME = "configuration.lock.json"


class LockedFile(BaseModel):
    model_config = ConfigDict(frozen=True)

    relative_path: str
    size: int
    sha256: str


class ConfigurationLock(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_id: str = "archivetrust.configuration_lock.v1"
    workspace_id: str
    files: tuple[LockedFile, ...]
    configuration_hash: str


class ConfigurationDriftError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lockable_files(workspace_root: Path) -> tuple[Path, ...]:
    files: list[Path] = []
    for relative_root in ("config", "models"):
        root = workspace_root / relative_root
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.name != LOCK_FILENAME and not path.is_symlink():
                files.append(path)
    metadata = workspace_root / "workspace.json"
    if metadata.exists():
        files.append(metadata)
    return tuple(sorted(files, key=lambda value: value.relative_to(workspace_root).as_posix()))


def build_configuration_lock(workspace_root: Path, *, workspace_id: str) -> ConfigurationLock:
    rows = tuple(
        LockedFile(
            relative_path=path.relative_to(workspace_root).as_posix(),
            size=path.stat().st_size,
            sha256=_sha256(path),
        )
        for path in _lockable_files(workspace_root)
    )
    canonical = json.dumps(
        [row.model_dump(mode="json") for row in rows], sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return ConfigurationLock(
        workspace_id=workspace_id,
        files=rows,
        configuration_hash=hashlib.sha256(canonical).hexdigest(),
    )


def lock_path(workspace_root: Path) -> Path:
    return workspace_root / "config" / LOCK_FILENAME


def write_configuration_lock(workspace_root: Path, *, workspace_id: str, replace: bool = False) -> ConfigurationLock:
    destination = lock_path(workspace_root)
    if destination.exists() and not replace:
        return ConfigurationLock.model_validate_json(destination.read_text(encoding="utf-8"))
    manifest = build_configuration_lock(workspace_root, workspace_id=workspace_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(destination)
    return manifest


def verify_configuration_lock(workspace_root: Path) -> ConfigurationLock:
    destination = lock_path(workspace_root)
    if not destination.exists():
        raise ConfigurationDriftError("workspace has no configuration lock")
    expected = ConfigurationLock.model_validate_json(destination.read_text(encoding="utf-8"))
    actual = build_configuration_lock(workspace_root, workspace_id=expected.workspace_id)
    if actual.files != expected.files or actual.configuration_hash != expected.configuration_hash:
        raise ConfigurationDriftError(
            f"configuration drift detected: expected {expected.configuration_hash}, got {actual.configuration_hash}"
        )
    return expected


def environment_reproducibility_report() -> dict:
    packages = sorted(
        {
            str(dist.metadata.get("Name")): str(dist.version)
            for dist in distributions()
            if dist.metadata.get("Name")
        }.items()
    )
    return {
        "schema_id": "archivetrust.environment_report.v1",
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "operating_system": platform.platform(),
        "machine": platform.machine(),
        "packages": [{"name": name, "version": version} for name, version in packages],
        "secrets_included": False,
    }
