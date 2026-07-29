"""Verified full-deployment backup and clean-target restore."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, ConfigDict


MANIFEST_NAME = "backup-manifest.json"


class BackupFile(BaseModel):
    model_config = ConfigDict(frozen=True)

    relative_path: str
    size: int
    sha256: str


class BackupManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_id: str = "archivetrust.backup.v1"
    created_at: str
    files: tuple[BackupFile, ...]
    content_hash: str
    encryption: str = "none"


def _sha256_stream(handle) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _sha256_path(path: Path) -> str:
    with path.open("rb") as handle:
        return _sha256_stream(handle)


def _manifest_hash(files: tuple[BackupFile, ...]) -> str:
    payload = json.dumps(
        [item.model_dump(mode="json") for item in files], sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def create_backup(deployment_root: Path, destination: Path) -> BackupManifest:
    source = deployment_root.resolve()
    destination = destination.resolve()
    if not source.is_dir():
        raise ValueError("deployment root does not exist")
    if destination == source or source in destination.parents:
        raise ValueError("backup destination must be outside the deployment root")
    if destination.exists():
        raise FileExistsError(destination)
    paths = tuple(
        sorted(
            (path for path in source.rglob("*") if path.is_file() and not path.is_symlink()),
            key=lambda path: path.relative_to(source).as_posix(),
        )
    )
    files = tuple(
        BackupFile(
            relative_path=path.relative_to(source).as_posix(),
            size=path.stat().st_size,
            sha256=_sha256_path(path),
        )
        for path in paths
    )
    manifest = BackupManifest(
        created_at=datetime.now(timezone.utc).isoformat(),
        files=files,
        content_hash=_manifest_hash(files),
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
            for path, row in zip(paths, files, strict=True):
                archive.write(path, row.relative_path)
            archive.writestr(MANIFEST_NAME, manifest.model_dump_json(indent=2))
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return manifest


def _safe_member(name: str) -> PurePosixPath:
    if "\\" in name:
        raise ValueError(f"unsafe backup member {name!r}")
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"unsafe backup member {name!r}")
    return relative


def restore_backup(backup_path: Path, target_root: Path) -> BackupManifest:
    target = target_root.resolve()
    if target.exists() and any(target.iterdir()):
        raise FileExistsError("restore target must be absent or empty")
    with zipfile.ZipFile(backup_path, "r") as archive:
        names = archive.namelist()
        if MANIFEST_NAME not in names:
            raise ValueError("backup manifest is missing")
        manifest = BackupManifest.model_validate_json(archive.read(MANIFEST_NAME))
        if manifest.content_hash != _manifest_hash(manifest.files):
            raise ValueError("backup manifest hash mismatch")
        expected = {row.relative_path: row for row in manifest.files}
        actual = set(names) - {MANIFEST_NAME}
        if actual != set(expected):
            raise ValueError("backup members do not match the manifest")
        for name in sorted(actual):
            relative = _safe_member(name)
            destination = target.joinpath(*relative.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(name, "r") as source, destination.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            row = expected[name]
            if destination.stat().st_size != row.size or _sha256_path(destination) != row.sha256:
                raise ValueError(f"restored content hash mismatch for {name}")
    return manifest


def verify_backup(backup_path: Path) -> BackupManifest:
    with zipfile.ZipFile(backup_path, "r") as archive:
        manifest = BackupManifest.model_validate_json(archive.read(MANIFEST_NAME))
        if manifest.content_hash != _manifest_hash(manifest.files):
            raise ValueError("backup manifest hash mismatch")
        for row in manifest.files:
            _safe_member(row.relative_path)
            with archive.open(row.relative_path, "r") as handle:
                if _sha256_stream(handle) != row.sha256:
                    raise ValueError(f"backup content hash mismatch for {row.relative_path}")
        return manifest
