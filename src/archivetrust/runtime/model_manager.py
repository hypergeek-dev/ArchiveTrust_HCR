"""The Model Manager (Part 11): detects installed models, validates compatibility, and reports
disk usage, size, backend, revision, and a hash when cheaply available. Read-only and local —
downloading is explicitly out of scope here (`ModelSource` in `models.py` is the seam a future
downloader would consume), so nothing in this module makes a network call.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from archivetrust.runtime.contracts import RuntimeCapabilities
from archivetrust.runtime.deployment_layout import DeploymentLayout
from archivetrust.runtime.models import (
    InstalledModel,
    ModelDescriptor,
    ModelSource,
    ModelSourceKind,
)

_MANIFEST_FILENAME = "archivetrust_model.json"
"""Each installed model directory may carry a small manifest naming its descriptor; a directory
with no manifest is still reported (Part 11: "detect installed models"), just with a
best-effort descriptor derived from its directory name, distinguished from a fully-declared one."""

_HASH_SIZE_LIMIT_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB
"""Above this size, computing a whole-directory sha256 is judged prohibitively slow for routine
Model Manager display; `InstalledModel.sha256` is left `None` rather than blocking the UI on a
multi-minute hash (Part 11: "display hashes when available" -- never fabricated, never blocking)."""


class IncompatibleModelError(ValueError):
    """Raised by `validate_compatibility` when an installed model cannot be used with a given
    runtime kind's declared capabilities."""


def _directory_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _sha256_of_directory(path: Path) -> str | None:
    size = _directory_size(path)
    if size > _HASH_SIZE_LIMIT_BYTES:
        return None
    digest = hashlib.sha256()
    for file_path in sorted(path.rglob("*")):
        if file_path.is_file():
            digest.update(file_path.name.encode("utf-8"))
            digest.update(file_path.read_bytes())
    return digest.hexdigest()


def scan_installed_models(layout: DeploymentLayout, *, compute_hashes: bool = True) -> tuple[InstalledModel, ...]:
    """Detects every model directory under `layout.models_dir` (Part 11). Each immediate
    subdirectory of `models_dir` is treated as one installed model — this mirrors how a browsed-to
    local model (Part 12's "Browse for local model") is expected to be laid out: one directory per
    model, containing whatever files its runtime needs.
    """
    if not layout.models_dir.exists():
        return ()
    installed: list[InstalledModel] = []
    for entry in sorted(layout.models_dir.iterdir()):
        if not entry.is_dir():
            continue
        descriptor = _describe(entry)
        size = _directory_size(entry)
        digest = _sha256_of_directory(entry) if compute_hashes else None
        installed.append(
            InstalledModel(descriptor=descriptor, local_path=entry, size_bytes=size, sha256=digest)
        )
    return tuple(installed)


def _describe(model_dir: Path) -> ModelDescriptor:
    manifest_path = model_dir / _MANIFEST_FILENAME
    if manifest_path.exists():
        import json

        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        return ModelDescriptor(
            model_id=data["model_id"],
            display_name=data.get("display_name", data["model_id"]),
            source=ModelSource(
                kind=ModelSourceKind.LOCAL_PATH, identifier=model_dir.name, revision=data.get("revision")
            ),
            runtime_kind=data["runtime_kind"],
        )
    # No manifest: a best-effort descriptor from the directory name alone, so the model is still
    # visible (Part 11) rather than silently skipped -- runtime_kind is left as "unknown" so a
    # settings UI can flag it for the operator to configure, never guessed.
    return ModelDescriptor(
        model_id=model_dir.name,
        display_name=model_dir.name,
        source=ModelSource(kind=ModelSourceKind.LOCAL_PATH, identifier=model_dir.name),
        runtime_kind="unknown",
    )


def validate_compatibility(
    installed: InstalledModel, capabilities: RuntimeCapabilities
) -> None:
    """Raises `IncompatibleModelError` if this model's declared runtime cannot be resolved against
    the given capabilities (e.g. a model whose manifest requires GPU-only inference paired with a
    CPU-only runtime configuration). A no-op (returns normally) when compatible.
    """
    if installed.descriptor.runtime_kind == "unknown":
        raise IncompatibleModelError(
            f"model at {installed.local_path} has no declared runtime_kind "
            f"(missing {_MANIFEST_FILENAME}) -- cannot validate compatibility"
        )
    if not (capabilities.supports_gpu or capabilities.supports_cpu):
        raise IncompatibleModelError("runtime declares support for neither GPU nor CPU execution")
