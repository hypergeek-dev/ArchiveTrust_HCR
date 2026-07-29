"""Model descriptors — the vocabulary the Model Registry and Model Manager both speak.

A `ModelDescriptor` names *what* a model is (id, revision, source); a `ResolvedModel` is what the
Model Registry hands back after resolving a logical identifier (Part 3) — the descriptor plus the
concrete runtime kind, device, precision, and on-disk location to actually run it. Providers only
ever see a `ResolvedModel`; they never construct one themselves (Part 3: "Providers must never
resolve model paths themselves").
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from archivetrust.runtime.contracts import DeviceSelection, Precision


class ModelSourceKind(str, Enum):
    """Where a model's files come from (Part 12's setup-wizard options)."""

    LOCAL_PATH = "local_path"
    HUGGING_FACE = "hugging_face"
    ENTERPRISE_REPOSITORY = "enterprise_repository"
    NETWORK_LOCATION = "network_location"


class ModelSource(BaseModel):
    """Where to obtain a model, and how to name it once obtained. Downloading itself is out of
    scope (Part 11: "design the architecture for future downloading, but downloading itself is
    optional") — this type is the seam a future downloader would consume; nothing here fetches.
    """

    model_config = ConfigDict(frozen=True)

    kind: ModelSourceKind
    identifier: str
    """A path (LOCAL_PATH), a repo id (HUGGING_FACE / ENTERPRISE_REPOSITORY), or a URL
    (NETWORK_LOCATION) — interpretation is `kind`-specific, never guessed."""
    revision: str | None = None


class ModelDescriptor(BaseModel):
    """Identifies one model, independent of where its files currently live on disk."""

    model_config = ConfigDict(frozen=True)

    model_id: str
    """A short, stable name ("qwen2.5-vl-7b-instruct") — recorded in telemetry provenance."""
    display_name: str
    source: ModelSource
    runtime_kind: str
    """Which `InferenceRuntime` implementation this model requires ("transformers",
    "openai_compatible", ...) — resolved by the Model Registry, never guessed by a provider."""


class InstalledModel(BaseModel):
    """A model whose files were actually found on disk (Model Manager, Part 11) — descriptor plus
    what was observed there: size, a content hash where cheaply available, and the resolved local
    path. `sha256` is `None` when computing it would be prohibitively slow for a large model
    directory (Part 11: "display hashes when available" — never fabricated when absent).
    """

    model_config = ConfigDict(frozen=True)

    descriptor: ModelDescriptor
    local_path: Path
    size_bytes: int
    sha256: str | None = None


class ResolvedModel(BaseModel):
    """What the Model Registry hands a runtime constructor: a model plus how to run it. This is
    the only thing a provider-facing bridge (e.g. `providers/qwen_vl/runtime_backend.py`) ever
    receives — it never independently resolves a path, a device, or a precision.
    """

    model_config = ConfigDict(frozen=True)

    descriptor: ModelDescriptor
    local_path: Path | None
    """`None` for a remote runtime (e.g. an OpenAI-compatible endpoint) where there is no local
    model directory to point at."""
    device: DeviceSelection
    precision: Precision
    cache_dir: Path
