"""The Model Registry (Part 3): resolves a logical identifier ("Primary Vision Provider") to a
runtime, model, revision, location, device, and cache — the one place path/device/runtime
resolution happens. Providers request inference through a runtime; they never resolve any of this
themselves.

`RuntimeFactory` is the seam a deployment registers concrete runtime constructors through — the
registry itself never imports `TransformersRuntime` or `OpenAICompatibleRuntime` directly, so a
third-party `plugins/` runtime can be registered identically (Part 14: "the executable remains
generic"). Resolution and compatibility-checking are separated from runtime *construction* so both
are independently testable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Protocol

from pydantic import BaseModel, ConfigDict, TypeAdapter

from archivetrust.runtime.contracts import (
    DeviceSelection,
    InferenceRuntime,
    LayoutDetectionRuntime,
    Precision,
)
from archivetrust.runtime.deployment_layout import DeploymentLayout
from archivetrust.runtime.models import ModelDescriptor, ModelSourceKind, ResolvedModel


class UnknownLogicalModelError(KeyError):
    """Raised when a logical identifier ("Primary Vision Provider") has no registered binding."""


class UnsupportedRuntimeError(KeyError):
    """Raised when a `ModelDescriptor.runtime_kind` has no registered `RuntimeFactory` — a
    deployment configuration error, never silently substituted with a different runtime.
    """


class IncompatibleModelError(ValueError):
    """Raised when a requested device/precision is not supported by the resolved runtime kind
    (checked against `RuntimeCapabilities` at construction time, before any inference is attempted).
    """


class RuntimeFactory(Protocol):
    """Constructs a concrete runtime from a `ResolvedModel`. One factory per `runtime_kind`,
    registered by a deployment (`ModelRegistry.register_runtime_factory`). Returns either an
    `InferenceRuntime` (image+prompt->text runtimes: transformers/vllm/...) or a
    `LayoutDetectionRuntime` (Phase 31: PP-DocLayoutV2's `"paddle_layout"` kind, image->structured
    regions) — `ModelRegistry.create_runtime` itself never branches on which; both expose the same
    generic `capabilities()`/`warm_up()`/`shut_down()` members it actually uses.
    """

    def __call__(self, resolved: ResolvedModel) -> InferenceRuntime | LayoutDetectionRuntime:
        ...


class LogicalModelBinding(BaseModel):
    """What a logical identifier currently points at — the thing an operator changes via
    configuration (Provider Manager / General Settings), never by editing code."""

    model_config = ConfigDict(frozen=True)

    logical_name: str
    descriptor: ModelDescriptor
    device: DeviceSelection = DeviceSelection.AUTOMATIC
    precision: Precision = Precision.AUTOMATIC
    provider_id: str = "qwen2.5-vl"
    """Which registered `ProviderAdapter` this binding activates (Multi-Provider Activation
    milestone) — the logical name ("Primary Vision Provider") identifies the *slot*; this field
    identifies *which VLM adapter* fills it, so the composition layer can construct the matching
    adapter instead of assuming one hardcoded class. Defaults to `"qwen2.5-vl"` for backward
    compatibility: every binding this codebase ever persisted before this field existed was a Qwen
    binding (the only VLM adapter that existed), so an old `model_bindings.json` on disk — lacking
    this field entirely — resolves to exactly the adapter it always meant, never silently
    reinterpreted as the new default."""


class ModelRegistry:
    """Resolves logical identifiers to `ResolvedModel`s and constructs runtimes for them."""

    def __init__(self, layout: DeploymentLayout) -> None:
        self._layout = layout
        self._bindings: dict[str, LogicalModelBinding] = {}
        self._runtime_factories: dict[str, RuntimeFactory] = {}

    def register_runtime_factory(self, runtime_kind: str, factory: RuntimeFactory) -> None:
        self._runtime_factories[runtime_kind] = factory

    def bind(self, binding: LogicalModelBinding) -> None:
        """Points a logical identifier at a model + device + precision. Re-binding (e.g. after an
        operator changes Provider Manager configuration) simply overwrites the prior binding —
        there is nothing to migrate, since a binding carries no state of its own.
        """
        self._bindings[binding.logical_name] = binding

    def bindings(self) -> tuple[LogicalModelBinding, ...]:
        return tuple(self._bindings.values())

    def bindings_path(self) -> Path:
        """Where bindings are persisted, under `config/` -- so a First-Launch choice ("Primary
        Vision Provider" -> some Hugging Face repo) survives a restart without the operator seeing
        the setup wizard again. Plain JSON, human-editable, no recompilation required (Part 14).
        """
        return self._layout.config_dir / "model_bindings.json"

    def save_bindings(self) -> None:
        adapter = TypeAdapter(list[LogicalModelBinding])
        self._layout.config_dir.mkdir(parents=True, exist_ok=True)
        self.bindings_path().write_bytes(adapter.dump_json(list(self._bindings.values()), indent=2))

    def load_bindings(self) -> None:
        """Loads previously-saved bindings, if any. A missing file means "nothing configured yet"
        (First Launch is still needed) -- never an error.
        """
        path = self.bindings_path()
        if not path.exists():
            return
        adapter = TypeAdapter(list[LogicalModelBinding])
        for binding in adapter.validate_json(path.read_text(encoding="utf-8")):
            self._bindings[binding.logical_name] = binding

    def resolve(self, logical_name: str) -> ResolvedModel:
        """Resolves the model/device/precision/cache a logical name currently points at. Does
        *not* require a runtime factory to be registered -- resolution (Part 3: "Runtime, Model,
        Revision, Location, Device, Compatibility, Cache") is a pure mapping question, independent
        of whether this process can currently construct that runtime; `create_runtime` is where
        construction, and therefore `UnsupportedRuntimeError`, applies.
        """
        try:
            binding = self._bindings[logical_name]
        except KeyError:
            raise UnknownLogicalModelError(logical_name) from None

        local_path = None
        if binding.descriptor.source.kind == ModelSourceKind.LOCAL_PATH:
            local_path = self._layout.models_dir / binding.descriptor.source.identifier

        return ResolvedModel(
            descriptor=binding.descriptor,
            local_path=local_path,
            device=binding.device,
            precision=binding.precision,
            cache_dir=self._layout.cache_dir,
        )

    def create_runtime(self, logical_name: str) -> InferenceRuntime | LayoutDetectionRuntime:
        """Resolves and constructs a ready-to-use runtime in one step — the call a provider bridge
        actually makes."""
        resolved = self.resolve(logical_name)
        if resolved.descriptor.runtime_kind not in self._runtime_factories:
            raise UnsupportedRuntimeError(resolved.descriptor.runtime_kind)
        factory = self._runtime_factories[resolved.descriptor.runtime_kind]
        runtime = factory(resolved)
        capabilities = runtime.capabilities()
        if resolved.device == DeviceSelection.GPU_ONLY and not capabilities.supports_gpu:
            raise IncompatibleModelError(
                f"runtime {resolved.descriptor.runtime_kind!r} does not support GPU_ONLY"
            )
        if (
            resolved.precision != Precision.AUTOMATIC
            and resolved.precision not in capabilities.supported_precisions
        ):
            raise IncompatibleModelError(
                f"runtime {resolved.descriptor.runtime_kind!r} does not support precision "
                f"{resolved.precision.value!r} (supports: "
                f"{[p.value for p in capabilities.supported_precisions]})"
            )
        return runtime
