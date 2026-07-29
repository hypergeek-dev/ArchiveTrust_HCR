from __future__ import annotations

import pytest

from archivetrust.runtime.contracts import DeviceSelection, Precision
from archivetrust.runtime.model_registry import (
    IncompatibleModelError,
    LogicalModelBinding,
    ModelRegistry,
    UnknownLogicalModelError,
    UnsupportedRuntimeError,
)
from archivetrust.runtime.models import ModelDescriptor, ModelSource, ModelSourceKind

from tests.runtime._fakes import FakeRuntime


def _descriptor(runtime_kind: str = "fake") -> ModelDescriptor:
    return ModelDescriptor(
        model_id="qwen2.5-vl",
        display_name="Qwen2.5-VL",
        source=ModelSource(kind=ModelSourceKind.LOCAL_PATH, identifier="qwen"),
        runtime_kind=runtime_kind,
    )


def test_resolve_computes_local_path_under_models_dir(tmp_path) -> None:
    from archivetrust.runtime.deployment_layout import DeploymentLayout

    layout = DeploymentLayout(root=tmp_path).ensure()
    registry = ModelRegistry(layout)
    registry.bind(LogicalModelBinding(logical_name="Primary Vision Provider", descriptor=_descriptor()))
    resolved = registry.resolve("Primary Vision Provider")
    assert resolved.local_path == layout.models_dir / "qwen"
    assert resolved.cache_dir == layout.cache_dir


def test_unknown_logical_name_raises(tmp_path) -> None:
    from archivetrust.runtime.deployment_layout import DeploymentLayout

    registry = ModelRegistry(DeploymentLayout(root=tmp_path).ensure())
    with pytest.raises(UnknownLogicalModelError):
        registry.resolve("Nonexistent Provider")


def test_resolve_succeeds_even_without_a_registered_runtime_factory(tmp_path) -> None:
    # Resolution (model/device/precision/cache) is independent of whether this process can
    # construct the runtime -- that check belongs to create_runtime, not resolve.
    from archivetrust.runtime.deployment_layout import DeploymentLayout

    registry = ModelRegistry(DeploymentLayout(root=tmp_path).ensure())
    registry.bind(LogicalModelBinding(logical_name="X", descriptor=_descriptor(runtime_kind="nonexistent")))
    resolved = registry.resolve("X")
    assert resolved.descriptor.runtime_kind == "nonexistent"


def test_unregistered_runtime_kind_raises_on_construction(tmp_path) -> None:
    from archivetrust.runtime.deployment_layout import DeploymentLayout

    registry = ModelRegistry(DeploymentLayout(root=tmp_path).ensure())
    registry.bind(LogicalModelBinding(logical_name="X", descriptor=_descriptor(runtime_kind="nonexistent")))
    with pytest.raises(UnsupportedRuntimeError):
        registry.create_runtime("X")


def test_create_runtime_constructs_via_registered_factory(tmp_path) -> None:
    from archivetrust.runtime.deployment_layout import DeploymentLayout

    layout = DeploymentLayout(root=tmp_path).ensure()
    registry = ModelRegistry(layout)
    registry.register_runtime_factory("fake", lambda resolved: FakeRuntime())
    registry.bind(LogicalModelBinding(logical_name="Primary Vision Provider", descriptor=_descriptor()))
    runtime = registry.create_runtime("Primary Vision Provider")
    assert runtime.runtime_kind == "fake"


def test_gpu_only_rejected_when_runtime_capabilities_lack_gpu(tmp_path) -> None:
    from archivetrust.runtime.deployment_layout import DeploymentLayout

    class NoGpuRuntime(FakeRuntime):
        def capabilities(self):
            caps = super().capabilities()
            return caps.model_copy(update={"supports_gpu": False})

    layout = DeploymentLayout(root=tmp_path).ensure()
    registry = ModelRegistry(layout)
    registry.register_runtime_factory("fake", lambda resolved: NoGpuRuntime())
    registry.bind(
        LogicalModelBinding(
            logical_name="X", descriptor=_descriptor(), device=DeviceSelection.GPU_ONLY
        )
    )
    with pytest.raises(IncompatibleModelError):
        registry.create_runtime("X")


def test_unsupported_precision_rejected(tmp_path) -> None:
    from archivetrust.runtime.deployment_layout import DeploymentLayout

    layout = DeploymentLayout(root=tmp_path).ensure()
    registry = ModelRegistry(layout)
    registry.register_runtime_factory("fake", lambda resolved: FakeRuntime())
    registry.bind(
        LogicalModelBinding(logical_name="X", descriptor=_descriptor(), precision=Precision.INT4)
    )
    with pytest.raises(IncompatibleModelError):
        registry.create_runtime("X")


def test_rebinding_overwrites_prior_binding(tmp_path) -> None:
    from archivetrust.runtime.deployment_layout import DeploymentLayout

    layout = DeploymentLayout(root=tmp_path).ensure()
    registry = ModelRegistry(layout)
    registry.bind(LogicalModelBinding(logical_name="X", descriptor=_descriptor()))
    registry.bind(LogicalModelBinding(logical_name="X", descriptor=_descriptor(runtime_kind="other")))
    assert len(registry.bindings()) == 1
    assert registry.bindings()[0].descriptor.runtime_kind == "other"


def test_bindings_persist_and_reload(tmp_path) -> None:
    from archivetrust.runtime.deployment_layout import DeploymentLayout

    layout = DeploymentLayout(root=tmp_path).ensure()
    registry = ModelRegistry(layout)
    registry.bind(LogicalModelBinding(logical_name="Primary Vision Provider", descriptor=_descriptor()))
    registry.save_bindings()
    assert registry.bindings_path().exists()

    reloaded = ModelRegistry(layout)
    reloaded.load_bindings()
    assert reloaded.bindings() == registry.bindings()


def test_load_bindings_is_a_no_op_when_nothing_was_saved(tmp_path) -> None:
    from archivetrust.runtime.deployment_layout import DeploymentLayout

    registry = ModelRegistry(DeploymentLayout(root=tmp_path).ensure())
    registry.load_bindings()  # must not raise
    assert registry.bindings() == ()


def test_logical_model_binding_provider_id_defaults_to_qwen(tmp_path) -> None:
    """Multi-Provider Activation milestone: `provider_id` defaults to `"qwen2.5-vl"` so a binding
    constructed exactly as every binding was before this field existed still resolves to the
    adapter it always meant."""
    binding = LogicalModelBinding(logical_name="Primary Vision Provider", descriptor=_descriptor())
    assert binding.provider_id == "qwen2.5-vl"


def test_pre_existing_bindings_json_without_provider_id_loads_as_qwen(tmp_path) -> None:
    """Existing Workspaces (bindings persisted before this milestone) must not be silently
    migrated to the new default provider on next load -- verified against the literal JSON shape
    a pre-milestone `save_bindings()` would have produced (no `provider_id` key at all)."""
    import json

    from archivetrust.runtime.deployment_layout import DeploymentLayout

    layout = DeploymentLayout(root=tmp_path).ensure()
    legacy_binding = {
        "logical_name": "Primary Vision Provider",
        "descriptor": {
            "model_id": "qwen2.5-vl",
            "display_name": "Qwen2.5-VL",
            "source": {"kind": "local_path", "identifier": "qwen", "revision": None},
            "runtime_kind": "fake",
        },
        "device": "automatic",
        "precision": "automatic",
        # deliberately no "provider_id" key -- the pre-milestone on-disk shape
    }
    registry = ModelRegistry(layout)
    registry.bindings_path().write_text(json.dumps([legacy_binding]), encoding="utf-8")

    registry.load_bindings()

    assert registry.bindings()[0].provider_id == "qwen2.5-vl"


def test_logical_model_binding_provider_id_is_settable_for_new_providers(tmp_path) -> None:
    binding = LogicalModelBinding(
        logical_name="Primary Vision Provider", descriptor=_descriptor(), provider_id="paddleocr-vl"
    )
    assert binding.provider_id == "paddleocr-vl"
