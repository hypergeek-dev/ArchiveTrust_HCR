from __future__ import annotations

import pytest

from archivetrust.presentation.first_launch_viewmodel import (
    DEFAULT_VISION_PROVIDER_ID,
    FirstLaunchViewModel,
    InvalidSetupInputError,
    SetupSourceKind,
    needs_first_launch,
    runtime_kind_for_provider,
)
from archivetrust.runtime.deployment_layout import DeploymentLayout
from archivetrust.runtime.model_registry import ModelRegistry


def test_needs_first_launch_true_when_no_models_installed(tmp_path) -> None:
    layout = DeploymentLayout(root=tmp_path).ensure()
    assert needs_first_launch(layout)


def test_needs_first_launch_false_after_a_model_directory_exists(tmp_path) -> None:
    layout = DeploymentLayout(root=tmp_path).ensure()
    (layout.models_dir / "qwen").mkdir()
    assert not needs_first_launch(layout)


def test_complete_setup_with_hugging_face_source(tmp_path) -> None:
    layout = DeploymentLayout(root=tmp_path).ensure()
    registry = ModelRegistry(layout)
    vm = FirstLaunchViewModel(layout=layout, model_registry=registry)

    result = vm.complete_setup(
        source_kind=SetupSourceKind.HUGGING_FACE,
        identifier="Qwen/Qwen2.5-VL-7B-Instruct",
        runtime_kind="transformers",
    )
    assert result.model_id == "Qwen/Qwen2.5-VL-7B-Instruct"
    assert result.runtime_kind == "transformers"
    resolved = registry.resolve("Primary Vision Provider")
    assert resolved.descriptor.source.identifier == "Qwen/Qwen2.5-VL-7B-Instruct"


def test_complete_setup_with_local_browse(tmp_path) -> None:
    layout = DeploymentLayout(root=tmp_path).ensure()
    model_dir = layout.models_dir / "my-local-model"
    model_dir.mkdir()
    registry = ModelRegistry(layout)
    vm = FirstLaunchViewModel(layout=layout, model_registry=registry)

    result = vm.complete_setup(
        source_kind=SetupSourceKind.BROWSE_LOCAL, identifier=str(model_dir), runtime_kind="transformers"
    )
    assert result.model_id == "my-local-model"


def test_complete_setup_rejects_missing_local_path(tmp_path) -> None:
    layout = DeploymentLayout(root=tmp_path).ensure()
    registry = ModelRegistry(layout)
    vm = FirstLaunchViewModel(layout=layout, model_registry=registry)
    with pytest.raises(InvalidSetupInputError):
        vm.complete_setup(
            source_kind=SetupSourceKind.BROWSE_LOCAL,
            identifier=str(layout.root / "does-not-exist"),
            runtime_kind="transformers",
        )


def test_complete_setup_rejects_empty_identifier(tmp_path) -> None:
    layout = DeploymentLayout(root=tmp_path).ensure()
    registry = ModelRegistry(layout)
    vm = FirstLaunchViewModel(layout=layout, model_registry=registry)
    with pytest.raises(InvalidSetupInputError):
        vm.complete_setup(source_kind=SetupSourceKind.HUGGING_FACE, identifier="   ", runtime_kind="transformers")


def test_needs_setup_reflects_current_layout_state(tmp_path) -> None:
    layout = DeploymentLayout(root=tmp_path).ensure()
    registry = ModelRegistry(layout)
    vm = FirstLaunchViewModel(layout=layout, model_registry=registry)
    assert vm.needs_setup()
    (layout.models_dir / "something").mkdir()
    assert not vm.needs_setup()


def test_runtime_kind_for_provider_resolves_paddleocr_vl_and_surya_to_vllm() -> None:
    assert runtime_kind_for_provider("paddleocr-vl") == "vllm"
    assert runtime_kind_for_provider("surya") == "vllm"
    assert runtime_kind_for_provider("qwen2.5-vl") == "transformers"


def test_runtime_kind_for_provider_falls_back_to_default_for_unknown_provider() -> None:
    from archivetrust.presentation.first_launch_viewmodel import DEFAULT_VISION_RUNTIME_KIND

    assert runtime_kind_for_provider("some-future-provider") == DEFAULT_VISION_RUNTIME_KIND


def test_complete_setup_without_runtime_kind_derives_it_from_provider_id(tmp_path) -> None:
    """Runtime Architecture Completion milestone, Part 11: the ingestion UI no longer exposes a
    runtime selector -- `complete_setup` must derive it from `provider_id` when omitted."""
    layout = DeploymentLayout(root=tmp_path).ensure()
    registry = ModelRegistry(layout)
    vm = FirstLaunchViewModel(layout=layout, model_registry=registry)

    result = vm.complete_setup(
        source_kind=SetupSourceKind.HUGGING_FACE,
        identifier="PaddlePaddle/PaddleOCR-VL",
        provider_id=DEFAULT_VISION_PROVIDER_ID,
    )
    assert result.runtime_kind == "vllm"
    resolved = registry.resolve("Primary Vision Provider")
    assert resolved.descriptor.runtime_kind == "vllm"


def test_complete_setup_explicit_runtime_kind_still_overrides_the_derived_default(tmp_path) -> None:
    layout = DeploymentLayout(root=tmp_path).ensure()
    registry = ModelRegistry(layout)
    vm = FirstLaunchViewModel(layout=layout, model_registry=registry)

    result = vm.complete_setup(
        source_kind=SetupSourceKind.HUGGING_FACE,
        identifier="PaddlePaddle/PaddleOCR-VL",
        provider_id=DEFAULT_VISION_PROVIDER_ID,
        runtime_kind="openai_compatible",
    )
    assert result.runtime_kind == "openai_compatible"


def test_needs_first_launch_false_after_a_binding_is_persisted_with_no_local_model(tmp_path) -> None:
    # A Hugging Face source has no local models_dir footprint -- needs_first_launch must still
    # recognize a persisted binding as "already configured" (regression: previously this only
    # scanned disk, so a non-local source would show the wizard again on every restart).
    layout = DeploymentLayout(root=tmp_path).ensure()
    registry = ModelRegistry(layout)
    vm = FirstLaunchViewModel(layout=layout, model_registry=registry)
    vm.complete_setup(
        source_kind=SetupSourceKind.HUGGING_FACE,
        identifier="Qwen/Qwen2.5-VL-7B-Instruct",
        runtime_kind="transformers",
    )
    assert not vm.needs_setup()

    # A fresh registry against the same layout, loading persisted bindings, must also see it.
    reloaded_registry = ModelRegistry(layout)
    reloaded_registry.load_bindings()
    reloaded_vm = FirstLaunchViewModel(layout=layout, model_registry=reloaded_registry)
    assert not reloaded_vm.needs_setup()
