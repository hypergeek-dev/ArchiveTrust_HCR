from __future__ import annotations

from archivetrust.presentation.settings_viewmodel import (
    AdvancedProviderSettingsViewModel,
    GeneralSettingsViewModel,
)
from archivetrust.providers.base import (
    DeploymentLocality,
    Introspectability,
    ProviderAdapter,
    ProviderRegistration,
    ProviderRunResult,
    Reproducibility,
)
from archivetrust.providers.registry import ProviderRegistry
from archivetrust.runtime.contracts import DeviceSelection, Precision
from archivetrust.runtime.provider_config import (
    DETERMINISTIC_LOCAL_CAPABILITIES,
    PROBABILISTIC_VISION_CAPABILITIES,
    ExecutionMode,
    ProcessingMode,
    ProviderConfigurationStore,
)


class _VisionAdapter(ProviderAdapter):
    registration = ProviderRegistration(
        provider_id="qwen2.5-vl",
        reproducibility=Reproducibility.PROBABILISTIC,
        deployment_locality=DeploymentLocality.OUT_OF_PROCESS,
        introspectability=Introspectability.BLACK_BOX,
    )

    def observe(self, *, document_ref, invocation_id, source) -> ProviderRunResult:
        raise NotImplementedError


def _registry() -> ProviderRegistry:
    registry = ProviderRegistry()
    registry.register(_VisionAdapter())
    return registry


def test_general_settings_default_execution_and_processing_mode() -> None:
    store = ProviderConfigurationStore()
    vm = GeneralSettingsViewModel(provider_registry=_registry(), configuration_store=store)
    vm.set_primary_vision_provider("qwen2.5-vl")
    assert vm.primary_vision_provider() == "qwen2.5-vl"
    assert vm.execution_mode() == ExecutionMode.AUTOMATIC
    assert vm.processing_mode() == ProcessingMode.BALANCED


def test_general_settings_set_execution_mode() -> None:
    store = ProviderConfigurationStore()
    vm = GeneralSettingsViewModel(provider_registry=_registry(), configuration_store=store)
    vm.set_primary_vision_provider("qwen2.5-vl")
    vm.set_execution_mode(ExecutionMode.GPU_ONLY)
    assert vm.execution_mode() == ExecutionMode.GPU_ONLY
    assert store.get("qwen2.5-vl").device == DeviceSelection.GPU_ONLY


def test_general_settings_set_processing_mode() -> None:
    store = ProviderConfigurationStore()
    vm = GeneralSettingsViewModel(provider_registry=_registry(), configuration_store=store)
    vm.set_primary_vision_provider("qwen2.5-vl")
    vm.set_processing_mode(ProcessingMode.MAXIMUM_QUALITY)
    assert vm.processing_mode() == ProcessingMode.MAXIMUM_QUALITY


def test_setting_primary_provider_disables_others() -> None:
    registry = ProviderRegistry()
    registry.register(_VisionAdapter())

    class _Other(ProviderAdapter):
        registration = ProviderRegistration(
            provider_id="glm-ocr",
            reproducibility=Reproducibility.PROBABILISTIC,
            deployment_locality=DeploymentLocality.OUT_OF_PROCESS,
            introspectability=Introspectability.BLACK_BOX,
        )

        def observe(self, *, document_ref, invocation_id, source):
            raise NotImplementedError

    registry.register(_Other())
    store = ProviderConfigurationStore()
    vm = GeneralSettingsViewModel(provider_registry=registry, configuration_store=store)
    vm.set_primary_vision_provider("qwen2.5-vl")
    assert store.get("qwen2.5-vl").enabled
    assert not store.get("glm-ocr").enabled


def test_advanced_settings_hides_groups_the_provider_does_not_support() -> None:
    store = ProviderConfigurationStore()
    vm = AdvancedProviderSettingsViewModel(
        provider_id="docling", configuration_store=store, capabilities=DETERMINISTIC_LOCAL_CAPABILITIES
    )
    groups = {f.group for f in vm.visible_fields()}
    assert "Prompt" not in groups
    assert "Device" not in groups
    assert "Grounding" in groups  # Docling still emits bounding boxes


def test_advanced_settings_shows_full_surface_for_probabilistic_provider() -> None:
    store = ProviderConfigurationStore()
    vm = AdvancedProviderSettingsViewModel(
        provider_id="qwen2.5-vl", configuration_store=store, capabilities=PROBABILISTIC_VISION_CAPABILITIES
    )
    groups = {f.group for f in vm.visible_fields()}
    assert {"Device", "Precision", "Performance", "Image Processing", "Prompt", "Determinism", "Grounding", "Output"} <= groups


def test_set_precision_updates_store() -> None:
    store = ProviderConfigurationStore()
    vm = AdvancedProviderSettingsViewModel(
        provider_id="qwen2.5-vl", configuration_store=store, capabilities=PROBABILISTIC_VISION_CAPABILITIES
    )
    vm.set_precision(Precision.INT8)
    assert store.get("qwen2.5-vl").precision == Precision.INT8


def test_deterministic_mode_shows_seed_only_when_enabled() -> None:
    store = ProviderConfigurationStore()
    vm = AdvancedProviderSettingsViewModel(
        provider_id="qwen2.5-vl", configuration_store=store, capabilities=PROBABILISTIC_VISION_CAPABILITIES
    )
    labels_before = {f.label for f in vm.visible_fields()}
    assert "Seed" not in labels_before
    vm.set_deterministic(deterministic=True, seed=42)
    labels_after = {f.label for f in vm.visible_fields()}
    assert "Seed" in labels_after
