from __future__ import annotations

from archivetrust.providers.base import (
    DeploymentLocality,
    Introspectability,
    ProviderAdapter,
    ProviderRegistration,
    ProviderRunResult,
    Reproducibility,
)
from archivetrust.providers.registry import ProviderRegistry
from archivetrust.presentation.provider_manager_viewmodel import ProviderManagerViewModel
from archivetrust.runtime.contracts import DeviceSelection
from archivetrust.runtime.deployment_layout import DeploymentLayout
from archivetrust.runtime.model_registry import LogicalModelBinding, ModelRegistry
from archivetrust.runtime.models import ModelDescriptor, ModelSource, ModelSourceKind
from archivetrust.runtime.provider_config import ProviderConfiguration, ProviderConfigurationStore
from archivetrust.runtime.provider_health import ProviderHealthTracker


class _FakeVisionAdapter(ProviderAdapter):
    registration = ProviderRegistration(
        provider_id="qwen2.5-vl",
        reproducibility=Reproducibility.PROBABILISTIC,
        deployment_locality=DeploymentLocality.OUT_OF_PROCESS,
        introspectability=Introspectability.BLACK_BOX,
    )

    def observe(self, *, document_ref, invocation_id, source) -> ProviderRunResult:
        raise NotImplementedError


class _FakeDeterministicAdapter(ProviderAdapter):
    registration = ProviderRegistration(
        provider_id="docling",
        reproducibility=Reproducibility.DETERMINISTIC,
        deployment_locality=DeploymentLocality.IN_PROCESS,
        introspectability=Introspectability.OPEN,
    )

    def observe(self, *, document_ref, invocation_id, source) -> ProviderRunResult:
        raise NotImplementedError


def _registry() -> ProviderRegistry:
    registry = ProviderRegistry()
    registry.register(_FakeVisionAdapter())
    registry.register(_FakeDeterministicAdapter())
    return registry


def test_entries_cover_every_registered_provider() -> None:
    vm = ProviderManagerViewModel(
        provider_registry=_registry(),
        configuration_store=ProviderConfigurationStore(),
        health_trackers={},
    )
    entries = vm.entries()
    assert {e.provider_id for e in entries} == {"qwen2.5-vl", "docling"}


def test_probabilistic_provider_gets_full_capabilities_deterministic_gets_narrow() -> None:
    vm = ProviderManagerViewModel(
        provider_registry=_registry(),
        configuration_store=ProviderConfigurationStore(),
        health_trackers={},
    )
    entries = {e.provider_id: e for e in vm.entries()}
    assert entries["qwen2.5-vl"].capabilities.has_prompt_management
    assert not entries["docling"].capabilities.has_prompt_management
    assert entries["qwen2.5-vl"].backend == "probabilistic"
    assert entries["docling"].backend == "deterministic"


def test_health_snapshot_reflects_tracker_state() -> None:
    tracker = ProviderHealthTracker("qwen2.5-vl")
    tracker.mark_ready(backend="transformers")
    vm = ProviderManagerViewModel(
        provider_registry=_registry(),
        configuration_store=ProviderConfigurationStore(),
        health_trackers={"qwen2.5-vl": tracker},
        availability={"qwen2.5-vl": True, "docling": True},
    )
    entries = {e.provider_id: e for e in vm.entries()}
    assert entries["qwen2.5-vl"].status == "Ready"
    assert entries["docling"].status == "Ready"


def test_unavailable_provider_is_reported_honestly() -> None:
    # Operational Completion milestone: a provider whose real client library isn't installed must
    # never claim "Ready" -- it is enabled by default (ProviderConfiguration.enabled == True) but
    # unavailable until the operator installs it.
    vm = ProviderManagerViewModel(
        provider_registry=_registry(),
        configuration_store=ProviderConfigurationStore(),
        health_trackers={},
        availability={"docling": False},
    )
    entries = {e.provider_id: e for e in vm.entries()}
    assert entries["docling"].status == "Unavailable"


def test_disabled_provider_reports_disabled_even_if_available() -> None:
    store = ProviderConfigurationStore()
    store.set(ProviderConfiguration(provider_id="docling", enabled=False))
    vm = ProviderManagerViewModel(
        provider_registry=_registry(),
        configuration_store=store,
        health_trackers={},
        availability={"docling": True},
    )
    entries = {e.provider_id: e for e in vm.entries()}
    assert entries["docling"].status == "Disabled"


def test_probabilistic_provider_available_but_not_ready_is_configuration_error() -> None:
    # A real "transformers" runtime is importable, but no model has been bound/warmed up yet.
    vm = ProviderManagerViewModel(
        provider_registry=_registry(),
        configuration_store=ProviderConfigurationStore(),
        health_trackers={},
        availability={"qwen2.5-vl": True},
    )
    entries = {e.provider_id: e for e in vm.entries()}
    assert entries["qwen2.5-vl"].status == "Configuration Error"


def test_running_provider_reports_running() -> None:
    running = {"docling"}
    vm = ProviderManagerViewModel(
        provider_registry=_registry(),
        configuration_store=ProviderConfigurationStore(),
        health_trackers={},
        availability={"docling": True},
        running_provider_ids=running,
    )
    entries = {e.provider_id: e for e in vm.entries()}
    assert entries["docling"].status == "Running"


def test_runtime_kind_resolved_from_model_registry(tmp_path) -> None:
    layout = DeploymentLayout(root=tmp_path).ensure()
    model_registry = ModelRegistry(layout)
    model_registry.bind(
        LogicalModelBinding(
            logical_name="qwen2.5-vl",
            descriptor=ModelDescriptor(
                model_id="qwen2.5-vl-7b",
                display_name="Qwen2.5-VL",
                source=ModelSource(kind=ModelSourceKind.LOCAL_PATH, identifier="qwen"),
                runtime_kind="transformers",
            ),
        )
    )
    vm = ProviderManagerViewModel(
        provider_registry=_registry(),
        configuration_store=ProviderConfigurationStore(),
        health_trackers={},
        model_registry=model_registry,
    )
    entries = {e.provider_id: e for e in vm.entries()}
    assert entries["qwen2.5-vl"].runtime_kind == "transformers"
    assert entries["docling"].runtime_kind is None  # no binding for docling


def test_set_enabled_updates_configuration_store() -> None:
    store = ProviderConfigurationStore()
    vm = ProviderManagerViewModel(provider_registry=_registry(), configuration_store=store, health_trackers={})
    vm.set_enabled("docling", False)
    assert not store.get("docling").enabled
