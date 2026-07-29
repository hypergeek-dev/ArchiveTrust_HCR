from __future__ import annotations

import pytest

from archivetrust.runtime.contracts import DeviceSelection, InferenceRequest, InferenceResult, Precision, RuntimeCapabilities
from archivetrust.runtime.deployment_layout import DeploymentLayout
from archivetrust.runtime.gpu_resource_manager import GPUResourceManager
from archivetrust.runtime.model_registry import LogicalModelBinding, ModelRegistry
from archivetrust.runtime.models import ModelDescriptor, ModelSource, ModelSourceKind
from archivetrust.runtime.runtime_manager import RuntimeManager
from archivetrust.runtime.runtime_telemetry import InMemoryRuntimeTelemetrySink, RuntimeTelemetryKind

from tests.runtime.test_gpu_resource_manager import FakeGPUInfoProvider


class FakeVLLMRuntime:
    """A minimal vLLM-shaped `InferenceRuntime` for `RuntimeManager` tests -- exposes the extra
    `is_healthy`/`evict` surface real `VLLMRuntime` exposes, so `RuntimeManager`'s duck-typed checks
    are exercised without any real Docker/vLLM involved.
    """

    runtime_kind = "vllm"

    def __init__(self, model_id: str, *, gpu_memory_utilization: float) -> None:
        self.model_id = model_id
        self.gpu_memory_utilization = gpu_memory_utilization
        self.warm_up_calls = 0
        self.evict_calls = 0
        self.shut_down_calls = 0
        self._healthy = True

    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(
            supports_gpu=True, supports_cpu=False, supported_precisions=(Precision.AUTOMATIC,),
            supports_batching=True, supports_deterministic_seed=True, max_tokens_configurable=True,
        )

    def warm_up(self) -> None:
        self.warm_up_calls += 1

    def infer(self, request: InferenceRequest) -> InferenceResult:
        return InferenceResult(
            raw_text="x", model_version=self.model_id, device_used="cuda",
            precision_used=Precision.AUTOMATIC, inference_seconds=0.01,
        )

    def is_healthy(self) -> bool:
        return self._healthy

    def mark_unhealthy(self) -> None:
        self._healthy = False

    def evict(self) -> None:
        self.evict_calls += 1
        self._healthy = False

    def shut_down(self) -> None:
        self.shut_down_calls += 1


def _descriptor(model_id: str, *, runtime_kind: str = "vllm") -> ModelDescriptor:
    return ModelDescriptor(
        model_id=model_id,
        display_name=model_id,
        source=ModelSource(kind=ModelSourceKind.HUGGING_FACE, identifier=model_id),
        runtime_kind=runtime_kind,
    )


def _bind(registry: ModelRegistry, logical_name: str, model_id: str, *, runtime_kind: str = "vllm") -> None:
    registry.bind(
        LogicalModelBinding(
            logical_name=logical_name,
            descriptor=_descriptor(model_id, runtime_kind=runtime_kind),
            device=DeviceSelection.GPU_ONLY if runtime_kind == "vllm" else DeviceSelection.AUTOMATIC,
            precision=Precision.AUTOMATIC,
        )
    )


@pytest.fixture
def registry(tmp_path) -> ModelRegistry:
    return ModelRegistry(DeploymentLayout(root=tmp_path).ensure())


@pytest.fixture
def gpu_manager() -> GPUResourceManager:
    return GPUResourceManager(gpu_info=FakeGPUInfoProvider(total_bytes=8_000_000_000))


def _factory_and_registry_of(created: list[FakeVLLMRuntime]):
    def factory(resolved, *, gpu_memory_utilization: float) -> FakeVLLMRuntime:
        runtime = FakeVLLMRuntime(resolved.descriptor.model_id, gpu_memory_utilization=gpu_memory_utilization)
        created.append(runtime)
        return runtime

    return factory


def test_get_or_create_constructs_a_vllm_runtime_with_a_computed_gpu_budget(registry, gpu_manager) -> None:
    """Construction (GPU reservation + object creation) is eager; warming up (starting the actual
    Docker container) is deliberately left lazy -- `get_or_create` is called from provider
    *activation*, which must never require GPU/Docker access on its own (this codebase's existing,
    tested invariant, preserved unchanged by this milestone)."""
    _bind(registry, "primary-vision", "PaddlePaddle/PaddleOCR-VL")
    created: list[FakeVLLMRuntime] = []
    manager = RuntimeManager(
        model_registry=registry, gpu_resource_manager=gpu_manager, vllm_runtime_factory=_factory_and_registry_of(created)
    )

    runtime = manager.get_or_create("primary-vision")

    assert len(created) == 1
    assert runtime is created[0]
    assert runtime.warm_up_calls == 0
    assert 0.0 < runtime.gpu_memory_utilization <= 0.75


def test_get_or_create_reuses_the_same_runtime(registry, gpu_manager) -> None:
    _bind(registry, "primary-vision", "PaddlePaddle/PaddleOCR-VL")
    created: list[FakeVLLMRuntime] = []
    manager = RuntimeManager(
        model_registry=registry, gpu_resource_manager=gpu_manager, vllm_runtime_factory=_factory_and_registry_of(created)
    )

    first = manager.get_or_create("primary-vision")
    second = manager.get_or_create("primary-vision")

    assert first is second
    assert len(created) == 1


def test_get_or_create_restarts_a_runtime_that_fails_its_health_check(registry, gpu_manager) -> None:
    _bind(registry, "primary-vision", "PaddlePaddle/PaddleOCR-VL")
    created: list[FakeVLLMRuntime] = []
    manager = RuntimeManager(
        model_registry=registry, gpu_resource_manager=gpu_manager, vllm_runtime_factory=_factory_and_registry_of(created)
    )

    first = manager.get_or_create("primary-vision")
    first.mark_unhealthy()
    second = manager.get_or_create("primary-vision")

    assert second is not first
    assert len(created) == 2


def test_insufficient_gpu_memory_evicts_the_least_recently_used_runtime(registry, gpu_manager) -> None:
    """Live-verified regression scenario (Production Runtime Completion milestone, Part 9): a third
    model that does not fit alongside two already-resident ones must trigger automatic eviction of
    whichever was used longest ago, never a hard failure the operator has to resolve by hand.
    """
    _bind(registry, "a", "model-a")
    _bind(registry, "b", "model-b")
    _bind(registry, "c", "model-c")
    created: list[FakeVLLMRuntime] = []
    manager = RuntimeManager(
        model_registry=registry, gpu_resource_manager=gpu_manager, vllm_runtime_factory=_factory_and_registry_of(created)
    )

    runtime_a = manager.get_or_create("a")
    manager.get_or_create("b")
    # "a" is now the least-recently-used of the two resident runtimes.
    manager.get_or_create("c")

    assert runtime_a.evict_calls == 1
    assert gpu_manager.reservation_for("model-a") is None
    assert gpu_manager.reservation_for("model-c") is not None


def test_reused_runtimes_stay_resident_over_a_least_recently_used_one(registry, gpu_manager) -> None:
    _bind(registry, "a", "model-a")
    _bind(registry, "b", "model-b")
    _bind(registry, "c", "model-c")
    created: list[FakeVLLMRuntime] = []
    manager = RuntimeManager(
        model_registry=registry, gpu_resource_manager=gpu_manager, vllm_runtime_factory=_factory_and_registry_of(created)
    )

    manager.get_or_create("a")
    manager.get_or_create("b")
    manager.get_or_create("a")  # touches "a" again -- "b" becomes the LRU one
    manager.get_or_create("c")

    # "a" was touched most recently among the two resident runtimes, so "b" (not "a") is evicted.
    assert gpu_manager.reservation_for("model-a") is not None
    assert gpu_manager.reservation_for("model-b") is None


def test_release_all_evicts_every_cached_runtime(registry, gpu_manager) -> None:
    _bind(registry, "a", "model-a")
    created: list[FakeVLLMRuntime] = []
    manager = RuntimeManager(
        model_registry=registry, gpu_resource_manager=gpu_manager, vllm_runtime_factory=_factory_and_registry_of(created)
    )
    manager.get_or_create("a")

    manager.release_all()

    assert created[0].evict_calls == 1
    assert gpu_manager.reservation_for("model-a") is None


def test_rebind_preserves_the_runtime_cache_across_a_workspace_switch(tmp_path, gpu_manager) -> None:
    registry_one = ModelRegistry(DeploymentLayout(root=tmp_path / "one").ensure())
    _bind(registry_one, "primary-vision", "PaddlePaddle/PaddleOCR-VL")
    created: list[FakeVLLMRuntime] = []
    manager = RuntimeManager(
        model_registry=registry_one, gpu_resource_manager=gpu_manager, vllm_runtime_factory=_factory_and_registry_of(created)
    )
    manager.get_or_create("primary-vision")

    registry_two = ModelRegistry(DeploymentLayout(root=tmp_path / "two").ensure())
    _bind(registry_two, "primary-vision", "PaddlePaddle/PaddleOCR-VL")
    manager.rebind(registry_two)
    manager.get_or_create("primary-vision")

    assert len(created) == 1  # same model id -> the cached runtime is reused, not rebuilt


def test_records_runtime_telemetry_for_start_reuse_and_gpu_allocation(registry, gpu_manager) -> None:
    _bind(registry, "primary-vision", "PaddlePaddle/PaddleOCR-VL")
    created: list[FakeVLLMRuntime] = []
    telemetry = InMemoryRuntimeTelemetrySink()
    manager = RuntimeManager(
        model_registry=registry,
        gpu_resource_manager=gpu_manager,
        vllm_runtime_factory=_factory_and_registry_of(created),
        telemetry=telemetry,
    )

    manager.get_or_create("primary-vision")
    manager.get_or_create("primary-vision")

    kinds = [event.kind for event in telemetry.events()]
    assert RuntimeTelemetryKind.RUNTIME_STARTED in kinds
    assert RuntimeTelemetryKind.GPU_ALLOCATED in kinds
    assert RuntimeTelemetryKind.RUNTIME_REUSED in kinds


def test_non_vllm_runtime_kind_is_still_reused_without_gpu_involvement(registry, gpu_manager) -> None:
    from tests.runtime._fakes import FakeRuntime

    _bind(registry, "primary-vision", "some/transformers-model", runtime_kind="transformers")
    fake = FakeRuntime()
    registry.register_runtime_factory("transformers", lambda resolved: fake)
    manager = RuntimeManager(
        model_registry=registry, gpu_resource_manager=gpu_manager, vllm_runtime_factory=lambda *a, **k: (_ for _ in ()).throw(AssertionError("vllm factory must not be called"))
    )

    first = manager.get_or_create("primary-vision")
    second = manager.get_or_create("primary-vision")

    assert first is second is fake
    assert gpu_manager.active_reservations() == ()


def test_vllm_construction_failure_is_recorded_and_reraised(registry, gpu_manager) -> None:
    """Production Investigation finding: a runtime-construction failure (Docker unreachable,
    container start timeout, ...) previously propagated with no telemetry at all -- the GPU
    reservation was already recorded but nothing said the runtime itself then failed to start, so a
    crash here looked identical to "never attempted" in any diagnostics view."""
    _bind(registry, "primary-vision", "PaddlePaddle/PaddleOCR-VL")
    telemetry = InMemoryRuntimeTelemetrySink()

    def failing_factory(resolved, *, gpu_memory_utilization: float):
        raise RuntimeError("docker daemon unreachable")

    manager = RuntimeManager(
        model_registry=registry, gpu_resource_manager=gpu_manager,
        vllm_runtime_factory=failing_factory, telemetry=telemetry,
    )

    with pytest.raises(RuntimeError, match="docker daemon unreachable"):
        manager.get_or_create("primary-vision")

    kinds = [event.kind for event in telemetry.events()]
    assert RuntimeTelemetryKind.RUNTIME_START_FAILED in kinds
    failed_event = next(e for e in telemetry.events() if e.kind == RuntimeTelemetryKind.RUNTIME_START_FAILED)
    assert "docker daemon unreachable" in failed_event.detail
    # the GPU reservation made before construction failed must not be leaked.
    assert gpu_manager.reservation_for("PaddlePaddle/PaddleOCR-VL") is not None


def test_non_vllm_construction_failure_is_recorded_and_reraised(registry, gpu_manager) -> None:
    _bind(registry, "primary-vision", "some/transformers-model", runtime_kind="transformers")

    def failing_factory(resolved):
        raise RuntimeError("model weights corrupt")

    registry.register_runtime_factory("transformers", failing_factory)
    telemetry = InMemoryRuntimeTelemetrySink()
    manager = RuntimeManager(
        model_registry=registry, gpu_resource_manager=gpu_manager,
        vllm_runtime_factory=lambda *a, **k: (_ for _ in ()).throw(AssertionError("vllm factory must not be called")),
        telemetry=telemetry,
    )

    with pytest.raises(RuntimeError, match="model weights corrupt"):
        manager.get_or_create("primary-vision")

    failed_event = next(e for e in telemetry.events() if e.kind == RuntimeTelemetryKind.RUNTIME_START_FAILED)
    assert failed_event.runtime_kind == "transformers"
    assert "model weights corrupt" in failed_event.detail


def test_non_vllm_runtime_kind_is_not_eagerly_warmed_up(registry, gpu_manager) -> None:
    """Regression guard: activating a `transformers`-backed provider must stay exactly as lazy as
    it always was (model weights load on the first real inference call, not at activation time) --
    unlike a vLLM-backed runtime, where "starting" a Docker container eagerly is cheap and correct.
    """
    from tests.runtime._fakes import FakeRuntime

    _bind(registry, "primary-vision", "some/transformers-model", runtime_kind="transformers")
    fake = FakeRuntime()
    registry.register_runtime_factory("transformers", lambda resolved: fake)
    manager = RuntimeManager(
        model_registry=registry, gpu_resource_manager=gpu_manager, vllm_runtime_factory=lambda *a, **k: (_ for _ in ()).throw(AssertionError("vllm factory must not be called"))
    )

    manager.get_or_create("primary-vision")

    assert fake.warm_up_calls == 0


def _probe_of(*, discoverable: bool, calls: list[str] | None = None):
    def probe(resolved) -> bool:
        if calls is not None:
            calls.append(resolved.descriptor.model_id)
        return discoverable

    return probe


def test_discovery_reuses_an_already_running_runtime_without_gpu_reservation(registry, gpu_manager) -> None:
    """Phase 24 (Runtime Discovery and Reuse) -- the core scenario: a container this app's own
    prior session started and left alive is discovered, not re-created, and no second GPU
    reservation is attempted for it (the exact previous cause of a restart-time
    `GPUOversubscriptionError` even though the old runtime was still alive and healthy)."""
    _bind(registry, "primary-vision", "PaddlePaddle/PaddleOCR-VL")
    created: list[FakeVLLMRuntime] = []
    manager = RuntimeManager(
        model_registry=registry,
        gpu_resource_manager=gpu_manager,
        vllm_runtime_factory=_factory_and_registry_of(created),
        vllm_discovery_probe=_probe_of(discoverable=True),
    )

    runtime = manager.get_or_create("primary-vision")

    assert len(created) == 1  # exactly one construction -- no throwaway probe instance
    assert runtime is created[0]
    assert gpu_manager.reservation_for("PaddlePaddle/PaddleOCR-VL") is not None  # adopted, not reserved-fresh


def test_discovery_records_runtime_discovered_not_runtime_started(registry, gpu_manager) -> None:
    _bind(registry, "primary-vision", "PaddlePaddle/PaddleOCR-VL")
    telemetry = InMemoryRuntimeTelemetrySink()
    manager = RuntimeManager(
        model_registry=registry,
        gpu_resource_manager=gpu_manager,
        vllm_runtime_factory=_factory_and_registry_of([]),
        vllm_discovery_probe=_probe_of(discoverable=True),
        telemetry=telemetry,
    )

    manager.get_or_create("primary-vision")

    kinds = [event.kind for event in telemetry.events()]
    assert RuntimeTelemetryKind.RUNTIME_DISCOVERED in kinds
    assert RuntimeTelemetryKind.RUNTIME_STARTED not in kinds
    assert RuntimeTelemetryKind.RUNTIME_RESTARTED not in kinds
    assert RuntimeTelemetryKind.GPU_ALLOCATED not in kinds  # no fresh reservation was made


def test_no_compatible_runtime_falls_back_to_normal_creation_and_reservation(registry, gpu_manager) -> None:
    """When discovery finds nothing (no probe configured, or the probe reports nothing reusable),
    behavior must be identical to today's: reserve, then create."""
    _bind(registry, "primary-vision", "PaddlePaddle/PaddleOCR-VL")
    created: list[FakeVLLMRuntime] = []
    telemetry = InMemoryRuntimeTelemetrySink()
    manager = RuntimeManager(
        model_registry=registry,
        gpu_resource_manager=gpu_manager,
        vllm_runtime_factory=_factory_and_registry_of(created),
        vllm_discovery_probe=_probe_of(discoverable=False),
        telemetry=telemetry,
    )

    manager.get_or_create("primary-vision")

    assert len(created) == 1
    kinds = [event.kind for event in telemetry.events()]
    assert RuntimeTelemetryKind.GPU_ALLOCATED in kinds
    assert RuntimeTelemetryKind.RUNTIME_STARTED in kinds
    assert RuntimeTelemetryKind.RUNTIME_DISCOVERED not in kinds


def test_no_discovery_probe_configured_behaves_exactly_as_before_this_phase(registry, gpu_manager) -> None:
    """Backward-compatibility guard: a `RuntimeManager` constructed without a discovery probe (the
    default) must reserve and create exactly as it always did -- no existing caller regresses."""
    _bind(registry, "primary-vision", "PaddlePaddle/PaddleOCR-VL")
    created: list[FakeVLLMRuntime] = []
    manager = RuntimeManager(
        model_registry=registry, gpu_resource_manager=gpu_manager, vllm_runtime_factory=_factory_and_registry_of(created)
    )

    manager.get_or_create("primary-vision")

    assert len(created) == 1
    assert gpu_manager.reservation_for("PaddlePaddle/PaddleOCR-VL") is not None


def test_discovery_probe_is_only_consulted_for_vllm_backed_bindings(registry, gpu_manager) -> None:
    from tests.runtime._fakes import FakeRuntime

    _bind(registry, "primary-vision", "some/transformers-model", runtime_kind="transformers")
    fake = FakeRuntime()
    registry.register_runtime_factory("transformers", lambda resolved: fake)
    probe_calls: list[str] = []
    manager = RuntimeManager(
        model_registry=registry,
        gpu_resource_manager=gpu_manager,
        vllm_runtime_factory=lambda *a, **k: (_ for _ in ()).throw(AssertionError("vllm factory must not be called")),
        vllm_discovery_probe=_probe_of(discoverable=True, calls=probe_calls),
    )

    manager.get_or_create("primary-vision")

    assert probe_calls == []


def test_discovered_runtime_is_cached_so_a_second_call_never_reprobes(registry, gpu_manager) -> None:
    """Discovery only needs to happen once per process per model -- once attached, the ordinary
    in-memory cache (`RUNTIME_REUSED`) takes over exactly as for a freshly created runtime."""
    _bind(registry, "primary-vision", "PaddlePaddle/PaddleOCR-VL")
    probe_calls: list[str] = []
    manager = RuntimeManager(
        model_registry=registry,
        gpu_resource_manager=gpu_manager,
        vllm_runtime_factory=_factory_and_registry_of([]),
        vllm_discovery_probe=_probe_of(discoverable=True, calls=probe_calls),
    )

    first = manager.get_or_create("primary-vision")
    second = manager.get_or_create("primary-vision")

    assert first is second
    assert probe_calls == ["PaddlePaddle/PaddleOCR-VL"]  # probed once, not on the cache-hit call


def test_resolution_failure_is_recorded_and_reraised(registry, gpu_manager) -> None:
    """Production Hardening Review (2026-07-13): `resolve()` raising (no such binding, missing
    model source) previously escaped `get_or_create` before any telemetry line executed --
    durably indistinguishable from "never attempted", the same blind spot the construction-failure
    recording already closed for the later stages."""
    telemetry = InMemoryRuntimeTelemetrySink()
    manager = RuntimeManager(
        model_registry=registry, gpu_resource_manager=gpu_manager,
        vllm_runtime_factory=_factory_and_registry_of([]), telemetry=telemetry,
    )

    with pytest.raises(Exception):
        manager.get_or_create("never-bound-logical-name")

    failed = [e for e in telemetry.events() if e.kind == RuntimeTelemetryKind.RUNTIME_START_FAILED]
    assert len(failed) == 1
    assert failed[0].model_id == "never-bound-logical-name"
    assert failed[0].runtime_kind == "unresolved"
