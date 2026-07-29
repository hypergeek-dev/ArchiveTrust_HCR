"""`RuntimeManager` (Production Runtime Completion milestone, Parts 3/4/6/9): the sole authority
for which `InferenceRuntime` instance backs a logical model binding at any moment, and for driving
GPU allocation through `GPUResourceManager`. Providers, adapters, and the Model Registry's own
`create_runtime` never decide this themselves any more for vLLM-backed bindings -- they ask this
module (Part 6: "Providers shall know only InferenceRequest -> InferenceResponse. They shall never
know Docker, vLLM, GPU budgets, HTTP, ports.").

Responsibilities, matching the prompt's numbered requirements one-to-one:

- Runtime reuse (Part 3): one constructed `InferenceRuntime` per model id, cached for the life of
  the process, never rebuilt just because a provider is invoked again.
- Automatic startup (Part 4): `get_or_create` locates an existing runtime, health-checks it,
  restarts it if the check fails, and only then returns it -- callers never see the difference
  between "already warm" and "just restarted".
- Intelligent runtime policy (Part 9): if a new vLLM-backed reservation cannot fit, the
  least-recently-used *other* runtime is evicted (its container stopped, its GPU reservation
  released) and the reservation is retried -- automatically, with no operator involvement.
"""

from __future__ import annotations

import logging
from typing import Protocol

from archivetrust.runtime.contracts import InferenceRuntime
from archivetrust.runtime.gpu_resource_manager import GPUOversubscriptionError, GPUResourceManager
from archivetrust.runtime.model_registry import ModelRegistry
from archivetrust.runtime.models import ResolvedModel
from archivetrust.runtime.runtime_telemetry import RuntimeTelemetryKind, RuntimeTelemetrySink, _event

logger = logging.getLogger(__name__)

_DISCOVERED_RUNTIME_UTILIZATION_PLACEHOLDER = 0.75
"""Passed only to satisfy `VLLMRuntimeFactory`'s required `gpu_memory_utilization` argument when
constructing a runtime for an already-discovered, already-running server. Never actually consumed:
that value only ever reaches `docker run` inside `VLLMRuntime.warm_up()`'s "start a new container"
branch, and a discovered runtime's `warm_up()` will find the server already reachable (the same
check discovery itself just performed) and take the reuse branch instead, every time."""


class VLLMRuntimeFactory(Protocol):
    """Constructs a vLLM-backed `InferenceRuntime` given an already-computed GPU budget -- the
    seam `RuntimeManager` uses instead of `ModelRegistry`'s plain `RuntimeFactory` (which has no
    way to receive a `gpu_memory_utilization` value at all). Registered once by the composition
    root (`clients/desktop/composition.py`), exactly like every other runtime factory in this
    codebase.
    """

    def __call__(self, resolved: ResolvedModel, *, gpu_memory_utilization: float) -> InferenceRuntime: ...


class VLLMDiscoveryProbe(Protocol):
    """Answers "is a compatible vLLM server already running and healthy for this model?" without
    constructing a runtime or touching GPU bookkeeping -- the seam `RuntimeManager` consults before
    `VLLMRuntimeFactory` (Phase 24: Runtime Discovery and Reuse). Optional: a `RuntimeManager`
    constructed without one (the default) behaves exactly as before this phase -- creation is
    always attempted, never discovery.
    """

    def __call__(self, resolved: ResolvedModel) -> bool: ...


class RuntimeManager:
    """Owns a cache of constructed runtimes (keyed by model id, not logical name -- two logical
    bindings that happen to name the same model id share one runtime) and the GPU reservations that
    back the vLLM-kind ones. Not itself a `RuntimeFactory`/`InferenceRuntime` -- it is the layer
    above both, the thing a composition root asks instead of calling `ModelRegistry.create_runtime`
    directly whenever GPU-aware reuse matters (which, in practice, is always, for a desktop app that
    may run for hours).
    """

    def __init__(
        self,
        *,
        model_registry: ModelRegistry,
        gpu_resource_manager: GPUResourceManager,
        vllm_runtime_factory: VLLMRuntimeFactory,
        vllm_discovery_probe: VLLMDiscoveryProbe | None = None,
        telemetry: RuntimeTelemetrySink | None = None,
    ) -> None:
        self._model_registry = model_registry
        self._gpu = gpu_resource_manager
        self._vllm_factory = vllm_runtime_factory
        self._vllm_discovery_probe = vllm_discovery_probe
        self._telemetry = telemetry
        self._runtimes: dict[str, InferenceRuntime] = {}
        self._lru: list[str] = []
        """Least-recently-used ordering of cached model ids, oldest first -- the eviction candidate
        list `_reserve_with_eviction` walks (Part 9)."""

    def rebind(self, model_registry: ModelRegistry) -> None:
        """Points this manager at a new `ModelRegistry` (e.g. after `AppContext.open_workspace`
        constructs a fresh, Workspace-scoped one) without discarding the runtime cache or GPU
        reservations -- those track physical resources (Docker containers, VRAM) that outlive any
        one Workspace's `ModelRegistry` object, so switching workspaces should not force every
        vLLM-backed provider to cold-start again.
        """
        self._model_registry = model_registry

    def get_or_create(self, logical_name: str) -> InferenceRuntime:
        """Resolves `logical_name` and returns the runtime for it -- constructing, restarting, or
        reusing exactly as needed, transparently to the caller (Part 4). Does not itself warm the
        runtime (start a container / load weights) -- that stays exactly as lazy as it always was,
        deferred to the first real inference call, so merely *activating* a provider never requires
        GPU/Docker/network access.
        """
        try:
            resolved = self._model_registry.resolve(logical_name)
        except Exception as exc:
            # A resolution failure (no such binding, missing model source, ...) previously raised
            # before any telemetry line in this method executed -- durably indistinguishable from
            # "never attempted" (Production Hardening Review, 2026-07-13: every pre-reservation
            # failure path must leave a trace, same rationale as the vLLM-construction case below).
            logger.exception("RuntimeManager: failed to resolve logical model %s", logical_name)
            self._record(
                RuntimeTelemetryKind.RUNTIME_START_FAILED,
                model_id=logical_name, runtime_kind="unresolved", detail=repr(exc),
            )
            raise
        model_id = resolved.descriptor.model_id

        cached = self._runtimes.get(model_id)
        if cached is not None:
            if self._is_still_healthy(cached):
                self._touch(model_id)
                self._record(RuntimeTelemetryKind.RUNTIME_REUSED, model_id=model_id, runtime_kind=resolved.descriptor.runtime_kind)
                return cached
            self._record(
                RuntimeTelemetryKind.HEALTH_FAILURE, model_id=model_id, runtime_kind=resolved.descriptor.runtime_kind
            )
            self._forget(model_id)

        if resolved.descriptor.runtime_kind == "vllm":
            try:
                runtime, discovered = self._create_vllm_runtime(resolved)
            except Exception as exc:
                # A construction failure here (Docker unreachable, container start timeout, port
                # conflict, ...) previously propagated with no telemetry at all -- the GPU_ALLOCATED
                # event above it already recorded the reservation, but nothing recorded that the
                # runtime itself then failed to come up, so a crash here looked identical to "never
                # attempted" in any diagnostics view. Record before re-raising -- never swallow it.
                logger.exception("RuntimeManager: failed to start vLLM runtime for %s", model_id)
                self._record(
                    RuntimeTelemetryKind.RUNTIME_START_FAILED,
                    model_id=model_id, runtime_kind="vllm", detail=repr(exc),
                )
                raise
            # `RUNTIME_DISCOVERED` was already recorded inside `_create_vllm_runtime` for the
            # discovery path -- recording STARTED/RESTARTED here too would misreport a reused,
            # already-running container as one this process just launched.
            if not discovered:
                self._record(RuntimeTelemetryKind.RUNTIME_RESTARTED if cached is not None else RuntimeTelemetryKind.RUNTIME_STARTED, model_id=model_id, runtime_kind="vllm")
        else:
            # Non-vLLM runtime kinds (e.g. "transformers") place themselves on whatever device the
            # binding's `DeviceSelection` names and are not GPU-oversubscription risks in the same
            # multi-container sense -- reuse still applies (Part 3), GPU budgeting does not.
            # Deliberately *not* warmed up here: unlike a vLLM-backed runtime (where "start" means
            # "launch a Docker container", cheap and idempotent to do eagerly, Part 4), warming a
            # `TransformersRuntime` means loading real model weights into memory immediately --
            # this construction step must stay exactly as lazy as it always was (the first real
            # `RuntimeBackedLocalInferenceRunner.run()` call still triggers `warm_up()` itself), so
            # merely *activating* a provider never pays a model-load cost it may not need yet.
            try:
                runtime = self._model_registry.create_runtime(logical_name)
            except Exception as exc:
                logger.exception(
                    "RuntimeManager: failed to construct %s runtime for %s",
                    resolved.descriptor.runtime_kind, model_id,
                )
                self._record(
                    RuntimeTelemetryKind.RUNTIME_START_FAILED,
                    model_id=model_id, runtime_kind=resolved.descriptor.runtime_kind, detail=repr(exc),
                )
                raise
            self._record(RuntimeTelemetryKind.RUNTIME_STARTED, model_id=model_id, runtime_kind=resolved.descriptor.runtime_kind)

        self._runtimes[model_id] = runtime
        self._touch(model_id)
        return runtime

    def release_all(self) -> None:
        """Evicts every cached runtime -- for a clean application shutdown or test teardown."""
        for model_id in list(self._runtimes):
            self._evict(model_id)

    def _is_still_healthy(self, runtime: InferenceRuntime) -> bool:
        is_healthy = getattr(runtime, "is_healthy", None)
        if is_healthy is None:
            return True  # no health-recheck concept (e.g. an in-process runtime) -- trust it
        return bool(is_healthy())

    def _create_vllm_runtime(self, resolved: ResolvedModel) -> tuple[InferenceRuntime, bool]:
        """Returns `(runtime, discovered)` -- `discovered` is `True` only when an already-running
        server was found and attached to, so `get_or_create` knows not to also record a
        STARTED/RESTARTED telemetry event for it.
        """
        model_id = resolved.descriptor.model_id

        # Runtime discovery (Phase 24: Runtime Discovery and Reuse), before any GPU reservation is
        # attempted: a compatible server may already be running -- most commonly, a container this
        # same app's own previous session started and left alive across a restart, still holding
        # its GPU allocation outside this fresh process's in-memory cache. Reserving again here
        # would double-book VRAM that is already spent, which is exactly what previously produced
        # `GPUOversubscriptionError` on a routine restart. Creation is the fallback, not the
        # default: only attempted when no probe is configured, or the probe finds nothing to reuse.
        # This is a deliberate, narrow exception to the "activation never touches GPU/Docker/
        # network" invariant documented below: discovery is one cheap, bounded HTTP reachability
        # check (no container start, no GPU I/O) -- everything after it stays exactly as lazy.
        if self._vllm_discovery_probe is not None and self._vllm_discovery_probe(resolved):
            self._gpu.adopt(runtime_key=model_id, model_id=model_id)
            self._record(RuntimeTelemetryKind.RUNTIME_DISCOVERED, model_id=model_id, runtime_kind="vllm")
            runtime = self._vllm_factory(
                resolved, gpu_memory_utilization=_DISCOVERED_RUNTIME_UTILIZATION_PLACEHOLDER
            )
            return runtime, True

        utilization = self._reserve_with_eviction(model_id)
        total = self._gpu.total_memory_bytes()
        reservation = self._gpu.reservation_for(model_id)
        self._record(
            RuntimeTelemetryKind.GPU_ALLOCATED,
            model_id=model_id,
            runtime_kind="vllm",
            utilization=utilization,
            reserved_bytes=reservation.reserved_bytes if reservation else None,
            total_bytes=total,
        )
        # Deliberately *not* warmed up here. `RuntimeManager.get_or_create` is called from
        # `activate_configured_vision_provider` -- provider *activation*, not provider
        # *invocation* -- and this codebase's existing, tested invariant is that activation never
        # requires GPU/Docker/network access (every runtime kind stays lazy until the first real
        # `RuntimeBackedLocalInferenceRunner.run()` call, which already triggers `warm_up()`
        # itself). GPU *reservation* above is pure bookkeeping (no Docker/GPU I/O) and stays eager,
        # since the computed `gpu_memory_utilization` must be known at construction time regardless
        # of when the container actually starts; only starting the container is deferred.
        return self._vllm_factory(resolved, gpu_memory_utilization=utilization), False

    def _reserve_with_eviction(self, model_id: str) -> float:
        while True:
            try:
                return self._gpu.reserve(runtime_key=model_id, model_id=model_id)
            except GPUOversubscriptionError:
                victim = self._least_recently_used_other_than(model_id)
                if victim is None:
                    raise
                logger.info(
                    "RuntimeManager: insufficient GPU memory for %s -- evicting least-recently-used "
                    "runtime %s to make room.",
                    model_id,
                    victim,
                )
                self._evict(victim)

    def _least_recently_used_other_than(self, model_id: str) -> str | None:
        for candidate in self._lru:
            if candidate != model_id:
                return candidate
        return None

    def _evict(self, model_id: str) -> None:
        runtime = self._runtimes.pop(model_id, None)
        if model_id in self._lru:
            self._lru.remove(model_id)
        self._gpu.release(model_id)
        if runtime is not None:
            evict = getattr(runtime, "evict", None)
            if evict is not None:
                evict()
            else:
                runtime.shut_down()
        self._record(RuntimeTelemetryKind.GPU_RELEASED, model_id=model_id)
        self._record(RuntimeTelemetryKind.RUNTIME_STOPPED, model_id=model_id)

    def _forget(self, model_id: str) -> None:
        """Drops a cached runtime whose health check failed, without going through the full
        `_evict` container-stop path -- the container is already gone/unreachable; there is nothing
        left to stop, only bookkeeping to clear before `get_or_create` reconstructs it.
        """
        self._runtimes.pop(model_id, None)
        if model_id in self._lru:
            self._lru.remove(model_id)
        self._gpu.release(model_id)

    def _touch(self, model_id: str) -> None:
        if model_id in self._lru:
            self._lru.remove(model_id)
        self._lru.append(model_id)

    def _record(self, kind: RuntimeTelemetryKind, **kwargs: object) -> None:
        if self._telemetry is None:
            return
        self._telemetry.record(_event(kind, **kwargs))  # type: ignore[arg-type]
