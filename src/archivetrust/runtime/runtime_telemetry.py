"""Operational runtime telemetry (Production Runtime Completion milestone, Part 7): describes what
the *runtime layer itself* did -- containers started/stopped/reused/restarted, GPU memory allocated
or released, and how long warm-up took. Deliberately kept out of `domain/telemetry/events.py`:
Constitution Article 16 states that module records changes in *document knowledge* and that
"process-level logging is a separate, non-canonical concern [with] no home here". It is also kept
distinct from `learning.analytics.provider_health.ProviderHealth` (behavioral trust, derived from
Trust Engine telemetry) and from `runtime.provider_health.ProviderHealthTracker` (per-provider
operational health: is it up, how fast). This module is the runtime-*infrastructure* layer's own
operational record -- who started which container, when, and why -- one level below either of
those, and is exactly what `RuntimeManager` populates as it makes allocation/eviction/restart
decisions.
"""

from __future__ import annotations

import threading
import time
from enum import Enum
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, TypeAdapter


class RuntimeTelemetryKind(str, Enum):
    RUNTIME_STARTED = "RuntimeStarted"
    RUNTIME_STOPPED = "RuntimeStopped"
    RUNTIME_REUSED = "RuntimeReused"
    RUNTIME_DISCOVERED = "RuntimeDiscovered"
    """A compatible runtime was found already running and healthy *outside* this process's own
    in-memory cache -- e.g. a container a previous session of this same app left alive -- and was
    attached to instead of reserving GPU memory for, and starting, a new one (Phase 24: Runtime
    Discovery and Reuse). Distinct from `RUNTIME_REUSED`, which is an in-process cache hit within
    the same `RuntimeManager` instance; this is the cross-process/cross-restart case."""
    RUNTIME_RESTARTED = "RuntimeRestarted"
    RUNTIME_START_FAILED = "RuntimeStartFailed"
    RUNTIME_RECYCLED = "RuntimeRecycled"
    """A long-lived in-process component (e.g. `RealDoclingClient`'s `DocumentConverter`) was
    deliberately torn down and rebuilt on a bounded schedule -- distinct from `RUNTIME_RESTARTED`,
    which is a *reaction* to a detected failure. Recorded proactively, before anything has gone
    wrong, as a bound on native/C-extension memory growth across a long-running batch (Production
    Incident, 2026-07-14: `RealDoclingClient` held one converter for an entire 1,000-document run
    with no recycling; the first `std::bad_alloc` symptom appeared at document #367 of 1,000)."""
    HEALTH_FAILURE = "HealthFailure"
    GPU_ALLOCATED = "GPUAllocated"
    GPU_RELEASED = "GPUReleased"
    WARMUP_TIME = "WarmupTime"


class RuntimeTelemetryEvent(BaseModel):
    """One fact about the runtime layer's own behavior. `recorded_at` is a bare `time.time()` epoch
    -- process-local wall-clock, deliberately not the domain telemetry stream's UTC ISO-8601
    `recorded_at` (Article 16's separation applies to the clock too: this stream is never merged
    into, or read as though it were, the canonical knowledge-evolution log).
    """

    model_config = ConfigDict(frozen=True)

    kind: RuntimeTelemetryKind
    recorded_at: float
    model_id: str
    runtime_kind: str | None = None
    gpu_device: str | None = None
    utilization: float | None = None
    reserved_bytes: int | None = None
    total_bytes: int | None = None
    seconds: float | None = None
    detail: str | None = None


def _event(kind: RuntimeTelemetryKind, *, model_id: str, **kwargs: object) -> RuntimeTelemetryEvent:
    return RuntimeTelemetryEvent(kind=kind, recorded_at=time.time(), model_id=model_id, **kwargs)  # type: ignore[arg-type]


class RuntimeTelemetrySink(Protocol):
    def record(self, event: RuntimeTelemetryEvent) -> None: ...


class InMemoryRuntimeTelemetrySink:
    """A process-local sink -- everything it records is lost at process exit. Useful for tests and
    as `FileRuntimeTelemetrySink`'s in-memory mirror; production wiring should prefer
    `FileRuntimeTelemetrySink` so a crash (a killed container, a native-library abort, a timeout)
    leaves a durable trace instead of silence.
    """

    def __init__(self) -> None:
        self._events: list[RuntimeTelemetryEvent] = []

    def record(self, event: RuntimeTelemetryEvent) -> None:
        self._events.append(event)

    def events(self) -> tuple[RuntimeTelemetryEvent, ...]:
        return tuple(self._events)


class FileRuntimeTelemetrySink:
    """Durable, append-only JSON-Lines runtime-telemetry stream -- the runtime-layer analogue of
    `infrastructure.storage.telemetry_sink.FileTelemetrySink`, same write-through-mirror design: the
    file is the single durable record, the in-memory list a cache of it loaded once at construction.

    This is what makes a container start/stop/health-failure/GPU-allocation history (and, since
    `RUNTIME_START_FAILED`, a runtime construction failure) survive an app crash -- previously this
    whole stream lived only in `InMemoryRuntimeTelemetrySink` and vanished the moment the process
    exited or was killed, which meant a crash mid-run left no runtime-level trace to diagnose it by,
    only whatever domain telemetry happened to be flushed already.
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._path.touch()
        self._events: list[RuntimeTelemetryEvent] = self._load()

    def _load(self) -> list[RuntimeTelemetryEvent]:
        adapter = TypeAdapter(RuntimeTelemetryEvent)
        events = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                events.append(adapter.validate_json(line))
        return events

    def record(self, event: RuntimeTelemetryEvent) -> None:
        with self._lock:
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(event.model_dump_json())
                handle.write("\n")
            self._events.append(event)

    def events(self) -> tuple[RuntimeTelemetryEvent, ...]:
        return tuple(self._events)
