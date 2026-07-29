"""`FileRuntimeTelemetrySink` durability (Production Investigation follow-up): runtime-layer
telemetry (container starts/stops, GPU allocation, `RUNTIME_START_FAILED`) previously lived only in
`InMemoryRuntimeTelemetrySink` and vanished the moment the process exited or was killed -- exactly
the gap that made a crash mid-run undiagnosable from telemetry alone. These tests only cover the
sink's own read/write/reload contract; `RuntimeManager`'s own behavior is covered by
`test_runtime_manager.py`, which only depends on the `RuntimeTelemetrySink` Protocol.
"""

from __future__ import annotations

from archivetrust.runtime.runtime_telemetry import FileRuntimeTelemetrySink, RuntimeTelemetryKind, _event


def test_recorded_events_survive_reconstruction(tmp_path) -> None:
    path = tmp_path / "runtime_events.jsonl"
    sink = FileRuntimeTelemetrySink(path)
    sink.record(_event(RuntimeTelemetryKind.RUNTIME_STARTED, model_id="PaddlePaddle/PaddleOCR-VL", runtime_kind="vllm"))
    sink.record(_event(RuntimeTelemetryKind.GPU_ALLOCATED, model_id="PaddlePaddle/PaddleOCR-VL", utilization=0.5))

    reloaded = FileRuntimeTelemetrySink(path)

    kinds = [event.kind for event in reloaded.events()]
    assert kinds == [RuntimeTelemetryKind.RUNTIME_STARTED, RuntimeTelemetryKind.GPU_ALLOCATED]


def test_a_construction_failure_recorded_before_a_crash_is_still_on_disk(tmp_path) -> None:
    """Simulates the scenario a crash mid-run previously lost entirely: the event is durable the
    instant `record()` returns, not only once some later graceful-shutdown path flushes it."""
    path = tmp_path / "runtime_events.jsonl"
    sink = FileRuntimeTelemetrySink(path)

    sink.record(
        _event(
            RuntimeTelemetryKind.RUNTIME_START_FAILED,
            model_id="PaddlePaddle/PaddleOCR-VL", runtime_kind="vllm",
            detail="docker daemon unreachable",
        )
    )

    # A fresh process re-reading the same file (the only thing a real crash leaves behind) must see it.
    events_on_disk = FileRuntimeTelemetrySink(path).events()
    assert len(events_on_disk) == 1
    assert events_on_disk[0].kind == RuntimeTelemetryKind.RUNTIME_START_FAILED
    assert events_on_disk[0].detail == "docker daemon unreachable"


def test_creates_parent_directories_and_starts_empty(tmp_path) -> None:
    path = tmp_path / "nested" / "runtime_events.jsonl"
    sink = FileRuntimeTelemetrySink(path)
    assert path.exists()
    assert sink.events() == ()
