from __future__ import annotations

import pytest

from archivetrust.runtime.contracts import InferenceResult, Precision
from archivetrust.runtime.provider_health import ProviderHealthTracker


def _result(seconds: float) -> InferenceResult:
    return InferenceResult(
        raw_text="{}", model_version="m@1", device_used="cuda", precision_used=Precision.FP16,
        inference_seconds=seconds,
    )


def test_snapshot_before_any_activity_is_not_ready() -> None:
    tracker = ProviderHealthTracker("qwen2.5-vl")
    snapshot = tracker.snapshot()
    assert not snapshot.installed
    assert not snapshot.ready
    assert snapshot.average_inference_seconds is None


def test_mark_ready_reports_installed_and_backend() -> None:
    tracker = ProviderHealthTracker("qwen2.5-vl")
    tracker.mark_ready(backend="transformers")
    snapshot = tracker.snapshot()
    assert snapshot.installed
    assert snapshot.ready
    assert snapshot.backend == "transformers"


def test_average_inference_time_over_recorded_calls() -> None:
    tracker = ProviderHealthTracker("qwen2.5-vl")
    tracker.record(_result(0.10))
    tracker.record(_result(0.20))
    snapshot = tracker.snapshot()
    assert snapshot.average_inference_seconds == pytest.approx(0.15)


def test_window_bounds_the_rolling_average() -> None:
    tracker = ProviderHealthTracker("qwen2.5-vl", window=2)
    tracker.record(_result(1.0))
    tracker.record(_result(0.1))
    tracker.record(_result(0.1))  # evicts the first 1.0s call
    assert tracker.snapshot().average_inference_seconds == 0.1


def test_model_revision_updates_on_record() -> None:
    tracker = ProviderHealthTracker("qwen2.5-vl")
    tracker.record(_result(0.1), model_revision="qwen2.5-vl@main")
    assert tracker.snapshot().model_revision == "qwen2.5-vl@main"
