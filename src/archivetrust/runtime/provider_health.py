"""Provider Health (Part 6): runtime information every provider exposes — installed, ready,
backend, GPU/VRAM, runtime kind, average inference time, last execution, model revision, device.

This is deliberately distinct from `learning.analytics.provider_health.ProviderHealth`, which
measures *behavioral* health from Trust Engine telemetry (invocation/rejection/correction counts —
"is this provider's output trustworthy over time?"). This module measures *operational* health from
the runtime itself ("is this provider's inference stack actually up, and how fast?"). Part 6 states
they feed the same Quality Center; they remain two different questions about two different things,
so they are two different types, never merged into one.
"""

from __future__ import annotations

import time
from collections import deque

from pydantic import BaseModel, ConfigDict

from archivetrust.runtime.contracts import InferenceResult, Precision


class ProviderHealthSnapshot(BaseModel):
    """A point-in-time operational health report for one provider (Part 6)."""

    model_config = ConfigDict(frozen=True)

    provider_id: str
    installed: bool
    ready: bool
    """`warm_up()` has succeeded and the runtime is prepared to serve inference."""
    backend: str
    """The runtime kind in use ("transformers", "openai_compatible", ...)."""
    device: str | None
    gpu_available: bool | None
    """`None` when unknown (e.g. a provider with no runtime yet configured); never guessed."""
    average_inference_seconds: float | None
    last_execution_epoch: float | None
    model_revision: str | None
    precision: Precision | None


class ProviderHealthTracker:
    """Accumulates `InferenceResult`s for one provider and produces `ProviderHealthSnapshot`s on
    demand. `window` bounds how many recent inferences the rolling average is computed over, so a
    provider's average reflects recent behavior (e.g. after a device/precision change) rather than
    being diluted by its entire lifetime history.
    """

    def __init__(self, provider_id: str, *, window: int = 50) -> None:
        self._provider_id = provider_id
        self._window = window
        self._recent: deque[InferenceResult] = deque(maxlen=window)
        self._ready = False
        self._backend: str | None = None
        self._last_execution_epoch: float | None = None
        self._model_revision: str | None = None

    def mark_ready(self, *, backend: str) -> None:
        self._ready = True
        self._backend = backend

    def record(self, result: InferenceResult, *, model_revision: str | None = None) -> None:
        self._recent.append(result)
        self._last_execution_epoch = time.time()
        if model_revision is not None:
            self._model_revision = model_revision

    def snapshot(self) -> ProviderHealthSnapshot:
        average = (
            sum(r.inference_seconds for r in self._recent) / len(self._recent)
            if self._recent
            else None
        )
        last = self._recent[-1] if self._recent else None
        return ProviderHealthSnapshot(
            provider_id=self._provider_id,
            installed=self._backend is not None,
            ready=self._ready,
            backend=self._backend or "unconfigured",
            device=last.device_used if last else None,
            gpu_available=(last.device_used == "cuda") if last else None,
            average_inference_seconds=average,
            last_execution_epoch=self._last_execution_epoch,
            model_revision=self._model_revision,
            precision=last.precision_used if last else None,
        )
