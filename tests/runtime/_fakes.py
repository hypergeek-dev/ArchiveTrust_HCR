from __future__ import annotations

from archivetrust.runtime.contracts import (
    DeviceSelection,
    InferenceRequest,
    InferenceResult,
    Precision,
    RuntimeCapabilities,
)


class FakeRuntime:
    """A minimal `InferenceRuntime` for tests -- no torch/transformers/network involved."""

    runtime_kind = "fake"

    def __init__(self, *, device_used: str = "cpu", precision: Precision = Precision.FP32) -> None:
        self.warm_up_calls = 0
        self.shut_down_calls = 0
        self.infer_calls: list[InferenceRequest] = []
        self._device_used = device_used
        self._precision = precision

    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(
            supports_gpu=True,
            supports_cpu=True,
            supported_precisions=(Precision.AUTOMATIC, Precision.FP32, Precision.FP16),
            supports_batching=False,
            supports_deterministic_seed=True,
            max_tokens_configurable=True,
        )

    def warm_up(self) -> None:
        self.warm_up_calls += 1

    def infer(self, request: InferenceRequest) -> InferenceResult:
        self.infer_calls.append(request)
        return InferenceResult(
            raw_text='{"page_number": 1, "items": []}',
            model_version="fake-model@1",
            device_used=self._device_used,
            precision_used=self._precision,
            inference_seconds=0.005,
        )

    def shut_down(self) -> None:
        self.shut_down_calls += 1


class FakeTorchAndTransformersFacade:
    """Stands in for the real torch/transformers libraries (never imported in this sandbox)."""

    def __init__(self, *, cuda_available: bool) -> None:
        self._cuda_available = cuda_available
        self.loaded: dict | None = None
        self.released: list[object] = []

    def cuda_is_available(self) -> bool:
        return self._cuda_available

    def load_model(self, *, local_path, model_id, revision, device, precision):
        handle = {"model_id": model_id, "device": device, "precision": precision}
        self.loaded = handle
        return handle

    def generate(self, handle, *, page_image_ref, prompt, max_tokens, seed):
        return f"generated:{handle['model_id']}:{prompt}"

    def release(self, handle) -> None:
        self.released.append(handle)


class FakeHttpTransport:
    """Stands in for a real HTTP client for `OpenAICompatibleRuntime` tests."""

    def __init__(self, response: dict) -> None:
        self._response = response
        self.calls: list[dict] = []

    def post_json(self, *, url: str, payload: dict, headers: dict[str, str]) -> dict:
        self.calls.append({"url": url, "payload": payload, "headers": headers})
        return self._response


class FakeContainerLifecycle:
    """Stands in for the real `docker` CLI for `VLLMRuntime` tests -- no Docker daemon involved."""

    def __init__(self, *, already_running: bool = False) -> None:
        self._running: set[str] = {"preexisting"} if already_running else set()
        self.start_calls: list[dict] = []
        self.stop_calls: list[str] = []

    def is_running(self, container_name: str) -> bool:
        return container_name in self._running

    def mark_running(self, container_name: str) -> None:
        """Test helper: simulates a container left running from a prior session."""
        self._running.add(container_name)

    def start(
        self,
        *,
        container_name: str,
        image: str,
        port: int,
        model_id: str,
        gpu_device: str | None,
        gpu_memory_utilization: float,
        max_model_len: int,
        max_num_seqs: int | None = None,
    ) -> None:
        self.start_calls.append(
            {
                "container_name": container_name,
                "image": image,
                "port": port,
                "model_id": model_id,
                "gpu_device": gpu_device,
                "gpu_memory_utilization": gpu_memory_utilization,
                "max_model_len": max_model_len,
                "max_num_seqs": max_num_seqs,
            }
        )
        self._running.add(container_name)

    def stop(self, container_name: str) -> None:
        self.stop_calls.append(container_name)
        self._running.discard(container_name)


class FakeHealthProbe:
    """Stands in for a real HTTP health check for `VLLMRuntime` tests. `healthy_after_calls=0`
    means reachable immediately; a positive value simulates a server that takes a few polls to
    come up, without a real `time.sleep` (tests pass `health_poll_interval_seconds=0`)."""

    def __init__(self, *, healthy_after_calls: int = 0) -> None:
        self._healthy_after_calls = healthy_after_calls
        self.calls = 0

    def is_reachable(self, url: str) -> bool:
        self.calls += 1
        return self.calls > self._healthy_after_calls
