"""`OpenAICompatibleRuntime` — a second, genuinely different runtime implementation, proving the
abstraction is real (Part 1: "the architecture should support multiple ... without requiring
recompilation").

Talks to any OpenAI-compatible vision endpoint (a hosted API, an enterprise gateway, a self-hosted
vLLM/llama.cpp server exposing the same wire format) via an injected `HttpTransport` — no concrete
HTTP client is imported here, for the same reason `providers/qwen_vl/backends.py` injects its
transport: which client (httpx, requests, urllib) to depend on is an infrastructure choice, and
this sandbox has no network access to exercise a real one against. The request/response shaping is
fully real; the wire call is a documented, swappable seam.
"""

from __future__ import annotations

import time
from typing import Protocol

from archivetrust.runtime.contracts import (
    DeviceSelection,
    InferenceRequest,
    InferenceResult,
    Precision,
    RuntimeCapabilities,
)


class HttpTransport(Protocol):
    """One HTTP POST to an OpenAI-compatible chat/vision endpoint."""

    def post_json(self, *, url: str, payload: dict, headers: dict[str, str]) -> dict:
        """Returns the parsed JSON response body."""
        ...


class OpenAICompatibleRuntime:
    """`InferenceRuntime` for any OpenAI-compatible vision endpoint. Device selection is meaningless
    for a remote endpoint (the server decides), so `capabilities()` reports GPU/CPU support as
    "whatever the endpoint does," and `device_used`/`precision_used` on the result reflect what the
    server reports, never a value this runtime invents.
    """

    runtime_kind = "openai_compatible"

    def __init__(
        self,
        *,
        base_url: str,
        model_id: str,
        api_key: str | None,
        transport: HttpTransport,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model_id = model_id
        self._api_key = api_key
        self._transport = transport
        self._warmed_up = False

    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(
            supports_gpu=True,
            supports_cpu=True,
            supported_precisions=(Precision.AUTOMATIC,),  # server-side; not client-configurable
            supports_batching=False,
            supports_deterministic_seed=True,
            max_tokens_configurable=True,
        )

    def warm_up(self) -> None:
        """A single lightweight ping, so connection setup doesn't penalize the first document."""
        if self._warmed_up:
            return
        headers = self._headers()
        self._transport.post_json(
            url=f"{self._base_url}/models", payload={}, headers=headers
        )
        self._warmed_up = True

    def infer(self, request: InferenceRequest) -> InferenceResult:
        started = time.monotonic()
        payload = {
            "model": self._model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": request.prompt},
                        {"type": "image_url", "image_url": {"url": request.page_image_ref}},
                    ],
                }
            ],
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.seed is not None:
            payload["seed"] = request.seed

        response = self._transport.post_json(
            url=f"{self._base_url}/chat/completions", payload=payload, headers=self._headers()
        )
        elapsed = time.monotonic() - started
        raw_text = response["choices"][0]["message"]["content"]
        model_version = response.get("model", self._model_id)
        return InferenceResult(
            raw_text=raw_text,
            model_version=model_version,
            device_used="remote",
            precision_used=Precision.AUTOMATIC,
            inference_seconds=elapsed,
        )

    def shut_down(self) -> None:
        """No persistent resource to release for a stateless HTTP client."""
        self._warmed_up = False

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers
