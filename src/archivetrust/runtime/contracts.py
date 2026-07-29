"""The `InferenceRuntime` abstraction (Part 1).

A runtime's whole job is: given an image reference and a prompt, run a model and return raw text
plus enough provenance to reproduce the run later. It knows nothing about documents, providers, or
the Canonical Observation Ontology — that separation is the entire point of this milestone.

Concrete runtimes (`TransformersRuntime`, `OpenAICompatibleRuntime`, and future
`OllamaRuntime`/`VLLMRuntime`/`LlamaCppRuntime`) all implement this one Protocol, so a provider (via
the bridge in `providers/qwen_vl/runtime_backend.py`) never branches on which runtime is in use.
"""

from __future__ import annotations

from enum import Enum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict


class DeviceSelection(str, Enum):
    """How the operator wants inference placed (Part 8/9's "Execution" setting)."""

    AUTOMATIC = "automatic"
    GPU_ONLY = "gpu_only"
    CPU_ONLY = "cpu_only"


class Precision(str, Enum):
    """Numeric precision for local inference (Part 9). Not every runtime/model supports every
    value — `RuntimeCapabilities.supported_precisions` states which are actually available, so a
    settings UI never offers a control that would silently fail (Part 9: "do not force every
    provider to share identical settings")."""

    AUTOMATIC = "automatic"
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"


class RuntimeCapabilities(BaseModel):
    """What a runtime *can* do — read by settings UIs to show only supported controls (Part 9),
    and by the Model Registry to reject an incompatible pairing before it fails at inference time.
    """

    model_config = ConfigDict(frozen=True)

    supports_gpu: bool
    supports_cpu: bool
    supported_precisions: tuple[Precision, ...]
    supports_batching: bool
    supports_deterministic_seed: bool
    max_tokens_configurable: bool


class InferenceRequest(BaseModel):
    """One inference call. `seed` is `None` unless deterministic mode is requested (Part 9); a
    runtime that cannot honor a seed must say so via `RuntimeCapabilities.supports_deterministic_seed`
    rather than silently ignoring it.
    """

    model_config = ConfigDict(frozen=True)

    page_image_ref: str
    prompt: str
    max_tokens: int | None = None
    seed: int | None = None


class InferenceResult(BaseModel):
    """One inference call's result, plus the provenance needed to reproduce it later (Part 13:
    "record complete runtime provenance"). `model_version` is what
    `providers.qwen_vl.native.QwenRawResponse.model_version` is populated from — the same field
    the existing adapter already requires (S3.5.5/S5.5: "model/version provenance").

    The fields below `inference_seconds` are additive (all optional, default `None`/`0`) — a
    runtime with nothing to report for one (e.g. `TransformersRuntime` has no network hop, so
    `network_seconds` stays `None`) simply leaves it unset; no existing runtime's behavior changes.
    `VLLMRuntime` is the first runtime to populate the fuller set (Runtime Architecture Completion
    milestone: server startup, network, retry, and queue provenance for an HTTP-served runtime).
    """

    model_config = ConfigDict(frozen=True)

    raw_text: str
    model_version: str
    device_used: str
    precision_used: Precision
    inference_seconds: float
    runtime_version: str | None = None
    """The concrete runtime's own version string (e.g. the vLLM server version reported at
    startup), distinct from `model_version` (which identifies the model/checkpoint)."""
    startup_seconds: float | None = None
    """How long this runtime spent becoming ready (server boot + health-check polling) before this
    call could proceed — `0.0` (not `None`) once warm, so a rolling average over many calls
    correctly shows startup cost amortizing to zero, never `None`-skipped into an undercount."""
    network_seconds: float | None = None
    """Time spent on the wire for an HTTP-served runtime; `None` for an in-process runtime."""
    retry_count: int = 0
    queue_wait_seconds: float | None = None
    """Time this request waited behind another before the server started processing it, when the
    runtime can observe that (e.g. from the server's own queue-position response headers)."""
    gpu_used: str | None = None
    """A human-readable GPU identifier the runtime actually used (e.g. `"cuda:0"` or a reported
    device name), when it can determine one — distinct from `device_used`, which is the coarse
    class of device ("cuda"/"cpu"/"remote")."""
    health_state: str | None = None
    """The runtime's own health-check verdict at the time of this call (e.g. "healthy",
    "degraded") when it exposes one; `None` for a runtime with no separate health concept."""


@runtime_checkable
class InferenceRuntime(Protocol):
    """Executes inference against one resolved model. A runtime is constructed already bound to a
    specific model (via the Model Registry, `model_registry.py`) — it does not itself decide which
    model to load.
    """

    runtime_kind: str
    """A short, stable identifier ("transformers", "openai_compatible", "ollama", ...) — recorded
    verbatim in telemetry provenance (Part 13), never inferred from a class name."""

    def capabilities(self) -> RuntimeCapabilities:
        ...

    def infer(self, request: InferenceRequest) -> InferenceResult:
        ...

    def warm_up(self) -> None:
        """Graceful initialization (Part 2): load weights / establish a connection before the first
        real request, so the first document processed isn't penalized by cold-start latency. A
        no-op for runtimes with nothing to warm (e.g. a stateless HTTP client)."""
        ...

    def shut_down(self) -> None:
        """Graceful shutdown (Part 2): release GPU memory / close connections. Must be safe to call
        even if `warm_up` was never called."""
        ...


class DetectedRegion(BaseModel):
    """One region a layout detector found on a page — real, provider-native geometry (Phase 31),
    never an inherited/estimated box. Coordinates are absolute pixels, top-left origin, matching the
    convention `domain/evidence/models.py`'s `BoundingBox` already uses for `Precision.PIXEL_ACCURATE`
    geometry from Docling/Tesseract.
    """

    model_config = ConfigDict(frozen=True)

    native_label: str
    x0: float
    y0: float
    x1: float
    y1: float
    confidence: float | None = None
    reading_order_index: int
    """The detector's own reported reading-order position — preserved verbatim, never re-derived or
    re-sorted by ArchiveTrust (Article 6: nothing the provider computed is silently discarded)."""


class LayoutDetectionRequest(BaseModel):
    """One layout-detection call — a whole rendered page, not a crop (cropping happens after
    detection, driven by its result)."""

    model_config = ConfigDict(frozen=True)

    page_image_ref: str


class LayoutDetectionResult(BaseModel):
    """One layout-detection call's result, plus enough provenance to reproduce it later (mirrors
    `InferenceResult`'s provenance discipline, Part 13) — deliberately its own type rather than
    shoehorned into `InferenceResult.raw_text`, since a detector's output is structured geometry,
    not text (see `LayoutDetectionRuntime`'s docstring)."""

    model_config = ConfigDict(frozen=True)

    regions: tuple[DetectedRegion, ...]
    model_version: str
    inference_seconds: float | None = None


@runtime_checkable
class LayoutDetectionRuntime(Protocol):
    """Executes layout detection against one resolved model — the sibling of `InferenceRuntime` for
    a runtime whose job is "find regions on a page," not "given an image and a prompt, generate
    text" (Phase 31: PaddleOCR-VL Structured Pipeline Integration, PP-DocLayoutV2).

    Deliberately NOT a variant of `InferenceRuntime`: forcing a detector's structured `DetectedRegion`
    output into `InferenceResult.raw_text` (e.g. as serialized JSON) would mean callers expecting
    text-shaped provenance have to know to re-parse it, and would misrepresent what this runtime
    actually does to anything inspecting `RuntimeCapabilities`/telemetry generically. Instead this
    Protocol reuses only the members that are genuinely generic across every runtime shape
    (`runtime_kind`, `capabilities`, `warm_up`, `shut_down`) and defines its own `detect` method with
    its own request/result types. A `RuntimeManager`/`ModelRegistry` consumer that only needs
    lifecycle management (not the inference call itself) can treat both Protocols identically; only
    the actual inference call site needs to know which one it has.
    """

    runtime_kind: str

    def capabilities(self) -> RuntimeCapabilities:
        ...

    def detect(self, request: LayoutDetectionRequest) -> LayoutDetectionResult:
        ...

    def warm_up(self) -> None:
        ...

    def shut_down(self) -> None:
        ...
