"""`TransformersRuntime` — the production Hugging Face Transformers runtime (Part 2).

No Ollama dependency anywhere in this codebase to begin with (verified before writing this
milestone: `providers/qwen_vl/backends.py` already only depends on injected Protocols). This is a
genuinely new runtime, not a replacement of one.

`torch`/`transformers` are **lazy-imported** inside `warm_up()`, never at module import time — this
sandbox (like the one that built the OCR/VLM provider adapters in Milestone 3) has neither
installed, and the same discipline applies: the mapping/orchestration logic here is fully real and
tested via dependency injection (`_TorchAndTransformers` is an injectable seam standing in for the
two real libraries), while the actual library call is a documented, swappable integration point
exercised for real once a GPU-bearing deployment installs the `transformers` extra.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from archivetrust.runtime.contracts import (
    DeviceSelection,
    InferenceRequest,
    InferenceResult,
    Precision,
    RuntimeCapabilities,
)

_SUPPORTED_PRECISIONS = (
    Precision.AUTOMATIC,
    Precision.FP32,
    Precision.FP16,
    Precision.BF16,
    Precision.INT8,
)
# INT4 requires bitsandbytes/quantization support this runtime does not assume is installed;
# omitted from supported_precisions so a settings UI never offers a control that would fail
# (Part 9: "do not force every provider to share identical settings").


class TorchAndTransformersFacade(Protocol):
    """The exact surface `TransformersRuntime` needs from the real `torch` + `transformers`
    libraries, isolated behind one Protocol so it can be constructed for real (production) or
    injected as a fake (tests, this sandbox). Never imported directly by anything outside
    `transformers_runtime.py`.
    """

    def cuda_is_available(self) -> bool:
        ...

    def load_model(
        self, *, local_path: Path | None, model_id: str, revision: str | None, device: str, precision: str
    ) -> object:
        """Loads (or lazily downloads into the HF cache, if `local_path` is None) the model and
        returns an opaque handle this facade's own `generate` understands."""
        ...

    def generate(
        self, handle: object, *, page_image_ref: str, prompt: str, max_tokens: int | None, seed: int | None
    ) -> str:
        ...

    def release(self, handle: object) -> None:
        """Frees GPU memory / unloads the model — graceful shutdown (Part 2)."""
        ...


_VISION_MODEL_CLASS_NAMES = ("AutoModelForImageTextToText", "AutoModelForVision2Seq")
"""Candidate Auto classes tried in order, per model load (Multi-Provider Activation milestone).

`transformers` renamed its generic vision-to-text auto class from `AutoModelForVision2Seq` to
`AutoModelForImageTextToText`; neither is guaranteed to exist in every installed version. Each
class in this list is *tried* against the actual model (catching the "unrecognized configuration"
failure and falling through), not merely checked for existence -- the model actually loaded here
(Qwen2.5-VL) registers under one of these two names depending on the installed `transformers`
version.

`AutoModelForCausalLM` previously appeared in this list as a third fallback, added specifically to
accommodate PaddleOCR-VL (whose `auto_map` only registers `AutoModel`/`AutoModelForCausalLM`, not
either vision2seq name). That accommodation is now dead: PaddleOCR-VL runs exclusively through
`VLLMRuntime` (Runtime Architecture Completion milestone) -- its `transformers`-version-skew bugs
(`AttributeError: 'PaddleOCRVLConfig' object has no attribute 'text_config'`, `KeyError: 'default'`
in `ROPE_INIT_FUNCTIONS`, documented in
`docs/PADDLEOCR_VL_RUNTIME_COMPATIBILITY_INVESTIGATION.md`) never reach this runtime at all, so the
`AutoModelForCausalLM` fallback (which would have silently mis-resolved any model registering under
that generic name, not only PaddleOCR-VL) was removed rather than kept as unused breadth."""


def _load_model_with_auto_class(*, source: str, revision: str | None, dtype, extra_kwargs: dict):
    """Tries each candidate Auto class against this specific model, in order, falling through only
    on the "this model doesn't register under this Auto class" failure -- any other error (a real
    download/auth/OOM failure) propagates immediately rather than being masked by three retries.
    """
    import transformers  # noqa: PLC0415

    last_error: Exception | None = None
    for name in _VISION_MODEL_CLASS_NAMES:
        cls = getattr(transformers, name, None)
        if cls is None:
            continue
        try:
            return cls.from_pretrained(
                source, revision=revision, torch_dtype=dtype, trust_remote_code=True, **extra_kwargs
            )
        except ValueError as exc:
            if "Unrecognized configuration class" not in str(exc):
                raise
            last_error = exc
            continue
    raise ImportError(
        "No Auto model class in "
        f"{_VISION_MODEL_CLASS_NAMES} could load {source!r} "
        f"(installed transformers=={transformers.__version__}). Last error: {last_error}"
    )


def real_torch_and_transformers_facade() -> TorchAndTransformersFacade:
    """Constructs the facade backed by the real `torch`/`transformers` libraries. Raises
    `ModuleNotFoundError` with a clear message if they are not installed — never silently falls
    back to a fake in production. Not imported at module load time; call this only when actually
    running with the `transformers` extra installed.
    """
    import torch  # noqa: PLC0415 (intentionally lazy — see module docstring)
    from transformers import AutoProcessor  # noqa: PLC0415

    class _RealFacade:
        def cuda_is_available(self) -> bool:
            return torch.cuda.is_available()

        def load_model(self, *, local_path, model_id, revision, device, precision):
            source = str(local_path) if local_path is not None else model_id
            dtype = {
                "fp32": torch.float32,
                "fp16": torch.float16,
                "bf16": torch.bfloat16,
            }.get(precision, "auto")
            # `load_in_8bit` only passed when actually requested (Multi-Provider Activation
            # milestone, discovered during validation): newer model classes (e.g.
            # Qwen2_5_VLForConditionalGeneration on current transformers releases) no longer
            # accept this legacy kwarg at all -- `from_pretrained(..., load_in_8bit=False)` raised
            # `TypeError: unexpected keyword argument 'load_in_8bit'` even when int8 was never
            # requested. Never passing it unless `precision == "int8"` keeps every non-int8 load
            # working regardless of whether a given model class still supports the legacy kwarg;
            # requesting real int8 quantization is unaffected and out of this milestone's scope.
            extra_kwargs = {"load_in_8bit": True} if precision == "int8" else {}
            # `attn_implementation="sdpa"` was tried here (PyTorch's built-in fused
            # scaled-dot-product-attention kernel) as a throughput experiment and reverted
            # (Runtime Architecture Completion milestone): live benchmarking
            # (`docs/VISION_PROVIDER_BENCHMARK_STATUS.md`) found it caused a CUDA crash regression
            # on Qwen2.5-VL-3B with no measured speed benefit over `transformers`' default `"eager"`
            # attention. `from_pretrained` is called with no `attn_implementation` override, so
            # `transformers` picks its own default per model/hardware, exactly as before that
            # experiment.
            model = _load_model_with_auto_class(
                source=source, revision=revision, dtype=dtype, extra_kwargs=extra_kwargs
            ).to(device)
            # `trust_remote_code=True`: required for any repo shipping custom modeling/processing
            # code via `auto_map`; inert for a standard repo like Qwen2.5-VL-7B-Instruct that ships
            # no custom code at all.
            processor = AutoProcessor.from_pretrained(source, revision=revision, trust_remote_code=True)
            return (model, processor)

        def generate(self, handle, *, page_image_ref, prompt, max_tokens, seed):
            model, processor = handle
            if seed is not None:
                torch.manual_seed(seed)
            # Official multimodal usage (Qwen2.5-VL, PaddleOCR-VL, and every other chat-templated
            # VLM this runtime may load): the processor only expands its image-placeholder token
            # into the right number of vision-feature slots when that placeholder already appears
            # in the input text, at the count `apply_chat_template` inserts for this exact image --
            # calling the processor directly on plain text (no placeholder at all) leaves zero
            # slots reserved regardless of how many real image features the vision encoder
            # produces, which `transformers` correctly rejects rather than silently mismatching.
            # `apply_chat_template` is the documented way to get that placeholder in place; a
            # processor with no chat template at all (rare, and not true of either shipped VLM
            # provider) falls back to the previous plain-text call rather than raising.
            if getattr(processor, "chat_template", None):
                messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
                templated_text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = processor(text=[templated_text], images=[page_image_ref], return_tensors="pt").to(
                    model.device
                )
            else:
                inputs = processor(text=prompt, images=page_image_ref, return_tensors="pt").to(model.device)
            output = model.generate(**inputs, max_new_tokens=max_tokens or 1024)
            # Decode only the newly generated tokens (official Qwen2.5-VL usage pattern) -- the
            # full sequence includes the echoed prompt, which would corrupt downstream JSON parsing
            # (`QwenPageResponse.model_validate_json`) with the system/user prompt text prepended.
            generated_ids = output[0][inputs["input_ids"].shape[-1] :]
            return processor.decode(generated_ids, skip_special_tokens=True)

        def release(self, handle) -> None:
            model, _processor = handle
            del model
            torch.cuda.empty_cache()

    return _RealFacade()


@dataclass
class _LoadedModel:
    handle: object
    device_used: str
    precision_used: Precision


class TransformersRuntime:
    """`InferenceRuntime` backed by Hugging Face `transformers`. Constructed already bound to one
    resolved model (`ResolvedModel`, via the Model Registry) — it never resolves its own model path.

    Device selection: `AUTOMATIC` picks GPU when `facade.cuda_is_available()`, else CPU (automatic
    GPU detection + automatic CPU fallback, Part 2); `GPU_ONLY` raises at `warm_up()` if no GPU is
    present rather than silently running on CPU (an operator who chose GPU-only should never
    unknowingly get CPU-speed inference); `CPU_ONLY` always runs on CPU.
    """

    runtime_kind = "transformers"

    def __init__(
        self,
        *,
        model_id: str,
        local_path: Path | None,
        revision: str | None,
        device: DeviceSelection,
        precision: Precision,
        facade: TorchAndTransformersFacade,
    ) -> None:
        self._model_id = model_id
        self._local_path = local_path
        self._revision = revision
        self._device_selection = device
        self._precision = precision
        self._facade = facade
        self._loaded: _LoadedModel | None = None

    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(
            supports_gpu=True,
            supports_cpu=True,
            supported_precisions=_SUPPORTED_PRECISIONS,
            supports_batching=True,
            supports_deterministic_seed=True,
            max_tokens_configurable=True,
        )

    def _resolve_device(self) -> str:
        gpu_available = self._facade.cuda_is_available()
        if self._device_selection == DeviceSelection.GPU_ONLY:
            if not gpu_available:
                raise RuntimeError(
                    "TransformersRuntime configured for GPU_ONLY but no CUDA device is available"
                )
            return "cuda"
        if self._device_selection == DeviceSelection.CPU_ONLY:
            return "cpu"
        return "cuda" if gpu_available else "cpu"  # AUTOMATIC

    def _resolve_precision(self, device: str) -> Precision:
        if self._precision != Precision.AUTOMATIC:
            return self._precision
        return Precision.FP16 if device == "cuda" else Precision.FP32

    def warm_up(self) -> None:
        """Graceful initialization: load weights once, before the first real request."""
        if self._loaded is not None:
            return
        device = self._resolve_device()
        precision = self._resolve_precision(device)
        handle = self._facade.load_model(
            local_path=self._local_path,
            model_id=self._model_id,
            revision=self._revision,
            device=device,
            precision=precision.value,
        )
        self._loaded = _LoadedModel(handle=handle, device_used=device, precision_used=precision)

    def infer(self, request: InferenceRequest) -> InferenceResult:
        self.warm_up()  # graceful: a caller that skipped warm_up still gets a correct, if slower, first call
        assert self._loaded is not None
        started = time.monotonic()
        raw_text = self._facade.generate(
            self._loaded.handle,
            page_image_ref=request.page_image_ref,
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            seed=request.seed,
        )
        elapsed = time.monotonic() - started
        model_version = f"{self._model_id}@{self._revision or 'default'}"
        return InferenceResult(
            raw_text=raw_text,
            model_version=model_version,
            device_used=self._loaded.device_used,
            precision_used=self._loaded.precision_used,
            inference_seconds=elapsed,
        )

    def shut_down(self) -> None:
        """Graceful shutdown: safe to call even if `warm_up` was never called."""
        if self._loaded is not None:
            self._facade.release(self._loaded.handle)
            self._loaded = None
