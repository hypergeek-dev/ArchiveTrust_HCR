"""`TransformersRuntime` — the general-purpose Hugging Face Transformers runtime (Part 2).

`torch`/`transformers` are **lazy-imported** inside `warm_up()`, never at module import time. The
mapping/orchestration logic here is fully real and tested via dependency injection
(`TorchAndTransformersFacade` is an injectable seam standing in for the two real libraries), while the
actual library call is a swappable integration point.

.. note::

   **No HTR method uses this module, and the Qwen-shaped assumptions in it are therefore inert.
   Verified by inspection on 2026-07-30, not assumed.**

   `docs/htr-repository-cleanup.md`'s Stage 5 record flagged this file as still carrying
   *"attention-implementation quirks, `Qwen2_5_VLForConditionalGeneration` class-name matching, a
   decode pattern tuned to Qwen's chat template"* and deferred the fix to "Florence-2's implementation
   phase" — on the unstated premise that Florence-2 would run through here. **It does not.** The
   premise was never checked. It is checked now, and two of the three flagged items turn out not to
   exist in code at all:

   1. **Florence-2 does not use this runtime.** `providers/florence2_htr/facade.py` loads its own model
      directly — `from transformers import AutoModelForCausalLM, AutoProcessor` inside its own
      `run_inference`. It imports nothing from `archivetrust.runtime`. Neither does SATRN (which runs
      in an isolated subprocess venv) nor Transkribus (which parses a file). `grep -rn "from
      archivetrust.runtime" src/archivetrust/providers/` returns **zero hits**. The two mentions of
      this module in the SATRN and Florence-2 facades are docstring citations of its *test-seam and
      lazy-import discipline*, not imports.

      The class list is in fact positive evidence of the separation: `_VISION_MODEL_CLASS_NAMES` no
      longer contains `AutoModelForCausalLM`, which is precisely the Auto class Florence-2 needs. If
      Florence-2 were routed through here it would fail to load at all, loudly, on the first attempt.

   2. **Nothing in `src/` calls `infer()` on any runtime.** `grep -rn "\\.infer("
      src/archivetrust/` matches only comments. The one code path that would have reached it —
      `composition.py::activate_configured_vision_provider` wrapping a runtime in a VLM adapter — looks
      the adapter up in `_VISION_PROVIDER_ADAPTER_FACTORIES`, which is an **empty dict** since Stage 5
      deleted all three VLM adapters. `warm_up()` has no caller in `src/` either, outside `infer()`
      itself. So `real_torch_and_transformers_facade`'s `generate` — where every genuinely
      Qwen-flavoured line lives — is unreachable from application code today.

   3. **Two of the three flagged "quirks" are comments, not behaviour.** Corrected for the record
      rather than repeated:

      * *"attention-implementation quirks"*: `from_pretrained` is called with **no**
        `attn_implementation` argument. The `sdpa` experiment was tried and reverted; only the comment
        recording that remains. `grep -rn attn_implementation src/` matches two comment lines and no
        code.
      * *"`Qwen2_5_VLForConditionalGeneration` class-name matching"*: no Qwen class name is matched
        anywhere. `_VISION_MODEL_CLASS_NAMES` holds two *generic* Auto class names
        (`AutoModelForImageTextToText`, `AutoModelForVision2Seq`) and each is **tried against the
        model** rather than matched by name. The Qwen class name appears only inside a comment
        explaining a `load_in_8bit` kwarg incompatibility.
      * *"a decode pattern tuned to Qwen's chat template"*: **this one is real.** `generate()` applies
        `processor.apply_chat_template` and then decodes only the newly-generated token span. It is
        guarded (`if getattr(processor, "chat_template", None)`) with a documented plain-text
        fallback, so it is correct for a chat-templated VLM and degrades rather than breaks for one
        without a template — but it is genuinely written around that family's usage pattern, and
        `_VISION_MODEL_CLASS_NAMES`'s own docstring is written around one model's registration
        behaviour.

   **What was deliberately NOT done, and why.** This module was **not** rewritten to be "generic".
   Generalising a code path that no caller reaches would be speculative work against an unknown future
   consumer, guided by no failing test and no real second model — and it would touch the one runtime a
   future non-HTR VLM binding still depends on. The honest state is: this is working, tested code for
   a chat-templated vision-language model, currently with no consumer, whose Qwen-shaped assumptions
   are inert because nothing HTR-related routes through it.

   **The isolation is enforced, not merely asserted here.**
   `tests/runtime/test_transformers_runtime.py::test_no_htr_provider_depends_on_the_shared_transformers_runtime`
   and its two siblings fail if any `providers/` module ever imports this one, or if Florence-2 stops
   loading its own model. If a future phase does route an HTR method through here, that test breaks
   first and the assumptions above stop being inert at exactly the right moment — which is the
   guarantee the Stage 5 deferral was missing.
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
            # Decode only the newly generated tokens (the chat-templated-VLM usage pattern) -- the
            # full sequence includes the echoed prompt, which would corrupt any downstream structured
            # parsing of this text by prepending the system/user prompt to it.
            #
            # This comment used to cite `QwenPageResponse.model_validate_json` as the concrete
            # downstream parser. That class was deleted with `providers/qwen_vl/` in migration Stage 5
            # and the citation dangled (it named a symbol that exists nowhere in `src/`); corrected
            # 2026-07-30 to state the reason generically, since there is no downstream parser at all
            # while this runtime has no consumer -- see the module docstring's note.
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
