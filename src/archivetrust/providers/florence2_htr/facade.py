"""The `torch`/`transformers` seam between `Florence2Adapter` and a real Florence-2 forward pass.

Unlike SATRN (`providers/satrn/facade.py`), no subprocess/isolated-venv boundary is needed here:
investigation confirmed Florence-2 loads through plain `transformers` (`AutoModelForCausalLM` +
`AutoProcessor`, `trust_remote_code=True`) using this project's *existing* main-venv `torch`/
`transformers` install -- once two real, narrow gaps were closed (see `adapter.py`'s module
docstring for the full investigation): (1) `transformers==5.13.1` (the version installed for this
session) raised `AttributeError: 'Florence2LanguageConfig' object has no attribute
'forced_bos_token_id'` while constructing Florence-2's remote `configuration_florence2.py` --
pinned down to `transformers==4.49.0` (still inside this project's declared `>=4.40,<6.0` range in
`pyproject.toml`, so no dependency-constraint change was needed, only a same-range downgrade of
what was actually installed); (2) Florence-2's remote modeling code additionally requires `timm`
(vision backbone) and `einops`, neither previously installed -- added, no version pin needed (pure
import-time dependencies of the model's own `trust_remote_code` module, not of this adapter).

Follows the same "narrow Protocol standing in for a real, hard-to-install dependency" discipline
`runtime/transformers_runtime.py::TorchAndTransformersFacade` and `providers/satrn/facade.py` both
use: tests inject a fake `Florence2WorkerFacade`-shaped object (fast, no model download, no GPU);
production code uses `real_florence2_facade()` (slow first call -- downloads + loads real weights,
fast on every call after that since the loaded model is cached in-process, unlike SATRN's
per-call subprocess restart).
"""

from __future__ import annotations

import math
import time
from typing import Any, Protocol


class Florence2WorkerResult(dict):
    """The result dict this module's facade returns -- kept as a plain dict (not a pydantic
    model) since this is an internal transport type between this module and `adapter.py`, which
    translates it into the public `RecognitionResult`/`Evidence`/`FailureRecord` types."""


class Florence2WorkerFacade(Protocol):
    """The exact surface `Florence2Adapter` needs from a Florence-2 recognition run, isolated
    behind one Protocol so it can be the real in-process model (production) or a fake (tests)."""

    def run_inference(
        self,
        *,
        image_path: str,
        device_request: str,
        model_id: str,
        model_revision: str | None,
        processor_model_id: str,
        processor_revision: str | None,
        task_prompt: str,
    ) -> Florence2WorkerResult: ...


def florence2_dependencies_available() -> tuple[bool, str]:
    """Cheap import-only check (no model download, no GPU allocation) -- used by
    `validate_environment()`/`health_check()`, which must not pay the multi-second-to-minutes
    weight-download/load cost just to answer "is this method usable at all"."""
    missing: list[str] = []
    versions: list[str] = []
    try:
        import torch  # noqa: PLC0415

        versions.append(f"torch=={torch.__version__} (cuda_available={torch.cuda.is_available()})")
    except ModuleNotFoundError:
        missing.append("torch")
    try:
        import transformers  # noqa: PLC0415

        versions.append(f"transformers=={transformers.__version__}")
    except ModuleNotFoundError:
        missing.append("transformers")
    try:
        import timm  # noqa: F401,PLC0415

        versions.append("timm importable")
    except ModuleNotFoundError:
        missing.append("timm")
    try:
        import einops  # noqa: F401,PLC0415

        versions.append("einops importable")
    except ModuleNotFoundError:
        missing.append("einops")

    if missing:
        return False, (
            f"Missing dependencies for Florence-2: {', '.join(missing)}. Install the "
            "'transformers' extra plus `timm`/`einops` (see providers/florence2_htr/README.md)."
        )
    return True, ", ".join(versions)


class _RealFlorence2Facade:
    """Loads the real fine-tuned Florence-2 checkpoint (+ base model's processor) via plain
    `transformers`, caching the loaded (model, processor) pair in-process per
    `(model_id, model_revision, processor_model_id, processor_revision, device)` so repeated
    `run_inference` calls in one process don't re-download/re-load weights."""

    def __init__(self) -> None:
        self._loaded: dict[tuple, Any] = {}

    def _load(
        self,
        *,
        model_id: str,
        model_revision: str | None,
        processor_model_id: str,
        processor_revision: str | None,
        device: str,
    ):
        key = (model_id, model_revision, processor_model_id, processor_revision, device)
        if key in self._loaded:
            return self._loaded[key]

        from transformers import AutoModelForCausalLM, AutoProcessor  # noqa: PLC0415

        model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=model_revision, trust_remote_code=True
        ).to(device)
        # The fine-tuned OCR checkpoint (`nazounoryuu/florence_base__mixed__line_bbox__ocr`) ships
        # only weights/config -- no `preprocessor_config.json`/`processing_florence2.py` (verified
        # by listing its HF repo siblings). Its tokenizer/image-processor are byte-identical to its
        # own `base_model` (`microsoft/Florence-2-base-ft`, per the fine-tuned repo's model card
        # metadata) since fine-tuning only touched the weights, not the vocabulary/preprocessing --
        # so the processor is loaded from the base model repo, pinned to its own resolved revision.
        processor = AutoProcessor.from_pretrained(
            processor_model_id, revision=processor_revision, trust_remote_code=True
        )
        self._loaded[key] = (model, processor)
        return self._loaded[key]

    def run_inference(
        self,
        *,
        image_path: str,
        device_request: str,
        model_id: str,
        model_revision: str | None,
        processor_model_id: str,
        processor_revision: str | None,
        task_prompt: str,
    ) -> Florence2WorkerResult:
        import torch  # noqa: PLC0415
        from PIL import Image, UnidentifiedImageError  # noqa: PLC0415

        if device_request == "cpu":
            device = "cpu"
        elif device_request == "cuda":
            device = "cuda"
        else:  # "auto"
            device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            image = Image.open(image_path).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            return Florence2WorkerResult(
                ok=False, category="malformed_input", message=f"{type(exc).__name__}: {exc}"
            )

        try:
            model, processor = self._load(
                model_id=model_id,
                model_revision=model_revision,
                processor_model_id=processor_model_id,
                processor_revision=processor_revision,
                device=device,
            )
        except Exception as exc:  # noqa: BLE001 -- any load failure is a real, reportable failure
            return Florence2WorkerResult(
                ok=False, category="model_load_failed", message=f"{type(exc).__name__}: {exc}"
            )

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        started = time.monotonic()
        try:
            inputs = processor(text=[task_prompt], images=[image], return_tensors="pt").to(device)
            with torch.no_grad():
                # `num_beams=3, do_sample=False` matches the source repo's own `predict()` helper
                # (`src/vlm/data_processing/florence.py`, verified by reading it directly) exactly
                # -- not an invented decoding configuration. `output_scores`/
                # `return_dict_in_generate` are this adapter's own addition (the source repo's
                # `predict()` does not request them) so `sequences_scores` -- a real, length-
                # normalized beam log-probability HF's beam search already computes internally --
                # becomes available as the honest confidence-proxy signal (see `adapter.py`).
                generated = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    do_sample=False,
                    num_beams=3,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
        except torch.OutOfMemoryError as exc:
            return Florence2WorkerResult(ok=False, category="cuda_oom", message=str(exc))
        except Exception as exc:  # noqa: BLE001
            return Florence2WorkerResult(
                ok=False, category="unknown_error", message=f"{type(exc).__name__}: {exc}"
            )
        elapsed = time.monotonic() - started

        # Genuinely raw -- skip_special_tokens=False, no parsing at all. Never overwritten.
        raw_decoded = processor.batch_decode(generated.sequences, skip_special_tokens=False)[0]

        try:
            parsed = processor.post_process_generation(
                raw_decoded, task=task_prompt, image_size=image.size
            )
        except Exception as exc:  # noqa: BLE001 -- a genuine structural-parse failure, not a crash
            return Florence2WorkerResult(
                ok=False,
                category="malformed_output",
                message=f"post_process_generation failed: {type(exc).__name__}: {exc}",
                raw_decoded=raw_decoded,
            )

        parsed_text = parsed.get(task_prompt) if isinstance(parsed, dict) else None
        if not parsed_text or not parsed_text.strip():
            return Florence2WorkerResult(
                ok=False,
                category="empty_output",
                message="Florence-2 produced no recognizable text for this input",
                raw_decoded=raw_decoded,
                parsed=parsed,
            )

        sequence_log_prob = None
        if getattr(generated, "sequences_scores", None) is not None:
            sequence_log_prob = float(generated.sequences_scores[0].item())

        peak_gpu_memory_mb = None
        gpu_name = None
        if device == "cuda":
            peak_gpu_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            gpu_name = torch.cuda.get_device_name(0)

        import transformers as _transformers  # noqa: PLC0415

        return Florence2WorkerResult(
            ok=True,
            raw_decoded=raw_decoded,
            parsed=parsed,
            parsed_text=parsed_text,
            sequence_log_prob=sequence_log_prob,
            elapsed_seconds=elapsed,
            device_used=device,
            peak_gpu_memory_mb=peak_gpu_memory_mb,
            gpu_name=gpu_name,
            software_environment={
                "torch": torch.__version__,
                "transformers": _transformers.__version__,
            },
            model_revision=model_revision,
        )


def real_florence2_facade() -> Florence2WorkerFacade:
    """Constructs the real, in-process facade. Not imported/constructed at module load time by
    `adapter.py` -- only when a caller actually wants to run real inference (mirrors
    `transformers_runtime.py::real_torch_and_transformers_facade`'s lazy-construction discipline).
    """
    return _RealFlorence2Facade()


def sequence_log_prob_to_confidence_proxy(sequence_log_prob: float | None) -> float | None:
    """Converts HF beam-search's `sequences_scores` (a length-normalized sum of per-token log
    probabilities, always <= 0) into a `(0, 1]` scalar via `exp(...)`. This is explicitly a
    **proxy**, not a calibrated or per-character confidence: it is the geometric mean of the
    per-token probabilities the beam search actually selected, exactly analogous to SATRN's mean
    decode probability (`providers/satrn/adapter.py`'s module docstring), but derived from a
    causal-LM's token-generation log-probabilities rather than an attention-based OCR decoder's
    own postprocessor. Never labeled "confidence" alone anywhere downstream without this
    docstring's caveat traveling with it (`MethodCapabilities`/README both say so explicitly).
    """
    if sequence_log_prob is None:
        return None
    return min(1.0, max(0.0, math.exp(sequence_log_prob)))
