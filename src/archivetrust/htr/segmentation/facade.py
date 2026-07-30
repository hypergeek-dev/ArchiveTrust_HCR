"""The `torch`/`transformers` seam between the line-detection adapter and a real Florence-2 `<OD>`
forward pass.

Same discipline, and deliberately the same shape, as
`providers/florence2_htr/facade.py::Florence2WorkerFacade`: a narrow Protocol standing in for a
real, slow, hard-to-install dependency, so tests inject a fake (fast, no download, no GPU) and
production uses `real_florence2_line_detector_facade()`. That module's investigation applies here
unchanged -- this is the *same model family*, loaded the same way (`AutoModelForCausalLM` +
`AutoProcessor`, `trust_remote_code=True`), under the same two verified constraints
(`transformers<5.0`, plus `timm`/`einops`); only the checkpoint and the task token differ.

**Two intentional divergences from `Florence2WorkerFacade`, both forced by what detection is:**

1. **It takes a decoded image, not an `image_path`.** Recognition reads one crop that exists as a
   file; detection reads a *window of a page* (one side of a spread) that has no file of its own.
   Requiring a path would mean writing every detector window to disk purely to satisfy a signature.
2. **It returns boxes, not text.** `post_process_generation(task="<OD>")` yields
   `{"bboxes": [[x0, y0, x1, y1], ...], "labels": [...]}` already rescaled to the passed
   `image_size`, so no `<loc_N>` de-quantization is reimplemented here.

**Box precision is `COARSE_ESTIMATE`, not `PIXEL_ACCURATE`.** Florence-2 emits geometry as
`<loc_N>` tokens quantized to 1000 bins per axis, then rescales; on a 3816 px-wide page one bin is
~3.8 px, and the box never had sub-bin precision to begin with. The adapter records
`Precision.COARSE_ESTIMATE` for exactly this reason -- `domain/evidence/models.py::Precision` exists
so a quantized box is never compared as though it were pixel-accurate.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:  # pragma: no cover - typing only
    from PIL.Image import Image as PillowImage


class LineDetectionWorkerResult(dict):
    """Internal transport dict between this module and the adapter, kept a plain dict for the same
    reason `Florence2WorkerResult` is: the adapter is what translates it into public domain types."""


class Florence2LineDetectorFacade(Protocol):
    """The exact surface the line-detection adapter needs from one real `<OD>` forward pass."""

    def detect_lines(
        self,
        *,
        image: "PillowImage",
        device_request: str,
        model_id: str,
        model_revision: str | None,
        processor_model_id: str,
        processor_revision: str | None,
        task_prompt: str,
        max_new_tokens: int,
    ) -> LineDetectionWorkerResult: ...


class _RealFlorence2LineDetectorFacade:
    """Loads the real fine-tuned line-detection checkpoint (+ the base model's processor) via plain
    `transformers`, caching the loaded pair in-process per
    `(model_id, model_revision, processor_model_id, processor_revision, device)`.

    The cache is what makes the cold/warm timing distinction the screening reports real rather than
    asserted: the first `detect_lines` call pays download+load, every later one does not, and
    `model_load_seconds` is reported as `0.0` on a cache hit rather than omitted.
    """

    def __init__(self) -> None:
        self._loaded: dict[tuple, Any] = {}

    def _resolve_device(self, device_request: str) -> str:
        import torch  # noqa: PLC0415

        if device_request in ("cpu", "cuda"):
            return device_request
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _load(
        self,
        *,
        model_id: str,
        model_revision: str | None,
        processor_model_id: str,
        processor_revision: str | None,
        device: str,
    ) -> tuple[Any, Any, float]:
        key = (model_id, model_revision, processor_model_id, processor_revision, device)
        if key in self._loaded:
            model, processor = self._loaded[key]
            return model, processor, 0.0

        from transformers import AutoModelForCausalLM, AutoProcessor  # noqa: PLC0415

        started = time.monotonic()
        model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=model_revision, trust_remote_code=True
        ).to(device)
        # The fine-tuned detection checkpoint ships only weights/config (verified by listing its HF
        # repo siblings: config.json, generation_config.json, model.safetensors, optimizer/scheduler
        # state, metrics.json -- no preprocessor_config.json, no processing_florence2.py), exactly
        # like its companion OCR checkpoint. Its processor therefore comes from the declared base
        # model, pinned to its own resolved revision -- the identical arrangement, and the identical
        # reasoning, as providers/florence2_htr/facade.py.
        processor = AutoProcessor.from_pretrained(
            processor_model_id, revision=processor_revision, trust_remote_code=True
        )
        elapsed = time.monotonic() - started
        self._loaded[key] = (model, processor)
        return model, processor, elapsed

    def detect_lines(
        self,
        *,
        image: "PillowImage",
        device_request: str,
        model_id: str,
        model_revision: str | None,
        processor_model_id: str,
        processor_revision: str | None,
        task_prompt: str,
        max_new_tokens: int,
    ) -> LineDetectionWorkerResult:
        import torch  # noqa: PLC0415

        device = self._resolve_device(device_request)

        try:
            model, processor, load_seconds = self._load(
                model_id=model_id,
                model_revision=model_revision,
                processor_model_id=processor_model_id,
                processor_revision=processor_revision,
                device=device,
            )
        except Exception as exc:  # noqa: BLE001 -- any load failure is a real, reportable failure
            return LineDetectionWorkerResult(
                ok=False, category="model_load_failed", message=f"{type(exc).__name__}: {exc}"
            )

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        started = time.monotonic()
        try:
            inputs = processor(text=[task_prompt], images=[image], return_tensors="pt").to(device)
            with torch.no_grad():
                # `do_sample=False, num_beams=3` matches the decoding configuration
                # providers/florence2_htr/facade.py already established for this model family from
                # the source repo's own `predict()` helper -- not re-invented for detection.
                generated = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=3,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
        except torch.OutOfMemoryError as exc:
            return LineDetectionWorkerResult(ok=False, category="cuda_oom", message=str(exc))
        except Exception as exc:  # noqa: BLE001
            return LineDetectionWorkerResult(
                ok=False, category="unknown_error", message=f"{type(exc).__name__}: {exc}"
            )
        elapsed = time.monotonic() - started

        raw_decoded = processor.batch_decode(generated.sequences, skip_special_tokens=False)[0]

        # A generation that stopped because it ran out of budget rather than because the model
        # emitted its stop token has silently dropped however many lines remained. Detected here by
        # the real token count, not guessed from the text.
        generated_token_count = int(generated.sequences.shape[-1]) - int(inputs["input_ids"].shape[-1])
        output_truncated = generated_token_count >= max_new_tokens

        try:
            parsed = processor.post_process_generation(
                raw_decoded, task=task_prompt, image_size=image.size
            )
        except Exception as exc:  # noqa: BLE001 -- a genuine structural-parse failure, not a crash
            return LineDetectionWorkerResult(
                ok=False,
                category="malformed_output",
                message=f"post_process_generation failed: {type(exc).__name__}: {exc}",
                raw_decoded=raw_decoded,
            )

        detection = parsed.get(task_prompt) if isinstance(parsed, dict) else None
        if not isinstance(detection, dict) or "bboxes" not in detection:
            return LineDetectionWorkerResult(
                ok=False,
                category="malformed_output",
                message=(
                    f"post_process_generation returned no 'bboxes' for task {task_prompt!r}; "
                    f"got {type(detection).__name__}"
                ),
                raw_decoded=raw_decoded,
            )

        peak_gpu_memory_mb = None
        gpu_name = None
        if device == "cuda":
            peak_gpu_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            gpu_name = torch.cuda.get_device_name(0)

        import transformers as _transformers  # noqa: PLC0415

        return LineDetectionWorkerResult(
            ok=True,
            boxes=tuple(tuple(float(v) for v in box) for box in detection.get("bboxes", ())),
            labels=tuple(detection.get("labels", ())),
            raw_decoded=raw_decoded,
            output_truncated=output_truncated,
            generated_token_count=generated_token_count,
            elapsed_seconds=elapsed,
            model_load_seconds=load_seconds,
            device_used=device,
            peak_gpu_memory_mb=peak_gpu_memory_mb,
            gpu_name=gpu_name,
            software_environment={
                "torch": torch.__version__,
                "transformers": _transformers.__version__,
            },
            model_revision=model_revision,
        )


def real_florence2_line_detector_facade() -> Florence2LineDetectorFacade:
    """Constructs the real, in-process facade. Never constructed at import time -- only when a
    caller actually wants real detection (same lazy-construction discipline as
    `real_florence2_facade()`)."""
    return _RealFlorence2LineDetectorFacade()
