"""The `torch`/`transformers` seam between `SwedishLionAdapter` and a real TrOCR forward pass.

Same "narrow Protocol standing in for a real, hard-to-install dependency" discipline as
`providers/florence2_htr/facade.py`: tests inject a fake `SwedishLionWorkerFacade`-shaped object
(fast, no model download, no GPU); production code uses `real_swedish_lion_facade()` (slow first
call -- downloads + loads real weights, fast on every call after that since the loaded model is
cached in-process).

No subprocess/isolated-venv boundary is needed (unlike SATRN): `Riksarkivet/trocr-base-handwritten-
hist-swe-2` is a standard `VisionEncoderDecoderModel` checkpoint, loadable through this project's
existing main-venv `torch`/`transformers` install with no extra dependencies beyond what Florence-2
already required `transformers` to provide -- no `timm`, no `einops`, no `trust_remote_code`. This
was confirmed by reading the model card's own usage example directly (`TrOCRProcessor` +
`VisionEncoderDecoderModel`, both core `transformers` classes), not assumed from the "TrOCR" name
alone.
"""

from __future__ import annotations

import time
from typing import Any, Protocol


class SwedishLionWorkerResult(dict):
    """The result dict this module's facade returns -- kept as a plain dict (not a pydantic
    model) since this is an internal transport type between this module and `adapter.py`, which
    translates it into the public `RecognitionResult`/`Evidence`/`FailureRecord` types."""


class SwedishLionWorkerFacade(Protocol):
    """The exact surface `SwedishLionAdapter` needs from a Swedish Lion recognition run, isolated
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
        num_beams: int,
    ) -> SwedishLionWorkerResult: ...


def swedish_lion_dependencies_available() -> tuple[bool, str]:
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

    if missing:
        return False, (
            f"Missing dependencies for Swedish Lion: {', '.join(missing)}. Install the "
            "'transformers' extra (see providers/swedish_lion/README.md)."
        )
    return True, ", ".join(versions)


class _RealSwedishLionFacade:
    """Loads the real fine-tuned TrOCR checkpoint (+ base model's processor) via plain
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

        from transformers import TrOCRProcessor, VisionEncoderDecoderModel  # noqa: PLC0415

        model = VisionEncoderDecoderModel.from_pretrained(model_id, revision=model_revision).to(device)
        # The fine-tuned checkpoint does ship its own preprocessor/tokenizer files (verified via
        # the HF API: `preprocessor_config.json`, `tokenizer.json`, `tokenizer_config.json`,
        # `vocab.json`, `merges.txt` are all present in its repo siblings) -- unlike Florence-2's
        # fine-tuned checkpoint, it does not strictly need its base model's processor. The model
        # card's own documented usage example loads the processor from
        # `microsoft/trocr-base-handwritten` regardless, and this facade follows that documented
        # example exactly rather than substituting an assumption of equivalence.
        processor = TrOCRProcessor.from_pretrained(processor_model_id, revision=processor_revision)
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
        num_beams: int,
    ) -> SwedishLionWorkerResult:
        import torch  # noqa: PLC0415
        from PIL import Image, UnidentifiedImageError  # noqa: PLC0415

        if device_request == "cpu":
            device = "cpu"
        elif device_request == "cuda":
            device = "cuda"
        else:  # "auto"
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Model card requirement: "the image has to be a single text line". Not the versioned
        # `RgbNormalization` preprocessing stage (`htr/preprocessing/rgb_normalization.py`) -- this
        # is TrOCR's own processor input requirement (a 3-channel tensor), exactly the same
        # deliberate distinction Florence-2's facade documents (see
        # `providers/florence2_htr/facade.py`): unversioned, unrecorded, produces no artifact,
        # because nothing downstream reasons about it.
        try:
            image = Image.open(image_path).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            return SwedishLionWorkerResult(
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
            return SwedishLionWorkerResult(
                ok=False, category="model_load_failed", message=f"{type(exc).__name__}: {exc}"
            )

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        started = time.monotonic()
        try:
            pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
            with torch.no_grad():
                # `num_beams=1` matches the source pipeline's own documented configuration
                # (`htrflow-swedish-htr/pipeline.yaml`'s `WordLevelTrOCR` step in
                # `hypergeek-dev/rigsarkivet_hcr_test`, verified by reading it directly) -- not an
                # invented decoding configuration.
                generated_ids = model.generate(pixel_values, num_beams=num_beams)
        except torch.OutOfMemoryError as exc:
            return SwedishLionWorkerResult(ok=False, category="cuda_oom", message=str(exc))
        except Exception as exc:  # noqa: BLE001
            return SwedishLionWorkerResult(
                ok=False, category="unknown_error", message=f"{type(exc).__name__}: {exc}"
            )
        elapsed = time.monotonic() - started

        # Genuinely raw -- skip_special_tokens=False, mirrors Florence-2's raw/parsed separation
        # even though TrOCR's decoder has no separate structural-parse stage (see adapter.py's
        # module docstring for why this adapter, like SATRN's, only ever populates two payload
        # stages rather than three).
        raw_decoded = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        if not text or not text.strip():
            return SwedishLionWorkerResult(
                ok=False,
                category="empty_output",
                message="Swedish Lion produced no recognizable text for this input",
                raw_decoded=raw_decoded,
            )

        peak_gpu_memory_mb = None
        gpu_name = None
        if device == "cuda":
            peak_gpu_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            gpu_name = torch.cuda.get_device_name(0)

        import transformers as _transformers  # noqa: PLC0415

        return SwedishLionWorkerResult(
            ok=True,
            raw_decoded=raw_decoded,
            text=text,
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


def real_swedish_lion_facade() -> SwedishLionWorkerFacade:
    """Constructs the real, in-process facade. Not imported/constructed at module load time by
    `adapter.py` -- only when a caller actually wants to run real inference."""
    return _RealSwedishLionFacade()
