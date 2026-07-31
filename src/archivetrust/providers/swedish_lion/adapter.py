"""`SwedishLionAdapter` -- the third real `HtrMethodAdapter` implementation, wrapping
`Riksarkivet/trocr-base-handwritten-hist-swe-2` ("Swedish Lion Libre", Swedish National Archives).

**Provenance and the correction that led to this adapter:** earlier in this project's technical
reliability screening benchmark, "Transkribus Swedish Lion I" was assumed to be a cloud-only,
external-upload method (`MethodCapabilities.external_upload_required=True`, no local execution
path) -- reflected in `docs/methods/transkribus-swedish-lion-1.md` and the `transkribus` provider
package. That assumption was wrong for the specific model examined here: the user's own reference
repository (`github.com/hypergeek-dev/rigsarkivet_hcr_test`) runs a real, local, downloadable
HTRFlow-style pipeline whose recognition step is `Riksarkivet/trocr-base-handwritten-hist-swe-2` --
confirmed by reading its `htrflow-swedish-htr/pipeline.yaml` directly (`WordLevelTrOCR` recognition
step, `num_beams: 1`). This model's own Hugging Face card names it "**Swedish Lion Libre**": "An HTR
model for historical Swedish developed by the Swedish National Archives" -- a TrOCR-base-handwritten
architecture fine-tuned from Microsoft's base checkpoint on eleven public archive datasets. No
Transkribus/READ-COOP reference appears anywhere on the model card itself; the naming similarity to
Transkribus's own "Swedish Lion I" branding is real but its lineage is not confirmed identical, so
this adapter is registered under its own `method_id` (`swedish_lion`), never merged into or
presented as the `transkribus` provider. The existing `transkribus` package and its cloud-upload
assumption are unmodified by this adapter -- both remain true statements about the distinct thing
each actually is.

**Why this is architecturally closer to SATRN than to Florence-2:** confirmed by reading
`Riksarkivet/trocr-base-handwritten-hist-swe-2`'s model card and HTRflow's own `TrOCR`/
`WordLevelTrOCR` model classes (`AI-Riksarkivet/htrflow`, `src/htrflow/models/huggingface/trocr.py`)
directly, not assumed from the "TrOCR" name:
- Input granularity is **line-level**, identical to SATRN and Florence-2: the model card states
  plainly "the image has to be a single text line". `WordLevelTrOCR` (the class HTRFlow's own
  pipeline actually instantiates) is a thin subclass of the base `TrOCR` class that additionally
  derives word-level bounding boxes from the model's attention weights -- it takes the exact same
  `list[Image]` line-crop input as the base class; this adapter's `recognize()` boundary only needs
  the base class's behavior (text out, no geometry -- see `get_capabilities()` below), so it talks
  to the plain `VisionEncoderDecoderModel`/`TrOCRProcessor` HF classes directly rather than vendoring
  HTRFlow's own `WordLevelTrOCR` wrapper or its `htrflow` package dependency chain.
- Decoder output has no task-token structural-parse stage like Florence-2's
  `post_process_generation` -- but `processor.batch_decode` still genuinely differs between
  `skip_special_tokens=False` (special tokens intact, e.g. `"<s> text </s>"`) and `=True` (clean
  text), a real two-step distinction confirmed by calling both against the facade, not assumed from
  SATRN's shape. This adapter therefore populates **three** transcription payload stages (raw,
  parsed, normalized) like Florence-2's, not SATRN's two -- SATRN's own decoder
  (`AttentionPostprocessor`) never exposes a special-token-laden string at all, so its adapter
  genuinely has only two stages to populate; this model does have a real "parsed" stage in between,
  so it is recorded rather than collapsed away.
- No confidence signal: `model.generate(...)` here is called without `output_scores=True`/
  `return_dict_in_generate=True` (unlike Florence-2's facade), because `VisionEncoderDecoderModel`'s
  `sequences_scores` under greedy/low-beam decoding is a real, available signal but this adapter
  does not yet request or expose it -- `MethodCapabilities.confidence_supported=False`, an honest
  `False` rather than a fabricated or silently-proxied value (Constitution Article 6). A future
  version may add it the same way Florence-2's facade does; until then, no confidence is invented.
- Same reuse as SATRN and Florence-2: because this model's input granularity is line-level and
  identical in kind (a single line crop, RGB), the byte-identical `InputCrop` files already saved by
  the completed reliability run (`docs/experiments/technical-reliability-screening/full-run/
  reliability-2026-07-31/crops/`) are directly reusable for a controlled comparison against this
  adapter -- no new segmentation is required.

**Why plain `transformers`, not a subprocess/isolated venv (unlike SATRN):** `VisionEncoderDecoderModel`
and `TrOCRProcessor` are both core `transformers` classes -- no `trust_remote_code`, no `timm`, no
`einops`, no incompatible pinned dependency chain like SATRN's `mmocr`/`mmcv`/`mmdet`. Loads through
this project's existing main-venv install exactly as Florence-2 does (see `facade.py`).
"""

from __future__ import annotations

from archivetrust.domain.evidence.models import Evidence, ProcessingStage
from archivetrust.domain.ontology.payloads.transcription import (
    NormalizedTranscriptionPayload,
    ParsedTranscriptionPayload,
    RawTranscriptionPayload,
)
from archivetrust.htr.experiment.models import FailureRecord
from archivetrust.providers.htr_adapter import (
    EnvironmentValidation,
    HealthCheckResult,
    MethodCapabilities,
    MethodMetadata,
    RecognitionInput,
    RecognitionResult,
)
from archivetrust.providers.swedish_lion.facade import (
    SwedishLionWorkerFacade,
    real_swedish_lion_facade,
    swedish_lion_dependencies_available,
)

METHOD_ID = "swedish_lion"
METHOD_NAME = "Swedish Lion Libre (Riksarkivet TrOCR)"
VENDOR = "Riksarkivet (Swedish National Archives)"

DEFAULT_MODEL_ID = "Riksarkivet/trocr-base-handwritten-hist-swe-2"
DEFAULT_MODEL_REVISION = "aa79fcb1850bf3155ebc442570d6c6bfc0ac8100"
"""Resolved commit for the fine-tuned checkpoint's `main` branch (via `huggingface_hub`'s model
API), read this session -- not a floating "main"."""

DEFAULT_PROCESSOR_MODEL_ID = "microsoft/trocr-base-handwritten"
"""The model card's own documented usage example loads the processor from this base model, not
from the fine-tuned repo itself, even though the fine-tuned repo does ship its own preprocessor/
tokenizer files -- see `facade.py`'s `_load` docstring for why this adapter follows the documented
example rather than substituting an assumption of equivalence."""
DEFAULT_PROCESSOR_REVISION = "eaacaf452b06415df8f10bb6fad3a4c11e609406"
"""Resolved commit for `microsoft/trocr-base-handwritten`'s `main` branch, read this session."""

DEFAULT_NUM_BEAMS = 1
"""Matches `hypergeek-dev/rigsarkivet_hcr_test`'s `htrflow-swedish-htr/pipeline.yaml` `WordLevelTrOCR`
step exactly (verified by reading it directly) -- not an invented decoding configuration."""

ADAPTER_VERSION = "1.0.0"
"""This adapter's own version -- distinct from `DEFAULT_MODEL_REVISION` (the pinned fine-tuned
checkpoint) and `DEFAULT_PROCESSOR_REVISION` (the pinned base-model processor). Bumped whenever
this file's request/response mapping changes in a way that could affect stored Evidence shape."""

KNOWN_FAILURE_CATEGORIES = (
    "model_load_failed",
    "cuda_oom",
    "malformed_input",
    "empty_output",
    "unknown_error",
)
"""Codepath categories `facade.py`'s real facade can report, each mapped to a `FailureRecord.category`
-- kept as a module-level tuple (not an Enum), same rationale as SATRN's and Florence-2's."""


def normalize_transcription(raw_text: str) -> str:
    """The only normalization this adapter performs -- Unicode NFC + outer whitespace stripping.
    Mirrors `providers/satrn/adapter.py::normalize_transcription` exactly (same rationale: nothing
    lexical, per the Constitution's "never silently normalize" rule)."""
    import unicodedata

    return unicodedata.normalize("NFC", raw_text).strip()


class SwedishLionAdapter:
    """Implements `archivetrust.providers.htr_adapter.HtrMethodAdapter` (verified via
    `isinstance(SwedishLionAdapter(), HtrMethodAdapter)` in the contract test)."""

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        model_revision: str | None = DEFAULT_MODEL_REVISION,
        processor_model_id: str = DEFAULT_PROCESSOR_MODEL_ID,
        processor_revision: str | None = DEFAULT_PROCESSOR_REVISION,
        num_beams: int = DEFAULT_NUM_BEAMS,
        device_request: str = "auto",
        facade: SwedishLionWorkerFacade | None = None,
    ) -> None:
        self._model_id = model_id
        self._model_revision = model_revision
        self._processor_model_id = processor_model_id
        self._processor_revision = processor_revision
        self._num_beams = num_beams
        self._device_request = device_request
        self._facade = facade if facade is not None else real_swedish_lion_facade()

    def get_metadata(self) -> MethodMetadata:
        return MethodMetadata(
            method_id=METHOD_ID,
            method_name=METHOD_NAME,
            vendor=VENDOR,
            model_revision=f"{self._model_id}@{self._model_revision or 'unpinned'}",
        )

    def get_capabilities(self) -> MethodCapabilities:
        return MethodCapabilities(
            # No confidence signal is requested/exposed yet -- see module docstring. An honest
            # False, not a fabricated value.
            confidence_supported=False,
            geometry_supported=False,
            line_level_supported=True,
            page_level_supported=False,
            local_execution_supported=True,
            external_upload_required=False,
        )

    def validate_environment(self) -> EnvironmentValidation:
        available, message = swedish_lion_dependencies_available()
        return EnvironmentValidation(valid=available, messages=(message,))

    def recognize(self, input: RecognitionInput) -> RecognitionResult:
        image_path = input.input_crop_id or input.page_image_ref
        if not image_path:
            return RecognitionResult(
                text=None,
                raw_response={
                    "ok": False,
                    "category": "malformed_input",
                    "message": "RecognitionInput carried neither input_crop_id nor page_image_ref",
                },
            )

        configured_device = input.configuration.get("device", self._device_request)
        num_beams = input.configuration.get("num_beams", self._num_beams)

        result = self._facade.run_inference(
            image_path=image_path,
            device_request=configured_device,
            model_id=self._model_id,
            model_revision=self._model_revision,
            processor_model_id=self._processor_model_id,
            processor_revision=self._processor_revision,
            num_beams=num_beams,
        )

        if not result.get("ok"):
            return RecognitionResult(text=None, raw_response=dict(result))

        return RecognitionResult(
            text=result["text"],
            confidence=None,
            raw_response=dict(result),
            execution_time_ms=result.get("elapsed_seconds", 0.0) * 1000.0 if result.get("elapsed_seconds") else None,
            model_revision=result.get("model_revision") or self._model_revision,
        )

    def health_check(self) -> HealthCheckResult:
        """Cheap liveness probe -- import-only dependency check, never loads real weights or runs
        inference (that cost belongs to an actual `recognize()` call)."""
        available, message = swedish_lion_dependencies_available()
        return HealthCheckResult(healthy=available, message=message)


def build_evidence(result: RecognitionResult, *, adapter_version: str = ADAPTER_VERSION) -> Evidence | None:
    """Populates `Evidence`'s HTR extension fields from a `RecognitionResult`. Returns `None` for a
    failed recognition -- callers should use `build_failure_record` for that case instead."""
    if result.text is None:
        return None

    raw_response = result.raw_response
    software_environment = raw_response.get("software_environment")
    hardware_environment = {"gpu_name": raw_response["gpu_name"]} if raw_response.get("gpu_name") else None

    return Evidence.create(
        provider=METHOD_ID,
        provider_version=adapter_version,
        raw_output=raw_response.get("raw_decoded", result.text),
        processing_stage=ProcessingStage.OCR,
        provider_confidence=result.confidence,
        model_revision=result.model_revision,
        execution_device=raw_response.get("device_used"),
        execution_time_ms=result.execution_time_ms,
        gpu_memory_mb=raw_response.get("peak_gpu_memory_mb"),
        software_environment=software_environment,
        hardware_environment=hardware_environment,
        supporting_metadata={
            "processor_model_id": DEFAULT_PROCESSOR_MODEL_ID,
            "processor_revision": DEFAULT_PROCESSOR_REVISION,
        },
    )


def build_observation_payloads(
    result: RecognitionResult,
) -> tuple[RawTranscriptionPayload, ParsedTranscriptionPayload, NormalizedTranscriptionPayload] | None:
    """Builds all **three** transcription payload stages as genuinely separate objects
    (Constitution: never silently normalize) -- `None` for a failed recognition. Mirrors
    `providers/florence2_htr/adapter.py`'s three stages, not SATRN's two -- see module docstring
    for why this model's decode genuinely has a raw/parsed distinction SATRN's does not."""
    if result.text is None:
        return None
    raw_response = result.raw_response
    raw_text = raw_response.get("raw_decoded", result.text)
    parsed_text = result.text  # already skip_special_tokens=True decoder output
    normalized_text = normalize_transcription(parsed_text)
    return (
        RawTranscriptionPayload(text=raw_text),
        ParsedTranscriptionPayload(text=parsed_text),
        NormalizedTranscriptionPayload(text=normalized_text),
    )


def build_failure_record(result: RecognitionResult, *, method_run_id: str) -> FailureRecord | None:
    """Builds a `FailureRecord` (htr/experiment/models.py) from a failed `RecognitionResult` --
    `None` when the result did not actually fail, so a caller can call this unconditionally after
    `recognize()` without branching first."""
    if result.text is not None:
        return None
    raw_response = result.raw_response
    category = raw_response.get("category")
    message = raw_response.get("message", "Swedish Lion recognition failed with no message")
    return FailureRecord.create(method_run_id=method_run_id, reason=message, category=category)
