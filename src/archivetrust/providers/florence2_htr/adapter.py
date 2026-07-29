"""`Florence2Adapter` -- the second real `HtrMethodAdapter` implementation (docs/htr-migration-plan.md
Stage 7), wrapping the Florence-2-based Swedish historical HTR pipeline from
`github.com/hoanghapham/vlm-htr` (commit `ced3b30222770911dcb900c3a1f83a247100d1a3`, the `main`
HEAD read during this investigation).

**Interface investigation (recorded here, not just in a report -- same discipline as
`providers/satrn/adapter.py`'s module docstring):**

`hoanghapham/vlm-htr` is a master's-thesis repository (Uppsala University, in collaboration with
Riksarkivet) comparing a "traditional" YOLO+TrOCR pipeline against a "VLM pipeline": **Florence-2
fine-tuned for two separate task-token-prompted stages** -- text *line detection* (`<OD>` task
token, full page in, quantized `<loc_N>` bounding boxes out) and text *line recognition* (`<OCR>`
task token, a rectangular line-bbox crop in, plain text out). This is confirmed by reading the
repo's own training scripts directly (`scripts/train/finetune_florence_od.py`,
`scripts/train/finetune_florence_ocr.py`) and its dataset/inference helper module
(`src/vlm/data_processing/florence.py`'s `FlorenceTask` class and `predict()` function) -- not
inferred from the README's prose alone.

**Base model + fine-tuned checkpoint (the brief's required disclosure):** base model is
`microsoft/Florence-2-base-ft` (loaded via `AutoModelForCausalLM.from_pretrained(...,
trust_remote_code=True)`, `revision="refs/pr/6"`, resolved this session to commit
`e0b8f375661041228a6431c950adac1a5c539b98`). A **publicly readable, non-gated** fine-tuned OCR
checkpoint for exactly this line-recognition stage was found and confirmed loadable:
`nazounoryuu/florence_base__mixed__line_bbox__ocr` (base_model tag: `microsoft/Florence-2-base-ft`,
`pipeline_tag: image-to-text`, `license: mit`, `gated: false`), resolved this session to commit
`994f47e8a0e8d77cb2e11528665efd07a855c3af`. It was located via the vlm-htr repo's linked Gradio
demo Space (`huggingface.co/spaces/nazounoryuu/vlm-htr`), whose README names both this OCR
checkpoint and its companion line-detection checkpoint
(`nazounoryuu/florence_base__mixed__page__line_od`) as the exact models it loads -- not guessed
from the model-name convention alone, though the name does match
`finetune_florence_ocr.py`'s own `--model-name florence_base__mixed__line_bbox__ocr` example
verbatim, corroborating it. **This is the non-gated case**: no user-provided HF token was required,
real fine-tuned-checkpoint inference (not merely base-model inference) is what this adapter
actually runs.

**Why only the OCR (recognition) stage is implemented here, not line detection:**
docs/htr-domain-design.md §7 makes segmentation an independent stage
(`htr/segmentation/SegmentationAdapter`), explicitly *not* part of `HtrMethodAdapter` (whose
`recognize()` returns text, not regions). The companion line-detection checkpoint
(`nazounoryuu/florence_base__mixed__page__line_od`) is documented here as a real, confirmed-to-exist
extension point for a future `SegmentationAdapter` implementation, not implemented in this stage --
implementing it inside this file would blur exactly the boundary the domain design draws on
purpose. See "Divergence from SATRN" below for what *is* in scope.

**Why plain `transformers`, not SATRN's isolated-venv/subprocess trick:** per the task's own
instruction, dependencies were checked first rather than assumed. Florence-2 *does* load through
this project's existing main-venv `torch`/`transformers` install (`pyproject.toml`'s `transformers`
extra), once two narrow, real gaps were closed -- not architecturally incompatible library stacks
like SATRN's `mmocr`/`mmcv`/`mmdet` chain:
1. `transformers==5.13.1` (what was installed for this session) raised `AttributeError:
   'Florence2LanguageConfig' object has no attribute 'forced_bos_token_id'` while constructing
   Florence-2's remote `configuration_florence2.py` (a real, verified incompatibility between very
   recent `transformers` internals and Florence-2's community `trust_remote_code` modeling file,
   not a guess). Downgrading to `transformers==4.49.0` -- still inside this project's existing
   declared `pyproject.toml` range (`>=4.40,<6.0`), so no dependency-constraint change was needed --
   resolves it; verified by loading the real checkpoint end-to-end afterward.
2. Florence-2's remote modeling code additionally imports `timm` (its vision backbone) and
   `einops`, neither previously installed; both added with `pip install timm einops` (see
   `README.md`'s install section), no version pin required.

Both fixes are recorded as facts, not silently worked around: see `facade.py`'s module docstring.

**Divergence from SATRN's workflow (the brief's explicit "do not force-fit" instruction, answered
concretely):**
- SATRN is a purpose-built CTC/attention OCR decoder: one task, one output shape, one honestly
  small confidence scalar from its own `AttentionPostprocessor`. Florence-2 is a general
  task-token-prompted VLM; this adapter must construct a prompt (`<OCR>`), and the model's raw
  output is not directly usable text -- it requires `processor.post_process_generation(...)`, a
  genuine **structural parsing step** SATRN's adapter never needed. This is why
  `build_observation_payloads` below populates all **three** transcription payload stages
  (`RawTranscriptionPayload`, `ParsedTranscriptionPayload`, `NormalizedTranscriptionPayload`) as
  three genuinely different strings, where SATRN's adapter only ever populated two (raw ==
  pre-normalization text already, no parsing stage existed for it to populate a third from).
- SATRN's confidence is a scalar the model's own postprocessor emits directly. Florence-2 (a
  causal-LM-style generator, not a CTC/attention OCR head) exposes no such thing; this adapter
  derives a confidence **proxy** from HF `generate(..., output_scores=True,
  return_dict_in_generate=True)`'s `sequences_scores` (length-normalized beam-search log
  probability), converted to `(0, 1]` via `exp(...)` -- see `facade.py`'s
  `sequence_log_prob_to_confidence_proxy` docstring for why this is explicitly labeled a proxy, not
  raw/calibrated confidence, per the task's hard rule.
- `Evidence.processing_stage` is `VLM_INFERENCE` here (not `OCR`, which SATRN's Evidence uses) --
  an honest architectural distinction: Florence-2 is a vision-language model prompted with a task
  token, SATRN is a dedicated non-VLM OCR decoder.
- SATRN required a second, isolated Python environment (`.venv-satrn`) reached over a subprocess
  boundary because its real dependency chain (`mmocr`/`mmcv`/`mmdet`) is incompatible with this
  project's main venv. Florence-2 runs in-process, in the main venv, through the already-declared
  `transformers` extra -- once the two narrow gaps above were closed. No subprocess, no second
  venv, no `ARCHIVETRUST_*_PYTHON` environment variable.
- The *real* vlm-htr pipeline this adapter is modeled on is two-stage end-to-end (its own line
  detection, `<OD>`, then OCR, `<OCR>`) -- unlike SATRN, which never does its own segmentation.
  This adapter deliberately implements only the recognition half (see above); `get_capabilities()`
  therefore reports the same `line_level_supported=True, page_level_supported=False` shape as
  SATRN's for this method's `recognize()` boundary specifically, even though the underlying model
  *family* (unlike SATRN's) is architecturally capable of the page-level detection half too -- just
  not through this adapter's `recognize()`, which is text-recognition-only by Protocol design.
"""

from __future__ import annotations

from archivetrust.domain.evidence.models import Evidence, ProcessingStage
from archivetrust.domain.ontology.payloads.transcription import (
    NormalizedTranscriptionPayload,
    ParsedTranscriptionPayload,
    RawTranscriptionPayload,
)
from archivetrust.htr.experiment.models import FailureRecord
from archivetrust.providers.florence2_htr.facade import (
    Florence2WorkerFacade,
    real_florence2_facade,
    sequence_log_prob_to_confidence_proxy,
    florence2_dependencies_available,
)
from archivetrust.providers.htr_adapter import (
    EnvironmentValidation,
    HealthCheckResult,
    MethodCapabilities,
    MethodMetadata,
    RecognitionInput,
    RecognitionResult,
)

METHOD_ID = "florence2_htr"
METHOD_NAME = "Florence-2 (vlm-htr line OCR)"
VENDOR = "Uppsala University / Riksarkivet (hoanghapham/vlm-htr thesis project)"

DEFAULT_MODEL_ID = "nazounoryuu/florence_base__mixed__line_bbox__ocr"
"""The fine-tuned OCR checkpoint (non-gated, confirmed public) -- see module docstring."""
DEFAULT_MODEL_REVISION = "994f47e8a0e8d77cb2e11528665efd07a855c3af"
"""The exact commit this session downloaded and ran real inference against (resolved via
`huggingface_hub.HfApi().model_info(...).sha` -- not a floating "main")."""

DEFAULT_PROCESSOR_MODEL_ID = "microsoft/Florence-2-base-ft"
"""The fine-tuned checkpoint ships no processor files of its own (verified: its HF repo siblings
are only `config.json`/`generation_config.json`/`model.safetensors`/optimizer+scheduler state/
`metrics.json` -- no `preprocessor_config.json`, no custom `processing_florence2.py`); its
tokenizer/image-processor are loaded from its own declared base model instead (see `facade.py`)."""
DEFAULT_PROCESSOR_REVISION = "e0b8f375661041228a6431c950adac1a5c539b98"
"""Resolved commit for `microsoft/Florence-2-base-ft@refs/pr/6` -- the exact revision the training
scripts in `hoanghapham/vlm-htr` pin (`REMOTE_MODEL_PATH`/`revision='refs/pr/6'`, verified by
reading `scripts/train/finetune_florence_ocr.py` directly), resolved to a real commit sha this
session rather than left as the floating `refs/pr/6` ref."""

VLM_HTR_REPO_REVISION = "ced3b30222770911dcb900c3a1f83a247100d1a3"
"""`github.com/hoanghapham/vlm-htr`'s `main` branch HEAD at the time of this investigation --
the source repo this adapter's task-token/decoding-parameter choices are read from (see module
docstring); this adapter vendors no code from it, it is a citation of provenance only."""

ADAPTER_VERSION = "1.0.0"
"""This adapter's own version -- distinct from `DEFAULT_MODEL_REVISION` (the pinned fine-tuned
checkpoint), `DEFAULT_PROCESSOR_REVISION` (the pinned base-model processor), and
`VLM_HTR_REPO_REVISION` (the source pipeline this adapter's design is read from). Bumped whenever
this file's request/response mapping changes in a way that could affect stored Evidence shape."""

TASK_PROMPT = "<OCR>"
"""Florence-2's built-in plain-text OCR task token (`FlorenceTask.OCR` in
`src/vlm/data_processing/florence.py`) -- not `<OCR_WITH_REGION>` (which additionally emits
per-character/word bounding boxes this adapter's `recognize()` contract has no field for;
`geometry_supported=False` reflects this)."""

KNOWN_FAILURE_CATEGORIES = (
    "model_load_failed",
    "cuda_oom",
    "malformed_input",
    "malformed_output",
    "empty_output",
    "unknown_error",
)
"""Codepath categories `facade.py`'s real facade can report, each mapped to a `FailureRecord.category`
-- kept as a module-level tuple (not an Enum), same rationale as SATRN's: a category this module
adds later doesn't require a lockstep enum change, and an unrecognized category still produces a
valid `FailureRecord` (see `build_failure_record`)."""


def normalize_transcription(parsed_text: str) -> str:
    """The only normalization this adapter performs -- Unicode NFC + outer whitespace stripping,
    applied to the already-*parsed* text (never to the raw special-token-laden decoder output).
    Deliberately nothing lexical (Constitution: never silently normalize) -- mirrors
    `providers/satrn/adapter.py::normalize_transcription` exactly, applied one stage later in this
    adapter's pipeline (after parsing) rather than directly on `recognize()`'s raw output, because
    Florence-2's raw output is not usable text at all until parsed (see module docstring).
    """
    import unicodedata

    return unicodedata.normalize("NFC", parsed_text).strip()


class Florence2Adapter:
    """Implements `archivetrust.providers.htr_adapter.HtrMethodAdapter` (verified via
    `isinstance(Florence2Adapter(), HtrMethodAdapter)` in the contract test)."""

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        model_revision: str | None = DEFAULT_MODEL_REVISION,
        processor_model_id: str = DEFAULT_PROCESSOR_MODEL_ID,
        processor_revision: str | None = DEFAULT_PROCESSOR_REVISION,
        task_prompt: str = TASK_PROMPT,
        device_request: str = "auto",
        facade: Florence2WorkerFacade | None = None,
    ) -> None:
        self._model_id = model_id
        self._model_revision = model_revision
        self._processor_model_id = processor_model_id
        self._processor_revision = processor_revision
        self._task_prompt = task_prompt
        self._device_request = device_request
        self._facade = facade if facade is not None else real_florence2_facade()

    def get_metadata(self) -> MethodMetadata:
        return MethodMetadata(
            method_id=METHOD_ID,
            method_name=METHOD_NAME,
            vendor=VENDOR,
            # The fine-tuned checkpoint's identity is the most meaningful single revision to
            # report here (MethodMetadata carries one revision field) -- the base
            # model/processor's own pinned revision is separately recorded on every Evidence's
            # `supporting_metadata` (see `build_evidence`), never dropped, just not duplicated
            # into this single-string field.
            model_revision=f"{self._model_id}@{self._model_revision or 'unpinned'}",
        )

    def get_capabilities(self) -> MethodCapabilities:
        return MethodCapabilities(
            # A real, measured proxy (exp of HF beam-search's length-normalized sequence
            # log-probability) -- not fabricated, not per-character/token granularity, not a
            # calibrated probability. See facade.py's sequence_log_prob_to_confidence_proxy.
            confidence_supported=True,
            geometry_supported=False,
            line_level_supported=True,
            page_level_supported=False,
            local_execution_supported=True,
            external_upload_required=False,
        )

    def validate_environment(self) -> EnvironmentValidation:
        available, message = florence2_dependencies_available()
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
        task_prompt = input.configuration.get("task_prompt", self._task_prompt)

        result = self._facade.run_inference(
            image_path=image_path,
            device_request=configured_device,
            model_id=self._model_id,
            model_revision=self._model_revision,
            processor_model_id=self._processor_model_id,
            processor_revision=self._processor_revision,
            task_prompt=task_prompt,
        )

        if not result.get("ok"):
            return RecognitionResult(text=None, raw_response=dict(result))

        confidence = sequence_log_prob_to_confidence_proxy(result.get("sequence_log_prob"))

        return RecognitionResult(
            text=result["parsed_text"],
            confidence=confidence,
            raw_response=dict(result),
            execution_time_ms=result.get("elapsed_seconds", 0.0) * 1000.0 if result.get("elapsed_seconds") else None,
            model_revision=result.get("model_revision") or self._model_revision,
        )

    def health_check(self) -> HealthCheckResult:
        """Cheap liveness probe -- import-only dependency check, never loads real weights or runs
        inference (that cost belongs to an actual `recognize()` call)."""
        available, message = florence2_dependencies_available()
        return HealthCheckResult(healthy=available, message=message)


def build_evidence(result: RecognitionResult, *, adapter_version: str = ADAPTER_VERSION) -> Evidence | None:
    """Populates `Evidence`'s HTR extension fields (docs/htr-domain-design.md §2) from a
    `RecognitionResult`. `raw_output` is the **genuinely raw** decoder text (special tokens
    intact, e.g. `"</s><s>...text...</s>"`), never `result.text` (which is already Florence-2's
    *parsed* transcription -- see module docstring's "Divergence from SATRN"). Returns `None` for
    a failed recognition -- callers should use `build_failure_record` for that case instead."""
    if result.text is None:
        return None

    raw_response = result.raw_response
    raw_decoded = raw_response.get("raw_decoded", result.text)
    software_environment = raw_response.get("software_environment")
    hardware_environment = {"gpu_name": raw_response["gpu_name"]} if raw_response.get("gpu_name") else None

    return Evidence.create(
        provider=METHOD_ID,
        provider_version=adapter_version,
        raw_output=raw_decoded,
        processing_stage=ProcessingStage.VLM_INFERENCE,
        provider_confidence=result.confidence,
        prompt=TASK_PROMPT,
        model_revision=result.model_revision,
        execution_device=raw_response.get("device_used"),
        execution_time_ms=result.execution_time_ms,
        gpu_memory_mb=raw_response.get("peak_gpu_memory_mb"),
        software_environment=software_environment,
        hardware_environment=hardware_environment,
        supporting_metadata={
            "processor_model_id": DEFAULT_PROCESSOR_MODEL_ID,
            "processor_revision": DEFAULT_PROCESSOR_REVISION,
            "vlm_htr_repo_revision": VLM_HTR_REPO_REVISION,
            "sequence_log_prob": raw_response.get("sequence_log_prob"),
        },
    )


def build_observation_payloads(
    result: RecognitionResult,
) -> tuple[RawTranscriptionPayload, ParsedTranscriptionPayload, NormalizedTranscriptionPayload] | None:
    """Builds all **three** transcription payload stages as genuinely separate objects
    (Constitution: never silently normalize/overwrite) -- `None` for a failed recognition.

    Unlike `providers/satrn/adapter.py::build_observation_payloads` (which only ever builds two:
    SATRN's raw output is already plain text, no parsing stage exists for it to populate a third
    from), Florence-2 genuinely has a distinct structural-parsing stage
    (`processor.post_process_generation`), so this adapter is the first to exercise
    `ParsedTranscriptionPayload` for real."""
    if result.text is None:
        return None
    raw_response = result.raw_response
    raw_text = raw_response.get("raw_decoded", result.text)
    parsed_text = result.text  # already Florence-2's post_process_generation plain-text output
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
    message = raw_response.get("message", "Florence-2 recognition failed with no message")
    # `category` is stored verbatim even if not in KNOWN_FAILURE_CATEGORIES (Constitution
    # Article 6: Full Exposure) -- a category this module adds later must never be silently
    # dropped or coerced, it just won't have been anticipated by name in this file yet.
    return FailureRecord.create(method_run_id=method_run_id, reason=message, category=category)
