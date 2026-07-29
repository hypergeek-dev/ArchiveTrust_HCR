"""`SatrnAdapter` -- the first real `HtrMethodAdapter` implementation (docs/htr-migration-plan.md
Stage 6), wrapping `Riksarkivet/satrn_htr` (Swedish National Archives SATRN HTR model).

**Interface investigation (recorded here per the Stage 6 task brief, not just in a report):**
`Riksarkivet/satrn_htr`'s Hugging Face repo declares `library_name: htrflow` and ships
`model.pth` + `config.py` (an OpenMMLab `mmengine` config), not a `transformers`-compatible
`config.json`/safetensors pair -- it is **not** loadable via `AutoModel.from_pretrained`.
Riksarkivet's own `htrflow` toolkit loads it through `mmocr.apis.TextRecInferencer`
(`htrflow.models.openmmlab.satrn.Satrn`, verified by reading that module's source directly). So
this adapter integrates through the model's actual supported inference path (OpenMMLab's
`mmocr`), not a from-scratch reimplementation of the SATRN architecture -- reimplementing a
12-layer transformer encoder + 6-layer transformer decoder from a bare state dict would risk
silently loading weights into slightly-wrong layer shapes and reporting a bogus "successful"
transcription, which Constitution Article 6 (Full Exposure) and Article 18 (failure must be a
recorded fact) both argue against.

**Why not the `htrflow` package itself:** its own PyPI wheel additionally pins
`nvidia-cu11-*`/`triton==2.0.0`/`ultralytics`/`imgaug` (things this SATRN-only adapter has no use
for) and does not even declare `mmocr`/`mmcv`/`mmdet`/`mmengine` as install-time dependencies at
all (they are an optional, separately-installed "openmmlabs" extra -- see
`htrflow.models.importer.all_models`). Only `htrflow.models.hf_utils.load_mmlabs`'s ~15 lines of
actual logic were needed; they are reimplemented directly in `_worker.py` instead.

**Why a subprocess, not an in-process import (the actually load-bearing decision):**
`mmocr`==1.0.1 requires `mmdet`<3.2, which requires `mmcv`<2.2 (verified by installing and reading
the version-assertion in `mmdet/__init__.py`), which pins to `numpy`<2 and ships prebuilt wheels
only for `torch`<=2.1. This project's main venv runs Python 3.13 / torch 2.13+cu130 / numpy 2.3
(`transformers` extra, already installed for a later Florence-2 adapter) -- installing the
OpenMMLab stack there would either fail outright (no compatible wheel) or force a downgrade that
breaks the ~979 other passing tests. A separate, isolated interpreter (`.venv-satrn`, Python
3.10 + torch 2.1.0+cu121 + mmengine/mmcv/mmdet/mmocr) runs the actual model; this adapter talks to
it over a subprocess boundary (`facade.py`). This was verified working end-to-end in this session
-- see `providers/satrn/README.md` for the exact measured output on the real fixture.

**Raw vs. normalized transcription (Constitution: never silently normalize):** `recognize()`
returns `RecognitionResult.text` as the model's completely unmodified output string. A *separate*
function, `normalize_transcription`, performs the only normalization this adapter defines
(Unicode NFC + leading/trailing whitespace stripping -- nothing lexical/spelling-related), and
`build_observation_payloads` stores the raw and normalized strings as two distinct payload
objects, never overwriting one with the other.

**Confidence honesty:** SATRN's decoder (`AttentionPostprocessor`) yields one scalar `scores`
value per line (verified from a real run: `0.6666...`) -- a mean per-token decode probability
from the attention head, not a per-character/token confidence array and not a calibrated
probability. `MethodCapabilities.confidence_supported=True` because a real, non-fabricated
confidence value is genuinely produced and returned in `RecognitionResult.confidence`; this
docstring and `README.md` both say plainly that it is a derived scalar, not per-character
granularity, so nothing downstream can mistake it for more than it is.
"""

from __future__ import annotations

from archivetrust.domain.evidence.models import Evidence, ProcessingStage
from archivetrust.domain.ontology.payloads.transcription import (
    NormalizedTranscriptionPayload,
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
from archivetrust.providers.satrn.facade import (
    SatrnWorkerFacade,
    SatrnWorkerTimeout,
    satrn_python_available,
    subprocess_satrn_facade,
)

METHOD_ID = "satrn"
METHOD_NAME = "SATRN (Riksarkivet)"
VENDOR = "Riksarkivet (Swedish National Archives)"
DEFAULT_MODEL_ID = "Riksarkivet/satrn_htr"
ADAPTER_VERSION = "1.0.0"
"""This adapter's own version -- distinct from `model_revision` (the pinned HF checkpoint) and
from `mmocr`/`mmcv`/`mmdet`/`torch` versions (the worker's software environment). Bumped whenever
this file's request/response mapping changes in a way that could affect stored Evidence shape."""

# Codepath categories `_worker.py` can report, each mapped to a `FailureRecord.category` --
# kept as a module-level tuple (not an Enum) so a category the worker adds later doesn't require
# a lockstep enum change here; unknown categories still produce a valid FailureRecord (see
# `build_failure_record`), just with whatever string the worker sent.
KNOWN_FAILURE_CATEGORIES = (
    "model_load_failed",
    "cuda_oom",
    "malformed_input",
    "empty_output",
    "timeout",
    "unknown_error",
)


def normalize_transcription(raw_text: str) -> str:
    """The only normalization this adapter performs -- Unicode NFC + outer whitespace stripping.
    Deliberately does nothing lexical (no spelling modernization, no punctuation repair): per the
    Constitution's "never silently normalize" rule, anything beyond whitespace/Unicode-form
    canonicalization would be an editorial claim about the text, not a formatting cleanup, and
    belongs in a `TranscriptionConvention`-governed downstream step, not baked into this adapter.
    """
    import unicodedata

    return unicodedata.normalize("NFC", raw_text).strip()


class SatrnAdapter:
    """Implements `archivetrust.providers.htr_adapter.HtrMethodAdapter` (verified via
    `isinstance(SatrnAdapter(), HtrMethodAdapter)` in the contract test -- the `Protocol` is
    `runtime_checkable`)."""

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        model_revision: str | None = None,
        device_request: str = "auto",
        facade: SatrnWorkerFacade | None = None,
    ) -> None:
        self._model_id = model_id
        self._model_revision = model_revision
        self._device_request = device_request
        self._facade = facade if facade is not None else subprocess_satrn_facade()

    def get_metadata(self) -> MethodMetadata:
        return MethodMetadata(
            method_id=METHOD_ID,
            method_name=METHOD_NAME,
            vendor=VENDOR,
            # The pinned commit this session actually downloaded and ran inference against
            # (Riksarkivet/satrn_htr@a40c7093232eaa47a83ce6469fc4abd033486bdc) -- used as the
            # default so `get_metadata()` never reports "main"/"latest" (Constitution: pinned,
            # reproducible provenance). A caller that constructs this adapter with an explicit
            # `model_revision=` overrides it.
            model_revision=self._model_revision or "a40c7093232eaa47a83ce6469fc4abd033486bdc",
        )

    def get_capabilities(self) -> MethodCapabilities:
        return MethodCapabilities(
            # A real, measured scalar confidence per line (mean decode probability from SATRN's
            # AttentionPostprocessor) -- not fabricated, not per-character (see module docstring).
            confidence_supported=True,
            geometry_supported=False,
            line_level_supported=True,
            page_level_supported=False,
            local_execution_supported=True,
            external_upload_required=False,
        )

    def validate_environment(self) -> EnvironmentValidation:
        messages: list[str] = []
        valid = True

        try:
            import torch  # noqa: PLC0415

            cuda_available = torch.cuda.is_available()
            messages.append(f"main-venv torch={torch.__version__}, cuda_available={cuda_available}")
        except ModuleNotFoundError:
            # Not actually required for this adapter (inference happens in the isolated venv),
            # but its absence is worth surfacing since it means no GPU-availability signal was
            # obtainable from the orchestrating process itself.
            messages.append("torch not importable in the orchestrating (main) venv -- informational only")

        satrn_python_ok, satrn_python_message = satrn_python_available()
        messages.append(satrn_python_message)
        if not satrn_python_ok:
            valid = False

        return EnvironmentValidation(valid=valid, messages=tuple(messages))

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

        try:
            result = self._facade.run_inference(
                image_path=image_path,
                device_request=configured_device,
                model_id=self._model_id,
                revision=self._model_revision,
            )
        except SatrnWorkerTimeout as exc:
            return RecognitionResult(
                text=None,
                raw_response={"ok": False, "category": "timeout", "message": str(exc)},
            )
        except FileNotFoundError as exc:
            # `_find_satrn_python`/`satrn_python_available` raised this -- the isolated venv is
            # missing entirely, distinct from a model-load failure inside a present venv.
            return RecognitionResult(
                text=None,
                raw_response={"ok": False, "category": "model_load_failed", "message": str(exc)},
            )

        if not result.get("ok"):
            return RecognitionResult(text=None, raw_response=dict(result))

        return RecognitionResult(
            text=result["text"],
            confidence=result.get("score"),
            raw_response=dict(result),
            execution_time_ms=result.get("elapsed_seconds", 0.0) * 1000.0 if result.get("elapsed_seconds") else None,
            model_revision=result.get("model_revision") or self._model_revision,
        )

    def health_check(self) -> HealthCheckResult:
        """Cheap liveness probe -- checks the isolated interpreter exists, never runs inference
        (that is `validate_environment`'s cost-tolerant cousin's job at pre-flight time; this is
        meant to be called often/cheaply)."""
        available, message = satrn_python_available()
        return HealthCheckResult(healthy=available, message=message)


def build_evidence(result: RecognitionResult, *, adapter_version: str = ADAPTER_VERSION) -> Evidence | None:
    """Populates `Evidence`'s HTR extension fields (docs/htr-domain-design.md §2) from a
    `RecognitionResult`. Returns `None` for a failed recognition (`text is None`) -- callers
    should use `build_failure_record` for that case instead; Evidence is only created for an
    actual (possibly low-confidence, possibly wrong) model output, never fabricated for a failure.
    """
    if result.text is None:
        return None

    raw_response = result.raw_response
    software_environment = raw_response.get("software_environment")
    hardware_environment = None
    if raw_response.get("gpu_name"):
        hardware_environment = {"gpu_name": raw_response["gpu_name"]}

    return Evidence.create(
        provider=METHOD_ID,
        provider_version=adapter_version,
        raw_output=result.text,
        processing_stage=ProcessingStage.OCR,
        provider_confidence=result.confidence,
        model_revision=result.model_revision,
        execution_device=raw_response.get("device_used"),
        execution_time_ms=result.execution_time_ms,
        gpu_memory_mb=raw_response.get("peak_gpu_memory_mb"),
        software_environment=software_environment,
        hardware_environment=hardware_environment,
        supporting_metadata={"config_revision": raw_response.get("config_revision")}
        if raw_response.get("config_revision")
        else None,
    )


def build_observation_payloads(
    result: RecognitionResult,
) -> tuple[RawTranscriptionPayload, NormalizedTranscriptionPayload] | None:
    """Builds the raw and normalized transcription payloads as two genuinely separate objects
    (Constitution: never silently normalize) -- `None` for a failed recognition."""
    if result.text is None:
        return None
    raw_payload = RawTranscriptionPayload(text=result.text)
    normalized_payload = NormalizedTranscriptionPayload(text=normalize_transcription(result.text))
    return raw_payload, normalized_payload


def build_failure_record(result: RecognitionResult, *, method_run_id: str) -> FailureRecord | None:
    """Builds a `FailureRecord` (htr/experiment/models.py) from a failed `RecognitionResult` --
    `None` when the result did not actually fail (`text is not None`), so a caller can call this
    unconditionally after `recognize()` without branching first."""
    if result.text is not None:
        return None
    raw_response = result.raw_response
    category = raw_response.get("category")
    message = raw_response.get("message", "SATRN recognition failed with no message")
    # `category` is stored verbatim even if it is not one of `KNOWN_FAILURE_CATEGORIES` -- a
    # category the worker adds later must never be silently dropped or coerced (Constitution
    # Article 6: Full Exposure), it just won't have been anticipated by name in this file yet.
    return FailureRecord.create(method_run_id=method_run_id, reason=message, category=category)
