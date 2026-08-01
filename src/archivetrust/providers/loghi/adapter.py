"""`LoghiAdapter` -- the page-level, containerized, multi-component `HtrMethodAdapter` implementation
for Loghi (Laypa layout analysis -> Loghi Tooling line extraction/reading order -> Loghi HTR
recognition).

**Modular pipeline, not one opaque recognizer.** Every other adapter in this codebase wraps a single
model call; Loghi wraps three independently-versioned components run inside Docker/WSL2 containers.
`get_metadata().model_revision` renders `LoghiComponentVersions.summary()` rather than one checkpoint
string, and every stage's real outcome is preserved on the `LoghiPipelineResult` this adapter's
`recognize()` gets back from its facade (`stage_results.py`) -- "only the final text is evidence" never
happens here.

**Why `line_level_supported=False`, deliberately, even though Loghi internally cuts lines.** The brief
is explicit: internal line-cutting does not earn this flag. It would only be `True` if ArchiveTrust
could intentionally hand Loghi's recognition stage one line crop and get a traceable result back under
a supported workflow -- that plumbing does not exist yet, so this stays an honest `False`.

**Why `image_color_normalization_required=True`.** `MethodCapabilities.image_color_normalization_
required`'s docstring (`providers/htr_adapter.py`) describes "a pipeline obligation... for a method
whose input ArchiveTrust prepares and hands to an external service." Loghi's containers are a process
boundary ArchiveTrust hands a prepared image to, exactly like Transkribus's external upload in every
respect except the transport (a mounted volume instead of a network upload) -- the same reasoning that
makes the flag `True` for Transkribus applies here, per the "shared page-input policy" the brief
requires (both Lion and Loghi read the same canonical RGB-normalized derivative).

**Why `local_execution_supported=True` and `external_upload_required=False` even though it runs in a
container.** No image or file ever leaves this machine -- Docker/WSL2 is a local execution boundary,
not a network upload, matching the same distinction SATRN's isolated-venv subprocess already draws.
"""

from __future__ import annotations

import tempfile
import time

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
from archivetrust.providers.loghi.environment import probe_loghi_environment
from archivetrust.providers.loghi.facade import LoghiWorkerFacade, subprocess_loghi_facade
from archivetrust.providers.loghi.models import LoghiComponentVersions
from archivetrust.providers.loghi.page_xml import parse_loghi_page_xml
from archivetrust.providers.loghi.parsing_models import LoghiParseError
from archivetrust.providers.loghi.pinned_versions import CURRENT_PINNED_VERSIONS

METHOD_ID = "loghi"
METHOD_NAME = "Loghi (Laypa + Loghi Tooling + Loghi HTR)"
VENDOR = "KNAW Humanities Cluster (knaw-huc/loghi)"
ADAPTER_VERSION = "1.0.0"
"""This adapter's own version -- distinct from every `LoghiComponentVersions` field, which pin the
external pipeline's own repositories/checkpoint/image, not this file's request/response mapping."""

KNOWN_FAILURE_CATEGORIES = (
    "environment_unavailable",
    "container_startup_failure",
    "model_loading_failure",
    "layout_analysis_failure",
    "line_extraction_failure",
    "recognition_failure",
    "page_xml_generation_failure",
    "page_xml_parse_failure",
    "external_import_failure",
    "mapping_failure",
    "timeout",
    "out_of_memory",
    "malformed_output",
    "empty_output",
)
"""The failure taxonomy from docs/methods/loghi.md, kept distinct on purpose -- a "which stage broke"
question this codebase's other, single-call adapters never had to answer. Deliberately excludes
`"suspicious_successful_output"`: a successful run with poor-looking text is not a failure (brief) --
see `LoghiPipelineResult.suspicious_output` instead, a flag on a *successful* result."""


def normalize_transcription(text: str) -> str:
    """The only lexical normalization this adapter performs -- Unicode NFC + outer whitespace
    stripping. Mirrors every other adapter's `normalize_transcription` exactly (Constitution: never
    silently normalize beyond this)."""
    import unicodedata

    return unicodedata.normalize("NFC", text).strip()


class LoghiAdapter:
    """Implements `archivetrust.providers.htr_adapter.HtrMethodAdapter` (verified via
    `isinstance(LoghiAdapter(), HtrMethodAdapter)` in the contract test)."""

    def __init__(
        self,
        *,
        component_versions: LoghiComponentVersions = CURRENT_PINNED_VERSIONS,
        facade: LoghiWorkerFacade | None = None,
        timeout_seconds: float = 1800.0,
    ) -> None:
        self._component_versions = component_versions
        self._facade = facade if facade is not None else subprocess_loghi_facade()
        self._timeout_seconds = timeout_seconds

    @property
    def component_versions(self) -> LoghiComponentVersions:
        """The pinned configuration this adapter instance was constructed with -- read-only, so
        callers (e.g. the smoke test's telemetry recording) never need private-attribute access."""
        return self._component_versions

    def get_metadata(self) -> MethodMetadata:
        return MethodMetadata(
            method_id=METHOD_ID,
            method_name=METHOD_NAME,
            vendor=VENDOR,
            model_revision=self._component_versions.summary(),
        )

    def get_capabilities(self) -> MethodCapabilities:
        return MethodCapabilities(
            # Real PAGE @conf, only when Loghi's own output states one -- never fabricated.
            confidence_supported=True,
            geometry_supported=True,
            # Deliberately False -- see module docstring's "line_level_supported" section.
            line_level_supported=False,
            page_level_supported=True,
            # Docker/WSL2 is a local execution boundary, not a network upload -- see module
            # docstring.
            local_execution_supported=True,
            external_upload_required=False,
            # The canonical RGB-normalized derivative is what gets mounted into the container --
            # see module docstring's "image_color_normalization_required" section.
            image_color_normalization_required=True,
        )

    def validate_environment(self) -> EnvironmentValidation:
        messages: list[str] = []
        valid = True

        if self._component_versions.is_placeholder():
            valid = False
            messages.append(
                "LoghiComponentVersions still carries UNPINNED placeholder values -- see "
                "providers/loghi/pinned_versions.py and docs/methods/loghi.md before any research run."
            )

        report = probe_loghi_environment()
        messages.append(
            f"host_os={report.host_os} docker_cli_present={report.docker_cli_present} "
            f"docker_version={report.docker_version!r} wsl_present={report.wsl_present} "
            f"wsl_distros={report.wsl_distros} linux_distribution={report.linux_distribution!r} "
            f"execution_mode={report.execution_mode}"
        )
        if report.execution_mode is None:
            valid = False
            messages.append(
                "No supported Loghi execution mode (Docker on Linux / Docker through WSL2 / native "
                "Linux) is currently available on this host."
            )

        return EnvironmentValidation(valid=valid, messages=tuple(messages))

    def recognize(self, input: RecognitionInput) -> RecognitionResult:
        image_path = input.page_image_ref or input.input_crop_id
        if not image_path:
            return RecognitionResult(
                text=None,
                raw_response={
                    "ok": False,
                    "category": "malformed_input",
                    "message": (
                        "RecognitionInput carried neither page_image_ref nor input_crop_id -- Loghi "
                        "is page_level_supported only (line_level_supported=False)."
                    ),
                },
            )

        report = probe_loghi_environment()
        if self._component_versions.is_placeholder() or report.execution_mode is None:
            return RecognitionResult(
                text=None,
                raw_response={
                    "ok": False,
                    "category": "environment_unavailable",
                    "message": "Loghi environment/pins not ready -- see validate_environment().",
                    "environment_report": report.model_dump(),
                },
            )

        with tempfile.TemporaryDirectory(prefix="loghi_output_") as output_dir:
            started = time.monotonic()
            pipeline_result = self._facade.run_pipeline(
                input_image_path=image_path,
                output_dir=output_dir,
                component_versions=self._component_versions,
                environment=report,
                timeout_seconds=self._timeout_seconds,
            )
            elapsed_ms = (time.monotonic() - started) * 1000.0

            if not pipeline_result.ok or pipeline_result.final_page_xml is None:
                failed_stage = next((s for s in pipeline_result.stages if not s.ok), None)
                return RecognitionResult(
                    text=None,
                    raw_response={
                        "ok": False,
                        "category": "recognition_failure" if pipeline_result.stages else "container_startup_failure",
                        "message": (
                            "; ".join(failed_stage.errors) if failed_stage and failed_stage.errors else
                            "Loghi pipeline did not produce a final PAGE XML output"
                        ),
                        "pipeline_result": pipeline_result.model_dump(),
                    },
                )

            try:
                parsed = parse_loghi_page_xml(pipeline_result.final_page_xml)
            except LoghiParseError as exc:
                return RecognitionResult(
                    text=None,
                    raw_response={
                        "ok": False,
                        "category": "page_xml_parse_failure",
                        "message": exc.message,
                        "pipeline_result": pipeline_result.model_dump(),
                    },
                )

            text = parsed.full_text()
            if not text.strip():
                return RecognitionResult(
                    text=None,
                    raw_response={
                        "ok": False,
                        "category": "empty_output",
                        "message": "Loghi produced a parseable PAGE XML with no recognizable text",
                        "pipeline_result": pipeline_result.model_dump(),
                        "parsed_page": parsed.model_dump(),
                    },
                )

            return RecognitionResult(
                text=text,
                confidence=parsed.mean_confidence(),
                raw_response={
                    "ok": True,
                    "pipeline_result": pipeline_result.model_dump(),
                    "parsed_page": parsed.model_dump(),
                    "suspicious_output": pipeline_result.suspicious_output,
                    "suspicious_output_reason": pipeline_result.suspicious_output_reason,
                },
                execution_time_ms=pipeline_result.total_duration_ms or elapsed_ms,
                model_revision=self._component_versions.summary(),
            )

    def health_check(self) -> HealthCheckResult:
        """Cheap liveness probe -- environment probing only, never a container start or model
        load."""
        report = probe_loghi_environment()
        healthy = report.execution_mode is not None and not self._component_versions.is_placeholder()
        return HealthCheckResult(
            healthy=healthy,
            message=f"execution_mode={report.execution_mode} pins_placeholder={self._component_versions.is_placeholder()}",
        )


def build_evidence(result: RecognitionResult, *, adapter_version: str = ADAPTER_VERSION) -> Evidence | None:
    """Populates `Evidence`'s HTR extension fields from a `RecognitionResult`. `raw_output` is the
    untouched final Loghi PAGE XML (not `result.text`, the already-parsed linear transcription) --
    same "raw is the file, not linearized text" discipline as `providers/transkribus/adapter.py`.
    Returns `None` for a failed run -- callers should use `build_failure_record` for that case."""
    if result.text is None:
        return None

    raw_response = result.raw_response
    pipeline_result = raw_response.get("pipeline_result", {})
    return Evidence.create(
        provider=METHOD_ID,
        provider_version=adapter_version,
        raw_output=pipeline_result.get("final_page_xml", result.text),
        processing_stage=ProcessingStage.OCR,
        provider_confidence=result.confidence,
        model_revision=result.model_revision,
        execution_time_ms=result.execution_time_ms,
        supporting_metadata={
            "stages": pipeline_result.get("stages"),
            "suspicious_output": raw_response.get("suspicious_output"),
            "suspicious_output_reason": raw_response.get("suspicious_output_reason"),
        },
    )


def build_observation_payloads(
    result: RecognitionResult,
) -> tuple[RawTranscriptionPayload, ParsedTranscriptionPayload, NormalizedTranscriptionPayload] | None:
    """Three genuinely separate stages, same discipline as every other adapter in this codebase
    (Constitution: never silently normalize). `raw` is the linearized PAGE XML text before this
    adapter's own normalization; `parsed` is identical to `raw` here because Loghi's own PAGE XML
    *is* the parsed structural extraction (there is no separate model-decode raw stage the way SATRN/
    Florence-2/Swedish-Lion have) -- `normalized` is the only stage this adapter actually transforms."""
    if result.text is None:
        return None
    raw_text = result.text
    parsed_text = result.text
    normalized_text = normalize_transcription(parsed_text)
    return (
        RawTranscriptionPayload(text=raw_text),
        ParsedTranscriptionPayload(text=parsed_text),
        NormalizedTranscriptionPayload(text=normalized_text),
    )


def build_failure_record(result: RecognitionResult, *, method_run_id: str) -> FailureRecord | None:
    """Builds a `FailureRecord` from a failed `RecognitionResult` -- `None` when the result did not
    actually fail. Mirrors every other adapter's `build_failure_record`."""
    if result.text is not None:
        return None
    raw_response = result.raw_response
    category = raw_response.get("category")
    message = raw_response.get("message", "Loghi recognition failed with no message")
    return FailureRecord.create(method_run_id=method_run_id, reason=message, category=category)
