"""`TranskribusAdapter` -- manual-import-only `HtrMethodAdapter` for Transkribus Swedish Lion I
(docs/htr-migration-plan.md Stage 8).

**Architecturally different from `providers/satrn/adapter.py` and `providers/florence2_htr/
adapter.py` on purpose:** this adapter never runs inference and makes zero network calls. It
parses a Transkribus export file that a researcher has *already produced by hand outside this
application* (Transkribus's web UI or desktop client, run manually, exported manually, saved to
local disk manually) and turns that file's content into the same `RecognitionResult`/`Evidence`
shape the other two adapters produce from a live model run. There is no code path anywhere in this
package that fetches, uploads, or polls anything over a network -- `recognize()` takes a *file
path*, not an image, and `parsing.py::parse_export_file` only ever calls `pathlib.Path.read_text`.
API mode (a live Transkribus REST integration) is explicitly out of scope for this phase per the
task brief ("Manual import mode must be implemented first... Never send documents to Transkribus
automatically") -- see README.md's "API mode is out of scope" section for what that would require
and why it is deliberately not started here.

**Capability-flag honesty (the task's explicit ambiguity to resolve, not paper over):**
`MethodCapabilities` has exactly two execution-shape flags, `local_execution_supported` and
`external_upload_required`. Neither alone describes this adapter's actual shape ("the recognition
was produced by an external service, at some point in the past, by a human's own action -- but
*this adapter* uploads nothing and calls no network API"). Concretely:
- `local_execution_supported=False` -- correct and unambiguous: no model runs in this process.
- `external_upload_required=False` -- the closest honest value, but incomplete on its own: it is
  correctly `False` because *this adapter* requires no upload (the file already exists locally by
  the time `recognize()` is ever called) -- but naively reading `external_upload_required=False`
  next to `local_execution_supported=False` could look like "fully local, no external service
  involved at all", which is not true either. There is no third boolean in the existing
  `MethodCapabilities` schema to say "external-service-originated, zero network calls made by this
  adapter" cleanly. Rather than overload either existing flag with a meaning it wasn't designed to
  carry, this is recorded as a **known representational gap** here, in `get_capabilities()`'s own
  inline comment, and in README.md -- not silently mis-flagged. `MethodMetadata.vendor` ("READ-COOP
  (Transkribus)") and `ExternalImport`'s very existence are what actually carry "this came from an
  external service" for any caller that reads them.

**Raw vs. parsed vs. normalized (Constitution: never silently normalize) -- one stage different
from SATRN/Florence-2:** for SATRN and Florence-2, "raw" means the model's own raw text output.
Here there is no model output to be raw *text* of -- the raw artifact is the export **file itself**
(PAGE/ALTO XML, or plain text), which is not yet a transcription at all, just a structured document
format. So `Evidence.raw_output` is the untouched file content (content-addressed, exactly as
Stage 8 requires: "original export file ... preserved, content-addressed like other raw evidence
-- do not silently discard/re-encode it"), and the transcription-payload chain this adapter
populates starts one stage later than SATRN/Florence-2's: `build_observation_payloads` returns
`(ParsedTranscriptionPayload, NormalizedTranscriptionPayload)` -- two stages, not the raw stage --
because extracting linear text from PAGE/ALTO regions/lines genuinely *is* the parsing step, not a
raw model emission.

**Timing honesty (the task's explicit "do not conflate" instruction):** `RecognitionResult.
execution_time_ms` is always `None` here -- no recognition execution happened in this process,
full stop, and reporting any number in that field would misrepresent it as if this adapter timed a
model run. The one real, measured cost this adapter *does* incur -- its own XML/text parsing --
is recorded separately as `raw_response["adapter_parse_time_ms"]`, never written into
`execution_time_ms`. How long Transkribus itself took to produce the export is not knowable from
the file alone in the general case and is recorded as `raw_response["external_execution_time"] =
"unknown -- this was imported after the fact, not measured"` rather than omitted silently.

**Device/GPU fields (the task's explicit "do not use misleading zero/default values"
instruction):** `Evidence.execution_device`, `execution_time_ms`, and `gpu_memory_mb` are always
`None` here, never `"cpu"`/`0.0` -- this adapter ran on no device at all; it read a file.
"""

from __future__ import annotations

import time
import unicodedata
from pathlib import Path

from archivetrust.domain.evidence.models import Evidence, ProcessingStage
from archivetrust.domain.ontology.payloads.transcription import (
    NormalizedTranscriptionPayload,
    ParsedTranscriptionPayload,
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
from archivetrust.providers.transkribus.parsing import parse_export_file
from archivetrust.providers.transkribus.parsing_models import TranskribusParseError

METHOD_ID = "transkribus_swedish_lion_1"
METHOD_NAME = "Transkribus Swedish Lion I"
VENDOR = "READ-COOP (Transkribus)"
ADAPTER_VERSION = "1.0.0"
"""This adapter's own version -- the request/response mapping between a parsed export file and
`RecognitionResult`/`Evidence`. Distinct from any Transkribus-side model version, which (per this
module's docstring) is often simply not present in the export file at all and is recorded per-run
in `supporting_metadata`/`raw_response`, never here."""

DEFAULT_METADATA_MODEL_REVISION = (
    "unpinned (Transkribus manual export declares no fixed checkpoint hash; "
    "see per-run RecognitionResult.model_revision / Evidence.supporting_metadata "
    "for any version string the specific import file states, if any)"
)
"""`get_metadata()` is called before any particular file is known, so it cannot report a per-run
value -- this honestly says why, rather than fabricating a version number (Stage 8's explicit
instruction: "do not fabricate a version number, record None/absent honestly if the file doesn't
state one")."""

KNOWN_FAILURE_CATEGORIES = (
    "file_not_found",
    "empty_file",
    "malformed_xml",
    "missing_required_element",
    "unsupported_format",
    "malformed_input",
)
"""Kept as a module-level tuple (not an Enum), same rationale as SATRN's/Florence-2's:
`TranskribusParseError.category` values a parser adds later are still carried verbatim into a
`FailureRecord` even if not named here (Constitution Article 6: Full Exposure)."""


def normalize_transcription(parsed_text: str) -> str:
    """The only normalization this adapter performs -- Unicode NFC + outer whitespace stripping,
    applied to the already-parsed linear transcription (never to the raw XML/text file content).
    Mirrors `providers/satrn/adapter.py::normalize_transcription` /
    `providers/florence2_htr/adapter.py::normalize_transcription` exactly: nothing lexical, per
    the Constitution's "never silently normalize" rule."""
    return unicodedata.normalize("NFC", parsed_text).strip()


class TranskribusAdapter:
    """Implements `archivetrust.providers.htr_adapter.HtrMethodAdapter` (verified via
    `isinstance(TranskribusAdapter(), HtrMethodAdapter)` in the contract test). Manual-import mode
    only -- see module docstring."""

    def __init__(
        self,
        *,
        model_name: str = METHOD_NAME,
        import_directory: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._import_directory = import_directory
        """Optional configured directory `validate_environment()`/`health_check()` probe for
        read/write access (Stage 8: "cheap check e.g. can we read/write the configured import
        directory"). `recognize()` never depends on this being set -- it always takes an explicit
        file path per call via `RecognitionInput.configuration["export_file_path"]`."""

    def get_metadata(self) -> MethodMetadata:
        return MethodMetadata(
            method_id=METHOD_ID,
            method_name=self._model_name,
            vendor=VENDOR,
            model_revision=DEFAULT_METADATA_MODEL_REVISION,
        )

    def get_capabilities(self) -> MethodCapabilities:
        return MethodCapabilities(
            # PAGE @conf / ALTO WC, only when the specific file actually carries one -- never
            # fabricated when absent (RecognitionResult.confidence is None in that case).
            confidence_supported=True,
            # PAGE Coords/Baseline points, ALTO HPOS/VPOS/WIDTH/HEIGHT -- real geometry when the
            # file states it.
            geometry_supported=True,
            line_level_supported=True,
            page_level_supported=True,
            # No model runs in this process -- see module docstring's capability-flag section.
            local_execution_supported=False,
            # This adapter uploads nothing (manual-import mode makes zero network calls) -- but
            # see module docstring: this flag cannot fully express "external-service-originated
            # output, zero network calls made by this adapter". False is the least-misleading
            # available value; the gap is documented, not silently accepted.
            external_upload_required=False,
            # Page-level Swedish Lion I runs consume an image ArchiveTrust prepares and a researcher
            # hands to an external service, so that image's colour representation is an experimental
            # variable and must be a recorded fact rather than whatever the source file carried. See
            # docs/methods/transkribus-swedish-lion-1.md §2.
            image_color_normalization_required=True,
        )

    def validate_environment(self) -> EnvironmentValidation:
        """Checks the *import file*/directory, never GPU/model prerequisites -- there is no model
        to check (Stage 8: "not GPU/model checks")."""
        messages: list[str] = []
        valid = True

        if self._import_directory is None:
            messages.append(
                "No import_directory configured -- recognize() requires an explicit "
                "export_file_path per call via RecognitionInput.configuration"
            )
            return EnvironmentValidation(valid=True, messages=tuple(messages))

        path = Path(self._import_directory)
        if not path.exists():
            valid = False
            messages.append(f"Configured import directory does not exist: {self._import_directory}")
        elif not path.is_dir():
            valid = False
            messages.append(f"Configured import directory is not a directory: {self._import_directory}")
        else:
            messages.append(f"Import directory OK (exists, is a directory): {self._import_directory}")

        return EnvironmentValidation(valid=valid, messages=tuple(messages))

    def recognize(self, input: RecognitionInput) -> RecognitionResult:
        """Takes a **file path to a pre-existing export** (Stage 8's explicit contract), never an
        image and never a live Transkribus session. The path is read from
        `input.configuration["export_file_path"]` -- `RecognitionInput.input_crop_id`/
        `page_image_ref` are documented (htr_adapter.py) as image references for methods that run
        inference on pixels, which this adapter never does; reusing either field for a file path
        would be misleading, so `configuration` (the Protocol's method-specific-configuration
        escape hatch) is used instead, and that choice is recorded here rather than silently
        overloading a field with the wrong documented meaning.
        """
        file_path = input.configuration.get("export_file_path")
        if not file_path:
            return RecognitionResult(
                text=None,
                raw_response={
                    "ok": False,
                    "category": "malformed_input",
                    "message": (
                        "RecognitionInput.configuration carried no 'export_file_path' -- this "
                        "adapter requires a path to a pre-existing Transkribus export file, "
                        "never an image reference"
                    ),
                },
            )

        export_format = input.configuration.get("export_format")

        parse_started = time.monotonic()
        try:
            parsed, raw_file_text = parse_export_file(file_path, export_format=export_format)
        except TranskribusParseError as exc:
            return RecognitionResult(
                text=None,
                raw_response={
                    "ok": False,
                    "category": exc.category,
                    "message": exc.message,
                    "source_file_path": file_path,
                },
            )
        adapter_parse_time_ms = (time.monotonic() - parse_started) * 1000.0

        parsed_text = parsed.full_text()
        confidence = parsed.mean_confidence()

        raw_response = {
            "ok": True,
            "export_format": parsed.export_format,
            "source_file_path": file_path,
            "raw_file_text": raw_file_text,
            "regions": [region.model_dump() for region in parsed.regions],
            "page_width": parsed.page_width,
            "page_height": parsed.page_height,
            "image_filename": parsed.image_filename,
            "processing_date": parsed.processing_date,
            "processing_date_source": "export_file" if parsed.processing_date is not None else "import_timestamp",
            "vendor_reported_accuracy": parsed.vendor_reported_accuracy,
            "model_version_hint": parsed.model_version_hint,
            "job_id_hint": parsed.job_id_hint,
            "transkribus_document_id_hint": parsed.transkribus_document_id_hint,
            "warnings": list(parsed.warnings),
            "adapter_parse_time_ms": adapter_parse_time_ms,
            "external_execution_time": "unknown -- this was imported after the fact, not measured",
        }

        return RecognitionResult(
            text=parsed_text,
            confidence=confidence,
            raw_response=raw_response,
            # Never conflated with adapter_parse_time_ms above -- see module docstring.
            execution_time_ms=None,
            model_revision=parsed.model_version_hint,
        )

    def health_check(self) -> HealthCheckResult:
        """Cheap liveness probe -- can the configured import directory be read, never a network
        ping to Transkribus (Stage 8: "not a network ping to Transkribus (there is no network
        call in this phase, full stop)")."""
        if self._import_directory is None:
            return HealthCheckResult(
                healthy=True,
                message="No import_directory configured; nothing to probe (recognize() takes an explicit file path per call)",
            )
        path = Path(self._import_directory)
        healthy = path.exists() and path.is_dir()
        return HealthCheckResult(
            healthy=healthy,
            message=f"import_directory={self._import_directory!r} exists_and_is_dir={healthy}",
        )


def build_evidence(result: RecognitionResult, *, adapter_version: str = ADAPTER_VERSION) -> Evidence | None:
    """Populates `Evidence`'s HTR extension fields from a `RecognitionResult`. `raw_output` is the
    **untouched export file content** (not `result.text`, which is already the parsed linear
    transcription -- see module docstring's "raw vs. parsed vs. normalized" section). Returns
    `None` for a failed parse -- callers should use `build_failure_record` for that case instead.
    """
    if result.text is None:
        return None

    raw_response = result.raw_response
    return Evidence.create(
        provider=METHOD_ID,
        provider_version=adapter_version,
        raw_output=raw_response.get("raw_file_text", result.text),
        processing_stage=ProcessingStage.OCR,
        provider_confidence=result.confidence,
        model_revision=result.model_revision,
        # No local execution happened -- explicit None, never a misleading 0/"cpu" default (see
        # module docstring).
        execution_device=None,
        execution_time_ms=None,
        gpu_memory_mb=None,
        software_environment=None,
        hardware_environment=None,
        supporting_metadata={
            "export_format": raw_response.get("export_format"),
            "source_file_path": raw_response.get("source_file_path"),
            "processing_date": raw_response.get("processing_date"),
            "processing_date_source": raw_response.get("processing_date_source"),
            "job_id_hint": raw_response.get("job_id_hint"),
            "transkribus_document_id_hint": raw_response.get("transkribus_document_id_hint"),
            "adapter_parse_time_ms": raw_response.get("adapter_parse_time_ms"),
            "external_execution_time": raw_response.get("external_execution_time"),
            # Preserved as audit-only metadata -- NEVER to be merged into or compared directly
            # against ArchiveTrust's own CER/WER metrics (evaluation/metrics.py). See README.md.
            "vendor_reported_accuracy": raw_response.get("vendor_reported_accuracy"),
        },
    )


def build_observation_payloads(
    result: RecognitionResult,
) -> tuple[ParsedTranscriptionPayload, NormalizedTranscriptionPayload] | None:
    """Builds the parsed and normalized transcription payloads as two genuinely separate objects
    (Constitution: never silently normalize) -- `None` for a failed parse. Only two stages, not
    three (unlike Florence-2's adapter) -- see module docstring: there is no separate "raw model
    text" stage here, the raw artifact is the export file itself, captured on `Evidence.raw_output`
    by `build_evidence`, not as a transcription payload."""
    if result.text is None:
        return None
    parsed_payload = ParsedTranscriptionPayload(text=result.text)
    normalized_payload = NormalizedTranscriptionPayload(text=normalize_transcription(result.text))
    return parsed_payload, normalized_payload


def build_failure_record(result: RecognitionResult, *, method_run_id: str) -> FailureRecord | None:
    """Builds a `FailureRecord` from a failed `RecognitionResult` -- `None` when the result did
    not actually fail, so a caller can call this unconditionally after `recognize()` without
    branching first. Mirrors `providers/satrn/adapter.py::build_failure_record`."""
    if result.text is not None:
        return None
    raw_response = result.raw_response
    category = raw_response.get("category")
    message = raw_response.get("message", "Transkribus manual import failed with no message")
    return FailureRecord.create(method_run_id=method_run_id, reason=message, category=category)
