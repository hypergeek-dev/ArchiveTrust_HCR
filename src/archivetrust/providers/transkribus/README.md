# Transkribus adapter -- manual import mode (Swedish Lion I)

Third real `HtrMethodAdapter` implementation (`docs/htr-migration-plan.md` Stage 8), and
architecturally the odd one out among the three: **it never runs inference and makes zero network
calls.** SATRN and Florence-2 wrap a local (or subprocess-local) model; this adapter wraps a
*parser* for files a researcher already produced by hand, outside this application, using
Transkribus's own web UI or desktop client.

## What "manual import mode" means, concretely

1. A researcher opens Transkribus (a separate, external, browser-based or desktop application --
   not reachable from or launched by ArchiveTrust) and runs Swedish Lion I recognition on a
   document there, using their own Transkribus account.
2. The researcher manually exports the result -- PAGE XML, ALTO XML, or plain text -- and manually
   saves it to local disk, somewhere ArchiveTrust can read it.
3. Only then does this adapter ever see the file: `TranskribusAdapter.recognize()` takes a
   **file path** (`RecognitionInput.configuration["export_file_path"]`), reads it with
   `pathlib.Path.read_text`, and parses it. That is the entire I/O surface of this package.

**There is no step where this application uploads a document to Transkribus, and no step where it
downloads or fetches a result from Transkribus over a network.** Every byte this adapter ever
touches was already on local disk before `recognize()` was called, placed there by the
researcher's own, separate, manual action.

## Why this matters (the task brief's explicit constraint)

> Manual import mode must be implemented first... Never send documents to Transkribus
> automatically. Uploading a document to an external service must require explicit user action...
> Never hardcode Transkribus credentials.

This is enforced by **adapter design**, not by a runtime check that could be bypassed or a
credential that could be forgotten-then-left-in-code:

- `recognize()`'s only parameter path into the export file is a local filesystem path. There is no
  `requests`/`httpx`/`urllib` import anywhere in this package (`grep -r "import requests\|import
  httpx\|urllib.request" src/archivetrust/providers/transkribus/` returns nothing) -- there is no
  Transkribus API client to call, so there is no automatic-upload code path to accidentally trigger.
- No credential of any kind (API key, username/password, session token) is read, stored, or
  referenced anywhere in this package. There is nothing to hardcode because there is nothing that
  authenticates to anything.
- `validate_environment()`/`health_check()` check local file/directory readability only -- never a
  network reachability probe to `transkribus.eu` or any Transkribus endpoint.

## API mode is explicitly out of scope for this phase

A live Transkribus REST API integration (submitting a job, polling for completion, fetching the
result automatically) is a real, documented Transkribus capability, but implementing it here would
directly contradict "manual import mode must be implemented first" and "never send documents to
Transkribus automatically" -- an API-mode adapter's `recognize()` would, by definition, need to
either upload a document or at minimum authenticate and poll an external service automatically.
That is deliberately not started in this phase: no HTTP client dependency was added, no credential
configuration surface was added, no `providers/transkribus/api_client.py` (or similar) exists. A
future phase that explicitly re-opens this scope would need its own explicit-user-action gate
(e.g. a UI "Upload to Transkribus" button the user must click per-document, never an automatic
background trigger) and its own credential-handling design (e.g. an OS keychain / user-supplied
env var, never a value committed to this repository) -- neither of which this phase specifies or
implements.

## Supported formats

- **PAGE XML** (`page_xml.py`) -- the PRImA Page format. Namespace-tolerant (matches on local tag
  names, not a pinned schema-version URI), parses `TextRegion`/`TextLine` geometry (`Coords`/
  `Baseline` point lists), per-line `TextEquiv/@conf` confidence, reading order (from an explicit
  `<ReadingOrder>` block, falling back to Transkribus's own `custom="readingOrder {index:N;}"`
  convention, falling back to document order), and vendor metadata via the standard
  `Metadata/MetadataItem` mechanism (`vendorReportedAccuracy`, `modelVersion`, `jobId`,
  `transkribusDocId`).
- **ALTO XML** (`alto_xml.py`) -- namespace-tolerant, parses `TextBlock`/`TextLine`/`String`
  structure, `HPOS`/`VPOS`/`WIDTH`/`HEIGHT` geometry, and per-word `WC` confidence (averaged per
  line -- ALTO has no standard per-line confidence attribute).
- **Plain text** (`plain_text.py`) -- the trivial case: each non-blank line becomes one line, no
  geometry, no confidence. Still wrapped in the same `ParsedDocument`/`ExternalImport` provenance
  chain as the two XML formats -- no shortcut around content-addressing.

Both XML parsers use Python's standard-library `xml.etree.ElementTree` -- a real, namespace-aware
parser. `lxml` was considered and deliberately not added: nothing here needs XPath 2.0, XInclude,
or validation against the actual PRImA/ALTO `.xsd` schemas (which would mean vendoring or fetching
them), and `ElementTree` handles everything this adapter actually needs. No new dependency was
added to `pyproject.toml` for this adapter.

## Capability-flag honesty: `local_execution_supported` / `external_upload_required`

`MethodCapabilities` has exactly two execution-shape booleans. Neither alone captures this
adapter's actual shape ("the recognition was produced by an external service, at some point in the
past, by a human's own separate action -- but this adapter itself uploads nothing and calls no
network API"):

- `local_execution_supported=False` -- unambiguously correct: no model runs in this process.
- `external_upload_required=False` -- correct for what it actually asks ("does *this adapter*
  require uploading anything to complete `recognize()`?" -- no, the file is already local by the
  time `recognize()` is called) but incomplete as a full picture: read next to
  `local_execution_supported=False`, it could be misread as "fully local, no external service
  involved at all", which is not true either -- Transkribus, an external service, genuinely
  produced the recognition this adapter parses.

There is no third boolean in the existing `MethodCapabilities` schema to express "external-service-
originated output, zero network calls made by this adapter" cleanly, and this adapter does not
invent one by overloading either existing flag with a meaning it wasn't designed to carry. This is
recorded as a known representational gap (see `adapter.py`'s module docstring and
`get_capabilities()`'s inline comment) rather than silently mis-flagged. `MethodMetadata.vendor`
("READ-COOP (Transkribus)") and the very existence of `ExternalImport` (never constructed for
SATRN/Florence-2, which have no external-origin story to record) are what actually carry "this came
from an external service" for any caller that reads them.

## Timing and device fields: what is real vs. what is honestly `None`

- `RecognitionResult.execution_time_ms` is always `None` -- no recognition execution happened in
  this process. The one real, measured cost this adapter incurs -- its own XML/text parsing -- is
  recorded separately as `raw_response["adapter_parse_time_ms"]`, never written into
  `execution_time_ms` (conflating the two would misrepresent parse time as if it were model
  inference time).
- How long Transkribus itself took to produce the export is unknowable from the file alone in the
  general case; `raw_response["external_execution_time"]` is the literal string `"unknown -- this
  was imported after the fact, not measured"`, not a fabricated number and not a silently omitted
  field.
- `Evidence.execution_device`, `execution_time_ms`, `gpu_memory_mb`, `software_environment`, and
  `hardware_environment` are always `None` -- this adapter ran on no device; it read a file.

## Raw vs. parsed vs. normalized (one stage different from SATRN/Florence-2)

For SATRN and Florence-2, "raw" means the model's own raw text output. Here there is no model
output to be raw *text* of -- the raw artifact is the **export file itself** (PAGE/ALTO XML, or
plain text), a structured document format, not yet a transcription. So:

- `Evidence.raw_output` = the untouched export file content, content-addressed exactly like any
  other Evidence (Stage 8's "original export file ... preserved, content-addressed like other raw
  evidence -- do not silently discard/re-encode it").
- `build_observation_payloads()` returns **two** stages, not three:
  `(ParsedTranscriptionPayload, NormalizedTranscriptionPayload)`. Extracting linear text from
  PAGE/ALTO regions/lines genuinely *is* the parsing step here -- there is no separate "raw model
  emission" stage to populate a `RawTranscriptionPayload` from.

## Vendor-reported accuracy: kept separate, never compared to ArchiveTrust's own metrics

> Do not assume that vendor-reported CER values are directly comparable with ArchiveTrust results.

When a PAGE XML export carries a `Metadata/MetadataItem name="vendorReportedAccuracy"` value (the
standard PAGE XML mechanism for arbitrary custom metadata -- not a fabricated schema extension),
it is preserved as:

- `ParsedDocument.vendor_reported_accuracy` / `raw_response["vendor_reported_accuracy"]` at parse
  time,
- `Evidence.supporting_metadata["vendor_reported_accuracy"]` on the built `Evidence`,
- `ExternalImport.vendor_reported_accuracy` if the caller threads it through when constructing the
  `ExternalImport` record.

It is **never** written into, averaged with, or otherwise merged into any value
`evaluation/metrics.py` computes. ALTO XML has no standard vendor-accuracy element at all; this
field is `None` for every ALTO import, honestly, not approximated from anything else in the file.

## Association: linking an import to an existing document/page/region/experiment

`external_import.py::associate_external_import(external_import, *, target_kind, target_id,
associated_at, associated_by=None)` produces an `ImportAssociation` -- a small, id-referencing,
append-only record linking one `ExternalImport` to one existing ArchiveTrust `Document` (an
`ArchiveObject`), `Page`, `Region`, or `Experiment` (`ImportTargetKind`). It never embeds or copies
the target entity (Graph-Reference Rule, Constitution Article 7) and never mutates the frozen
`ExternalImport` -- each call produces one additional, independent association record, so one
import can legitimately be associated with more than one target over time (e.g. first a `Page`,
later also the `Experiment` it was folded into) without losing either linkage.

## Failure handling

Every parser raises `TranskribusParseError(category, message)` -- never a bare
`xml.etree.ElementTree.ParseError` or `KeyError` -- for: `file_not_found`, `empty_file`,
`malformed_xml`, `missing_required_element`, `unsupported_format`. `adapter.py::recognize()`
catches these and returns a failed `RecognitionResult`
(`text=None, raw_response={"category": ..., "message": ...}`); `build_failure_record()` turns that
into a `FailureRecord`, preserved (never excluded) per the domain design. See
`tests/providers/transkribus/` for one exercised test per category, against real (hand-authored,
schema-conformant) malformed fixtures -- not just a mocked exception.

## Known limitations / left ambiguous

- No real Transkribus export was available to validate against in this environment (no
  network-reachable Transkribus account) -- test fixtures are hand-authored, schema-conformant
  PAGE/ALTO XML, not genuine Transkribus output. See `tests/fixtures/transkribus/README.md` for
  exact provenance.
- The mapping from `MetadataItem name="..."` conventions (`vendorReportedAccuracy`, `modelVersion`,
  `jobId`, `transkribusDocId`) to this adapter's hint fields is this adapter's own convention, not
  a Transkribus-published standard -- Transkribus's real exports may use different `MetadataItem`
  names (or none at all) for equivalent information; this adapter degrades to `None` for any it
  does not recognize, never guesses.
- `Evidence.processing_stage=ProcessingStage.OCR` was chosen for consistency with SATRN's Evidence
  (both represent a completed recognition result, not a VLM inference or layout-analysis stage) --
  an arguable choice given Transkribus's Swedish Lion I model is itself HTRflow/CRNN-family, not
  architecturally identical to SATRN; no `ProcessingStage` value exists specifically for "externally
  produced, manually imported recognition", and adding one was judged out of scope for this
  adapter alone.
- API mode (see above) is fully deferred, not partially started.
