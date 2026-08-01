# Loghi

Status: Current
Method id: `loghi`
Vendor: KNAW Humanities Cluster (`knaw-huc/loghi`)
Research status: `active` (current phase — `htr/research_status.py`)
Adapter: `src/archivetrust/providers/loghi/`

## What Loghi is, in this codebase

Loghi is a **page-level, containerized, multi-component pipeline** — not one model. Three
independently-versioned tools run in sequence inside Docker:

```
Original page
  → canonical RGB normalization (htr/preprocessing/, unchanged, shared with Swedish Lion I)
  → Loghi input package (providers/loghi/input_package.py)
  → Laypa            (layout analysis)
  → Loghi Tooling     (line extraction, reading order)
  → Loghi HTR         (recognition)
  → PAGE XML
  → ArchiveTrust parsing (providers/loghi/page_xml.py) / persistence
  → technical and human review
```

Every stage's outcome is recorded separately (`LoghiStageResult`, `providers/loghi/stage_results.py`)
— "only the final text is evidence" never happens here. See `docs/loghi-integration-audit.md` for how
this fits the rest of the codebase and `docs/CAPABILITY_MATRIX_HTR.md` §7 for the capability-flag
reasoning.

## Why this is not "one model_revision"

`LoghiComponentVersions` (`providers/loghi/models.py`) pins, independently: the top-level `loghi`
repository commit, every Git submodule commit (`laypa`, `loghi-htr`, `loghi-tooling`), the model
checkpoint id/hash, the Docker image tag/digest, the inference-script version, beam width,
reading-order/language-detection settings, GPU selection, and container runtime version. No field has a
default that could silently resolve to `"latest"`/`"main"` — see `pinned_versions.py`'s
`PLACEHOLDER_SENTINEL`, which is what `CURRENT_PINNED_VERSIONS` carries until a real environment is set
up, and which `validate_environment()` refuses.

**A later dependency update creates a new pipeline version.** Bumping any field in
`LoghiComponentVersions` is a new, reviewable configuration — never `git submodule update --remote`
inside a research run.

## Execution architecture

One of three explicitly-documented modes, probed (never assumed) by
`providers/loghi/environment.py::probe_loghi_environment()`:

* **Docker on Linux**
* **Docker through WSL2**
* **Native Linux**

Native Windows execution is not supported and is never claimed — Loghi's own tooling assumes Bash,
Linux paths and NVIDIA container tooling. `LoghiEnvironmentReport` records: host OS, WSL version,
Linux distribution, Docker version, NVIDIA Container Toolkit version, CUDA-visible device, resolved
execution mode. All read-only probes — no container is started to answer "is this available."

**On the machine this integration was built on:** Docker Desktop 29.6.1 and a `docker-desktop` WSL2
distro are installed, but no Loghi repos/images/checkpoints have been pulled — `validate_environment()`
honestly reports this exact state (see `docs/loghi-integration-audit.md` §13), not a blanket
"unavailable."

## Deterministic invocation, not an interactive shell

`providers/loghi/facade.py::_build_docker_argv` constructs one deterministic `docker run` (or
`wsl.exe -- docker run`) argument list from `LoghiComponentVersions` plus explicit mounted input/output
paths — never `shell=True`, never a hand-typed command. `_to_wsl_path` translates a Windows path
(`C:\Users\...`) to its WSL2 mount equivalent (`/mnt/c/Users/...`) for the WSL2 execution mode.

**What is and is not verified about this argv.** The `docker run`/`-v`/`--gpus`/`--rm` shape is
standard Docker CLI syntax. The container-side entrypoint and flag names (`/input`, `/output`,
`--beam-width`, ...) are this integration's placeholder convention, not yet confirmed against a real
pulled image's documented usage — no image has been pulled in this session to confirm against. Confirm
and adjust `_build_docker_argv` before the first real run.

## Capabilities (generated — `docs/CAPABILITY_MATRIX_HTR.md`)

| Capability | Value | Why |
|---|---|---|
| `page_level_supported` | `yes` | Loghi's native unit of work is a whole page. |
| `line_level_supported` | **`no`, deliberately** | Loghi cuts lines internally, but ArchiveTrust cannot yet intentionally hand its recognition stage a traceable line crop under a supported workflow — internal implementation detail does not earn this flag. |
| `geometry_supported` | `yes` | Real PAGE XML `Coords`/`Baseline` when the output states them. |
| `confidence_supported` | `yes` | Real PAGE `@conf`, only when stated — never fabricated. |
| `local_execution_supported` | `yes` | Docker/WSL2 is a local execution boundary; no image ever leaves this machine. |
| `external_upload_required` | `no` | Same reason. |
| `image_color_normalization_required` | `yes` | The canonical RGB-normalized page is what gets mounted into the container — a pipeline-handoff obligation, the same reasoning that makes this `yes` for Transkribus. |

## PAGE XML handling

`providers/loghi/page_xml.py::parse_loghi_page_xml` — its own parser, structurally similar to
`providers/transkribus/page_xml.py` (same stdlib `ElementTree`, same namespace-tolerant approach) but
returning its own `LoghiPageParseResult` type, **never merged with Transkribus's `ParsedDocument`**:
the two tools' field semantics are not guaranteed identical. Preserves:

* `source_xml_hash` — sha256 of the untouched output text.
* `page_schema_version` — read from the document's own namespace, never assumed.
* `omitted_fields` — what this parser deliberately does not extract (per-glyph geometry, text style,
  reading-order alternatives), stated rather than silently absent.

The original Loghi PAGE XML is preserved (`Evidence.raw_output`, via `adapter.py::build_evidence`);
`LoghiPageParseResult` is a separate, derived, ArchiveTrust-side representation.

## Failure taxonomy

`providers/loghi/adapter.py::KNOWN_FAILURE_CATEGORIES`: `environment_unavailable`,
`container_startup_failure`, `model_loading_failure`, `layout_analysis_failure`,
`line_extraction_failure`, `recognition_failure`, `page_xml_generation_failure`,
`page_xml_parse_failure`, `external_import_failure`, `mapping_failure`, `timeout`, `out_of_memory`,
`malformed_output`, `empty_output`. **`suspicious_successful_output` is deliberately not in this list**
— a successful run with poor-looking text is a flag on `LoghiPipelineResult.suspicious_output`, never a
`FailureRecord` category (brief: "A successful run with poor-looking text is not an execution
failure").

## Reproducibility

Every Loghi `ExperimentRun` gets a `ReproducibilityManifest` (`htr/experiment/models.py`, unchanged)
whose `software_environment`/`hardware_environment` carry the full `LoghiComponentVersions` +
`LoghiEnvironmentReport` + input/output hashes — a completed run's report is reconstructible from that
record without rerunning any container. `tests/htr/screening/test_loghi_smoke_test.py` demonstrates this
against a fake facade.

## Feasibility smoke test

`htr/screening/loghi_smoke_test.py::run_loghi_smoke_test` — 5 Swedish + 5 Dutch pages, exercising
environment validation → pipeline → PAGE XML → parse → telemetry → report, end to end. Run it via
`scripts/run_lion_loghi_comparison.py` option 4. On this machine, right now, it will honestly report
`environment_unavailable` for every page (pins are placeholders, no Docker image pulled) — this is the
correct, honest outcome at the current checkpoint, not a bug.

## Known limitations

* No real Loghi environment has been installed as part of this integration.
* Line-level input is not supported (see capabilities table above).
* The exact container entrypoint/argument contract is unconfirmed against a real image.
* Confidence values depend entirely on whether the specific pinned model/configuration emits PAGE
  `@conf` — many configurations may not.
