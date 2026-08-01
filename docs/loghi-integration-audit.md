# Loghi Integration Audit

Written before implementation, as the brief requires. Every claim below was verified by reading the
named file, not inferred from its name. Classification vocabulary:

```
Reuse unchanged | Extend | Generalize | Park from active workflow | Replace |
Delete only if obsolete | Requires investigation
```

## 0. A correction this audit exists to record

The brief's "Swedish Lion I" description (external upload, manual PAGE/ALTO download, "ArchiveTrust
never executes the model") matches `providers/transkribus/` (`method_id=transkribus_swedish_lion_1`),
**not** the adapter the user selected for the new active pair. The user selected `providers/
swedish_lion/` (`method_id=swedish_lion`), a real local TrOCR model
(`Riksarkivet/trocr-base-handwritten-hist-swe-2`) added in the commit immediately before this audit,
run in-process via plain `transformers` — no subprocess, no upload, no external queue. It was **not
yet wired into `composition.py::_build_htr_adapters()`** at the time of this audit; wiring it in is
part of this integration, not a pre-existing feature this document is merely describing.

Every section below that touches "Swedish Lion" describes `swedish_lion` unless `transkribus_swedish_lion_1`
is named explicitly. The Transkribus adapter is unmodified and remains available — see §12.

## 1. Method registry / adapter composition

**File:** `src/archivetrust/providers/htr_adapter.py` — `HtrMethodAdapter` Protocol, `MethodMetadata`,
`MethodCapabilities` (7 required booleans + 1 defaulted `image_color_normalization_required`),
`RecognitionInput`/`RecognitionResult`. **Reuse unchanged.** This Protocol is exactly what a Loghi
adapter should implement; nothing about it assumes a single-call recognizer, so a page-level
multi-stage pipeline fits it without modification.

**File:** `src/archivetrust/composition.py:793-826` — `AppContext.htr_method_adapters()` /
`_build_htr_adapters()`, a hardcoded `(module, class)` tuple. **Extend.** Two entries added:
`swedish_lion.adapter.SwedishLionAdapter` (fixing the pre-existing gap) and
`loghi.adapter.LoghiAdapter`. A new `active_htr_method_adapters()` method is added alongside it
(**Extend**, not **Generalize** — the existing unfiltered accessor is kept for historical/overview use,
a second accessor is added rather than changing the one method's meaning under existing callers).

**File:** `src/archivetrust/providers/htr_capability_matrix.py:32-45` — `HTR_ADAPTER_CLASSES`,
deliberately duplicated from `composition.py` with a drift test holding the two in sync. **Extend**
(same two additions, same duplication discipline preserved — importing `composition.py` here would
pull in PySide6-adjacent modules, as its own docstring explains).

**File:** `src/archivetrust/providers/base.py` (`ProviderAdapter`) + `providers/registry.py`
(`ProviderRegistry`). **Reuse unchanged / out of scope.** Legacy interface still load-bearing for
non-HTR providers (`application/pipeline.py`, `acquisition/processing.py`); no HTR method should be
added to it, and nothing here touches it.

**File:** `src/archivetrust/htr/screening/run_configuration.py:81-88` — `RECOGNITION_METHOD_IDS =
(SATRN_METHOD_ID, FLORENCE2_METHOD_ID)`, hardcoded pair, threaded through `runner.py`/`run_state.py`.
**Park from active workflow.** This harness produced the sealed reliability run; it is not generalized
to a variable method set (that would risk mutating the meaning of an already-sealed configuration hash)
and not extended to include Loghi/Lion. It remains runnable as-is for historical reproduction. The new
Lion-vs-Loghi comparison gets its own module (`htr/screening/lion_loghi_experiment.py`) and its own
launcher (`scripts/run_lion_loghi_comparison.py`), following the same precedent
`scripts/run_swedish_lion_comparison.py` already set for not generalizing this harness.

## 2. Adapter isolation pattern

**File:** `src/archivetrust/providers/satrn/facade.py` (`SatrnWorkerFacade` Protocol,
`_find_satrn_python()`, `_SubprocessSatrnWorkerFacade`) + `providers/satrn/_worker.py`. **Reuse as a
pattern, not as code.** This subprocess/isolated-interpreter seam (Protocol, real-vs-fake, explicit
timeout, no interactive shell, JSON-over-stdout) is the direct template for `providers/loghi/facade.py`,
substituting a `docker run` (or `wsl.exe -- docker run`) argv for a Python interpreter path. No shared
code is extracted — the two facades solve the same shape of problem for different backends, and the
codebase's existing convention (SATRN vs. Florence-2 vs. Swedish-Lion facades are each independent) is
to keep each provider's facade self-contained.

## 3. Capability matrix generation

**File:** `src/archivetrust/providers/htr_capability_matrix.py` (generator) +
`docs/CAPABILITY_MATRIX_HTR.md` (generated doc) + `tests/providers/test_htr_capability_matrix.py`
(drift tests) + `scripts/generate_capability_matrix.py` (CLI). **Extend.** The generation architecture
(adapters are the source of truth, generated region delimited by markers, drift test re-renders and
byte-compares) is correct for Loghi and needs no redesign — only: two more adapters in the identity/
capability tables, and one new generated table (research status) sourced from a different function
(`research_status.status_for`), added as its own section so status and capability stay visibly
separate per the brief.

## 4. Active-method selection — no status concept exists

Confirmed by search: no `MethodStatus`/`is_active`/`ACTIVE` enum scoped to HTR methods anywhere in
`src/archivetrust`. Every consumer of `htr_method_adapters()` — `experiment_builder_viewmodel()`
(`composition.py:856-874`), `method_overview_viewmodel()` (`composition.py:839-845`),
`first_launch_viewmodel()` (`composition.py:943-951`) — receives the full, unfiltered list. **Requires
new code**, not extension of an existing enum: `htr/research_status.py` (§ below) is new.
`FirstLaunchViewModel.htr_method_readiness()` reports environment *readiness*, a materially different
concept from research *status*, and is **left unchanged** — a method can be ready-to-run and still be
archived from the current phase (SATRN's environment might validate fine; it is archived anyway).

## 5. Experiment / ExperimentRun model

**File:** `src/archivetrust/htr/experiment/models.py`. **Extend.** `Experiment` → `ExperimentVersion`
→ `ExperimentRun` → `MethodRun` is reused unchanged as the shape for all four Lion/Loghi cells — no
redesign needed, the model already supports "one `ExperimentRun` per configuration." Additions: three
optional fields on `ExperimentVersion` (`corpus_language`, `method_primary_language_domain`,
`domain_relationship`) and a new `ExperimentComparisonGroup` type (parent-grouping id for the four
cells, modeled on `htr/corpus/models.py::Collection`'s "named group of related things, referenced by
id" shape). Both additions are optional/new-model, so every serialized `ExperimentVersion` from the
sealed benchmark still deserializes unchanged — nothing is renamed or required-ified.

## 6. Prior benchmark storage — what must not be touched

`docs/experiments/technical-reliability-screening/full-run/reliability-2026-07-31/` (events.jsonl +
hash-chain sidecar + report + crops) and `.../reliability-2026-07-31-swedish-lion/` (the
already-completed Swedish-Lion-Libre comparison against the same crops). **Reuse unchanged — read-only
historical evidence.** Nothing in this integration writes to, re-derives, or re-hashes these
directories. `docs/research-findings.md`'s five findings and their `htr/knowledge/models.py` types are
likewise **reuse unchanged**.

## 7. Dataset structure

`dataset-rgb/` — 31 collection subdirectories, flat page images, Swedish witch-trial records
(1584–1764). **Reuse unchanged, documented explicitly as the Swedish corpus** (not renamed). Its
manifest/inventory (`docs/experiments/technical-reliability-screening/dataset-manifest.json`,
`dataset-inventory.csv`) stays as the sealed run's own record; the new comparison does not reuse those
specific files (different sampling scope) but follows their JSON/CSV shape for the new
`dataset-dutch-rgb-manifest.json`/inventory.

`dataset-dutch-rgb/` — **new.** Populated from `D:\Downloads\republic7.zip` (515 pages, `train`/`val`
split, PAGE-XML ground truth, Nationaal Archief scans — see
`docs/experiments/lion-loghi-comparison/dataset-provenance-dutch.md` for the full provenance record).
Not auto-selected or downloaded — a real file the user pointed to, extracted with its provenance
recorded, not silently assumed licensed for redistribution.

## 8. RGB normalization

`src/archivetrust/htr/preprocessing/{rgb_normalization.py,models.py,normalization_service.py,
export_package.py}`. **Reuse unchanged.** Version, config, and the AST-scanned no-enhancement
contract (`tests/htr/preprocessing/test_no_enhancement_contract.py`) apply identically to Dutch pages —
nothing about the pipeline is Swedish-specific. Both Lion and Loghi consume the same normalized
derivative per the "shared page-input policy"; Loghi's own internal preprocessing (inside its
containers) is recorded as part of the Loghi pipeline's stage results, never substituted for or
compared against ArchiveTrust's own normalization step.

## 9. Transkribus export/import and PAGE XML handling

`src/archivetrust/providers/transkribus/{page_xml.py,alto_xml.py,parsing.py,external_import.py,
adapter.py}`. **Reuse unchanged** as the Transkribus workflow (now `INACTIVE`, not archived — see §12);
**Extend by pattern, not by import**, for Loghi: `providers/loghi/page_xml.py` is a new, Loghi-specific
parser returning its own `LoghiPageParseResult` type, reusing only the generic low-level PAGE-schema
element helpers (Coords/Baseline/TextLine parsing) where the standard itself, not any Transkribus-
specific convention, is what's being read. The two parse-result types are never merged — different
tools' exports are not guaranteed field-identical.

## 10. Segmentation abstractions

`src/archivetrust/htr/segmentation/` (`Florence2LineDetectorAdapter`, `SegmentationService`).
**Reuse unchanged, not extended to Loghi.** Loghi performs its own layout analysis and line extraction
inside its own containers (Laypa, Loghi Tooling) — that is a pipeline-internal stage recorded on
`LoghiPipelineResult`, not a call into ArchiveTrust's own `SegmentationService`. Reusing that service
for Loghi would misrepresent an external tool's segmentation as ArchiveTrust's own.

## 11. Method-run persistence, telemetry, reproducibility manifests

`src/archivetrust/htr/persistence/durable_store.py` (`DurableHtrResearchStore`, ~40 `register_*`/
`record_*` methods, single write path). **Extend** — new `record_loghi_*` / `record_method_research_status_changed`
/ `record_domain_relationship` methods added following the exact `record_normalization_started/
completed/failed` triad pattern already established. `domain/telemetry/events.py`
(`TelemetryEventKind`, `TelemetryEvent`). **Extend** — 9 new kinds (§ telemetry section of the plan),
same rationale-docstring discipline as the existing 74. `htr/experiment/models.py::ReproducibilityManifest`.
**Reuse unchanged** — its free-form `software_environment`/`hardware_environment` dicts already have
room for `LoghiComponentVersions`/`LoghiEnvironmentReport` without a schema change.

## 12. Research observations, findings, report generation

`htr/knowledge/{models.py,baseline_knowledge.py,satrn_repetition_knowledge.py}`,
`docs/research-findings.md`, `docs/research-observations.md`. **Reuse unchanged.** No existing finding
is edited, superseded, or reinterpreted by this integration. New findings about Lion-vs-Loghi are
explicitly **not** created yet (brief: no fabricated results) — the four `ExperimentVersion`s this
integration builds carry no results until real runs happen.

`htr/screening/run_report.py::write_report()`. **Reuse unchanged as a pattern**; the Lion-vs-Loghi
launcher composes it directly rather than generalizing it, matching `run_swedish_lion_comparison.py`'s
precedent (§1).

## 13. Current Docker / Windows / external-process support

No Docker or WSL invocation exists anywhere in `src/` or `scripts/` today (confirmed by search).
**Requires investigation → resolved as: new code, SATRN-facade-patterned (§2).** Windows support for
Loghi specifically means **not** claiming native Windows execution — Loghi's own tooling assumes Bash/
Linux paths/NVIDIA container tooling, so the adapter documents and probes for exactly one of: Docker on
Linux, Docker through WSL2, or native Linux. This machine (checked directly): Docker Desktop 29.6.1
installed, a `docker-desktop` WSL2 distro present but **stopped**, no Loghi images/repos/checkpoints
pulled. `validate_environment()` reports this precise state, never a blanket unavailable and never a
false "ready."

## Summary table

| Component | Classification |
|---|---|
| `HtrMethodAdapter` Protocol | Reuse unchanged |
| `composition.py` adapter tuple | Extend |
| `htr_capability_matrix.py` | Extend |
| Legacy `ProviderAdapter`/`registry.py` | Reuse unchanged (out of scope) |
| `run_configuration.py` / reliability harness | Park from active workflow |
| SATRN facade pattern | Reuse as pattern (not code) |
| Research-status concept | New (did not exist) |
| `FirstLaunchViewModel` readiness | Reuse unchanged |
| `htr/experiment/models.py` | Extend |
| Sealed reliability-run evidence | Reuse unchanged (read-only) |
| `dataset-rgb/` | Reuse unchanged, documented as Swedish corpus |
| `dataset-dutch-rgb/` | New, populated from provided archive |
| RGB normalization pipeline | Reuse unchanged |
| Transkribus PAGE/ALTO parsing | Reuse unchanged; Loghi gets its own parser by pattern |
| Segmentation service (Florence-2 detector) | Reuse unchanged, not extended to Loghi |
| `DurableHtrResearchStore` | Extend |
| `TelemetryEventKind` | Extend |
| `ReproducibilityManifest` | Reuse unchanged |
| Findings / research-knowledge | Reuse unchanged |
| Docker/WSL support | New (none existed) |
| Windows native Loghi execution | Not supported — explicitly documented boundary |
