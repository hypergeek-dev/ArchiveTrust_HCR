# HTR Domain Design

Phase 4 deliverable. Defines the entity relationships, versioning strategy, and evidence relationships for the HTR Research Center domain model, built on top of the retained Evidence/Observation/Canonical substrate described in `docs/htr-transformation-audit.md`.

Naming convention: Python, `PascalCase` classes as immutable Pydantic models (matching the existing codebase convention), one module per concern under `src/archivetrust/domain/`, `src/archivetrust/htr/`, and `src/archivetrust/research/`.

## 1. Entity map

```
ResearchProject
 └─ Dataset ──────────────── DatasetVersion (immutable snapshot)
     └─ Collection
         └─ Document (an ArchiveObject; retained concept, renamed conceptually to source document)
             └─ Page
                 └─ Region ── (from segmentation)
                     └─ TextLine ── (from segmentation)
                         └─ InputCrop (hash-addressed image bytes; shared across methods)

TranscriptionConvention (versioned, immutable once used)
 └─ GroundTruthItem (page | region | line | word | entity | custom segment)
     ├─ ReviewAssignment (reviewer A, reviewer B — blind)
     │    └─ ReviewSubmission
     ├─ AgreementResult (computed from the two submissions)
     └─ Adjudication (only when AgreementResult requires it)

Method (SATRN | Florence-2 | Transkribus Swedish Lion I)
 └─ ModelVersion
     └─ MethodCapability (per-method declared capability flags)

SegmentationConfiguration ─┐
PipelineConfiguration ─────┼─ referenced by ExperimentVersion
                           │
Experiment
 └─ ExperimentVersion (immutable once ExperimentRun exists)
     └─ ExperimentRun
         └─ MethodRun (one per method x input)
             ├─ RawResult
             ├─ ParsedResult
             ├─ NormalizedResult
             ├─ FailureRecord (if applicable — preserved, never excluded)
             └─ MetricResult (per MetricDefinition)

CanonicalResult
 ├─ pointers to selected MethodRun/segment per selected span
 └─ CanonicalizationStrategy + strategy version

EvidenceRecord — cross-cutting: every stage above emits one, chaining to its inputs
ReproducibilityManifest — one per ExperimentRun
ResearchReport — generated from one or more ExperimentRuns
ExternalImport — Transkribus manual-import provenance wrapper around a MethodRun
```

## 2. Relationship to the retained substrate

The existing `Evidence`/`Observation`/`CanonicalObservation`/`CanonicalDocument` pattern is **not replaced** — it is the implementation mechanism for `EvidenceRecord`/`RawResult`/`ParsedResult`/`NormalizedResult`/`CanonicalResult`:

- `Evidence` gains an HTR extension: new optional fields (`model_revision`, `execution_device`, `execution_time_ms`, `gpu_memory_mb`, `software_environment`, `hardware_environment`) rather than a new incompatible type. Content-addressing (`content_address(...)`) is extended to include `model_revision` and `pipeline_configuration_hash` so line-level HTR results content-address correctly per the brief's "hash of each shared input crop" requirement.
- `Observation` gains new `ObservationType` values: `TEXT_LINE`, `REGION`, `RAW_TRANSCRIPTION`, `PARSED_TRANSCRIPTION`, `NORMALIZED_TRANSCRIPTION`. These replace `HANDWRITTEN_NOTE` (deleted per cleanup inventory) at line granularity instead of paragraph/note granularity.
- `CanonicalObservation`/`CanonicalDocument` are extended with an explicit `source_method_run_id` per selected span and a `CanonicalizationStrategy` enum + version, satisfying "every selected segment must preserve a pointer to its source result."
- The append-only telemetry event schema (`domain/telemetry/events.py`) gains HTR-specific events: `SegmentationRunCompleted`, `MethodRunCompleted`, `ReviewSubmissionRecorded`, `AdjudicationRecorded`, `CanonicalResultCreated` (renaming `CanonicalDocumentCreated` semantics where line-level).

New entities that have **no** analogue in the old system (`ResearchProject`, `Dataset`/`DatasetVersion`, `Experiment`/`ExperimentVersion`/`ExperimentRun`, `MethodRun`, `MetricDefinition`/`MetricResult`, `TranscriptionConvention`, `GroundTruthItem`, `ReviewAssignment`/`ReviewSubmission`/`AgreementResult`/`Adjudication`, `ReproducibilityManifest`, `ExternalImport`) are additive Pydantic models under new packages — they do not require modifying `domain/evidence` or `domain/ontology` beyond the extensions above.

## 3. Versioning strategy

- **`DatasetVersion`**: immutable snapshot of a `Dataset`'s document/page membership at a point in time; new documents create a new `DatasetVersion`, never mutate an existing one. `ExperimentVersion` references a specific `DatasetVersion`, never a mutable `Dataset`.
- **`TranscriptionConvention`**: versioned (`convention_id`, `version`); once any `GroundTruthItem` references a `(convention_id, version)` pair, that pair is frozen — a change creates a new version, and existing ground truth keeps pointing at the old one. Enforced the same way `Evidence` enforces content-addressed immutability: a validator on `GroundTruthItem` construction.
- **`ExperimentVersion`**: an `Experiment` is mutable only until its first `ExperimentRun` is created (mirrors the existing "supersede, don't overwrite" pattern used by `CanonicalObservation`). Any edit after that creates a new `ExperimentVersion`; every `ExperimentRun` points at the exact `ExperimentVersion` used.
- **`CanonicalizationStrategy`**: each strategy implementation carries a `strategy_version`; `CanonicalResult.strategy_version` is stored so a later change to strategy logic doesn't retroactively reinterpret old canonical results.
- **`MetricDefinition`**: versioned like `METRICS_VERSION` already is in `evaluation/metrics.py` — extended, not replaced.

## 4. Evidence relationships (traceability chain)

Matches the brief's required chain exactly, implemented as a linked sequence of content-addressed ids rather than foreign keys (no DB):

```
CanonicalResult.source_method_run_id
  → MethodRun.id
    → NormalizedResult.parsed_result_id → ParsedResult.raw_result_id → RawResult.evidence_id
    → Evidence.id (content-addressed: method + model_revision + pipeline_configuration_hash + input_crop_hash)
      → InputCrop.hash → TextLine.id → Region.id → Page.id → Document.id → Collection.id → Dataset.id (+ DatasetVersion) → ResearchProject.id
```

Every hop is a stored id reference, never an inline copy — this is the same "reference by id, never embed" rule already enforced on `Observation.evidence_ids`.

## 5. Data ownership

- `ResearchProject`/`Dataset`/`DatasetVersion`/`Collection`/`Document`/`Page`/`Region`/`TextLine`/`InputCrop` — owned by a new `src/archivetrust/htr/corpus/` package. `Document`/`Page` reuse the existing `ArchiveObject` acquisition/blob-store mechanism (`acquisition/`, `infrastructure/storage/blob_store.py`) rather than reinventing file storage.
- `Method`/`ModelVersion`/`MethodCapability`/adapters — owned by `src/archivetrust/providers/` (retained package name; contents replaced per cleanup inventory).
- `Experiment*`/`MethodRun`/`MetricDefinition`/`MetricResult`/`FailureRecord`/`ReproducibilityManifest` — owned by a new `src/archivetrust/htr/experiment/` package, orchestrated by an extended `application/pipeline.py`.
- `TranscriptionConvention`/`GroundTruthItem`/`ReviewAssignment`/`ReviewSubmission`/`AgreementResult`/`Adjudication` — owned by `evaluation/` (ground truth) and `review/` (assignment/blind isolation/adjudication), extending existing packages rather than new ones, since the workflow shape is retained.
- `CanonicalResult` — owned by `domain/canonical/` (extends `CanonicalObservation`/`CanonicalDocument`).
- `ExternalImport` — owned by a new `src/archivetrust/providers/transkribus/` adapter package.
- `ResearchReport` — owned by a new `src/archivetrust/research/reports/` package, reading only from telemetry/read-models (existing `research/` package is already read-only by convention).

## 6. Cleanup implications

See `docs/htr-repository-cleanup.md` domain-model table for the specific `ObservationType` deletions (`TABLE_CELL`, `NAMED_ENTITY`, `RELATIONSHIP`, `HANDWRITTEN_NOTE`) this design requires, and the `DeploymentProfile.STRUCTURED_PIPELINE` removal in favor of a generic, independent segmentation stage (§7).

## 7. Segmentation as an independent stage

Per the brief's pipeline requirement, segmentation must not be buried inside one method's adapter. New package `src/archivetrust/htr/segmentation/`:

```python
class SegmentationAdapter(Protocol):
    def detect_regions(self, page_image: PageImage) -> tuple[Region, ...]: ...
    def detect_lines(self, region: Region) -> tuple[TextLine, ...]: ...
    def order_lines(self, lines: tuple[TextLine, ...]) -> tuple[TextLine, ...]: ...
    def crop_lines(self, lines: tuple[TextLine, ...]) -> tuple[InputCrop, ...]: ...
```

A `MethodRun` for a *controlled* experiment consumes `InputCrop`s produced once by a chosen `SegmentationAdapter` and shared byte-identical across all local recognizers (hash-verified — `InputCrop.hash` must match across all `MethodRun`s in a controlled comparison, enforced by an assertion in the experiment runner). A `MethodRun` for an *end-to-end* experiment is allowed to use its own method-preferred segmentation, and the `ExperimentRun` record labels which mode was used, per the brief's "do not present end-to-end comparisons as pure recognizer comparisons."

## 8. Legacy experiment handling

Any `MethodRun` whose `Method` is one of the deleted OCR providers (Docling, Tesseract+LayoutParser, Qwen2.5-VL, PaddleOCR-VL, Surya) and that existed before this transformation is preserved as a **legacy experiment**: readable via `application/journal.py` replay, clearly labeled `legacy: true` in any report or UI surface, and never included in HTR baseline aggregate statistics. No legacy `MethodRun` is deleted — see `docs/htr-migration-plan.md` §4 for the concrete migration step.
