# HTR Transformation Audit

Status: Phase 1 deliverable of the HTR Research Center transformation. Written 2026-07-29 from direct inspection of the repository at this commit (`0951e91`, "Initial commit"). Supersedes nothing yet — this is the first HTR-transformation document.

## 1. What this repository actually is today

ArchiveTrust is **not** a lightweight OCR-comparison demo. It is a mature, ~evidence-first, provider-agnostic document-understanding trust layer:

- **Pure Python 3.11+**, no JavaScript/web frontend anywhere. The UI is a **PySide6/Qt desktop app** (`src/archivetrust/clients/desktop_v2/`, the one supported client; `clients/desktop/` is an older MVVM client kept for reference).
- **No database.** Persistence is append-only JSON Lines telemetry (`infrastructure/storage/telemetry_sink.py`), a content-addressed SHA-256 blob store, and hash-chain integrity manifests. "Current state" is a projection replayed from telemetry, not queried from a DB.
- Hard runtime dependency: `pydantic` only. Every provider/ML library (`docling`, `pytesseract`, `transformers`, `paddleocr`, vLLM client) is an optional extra, lazily imported and runtime-probed.
- The core domain separation — **Evidence → Observation → Comparison → CanonicalObservation → CanonicalDocument**, all content-addressed, append-only, supersede-not-overwrite — is already exactly the "evidence-first" philosophy this transformation is asked to preserve. This is the strongest asset in the repo and the reason the transformation should be a **redesign around HTR concepts on top of this substrate**, not a rewrite of the substrate itself.
- A governance document, `ARCHITECTURAL_CONSTITUTION.md`, states invariants ("Articles") that are cited by number throughout docstrings and tests. Any domain redesign must either satisfy these invariants or explicitly amend the constitution — this is not a green-field repo.

This materially changes how the task brief's frontend/database instructions should be read: "frontend" in this repo means the Qt desktop client and its ViewModel layer (`presentation/`), not a web app; "database cleanup" means the domain model (Pydantic entities + telemetry event schemas + workspace JSON layout), not SQL migrations.

## 2. Current pipeline / data flow

```
Archive Object (immutable) → Provider Output → Evidence → Observation → Comparison
    → Canonical Observation → Canonical Document
```
```
Desktop v2 → durable worker command → local worker process → providers
     |                                     |
     +-- current-state / read models <- append-only telemetry + blob store
                        |
                        +-- operational review -> superseding observation/document
                        +-- JSON / PAGE / ALTO / METS release export
                        +-- replay and diagnostics

Separate: workspace/evaluation -> blinded assignments -> two reviews -> adjudication
```

There is already: a comparison/reconciliation engine (Phases A–F), a confidence engine, a human-review subsystem with triage/sampling/closure, and an evaluation subsystem with a blinded double-annotation + adjudication ground-truth store and CER/WER metrics. These map closely onto the HTR brief's "blind dual review," "adjudication," and "recognition metrics" requirements — they need to be *extended to line-level HTR granularity*, not invented from scratch.

## 3. Provider adapters — current state

Interface: `ProviderAdapter` (`src/archivetrust/providers/base.py`), with `DeterministicProviderAdapter` / `ProbabilisticProviderAdapter` subclasses. Single method: `observe(*, document_ref, invocation_id, source) -> ProviderRunResult`. `ProviderRunResult` = evidence tuple + observations tuple + rejections tuple + failure_reason. This shape is close to the brief's `HtrMethodAdapter` interface conceptually (metadata/capabilities/health-check are missing today and must be added).

Implemented providers, all page/paragraph-granularity general-document OCR:

| Provider | Status | Notes |
|---|---|---|
| Docling | implemented | native `handwritten_text` label → `HandwrittenNotePayload` |
| Tesseract + LayoutParser (combined) | implemented | near-garbage on cursive handwriting per own validation report |
| Qwen2.5-VL | implemented | probabilistic VLM via Transformers runtime |
| PaddleOCR-VL (raw) + PaddleOCR-VL Structured Pipeline | implemented | two deployment profiles of one vendor family |
| Surya | implemented | via vLLM client |
| DeepSeek OCR, LayoutParser standalone, Donut, Document AI | **not present** — mentioned only as illustrative names in docs |

None of these are Swedish-historical-HTR-capable; none produce line-level segmentation output; the "handwritten" concept that exists (`HANDWRITTEN_NOTE`) is a paragraph-level Docling label, not a recognition result. **All five provider adapters are deletion/archival candidates** under the "does not merely rename OCR providers" instruction — none map onto SATRN/Florence-2/Transkribus semantics.

## 4. Domain model gaps for HTR

Present and reusable as-is or with light extension:
- `Evidence` (content-addressed raw capture) — reusable, needs a line-level granularity option and HTR-specific metadata fields (device, execution time, GPU memory, etc. — currently no such fields on any Evidence/Observation).
- `Observation` / ontology types — the enum-based `ObservationType` approach is reusable; needs new types, not a new pattern.
- `CanonicalObservation` / `CanonicalDocument` — reusable pattern; canonicalization strategy is currently implicit/single-strategy, needs the 5 named strategies from the brief plus per-segment source pointers (partially present via evidence_ids, needs to be surfaced explicitly).
- CER/WER (`evaluation/metrics.py`) — directly reusable, already provider-agnostic and versioned.
- Ground-truth blinded double-annotation + adjudication (`evaluation/ground_truth.py`) — directly reusable pattern, needs a `TranscriptionConvention` concept and line/region granularity added.
- Review triage/sampling/closure (`review/`) — reusable workflow shape, needs HTR-specific review packet content (line crops, not page-level claims).

Missing entirely (must be built new, not renamed):
- Any line/region/segmentation-first primitive (`TextLine`, `Region`, `InputCrop`) — today the smallest geometric unit is a single `BoundingBox` on `Evidence`.
- `ResearchProject`, `Dataset`/`DatasetVersion`, `Experiment`/`ExperimentVersion`/`ExperimentRun`, `MethodRun`, `MetricDefinition`/`MetricResult`, `ReproducibilityManifest`, `FailureRecord` as first-class entities.
- Segmentation as an independent pipeline stage — today segmentation (where it exists, e.g. PaddleOCR structured pipeline's detector/cropper) is buried inside one provider's adapter, not a shared stage other methods can reuse or be evaluated against independently.

OCR-specific concepts that do **not** map to HTR and should be removed rather than repurposed: `TABLE_CELL`/table reconciliation, `NAMED_ENTITY`/`RELATIONSHIP`, `DeploymentProfile.STRUCTURED_PIPELINE` as currently modeled (too PaddleOCR-specific), `BoundingBox.Precision.COARSE_ESTIMATE` semantics tied to page-level layout detection.

## 5. Documentation state

Root has 15 `.md` files. Live/authoritative: `README.md`, `ARCHITECTURAL_CONSTITUTION.md`, `AGENTS.md`, `CONTRIBUTING.md`. Historical/superseded planning snapshots (keep as archive, stop treating as current): `ROADMAP.md`, `ROADMAP_V2.md`, `MILESTONE0_REVIEW.md`, `MILESTONE1_DOMAIN_MODEL.md`, `MILESTONE4_COMPARISON_ENGINE.md`, `PROPOSAL_REVIEW_CANONICAL_CHUNK.md`, `IMPLEMENTATION_STATUS.md`, `ARCHITECTURE_CHANGELOG.md`, `ARCHITECTURE_READINESS_REVIEW.md`, `ARCHITECTURE_VALIDATION_REPORT.md`, `HUMAN_REVIEW_SPECIFICATION.md` (spec content reusable, needs HTR rewrite).

`docs/` (~100+ files) is dominated by dated phase/workstream audit trail (`PHASE_14…` through `PHASE_35…`, `A1`–`F5` roadmaps, production-incident reports) — this is provenance, not living reference documentation. `docs/CURRENT_SYSTEM.md` is the one doc that self-labels as current authority.

`artifacts/` is ~3.3 GB of generated benchmark/qualification-run output — reproducible, not evidence with unique research value, and not part of the HTR research center. `"For Chat-GPT project folder/"` is a git-ignored external context mirror, not source of truth.

## 6. Recommended classification (see also `docs/htr-repository-cleanup.md`)

| Component | Classification | Reason |
|---|---|---|
| `domain/evidence`, `domain/ontology` (base pattern), `domain/canonical`, `domain/telemetry`, `infrastructure/storage/*` | Retain, extend | Evidence-first substrate is exactly what the brief asks to preserve |
| `domain/comparison/*`, `domain/confidence/*`, `domain/alignment/*` | Reuse with modification | Reconciliation logic needs line-level + explicit strategy support, not a rewrite |
| `evaluation/metrics.py`, `evaluation/ground_truth.py`, `evaluation/workflow.py` | Reuse with modification | Add TranscriptionConvention + line granularity |
| `review/*` | Reuse with modification | Add blind-review isolation semantics, line-crop packets |
| `providers/docling`, `providers/tesseract_layoutparser`, `providers/qwen_vl`, `providers/paddleocr_vl*`, `providers/surya`, `runtime/vllm_runtime.py`, `runtime/pp_doclayout_runtime.py` | **Delete** (export as legacy fixtures where tests rely on them) | Not HTR methods, don't map onto SATRN/Florence-2/Transkribus |
| `domain/comparison/table_reconciliation.py`, `ObservationType.TABLE_CELL/NAMED_ENTITY/RELATIONSHIP` | Delete | Semantics don't fit HTR |
| `benchmark/`, `artifacts/` | Archive outside active app / delete generated output | Reproducible benchmark output, not HTR research evidence |
| Dated `docs/PHASE_*`, `docs/A*-F*` roadmaps | Archive to `docs/archive/` | Historical audit trail, not living docs |
| `ROADMAP*.md`, `MILESTONE*.md`, `IMPLEMENTATION_STATUS.md`, `ARCHITECTURE_CHANGELOG.md`, `ARCHITECTURE_READINESS_REVIEW.md`, `ARCHITECTURE_VALIDATION_REPORT.md`, `PROPOSAL_REVIEW_CANONICAL_CHUNK.md` | Archive to `docs/archive/` | Superseded planning snapshots |
| `clients/desktop/` (legacy Qt client) | Investigate then likely delete | README already says desktop_v2 is the only supported client |
| `"For Chat-GPT project folder/"` | Delete or leave untouched (git-ignored, not part of app) | Not source of truth, out of scope |

## 7. Major risks

1. **Scale.** This is a governed, tested (228 test files), invariant-enforcing codebase, not a prototype. A full 14-phase rewrite in one sitting risks silently violating `ARCHITECTURAL_CONSTITUTION.md` invariants or destroying test coverage faster than it can be replaced.
2. **No web frontend exists.** The brief's UI section (dashboard, comparison view, evidence view, review center) must be re-targeted at the Qt desktop client's ViewModel/pages architecture, not built as a new web app — doing the latter would contradict "keep the implementation modular" / "prefer coherent architecture" guidance and massively balloon scope.
3. **External model dependencies.** SATRN (`Riksarkivet/satrn_htr`) and the Florence-2 pipeline (`vlm-htr`) require downloading real model weights and a GPU-capable environment to validate end-to-end; this sandbox may not have GPU/network access to actually run inference. Adapter code, contracts, and tests-against-fixtures can be built and validated without real inference; real-model integration tests should be marked accordingly.
4. **Irreversible deletion of a mature system.** Several components slated for deletion (five provider adapters, table/NER ontology types, legacy desktop client) are large, tested, working subsystems. Given the size of the blast radius, this should proceed as a staged, reviewable sequence of commits rather than one large destructive commit.

## 8. Recommended implementation order

1. `docs/htr-repository-cleanup.md` (classification detail, this doc's companion) — done alongside this audit.
2. `docs/htr-domain-design.md` + `docs/htr-migration-plan.md` (Phase 4).
3. Core HTR abstractions (Phase 5): `HtrMethodAdapter`, `TextLine`/`Region`/`InputCrop`, experiment/run entities, reproducibility manifest — additive, does not require deleting anything yet.
4. Remove replaced OCR provider architecture (Phase 6) once the new core compiles and has adapter contract tests.
5. SATRN adapter (simplest local baseline) → Florence-2 adapter → Transkribus manual-import adapter.
6. Evaluation engine extensions, review workflow extensions.
7. Qt desktop client re-target (research dashboard, comparison/evidence views).
8. Baseline experiment + final cleanup pass.

This order is chosen so the repository is never left in a state where it fails to build/import: additive core work lands before deletions, and deletions land only after replacement functionality has adapter-contract test coverage.
