# HTR Migration Plan

Phase 4 deliverable. Concrete, staged sequence for landing `docs/htr-domain-design.md` on top of the audited repository, chosen so the repo never fails to import/build between steps.

## Staging principle

Additive work lands before deletions. Each stage ends with: `pytest --collect-only` succeeds, `import archivetrust` succeeds, and any new adapter/entity has at least a contract test. This mirrors the audit's recommended implementation order (§8).

## Stage-by-stage plan

1. **Core HTR entities (additive)** — add new Pydantic models (`ResearchProject`, `Dataset`/`DatasetVersion`, `Region`, `TextLine`, `InputCrop`, `Experiment`/`ExperimentVersion`/`ExperimentRun`, `MethodRun`, `MetricDefinition`/`MetricResult`, `FailureRecord`, `TranscriptionConvention`, `GroundTruthItem` extensions, `ReproducibilityManifest`) under the new packages listed in the domain design's §5 data-ownership table. Nothing existing is modified yet. Risk: none (purely additive).
2. **`HtrMethodAdapter` interface (additive)** — add `providers/htr_adapter.py` alongside the existing `ProviderAdapter` (both exist temporarily). New `ObservationType` values added alongside old ones (both exist temporarily). Risk: low.
3. **Extend `Evidence`/telemetry (in place, backward-compatible)** — add the new optional fields to `Evidence` and the new telemetry event types. Existing events/fields are untouched, so all 1218 existing tests keep passing unmodified. Risk: low; validated by running the full existing test suite before proceeding.
4. **Legacy experiment tagging** — before any provider deletion, run a one-time script (`scripts/migrate_legacy_method_runs.py`, deleted after use) that walks existing telemetry for the five OCR providers and confirms every resulting canonical/telemetry record round-trips through `application/journal.py` replay unchanged, then tags them `legacy: true` in a manifest under `docs/archive/legacy-experiments-manifest.json`. This is the "protect evidence before deletion" step (audit Phase 2) made concrete. No data is deleted at this stage — only confirmed replayable and manifested.
5. **Remove replaced provider architecture** — delete `providers/docling/`, `providers/tesseract_layoutparser/`, `providers/qwen_vl/`, `providers/paddleocr_vl/`, `providers/paddleocr_vl_structured/`, `providers/surya/`, `runtime/vllm_runtime.py`, `runtime/pp_doclayout_runtime.py`, delete `ObservationType.TABLE_CELL`/`NAMED_ENTITY`/`RELATIONSHIP`/`HANDWRITTEN_NOTE` and their payload/reconciliation code, delete the old `ProviderAdapter`/`ProviderRunResult` interface (now fully superseded by `HtrMethodAdapter`), remove the corresponding `pyproject.toml` extras (`docling`, `tesseract`, `vllm`, `paddle-layout`), delete their test suites per file (not skip). Relocate `clients/desktop/composition.py::AppContext` to `src/archivetrust/composition.py` and update its two import sites (`admin/qualification.py`, `worker/service.py`) before deleting the rest of `clients/desktop/`. Risk: high blast radius — done as one reviewable commit, full test suite run immediately after.
6. **SATRN adapter** — first real `HtrMethodAdapter` implementation, local/deterministic, no external service dependency. Validates the new interface end-to-end before the architecturally different Florence-2 pipeline is built on top of it.
7. **Florence-2 adapter** — built against the same `HtrMethodAdapter` interface but permitted (per the brief) to diverge in pipeline shape internally; its own segmentation/prompting stages live inside `providers/florence2_htr/` rather than being forced through SATRN's exact flow.
8. **Transkribus manual-import adapter** — `ExternalImport` + PAGE/ALTO XML parsers; no network call is ever made without explicit user action (enforced by adapter design: `recognize()` for this adapter requires a pre-supplied export file path, there is no auto-fetch code path).
9. **Evaluation engine extension** — add segmentation metrics, historical-feature evaluator plugin interface, failure classification, on top of the retained CER/WER primitives.
10. **Review workflow extension** — add blind-isolation enforcement (a `ReviewSubmission` write is rejected if the same `GroundTruthItem`'s other assignment isn't yet closed and the reviewer requests to view it) and `AgreementResult`/`Adjudication` on top of the retained triage/sampling/closure shape.
11. **Desktop client (v2) rework** — add research dashboard, method overview, dataset explorer, experiment builder, comparison view, evidence view, review center pages to `clients/desktop_v2/pages.py`, backed by new ViewModels in `presentation/`. Remove OCR-provider-specific pages/cards.
12. **Baseline experiment** — the "Swedish Historical HTR Baseline Comparison" template, executed against a small representative fixture set.
13. **Final cleanup pass** — remaining dead code/doc sweep, dependency lockfile regeneration, terminology grep per the audit's obsolete-term list.

## Compatibility strategy

No compatibility shim is introduced between the old `ProviderAdapter` and new `HtrMethodAdapter` interfaces — stage 5 deletes the old interface outright once stage 6 proves the new one works, per "prefer coherent architecture over superficial backward compatibility." The only backward-compatibility concern honored is **legacy evidence readability** (stage 4's manifest + journal replay guarantee), not API compatibility.

## Rollback

Each stage is a separate commit. If a stage's post-condition (import + collect + targeted tests) fails, the fix happens within that stage's commit scope before moving on — no stage is allowed to leave the tree in a broken state for a later stage to fix.
