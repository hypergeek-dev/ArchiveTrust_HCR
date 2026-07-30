# HTR Repository Cleanup Inventory

Companion to `docs/htr-transformation-audit.md`. Tracks every substantial deletion/replacement/archival decision made during the HTR transformation, per the classification scheme: **Retain unchanged / Reuse with modification / Replace / Archive outside active application / Delete / Requires investigation**.

This document is updated as each cleanup phase executes; entries are not retroactively removed even after deletion, so it remains a durable record of what happened and why.

## Final state summary (Stage 13 / Phase 14 closing pass)

The transformation is complete through Migration Plan Stage 13. Final measurements:

- **Tests**: 1345 passed, 2 skipped (started at 1218 collected at the pre-transformation baseline;
  net growth reflects real new HTR functionality — SATRN, Florence-2, Transkribus adapters,
  evaluation engine, blind review, research-interface ViewModels, baseline experiment — outweighing
  the ~280 tests deleted for the five removed OCR providers).
- **`docs/` root**: reduced from ~100 active dated audit/roadmap files to 16 active files (12
  pre-existing "authority" docs, now annotated with HTR-transformation supersession notices where
  their content is stale, plus the 4 new `htr-*.md` planning documents) + `docs/archive/` (101
  files, all superseded planning/audit/investigation documents from the OCR-era project, preserved
  as historical record) + `docs/experiments/baseline-comparison/` (a real executed example).
- **Root `.md` files**: reduced from 15 to 5 (`README.md`, `ARCHITECTURAL_CONSTITUTION.md`,
  `AGENTS.md`, `CONTRIBUTING.md`, `HUMAN_REVIEW_SPECIFICATION.md`) — 10 archived to `docs/archive/`.
- **Deleted entirely** (not archived — reproducible or irrelevant): `artifacts/` (~3.3GB generated
  benchmark output), `qualification_runs/`, `run_profiles/`, `benchmarks/`, `test-artifacts/`,
  `audit-artifacts/`, `tmp/`, `.codex_tmp/`, three ad-hoc personal scripts operating on an external
  non-repo corpus (`Analyzer.py`, `picker.py`, `pdf_analysis.csv`), `qualification_run_config.json`,
  `"For Chat-GPT project folder/"`, five OCR provider packages and their tests, two runtime modules,
  two HTML/labeled-block decoders, table/NER/handwritten-note ontology types and their payloads,
  four unused `pyproject.toml` extras (`docling`, `tesseract`, `vllm`, `paddle-layout`), and a leftover
  `surya-ocr` pip package (installed in the environment but no longer declared anywhere — removed
  to fix a real `pip check` conflict with the `transformers` pin Florence-2 requires).
- **`pip check`**: clean (no broken requirements) as of this pass.
- **Terminology sweep**: obsolete-provider-name references remaining in active code are, on
  sampling, legitimate historical citations (explaining *why* a deleted type/module was removed,
  e.g. `providers/base.py`'s "PaddleOCR-VL Structured's now-deleted two-stage profile") or
  still-functional legacy-compatibility code deliberately retained in Stage 5 (`providers/base.py`,
  `providers/registry.py`, `domain/comparison/capability_matrix_data.py` — load-bearing for
  `application/pipeline.py`'s still-live legacy path, per that stage's documented judgment call).
  Not exhaustively rewritten file-by-file in this pass; flagged as a residual item below.
- **Repository is smaller and more focused** than the pre-transformation state per the brief's
  explicit goal, while every deletion is recorded in this document with its reason.

### Residual gaps honestly carried forward (not closed in this pass)

- `presentation/first_launch_viewmodel.py`'s `VISION_PROVIDER_CHOICES` still lists three deleted
  providers in one first-run wizard flow (flagged during Stage 11, not yet redesigned).
- `docs/CAPABILITY_MATRIX.md` and `docs/EVALUATION_PROTOCOL.md` were annotated with supersession
  notices rather than fully rewritten for SATRN/Florence-2/Transkribus; a from-scratch HTR capability
  matrix document does not yet exist as a standalone artifact (the adapters' own `get_capabilities()`
  plus their READMEs are the current source of truth).
- `ARCHITECTURAL_CONSTITUTION.md`'s per-article provenance citations (e.g. "*Sources: ROADMAP.md
  §5.9*") still name files by their pre-archival root-level names; those files now live under
  `docs/archive/` with the same filenames, so the citations are stale paths but not incorrect
  content — left as-is rather than mechanically rewriting a governance document's citation style
  in a cleanup pass.
- No production dependency lockfile exists in this project's convention (pip/setuptools,
  `pyproject.toml`-only) — there was nothing to regenerate.

### Post-closure amendment: `docs/archive/` deleted outright

After this Phase 14 pass completed, the user reviewed the result and explicitly requested that
`docs/archive/` (the ~101 archived pre-HTR planning/audit/investigation documents referenced
throughout this document's "Archive → `docs/archive/`" rows above) be **deleted outright** rather
than retained, on the grounds that it was unnecessary bulk cluttering the repository. This was
confirmed with the user as irreversible before acting (this repository has no prior git history to
recover from — nothing had ever been committed). Executed: `docs/archive/` and its contents
removed; `README.md`'s reference to it corrected. The table rows above that say "Archive →
`docs/archive/`" are left unchanged as an accurate historical record of the *original* Stage 5/13
classification decision — they describe what was decided and done at the time, not the final
state. The final state is: those source documents no longer exist anywhere in this repository.
Also deleted in this same pass, as generated/regenerable clutter unrelated to the archive decision:
`archivetrust_data/` (default runtime data directory — config/cache/telemetry/logs written by the
app and integration tests) and `test-artifacts/` (screenshots written by
`tests/clients/test_desktop_v2_interaction_audit.py`); both added to `.gitignore` so they stop
reappearing as untracked clutter.

## Root-level documentation

| Item | Previous purpose | Classification | Reason | Replacement | Deletion risk | Evidence value |
|---|---|---|---|---|---|---|
| `ROADMAP.md`, `ROADMAP_V2.md` | Forward planning for the general OCR-comparison platform | Archive → `docs/archive/` | Superseded by execution; still cited by docstring provenance comments | `docs/htr-migration-plan.md` | Low — historical record only | Keep as archive for docstring citation trail |
| `MILESTONE0_REVIEW.md`, `MILESTONE1_DOMAIN_MODEL.md`, `MILESTONE4_COMPARISON_ENGINE.md` | Design records for the Evidence/Observation/Comparison substrate | Archive → `docs/archive/` | Describes a process already implemented; the substrate itself is retained, the planning narrative is not current | `docs/htr-domain-design.md` | Low | High — explains why the retained substrate looks the way it does |
| `PROPOSAL_REVIEW_CANONICAL_CHUNK.md` | Rejected/accepted ontology-type proposals (incl. discussion of Handwritten Note vs Signature/Stamp) | Archive → `docs/archive/` | Point-in-time proposal review, decisions already absorbed into code | n/a | Low | Medium — explains a design decision |
| `IMPLEMENTATION_STATUS.md` | Chronological implementation log of the OCR platform | Archive → `docs/archive/` | Superseded as "current state" doc by `docs/CURRENT_SYSTEM.md`; too large/stale to keep live | `docs/CURRENT_SYSTEM.md` (to be rewritten for HTR) | Low | Medium — historical audit trail |
| `ARCHITECTURE_CHANGELOG.md`, `ARCHITECTURE_READINESS_REVIEW.md`, `ARCHITECTURE_VALIDATION_REPORT.md` | Point-in-time architecture review artifacts | Archive → `docs/archive/` | Completed review snapshots for the OCR-era architecture | n/a | Low | Medium |
| `HUMAN_REVIEW_SPECIFICATION.md` | Spec for the general-document review workflow | Reuse with modification | Review workflow shape (triage/sampling/closure/adjudication) is reusable; content must be rewritten for line-level HTR review + blind dual review semantics | New `docs/review/` spec | Medium — active spec, must not leave stale copy alongside new one | High — captures a working spec |
| `ARCHITECTURAL_CONSTITUTION.md` | Governance invariants | Retain, amend in place | Invariants (evidence-first, append-only, supersede-not-overwrite, provider independence) still apply to HTR; add HTR-specific articles rather than replacing the document | n/a | High if violated silently | Governs all other decisions |
| `README.md` | Project overview | Replace (rewrite) | Must describe HTR Research Center purpose per task brief | New content, same file | Low | n/a |
| `AGENTS.md`, `CONTRIBUTING.md`, `LICENSE` | Contributor/agent instructions, license | Retain unchanged | Not OCR-specific | n/a | None | n/a |
| `"For Chat-GPT project folder/"` | Git-ignored external context mirror for a prior oversight process | Delete | Not source of truth, not part of the application, mirrors docs being archived anyway | n/a | None (git-ignored, external mirror) | None — mirror, not original |

## `docs/` directory

| Item | Classification | Reason |
|---|---|---|
| `docs/PHASE_14…` through `PHASE_35…`, workstream roadmaps `A1`–`F5`, production-incident reports | Archive → `docs/archive/` | Dated sequential audit trail for the OCR-era production-readiness effort; valuable provenance, not living reference |
| `docs/CURRENT_SYSTEM.md` | Replace (rewrite for HTR) | Must describe the HTR-era architecture as current authority |
| `docs/CAPABILITY_MATRIX.md`, `docs/EVALUATION_PROTOCOL.md` | Replace | Capability matrix must be rebuilt for SATRN/Florence-2/Transkribus capabilities; evaluation protocol must describe HTR metrics |
| `docs/OPERATIONS_RUNBOOK.md`, `docs/SECURITY_AND_DATA_HANDLING.md`, `docs/DATA_HANDLING_POLICY.md` | Reuse with modification | Operational/security concerns (external upload consent, no hardcoded credentials) apply directly to Transkribus integration; needs an HTR-specific addendum, not a rewrite |
| `docs/ARCHITECTURE_TELEMETRY_STANDARD.md`, `docs/TELEMETRY_STANDARD_V1.md`, PROV/OpenLineage mapping doc | Retain, extend | Telemetry schema is provider-agnostic; add HTR-specific event types rather than replacing |
| `docs/investigation/` (Milestone-0 foundational design docs, `PROVIDER_ANALYSIS.md` etc.) | Archive → `docs/archive/` | Analyzes the five general OCR providers being deleted; historical rationale only |
| `docs/archive/` (existing) | Retain, append | Already the designated home for retired docs |

## Providers / runtime (source code)

| Item | Previous purpose | Classification | Reason | Replacement | Dependencies | Migration implications | Executed |
|---|---|---|---|---|---|---|---|
| `providers/docling/` | General-document layout+text extraction | Delete | Not an HTR method; `handwritten_text` label is paragraph-level, not line-level recognition | `providers/satrn/`, `providers/florence2_htr/`, `providers/transkribus/` | `docling` optional extra (removed from `pyproject.toml`) | `ObservationType.HANDWRITTEN_NOTE` payload mapping removed with it — see domain model row below | — **EXECUTED (Stage 5)**: `src/archivetrust/providers/docling/` and `tests/providers/docling/` deleted. |
| `providers/tesseract_layoutparser/` | General OCR + layout detection | Delete | Own validation report documents near-garbage cursive handwriting output; not a research-relevant HTR baseline | — | `tesseract` optional extra, system Tesseract binary requirement (removed) | none — isolated adapter | — **EXECUTED (Stage 5)**: `src/archivetrust/providers/tesseract_layoutparser/` and `tests/providers/tesseract_layoutparser/` deleted. |
| `providers/qwen_vl/` (+ `runtime/transformers_runtime.py` usage) | General VLM document understanding | Delete | Not a Swedish-historical-HTR-specialized method; out of scope per brief's three named methods | — | `transformers` extra retained (Florence-2 also needs Transformers) | `runtime/transformers_runtime.py` retained/adapted for Florence-2, not deleted | — **EXECUTED (Stage 5)**: `src/archivetrust/providers/qwen_vl/` and `tests/providers/qwen_vl/` deleted. `runtime/transformers_runtime.py` retained as planned, but still has Qwen-specific baked-in assumptions (attention-implementation quirks, `Qwen2_5_VLForConditionalGeneration` class-name matching, a decode pattern tuned to Qwen's chat template) that were **not** removed in this stage — flagged for Florence-2's implementation phase to address, not silently left as a false "already generic" claim. |
| `providers/paddleocr_vl/`, `providers/paddleocr_vl_structured/` (+ `runtime/vllm_runtime.py`, `runtime/pp_doclayout_runtime.py`) | General VLM OCR, single-pass and structured-pipeline profiles | Delete | Not HTR-specialized; `DeploymentProfile.STRUCTURED_PIPELINE` concept was modeled too tightly around this one vendor's detect-crop-recognize shape | Independent segmentation stage (Phase 5 core abstractions) generalizes the reusable idea | `vllm`, `paddle-layout` optional extras (removed) | `DeploymentProfile.STRUCTURED_PIPELINE` enum value removed/replaced by the new segmentation-stage model | — **EXECUTED (Stage 5)**: both provider packages and their tests deleted; `DeploymentProfile.STRUCTURED_PIPELINE` removed from `providers/base.py` (`SINGLE_PASS` is the only remaining value); `runtime/pp_doclayout_runtime.py` deleted. |
| `providers/surya/` | General OCR via vLLM | Delete | Same as above | — | shares `vllm` extra with PaddleOCR-VL (extra removed once both gone) | none | — **EXECUTED (Stage 5)**: `src/archivetrust/providers/surya/` and `tests/providers/surya/` deleted. |
| `providers/decoding/` (`html_decoder.py`, `labeled_block_decoder.py`, `plain_text_decoder.py`) | Shared raw-VLM-response parsing for the deleted providers | Requires investigation → likely Reuse with modification | `plain_text_decoder.py` pattern (raw text → parsed transcription) is generically useful for Florence-2/SATRN parsing; the HTML/labeled-block decoders were Docling/PaddleOCR-specific | New `providers/parsing/` if reused | Evaluated during Phase 8 (Florence-2) implementation | — | **EXECUTED (Stage 5)**: `html_decoder.py`/`labeled_block_decoder.py` and their tests deleted (confirmed sole consumers were the deleted Docling/PaddleOCR adapters); `plain_text_decoder.py`/`base.py` and their tests kept in place (`providers/decoding/`, not relocated to a new `parsing/` package — that relocation, if still wanted, is deferred to Phase 8 as this row already anticipated). |
| `providers/ground_truth/` | Reserved provider identity, currently empty (`__pycache__` only) | Retain unchanged | Already a placeholder for evaluation-reference plumbing; compatible with HTR ground truth | n/a | none | none | Untouched, as planned. |
| `runtime/transformers_runtime.py` | HF Transformers model runtime | Reuse with modification | Needed by Florence-2 (and optionally SATRN if run through Transformers rather than HTRFlow) | n/a | `transformers`, `torch` extras retained | Remove Qwen-specific assumptions | **Partially executed**: retained per plan; Qwen-specific assumptions **not yet removed** (see the `providers/qwen_vl/` row above) — deferred to the Florence-2 implementation phase rather than done speculatively here. |
| `runtime/vllm_runtime.py`, `runtime/pp_doclayout_runtime.py` | vLLM client, PP-DocLayoutV2 runtime | Delete | Only consumers were the deleted PaddleOCR/Surya adapters | n/a | `vllm`, `paddle-layout` extras removed | none | **EXECUTED (Stage 5)**: both files and `tests/runtime/test_vllm_runtime.py`/`test_pp_doclayout_runtime.py` deleted, after confirming (grep) no consumer outside the deleted providers/`clients/desktop/composition.py`. `bootstrap/providers.py` (Docker lifecycle CLI for `paddleocr-vl`/`surya`, the only consumer of `vllm_runtime.py` outside `composition.py`) was also deleted, and its `providers start/stop/status/logs` subcommand removed from `bootstrap/cli.py` — not explicitly listed in the cleanup doc's tables, but a direct, necessary consequence of this row. |
| `providers/base.py` (`ProviderAdapter`, `ProviderRegistration`, `ProviderRunResult`) | Provider adapter interface | Replace | Becomes `HtrMethodAdapter` per brief's interface shape (adds `getMetadata`/`getCapabilities`/`validateEnvironment`/`healthCheck`); existing `observe()` contract and evidence/observation/rejection triple is retained conceptually | `providers/htr_adapter.py` (Phase 5) | All adapters | Registry keying scheme reused | **Judgment call — NOT deleted, deviating from this row's literal plan.** Investigation found `ProviderAdapter`/`ProviderRunResult`/`ProviderRegistry` are still actively load-bearing in `application/pipeline.py`, `acquisition/processing.py`, `presentation/provider_manager_viewmodel.py`, and the composition root — none of which have migrated to `HtrMethodAdapter` (a different-granularity interface: per-crop/line `recognize()`, not per-document `observe()`). Deleting `providers/base.py` now would break the application pipeline with no replacement, not complete a finished migration. What *was* removed: `DeploymentProfile.STRUCTURED_PIPELINE` (see row above). Full explanation in `providers/base.py`'s own module docstring. Revisit when SATRN (Stage 6) or a later stage actually threads an `HtrMethodAdapter`-based method through document-level orchestration. |
| `providers/registry.py` | Provider lookup by `(provider_id, deployment_profile)` | Reuse with modification | Lookup pattern is fine; `deployment_profile` axis needs redefinition now that STRUCTURED_PIPELINE is gone | n/a | — | — | **Kept unchanged** (reuse, no modification needed): the `(provider_id, deployment_profile)` keyed-lookup pattern is generic and still correct now that `DeploymentProfile` has only `SINGLE_PASS` — no redefinition was actually required, just one fewer enum value to key against. Consistent with `providers/base.py`'s row above: still load-bearing, not dead code. |

## Domain model

| Item | Classification | Reason | Migration implications | Executed |
|---|---|---|---|---|
| `ObservationType.TABLE_CELL`, `domain/comparison/table_reconciliation.py` | Delete | Table semantics don't fit line-level HTR research; no HTR method in scope produces structured tables | Any comparison-engine code branching on `TABLE`/`TABLE_CELL` needs the branch removed, not stubbed | **EXECUTED (Stage 5)**: `ObservationType.TABLE_CELL` and `TableCellPayload` deleted; `domain/comparison/table_reconciliation.py` and its test deleted; `domain/comparison/engine.py`'s `TABLE`-cluster special-case (cell-level reconciliation via `reconcile_table`) removed, `TABLE` now reconciles via the same generic single-deterministic-pick path as any other non-text-bearing type. `ObservationType.TABLE`/`TablePayload` retained. |
| `ObservationType.NAMED_ENTITY`, `ObservationType.RELATIONSHIP` | Delete | NER-style entity/relationship extraction is out of scope; historical-feature comparison (names, dates, places) is implemented as an *evaluator*, not an ontology observation type | Comparison-engine `_TEXT_BEARING_TYPES` set updated | **EXECUTED (Stage 5)**: both `ObservationType` values and `NamedEntityPayload`/`RelationshipPayload` deleted. `_TEXT_BEARING_TYPES` didn't reference either (only `HANDWRITTEN_NOTE` needed removing there); `review/triage.py`'s `_STRUCTURAL_TYPES` (had `RELATIONSHIP`) and `review/service.py`'s `_EDIT_CATEGORY_BY_TYPE` (had both) updated. |
| `ObservationType.HANDWRITTEN_NOTE`, `HandwrittenNotePayload` | Delete | Paragraph-level "is this handwritten" flag from Docling; replaced by a first-class line-level HTR recognition result, not a boolean note | Replaced by `TextLine`/`RecognitionResult` (Phase 5) | **EXECUTED (Stage 5)**: type and payload deleted; removed from `domain/comparison/engine.py::_TEXT_BEARING_TYPES`, `review/triage.py::DEFAULT_SINGLE_SOURCE_REVIEW_TYPES`, and `domain/comparison/capability_matrix_data.py`'s rating tables. `clients/desktop/demo_data.py`'s one real usage swapped to `ParagraphPayload` (smallest correct fix, per this doc's own suggested options). |
| `ObservationType.ARCHIVE_BOUNDARY` | Requires investigation | May still be relevant for multi-document archival collections; decide during Phase 4 domain design based on whether Collection/Document modeling needs it | — | Not decided in Stage 5 (out of scope) — still present, unchanged. |
| `BoundingBox.Precision` (`PIXEL_ACCURATE`/`COARSE_ESTIMATE`) | Reuse with modification | Geometry concept retained; needs to attach to `TextLine`/`Region` (line/region-level), not only single page-level boxes | — | Not touched in Stage 5 (out of scope). |
| `Evidence`, `Observation`, `CanonicalObservation`, `CanonicalDocument`, content-addressing (`domain/shared/ids.py`) | Retain, extend | Core evidence-first pattern this transformation is required to preserve | Add HTR-specific metadata fields (device, execution time, GPU memory, model revision, etc.) as new optional fields, not new incompatible types | Not touched in Stage 5 (already extended in an earlier stage per this row's own note). |

## Benchmark / artifacts / legacy client

| Item | Classification | Reason | Migration implications |
|---|---|---|---|
| `benchmark/` | Requires investigation → likely Reuse with modification | The metrics-group/aggregation/reporting *framework* is generic; its current metric definitions are OCR-platform "production readiness" oriented, not HTR research metrics | Decide during Phase 10 whether to retarget at HTR operational metrics or delete in favor of `evaluation/` |
| `artifacts/` (~3.3 GB generated benchmark/qualification output) | Delete | Fully reproducible generated output, not unique evidence, not part of the HTR research center, too large for the repository | None — regenerable by rerunning benchmarks if ever needed |
| `qualification_runs/`, `run_profiles/`, `benchmarks/`, `test-artifacts/`, `audit-artifacts/`, `tmp/`, `.codex_tmp/` (untracked working dirs at repo root) | Delete | Local generated/temporary working directories, not committed evidence | Verify nothing inside is unique before deletion (Phase 2) |
| `Analyzer.py`, `picker.py`, `pdf_analysis.csv` | Delete | Ad-hoc personal scripts classifying/sampling PDFs from an external corpus at `D:\ArchiveTrust\archive` outside this repo (born-digital vs. scanned classification, random sampling for provider qualification). Not HTR-relevant, output is reproducible from the external corpus, not tracked evidence | none | None — operates on external, non-repo data | None |
| `qualification_run_config.json` | Delete | Config for the old OCR-provider qualification run (`docling`, `tesseract_layoutparser`, `paddleocr-vl`, `surya` — all being deleted) | superseded by HTR experiment config (Phase 4 domain design) | None | None — reproducible config |
| `clients/desktop/` (legacy Qt client) | **Resolved: split.** `clients/desktop/composition.py::AppContext` is a live dependency (imported by `admin/qualification.py`, `worker/service.py` — this is the app's composition root/DI wiring, not UI). The rest of `clients/desktop/` (legacy Qt pages/viewmodels superseded by `desktop_v2`) is dead UI. | `composition.py` → Reuse with modification (relocate out of `clients/desktop/` to a neutral location, e.g. `src/archivetrust/composition.py`, so non-UI code doesn't import through a "desktop" package); rest of `clients/desktop/` → Delete | `desktop_v2` is the only supported UI per README | Update the two non-UI import sites after relocating `AppContext` | Medium — must not break `admin`/`worker` composition wiring | Low — `AppContext` logic is retained, just moved | **Partially executed, correcting a false premise in this row.** `composition.py` relocated to `src/archivetrust/composition.py` as planned, and all its deleted-provider wiring (Docling/Tesseract/Qwen/PaddleOCR/Surya adapter construction, `vllm`/`paddle_layout` runtime factories) stripped. Import sites updated -- but there were **more than the two named here**: `desktop_v2/app.py` and `desktop_v2/pages.py` *also* imported `AppContext` from `clients.desktop.composition` directly, plus every remaining `clients/desktop/*` module (`main.py`, `queue_worker.py`, `workspace_wizard.py`, `pages/settings.py`, `pages/provider_manager.py`). All updated to `archivetrust.composition`. **The "rest of `clients/desktop/` is dead UI" premise this row asserts is false as of this investigation**: `desktop_v2/app.py` and `desktop_v2/pages.py` import `ui.py` (`STYLESHEET`), `pdf_render.py`, `review_document_viewer.py`, `workspace_wizard.py`, and `pages/provider_manager.py`/`pages/settings.py` directly from `clients/desktop/` -- these are live shared infrastructure `desktop_v2` depends on, not superseded legacy UI. **`clients/desktop/` was NOT deleted** (only `composition.py` was removed from it, via relocation) -- deleting the rest as this row instructs would have broken the supported `desktop_v2` UI. Flagged for a follow-up investigation/doc correction, not acted on destructively based on a stale premise. |

## Dependencies

| Extra | Classification | Reason | Executed |
|---|---|---|---|
| `docling`, `tesseract` | Delete | Only consumers deleted | **EXECUTED (Stage 5)**: both `[project.optional-dependencies]` groups removed from `pyproject.toml`. |
| `vllm`, `paddle-layout` | Delete | Only consumers deleted | **EXECUTED (Stage 5)**: both groups removed from `pyproject.toml`. |
| `transformers` | Retain, redefine consumer | Now serves Florence-2 (and possibly SATRN); Qwen-specific code removed | **Partially executed**: extra retained in `pyproject.toml` as planned; the actual Qwen-specific code in `runtime/transformers_runtime.py` was **not** removed in this stage (see the `providers/qwen_vl/` row above) — this row's "Qwen-specific code removed" was aspirational for this stage and is now corrected to reflect reality. |
| `gui`, `watch`, `dev` | Retain unchanged | Not OCR-provider-specific | Untouched, as planned. |
| New: SATRN inference deps (HTRFlow or model's supported inference path), Florence-2 deps (`transformers`, image preprocessing), Transkribus import deps (PAGE XML / ALTO XML parsers) | Add | Required by Phases 7–9 | Not yet added — future-phase work, out of Stage 5's scope. |

## Validation performed

- This inventory is a planning artifact (Phase 1 deliverable); no deletions have been executed yet as of this writing.
- Each row's actual deletion will be validated per the project's standard checklist (compilation, type checking, tests, adapter contract tests, evidence-chain integrity) before being marked executed in a future revision of this document.

## Stage 11 execution record (docs/htr-migration-plan.md Stage 11 -- desktop client rework)

Executed. The research interface was built at the ViewModel layer (`presentation/htr_*_viewmodel.py`,
all Qt-independent per the existing architecture's own principle) with Qt page wiring in a new
`clients/desktop_v2/htr_pages.py`. `pages.py` was left at its existing 1697 lines rather than grown
further; the research pages have a different backing layer (the in-memory `HtrResearchStore`) than
the operational pages (telemetry read-models), so the split is along a real seam.

New in this stage:

| Item | Classification | Reason |
|---|---|---|
| `src/archivetrust/htr/research_store.py` | Add | Stages 1-10 landed the corpus/experiment entities as frozen models with **no store, no repository, and no telemetry event carrying them** (verified by grep: their only consumers were each other and `research/reports/models.py`). The Stage 11 ViewModels are required to read real domain data, so this is the missing holder. In-memory by design, mirroring `review/blind_review/store.py`'s stated rationale. Persisting HTR experiment entities into the append-only telemetry stream is real schema work (`docs/htr-domain-design.md` §2 lists the events it needs) and was **not** invented as a side effect of building a UI layer. |
| `src/archivetrust/research/reports/export.py` | Add | JSON + CSV export per the brief's "Support JSON and CSV first". **HTML and PDF deliberately not implemented**: the brief permits them "only if the architecture already supports it cleanly", and it does not -- no template engine, no HTML renderer, no PDF writer exists in the dependency set (`clients/desktop/pdf_render.py` *reads* PDFs via Qt, it does not write them). `UNSUPPORTED_FORMATS` names them so a caller gets a clear refusal rather than a silently missing option. |
| `presentation/display_names.py` `_PROVIDER_LABELS` | Replace | Became `_METHOD_LABELS` (the three real methods, keyed on each adapter's own `METHOD_ID` constant) plus `_LEGACY_METHOD_LABELS`. Deleted OCR providers are **not** erased from the label table: `docs/htr-domain-design.md` §8 requires legacy runs to stay readable and "clearly labeled", and stored telemetry still names them. They now render with a `" (retired OCR method)"` suffix so they cannot be mistaken for a fourth supported HTR method. `provider_label` is retained as an exact alias of the new `method_label`, because `providers/base.py`'s `ProviderAdapter` was deliberately not deleted and the legacy provider-admin pages still administer it under that name. |
| `presentation/first_launch_viewmodel.py::VISION_PROVIDER_CHOICES` | **Requires investigation -- flagged, not fixed** | Offers `paddleocr-vl`/`qwen2.5-vl`/`surya` as selectable "vision reading engines"; all three adapters were deleted in Stage 5, and `AppContext.provider_registry` was verified to register **zero** adapters, so the first-launch wizard offers three choices nothing can activate. Repairing it means deciding what a first-launch flow should offer in an HTR research centre -- a first-launch redesign, out of Stage 11's scope. A prominent `.. warning::` block was added to that module's docstring recording the finding and the reasoning, rather than leaving it to be rediscovered. |
| `clients/desktop/demo_data.py` (`_LAYOUT`/`_OCR`/`_VLM` = docling/tesseract/qwen2.5-vl) | Requires investigation | Demo-only seed data naming deleted providers. Left untouched: it is exercised by the demo-context tests and only feeds `build_demo_context`, so it produces legacy-labeled (not broken) output under the new label table. |

Navigation: seven research entries (Research Overview, HTR Methods, Datasets, Experiments,
Comparison, Evidence Chain, Review Center) were added ahead of the nine retained operational
entries, and the shell's default landing page moved from Workspaces to Research Overview. **No
operational page was deleted**: those administer the acquisition/processing/release machinery
`docs/htr-transformation-audit.md` classifies as "retain, extend", and Stage 11 does not replace
them. `MainWindowV2._build_page`'s if-chain was replaced by an exhaustive table so a page added to
the enum without a widget now fails loudly instead of silently falling through to Administration.

Two existing tests were updated rather than deleted, both for intended behavior changes: the
9-page rail assertions became `len(DesktopV2Page)` (asserted against the enum, so the next page
addition does not break them again), and `test_document_lifecycle.py`'s `"Docling" in row.providers`
became the retired-label form.

Validation: `1324 passed, 5 skipped` (from a `1193 passed, 5 skipped` pre-stage baseline in the
same tree; +131 new tests). The three real SATRN-inference tests skip only where `.venv-satrn` is
absent.

## Stage 5 execution record (docs/htr-migration-plan.md Stage 5)

Executed. Every row this stage's scope covered is marked "Executed"/"Partially executed" in its own table above, with the specific deviations from the plan's literal text called out inline rather than silently glossed over. Summary of the deviations (full reasoning lives on each affected row):

1. **`providers/base.py` was not deleted.** `ProviderAdapter`/`ProviderRunResult`/`ProviderRegistry` are still load-bearing for `application/pipeline.py`, `acquisition/processing.py`, `presentation/provider_manager_viewmodel.py`, and the composition root — none of which have migrated to `HtrMethodAdapter` (a different-granularity interface). What *was* removed: `DeploymentProfile.STRUCTURED_PIPELINE`.
2. **`runtime/transformers_runtime.py` still has Qwen-specific assumptions** (attention-implementation quirks, class-name matching, a Qwen-chat-template-tuned decode pattern) not removed in this stage — flagged for Florence-2's implementation phase.
3. **`clients/desktop/` was not deleted beyond `composition.py`.** The premise that the rest of `clients/desktop/` is dead UI superseded by `desktop_v2` is false: `desktop_v2/app.py` and `desktop_v2/pages.py` import `ui.py`, `pdf_render.py`, `review_document_viewer.py`, `workspace_wizard.py`, and two `clients/desktop/pages/*` modules directly. Only `composition.py` was removed (relocated to `src/archivetrust/composition.py`); every import site (more than the two named in this doc's original `clients/desktop/` row) was updated to the new location.
4. **Test suites**: every test directly importing a deleted provider module was deleted outright (not skipped), per the project's stated no-skipped-obsolete-tests rule. Where a test's principle was generic (e.g. `MappingTable`/`ObservationPayload` mechanics, "a provider may legitimately share one Observation across two Canonical Observations") it was preserved using a synthetic/generic fixture instead of a deleted provider's real payload/adapter. Where a test's principle was itself provider-specific and now belongs to a future adapter phase (e.g. "OCR provider provenance", "vision provider activation", the Milestone-7 3-OCR-provider end-to-end acceptance corpus), it was deleted with an inline TODO note for that future phase (SATRN, Stage 6, in most cases) rather than reimplemented speculatively here.
5. **`bootstrap/providers.py`** (Docker lifecycle CLI for `paddleocr-vl`/`surya`, not named in this doc's original tables) was deleted as a direct consequence of `runtime/vllm_runtime.py`'s deletion, along with its `providers start/stop/status/logs` subcommand in `bootstrap/cli.py`.

Full validation: `PYTHONPATH=src .venv/Scripts/python.exe -c "import archivetrust"` succeeds; `PYTHONPATH=src .venv/Scripts/python.exe -m pytest tests -q` is green (979 passed, 2 skipped — down from the pre-Stage-5 baseline of 1270 passed/2 skipped, entirely accounted for by deleted provider/decoding/runtime/table-reconciliation/vision-activation test suites, not by silent breakage).

## HTR telemetry-persistence pass execution record (2026-07-30)

Scope: give the HTR research entities durable, append-only, telemetry-backed persistence, replacing
their in-memory-only storage. The built architecture is documented in
`docs/architecture/htr-telemetry.md`; this section records only what was **deleted or replaced**, per
this file's purpose. Existing content above is untouched.

### Nothing was deleted

No module, class, function, test, or event kind was removed in this pass. That is worth stating
explicitly, because the obvious reading of "replace `HtrResearchStore`'s in-memory-only storage"
would be to delete the class. It was not deleted, and the reasoning is recorded in
`docs/architecture/htr-telemetry.md` §3 (a four-option comparison): the gap analysis §10 is explicit
that its query interface is correct and that "the gap is purely that nothing durable backs it", so
deleting a correct interface to reintroduce an equivalent one would have been churn that also forced
changes to four ViewModels and their tests for no behavioural gain.

### What was replaced (in role, not in code)

| Thing | Was | Is now |
|---|---|---|
| `htr/research_store.py::HtrResearchStore` | The sole source of truth for 19 buckets of HTR entities; in-memory, no telemetry, no disk (gap analysis §2) | The in-memory **projection** over a durable event log. Query interface unchanged; module docstring rewritten to state its four current jobs. No longer a source of truth. |
| `composition.py::AppContext.htr_research_store` | Constructed a bare `HtrResearchStore()`, process-wide, empty on every launch | Constructs a `DurableHtrResearchStore` per Workspace, hydrated by replaying that Workspace's `telemetry/htr_research_events.jsonl`. Reset by `open_workspace` so switching Workspaces switches research histories. |
| The five HTR event kinds added by the prior transformation (`SegmentationRunCompleted`, `MethodRunCompleted`, `ReviewSubmissionRecorded`, `AdjudicationRecorded`, `CanonicalResultCreated`) | Defined with **zero producers anywhere in `src/`** and no `Journal` replay branch (gap analysis §1) | All five have real producers in `htr/persistence/durable_store.py`. `MethodRunCompleted` and `CanonicalResultCreated` additionally participate in replay. |

### Additive changes to existing files

* `domain/telemetry/events.py`: +34 `TelemetryEventKind` members (68 total), `+correlation_id`/
  `+causation_id` on the shared `TelemetryEvent` base, new `HtrActorType` enum and
  `HtrTelemetryEvent` base class, new event classes, all registered in `EVENT_TYPE_BY_KIND`. The five
  pre-existing HTR event classes were re-parented from `TelemetryEvent` to `HtrTelemetryEvent`
  (additive -- only optional defaulted fields are gained). Three gained one optional field each
  (`CanonicalResultCreated.canonical_result`, `ReviewSubmissionRecorded.record`,
  `AdjudicationRecorded.record`) so replay can reconstruct them; reasoning in
  `docs/architecture/htr-telemetry.md` §5.
* `application/journal.py`: one new `_apply` branch (HTR events -> documented no-op pointing at
  `HtrJournal`). No existing branch changed.
* `htr/research_store.py`: three projection-support methods added (`advance_experiment_run`,
  `apply_transcript_stage`, `adopt_projection`), one new bucket (`_conventions`) with its
  `register_convention`/`convention`/`conventions` accessors, one new error type
  (`UnknownEntityError`). No existing method changed.
* `docs/TELEMETRY_STANDARD_V1.md`: the 34 new kinds and the correlation/causation fields documented,
  as `tests/test_telemetry_standard_publication.py` requires of every kind in the enum.

### New files

* `src/archivetrust/htr/persistence/__init__.py`
* `src/archivetrust/htr/persistence/durable_store.py` (`DurableHtrResearchStore`,
  `HtrCoarseEntitySnapshot`)
* `src/archivetrust/application/htr_journal.py` (`HtrJournal`, `causation_chain`)
* `docs/architecture/htr-telemetry.md`
* `tests/htr/persistence/{__init__,_fixtures}.py`,
  `tests/htr/persistence/test_read_model_reconstruction.py`,
  `tests/htr/persistence/test_correlation_and_causation.py`

### Tests updated rather than deleted

One assertion, for an intended behaviour change:
`tests/domain/telemetry/test_events.py::test_all_canonical_event_kinds_are_registered` -- the
hard-coded event-kind count `34` became `68`, with its running comment extended to account for the
30 kinds from `docs/architecture/htr-event-model.md` §3 plus the 4 documented additions that list
omitted. No test was deleted or skipped.

### Deliberately not done (deferred, not overlooked)

* `review/blind_review/store.py::BlindReviewStore` is **not** replaced. Its four record types now
  have durable, correlated telemetry, but its own in-memory persistence and query surface are
  unchanged -- reasoning in `docs/architecture/htr-telemetry.md` §7.
* `htr/experiment/baseline_execution.py` is unchanged. It already accepts an injected store and a
  `DurableHtrResearchStore` now satisfies that parameter, so the wiring is ready; actually re-running
  the baseline through it is Phase 4.
* `research/reports/*` still reads in-process dataclasses rather than durable records
  (gap analysis §8). `ResearchReportGenerated` exists for when that is re-pointed.
* Five event kinds (`ResearchObservationCreated`, `CandidateFindingCreated`, `FindingReviewed`,
  `FindingStatusChanged`, `ResearchReportGenerated`) are schema-only with no producer, disclosed in
  their docstrings and in `docs/TELEMETRY_STANDARD_V1.md` -- the knowledge lifecycle is a later phase.
