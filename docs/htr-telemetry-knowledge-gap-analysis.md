# HTR Telemetry and Research-Knowledge Gap Analysis

Phase 1 deliverable of the telemetry/provenance/knowledge follow-up. Written from direct code
inspection (file:line references below), not from prior planning documents' stated intentions.
Where this document contradicts `docs/htr-domain-design.md`, `docs/htr-repository-cleanup.md`, or
`README.md`, the contradiction is called out explicitly rather than silently resolved — per this
follow-up's own instruction to treat current, tested code as technical reality and document
discrepancies with prior planning.

## 0. Important correction to the record before anything else

**`docs/experiments/baseline-comparison/` does not exist.** It is not present on disk, was never
committed to git (`git log --oneline -- docs/experiments` returns nothing), and is not excluded by
`.gitignore` (it simply isn't there). This directly contradicts `README.md`'s "Reproducibility" and
"Experiment workflow" sections and `docs/htr-repository-cleanup.md`'s "Final state summary", both of
which describe it as "a real executed example" already in the repository.

This is not a fabrication in this document — it is a direct, reproducible observation (`ls
docs/experiments/` fails; `git show --stat HEAD -- docs/experiments` is empty). The most likely
explanation, consistent with an unrelated transient issue observed earlier in this same session
(`AGENTS.md` briefly disappearing from directory listings under heavy concurrent I/O before
reappearing intact), is that the three report files were written to disk by
`scripts/run_baseline_comparison.py` during Phase 13, verified once by direct file read, and then
lost before the final `git add -A`/commit — never restored the way `AGENTS.md` happened to be. There
is no way to distinguish "lost to a filesystem glitch" from "never actually survived its worktree
copy-back" from the evidence available now, and this document does not guess further.

**Practical consequence for this follow-up**: the instruction to "use the existing baseline as the
first real migration case" and "do not rerun expensive inference unnecessarily" cannot be honored
literally — there is no durable existing baseline artifact to migrate. §9 below (Phase 4 plan)
addresses this directly: the baseline will be re-executed exactly once, through the new durable
persistence path being built in this follow-up, and the re-run will be documented as a fresh
`ExperimentRun` with an honest note that it supersedes an earlier run whose durable evidence was
lost before commit — not presented as "migrated" data it is not.

## 1. The pre-existing generic telemetry/evidence architecture

`domain/telemetry/events.py` defines `TelemetryEventKind` (~30 kinds). Two populations:

- **Pre-existing (OCR-comparison era) kinds — the only ones any code actually constructs today**:
  `PROVIDER_OBSERVATION_ATTEMPTED`, `EVIDENCE_REJECTED`, `EVIDENCE_CREATED`, `OBSERVATION_CREATED`,
  `OBSERVATION_MAPPED`, `ALIGNMENT_ATTEMPTED`, `OBSERVATION_ALIGNED`, `OBSERVATION_LEFT_UNALIGNED`,
  `OBSERVATION_COMPARED`, `OBSERVATION_ACCEPTED`, `OBSERVATION_REJECTED`, `OBSERVATION_MERGED`,
  `AGREEMENT_CALCULATED`, `CANONICAL_DECISION_CREATED`, `CONFIDENCE_CHANGED`, `KNOWLEDGE_MERGED`,
  `KNOWLEDGE_DISCARDED`, `CANONICAL_DOCUMENT_CREATED`, `HUMAN_CORRECTION_SUBMITTED/APPLIED`,
  `DATASET_CANDIDATE_CREATED`, `PROVENANCE_CONTEXT_ESTABLISHED`, `CANDIDATE_EXCLUDED[_BATCH]`,
  `REVIEW_OUTCOME_RECORDED`, `REVIEW_PACKET_*`.
- **HTR kinds added in the prior transformation (Stage 3), defined but never constructed anywhere**
  (`events.py` lines 617–683): `SEGMENTATION_RUN_COMPLETED`, `METHOD_RUN_COMPLETED`,
  `REVIEW_SUBMISSION_RECORDED`, `ADJUDICATION_RECORDED`, `CANONICAL_RESULT_CREATED`. Zero producers
  anywhere in `src/`. `application/journal.py::Journal._apply`'s exhaustive elif-chain (lines
  326–396) has no branch for any of the five, and its explicit "no further state to reconstruct"
  comment list doesn't name them either — they are absent from replay, not merely no-ops.

**No `correlation_id`/`causation_id` concept exists anywhere in the codebase** (zero grep hits).
Correlation today is implicit, via shared domain ids embedded per-event (`document_ref`,
`semantic_slot_id`, `evidence_id`).

`infrastructure/storage/telemetry_sink.py::FileTelemetrySink` is the real, working, durable
mechanism: append-only JSONL, lazy byte-offset index, corrupt-line isolation, content-addressed
externalization of large raw output (>4096 bytes) into `blob_store.py`, and an O(1) incremental
hash-chain sidecar (`HashChainAppender`). `InMemoryTelemetrySink` is the non-durable counterpart,
explicitly documented as being for tests or as a lower layer beneath a durable sink.

`application/journal.py::Journal.replay(events) -> JournalState` is a **working, exercised**
read-model-reconstruction pattern for the pre-existing substrate — proof that "rebuild current state
from an append-only log, never by re-running a provider" already works in this codebase. It has zero
code path for any HTR entity today.

`domain/evidence/models.py::Evidence` was already extended in the prior transformation (Stage 3, not
this follow-up) with HTR-relevant optional fields (`model_revision`, `pipeline_configuration_hash`,
`execution_device`, `execution_time_ms`, `gpu_memory_mb`, `software_environment`,
`hardware_environment`) and remains content-addressed and immutable. This part of the retained
substrate is already correctly HTR-ready; the gap is entirely downstream of `Evidence`.

`docs/TELEMETRY_STANDARD_PROV_OPENLINEAGE_MAPPING.md` is explicitly "mapping guidance, not an
implemented exporter" and its mapping tables cite only pre-existing event kinds — none of the five
HTR kinds appear anywhere in it, despite being added afterward.

## 2. `src/archivetrust/htr/research_store.py` — the core of the gap

`HtrResearchStore` is a plain in-process object holding 19 `dict[str, Model]` buckets (projects,
datasets, dataset versions, collections, pages, regions, text lines, crops, experiments, experiment
versions, experiment runs, method runs + their transcripts, failures, metric definitions/results,
reproducibility manifests, canonical results, external imports) plus one ground-truth-text dict.

- **No telemetry emission**: zero references to `TelemetrySink`/`TelemetryEvent` in the file.
  `register_*` methods call only an internal `_put` (dict assignment under a lock).
- **No file/disk backing whatsoever**: zero `Path`/`open()` references.
- **Explicitly, honestly disclosed as in-memory-by-design in its own docstring** at the time it was
  written — this was a deliberate, disclosed scope decision (mirroring `BlindReviewStore`), not an
  oversight. This follow-up's job is to act on that disclosed deferral, not to treat it as a bug.
- **Not a true singleton**: `composition.py::AppContext.htr_research_store` lazily constructs one
  instance per `AppContext`; `baseline_execution.py::run_baseline_comparison` independently
  constructs a *second*, throwaway instance per script run unless a caller injects one. Neither is
  ever connected to disk or to each other. Both vanish on process exit.

## 3. `htr/experiment/models.py` and `htr/corpus/models.py`

Pure frozen Pydantic domain types. No `save`/`load` methods anywhere. `experiment/models.py`'s own
docstring already states these entities are "orchestrated by `application/pipeline.py` (a later
stage)" — confirmed that orchestration was never built (`application/pipeline.py` has zero
references to `MethodRun`/`HtrMethodAdapter`). `assert_experiment_mutable` is a pure function; its
"immutability after first run" guarantee only holds if a caller actually queries a real store before
calling it — today nothing calls it at all in the baseline flow (confirmed by grep).

## 4. Baseline experiment's actual (as-built) data flow

Traced through `baseline_execution.py::run_baseline_comparison` (the ~600-line orchestration
function) and `scripts/run_baseline_comparison.py::main`:

1. A fresh, empty `HtrResearchStore()` is created per run (no caller ever injects a durable one).
2. Every entity is registered into that in-memory store via `store.register_*` calls. **No
   telemetry event is constructed or appended anywhere in the function.**
3. A code comment in the function itself states `Evidence` records are deliberately *not*
   registered into `store` ("Evidence belongs to the retained Evidence/Observation substrate, not
   this research-facing store... the real Evidence objects are carried on
   `BaselineRunResult.satrn.evidence`") — but those `Evidence` objects are also never appended to a
   `TelemetrySink`, so in practice they only ever lived in one Python process's memory.
4. `build_research_report(result)` builds the `ResearchReport` **directly from the in-memory
   `BaselineRunResult`/`MethodOutcome` dataclasses** — not from `store`, not from telemetry.
5. The only disk writes in the entire flow are the three final export files (JSON + 2 CSVs),
   produced once, at the very end, from those in-memory objects — and (§0 above) those specific
   files no longer exist in this repository.

**This confirms, precisely, the follow-up's stated architectural gap**: HTR research entities are
domain models with no persistence path connecting them to the retained append-only telemetry
architecture at all — not "partially wired", but entirely absent from it. The generated reports were
never more than a snapshot of one process's memory, and in this case that snapshot did not survive.

## 5. Presentation/ViewModel layer — two structurally different backing patterns coexist

`presentation/htr_dashboard_viewmodel.py`, `htr_comparison_viewmodel.py`, `htr_dataset_viewmodel.py`,
`htr_evidence_viewmodel.py` all take `HtrResearchStore` directly in their constructors and call its
accessors synchronously on every `snapshot()` — no caching, no cursor, no incremental update.

By contrast, `presentation/read_model/facade.py::ReadModel` **is** a real, working, telemetry-backed
projection layer for the pre-existing system: it wraps a `CoreAggregate` updated incrementally via
`self.core.update(self._telemetry_source.all_events())`, exploiting `FileTelemetrySink`'s lazy
sequence view so a refresh only parses events recorded since the last one. The non-HTR ViewModels
(`quality_viewmodel`, `evolution_viewmodel`, `evidence_explorer_viewmodel`) all source from
`telemetry`/`read_model`, not from a raw in-memory dict store.

`docs/htr-repository-cleanup.md`'s Stage 11 record already states this split plainly ("the research
pages have a different backing layer... than the operational pages... so the split is along a real
seam") — i.e. this divergence was disclosed at the time, not discovered now. This follow-up's job is
to close that seam by giving the HTR ViewModels the same kind of durable, incrementally-updated
projection the pre-existing ViewModels already have — reusing the `read_model` *pattern*, not
necessarily its exact `CoreAggregate` implementation (which is keyed to pre-existing event kinds).

## 6. Review/evaluation persistence — the durability inconsistency, made concrete

- `review/blind_review/store.py::BlindReviewStore`: same in-memory, no-telemetry pattern as
  `HtrResearchStore`, explicitly modeled on it by its own docstring.
- `evaluation/ground_truth.py::FileGroundTruthStore`: the **opposite** pattern, already in this
  codebase — genuinely durable, append-only JSONL, content-hash-verified against the immutable
  Archive Object, supersession-chained, with an `export_with_provenance()` that writes an integrity
  manifest sidecar (reusing `infrastructure/storage/integrity.py`).

**The inconsistency is real and specific**: the pre-HTR ground-truth store already solved "durable,
append-only, provenance-verified persistence for research-relevant records" in this exact codebase.
`HtrResearchStore` and `BlindReviewStore` were built without reusing that solved pattern, despite
`docs/htr-domain-design.md` §5 stating the ground-truth package is exactly where
`TranscriptionConvention`/`GroundTruthItem` extensions should live. This is the strongest existing
precedent this follow-up should extend, rather than reinventing file-backed storage from scratch.

## 7. `htr/evaluation/*` — pure functions, correctly so

`recognition.py::compute_recognition_metrics`, `failures.py::classify_reliability`,
`aggregate_method_run_metrics` etc. are pure functions with no I/O, no store reference, no telemetry
— by design, and correctly so (they're computation, not persistence). The caller
(`baseline_execution.py::_evaluate_method_run`) is entirely responsible for persisting their return
values; today it only registers them into the throwaway in-memory store. **Classification: Already
sufficient as pure functions — the gap is entirely in what the caller does with their output, not in
these modules themselves.**

## 8. Reproducibility manifest and research report — confirmed entirely in-memory-sourced

`ReproducibilityManifest.create()` builds from values captured during the same process run and is
registered only into the throwaway store. `research/reports/models.py::ResearchReport`'s own
docstring already states "the generator that reads telemetry/read-models to populate one is a later
phase" — that generator (`build_research_report`) was in fact built, but reads neither telemetry nor
`HtrResearchStore`; it reads the narrower in-process `BaselineRunResult` dataclass directly. This is
a third, even-narrower data source than either of the two the design docs anticipated.

`workspace/store.py::WorkspaceStore` is confirmed as a second, simpler, already-working durable
persistence idiom in this codebase (whole-object JSON round-trip via Pydantic, overwrite-in-place,
no event sourcing) — a legitimate lighter-weight alternative to full telemetry+replay for entities
that don't need version history, worth considering for parts of this problem that don't need full
event sourcing (see §11).

## 9. Classification table

| Component | Classification | Reason |
|---|---|---|
| `domain/telemetry/events.py` (pre-existing kinds) | Retain unchanged | Correct, working, in active use |
| `domain/telemetry/events.py` (5 HTR kinds) | Extend | Schema exists; needs real producers and `Journal` replay support, plus more HTR-specific kinds per this follow-up's required event list |
| `infrastructure/storage/telemetry_sink.py::FileTelemetrySink` | Retain, reuse directly | Already durable, append-only, hash-chained, content-addressing large payloads — exactly what HTR needs, no HTR-specific variant required |
| `application/journal.py::Journal` | Extend | Working replay pattern; needs new `_apply` branches for HTR event kinds |
| `domain/evidence/models.py::Evidence` | Already sufficient | Already extended with HTR fields in the prior transformation |
| `htr/research_store.py::HtrResearchStore` | Generalize into a durable-backed repository; retain its *interface* as an in-memory/cache/test-fixture role | Its query interface is fine; its persistence mechanism is the gap |
| `review/blind_review/store.py::BlindReviewStore` | Requires investigation, likely extend in parallel using the same durable pattern | Same gap as `HtrResearchStore`, same disclosed deferral; in scope wherever review telemetry is required by this follow-up's Level 6, but not the primary target |
| `evaluation/ground_truth.py::FileGroundTruthStore` | Retain, use as the concrete durable-persistence precedent | Solves the exact problem already; new HTR persistence should follow its shape (or extend it) rather than invent a third pattern |
| `htr/evaluation/*` (metrics, failures) | Already sufficient | Correctly pure; no change needed |
| `research/reports/*` | Extend | Report generation must read from durable records, not in-process dataclasses, once those durable records exist |
| `presentation/htr_*_viewmodel.py` | Extend | Must be re-pointed at a durable-backed repository/read-model instead of a raw in-memory store, per this follow-up's read-model-reconstruction requirement |
| `presentation/read_model/facade.py::ReadModel` | Reference pattern, not directly reusable | Keyed to pre-existing event kinds; the *pattern* (incremental telemetry-sourced projection) should be replicated for HTR, not the exact class |
| `workspace/store.py::WorkspaceStore` | Reference pattern for non-versioned entities | Simpler whole-object JSON persistence, a legitimate lighter alternative for parts of the HTR domain that don't need append-only history (see §11) |
| `docs/TELEMETRY_STANDARD_PROV_OPENLINEAGE_MAPPING.md` | Extend | Needs the 5+ new HTR event kinds added to its mapping tables |
| `docs/experiments/baseline-comparison/` | Does not exist — recreate via re-run, not "migrate" | See §0 |

## 10. What this follow-up must NOT do (confirmed by this investigation)

- Must not treat `HtrResearchStore`'s dict-of-Pydantic-models query interface as wrong — it is a
  reasonable read/query shape; the gap is purely that nothing durable backs it.
- Must not invent a second, incompatible event-sourcing mechanism — `FileTelemetrySink` +
  `Journal.replay` is a working, adequate mechanism; it needs new event kinds and new replay
  branches, not a parallel system.
- Must not claim the baseline was "migrated" — the honest description is "re-executed once, through
  new durable persistence, because the original run's durable artifacts do not exist" (§0).
- Must not silently fold `htr/evaluation/*`'s pure computation functions into a persistence layer —
  keep them pure; persistence is the caller's job, as it correctly is today.

## 11. Recommended shape for durable HTR persistence (input to Phase 3, not yet implemented here)

Two viable, non-exclusive approaches given the evidence above:

1. **Append-only telemetry + replay** (matches the existing `Evidence`/`Observation`/`Journal`
   pattern exactly): every `HtrResearchStore.register_*` call becomes "construct a `TelemetryEvent`,
   append it to a `FileTelemetrySink`, and update an in-memory projection the same shape as
   `HtrResearchStore` already has." A new `HtrJournal.replay(events) -> HtrResearchState` (modeled
   directly on `Journal.replay`) rebuilds that projection from the event log alone. This is the
   pattern this follow-up's "read-model reconstruction" and "durable event coverage" sections
   describe, and it is the one that keeps HTR aligned with the retained substrate's own idioms.
2. **Whole-entity JSON persistence** (matches `WorkspaceStore`): for entities that don't need
   version/revision history exposed as events (e.g. `ResearchProject`, `Dataset` registration),
   simple durable JSON-file storage may be sufficient and simpler, avoiding event-log overhead for
   data that doesn't change after creation.

Given this follow-up's explicit emphasis on correlation/causation, replayable provenance, and
"reconstruct the read model after restart from durable events" as a *demonstrated, tested*
requirement, approach (1) is the primary mechanism for anything on the experiment-execution path
(experiments, runs, method runs, results, metrics, reliability, observations, findings); approach
(2) is reserved for coarse, rarely-changing registration entities where full event history adds
no research value. The exact split is finalized in Phase 3's implementation, not this document.
