# HTR Telemetry Persistence — The Built Implementation

Companion to `docs/architecture/htr-event-model.md` (the mapping and design) and
`docs/htr-telemetry-knowledge-gap-analysis.md` (the evidence-based investigation of the gap). Those
two describe what *should* exist and why. This document describes what *does* exist, in code, as of
2026-07-30 — including the places where implementation forced a decision the design documents left
open, and the places where it forced a decision *against* them.

## 1. What changed, in one paragraph

HTR research entities previously lived only in `HtrResearchStore`'s 19 in-process `dict`s: no
telemetry emission, no disk backing, gone on process exit (gap analysis §2). They now have a durable
append-only home. `htr/persistence/durable_store.py::DurableHtrResearchStore` appends a telemetry
event for every registration to a `FileTelemetrySink` and *then* updates an in-memory projection;
`application/htr_journal.py::HtrJournal` rebuilds that projection from the event log alone;
`composition.py::AppContext.htr_research_store` hands out the durable store instead of a bare dict.
34 new `TelemetryEventKind` members were added, the five HTR kinds the prior transformation left
producerless got real producers, and `correlation_id`/`causation_id` were added to the shared
`TelemetryEvent` base. No new storage mechanism was invented: `FileTelemetrySink` is reused entirely
unmodified, hash-chain sidecar and blob externalization included.

## 2. Entity relationships and where each one is persisted

```
                                  ┌─ event-sourced ──────────────┐
ResearchProject ─────────────────── ResearchProjectCreated       │  every entity in this
 └─ Dataset ─────────────────────── DatasetCreated               │  column is reconstructed
     ├─ DatasetVersion  (immutable)─ DatasetVersionCreated        │  field-for-field by
     └─ Collection ───────────────── CollectionCreated            │  HtrJournal.replay from
         └─ Document (=ArchiveObject) DocumentRegistered          │  the event log alone
             └─ Page ─────────────── PageRegistered               │
                 └─ Region ───────── RegionDetected               │
                     └─ TextLine ─── TextLineDetected             │
                         └─ InputCrop  InputCropCreated           │
                            (content-addressed: InputCrop.hash)   │
                                                                  │
TranscriptionConvention ─────────── TranscriptionConventionRegistered
                                                                  │
Experiment ──────────────────────── ExperimentCreated             │
 └─ ExperimentVersion ───────────── ExperimentVersionCreated      │
     └─ ExperimentRun ───────────┬── ExperimentRunStarted    ◄── correlation root
         │                       ├── ExperimentRunCompleted       │
         │                       └── ExperimentRunFailed          │
         ├─ ReproducibilityManifest  ReproducibilityManifestRecorded
         └─ MethodRun ───────────┬── MethodRunStarted  (carries the full MethodRun)
             │                   ├── MethodRunCompleted  (terminal marker)
             │                   └── MethodRunFailed  (carries the FailureRecord)
             ├─ RawResult ───────── RawMethodResultRecorded   ┐
             ├─ ParsedResult ────── ParsedMethodResultRecorded │ MethodRunTranscript's
             ├─ NormalizedResult ── NormalizedMethodResultRecorded │ four stages, never
             ├─ (reviewed text) ─── ReviewedResultRecorded     ┘ collapsed into one
             ├─ MetricResult ────── MetricCalculated
             │   └─ MetricDefinition  MetricDefinitionRegistered
             └─ (reliability) ───── ReliabilityIssueClassified

CanonicalResult ─────────────────── CanonicalResultCreated
ExternalImport ──────────────────── ExternalResultImported
ground truth text (per TextLine) ── GroundTruthTextRecorded

                                  ┌─ events durable, projection deferred ─┐
ReviewAssignment ────────────────── ReviewAssigned                        │  see §7
 └─ ReviewSubmission ────────────── ReviewSubmissionRecorded              │
     ├─ AgreementResult ─────────── AgreementCalculatedHtr                │
     └─ Adjudication ────────────── AdjudicationRecorded                  │

                                  ┌─ schema only, no producer ─┐
ResearchObservation ─────────────── ResearchObservationCreated │  later phase
ResearchFinding ─────────────────┬─ CandidateFindingCreated    │
                                 ├─ FindingReviewed            │
                                 └─ FindingStatusChanged       │
ResearchReport ──────────────────── ResearchReportGenerated    │
```

The traceability chain `docs/htr-domain-design.md` §4 specifies is preserved exactly, as stored id
references rather than embedded copies:

```
CanonicalResult.spans[].source_method_run_id
  → MethodRun.method_run_id
    → MethodRun.evidence_id → Evidence.id  (the retained substrate, untouched)
    → MethodRun.input_crop_id → InputCrop.crop_id
       → InputCrop.hash  (content address of the crop bytes)
       → InputCrop.text_line_id → TextLine.region_id → Region.page_id
       → Page.archive_object_ref → ArchiveObject  (the immutable acquisition record)
    → MethodRun.experiment_run_id → ExperimentRun.experiment_version_id
       → ExperimentVersion.dataset_version_id → DatasetVersion.dataset_id → Dataset.project_id
```

## 3. `HtrResearchStore`'s final role, and why

**Decision: kept, with its query interface untouched, but demoted from source of truth to
projection — and subclassed to add durability.** It now has exactly four jobs, enumerated in its own
module docstring: the query shape the ViewModels consume, the state `HtrJournal.replay` returns, the
base class `DurableHtrResearchStore` extends, and a legitimate test fixture when constructed bare.

The four options considered, and why three were rejected:

| Option | Rejected because |
|---|---|
| Delete and replace with a new durable repository class | The gap analysis §10 is explicit that the store's dict-of-Pydantic query interface "is a reasonable read/query shape; the gap is purely that nothing durable backs it". Deleting a correct interface to reintroduce an equivalent one is churn, and it would have forced changes to four ViewModels and their tests for no behavioural gain. |
| Keep it as a pure in-memory cache *in front of* a separate durable store | Two objects answering the same questions, with a cache-invalidation problem between them. The projection *is* the cache; a second layer adds a failure mode without adding an answer. |
| Wrap it by composition (`DurableStore` holding an `HtrResearchStore`) | Would require re-declaring ~40 delegating query methods, and every ViewModel constructor signature (`store: HtrResearchStore`) would have had to change or widen to a Protocol. Subclassing makes `isinstance(durable, HtrResearchStore)` true, so **zero** ViewModel or ViewModel-test changes were needed. |
| **Chosen: keep as the projection; `DurableHtrResearchStore(HtrResearchStore)` overrides every `register_*` to emit-then-project** | Liskov holds — the subclass answers every query identically and additionally returns the emitted `event_id` from registrations (a widened return the parent's `None`-returning callers ignore harmlessly). |

What is *no longer supported* is production code treating a bare `HtrResearchStore()` as storage.
`AppContext` no longer does. Tests that construct one directly still do, and still mean what they
say: a projection with no sink behind it is precisely what a unit-test fixture wants.

Two projection-only methods were added to support replay, both documented as such:
`advance_experiment_run` (a terminal marker carrying a later state of an already-registered run) and
`apply_transcript_stage` (incremental assembly of `MethodRunTranscript` from its four stage events).
Advancing a projection entry is not rewriting history — the durable log holds every event in append
order and a fresh replay is deterministic — and it is exactly what `Journal._apply` already does
when it overwrites `_alignment_state_by_observation[...]`. A third method, `adopt_projection`, exists
solely so `DurableHtrResearchStore.open` can hydrate from a replay *without* going through the
emitting overrides; see §8 for the bug that made this necessary.

## 4. The two persistence patterns, and the split between them

Gap analysis §11 offered two shapes and deferred the exact split to implementation. Here it is.

**Event-sourced with replay** — everything. Every entity in §2's first and second groups emits a
telemetry event and is reconstructed by `HtrJournal.replay`. This includes the coarse registration
entities (`ResearchProject`, `Dataset`, `DatasetVersion`, `Collection`, `TranscriptionConvention`),
which §11 offered as candidates for JSON-only storage.

**Whole-object JSON snapshot** — additionally, and only as a *derived cache*, for those same five
coarse entities, via `HtrCoarseEntitySnapshot` in `workspace/store.py::WorkspaceStore`'s idiom (one
JSON file, Pydantic round-trip, overwrite in place, no history).

**Why coarse entities got events too, rather than JSON only.** Three concrete reasons, in order of
weight:

1. **The required event list demands it.** `docs/architecture/htr-event-model.md` §3 lists
   `ResearchProjectCreated`, `DatasetCreated`, `DatasetVersionCreated` as required kinds. Emitting
   them and then *not* replaying them would leave events written but unreadable — the exact
   defined-but-never-constructed condition gap analysis §1 flags as the prior transformation's
   mistake.
2. **The corpus tree is not actually flat.** A `DatasetVersion` chains by `supersedes`;
   `TranscriptionConvention` freezes per `(convention_id, version)` and must keep old versions
   resolvable (`docs/htr-domain-design.md` §3). "Doesn't change after creation" is true of each
   *record* but not of the *entity's* history, which is the thing event sourcing preserves.
3. **A dataset's membership is research-relevant provenance.** Which collections a
   `DatasetVersion` snapshotted, and when, is part of what a published result rests on.

**So what is the JSON snapshot for?** A cheap "what projects exist?" answer for a first-paint UI
without parsing the whole event log — which is the *only* thing §11 offered `WorkspaceStore`'s
pattern for. It is a cache, and the test of that claim is that deleting the file loses nothing:
`test_coarse_snapshot_is_a_derived_cache_and_deleting_it_loses_nothing` deletes it, replays, gets
every entity back, and regenerates the snapshot via `HtrCoarseEntitySnapshot.rebuild`.

## 5. Where events embed typed objects, and where they carry dicts

`domain/telemetry/events.py`'s own docstring requires that events carry "the *full* domain object(s)
[they announce] (not just an id), because replay must reconstruct [state] from telemetry alone". Two
constraints cut across that:

* `tests/domain/test_dependency_direction.py::test_domain_layer_never_imports_providers` forbids
  any `domain/*` module from importing `archivetrust.providers.*` at all.
* `ReviewOutcomeRecorded`'s own docstring states the rule for the rest: "`review` depends on
  `domain.telemetry`, never the reverse". Since `review/service.py` imports `domain.telemetry`, a
  reverse import would create a package-level cycle.

So the split is drawn on layering, not convenience:

| Entity source | Carried as | Why |
|---|---|---|
| `htr/corpus/models.py`, `htr/experiment/models.py` | **Typed embedded object** | Pure frozen Pydantic types whose only imports are `domain.evidence.models` and `domain.shared.ids`; their package `__init__`s pull nothing else and `htr/__init__.py` is docstring-only. Cycle-free, and passes both guard tests. |
| `domain/canonical/result.py::CanonicalResult` | **Typed embedded object** | Already `domain/`. |
| `providers/transkribus/external_import.py::ExternalImport` | `record: dict` | Forbidden outright by the guard test. |
| `review/htr_models.py`, `evaluation/ground_truth.py::TranscriptionConvention` | `record: dict` | Would invert the stated dependency direction and cycle. |

The dicts are `model_dump(mode="json")` of the frozen model, and
`application/htr_journal.py` — an application-layer module that *may* import `review`,
`evaluation` and `providers` — reconstructs the typed model with `model_validate` on replay. Nothing
is lost; the type is simply recovered one layer out.

One pre-existing HTR event kind gained a field: `CanonicalResultCreated.canonical_result`, optional
and defaulted. Its original id-only design was justified on the grounds that "a `CanonicalResult`
can be read back from its own store by id" — reasoning that no longer holds now that the telemetry
stream *is* the store. Leaving it id-only would have made `CanonicalResult` the single HTR entity
replay could not reconstruct. `ReviewSubmissionRecorded` and `AdjudicationRecorded` gained an
optional `record` dict for the same reason. All three additions are optional with defaults, so the
pre-existing construction shapes in `tests/domain/telemetry/test_htr_events.py` still validate
unchanged.

## 6. The event vocabulary: 68 kinds

34 pre-existing + 34 new. Of the 34 new, 30 are exactly the list in event-model doc §3. Four are
documented additions that list omitted, each justified in its own enum-member docstring:

| Addition | Why §3's list was insufficient |
|---|---|
| `CollectionCreated` | §3 lists `DocumentRegistered` but not the `Collection` that owns the documents. `HtrResearchStore` has had a `Collection` bucket since Stage 11 and `htr-domain-design.md` §1 places it between `DatasetVersion` and `Document`; replay could not rebuild the corpus tree without it. |
| `ReviewedResultRecorded` | §3 names the three *machine* transcript stages. `MethodRunTranscript` has a fourth, `reviewed_text`, whose docstring insists it "belongs to a different lineage" (human, not method). Folding it into `ReviewSubmissionRecorded` would conflate the blind-review target lineage with the method-run lineage that docstring separates. |
| `MetricDefinitionRegistered` | §3 lists `MetricCalculated` (the result) but not the versioned `MetricDefinition` it was computed against, which the store holds and which `htr-domain-design.md` §3 requires be independently versioned so replay knows which calibration produced a value. |
| `GroundTruthTextRecorded` | The store carries a resolved `text_line_id → reference transcription` mapping that drives every metric display. Without it, a replayed store shows metrics with no reference text explaining them. Records only the resolved string; `evaluation/ground_truth.py` still owns the `GroundTruthAnnotation` workflow. |

Five of the 34 (`ResearchObservationCreated`, `CandidateFindingCreated`, `FindingReviewed`,
`FindingStatusChanged`, `ResearchReportGenerated`) are **schema-only with no producer anywhere in
`src/`**, disclosed as such in their docstrings and in `docs/TELEMETRY_STANDARD_V1.md`. The
knowledge lifecycle they belong to is explicitly a later phase. Landing the closed vocabulary now
means the enum need not be reopened for it, and each references its subject by id and carries no
entity object precisely so the later phase can model `ResearchObservation`/`ResearchFinding` freely
without migrating a schema guessed at here. This is the same disclosure `ObservationMapped` has
carried since 2026-07-14.

### `document_ref` for research-scoped events

`TelemetryEvent.document_ref` is non-optional and documented as scoping an event to the Archive
Object it concerns. Most HTR research entities have no such scope — an `Experiment` spans a whole
`DatasetVersion`. Rather than loosen the base field or invent a plausible-looking document
reference, HTR events carry the explicit sentinel `HTR_RESEARCH_SCOPE = "htr:research"` and do their
real scoping through `HtrTelemetryEvent`'s purpose-built `project_id`/`dataset_id`/
`dataset_version_id`/`experiment_id`/`experiment_version_id`/`experiment_run_id`/`method_run_id`/
`subject_id` fields — §3's own mandated field list. `PageRegistered` and `DocumentRegistered` are
the exceptions: a page and a document member genuinely belong to exactly one Archive Object, so they
carry the real `archive_object_ref` and per-document replay of page-scoped HTR history keeps working
(`test_page_events_stay_scoped_to_their_real_archive_object`).

## 7. Scope decisions

**`BlindReviewStore`: events durable now, projection deferred.** The brief left this call open. The
four review record types (`ReviewAssignment`, `ReviewSubmission`, `AgreementResult`, `Adjudication`)
now have durable, correlated telemetry via `DurableHtrResearchStore.record_review_*`, which is what
gives `REVIEW_SUBMISSION_RECORDED` and `ADJUDICATION_RECORDED` their first real producers. What is
*not* done is replacing `BlindReviewStore`'s own in-memory persistence or projecting these events
into a query surface. Reasoning: `BlindReviewStore` owns blind-isolation *workflow* logic (which
reviewer may see what, when) that is a different concern from durability, its query surface is
consumed by `ReviewCenterViewModel`, and rewriting it in the same pass as the experiment-execution
path would have doubled the blast radius of this change for a subsystem the follow-up names as "not
the primary target". `HtrJournal._apply` names all four as explicit no-ops with this reason.

**`Journal` extended, not merged.** `application/journal.py::Journal._apply` gained one branch:
`isinstance(event, HtrTelemetryEvent)` → documented no-op, pointing at `HtrJournal`. The two states
reconstruct disjoint entity sets from disjoint event kinds; merging them would have produced one
class with two unrelated halves where every caller uses one and ignores the other. The branch exists
so a hand-merged archive produces an explicit documented no-op rather than a silent fall-through
past the end of the chain. This is the choice gap analysis §11 anticipated and event-model doc §2
pre-committed to.

**Two streams, one mechanism.** HTR events go to `htr_research_events.jsonl`, not `events.jsonl`.
`presentation/read_model/facade.py::ReadModel` maintains an incremental cursor over `events.jsonl`
keyed to the pre-existing kinds; interleaving ~35 HTR kinds it does not project would advance that
cursor on every HTR registration for nothing. Same sink class, same append-only guarantees, same
hash-chain sidecar — which is not the "second, incompatible event-sourcing mechanism" gap analysis
§10 forbids.

**Baseline re-execution: ~~not done here~~ done in Phase 4 (2026-07-30).** This section originally
read "`baseline_execution.py` still constructs a throwaway `HtrResearchStore()` when no store is
injected ... actually re-running the baseline through it is Phase 4, a separate task." That task is
now complete and the statement no longer holds. `run_baseline_comparison` now *defaults* to a
`DurableHtrResearchStore` over an `InMemoryTelemetrySink` (so it always event-sources; only the medium
varies), threads `caused_by` through every registration, scopes each `ExperimentRun`'s events with
`correlated_to`, and emits `ExperimentRunCompleted` for both runs. The re-executed run's durable log
and reports are committed under `docs/experiments/baseline-comparison/`, with
`docs/experiments/baseline-comparison/README.md` stating plainly that it is a fresh run superseding
one whose evidence did not survive (gap analysis §0) rather than a migration of it.

Phase 4 also closed two gaps this document had listed as later work, and added one producer:

* **`Evidence` now reaches a sink.** Gap analysis §4 point 3 noted that the baseline's `Evidence`
  records "are also never appended to a `TelemetrySink`". `DurableHtrResearchStore.record_evidence`
  appends them as the retained substrate's own `EvidenceCreated` — no new kind invented — making this
  stream a deliberately mixed one, which `HtrJournal.replay` already documents as supported.
  `HtrResearchStore` still has no `Evidence` bucket; `evidence_from_events` reads them back off the
  stream instead.
* **Report generation reads durable records** (see §11's revised entry below).
* **`ReliabilityIssueClassified` has a producer**, emitted alongside the `MethodRunFailed` that
  carries the same `FailureRecord` — the former is the semantically correct evaluation-layer kind, the
  latter is what this projection actually indexes. Both, and the honest caveat about applying
  `MethodRunFailed` to a succeeded run, are documented at the emitting call site.

## 8. How correlation and causation work, concretely

Both are optional fields on the shared `TelemetryEvent` base — not on an HTR subclass, and not in a
parallel envelope — because event-model doc §4 specifies they apply to all events, old and new.
Purely additive: every pre-existing construction call site keeps working, and pre-existing producers
deliberately still leave both `None` rather than having a correlation invented for them
retroactively.

**`correlation_id` = the `ExperimentRun.id`.** Chosen because it is already unique, already stable,
already meaningful to a researcher reading the log, and needs no second identifier minted alongside
it. Scoped by a context manager:

```python
run_event = store.register_experiment_run(run, caused_by=version_event)
with store.correlated_to(run.experiment_run_id):
    method_event = store.register_method_run(method_run, caused_by=run_event)
    store.complete_method_run(method_run, caused_by=method_event)
```

`correlated_to` nests and restores, so an inner unit of work cannot leak into the outer one.
`register_experiment_run` defaults the correlation to the run's own id when no scope is active: that
event is the causal root of the run, so it belongs to that run by definition rather than by a caller
remembering to say so. Outside any run, `correlation_id` is an honest `None` — a bare
`ResearchProject` registration genuinely belongs to no run.

**`causation_id` = the `event_id` of the directly-causing event.** Every registration method returns
the `event_id` it emitted; the caller threads it into the next call's `caused_by`. That return value
is the whole mechanism — there is no implicit "previous event" state, so a chain is only as good as
the caller's honesty about what actually caused what.

`application/htr_journal.py::causation_chain(events, from_event_id=...)` walks the DAG forward using
**only** `causation_id → event_id` pointers. It consults no timestamp, no append position, and no
shared domain id. Where one event caused several, it follows the branch that continues furthest, so
the result is the longest causal path rather than an arbitrary sibling.

## 9. How replay reconstructs state

```
FileTelemetrySink(path).all_events()      # lazy, snapshot-consistent Sequence
        │
        ▼
HtrJournal().replay(events) ──► fresh HtrResearchStore()
        │
        │  for each event, exactly one branch of an exhaustive isinstance chain:
        ├─ ResearchProjectCreated  ──► store.register_project(event.project)
        ├─ ExperimentRunStarted    ──► store.register_experiment_run(event.experiment_run)
        ├─ ExperimentRunCompleted  ──► store.advance_experiment_run(event.experiment_run)
        ├─ RawMethodResultRecorded ──► store.apply_transcript_stage(stage="raw", ...)
        ├─ ExternalResultImported  ──► ExternalImport.model_validate(event.record)
        └─ ... 14 kinds are documented no-ops, each with its reason stated inline
```

Replay is stateless across calls: each `replay()` starts a fresh store, so a reconstruction is
reproducible purely from stored events. Non-HTR events are skipped rather than rejected, since a
caller may legitimately hand it a mixed stream.

An `ExperimentRunCompleted` whose `ExperimentRunStarted` is missing raises `UnknownEntityError`
rather than being silently promoted to a first registration — an incomplete log must surface
(`test_replay_refuses_to_invent_an_entity_a_terminal_event_advances`).

### The restart path, and one bug worth recording

`DurableHtrResearchStore.open(sink)` is what `composition.py` uses. Its first implementation
replayed directly into `self`, which went through this class's *emitting* `register_*` overrides and
appended a second copy of every historical event — the log would have doubled on every process
start. `test_open_rebuilds_a_durable_store_ready_to_keep_appending` caught it (40 events where 23
were expected). `open` now replays into a plain `HtrResearchStore` and adopts its buckets via
`adopt_projection`, which writes to the projection only. This is the concrete reason
`adopt_projection` exists and why bypassing `register_*` there is correct rather than a shortcut.

### Write ordering

`_emit` does validate → append → project, in that order — `FileGroundTruthStore.append`'s order,
deliberately. A crash can therefore leave an event the projection has not yet seen (harmless; the
next replay picks it up) but never a projection entry with no durable event behind it (which would
vanish on restart — exactly the failure this pass exists to remove).

## 10. What is proven by test, not asserted by prose

`tests/htr/persistence/` — 17 tests. The two the follow-up specifically asks for:

**Destroy-and-reconstruct** (`test_read_model_survives_destruction_of_the_store`): registers a
project/dataset/collection/dataset-version/page/region/line/crop/experiment/experiment-version/
experiment-run/method-run/transcript/metric-definition/metric-result/manifest through the durable
store; confirms the events are on disk; `del`s the store and forces `gc.collect()`; constructs a
**brand-new** `FileTelemetrySink` over the same path; replays; then asserts field-for-field equality
on all of it. Deliberately not a whole-store blob round trip, which the brief says does not count.

**Causation chain** (`test_causation_chain_of_four_or_more_links`): walks a 7-link chain
(`ExperimentRunStarted → MethodRunStarted → RawMethodResultRecorded →
ParsedMethodResultRecorded → NormalizedMethodResultRecorded → MetricCalculated →
ReproducibilityManifestRecorded → ExperimentRunCompleted`) purely by pointer, asserting
`child.causation_id == parent.event_id` for every consecutive pair.
`test_the_chain_is_not_merely_append_order` guards it from passing trivially by proving the causal
sequence and the file's append order are genuinely different — `MethodRunCompleted` is appended
mid-chain but is a causal *sibling*, not a link.

Also covered: events reach a real file with the hash-chain sidecar; `InputCrop.hash` survives the
round trip unchanged and still equals `InputCrop.compute_hash` of the original bytes; the terminal
run state (not the started one) is what replay yields; `open` supports continued appending and still
refuses duplicates after a restart; page events keep their real Archive Object scope; the coarse
JSON snapshot is a deletable cache; an in-memory sink honestly reports `is_durable is False`;
correlation scopes nest and restore; and pre-existing event kinds still construct without the two
new base fields.

## 11. Known limitations for the next phase

* **Review projection deferred** (§7). The four review event kinds are durable and correlated but
  are not projected into any query surface; `BlindReviewStore` is still in-memory.
* **Five event kinds have no producer** (§6), by design, pending the knowledge lifecycle.
* **`ExperimentRunFailed` and `ReliabilityIssueClassified` are durable but unindexed.** Both are
  written and replayable from the sink; neither has a bucket in `HtrResearchStore`'s query surface.
  Named as no-ops in `HtrJournal._apply` rather than left implicit. As of Phase 4
  `ReliabilityIssueClassified` has a real producer, which is why the baseline also records each
  classification as a `MethodRunFailed` — that is the only one of the two the projection indexes. A
  reliability bucket on `HtrResearchStore` would let the duplication go away; that is the next phase's
  call to make.
* ~~**`baseline_execution.py` is ready but unwired** (§7) — Phase 4.~~ Done, 2026-07-30 — see §7.
* ~~**`research/reports/*` still reads in-process dataclasses**, not durable records (gap analysis
  §8).~~ Done, 2026-07-30. `htr/experiment/baseline_execution.py::build_research_report_from_store`
  sources every field from store queries plus the stream's `EvidenceCreated` records;
  `build_research_report(result)` is now a thin wrapper over it, so there is one code path and the
  live and replayed reports cannot drift. Proven by regenerating the committed report from a replay
  and comparing field for field
  (`tests/htr/persistence/test_real_baseline_reconstruction.py::test_the_committed_report_is_reproducible_from_the_committed_log`).
  `research/reports/models.py`/`export.py` themselves were not changed — they were never the problem;
  the *generator* was. `ResearchReportGenerated` still has no producer, for the reason now recorded in
  its own docstring: announcing a report is a research-knowledge-lifecycle concern, and Article 33
  forbids a projection from emitting telemetry about itself.
* **`assert_experiment_mutable` still has no caller.** The durable store now makes a real store
  queryable at the point a mutation would occur, so the enforcement the function was written for is
  finally *possible* — but wiring it is an experiment-editing concern, not a persistence one, and
  no experiment-editing entry point exists yet.
* **Single-writer.** Inherited unchanged from `FileTelemetrySink`: a concurrent writer in another
  process is not reflected until the sink is reconstructed.
