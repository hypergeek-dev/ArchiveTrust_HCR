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

                                  ┌─ event-sourced (added 2026-07-30, §12) ─┐
ResearchObservation ─────────────── ResearchObservationCreated               │
ResearchFinding ─────────────────┬─ CandidateFindingCreated                  │
                                 ├─ FindingReviewed  (act of review)         │
                                 └─ FindingStatusChanged  (carries the       │
                                    whole post-transition finding)           │

                                  ┌─ schema only, no producer ─┐
ResearchReport ──────────────────── ResearchReportGenerated    │  see §6
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

## 6. The event vocabulary: 70 kinds

34 pre-existing + 36 new. Of the 36 new, 30 are exactly the list in event-model doc §3. Six are
documented additions that list omitted, each justified in its own enum-member docstring — the four
below, plus `ResearchQuestionRaised` and `ExperimentDraftedFromQuestion`, added 2026-07-30 with the
research-question feedback loop (§13). Those last two are beyond §3's list because §3's layers 10-12 run
one way, event → observation → finding, and enumerate no kind for the edge *back*, which the knowledge
follow-up requires ("allow an observation, finding or contradiction to create a new research question").

| Addition | Why §3's list was insufficient |
|---|---|
| `CollectionCreated` | §3 lists `DocumentRegistered` but not the `Collection` that owns the documents. `HtrResearchStore` has had a `Collection` bucket since Stage 11 and `htr-domain-design.md` §1 places it between `DatasetVersion` and `Document`; replay could not rebuild the corpus tree without it. |
| `ReviewedResultRecorded` | §3 names the three *machine* transcript stages. `MethodRunTranscript` has a fourth, `reviewed_text`, whose docstring insists it "belongs to a different lineage" (human, not method). Folding it into `ReviewSubmissionRecorded` would conflate the blind-review target lineage with the method-run lineage that docstring separates. |
| `MetricDefinitionRegistered` | §3 lists `MetricCalculated` (the result) but not the versioned `MetricDefinition` it was computed against, which the store holds and which `htr-domain-design.md` §3 requires be independently versioned so replay knows which calibration produced a value. |
| `GroundTruthTextRecorded` | The store carries a resolved `text_line_id → reference transcription` mapping that drives every metric display. Without it, a replayed store shows metrics with no reference text explaining them. Records only the resolved string; `evaluation/ground_truth.py` still owns the `GroundTruthAnnotation` workflow. |

~~Five of the 34~~ **One of the 34** (`ResearchReportGenerated`) is **schema-only with no producer
anywhere in `src/`**, disclosed as such in its docstring and in `docs/TELEMETRY_STANDARD_V1.md`. Its
reason is specific to it and is recorded there: announcing a report as a *published research artifact*
is a different act from generating one, and Article 33 forbids a projection emitting telemetry about
itself.

**Revised 2026-07-30 (§12).** This section originally listed five: the four knowledge kinds
(`ResearchObservationCreated`, `CandidateFindingCreated`, `FindingReviewed`, `FindingStatusChanged`)
alongside `ResearchReportGenerated`, on the grounds that "the knowledge lifecycle they belong to is
explicitly a later phase" and that each "references its subject by id and carries no entity object
precisely so the later phase can model `ResearchObservation`/`ResearchFinding` freely without migrating
a schema guessed at here". That deferral has now been taken up, and it paid off exactly as intended: the
four kinds got real producers and real typed payloads without a single field being renamed or migrated,
because none had been guessed at. See §12.

Landing the closed vocabulary early is what made that possible — the enum did not have to be reopened.
This is the same disclosure `ObservationMapped` has carried since 2026-07-14.

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
* ~~**Five event kinds have no producer** (§6), by design, pending the knowledge lifecycle.~~ Four of
  the five got real producers on 2026-07-30 — see §12. `ResearchReportGenerated` is the remaining one,
  and its reason is now specific to it rather than shared.
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
* **Four streams and no export or retention.** `events.jsonl`, `htr_research_events.jsonl`,
  `htr_knowledge_events.jsonl` and `htr_knowledge_feedback_events.jsonl` (§13). Nothing prunes any of
  them and there is no knowledge export format — Phase 12, and the one gap the knowledge phases
  deliberately did not touch.
* **Research questions cannot yet be answered, superseded or withdrawn** (§13). The `Answered` status
  and its required finding pointer are enforced and `ResearchQuestion.answered_by` exists, but nothing
  calls it: answering needs a reviewed finding from the drafted experiment, which needs the run.

## 12. The knowledge lifecycle (added 2026-07-30)

The layer 10–12 half of this architecture, built on top of everything above. Full documentation:
[`docs/research-observations.md`](../research-observations.md),
[`docs/research-findings.md`](../research-findings.md),
[`docs/knowledge-lifecycle.md`](../knowledge-lifecycle.md).

`src/archivetrust/htr/knowledge/` — a new package sibling to `htr/corpus/` and `htr/experiment/`,
following this codebase's one-package-per-concern convention. Its `models.py` imports nothing but
`pydantic` and `domain.shared.ids`, which is §5's condition for typed embedding, so
`ResearchObservation` and `ResearchFinding` travel on the wire as **typed objects** rather than `record`
dicts and `HtrJournal.replay` reconstructs them field-for-field.

Four kinds gained real producers on `DurableHtrResearchStore`: `register_research_observation`,
`register_candidate_finding`, and `record_finding_transition` (which emits `FindingReviewed` then
`FindingStatusChanged` caused by it). Three event classes gained an optional, defaulted typed payload
field — `ResearchObservationCreated.observation`, `CandidateFindingCreated.finding`,
`FindingStatusChanged.finding` — the same additive shape `CanonicalResultCreated.canonical_result` used
in §5, so every pre-existing construction still validates. `FindingReviewed` carries no entity
deliberately: the reviewed state is the outcome, which `FindingStatusChanged` carries, and two copies of
one state in the log with nothing saying which is authoritative would be a bug rather than redundancy.

`HtrResearchStore` gained two projection buckets (`_observations`, `_findings`, both in
`_PROJECTION_BUCKETS`), the accessors `research_observations`/`research_observation`/`findings`/
`finding`/`findings_contradicting`, and `advance_finding` — the `advance_experiment_run` pattern applied
to the one mutation a finding permits. `HtrJournal._apply` gained three branches and lost four names from
its no-op list; `FindingReviewed` stays a documented no-op.

**A third stream.** Knowledge events go to `htr_knowledge_events.jsonl`, not into the run's
`htr_research_events.jsonl`, because that file is a committed research artifact documented as exactly one
run's 100 events and read as such by a test — appending would falsify its own description. §7's "two
streams, one mechanism" reasoning applies unchanged: same sink class, same append-only guarantees, same
hash-chain sidecar. `causation_id`s cross the file boundary and `HtrJournal().replay(run + knowledge)`
reconstructs one graph, proven in `tests/htr/knowledge/test_baseline_knowledge.py`.

**What the enforcement actually is.** The three arrows of event-model doc §1 are blocked by construction:
`supporting_evidence` and `supporting_observations` are both non-empty-by-construction, and
`ResearchFinding` cannot be *constructed* at any status past `Draft`/`Candidate` — enforced twice, on
`create` and on a model validator, so bypassing the classmethod does not help. `ResearchScope` makes a
scoped result structurally unreadable as a general one: `sample_size` is a derived property with
`extra="forbid"` so it cannot be asserted, `covered_unit_ids` must enumerate real ids with wildcards
refused, and every named method must carry its exact model revision.

**Honest limits of the demonstration**, recorded here as well as in `docs/knowledge-lifecycle.md`: no
`Supported` and no `Superseded` finding exists, because both need a second experiment run and this
repository has one; the four demonstrated transitions carry a single reviewer, the repository maintainer,
with no independent review or adjudication.

`ResearchReportGenerated` remains the single producerless kind (§6).

## 13. The research-question feedback loop and its frontend (added 2026-07-30)

Full documentation: [`docs/knowledge-lifecycle.md`](../knowledge-lifecycle.md), sections "The feedback
loop" and "The frontend".

**Two new kinds, two new producers.** `ResearchQuestionRaised` and `ExperimentDraftedFromQuestion`, both
on `DurableHtrResearchStore` (`register_research_question`,
`record_experiment_drafted_from_question`). `ResearchQuestion` and `Hypothesis` live in
`htr/knowledge/models.py`, which still imports nothing but `pydantic` and `domain.shared.ids`, so both
travel on the wire as **typed objects** under §5's rule and `HtrJournal.replay` reconstructs a question
and its hypotheses field-for-field.

`ExperimentDraftedFromQuestion` deliberately carries the question but *not* the `Experiment` or
`ExperimentVersion`: both are announced with their full objects by
`ExperimentCreated`/`ExperimentVersionCreated`, and a second copy would put two versions of one entity
state in the log with nothing saying which is authoritative — the same reason `FindingReviewed` carries
no finding. It carries the *post-draft* question, which `HtrJournal` projects via
`advance_research_question`: the `ExperimentRunCompleted`/`FindingStatusChanged` pattern applied to the
one entity that legitimately gains pointers after registration. `advance_research_question` refuses a
question whose `ResearchQuestionRaised` is missing, so an incomplete log surfaces
(`test_replay_refuses_to_invent_a_question_a_draft_event_advances`) rather than producing a question that
appears already under investigation with nothing recording that it was ever asked.

`HtrResearchStore` gained one projection bucket (`_research_questions`, in `_PROJECTION_BUCKETS`),
`register_research_question`/`advance_research_question`, and three accessors:
`research_questions(status=, originating_observation_id=, originating_finding_id=)`,
`research_questions_for_experiment(experiment_id)` and `research_question(id)`. The last is the reverse
of `ResearchQuestion.created_experiment_id`, so an experiment surface can answer "why does this exist?"
without parsing `pipeline_configuration_ref`. Both directions are stored and neither is derived from the
other.

**A fourth stream.** Feedback events go to `htr_knowledge_feedback_events.jsonl`, because §12's
`htr_knowledge_events.jsonl` is documented as exactly 18 events of 4 kinds and read as that by a test.
§7's "two streams, one mechanism" reasoning applies unchanged, now four times over.
`scripts/register_research_question.py` *replays* the committed run and knowledge logs to find the real
observation and finding, rather than re-registering the baseline knowledge — which would mint fresh ids
and silently invalidate every id the three knowledge documents quote.

**One additive change outside this package.** `htr/experiment/baseline_template.py` is unmodified;
`htr/knowledge/questions.py::DraftedExperimentDefinition` *subclasses*
`BaselineExperimentDefinition`, adding four id fields, and reuses `build_baseline_experiment` for the
construction itself. Because the base model's own shape is untouched, no committed artifact changes
meaning, and `build_research_report_from_store`'s
`BaselineExperimentDefinition.model_validate_json(pipeline_configuration_ref)` keeps parsing both the old
refs and the new one (Pydantic ignores extra keys by default) —
`test_the_drafted_definition_still_parses_as_a_baseline_definition` asserts it.

**The frontend.** `presentation/htr_knowledge_viewmodel.py` over this projection, rendered by
`clients/desktop_v2/htr_knowledge_page.py` as `DesktopV2Page.RESEARCH_KNOWLEDGE`, the eighth research
surface. It reuses `htr_evidence_viewmodel.py::HtrEvidenceChainViewModel` to resolve an evidence
reference to a breadcrumb rather than walking the corpus a second time, and its
`_TARGET_PAGE_BY_EVIDENCE_KIND` table is exhaustive over `EvidenceReferenceKind` with two honest
`None`s: no page in this application replays the HTR telemetry stream by event id (the operational
Document Evidence page replays per Archive Object, and HTR research events carry the `htr:research`
sentinel), and an `external_document` reference names a committed repository file rather than a record
this application stores. Those two are why a `telemetry_event` or `evidence_record` reference comes back
`resolved=False` **with the reason it is a property of the read model rather than of the evidence** —
which is not the same fact as a dangling id, and the two are reported differently.
