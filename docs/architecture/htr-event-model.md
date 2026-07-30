# HTR Event Model — Evidence-to-Knowledge Mapping

Phase 2 deliverable of the telemetry/knowledge follow-up (`docs/htr-telemetry-knowledge-gap-analysis.md`
is Phase 1; read that first). Documents the mapping this follow-up implements:

```
Existing ArchiveTrust evidence concepts → HTR research entities → Telemetry events →
Durable repositories → Read models → Research observations → Research findings
```

## 1. Information layers (explicit, not undifferentiated logs)

Per the follow-up's requirement to distinguish layers rather than store everything as unstructured
JSON, this is the layering implemented:

| Layer | Represented by | Durable? | Mutable? |
|---|---|---|---|
| 1. Operational logging | existing app logs (unrelated to this follow-up) | n/a | n/a |
| 2. Security logging | `worker/` security/ACL logs (unrelated to this follow-up) | n/a | n/a |
| 3. Execution telemetry | `HtrTelemetryEvent` subtypes (new, §3) appended to `FileTelemetrySink` | Yes, append-only | No — append-only |
| 4. Provenance evidence | `Evidence` (retained, already HTR-extended) | Yes | No — content-addressed, immutable |
| 5. Segmentation evidence | `RegionDetected`/`TextLineDetected`/`InputCropCreated` events + `htr/corpus/models.py` entities | Yes | No |
| 6. Method-result evidence | `RawMethodResultRecorded`/`ParsedMethodResultRecorded`/`NormalizedMethodResultRecorded` events | Yes | No |
| 7. Evaluation evidence | `MetricCalculated`/`ReliabilityIssueClassified` events | Yes | No |
| 8. Human-review evidence | `ReviewAssigned`/`ReviewSubmitted`/`AgreementCalculated`/`AdjudicationRecorded` events | Yes | No |
| 9. Canonicalization evidence | `CanonicalResultCreated` event | Yes | No |
| 10. Research observations | `ResearchObservation` entity, `ResearchObservationCreated` event | Yes | No — new observations supersede scope, don't overwrite |
| 11. Candidate findings | `ResearchFinding` (status=Candidate), `CandidateFindingCreated` event | Yes | Status transitions only, via new `FindingRevision` events |
| 12. Reviewed research knowledge | `ResearchFinding` (status=Supported/Disputed/...), `FindingReviewed`/`FindingStatusChanged` events | Yes | Status transitions only |

**Hard rule enforced by construction, not convention**: a telemetry event (layer 3) never directly
becomes a `ResearchObservation` (layer 10) — an explicit extraction step reads events/durable
records and constructs an observation with its own evidence-link list. A `ResearchObservation` never
auto-promotes to a `ResearchFinding` — a separate, explicit "candidate finding" construction step is
required, and a finding never auto-promotes past `Candidate` status without a `KnowledgeReview`
event recording a human actor. See `docs/knowledge-lifecycle.md`.

## 2. Existing evidence concepts → what they become

| Existing concept | Role in the HTR event model |
|---|---|
| `Evidence` (content-addressed raw capture) | Unchanged. Every `RawMethodResultRecorded` event references an `Evidence.id`; the event does not duplicate the raw payload (`FileTelemetrySink` already externalizes payloads >4KB into the blob store — reused as-is). |
| `Observation` (typed claim referencing Evidence) | Unchanged; the pre-existing `TEXT_LINE`/`REGION`/`RAW_TRANSCRIPTION`/`PARSED_TRANSCRIPTION`/`NORMALIZED_TRANSCRIPTION` ontology types (added in the prior transformation) are the Observation-layer counterparts of the new events below — an event records *that* something happened (with a timestamp, correlation id, actor), an Observation records the *typed content claim* it produced. Both are written; neither replaces the other. |
| `CanonicalObservation`/`CanonicalDocument` | Pattern reused by the HTR-specific `CanonicalResult` (already modeled in `domain/canonical/result.py`); `CanonicalResultCreated` is its telemetry counterpart, now actually emitted (previously defined-but-unused). |
| `FileTelemetrySink` | Reused directly, unmodified, as the durable append-only store for all new HTR event kinds. No parallel sink is created. |
| `Journal.replay` | Extended (not replaced) with new `_apply` branches for HTR event kinds, producing an `HtrJournalState` alongside the existing `JournalState` (kept separate because they reconstruct disjoint entity sets — see `docs/architecture/htr-telemetry.md` §2 for why a single merged state was rejected). |
| `content_address(...)` | Reused for `InputCrop.hash` (already implemented) and extended to identify `MethodRun` inputs; no second hashing scheme introduced. |
| `HashChainAppender` | Reused unmodified — the same tamper-evidence sidecar mechanism applies to the HTR event stream, since it operates on `FileTelemetrySink` files generically, not on event *kind*. |
| `FileGroundTruthStore` (evaluation/ground_truth.py) | Used as the structural precedent for the new `HtrDurableStore` (§4 of `docs/architecture/htr-telemetry.md`) — append validate-then-write, `latest()` supersession resolution, `export_with_provenance()` integrity sidecar. Not merged into it directly (different entity shapes), but the same idiom. |
| `WorkspaceStore` (whole-object JSON round-trip) | Used for the small number of coarse, rarely-revised registration entities (`ResearchProject`, `Dataset`, `Collection`) where full event-sourced history adds no research value — see gap analysis §11. |

## 3. New telemetry event kinds (this follow-up)

Extends `TelemetryEventKind` (existing enum) with HTR-specific kinds, split into two groups:

**Already defined in the prior transformation, now actually wired to producers and replay** (no
rename — extending existing schema per the follow-up's explicit "do not rename, migrate semantics
correctly" instruction):
`SEGMENTATION_RUN_COMPLETED`, `METHOD_RUN_COMPLETED`, `REVIEW_SUBMISSION_RECORDED`,
`ADJUDICATION_RECORDED`, `CANONICAL_RESULT_CREATED`.

**New in this follow-up** (adapted to this project's naming convention — PascalCase event class,
SCREAMING_SNAKE enum member — not copied mechanically from the task brief's illustrative names):
`RESEARCH_PROJECT_CREATED`, `DATASET_CREATED`, `DATASET_VERSION_CREATED`, `DOCUMENT_REGISTERED`,
`PAGE_REGISTERED`, `REGION_DETECTED`, `TEXT_LINE_DETECTED`, `INPUT_CROP_CREATED`,
`TRANSCRIPTION_CONVENTION_REGISTERED`, `EXPERIMENT_CREATED`, `EXPERIMENT_VERSION_CREATED`,
`EXPERIMENT_RUN_STARTED`, `EXPERIMENT_RUN_COMPLETED`, `EXPERIMENT_RUN_FAILED`,
`METHOD_RUN_STARTED`, `METHOD_RUN_FAILED`, `RAW_METHOD_RESULT_RECORDED`,
`PARSED_METHOD_RESULT_RECORDED`, `NORMALIZED_METHOD_RESULT_RECORDED`, `METRIC_CALCULATED`,
`RELIABILITY_ISSUE_CLASSIFIED`, `REVIEW_ASSIGNED`, `AGREEMENT_CALCULATED_HTR` (suffixed — the
pre-existing `AGREEMENT_CALCULATED` kind means something different, cross-provider structural
agreement, not reviewer-pair textual agreement; reusing the name for a different meaning would
violate the "do not rename when semantics differ" rule, so this is a deliberately distinct kind, not
a rename), `RESEARCH_OBSERVATION_CREATED`, `CANDIDATE_FINDING_CREATED`, `FINDING_REVIEWED`,
`FINDING_STATUS_CHANGED`, `REPRODUCIBILITY_MANIFEST_RECORDED`, `EXTERNAL_RESULT_IMPORTED`,
`RESEARCH_REPORT_GENERATED`.

Every event carries: `event_id`, `kind`, `schema_version`, `timestamp`, `project_id`, `dataset_id`
(optional — not all events are dataset-scoped), `dataset_version_id` (optional), `experiment_id`
(optional), `experiment_version_id` (optional), `experiment_run_id` (optional), `method_run_id`
(optional), `subject_id` (the document/page/region/line/crop id the event is about, when
applicable), `actor_type` (`system`/`human`/`method`), `actor_id` (optional), `source_component`,
`correlation_id`, `causation_id`, `application_commit`, `payload` (event-specific fields), plus
`environment`-related fields only where meaningful (method-execution events).

## 4. Correlation and causation — concrete mechanism

`correlation_id`: one value shared by every event belonging to the same logical unit of work — for
the baseline experiment, one correlation id per `ExperimentRun` (all events for that run's segments,
method runs, evaluations, reviews share it). `causation_id`: the `event_id` of the event that
directly caused this one (e.g. a `MethodRunCompleted` event's `causation_id` is the
`MethodRunStarted` event's `event_id`; a `MetricCalculated` event's `causation_id` is the
`ParsedMethodResultRecorded`/`NormalizedMethodResultRecorded` event it evaluated). This gives an
explicit, queryable DAG rather than the "infer from timestamps and shared domain ids" approach the
gap analysis found the pre-existing system relies on — implemented as a genuinely new mechanism
(§1's "do not create a duplicate telemetry system, but this concept did not exist at all before"),
added as new optional fields on the shared `TelemetryEvent` base rather than a parallel envelope.

## 5. What is explicitly NOT changed

- The pre-existing ~25 OCR-era telemetry event kinds, their producers, and `Journal`'s existing
  branches: untouched.
- `Evidence`, `Observation`, `CanonicalObservation`, `CanonicalDocument`: untouched (already
  HTR-ready per the gap analysis).
- `htr/evaluation/*`'s pure metric/failure-classification functions: untouched (already correctly
  pure — see gap analysis §7).
- `htr/corpus/models.py`, `htr/experiment/models.py`: the Pydantic model *definitions* are retained
  as-is; only their persistence path changes (they gain durable backing, not new fields, except
  where a genuinely new concept like `ResearchObservation` requires a new model).

Full entity relationship diagram and the durable-repository implementation this mapping drives are
in `docs/architecture/htr-telemetry.md` (Phase 3).
