# ArchiveTrust Telemetry Standard v1

**Profile id:** `archivetrust.telemetry.standard.v1`  
**Status:** Current production behavior profile.  
**Date:** 16 July 2026  
**Reference implementation:** `src/archivetrust/domain/telemetry/events.py`, `src/archivetrust/application/journal.py`, `src/archivetrust/infrastructure/storage/telemetry_sink.py`, `src/archivetrust/infrastructure/storage/integrity.py`.

## 1. Boundary

This profile describes only behavior implemented in `src/` at the time of writing. It does not
claim that every aspiration in `ARCHITECTURE_TELEMETRY_STANDARD.md` is complete.

An implementation conforms to this profile when it can:

- persist immutable document-scoped telemetry events;
- parse events by `kind`;
- replay document reasoning without re-running providers;
- preserve raw evidence, including rejected evidence;
- distinguish provider silence, no-observation outcomes, and failure;
- record alignment, exclusion, comparison, confidence, canonical, correction, review, and packet
  closure facts;
- verify file-backed event-stream integrity when hash-chain sidecars exist.

## 2. Event Stream Model

The canonical stream is append-only JSON Lines. Each line contains one serialized event. Event order
is semantic: replay consumes events in file order.

Every event has:

- `kind`
- `event_id`
- `document_ref`
- `schema_version`
- optional `recorded_at`
- optional policy/version stamps:
  - `reconciliation_policy_version`
  - `capability_matrix_version`
  - `confidence_policy_version`
  - `feedback_policy_version`
  - `alignment_algorithm_version`

Unknown fields are not part of this profile. Unknown `kind` values must not be silently accepted by
a conforming v1 parser.

## 3. Production Event Kinds

The v1 document-scoped event vocabulary is:

| Event kind | Purpose |
|---|---|
| `ProviderObservationAttempted` | Records that a provider was invoked and what invocation outcome was observed. |
| `EvidenceRejected` | Preserves malformed/rejected raw provider output. |
| `EvidenceCreated` | Records immutable raw provider evidence. |
| `ObservationCreated` | Records a mapped ontology observation. |
| `ObservationMapped` | Schema-ready mapping event for provider-to-ontology lineage. |
| `AlignmentAttempted` | Records candidate and selected observations for an alignment group. |
| `ObservationAligned` | Records an observation placed into an alignment group. |
| `ObservationLeftUnaligned` | Records an observation explicitly left unaligned. |
| `ObservationCompared` | Records observations compared for a semantic slot. |
| `ObservationAccepted` | Records an observation accepted into a canonical observation. |
| `ObservationRejected` | Records an observation rejected with reason. |
| `ObservationMerged` | Records merged observations. |
| `AgreementCalculated` | Records comparison confidence for a semantic slot. |
| `CanonicalDecisionCreated` | Records a canonical observation decision. |
| `ConfidenceChanged` | Records provider, comparison, or canonical confidence changes. |
| `KnowledgeMerged` | Records merged knowledge ids. |
| `KnowledgeDiscarded` | Records discarded knowledge ids. |
| `CanonicalDocumentCreated` | Records an assembled canonical document snapshot. |
| `HumanCorrectionSubmitted` | Records a human correction proposal. |
| `HumanCorrectionApplied` | Records the superseding canonical observation created from a correction. |
| `DatasetCandidateCreated` | Records a correction-linked dataset candidate. |
| `ProvenanceContextEstablished` | Records run/environment context for a document. |
| `CandidateExcluded` | Records one structurally excluded candidate pair. |
| `CandidateExcludedBatch` | Records many structurally excluded candidate pairs compactly. |
| `ReviewOutcomeRecorded` | Records one terminal human review decision. |
| `ReviewPacketCreated` | Records creation of a durable operational review packet. |
| `ReviewPacketDispatched` | Records that a projected packet was handed to a reviewer. |
| `ReviewPacketOpened` | Records that the assigned reviewer opened or claimed the packet. |
| `ReviewPacketClosed` | Records terminal lifecycle closure for a dispatched review packet. |
| `SegmentationRunCompleted` | Records one segmentation-adapter run over a page (docs/htr-domain-design.md §7). |
| `MethodRunCompleted` | Records one HTR method run reaching a terminal outcome. |
| `ReviewSubmissionRecorded` | Records one blind ground-truth review submission. |
| `AdjudicationRecorded` | Records resolution of a disagreeing ground-truth agreement result. |
| `CanonicalResultCreated` | Records creation of a line/region-level canonical HTR result. |

### HTR research-persistence kinds

Added by the HTR telemetry/provenance pass (`docs/architecture/htr-telemetry.md`;
mapping in `docs/architecture/htr-event-model.md` §3). These are **research-scoped, not
document-scoped**: most concern a project, dataset, experiment or metric rather than one Archive
Object, and carry the sentinel `document_ref` value `htr:research` with their real scoping in the
dedicated `project_id`/`dataset_id`/`experiment_id`/`experiment_run_id`/`method_run_id`/`subject_id`
fields. `PageRegistered` and `DocumentRegistered` are the exceptions and carry a genuine
`archive_object_ref`. They are appended to a separate stream (`htr_research_events.jsonl`) through
the same `FileTelemetrySink`, and replayed by `application/htr_journal.py::HtrJournal`, not by
`Journal`.

| Event kind | Purpose |
|---|---|
| `ResearchProjectCreated` | Records registration of a research project. |
| `DatasetCreated` | Records registration of a dataset within a project. |
| `DatasetVersionCreated` | Records an immutable dataset-membership snapshot. |
| `CollectionCreated` | Records registration of a document collection within a dataset. |
| `DocumentRegistered` | Records one Archive Object joining a collection. |
| `PageRegistered` | Records registration of one page of a document. |
| `RegionDetected` | Records a segmentation-detected region on a page. |
| `TextLineDetected` | Records a segmentation-detected text line within a region. |
| `InputCropCreated` | Records a content-addressed input crop for one text line. |
| `TranscriptionConventionRegistered` | Records a versioned transcription convention. |
| `ExperimentCreated` | Records registration of an experiment. |
| `ExperimentVersionCreated` | Records one versioned experiment configuration. |
| `ExperimentRunStarted` | Records the start of an experiment run; its id is the run's correlation id. |
| `ExperimentRunCompleted` | Records an experiment run's terminal success. |
| `ExperimentRunFailed` | Records that an experiment run could not complete. |
| `MethodRunStarted` | Records one method-by-input execution, carrying the full method-run record. |
| `MethodRunFailed` | Records a method run's failure, carrying the preserved failure record. |
| `RawMethodResultRecorded` | Records transcript stage 1: raw method output. |
| `ParsedMethodResultRecorded` | Records transcript stage 2: parsed output. |
| `NormalizedMethodResultRecorded` | Records transcript stage 3: normalized output. |
| `ReviewedResultRecorded` | Records transcript stage 4: human-reviewed text (a distinct lineage). |
| `MetricDefinitionRegistered` | Records a versioned metric definition. |
| `MetricCalculated` | Records one metric computed for one method run. |
| `ReliabilityIssueClassified` | Records a reliability/failure classification for a method run. |
| `GroundTruthTextRecorded` | Records the resolved reference transcription for one text line. |
| `ReviewAssigned` | Records assignment of one blind reviewer to one target. |
| `AgreementCalculatedHtr` | Records reviewer-pair textual agreement (distinct from `AgreementCalculated`). |
| `ReproducibilityManifestRecorded` | Records the reproducibility manifest for one experiment run. |
| `ExternalResultImported` | Records a manually imported external result's provenance. |
| `ResearchObservationCreated` | Records an extracted research observation; carries the full `ResearchObservation`. |
| `CandidateFindingCreated` | Records a candidate research finding; carries the full `ResearchFinding`. |
| `FindingReviewed` | Records a named human's review of a finding. Carries no entity — the outcome is on the `FindingStatusChanged` it causes. |
| `FindingStatusChanged` | Records a finding's status transition; carries the whole post-transition finding, including its full revision history. |
| `ResearchQuestionRaised` | Records a research question raised by an observation, finding or contradiction; carries the full `ResearchQuestion` including its attached hypotheses. `actor_type` is `human`. |
| `ExperimentDraftedFromQuestion` | Records that an `ExperimentVersion` was drafted from a research question. Carries the post-draft question, and the experiment/version by id only — both are announced with their full objects by `ExperimentCreated`/`ExperimentVersionCreated`. Records no execution: a drafted version has no `ExperimentRun`. |
| `ResearchReportGenerated` | Schema-ready: records generation of a research report. No producer yet. |
| `ImageNormalizationStarted` | Records the start of the versioned `RgbNormalization` preprocessing stage for one source page image. |
| `ImageNormalizationCompleted` | Records successful RGB normalization; carries the full `NormalizedPageArtifact` provenance (original/normalized hashes, source and output color modes, configuration hash). |
| `ImageNormalizationFailed` | Records a failed normalization attempt (decode failure, invalid output, unwritable artifact) as durable evidence — never silently swallowed, and the unnormalized original is never substituted for the failed derivative. |
| `DerivedImageArtifactCreated` | Records that a normalized page image artifact now exists as a distinct, hash-addressed derivative of its original, never overwriting it. |
| `MethodResearchStatusChanged` | Records a new `htr.research_status.ResearchPhase` declaration — which methods are active/archived/inactive/experimental/unavailable in the current research phase, and why. Carries the full phase. |
| `LoghiEnvironmentValidated` | Records one Loghi `validate_environment()` probe result (Docker/WSL2 presence, resolved execution mode, pin placeholder status) as a dict, since the typed report lives in `providers/loghi/` and this domain module may not import `providers/*`. |
| `LoghiPipelineStarted` | Records that one Loghi pipeline invocation began for one `MethodRun` — the input reference and the pinned component-versions configuration (as a dict, same reason as above). |
| `LoghiStageStarted` | Records that one Loghi pipeline stage (Laypa / Loghi Tooling / Loghi HTR) began. |
| `LoghiStageCompleted` | Records one Loghi pipeline stage's successful completion. |
| `LoghiStageFailed` | Records one Loghi pipeline stage's failure, preserved as durable evidence — never silently swallowed. |
| `LoghiPageXmlGenerated` | Records that Loghi produced a final PAGE XML output for one page — the source XML's own content hash and declared schema version. |
| `DomainRelationshipRecorded` | Records an `ExperimentVersion`'s `corpus_language`/`method_primary_language_domain`/`domain_relationship` classification as its own queryable fact. |
| `CrossDomainComparisonCreated` | Records an `ExperimentComparisonGroup` — a named parent grouping over several `Experiment`s (e.g. the four Lion-vs-Loghi cells) that never merges their results. |
| `TrainingSessionStarted` | Records that one resumable Loghi fine-tuning training session began — `run_id`, `session_id`, `configuration_hash`, and the initial epoch/global-step/source-checkpoint it is resuming from (or `0`/the pinned parent checkpoint, for a run's first-ever session). |
| `TrainingSessionCheckpointed` | Records one epoch's checkpoint (`"latest"` / `"best_val"` / `"end_of_session"`) — epoch, global step, checkpoint directory, and the real train/val CER/WER this epoch reported, if any. |
| `TrainingSessionCompleted` | Records a training session's end — the honest `stop_reason` (`"time_budget_reached"` / `"stop_requested"` / `"epoch_would_not_fit"` / `"epoch_failed"` / `"target_epochs_reached"`), final epoch/global step, and the latest/best checkpoint directories. |
| `TrainingSessionFailed` | Records a training session's real process/container failure — distinct from any clean `stop_reason` `TrainingSessionCompleted` covers. |
| `RunWarningRecorded` | Records one `run_health.py` finding (e.g. `"stale_heartbeat"`, `"low_disk_space"`) as durable evidence, so it is still visible after a dashboard or trainer restart, not just a live-computed value. |

**Updated 2026-08-01.** Nine kinds added for the active-method transition and Loghi integration
(`docs/loghi-integration-audit.md`), each with a real producer on `DurableHtrResearchStore` from the
outset. `ExternalResultImported` and `ReproducibilityManifestRecorded` (already in this table) are
reused as-is for Transkribus imports and Loghi's reproducibility manifest respectively — neither needed
a Loghi-specific counterpart. Three more kinds added the same day for the Swedish Loghi fine-tuning
pilot (`docs/methods/loghi-swedish-finetuning.md`): `TrainingSessionStarted`,
`TrainingSessionCheckpointed`, `TrainingSessionCompleted` — a training session's lifecycle (resumable
across process boundaries, one checkpoint per epoch) is materially different from an inference
`MethodRun`'s, so it gets its own three kinds rather than reusing `MethodRunCompleted`. Two more added
the same day for the read-only training dashboard (`docs/methods/loghi-training-dashboard.md`):
`TrainingSessionFailed`, `RunWarningRecorded`.

**Updated 2026-07-30.** The four knowledge kinds above previously read "Schema-ready … No producer
yet", and this section previously said "the last five are deliberately declared without producers …
the research-knowledge lifecycle they belong to is a later phase". That phase is now built
(`docs/architecture/htr-telemetry.md` §12, `docs/knowledge-lifecycle.md`) and all four have real
producers on `DurableHtrResearchStore`. Landing the closed vocabulary early is what let them acquire
typed payloads without a single field being renamed.

`ResearchReportGenerated` is now the only kind here without a producer, and its reason is specific to
it rather than shared: announcing a report as a *published research artifact* is a different act from
generating one, and Article 33 forbids a projection emitting telemetry about itself.
`ObservationMapped` above has the same disclosed-without-a-producer status.

### Correlation and Causation

Every event kind in this standard, old and new, carries two optional fields:

| Field | Meaning |
|---|---|
| `correlation_id` | One value shared by every event in the same logical unit of work. For HTR experiment execution this is the `ExperimentRun` id. |
| `causation_id` | The `event_id` of the event that *directly caused* this one, forming an explicit provenance DAG. |

Both are `null` when no correlated unit of work applies, which includes every pre-existing
document-scoped producer: they are not backfilled, and a `null` here means "no recorded linkage",
never an inferred one. A conforming reader must not infer causation from append order or from shared
domain ids when these fields are absent.

## 4. Replay Requirements

Replay reconstructs document state from events alone. It must not:

- invoke providers;
- create missing evidence;
- infer absent provider invocations;
- mutate historical events;
- emit telemetry for ordinary query-time projections.

Replay must reconstruct at least:

- evidence records;
- observations and provider observation graphs;
- canonical observations and supersession history;
- canonical document snapshots;
- confidence evolution;
- comparison decisions and agreement calculations;
- alignment attempts and terminal alignment states;
- candidate exclusions, including batched exclusions;
- review outcomes.

## 5. Encoding Requirements

### Candidate Exclusions

Writers should use `CandidateExcludedBatch` for new high-volume exclusion sets. Readers must support
both `CandidateExcluded` and `CandidateExcludedBatch`. Replay must expand batched exclusions to the
same query surface as legacy single-pair exclusions.

Current production exclusion reason codes are:

- `AMBIGUOUS_MULTI_PER_PROVIDER`: structural ordinal-position ambiguity.
- `SCOPE_MISMATCH`: F5 scope-compatibility policy gate; `structural` is `false`.

### Raw Output Blob Indirection

File-backed sinks may replace large raw-output fields with content-addressed blob references.
Readers must transparently rehydrate:

- `EvidenceCreated.evidence.raw_output`
- `EvidenceRejected.raw_output`

The logical event object after read must preserve the original raw string.

### Review Closure

`ReviewOutcomeRecorded` records what a human decided. `ReviewPacketCreated`,
`ReviewPacketDispatched`, `ReviewPacketOpened`, and `ReviewPacketClosed` record the lifecycle of a
packet handed to a reviewer.

Queue construction, triage, packet assembly, and aging queries remain deterministic projections and
must not emit telemetry merely because they were run.

## 6. Integrity Requirements

For file-backed domain telemetry, the reference implementation writes a detached hash-chain sidecar:

- sidecar filename: `<stream>.chain.json`
- algorithm: `sha256`
- protected payload: exact non-empty JSONL line bytes in order
- root hash: chained over per-line hashes

Verification must detect:

- bit flips;
- truncation;
- line reorder.

Exported JSON artifacts may use detached file-digest sidecars:

- sidecar filename: `<export>.integrity.json`
- algorithm: `sha256`
- recorded fields: source name, byte size, digest.

## 7. Compatibility

Conforming v1 readers must continue to parse older v1-compatible streams that contain:

- legacy `CandidateExcluded` events;
- inline raw-output fields;
- no hash-chain sidecar;
- historical events with missing optional `recorded_at` or policy-version fields.

Absence of an optional field is represented as unknown or historical, not guessed.

## 8. Non-Goals

This v1 profile does not standardize:

- acquisition telemetry;
- research telemetry;
- public-key signatures;
- PROV or OpenLineage export format;
- provider SDK conformance;
- future provider scope declarations;
- calibration decision policies.

Those are adjacent ArchiveTrust capabilities, not required for document-scoped telemetry v1
conformance.
