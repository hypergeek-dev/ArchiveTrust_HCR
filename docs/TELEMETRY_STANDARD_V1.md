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
