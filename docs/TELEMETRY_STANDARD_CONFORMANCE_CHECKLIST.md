# ArchiveTrust Telemetry Standard v1 Conformance Checklist

**Profile:** `archivetrust.telemetry.standard.v1`  
**Status:** Checklist for implementers and reviewers.

## 1. Event Vocabulary

A conforming implementation:

- [ ] Uses a closed `kind` vocabulary.
- [ ] Rejects or quarantines unknown `kind` values.
- [ ] Records `event_id`, `document_ref`, `schema_version`, and optional `recorded_at`.
- [ ] Preserves policy/version stamps where applicable.
- [ ] Supports every event kind listed in `TELEMETRY_STANDARD_V1.md`.

## 2. Storage

A conforming implementation:

- [ ] Persists document-scoped events append-only.
- [ ] Preserves event order.
- [ ] Reads events without mutating them.
- [ ] Can operate with no integrity sidecar for historical streams.
- [ ] Detects bit flips, truncation, and reorder when hash-chain sidecars are present.

## 3. Replay

A conforming implementation:

- [ ] Replays without invoking providers.
- [ ] Reconstructs Evidence.
- [ ] Reconstructs Observations.
- [ ] Reconstructs CanonicalObservations and supersession history.
- [ ] Reconstructs CanonicalDocuments.
- [ ] Reconstructs confidence evolution.
- [ ] Reconstructs alignment states.
- [ ] Reconstructs candidate exclusions from both legacy and batched encodings.
- [ ] Reconstructs review outcomes.

## 4. Provider Outcomes

A conforming implementation:

- [ ] Records attempted provider invocations.
- [ ] Distinguishes no observations from failure.
- [ ] Preserves failure reason when known.
- [ ] Preserves rejected raw output.
- [ ] Never infers a provider invocation from later objects.

## 5. Encoding Repair

A conforming implementation:

- [ ] Reads `CandidateExcluded`.
- [ ] Reads `CandidateExcludedBatch`.
- [ ] Produces equivalent replay state for legacy and batched candidate exclusions.
- [ ] Preserves exclusion `basis_code`, including `AMBIGUOUS_MULTI_PER_PROVIDER` and `SCOPE_MISMATCH`.
- [ ] Preserves the exclusion `structural` flag; `SCOPE_MISMATCH` is a policy exclusion, not a structural geometry exclusion.
- [ ] Rehydrates externalized raw-output blobs transparently.
- [ ] Does not truncate raw output.

## 6. Review Closure

A conforming implementation:

- [ ] Records `ReviewOutcomeRecorded` for terminal human decisions.
- [ ] Records `ReviewPacketDispatched` when a packet is actually handed to a reviewer.
- [ ] Records `ReviewPacketClosed` when dispatched packet lifecycle ends.
- [ ] Supports closure kinds `resolved`, `deferred`, `expired`, and `withdrawn`.
- [ ] Keeps triage, packet assembly, and aging as side-effect-free projections.

## 7. Export And Integrity

A conforming implementation:

- [ ] Can emit versioned canonical JSON exports.
- [ ] Can emit detached digest sidecars for export artifacts.
- [ ] Keeps code license and dataset license distinct.
- [ ] Treats telemetry archives as potentially containing document-derived data.

## 8. PROV/OpenLineage Mapping

A conforming implementation is not required to export PROV-O or OpenLineage.

If it does, it should:

- [ ] Preserve ArchiveTrust ids.
- [ ] Map Evidence/Observation/CanonicalObservation/CanonicalDocument to PROV entities.
- [ ] Map provider invocation, comparison, reconciliation, and review to PROV activities.
- [ ] Map document processing runs to OpenLineage runs/jobs at a declared granularity.
- [ ] Carry ArchiveTrust-specific semantics as explicit extension attributes or facets.
- [ ] Document any information loss.

## 9. Non-Conformance Examples

The following are not conforming:

- accepting unknown event kinds silently;
- dropping rejected raw output;
- treating missing provider output as provider failure without an attempted invocation event;
- rewriting historical events in place;
- making replay call live providers;
- flattening review outcome and packet closure into one ambiguous field;
- reporting an integrity sidecar as verified without recomputing the chain or digest.
