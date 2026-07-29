"""Replays Benchmark #1's full corpus through the *current* Comparison Engine, Confidence Engine,
Canonical Assembly, and Review Engine, using `review.regenerate.regenerate_review_queue`
(`ROADMAP.md`'s "Make review-queue regeneration a first-class replay capability", 2026-07-14).

Read-only over `archivetrust_data/` (Article 34's standing constraint): no provider is invoked, no
document is reprocessed, nothing is written back to the workspace. Only persisted
`ProviderObservationAttempted`/`EvidenceRejected`/`EvidenceCreated`/`ObservationCreated` telemetry
is replayed as input to today's Comparison -> Confidence -> Canonical -> Review pipeline; the
*original* recorded Canonical layer is used only to compute the "before" side of the comparison,
never mutated or re-derived from.

Writes only to `benchmarks/benchmark_1_replay_results.json` (new file) -- never overwrites
`benchmarks/BENCHMARK_1_REPORT.md`, `benchmarks/calibration_corpus_1.jsonl`, or any existing review
packet/telemetry artifact.

Run: `python scripts/replay_benchmark_1_current_engine.py`
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

WORKSPACE_DIR = REPO_ROOT / "archivetrust_data" / "workspaces" / "46e94f15d73347c2be79f1527e646c11"
TELEMETRY_PATH = WORKSPACE_DIR / "telemetry" / "events.jsonl"
OUTPUT_PATH = REPO_ROOT / "benchmarks" / "benchmark_1_replay_results.json"


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _confidence_bucket(value: float) -> str:
    if value < 0.2:
        return "0.0-0.2"
    if value < 0.4:
        return "0.2-0.4"
    if value < 0.6:
        return "0.4-0.6"
    if value < 0.8:
        return "0.6-0.8"
    return "0.8-1.0"


def main() -> None:
    from archivetrust.application.journal import Journal
    from archivetrust.application.pipeline import run_comparison_and_assembly
    from archivetrust.domain.comparison.capability_matrix_data import production_capability_matrix
    from archivetrust.domain.comparison.policy import ReconciliationPolicy
    from archivetrust.domain.confidence.policy import ConfidencePolicy
    from archivetrust.domain.graph.provider_graph import ProviderObservationGraph
    from archivetrust.domain.telemetry.events import (
        EvidenceCreated,
        EvidenceRejected,
        ObservationCreated,
        ProviderObservationAttempted,
    )
    from archivetrust.infrastructure.storage.telemetry_sink import FileTelemetrySink
    from archivetrust.review.assembler import assemble_packet
    from archivetrust.review.triage import TriagePolicy, triage_review_queue
    from scripts.semantic_alignment_audit import classify_packet

    if not TELEMETRY_PATH.exists():
        raise SystemExit(f"expected real corpus telemetry at {TELEMETRY_PATH}, not found")

    hash_before = _sha256(TELEMETRY_PATH)
    _log(f"events.jsonl sha256 (before): {hash_before}")

    _log("Loading domain telemetry (events.jsonl, ~977MB) -- this takes about a minute...")
    sink = FileTelemetrySink(TELEMETRY_PATH)
    all_events = list(sink.all_events())
    _log(f"Loaded {len(all_events)} domain events.")

    doc_refs = sorted({getattr(e, "document_ref", None) for e in all_events} - {None})
    _log(f"{len(doc_refs)} distinct documents in this corpus.")
    if len(sys.argv) > 1:
        limit = int(sys.argv[1])
        doc_refs = doc_refs[:limit]
        _log(f"--limit {limit}: restricting this run to the first {len(doc_refs)} documents.")

    _PRE = (ProviderObservationAttempted, EvidenceRejected, EvidenceCreated, ObservationCreated)

    journal = Journal()
    triage_policy = TriagePolicy()
    reconciliation_policy = ReconciliationPolicy(policy_version=1)
    capability_matrix = production_capability_matrix()
    confidence_policy = ConfidencePolicy(confidence_policy_version=1)

    replay_failures: list[dict] = []
    documents_replayed = 0

    original_review_reason_counts: Counter = Counter()
    regenerated_review_reason_counts: Counter = Counter()
    original_classification_counts: Counter = Counter()
    regenerated_classification_counts: Counter = Counter()
    original_queue_classification_counts: Counter = Counter()
    regenerated_queue_classification_counts: Counter = Counter()
    original_confidence_buckets: Counter = Counter()
    regenerated_confidence_buckets: Counter = Counter()

    original_contested_total = 0
    original_single_source_total = 0
    regenerated_contested_total = 0
    regenerated_single_source_total = 0

    no_provider_rerun_violations: list[str] = []
    category_records: list[dict] = []
    newly_contested: list[dict] = []

    events_by_document: dict[str, list] = defaultdict(list)
    for event in all_events:
        ref = getattr(event, "document_ref", None)
        if ref is not None:
            events_by_document[ref].append(event)
    del all_events  # this corpus is large; release the flat list once grouped

    for i, doc_ref in enumerate(doc_refs):
        if i % 50 == 0:
            _log(f"  processing document {i + 1}/{len(doc_refs)}...")
        events = events_by_document.pop(doc_ref)

        pre_events = tuple(e for e in events if isinstance(e, _PRE))
        if not pre_events:
            continue

        # -- original: replay exactly what was recorded, no re-derivation --
        original_state = journal.replay(events)
        original_packets = tuple(
            assemble_packet(original_state, item, document_ref=doc_ref, archive_object_ref=doc_ref)
            for item in triage_review_queue(original_state, triage_policy)
        )

        # -- regenerated: current Comparison -> Confidence -> Canonical -> Review, from persisted
        # Evidence/Observations only. No provider adapter is imported or invoked anywhere in this
        # script or in run_comparison_and_assembly/regenerate_review_queue's own call graph.
        pre_state = journal.replay(pre_events)
        obs_by_provider: dict[tuple[str, str], list] = defaultdict(list)
        for observation in pre_state.all_observations():
            obs_by_provider[(observation.provider_id, observation.provider_version)].append(observation)
        if not obs_by_provider:
            continue
        provider_graphs = tuple(
            ProviderObservationGraph(
                provider_id=p, provider_version=v,
                invocation_id=f"regenerated:{p}:{v}", observations=tuple(obs),
            )
            for (p, v), obs in sorted(obs_by_provider.items())
        )
        evidence_by_id = {e.evidence_id: e for e in pre_state.all_evidence()}

        try:
            result = run_comparison_and_assembly(
                document_ref=doc_ref,
                archive_object_ref=doc_ref,
                provider_graphs=provider_graphs,
                evidence_by_id=evidence_by_id,
                reconciliation_policy=reconciliation_policy,
                capability_matrix=capability_matrix,
                confidence_policy=confidence_policy,
            )
        except Exception as exc:  # noqa: BLE001 -- recorded as data, corpus-wide run must continue
            replay_failures.append({"document_ref": doc_ref, "error": f"{type(exc).__name__}: {exc}"})
            continue

        for event in result.events:
            if isinstance(event, _PRE):
                no_provider_rerun_violations.append(doc_ref)

        regenerated_state = journal.replay(pre_events + result.events)
        regenerated_packets = tuple(
            assemble_packet(regenerated_state, item, document_ref=doc_ref, archive_object_ref=doc_ref)
            for item in triage_review_queue(regenerated_state, triage_policy)
        )
        documents_replayed += 1

        # -- corpus-wide classification/confidence distributions (every known slot, not just queued) --
        for slot in original_state.known_semantic_slots():
            canonical = original_state.canonical_observation_history(slot)[-1]
            original_classification_counts[canonical.comparison_confidence.classification.value] += 1
            if canonical.canonical_confidence is not None:
                original_confidence_buckets[_confidence_bucket(canonical.canonical_confidence.value)] += 1
        for slot in regenerated_state.known_semantic_slots():
            canonical = regenerated_state.canonical_observation_history(slot)[-1]
            regenerated_classification_counts[canonical.comparison_confidence.classification.value] += 1
            if canonical.canonical_confidence is not None:
                regenerated_confidence_buckets[_confidence_bucket(canonical.canonical_confidence.value)] += 1

        for packet in original_packets:
            original_review_reason_counts[packet.review_reason.value] += 1
            classification = packet.comparison_confidence.classification.value
            original_queue_classification_counts[classification] += 1
            if classification == "contested":
                original_contested_total += 1
            elif classification == "uncorroborated_single_source":
                original_single_source_total += 1

        # original observation-set -> classification, for "newly contested" detection
        original_sets: list[tuple[frozenset, str]] = []
        for slot in original_state.known_semantic_slots():
            canonical = original_state.canonical_observation_history(slot)[-1]
            obs_ids = frozenset(ref.observation_id for ref in canonical.contributing_observations)
            original_sets.append((obs_ids, canonical.comparison_confidence.classification.value))

        for packet in regenerated_packets:
            regenerated_review_reason_counts[packet.review_reason.value] += 1
            classification = packet.comparison_confidence.classification.value
            regenerated_queue_classification_counts[classification] += 1
            if classification == "contested":
                regenerated_contested_total += 1
            elif classification == "uncorroborated_single_source":
                regenerated_single_source_total += 1

            if classification != "contested":
                continue
            regenerated_obs_ids = frozenset(c.observation_id for c in packet.candidates)
            exact_match_was_contested = any(
                obs_ids == regenerated_obs_ids and cls == "contested" for obs_ids, cls in original_sets
            )
            if not exact_match_was_contested:
                overlapping = [
                    (sorted(obs_ids), cls) for obs_ids, cls in original_sets
                    if obs_ids & regenerated_obs_ids
                ]
                newly_contested.append({
                    "document_ref": doc_ref,
                    "semantic_slot_id": packet.semantic_slot_id,
                    "regenerated_observation_ids": sorted(regenerated_obs_ids),
                    "original_overlapping_groupings": overlapping[:3],
                    "explanation": (
                        "no original canonical decision classified exactly this observation set as "
                        "contested -- either these observations were grouped differently by the "
                        "original (pre-fix) clustering, or at least one was previously "
                        "single-source/corroborated on its own"
                        if overlapping else
                        "no original canonical decision shares any observation with this regenerated "
                        "contested packet -- these observations were not part of any recorded "
                        "canonical decision in the original benchmark (e.g. a provider whose "
                        "capability-matrix participation changed, or an alignment change)"
                    ),
                })

            if len({c.provider_id for c in packet.candidates}) >= 2:
                canonical_by_id = {
                    c.canonical_observation_id: c
                    for c in result.reconciled_graph.canonical_observations
                }
                canonical = canonical_by_id.get(packet.canonical_observation_id)
                if canonical is not None:
                    category_records.append(classify_packet(packet, canonical))

    hash_after = _sha256(TELEMETRY_PATH)
    _log(f"events.jsonl sha256 (after):  {hash_after}")
    if hash_after != hash_before:
        raise SystemExit(
            "IMMUTABILITY VIOLATION: events.jsonl changed during replay -- aborting without "
            "writing a report. This must never happen; investigate before re-running."
        )

    category_counts = Counter(r["category"] for r in category_records)

    def _reduction_pct(before: int, after: int) -> float | None:
        if before == 0:
            return None
        return round((before - after) / before * 100, 1)

    results = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "corpus": {
            "workspace_id": "46e94f15d73347c2be79f1527e646c11",
            "documents_in_corpus": len(doc_refs),
            "documents_replayed": documents_replayed,
        },
        "replay_failures": replay_failures,
        "immutability": {
            "events_jsonl_sha256_before": hash_before,
            "events_jsonl_sha256_after": hash_after,
            "unchanged": hash_after == hash_before,
        },
        "no_provider_rerun_violations": no_provider_rerun_violations,
        "review_queue_totals": {
            "original": {
                "contested": original_contested_total,
                "single_source": original_single_source_total,
                "total": sum(original_review_reason_counts.values()),
            },
            "regenerated": {
                "contested": regenerated_contested_total,
                "single_source": regenerated_single_source_total,
                "total": sum(regenerated_review_reason_counts.values()),
            },
            "reduction_pct": {
                "contested": _reduction_pct(original_contested_total, regenerated_contested_total),
                "single_source": _reduction_pct(original_single_source_total, regenerated_single_source_total),
                "total": _reduction_pct(
                    sum(original_review_reason_counts.values()), sum(regenerated_review_reason_counts.values())
                ),
            },
        },
        "review_reason_distribution": {
            "original": dict(original_review_reason_counts),
            "regenerated": dict(regenerated_review_reason_counts),
        },
        "comparison_classification_distribution_all_slots": {
            "original": dict(original_classification_counts),
            "regenerated": dict(regenerated_classification_counts),
        },
        "comparison_classification_distribution_queued_only": {
            "original": dict(original_queue_classification_counts),
            "regenerated": dict(regenerated_queue_classification_counts),
        },
        "canonical_confidence_distribution": {
            "original": dict(original_confidence_buckets),
            "regenerated": dict(regenerated_confidence_buckets),
        },
        "top_categories_remaining_contested_multi_provider": category_counts.most_common(),
        "newly_contested_count": len(newly_contested),
        "newly_contested_examples": newly_contested[:20],
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2))
    _log(f"Wrote {OUTPUT_PATH}")
    print(json.dumps(results, indent=2)[:4000])


if __name__ == "__main__":
    main()
