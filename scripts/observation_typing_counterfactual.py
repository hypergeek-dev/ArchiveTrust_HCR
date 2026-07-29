"""Phase 5 counterfactual replay (SEMANTIC_CONTRACT_AUDIT follow-up, 2026-07-14).

Question: if tesseract_layoutparser's native_label=="Text" observations had been typed as the
architecture's own honest fallback (LayoutRegionPayload, ObservationType.LAYOUT_REGION) instead of
unconditionally as ObservationType.PARAGRAPH, how many contested review packets remain?

No new heuristic is introduced -- LayoutRegionPayload already exists precisely for "no universal
ontology" native labels (providers/tesseract_layoutparser/importer.py's own docstring, S2.2) and is
already applied to every OTHER unmapped Tesseract label. This script only removes the special-case
"Text"->paragraph entry and re-derives Comparison/Confidence/Triage from the same persisted
Evidence, using the exact same pipeline call and classify_packet() as the baseline
(scripts/semantic_alignment_audit.py) so the two runs are apples-to-apples comparable.

Mechanism: does NOT re-run the provider or re-decode raw output. Takes each already-persisted
Observation (from ObservationCreated telemetry) whose provider is tesseract_layoutparser and whose
Evidence.raw_output records native_label=="Text", and replaces only its observation_type/payload
(model_copy) with the LayoutRegionPayload equivalent -- same evidence_ids, same provider_confidence,
same everything else. This is the minimal counterfactual: "what if this Observation had been typed
honestly from the start."
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

WORKSPACE_DIR = REPO_ROOT / "archivetrust_data" / "workspaces" / "46e94f15d73347c2be79f1527e646c11"
OUT = REPO_ROOT / "benchmarks" / "observation_typing_counterfactual.json"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def native_label_of(observation, evidence_by_id) -> str | None:
    if not observation.evidence_ids:
        return None
    ev = evidence_by_id.get(observation.evidence_ids[0])
    if ev is None:
        return None
    try:
        raw = json.loads(ev.raw_output)
    except (json.JSONDecodeError, TypeError):
        return None
    return raw.get("native_label")


def main():
    from archivetrust.infrastructure.storage.telemetry_sink import FileTelemetrySink
    from archivetrust.application.journal import Journal
    from archivetrust.application.pipeline import run_comparison_and_assembly
    from archivetrust.domain.graph.provider_graph import ProviderObservationGraph
    from archivetrust.domain.telemetry.events import (
        EvidenceCreated,
        EvidenceRejected,
        ObservationCreated,
        ProviderObservationAttempted,
    )
    from archivetrust.domain.comparison.policy import ReconciliationPolicy
    from archivetrust.domain.comparison.capability_matrix_data import production_capability_matrix
    from archivetrust.domain.confidence.policy import ConfidencePolicy
    from archivetrust.domain.ontology.types import ObservationType
    from archivetrust.domain.ontology.payloads import LayoutRegionPayload
    from archivetrust.review.triage import TriagePolicy, triage_review_queue, ReviewReason
    from archivetrust.review.assembler import assemble_packet

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from semantic_alignment_audit import classify_packet  # reuse unmodified, same as baseline

    _PRE = (ProviderObservationAttempted, EvidenceRejected, EvidenceCreated, ObservationCreated)

    log("Loading telemetry (932MB)...")
    sink = FileTelemetrySink(WORKSPACE_DIR / "telemetry" / "events.jsonl")
    all_events = sink._events
    log(f"Loaded {len(all_events)} events")
    doc_refs = sorted({getattr(e, "document_ref", None) for e in all_events} - {None})

    journal = Journal()
    triage_policy = TriagePolicy()
    recon_policy = ReconciliationPolicy(policy_version=1)
    cap_matrix = production_capability_matrix()
    conf_policy = ConfidencePolicy(confidence_policy_version=1)

    retyped_total = 0
    contested_packets = []  # baseline-shape records, same fields as calibration_corpus_1.jsonl
    category_counts = Counter()
    failed_docs = 0

    # Track per-document canonical-slot classification shift: for the SAME semantic content, did
    # retyping change a slot's fate (contested->not, or vice versa is impossible here since we only
    # ever remove paragraph-pool membership, never add it)? We approximate this by comparing counts
    # of multi-provider contested canonical observations whose contributing observations include at
    # least one retyped one, against how many total retyped observations existed.
    canonicals_touching_retyped_before_after = {"contested_before_retype_removed": 0}

    for idx, doc_ref in enumerate(doc_refs):
        events = list(sink.events_for_document(doc_ref))
        if not events:
            continue
        pre = tuple(e for e in events if isinstance(e, _PRE))
        source_state = journal.replay(pre)

        obs_by_provider = defaultdict(list)
        for o in source_state.all_observations():
            obs_by_provider[(o.provider_id, o.provider_version)].append(o)
        if not obs_by_provider:
            continue

        evidence_by_id = {e.evidence_id: e for e in source_state.all_evidence()}

        # -- Apply the counterfactual retyping --
        doc_retyped = 0
        new_obs_by_provider = {}
        for key, obs_list in obs_by_provider.items():
            p, v = key
            if p != "tesseract_layoutparser":
                new_obs_by_provider[key] = obs_list
                continue
            retyped_list = []
            for o in obs_list:
                if o.observation_type == ObservationType.PARAGRAPH and native_label_of(o, evidence_by_id) == "Text":
                    retyped_list.append(
                        o.model_copy(
                            update={
                                "observation_type": ObservationType.LAYOUT_REGION,
                                "payload": LayoutRegionPayload(native_label="Text"),
                            }
                        )
                    )
                    doc_retyped += 1
                else:
                    retyped_list.append(o)
            new_obs_by_provider[key] = retyped_list
        retyped_total += doc_retyped
        if doc_retyped == 0:
            continue  # no counterfactual effect possible for this document; skip re-running pipeline

        graphs = tuple(
            ProviderObservationGraph(
                provider_id=p, provider_version=v, invocation_id=f"regenerated:{p}:{v}", observations=tuple(obs)
            )
            for (p, v), obs in sorted(new_obs_by_provider.items())
        )
        try:
            result = run_comparison_and_assembly(
                document_ref=doc_ref,
                archive_object_ref=doc_ref,
                provider_graphs=graphs,
                evidence_by_id=evidence_by_id,
                reconciliation_policy=recon_policy,
                capability_matrix=cap_matrix,
                confidence_policy=conf_policy,
            )
        except Exception as exc:
            failed_docs += 1
            log(f"  FAILED {doc_ref}: {exc}")
            continue

        canonical_by_id = {c.canonical_observation_id: c for c in result.reconciled_graph.canonical_observations}
        regenerated_state = journal.replay(pre + result.events)

        for item in triage_review_queue(regenerated_state, triage_policy):
            if item.reason != ReviewReason.SOURCES_DISAGREE:
                continue
            packet = assemble_packet(regenerated_state, item, document_ref=doc_ref, archive_object_ref=doc_ref)
            if len({c.provider_id for c in packet.candidates}) < 2:
                continue
            canonical = canonical_by_id.get(packet.canonical_observation_id)
            if canonical is None:
                continue
            rec = classify_packet(packet, canonical)
            rec["document_ref"] = doc_ref
            contested_packets.append(rec)
            category_counts[rec["category"]] += 1

        if (idx + 1) % 50 == 0:
            log(f"  ...{idx + 1}/{len(doc_refs)} docs processed, {retyped_total} retyped so far")

    log(f"\nFailed docs: {failed_docs}")
    log(f"Total tesseract_layoutparser observations retyped (native_label=='Text' -> LAYOUT_REGION): {retyped_total}")
    log(f"Total contested multi-provider packets AFTER retyping: {len(contested_packets)}")
    log("Category distribution AFTER retyping:")
    for cat, n in category_counts.most_common():
        log(f"  {cat}: {n}")

    summary = {
        "retyped_observation_count": retyped_total,
        "contested_packets_after_retyping": len(contested_packets),
        "category_counts_after_retyping": dict(category_counts),
        "failed_docs": failed_docs,
        "total_docs": len(doc_refs),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
