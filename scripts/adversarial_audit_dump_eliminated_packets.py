"""Adversarial Architecture Audit (2026-07-14) -- data collection.

The RCDP report (RCDP_REMAINING_CLUSTERING_DEFECT_2026-07-14.md) found a counterfactual that
drives contested packets from 332 to 0, by preventing Docling/PaddleOCR-VL from ever clustering
when PaddleOCR-VL has no geometry. This script does NOT apply that counterfactual. It captures the
full, pre-counterfactual state (post-typing-fix, 332 contested packets) with actual candidate text
and full page context, so the elimination can be audited for information loss rather than trusted
on packet-count alone.

For every one of the 332 packets, records:
  - both candidates' full text
  - every OTHER Docling Observation on the same page (to check whether PaddleOCR-VL's content is
    captured redundantly elsewhere, or is the only place it appears)
  - containment/semantic_relation exactly as classify_packet computed them originally

Read-only. No src/ file modified. No counterfactual applied here -- this is the "before" snapshot.
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

WORKSPACE_DIR = REPO_ROOT / "archivetrust_data" / "workspaces" / "46e94f15d73347c2be79f1527e646c11"
OUT = REPO_ROOT / "benchmarks" / "adversarial_audit_eliminated_packets.jsonl"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def native_label_of(observation, evidence_by_id):
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
    from archivetrust.domain.comparison.geometry import page_of

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from semantic_alignment_audit import classify_packet  # reused unmodified

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

    records = []
    failed_docs = 0

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

        # Same typing fix as the established baseline (this dump is the "before missing-geometry
        # guard" / "after typing fix" state -- exactly the 332-packet population under audit).
        new_obs_by_provider = {}
        for key, obs_list in obs_by_provider.items():
            p, v = key
            if p != "tesseract_layoutparser":
                new_obs_by_provider[key] = obs_list
                continue
            retyped = []
            for o in obs_list:
                if o.observation_type == ObservationType.PARAGRAPH and native_label_of(o, evidence_by_id) == "Text":
                    retyped.append(
                        o.model_copy(
                            update={"observation_type": ObservationType.LAYOUT_REGION, "payload": LayoutRegionPayload(native_label="Text")}
                        )
                    )
                else:
                    retyped.append(o)
            new_obs_by_provider[key] = retyped

        graphs = tuple(
            ProviderObservationGraph(provider_id=p, provider_version=v, invocation_id=f"regenerated:{p}:{v}", observations=tuple(obs))
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

        # Full per-page Docling observation text index (any type), for the "captured elsewhere?"
        # check -- built once per document from the (retyped) observation pool.
        docling_text_by_page = defaultdict(list)
        for (p, v), obs_list in new_obs_by_provider.items():
            if p != "docling":
                continue
            for o in obs_list:
                text = getattr(o.payload, "text", None)
                if not text:
                    continue
                page = page_of(o, evidence_by_id)
                docling_text_by_page[page].append({"observation_id": o.observation_id, "observation_type": o.observation_type.value, "text": text})

        for item in triage_review_queue(regenerated_state, triage_policy):
            if item.reason != ReviewReason.SOURCES_DISAGREE:
                continue
            packet = assemble_packet(regenerated_state, item, document_ref=doc_ref, archive_object_ref=doc_ref)
            providers = {c.provider_id for c in packet.candidates}
            if len(providers) < 2:
                continue
            canonical = canonical_by_id.get(packet.canonical_observation_id)
            if canonical is None:
                continue
            rec = classify_packet(packet, canonical)
            if providers != {"docling", "paddleocr-vl"}:
                continue  # this audit is scoped to the exact population the counterfactual eliminated

            candidates_detail = [
                {"provider_id": c.provider_id, "value": c.value, "page": (c.evidence[0].page if c.evidence else None)}
                for c in packet.candidates
            ]
            page = candidates_detail[0]["page"] if candidates_detail else None
            other_docling_on_page = [
                d for d in docling_text_by_page.get(page, []) if d["observation_id"] not in {c.observation_id for c in packet.candidates}
            ]

            records.append(
                {
                    "document_ref": doc_ref,
                    "canonical_observation_id": canonical.canonical_observation_id,
                    "category": rec["category"],
                    "page": page,
                    "candidates": candidates_detail,
                    "containment_ratio": rec["containment_ratio"],
                    "semantic_relation": rec["semantic_relation"],
                    "candidate_lengths": rec["candidate_lengths"],
                    "other_docling_observations_same_page": other_docling_on_page,
                }
            )

        if (idx + 1) % 50 == 0:
            log(f"  ...{idx + 1}/{len(doc_refs)} docs processed, {len(records)} eliminated-population packets so far")

    log(f"\nFailed docs: {failed_docs}")
    log(f"Total docling/paddleocr-vl contested packets captured: {len(records)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"Wrote {len(records)} records to {OUT}")


if __name__ == "__main__":
    main()
