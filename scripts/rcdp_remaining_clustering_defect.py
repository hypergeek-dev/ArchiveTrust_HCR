"""Root Cause Discovery Protocol -- remaining_clustering_defect (2026-07-14).

Investigates exactly one category, in the *post-typing-fix* state (the counterfactual established
in OBSERVATION_TYPING_PREVALENCE_AUDIT_2026-07-14.md: tesseract_layoutparser's native_label=="Text"
Observations retyped to LayoutRegionPayload/ObservationType.LAYOUT_REGION). remaining_clustering_defect
was the largest category in that post-fix state (126 packets) and is investigated here because it is
classify_packet's residual bucket: distinct textual content, no 2-candidate pixel-accurate IoU
available to explain why the two candidates were ever compared at all.

For every packet in this category, walks back into the Comparison Engine's own Cluster object (not
just its post-hoc string summary) to recover the *exact* deterministic mechanism that put the two
disagreeing Observations in the same cluster: which pairwise affinity basis fired
(domain/comparison/clustering.py::_pairwise_affinity), what score, and how many members the
resulting connected component had. No provider is re-invoked; no new heuristic is introduced; this
is read-only over already-computed, deterministic Comparison Engine output.
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
OUT = REPO_ROOT / "benchmarks" / "rcdp_remaining_clustering_defect.jsonl"
SUMMARY_OUT = REPO_ROOT / "benchmarks" / "rcdp_remaining_clustering_defect_summary.json"


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
    from archivetrust.domain.comparison.engine import run_comparison_engine
    from archivetrust.domain.confidence.engine import apply_confidence_engine
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
    from archivetrust.domain.evidence.models import Precision
    from archivetrust.domain.comparison.geometry import bounding_box_of
    from archivetrust.review.triage import TriagePolicy, triage_review_queue, ReviewReason
    from archivetrust.review.assembler import assemble_packet

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from semantic_alignment_audit import classify_packet  # reuse unmodified, same as prior audits

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

        # -- Apply the SAME retyping counterfactual as the prevalence audit --
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
                else:
                    retyped_list.append(o)
            new_obs_by_provider[key] = retyped_list

        graphs = tuple(
            ProviderObservationGraph(
                provider_id=p, provider_version=v, invocation_id=f"regenerated:{p}:{v}", observations=tuple(obs)
            )
            for (p, v), obs in sorted(new_obs_by_provider.items())
        )
        observations_by_id = {o.observation_id: o for g in graphs for o in g.observations}

        try:
            # Call the Comparison Engine directly (not run_comparison_and_assembly) so the
            # AlignmentResult -- and therefore each Cluster's affinity_edges -- is retained; the
            # pipeline wrapper discards it after building telemetry events.
            comparison_result = run_comparison_engine(graphs, evidence_by_id, cap_matrix, recon_policy)
            completed_graph = apply_confidence_engine(
                comparison_result.reconciled_graph, observations_by_id, conf_policy
            )
        except Exception as exc:
            failed_docs += 1
            log(f"  FAILED (direct engine) {doc_ref}: {exc}")
            continue

        clusters_by_member: dict[str, object] = {}
        for cluster in comparison_result.alignment.clusters:
            for oid in cluster.member_observation_ids:
                clusters_by_member[oid] = cluster

        # -- Re-run the same document through the standard wrapper for triage/assembly, exactly as
        # the prevalence audit did, so packet classification is identical and comparable. --
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
            log(f"  FAILED (wrapper) {doc_ref}: {exc}")
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
            if rec["category"] != "remaining_clustering_defect":
                continue

            contributing_ids = [ref.observation_id for ref in canonical.contributing_observations]
            cluster = clusters_by_member.get(contributing_ids[0]) if contributing_ids else None

            member_precisions = []
            member_providers = set()
            if cluster is not None:
                for oid in cluster.member_observation_ids:
                    obs = observations_by_id.get(oid)
                    if obs is None:
                        continue
                    member_providers.add(obs.provider_id)
                    box = bounding_box_of(obs, evidence_by_id)
                    member_precisions.append(box.precision.value if box is not None else "missing")

            # Find the direct edge (if any) between the two lowest-and-highest-id contributing
            # observations (classify_packet's own candidate set) -- report ALL edges touching this
            # canonical's contributing observations, since >2-member clusters can connect them only
            # transitively (Step 5's "exact code path" requirement).
            edge_bases = []
            direct_edge_between_contributors = False
            if cluster is not None:
                contrib_set = set(contributing_ids)
                for edge in cluster.affinity_edges:
                    if edge.observation_id_a in contrib_set or edge.observation_id_b in contrib_set:
                        edge_bases.append({"basis": edge.basis, "score": edge.score})
                    if edge.observation_id_a in contrib_set and edge.observation_id_b in contrib_set:
                        direct_edge_between_contributors = True

            records.append(
                {
                    "document_ref": doc_ref,
                    "canonical_observation_id": canonical.canonical_observation_id,
                    "observation_type": rec["observation_type"],
                    "providers": rec["providers"],
                    "candidate_lengths": rec["candidate_lengths"],
                    "cluster_id": cluster.cluster_id if cluster is not None else None,
                    "cluster_member_count": len(cluster.member_observation_ids) if cluster is not None else None,
                    "cluster_distinct_providers": len(member_providers) if cluster is not None else None,
                    "member_bbox_precisions": member_precisions,
                    "direct_edge_between_contributors": direct_edge_between_contributors,
                    "edge_bases_touching_contributors": edge_bases,
                }
            )

        if (idx + 1) % 50 == 0:
            log(f"  ...{idx + 1}/{len(doc_refs)} docs processed, {len(records)} remaining_clustering_defect packets so far")

    log(f"\nFailed docs: {failed_docs}")
    log(f"Total remaining_clustering_defect packets found: {len(records)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"Wrote {len(records)} records to {OUT}")

    # -- Aggregate the exact mechanism --
    basis_counter = Counter()
    member_count_counter = Counter()
    direct_edge_counter = Counter()
    precision_pattern_counter = Counter()
    for r in records:
        member_count_counter[r["cluster_member_count"]] += 1
        direct_edge_counter[r["direct_edge_between_contributors"]] += 1
        for eb in r["edge_bases_touching_contributors"]:
            basis_counter[eb["basis"]] += 1
        precisions = tuple(sorted(r["member_bbox_precisions"]))
        precision_pattern_counter[precisions] += 1

    log("\n=== Cluster member count distribution ===")
    for k, v in sorted(member_count_counter.items(), key=lambda kv: -kv[1]):
        log(f"  {k} members: {v} packets")

    log("\n=== Direct edge between the two disagreeing contributors? ===")
    for k, v in direct_edge_counter.most_common():
        log(f"  {k}: {v}")

    log("\n=== Affinity edge basis (touching contributors, may be >1 per packet in >2-member clusters) ===")
    for k, v in basis_counter.most_common():
        log(f"  {k}: {v}")

    log("\n=== Bounding-box precision pattern across cluster members ===")
    for k, v in precision_pattern_counter.most_common(10):
        log(f"  {k}: {v}")

    summary = {
        "total_packets": len(records),
        "cluster_member_count_distribution": {str(k): v for k, v in member_count_counter.items()},
        "direct_edge_between_contributors": {str(k): v for k, v in direct_edge_counter.items()},
        "affinity_edge_basis_touching_contributors": dict(basis_counter),
        "bbox_precision_patterns": {str(k): v for k, v in precision_pattern_counter.items()},
        "failed_docs": failed_docs,
        "total_docs": len(doc_refs),
    }
    SUMMARY_OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"\nWrote {SUMMARY_OUT}")


if __name__ == "__main__":
    main()
