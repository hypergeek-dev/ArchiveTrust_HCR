"""RCDP Step 7 -- counterfactual replay for the remaining_clustering_defect mechanism.

Mechanism under test (established by scripts/rcdp_remaining_clustering_defect.py against the full
corpus, 126/126 = 100%): every remaining_clustering_defect packet (post-typing-fix state) is a
docling/paddleocr-vl pair, unambiguously ordinal-merged (each provider contributed exactly one
paragraph-type Observation to the page) despite one side having NO bounding box at all -- i.e. no
known spatial extent, as opposed to merely coarse geometry. `_pairwise_affinity`'s ordinal fallback
(domain/comparison/clustering.py) does not distinguish "coarse geometry" from "no geometry
whatsoever" -- both fall through to the same ordinal-position guess.

Smallest possible architectural change that removes only this mechanism: use the EXISTING
AlignmentService seam (domain/alignment/service.py, ROADMAP.md S5.12 -- built exactly for swapping
grouping strategies without touching the Comparison Engine) to substitute one guard clause in the
pairwise affinity function: when either side has NO bounding box at all (not merely non-pixel-
-accurate), do not fall through to an ordinal-position guess -- return the same "undefined" signal
already used for the ambiguous multi-per-page case. This is a read-only experiment: no file under
src/ is modified. The real `cluster_observations`/`_pairwise_affinity` are left untouched; this
script provides an alternate `AlignmentService` implementation for A/B replay only.
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
OUT = REPO_ROOT / "benchmarks" / "rcdp_ordinal_geometry_counterfactual.json"


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


def build_patched_alignment_service():
    """Constructs a AlignmentService whose only behavioral difference from the production
    ClusteringAlignmentService is the one guard clause described in the module docstring. Copies
    domain.comparison.clustering's connected-components/pool-formation logic verbatim (imported,
    not reimplemented) and only replaces `_pairwise_affinity`.
    """
    from archivetrust.domain.alignment.models import (
        ALIGNMENT_ALGORITHM_VERSION,
        AlignmentAttempt,
        AlignmentObservationState,
        AlignmentOutcome,
        AlignmentResult,
    )
    from archivetrust.domain.comparison.clustering import AffinityEdge, Cluster, _connected_components
    from archivetrust.domain.comparison.geometry import (
        bounding_box_of,
        both_pixel_accurate,
        intersection_over_union,
        page_of,
    )
    from archivetrust.domain.ontology.types import ObservationType
    from archivetrust.domain.shared.ids import new_id

    def _ordinal_affinity(index_a: int, index_b: int) -> float:
        distance = abs(index_a - index_b)
        if distance == 0:
            return 1.0
        return max(0.0, 1.0 - 0.5 * distance)

    def _pairwise_affinity_patched(obs_a, obs_b, index_a, index_b, evidence_by_id, provider_counts):
        box_a = bounding_box_of(obs_a, evidence_by_id)
        box_b = bounding_box_of(obs_b, evidence_by_id)
        if both_pixel_accurate(box_a, box_b):
            return intersection_over_union(box_a, box_b), "pixel-accurate IoU"

        # -- The one changed line vs. production clustering.py: a missing bounding box (no known
        # spatial extent at all) is not the same fact as "coarse geometry", and must not be treated
        # as interchangeable with it by the ordinal-position fallback. --
        if box_a is None or box_b is None:
            return 0.0, "ordinal position undefined (one side has no geometry at all -- scope cannot be confirmed)"

        key_a = (obs_a.provider_id, obs_a.provider_version)
        key_b = (obs_b.provider_id, obs_b.provider_version)
        if provider_counts.get(key_a, 0) > 1 or provider_counts.get(key_b, 0) > 1:
            return 0.0, "ordinal position undefined (ambiguous: a provider contributed more than one same-type Observation to this page)"
        return _ordinal_affinity(index_a, index_b), "ordinal position (coarse geometry, both sides have some geometry)"

    def cluster_observations_patched(observations, evidence_by_id, policy):
        pools: dict = {}
        for obs in sorted(observations, key=lambda o: o.observation_id):
            page = page_of(obs, evidence_by_id)
            pools.setdefault((obs.observation_type, page), []).append(obs)

        clusters = []
        for (observation_type, page), pool in sorted(pools.items(), key=lambda kv: (kv[0][0].value, kv[0][1] or -1)):
            member_ids = tuple(o.observation_id for o in pool)
            provider_counts: dict = {}
            for obs in pool:
                key = (obs.provider_id, obs.provider_version)
                provider_counts[key] = provider_counts.get(key, 0) + 1
            edges = []
            affinity_records = []
            for i in range(len(pool)):
                for j in range(i + 1, len(pool)):
                    score, basis = _pairwise_affinity_patched(pool[i], pool[j], i, j, evidence_by_id, provider_counts)
                    score = round(score, policy.rounding_ndigits)
                    edges.append((pool[i].observation_id, pool[j].observation_id, score))
                    affinity_records.append(
                        AffinityEdge(observation_id_a=pool[i].observation_id, observation_id_b=pool[j].observation_id, score=score, basis=basis)
                    )
            components = _connected_components(member_ids, edges, policy.cluster_join_threshold)
            for component in components:
                component_edges = tuple(e for e in affinity_records if e.observation_id_a in component and e.observation_id_b in component)
                basis = (
                    f"type={observation_type.value}, page={page}, cluster_join_threshold={policy.cluster_join_threshold} "
                    f"(Policy v{policy.policy_version}), {len(component)} member(s) [PATCHED alignment: missing-geometry guard]"
                )
                clusters.append(
                    Cluster(
                        cluster_id=new_id("cluster"),
                        observation_type=observation_type,
                        page=page,
                        member_observation_ids=tuple(component),
                        clustering_basis=basis,
                        affinity_edges=component_edges,
                    )
                )
        return tuple(clusters)

    class PatchedAlignmentService:
        algorithm_name = "connected-components-geometry-clustering-missing-geometry-guard-experiment"
        algorithm_version = ALIGNMENT_ALGORITHM_VERSION

        def align(self, observations, evidence_by_id, policy):
            clusters = cluster_observations_patched(observations, evidence_by_id, policy)
            candidates_by_key: dict = {}
            for obs in observations:
                key = (obs.observation_type, page_of(obs, evidence_by_id))
                candidates_by_key.setdefault(key, []).append(obs.observation_id)

            attempts = []
            states = []
            for cluster in clusters:
                key = (cluster.observation_type, cluster.page)
                candidates = tuple(sorted(candidates_by_key.get(key, [])))
                members = cluster.member_observation_ids
                attempt = AlignmentAttempt(
                    alignment_attempt_id=new_id("alignment_attempt"),
                    algorithm_name=self.algorithm_name,
                    algorithm_version=self.algorithm_version,
                    comparison_group_id=cluster.cluster_id,
                    observation_type=cluster.observation_type,
                    page=cluster.page,
                    candidate_observation_ids=candidates,
                    selected_observation_ids=members,
                    alignment_rationale=cluster.clustering_basis,
                    metadata={
                        "member_count": str(len(members)),
                        "candidate_count": str(len(candidates)),
                        "affinity_edge_count": str(len(cluster.affinity_edges)),
                    },
                )
                attempts.append(attempt)
                outcome = AlignmentOutcome.ALIGNED if len(members) > 1 else AlignmentOutcome.UNALIGNED
                for observation_id in members:
                    states.append(
                        AlignmentObservationState(
                            observation_id=observation_id,
                            outcome=outcome,
                            comparison_group_id=cluster.cluster_id,
                            alignment_attempt_id=attempt.alignment_attempt_id,
                        )
                    )
            return AlignmentResult(
                algorithm_name=self.algorithm_name,
                algorithm_version=self.algorithm_version,
                clusters=clusters,
                attempts=tuple(attempts),
                observation_states=tuple(states),
            )

    return PatchedAlignmentService()


def main():
    from archivetrust.infrastructure.storage.telemetry_sink import FileTelemetrySink
    from archivetrust.application.journal import Journal
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
    from archivetrust.domain.confidence.models import ComparisonClassification
    from archivetrust.domain.ontology.types import ObservationType
    from archivetrust.domain.ontology.payloads import LayoutRegionPayload
    from archivetrust.review.triage import review_reason_for, TriagePolicy, ReviewReason

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from semantic_alignment_audit import classify_packet  # noqa: E402 -- reused unmodified

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
    patched_service = build_patched_alignment_service()

    from archivetrust.review.assembler import assemble_packet
    from archivetrust.review.triage import triage_review_queue

    category_counts = Counter()
    total_contested = 0
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

        # -- Base state for this experiment is the POST-typing-fix state established in the
        # prevalence audit (retype tesseract native_label=='Text' -> LAYOUT_REGION), so this
        # experiment is additive on top of the already-validated typing fix, not a fresh baseline. --
        new_obs_by_provider = {}
        for key, obs_list in obs_by_provider.items():
            p, v = key
            if p != "tesseract_layoutparser":
                new_obs_by_provider[key] = obs_list
                continue
            retyped = []
            for o in obs_list:
                if o.observation_type == ObservationType.PARAGRAPH and native_label_of(o, evidence_by_id) == "Text":
                    retyped.append(o.model_copy(update={"observation_type": ObservationType.LAYOUT_REGION, "payload": LayoutRegionPayload(native_label="Text")}))
                else:
                    retyped.append(o)
            new_obs_by_provider[key] = retyped

        graphs = tuple(
            ProviderObservationGraph(provider_id=p, provider_version=v, invocation_id=f"regenerated:{p}:{v}", observations=tuple(obs))
            for (p, v), obs in sorted(new_obs_by_provider.items())
        )
        observations_by_id = {o.observation_id: o for g in graphs for o in g.observations}

        try:
            comparison_result = run_comparison_engine(
                graphs, evidence_by_id, cap_matrix, recon_policy, alignment_service=patched_service
            )
            completed_graph = apply_confidence_engine(comparison_result.reconciled_graph, observations_by_id, conf_policy)
        except Exception as exc:
            failed_docs += 1
            log(f"  FAILED {doc_ref}: {exc}")
            continue

        # Build a minimal JournalState-free packet/triage pass, reusing the real assemble_packet by
        # replaying only the events this patched comparison run would have produced -- construct
        # them the same way run_comparison_and_assembly does.
        from archivetrust.domain.comparison.telemetry import comparison_result_to_events
        from archivetrust.domain.confidence.telemetry import confidence_changed_events
        from archivetrust.domain.alignment.telemetry import alignment_events

        events_out = []
        events_out.extend(alignment_events(comparison_result.alignment, document_ref=doc_ref))
        events_out.extend(
            comparison_result_to_events(
                type(comparison_result)(reconciled_graph=completed_graph, bundles=comparison_result.bundles, alignment=comparison_result.alignment),
                document_ref=doc_ref,
                policy=recon_policy,
                capability_matrix=cap_matrix,
            )
        )
        events_out.extend(confidence_changed_events(completed_graph.canonical_observations, document_ref=doc_ref, policy=conf_policy))

        regenerated_state = journal.replay(pre + tuple(events_out))
        canonical_by_id = {c.canonical_observation_id: c for c in completed_graph.canonical_observations}

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
            total_contested += 1
            category_counts[rec["category"]] += 1

        if (idx + 1) % 50 == 0:
            log(f"  ...{idx + 1}/{len(doc_refs)} docs processed, {total_contested} contested so far")

    log(f"\nFailed docs: {failed_docs}")
    log(f"Total contested packets AFTER missing-geometry guard (on top of typing fix): {total_contested}")
    log("Category distribution:")
    for cat, n in category_counts.most_common():
        log(f"  {cat}: {n}")

    summary = {
        "total_contested_packets": total_contested,
        "category_counts": dict(category_counts),
        "failed_docs": failed_docs,
        "total_docs": len(doc_refs),
    }
    OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
