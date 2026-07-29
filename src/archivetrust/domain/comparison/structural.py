"""Phase E -- Structural Reconciliation (MILESTONE4_COMPARISON_ENGINE.md S4, Constitution
Article 9).

Elects `ReconciledObservationGraph` edges from multiple providers' intra-provider parent/child
claims, weighted by **capability tag, never provider identity** (C6) -- the mechanism that
prevents one provider's structure (Docling's, most temptingly) from becoming canonical by default.
Conflicting hierarchies are recorded, never silently resolved (Constitution Article 11); cycles are
broken by a deterministic lowest-weighted-edge drop, logged.

**Documented simplification:** edge weight uses the *parent* cluster's `observation_type` as the
structural-capability lookup key (`capability_matrix.capability_for(..., parent's type)`) as a
proxy for "this provider's structural/hierarchy capability for this kind of relationship." The
spec (S4) does not name a finer-grained "structural capability" axis separate from per-type
capability, so this reuses the same Capability Matrix rather than inventing a new one.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from archivetrust.domain.comparison.capability_matrix import CapabilityMatrix
from archivetrust.domain.comparison.clustering import Cluster
from archivetrust.domain.comparison.policy import ReconciliationPolicy
from archivetrust.domain.confidence.models import ComparisonClassification, ComparisonConfidence
from archivetrust.domain.ontology.base import Observation


class EdgeSupport(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider_id: str
    provider_version: str
    weight: float


class ElectedEdge(BaseModel):
    model_config = ConfigDict(frozen=True)

    parent_cluster_id: str
    child_cluster_id: str
    elected: bool
    weighted_support: float
    total_asserting_weight: float
    supporting_providers: tuple[EdgeSupport, ...]
    comparison_confidence: ComparisonConfidence


class StructuralReconciliationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    edges: tuple[ElectedEdge, ...]
    dropped_for_cycle: tuple[str, ...]
    """`"{parent_cluster_id}->{child_cluster_id}"` keys of edges dropped to preserve acyclicity."""
    reading_order: dict[str, tuple[str, ...]]
    """parent_cluster_id -> ordered tuple of child_cluster_ids (deterministic weighted rank
    aggregation, S4)."""


def _cluster_of(observation_id: str, cluster_by_observation_id: dict[str, str]) -> str | None:
    return cluster_by_observation_id.get(observation_id)


def reconcile_structure(
    clusters: tuple[Cluster, ...],
    observations_by_id: dict[str, Observation],
    capability_matrix: CapabilityMatrix,
    policy: ReconciliationPolicy,
) -> StructuralReconciliationResult:
    cluster_by_observation_id: dict[str, str] = {}
    cluster_by_id: dict[str, Cluster] = {}
    for cluster in clusters:
        cluster_by_id[cluster.cluster_id] = cluster
        for obs_id in cluster.member_observation_ids:
            cluster_by_observation_id[obs_id] = cluster.cluster_id

    # Candidate edges: (parent_cluster, child_cluster) -> {(provider_id, provider_version): weight}
    candidates: dict[tuple[str, str], dict[tuple[str, str], float]] = {}
    all_providers_per_pair: dict[tuple[str, str], set[tuple[str, str]]] = {}

    for observation in observations_by_id.values():
        parent_cluster = _cluster_of(observation.observation_id, cluster_by_observation_id)
        if parent_cluster is None:
            continue
        for child_id in observation.child_observations:
            child_cluster = _cluster_of(child_id, cluster_by_observation_id)
            if child_cluster is None or child_cluster == parent_cluster:
                continue
            key = (parent_cluster, child_cluster)
            provider_key = (observation.provider_id, observation.provider_version)
            capability = capability_matrix.capability_for(
                provider_id=observation.provider_id,
                provider_version=observation.provider_version,
                observation_type=cluster_by_id[parent_cluster].observation_type,
            )
            weight = policy.capability_weight.get(capability, 0.0)
            candidates.setdefault(key, {})[provider_key] = weight
            all_providers_per_pair.setdefault(key, set()).add(provider_key)

    edges: list[ElectedEdge] = []
    for (parent_cluster, child_cluster), provider_weights in sorted(candidates.items()):
        weighted_support = policy.round_score(sum(provider_weights.values()))
        total_possible = policy.round_score(
            sum(
                policy.capability_weight.get(
                    capability_matrix.capability_for(
                        provider_id=p, provider_version=v,
                        observation_type=cluster_by_id[parent_cluster].observation_type,
                    ),
                    0.0,
                )
                for p, v in all_providers_per_pair[(parent_cluster, child_cluster)]
            )
        )
        elected = weighted_support >= policy.edge_accept_threshold
        magnitude = (
            policy.round_score(weighted_support / total_possible) if total_possible > 0 else None
        )
        if len(provider_weights) <= 1:
            classification = ComparisonClassification.UNCORROBORATED_SINGLE_SOURCE
            magnitude = None
        elif elected:
            classification = ComparisonClassification.CORROBORATED
        else:
            classification = ComparisonClassification.CONTESTED

        edges.append(
            ElectedEdge(
                parent_cluster_id=parent_cluster,
                child_cluster_id=child_cluster,
                elected=elected,
                weighted_support=weighted_support,
                total_asserting_weight=total_possible,
                supporting_providers=tuple(
                    EdgeSupport(provider_id=p, provider_version=v, weight=w)
                    for (p, v), w in sorted(provider_weights.items())
                ),
                comparison_confidence=ComparisonConfidence(
                    classification=classification,
                    magnitude=magnitude,
                    basis=(
                        f"capability-weighted support {weighted_support}/{total_possible} "
                        f"(edge_accept_threshold={policy.edge_accept_threshold}, "
                        f"Policy v{policy.policy_version})"
                    ),
                ),
            )
        )

    elected_edges = [e for e in edges if e.elected]
    dropped = _break_cycles(elected_edges)
    dropped_keys = {f"{e.parent_cluster_id}->{e.child_cluster_id}" for e in dropped}
    final_edges = tuple(
        e if f"{e.parent_cluster_id}->{e.child_cluster_id}" not in dropped_keys else e.model_copy(update={"elected": False})
        for e in edges
    )

    reading_order = _aggregate_reading_order(
        final_edges, observations_by_id, cluster_by_observation_id, capability_matrix, policy
    )

    return StructuralReconciliationResult(
        edges=final_edges, dropped_for_cycle=tuple(sorted(dropped_keys)), reading_order=reading_order
    )


def _break_cycles(elected_edges: list[ElectedEdge]) -> list[ElectedEdge]:
    """Deterministic lowest-weighted-edge drop until the parent/child graph is acyclic
    (Constitution Article 9's acyclicity invariant, S4's cycle-handling rule)."""
    remaining = list(elected_edges)
    dropped: list[ElectedEdge] = []

    def has_cycle(edges: list[ElectedEdge]) -> list[ElectedEdge] | None:
        graph: dict[str, list[str]] = {}
        for e in edges:
            graph.setdefault(e.parent_cluster_id, []).append(e.child_cluster_id)
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {}
        nodes = {e.parent_cluster_id for e in edges} | {e.child_cluster_id for e in edges}
        for node in nodes:
            color[node] = WHITE

        def visit(node: str) -> list[ElectedEdge] | None:
            color[node] = GRAY
            for child in graph.get(node, []):
                if color.get(child, WHITE) == GRAY:
                    cycle_edges = [e for e in edges if e.parent_cluster_id == node and e.child_cluster_id == child]
                    return cycle_edges
                if color.get(child, WHITE) == WHITE:
                    result = visit(child)
                    if result:
                        return result
            color[node] = BLACK
            return None

        for node in sorted(nodes):
            if color[node] == WHITE:
                result = visit(node)
                if result:
                    return result
        return None

    while True:
        cycle_edges = has_cycle(remaining)
        if not cycle_edges:
            break
        lowest = sorted(cycle_edges, key=lambda e: (e.weighted_support, e.parent_cluster_id, e.child_cluster_id))[0]
        remaining.remove(lowest)
        dropped.append(lowest)

    return dropped


def _aggregate_reading_order(
    edges: tuple[ElectedEdge, ...],
    observations_by_id: dict[str, Observation],
    cluster_by_observation_id: dict[str, str],
    capability_matrix: CapabilityMatrix,
    policy: ReconciliationPolicy,
) -> dict[str, tuple[str, ...]]:
    """Deterministic weighted Borda-count rank aggregation (S4's positional-median-style
    fallback, used unconditionally here rather than only above the Policy cap -- exact Kemeny
    aggregation is not implemented, since provider counts in this project are always small (<=3)
    and Borda already gives a deterministic, capability-weighted, tie-broken order under that
    bound; see IMPLEMENTATION_STATUS.md).
    """
    children_by_parent: dict[str, set[str]] = {}
    for edge in edges:
        if edge.elected:
            children_by_parent.setdefault(edge.parent_cluster_id, set()).add(edge.child_cluster_id)

    result: dict[str, tuple[str, ...]] = {}
    for parent_cluster, children in children_by_parent.items():
        if len(children) > policy.max_rank_aggregation_children:
            result[parent_cluster] = tuple(sorted(children))
            continue

        scores: dict[str, float] = dict.fromkeys(children, 0.0)
        for observation in observations_by_id.values():
            if cluster_by_observation_id.get(observation.observation_id) != parent_cluster:
                continue
            local_children = [
                cluster_by_observation_id.get(cid)
                for cid in observation.child_observations
                if cluster_by_observation_id.get(cid) in children
            ]
            capability = capability_matrix.capability_for(
                provider_id=observation.provider_id,
                provider_version=observation.provider_version,
                observation_type=observation.observation_type,
            )
            weight = policy.capability_weight.get(capability, 0.0)
            for rank, child_cluster in enumerate(local_children):
                if child_cluster is not None:
                    scores[child_cluster] += weight * rank

        result[parent_cluster] = tuple(sorted(children, key=lambda c: (scores[c], c)))

    return result
