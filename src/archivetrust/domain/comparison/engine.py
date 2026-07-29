"""The Comparison Engine (MILESTONE4_COMPARISON_ENGINE.md S1): orchestrates Phases A-F into one
pure, deterministic function producing the `ReconciledObservationGraph` plus a per-cluster
Comparison bundle (`AgreementReport`, `EvidenceReport`, `TrustScore`, `RiskScore`).

**Scope boundary (S0, restated):** this engine does not compute Canonical Confidence (Milestone 5)
and does not blend raw Provider Confidence across providers (C8) -- every `CanonicalObservation`
this produces is complete except `canonical_confidence`, exactly as
`MILESTONE1_DOMAIN_MODEL.md` S1.3 requires.

**Table handling (docs/htr-repository-cleanup.md domain-model table -- EXECUTED, Stage 5):**
`TableCellPayload`/`ObservationType.TABLE_CELL` and the cell-level neutral-lattice reconciliation
pass (`table_reconciliation.py`) were deleted -- table semantics don't fit line-level HTR research
and no in-scope HTR method produces structured tables. `ObservationType.TABLE` itself is retained
(it is a distinct, non-cell type), but a `TABLE` cluster is now reconciled through the same generic
path as any other non-text-bearing type (a single deterministic pick), not a dedicated cell pass.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from archivetrust.domain.alignment.models import AlignmentResult
from archivetrust.domain.alignment.service import AlignmentService, ClusteringAlignmentService
from archivetrust.domain.canonical.observation import CanonicalGraphEdge, CanonicalObservation
from archivetrust.domain.comparison.capability_matrix import CapabilityMatrix
from archivetrust.domain.comparison.matching import MatchCardinality, match_cluster
from archivetrust.domain.comparison.policy import ReconciliationPolicy
from archivetrust.domain.comparison.reporting import (
    AgreementReport,
    EvidenceReport,
    RiskScore,
    TrustScore,
    build_agreement_report,
    build_evidence_report,
    compute_risk_score,
    compute_trust_score,
)
from archivetrust.domain.comparison.structural import reconcile_structure
from archivetrust.domain.comparison.text_reconciliation import (
    ReconciliationBasisCode,
    TextCandidate,
    reconcile_text,
)
from archivetrust.domain.confidence.models import ComparisonClassification, ComparisonConfidence
from archivetrust.domain.evidence.models import Evidence
from archivetrust.domain.graph.provider_graph import ProviderObservationGraph
from archivetrust.domain.graph.reconciled_graph import ReconciledObservationGraph
from archivetrust.domain.ontology.base import Observation
from archivetrust.domain.ontology.types import ObservationType

_TEXT_BEARING_TYPES = {
    ObservationType.HEADING,
    ObservationType.PARAGRAPH,
    ObservationType.CAPTION,
    ObservationType.FOOTNOTE,
}


class ComparisonBundle(BaseModel):
    model_config = ConfigDict(frozen=True)

    agreement_report: AgreementReport
    evidence_report: EvidenceReport
    trust_score: TrustScore
    risk_score: RiskScore


class ComparisonEngineResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    reconciled_graph: ReconciledObservationGraph
    bundles: dict[str, ComparisonBundle]
    """Keyed by `canonical_observation_id`."""
    alignment: AlignmentResult
    """The observable record of how the input Observations were grouped into comparison groups
    (ROADMAP.md S5.12, Article 24) -- the grouping the rest of this engine consumed, its rationale,
    and every Observation's Aligned/Unaligned outcome. The grouping decisions are unchanged from the
    prior direct clustering call; this field only makes them observable."""


def _existence_confidence(independent_count: int, policy: ReconciliationPolicy) -> ComparisonConfidence:
    if independent_count <= 1:
        return ComparisonConfidence(
            classification=ComparisonClassification.UNCORROBORATED_SINGLE_SOURCE,
            magnitude=None,
            basis="only one independent provider observed this slot",
        )
    return ComparisonConfidence(
        classification=ComparisonClassification.CORROBORATED,
        magnitude=policy.round_score(min(1.0, independent_count / 3)),
        basis=f"{independent_count} independent providers observed this slot (existence-only agreement)",
    )


def run_comparison_engine(
    provider_graphs: tuple[ProviderObservationGraph, ...],
    evidence_by_id: dict[str, Evidence],
    capability_matrix: CapabilityMatrix,
    policy: ReconciliationPolicy,
    reconciliation_sequence: int = 0,
    alignment_service: AlignmentService | None = None,
) -> ComparisonEngineResult:
    observations_by_id: dict[str, Observation] = {}
    for graph in provider_graphs:
        for observation in graph.observations:
            observations_by_id[observation.observation_id] = observation
    all_observations = tuple(observations_by_id.values())

    participating_providers = tuple(
        sorted({(g.provider_id, g.provider_version) for g in provider_graphs})
    )

    # Alignment (grouping) is now performed through a replaceable seam (ROADMAP.md S5.12). The
    # default service wraps the same clustering that was previously called directly here, so the
    # grouping -- and therefore every downstream result -- is byte-for-byte unchanged; the seam only
    # makes the grouping an observable, versioned, replaceable responsibility.
    alignment_service = alignment_service or ClusteringAlignmentService()
    alignment = alignment_service.align(all_observations, evidence_by_id, policy)
    clusters = alignment.clusters

    canonical_by_cluster: dict[str, CanonicalObservation] = {}
    bundles: dict[str, ComparisonBundle] = {}

    for cluster in clusters:
        members = tuple(observations_by_id[oid] for oid in cluster.member_observation_ids)
        contributions = match_cluster(cluster, observations_by_id, participating_providers, capability_matrix)
        independent = [
            c for c in contributions if c.cardinality in (MatchCardinality.ONE_TO_ONE, MatchCardinality.MANY_TO_ONE)
        ]

        first_payload = members[0].payload
        if cluster.observation_type in _TEXT_BEARING_TYPES and hasattr(first_payload, "text"):
            candidates = tuple(
                TextCandidate(observation_id=m.observation_id, provider_id=m.provider_id, text=getattr(m.payload, "text") or "")
                for m in members
            )
            text_result = reconcile_text(candidates, policy)
            payload = first_payload.model_copy(update={"text": text_result.accepted_text})
            comparison_confidence = ComparisonConfidence(
                classification=ComparisonClassification(text_result.classification.value),
                magnitude=text_result.magnitude,
                basis=text_result.reconciliation_basis,
            )
            reconciliation_basis = text_result.reconciliation_basis
            reconciliation_basis_code = text_result.reconciliation_basis_code.value
        else:
            reference = sorted(members, key=lambda m: m.observation_id)[0]
            payload = reference.payload
            comparison_confidence = _existence_confidence(len(independent), policy)
            reconciliation_basis = (
                f"single deterministic pick (observation_id={reference.observation_id}) -- "
                "no content-level reconciliation defined for this observation type"
            )
            reconciliation_basis_code = ReconciliationBasisCode.NON_TEXT_DETERMINISTIC_PICK.value

        canonical = CanonicalObservation.reconcile(
            semantic_slot_id=cluster.cluster_id,
            payload=payload,
            contributing_observations=members,
            comparison_confidence=comparison_confidence,
            clustering_basis=cluster.clustering_basis,
            reconciliation_basis=reconciliation_basis,
            reconciliation_basis_code=reconciliation_basis_code,
            reconciliation_sequence=reconciliation_sequence,
        )
        canonical_by_cluster[cluster.cluster_id] = canonical

        agreement_report = build_agreement_report(
            cluster_id=cluster.cluster_id,
            observation_type=cluster.observation_type,
            clustering_basis=cluster.clustering_basis,
            contributions=contributions,
            value_comparison_confidence=comparison_confidence
            if cluster.observation_type in _TEXT_BEARING_TYPES
            else None,
            policy=policy,
        )
        evidence_report = build_evidence_report(
            cluster_id=cluster.cluster_id, contributing_observations=members, evidence_by_id=evidence_by_id
        )
        bundles[canonical.canonical_observation_id] = ComparisonBundle(
            agreement_report=agreement_report,
            evidence_report=evidence_report,
            trust_score=compute_trust_score(agreement_report, policy),
            risk_score=compute_risk_score(agreement_report, policy),
        )

    structural_result = reconcile_structure(clusters, observations_by_id, capability_matrix, policy)

    edges_by_parent: dict[str, list] = {}
    parent_of_child: dict[str, str] = {}
    for edge in structural_result.edges:
        if not edge.elected:
            continue
        edges_by_parent.setdefault(edge.parent_cluster_id, []).append(edge)
        parent_of_child[edge.child_cluster_id] = edge.parent_cluster_id

    final_canonicals: list[CanonicalObservation] = []
    for cluster_id, canonical in canonical_by_cluster.items():
        order = structural_result.reading_order.get(cluster_id)
        child_cluster_ids = order or tuple(sorted(c.child_cluster_id for c in edges_by_parent.get(cluster_id, [])))
        child_edges = tuple(
            CanonicalGraphEdge(
                target_canonical_observation_id=canonical_by_cluster[child_id].canonical_observation_id,
                comparison_confidence=next(
                    (e.comparison_confidence for e in edges_by_parent.get(cluster_id, []) if e.child_cluster_id == child_id),
                    None,
                ),
            )
            for child_id in child_cluster_ids
            if child_id in canonical_by_cluster
        )
        parent_cluster_id = parent_of_child.get(cluster_id)
        parent_edges = ()
        if parent_cluster_id and parent_cluster_id in canonical_by_cluster:
            matching_edge = next(
                (e for e in edges_by_parent.get(parent_cluster_id, []) if e.child_cluster_id == cluster_id), None
            )
            parent_edges = (
                CanonicalGraphEdge(
                    target_canonical_observation_id=canonical_by_cluster[parent_cluster_id].canonical_observation_id,
                    comparison_confidence=matching_edge.comparison_confidence if matching_edge else None,
                ),
            )

        updated = canonical.model_copy(
            update={
                "parent_observations": parent_edges,
                "child_observations": child_edges,
            }
        )
        final_canonicals.append(updated)

    reconciled_graph = ReconciledObservationGraph(
        reconciliation_sequence=reconciliation_sequence, canonical_observations=tuple(final_canonicals)
    )
    return ComparisonEngineResult(
        reconciled_graph=reconciled_graph, bundles=bundles, alignment=alignment
    )
