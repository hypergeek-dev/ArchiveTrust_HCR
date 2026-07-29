"""The `AlignmentService` seam (ROADMAP.md S5.12; Constitution Article 24).

The stable interface between Observations and Comparison. The Comparison Engine consumes an
`AlignmentService`'s `AlignmentResult` instead of calling clustering directly, so alignment is an
explicit, replaceable responsibility rather than logic embedded in the engine.

`ClusteringAlignmentService` is the current — and only — implementation. It **wraps the existing
clustering unchanged**: it calls `domain.comparison.clustering.cluster_observations` and returns its
clusters verbatim as the authoritative grouping, then builds the observability record (candidates,
selections, per-Observation Aligned/Unaligned states) *from* that result. It adds no grouping logic,
no heuristic, and no scoring (S5.12 non-goals). A future strategy (heuristic, PDF-anchor,
graph-based, ML-assisted) implements the same `AlignmentService` protocol and is swapped in with no
change to the Comparison Engine or anything downstream of it.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from archivetrust.domain.alignment.models import (
    ALIGNMENT_ALGORITHM_VERSION,
    AlignmentAttempt,
    AlignmentObservationState,
    AlignmentOutcome,
    AlignmentResult,
    ExcludedCandidatePair,
)
from archivetrust.domain.comparison.clustering import (
    ClusteringBasisCode,
    _pairwise_affinity,
    alignment_scope_bucket_of,
    cluster_observations,
)
from archivetrust.domain.comparison.geometry import page_of
from archivetrust.domain.comparison.policy import ReconciliationPolicy
from archivetrust.domain.evidence.models import Evidence
from archivetrust.domain.ontology.base import Observation
from archivetrust.domain.ontology.types import ObservationType
from archivetrust.domain.shared.ids import new_id


@runtime_checkable
class AlignmentService(Protocol):
    """Groups Observations into comparison groups and records the grouping as observable evidence.

    Implementations must be deterministic in the same sense as the rest of the engine: identical
    `(observations, evidence, policy)` produce identical grouping *decisions* (Constitution Article
    24). The `AlignmentResult.clusters` it returns is the sole grouping the Comparison Engine
    consumes; the rest of the result is observability.
    """

    algorithm_name: str
    algorithm_version: int

    def align(
        self,
        observations: tuple[Observation, ...],
        evidence_by_id: dict[str, Evidence],
        policy: ReconciliationPolicy,
    ) -> AlignmentResult:
        ...


_EXCLUSION_BY_CODE = {
    ClusteringBasisCode.AMBIGUOUS_MULTI_PER_PROVIDER: (
        "clustering.ambiguous_multi_per_provider_guard",
        True,
    ),
    ClusteringBasisCode.SCOPE_MISMATCH: (
        "alignment.scope_compatibility_gate",
        False,
    ),
}
"""Maps exclusion codes to a human-readable mechanism name and whether the exclusion is structural.

`AMBIGUOUS_MULTI_PER_PROVIDER` is the original Article 27 structural exclusion. `SCOPE_MISMATCH`
is F5's policy exclusion: auditable in the same candidate-exclusion stream, but not structural in
the geometry/topology sense.
"""


class ClusteringAlignmentService:
    """The current alignment strategy: the existing content-independent clustering, made observable.

    Grouping is **entirely** delegated to `cluster_observations`; this class never decides which
    Observations belong together. It only reconstructs, for each produced group, the pool of
    candidates that group was drawn from, and records each Observation's terminal Aligned/Unaligned
    state — the observability the raw clustering output did not carry.
    """

    algorithm_name = "connected-components-geometry-clustering"
    algorithm_version = ALIGNMENT_ALGORITHM_VERSION

    @staticmethod
    def _excluded_pairs(
        observations: tuple[Observation, ...],
        evidence_by_id: dict[str, Evidence],
        policy: ReconciliationPolicy,
    ) -> tuple[ExcludedCandidatePair, ...]:
        """Recomputes clustering's own deterministic pool formation (Article 10: content-
        independent, same `{type, page}` grouping `cluster_observations` uses internally) purely to
        surface which pairwise comparisons `_pairwise_affinity` structurally excluded (Article 27).
        A pure, read-only re-derivation -- never a second grouping decision (this class never
        decides groupings, per the class docstring); identical inputs always re-derive identical
        exclusions, since `_pairwise_affinity` is itself deterministic (Article 24).
        """
        pools: dict[tuple[ObservationType, int | None], list[Observation]] = {}
        for obs in sorted(observations, key=lambda o: o.observation_id):
            page = page_of(obs, evidence_by_id)
            pools.setdefault((obs.observation_type, page), []).append(obs)

        excluded: list[ExcludedCandidatePair] = []
        for pool in pools.values():
            provider_counts: dict[tuple[str, str], int] = {}
            for obs in pool:
                key = (obs.provider_id, obs.provider_version)
                provider_counts[key] = provider_counts.get(key, 0) + 1
            for i in range(len(pool)):
                for j in range(i + 1, len(pool)):
                    _score, _basis, basis_code = _pairwise_affinity(
                        pool[i],
                        pool[j],
                        i,
                        j,
                        evidence_by_id,
                        provider_counts,
                        policy.enforce_scope_compatible_alignment,
                    )
                    # Only AMBIGUOUS_MULTI_PER_PROVIDER is a structural exclusion (Article 27);
                    # PIXEL_ACCURATE_IOU/ORDINAL_POSITION_FALLBACK are ordinary scored outcomes,
                    # even when their score happens to be low -- Article 27 governs exclusion
                    # mechanisms, never merely-low-affinity pairs.
                    if basis_code in _EXCLUSION_BY_CODE:
                        mechanism, structural = _EXCLUSION_BY_CODE[basis_code]
                        excluded.append(
                            ExcludedCandidatePair(
                                candidate_observation_id=pool[i].observation_id,
                                compared_against_observation_id=pool[j].observation_id,
                                excluding_mechanism=mechanism,
                                basis_code=basis_code,
                                structural=structural,
                            )
                        )
        return tuple(excluded)

    def align(
        self,
        observations: tuple[Observation, ...],
        evidence_by_id: dict[str, Evidence],
        policy: ReconciliationPolicy,
    ) -> AlignmentResult:
        clusters = cluster_observations(observations, evidence_by_id, policy)
        excluded_pairs = self._excluded_pairs(observations, evidence_by_id, policy)

        # Reconstruct the candidate pools exactly as clustering formed them, so an
        # AlignmentAttempt can record what was *considered*, not only what was *selected*.
        candidates_by_key: dict[
            tuple[ObservationType, int | None]
            | tuple[ObservationType, int | None, str],
            list[str],
        ] = {}
        for obs in observations:
            if policy.enforce_scope_compatible_alignment:
                key = (
                    obs.observation_type,
                    page_of(obs, evidence_by_id),
                    alignment_scope_bucket_of(obs),
                )
            else:
                key = (obs.observation_type, page_of(obs, evidence_by_id))
            candidates_by_key.setdefault(key, []).append(obs.observation_id)

        attempts: list[AlignmentAttempt] = []
        states: list[AlignmentObservationState] = []
        for cluster in clusters:
            members = cluster.member_observation_ids
            if policy.enforce_scope_compatible_alignment:
                first_member = next(
                    obs for obs in observations if obs.observation_id == members[0]
                )
                key = (
                    cluster.observation_type,
                    cluster.page,
                    alignment_scope_bucket_of(first_member),
                )
            else:
                key = (cluster.observation_type, cluster.page)
            candidates = tuple(sorted(candidates_by_key.get(key, [])))
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

            # Aligned iff grouped with at least one other Observation; a singleton group is the
            # explicit "no other Observation aligned to it" state (Article 24), never dropped.
            outcome = (
                AlignmentOutcome.ALIGNED if len(members) > 1 else AlignmentOutcome.UNALIGNED
            )
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
            excluded_pairs=excluded_pairs,
        )
