"""Phase B -- Observation Clustering (MILESTONE4_COMPARISON_ENGINE.md S2, Constitution Article 10).

**The load-bearing invariant of the whole engine (C3+C4):** clustering groups Observations using
only `{type, geometry, precision, page/ordinal position}` -- payload content is never in scope,
enforced structurally by this module never importing `ObservationPayload` subclasses or reading
`observation.payload` at all.

**Deliberately deferred, documented, not silently dropped** (all explicitly gated behind corpus
validation by S19, not blockers to building the engine):
- The bimodal split pass (S2 step 4) that would separate a wrongly-merged connected component.
- Cross-page continuity clustering (S2 step 5, S18 R3) -- this implementation clusters within one
  page only; a table or paragraph spanning a page break becomes two single-page clusters, which is
  the same "conservative, prefer-split" outcome the spec asks for on uncertainty, just without the
  explicit `related` "continues" edge candidate the full design proposes.
- Containment-claim reinforcement (S2 step 3) -- an optional affinity *boost*, never load-bearing
  on its own; omitting it slightly weakens recall, never correctness (a provider's own containment
  claim was never sufficient to cluster alone, C5).
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict

from archivetrust.domain.comparison.geometry import (
    bounding_box_of,
    both_pixel_accurate,
    intersection_over_union,
    page_of,
)
from archivetrust.domain.comparison.policy import ReconciliationPolicy
from archivetrust.domain.evidence.models import Evidence
from archivetrust.domain.ontology.base import Observation
from archivetrust.domain.ontology.scope import (
    ObservationScopeClassification,
    ObservationScopeMeasurement,
    classify_observation_scope,
)
from archivetrust.domain.ontology.types import ObservationType
from archivetrust.domain.shared.ids import new_id


class ClusteringBasisCode(str, Enum):
    """Versioned, closed vocabulary for `_pairwise_affinity`'s decision paths (Constitution
    Article 26). Every member corresponds to a real, currently-reachable code path in this module,
    verified against `ARCHITECTURE_TELEMETRY_STANDARD.md` S1.1's evidentiary discipline before being
    added -- a code is never defined for a mechanism this module doesn't actually contain.
    """

    PIXEL_ACCURATE_IOU = "PIXEL_ACCURATE_IOU"
    """Both sides have pixel-accurate geometry; score is their bounding-box IoU."""
    ORDINAL_POSITION_FALLBACK = "ORDINAL_POSITION_FALLBACK"
    """At least one side lacks pixel-accurate geometry (coarse or absent alike -- this module does
    not distinguish the two, S1.1); score is distance-decayed ordinal position within the pool."""
    AMBIGUOUS_MULTI_PER_PROVIDER = "AMBIGUOUS_MULTI_PER_PROVIDER"
    """A structural exclusion (Constitution Article 27), not an ordinary low score: a provider
    contributed more than one same-type Observation to this page, so ordinal position within the
    arbitrary id-sorted pool carries no spatial meaning -- score is fixed at 0.0 regardless of
    geometry. The *only* structural exclusion mechanism currently shipped in this module; the
    geometry-precision guard some prior analysis evaluated remains an unshipped counterfactual
    (`ARCHITECTURE_TELEMETRY_STANDARD.md` S1.1 category 3 registry)."""
    CONTAINMENT_REINFORCEMENT = "CONTAINMENT_REINFORCEMENT"
    """Reserved, not yet reachable. This module's own docstring documents containment-claim
    reinforcement (S2 step 3) as deliberately deferred, not implemented -- kept here so the
    vocabulary doesn't need amending twice when it is."""
    SCOPE_MISMATCH = "SCOPE_MISMATCH"
    """F5 scope-aware alignment exclusion: two same-type, same-page observations describe
    incompatible semantic scales (for example a fragment vs. a whole-page extended transcript), so
    they are not candidates for one comparison group even when geometry/ordinal fallback might
    otherwise connect them."""


class AffinityEdge(BaseModel):
    model_config = ConfigDict(frozen=True)

    observation_id_a: str
    observation_id_b: str
    score: float
    basis: str
    basis_code: ClusteringBasisCode
    """Structured counterpart to `basis` (Constitution Article 26) -- in addition to, never instead
    of, the free-text explanation."""


class Cluster(BaseModel):
    model_config = ConfigDict(frozen=True)

    cluster_id: str
    observation_type: ObservationType
    page: int | None
    member_observation_ids: tuple[str, ...]
    clustering_basis: str
    affinity_edges: tuple[AffinityEdge, ...] = ()


def scope_classification_of(obs: Observation) -> ObservationScopeClassification:
    """Live policy interpretation of an Observation's stored scope measurement.

    Historical telemetry can deserialize with `scope=None`; treating that as UNIT preserves the
    pre-F5 ordinary single-unit assumption without inventing a migration.
    """
    measurement = obs.scope or ObservationScopeMeasurement(
        character_count=None,
        word_count=None,
        sentence_count=None,
        has_bounding_box=False,
        page_area_fraction=None,
    )
    return classify_observation_scope(measurement)


def alignment_scope_bucket_of(obs: Observation) -> str:
    """F5 compatibility bucket for grouping.

    `FRAGMENT` and `UNIT` remain comparable: a heading, label, or short paragraph can reasonably be
    another provider's ordinary unit-scale claim. `EXTENDED` is isolated because it represents the
    page-scale transcript failure mode F5 closes.
    """
    classification = scope_classification_of(obs)
    if classification == ObservationScopeClassification.EXTENDED:
        return ObservationScopeClassification.EXTENDED.value
    return "unit_or_fragment"


def _ordinal_affinity(index_a: int, index_b: int) -> float:
    """Fallback used when at least one Observation lacks a pixel-accurate box (S2 step 2: "a
    coarse box may confirm 'these are in the same region' but never 'these are the same object'
    on geometry alone" -- ordinal position within the type+page pool is the structural signal
    used instead). Distance-decayed, capped at 1.0 for the same position.
    """
    distance = abs(index_a - index_b)
    if distance == 0:
        return 1.0
    return max(0.0, 1.0 - 0.5 * distance)


def _pairwise_affinity(
    obs_a: Observation,
    obs_b: Observation,
    index_a: int,
    index_b: int,
    evidence_by_id: dict[str, Evidence],
    provider_counts: dict[tuple[str, str], int],
    enforce_scope_compatibility: bool = True,
) -> tuple[float, str, ClusteringBasisCode]:
    """Returns `(score, basis, basis_code)` -- every branch now identifies which of
    `ClusteringBasisCode`'s real code paths it took (Constitution Article 26), never `None`, since
    every reachable branch in this module is already enumerable. `cluster_observations`'s own
    clustering decision is unchanged by this addition -- it uses `score`/`basis` exactly as before;
    `basis_code` flows into `AffinityEdge` (this module) and, for the one code that represents a
    structural exclusion rather than an ordinary score (`AMBIGUOUS_MULTI_PER_PROVIDER`, Article 27),
    into `domain/alignment/service.py`'s `CandidateExcluded` emission.
    """
    if enforce_scope_compatibility:
        scope_a = alignment_scope_bucket_of(obs_a)
        scope_b = alignment_scope_bucket_of(obs_b)
        if scope_a != scope_b:
            return (
                0.0,
                f"scope mismatch ({scope_a} vs {scope_b})",
                ClusteringBasisCode.SCOPE_MISMATCH,
            )

    box_a = bounding_box_of(obs_a, evidence_by_id)
    box_b = bounding_box_of(obs_b, evidence_by_id)
    if both_pixel_accurate(box_a, box_b):
        assert box_a is not None and box_b is not None
        return (
            intersection_over_union(box_a, box_b),
            "pixel-accurate IoU",
            ClusteringBasisCode.PIXEL_ACCURATE_IOU,
        )

    # Ordinal position is only a meaningful proxy for "these describe the same object" when each
    # provider involved contributed exactly one Observation of this type to this page -- an
    # unambiguous 1:1 echo (e.g. one coarse-geometry Heading answering one pixel-accurate Heading).
    # `sorted(observations, key=observation_id)` establishes pool order for determinism only (C7);
    # it carries no spatial meaning, so once either provider contributed more than one same-type
    # Observation to the page there is no principled correspondence between "adjacent in this
    # arbitrary pool" and "adjacent on the page" -- guessing one produces exactly the failure mode
    # this fallback exists to avoid (unrelated content silently fused into one comparison). Verified
    # against the 2026-07-13 benchmark corpus: this ambiguous case is what fused PaddleOCR-VL's
    # single whole-page transcript (no geometry) into an arbitrary one of many same-page Tesseract
    # paragraph fragments in ~85% of that corpus's contested review packets.
    key_a = (obs_a.provider_id, obs_a.provider_version)
    key_b = (obs_b.provider_id, obs_b.provider_version)
    if provider_counts.get(key_a, 0) > 1 or provider_counts.get(key_b, 0) > 1:
        return (
            0.0,
            (
                "ordinal position undefined (ambiguous: a provider contributed more than one "
                "same-type Observation to this page)"
            ),
            ClusteringBasisCode.AMBIGUOUS_MULTI_PER_PROVIDER,
        )
    return (
        _ordinal_affinity(index_a, index_b),
        "ordinal position (coarse or missing geometry)",
        ClusteringBasisCode.ORDINAL_POSITION_FALLBACK,
    )


def _connected_components(
    member_ids: tuple[str, ...], edges: list[tuple[str, str, float]], threshold: float
) -> list[list[str]]:
    parent: dict[str, str] = {oid: oid for oid in member_ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        root_x, root_y = find(x), find(y)
        if root_x != root_y:
            # Deterministic union target: lexicographically smaller id wins as root.
            if root_x < root_y:
                parent[root_y] = root_x
            else:
                parent[root_x] = root_y

    for a, b, score in edges:
        if score >= threshold:
            union(a, b)

    components: dict[str, list[str]] = {}
    for oid in sorted(member_ids):
        root = find(oid)
        components.setdefault(root, []).append(oid)
    return [sorted(members) for members in components.values()]


def cluster_observations(
    observations: tuple[Observation, ...],
    evidence_by_id: dict[str, Evidence],
    policy: ReconciliationPolicy,
) -> tuple[Cluster, ...]:
    """Deterministic given `(observations, evidence_by_id, policy)` (C7): pool ordering is sorted
    by `observation_id` before any pairwise comparison, so input order never affects the result.
    """
    pool_key: tuple[ObservationType, int | None] | tuple[ObservationType, int | None, str]
    pools: dict[
        tuple[ObservationType, int | None] | tuple[ObservationType, int | None, str],
        list[Observation],
    ] = {}
    for obs in sorted(observations, key=lambda o: o.observation_id):
        page = page_of(obs, evidence_by_id)
        if policy.enforce_scope_compatible_alignment:
            pool_key = (obs.observation_type, page, alignment_scope_bucket_of(obs))
        else:
            pool_key = (obs.observation_type, page)
        pools.setdefault(pool_key, []).append(obs)

    clusters: list[Cluster] = []
    for key, pool in sorted(
        pools.items(),
        key=lambda kv: (
            kv[0][0].value,
            kv[0][1] or -1,
            kv[0][2] if len(kv[0]) > 2 else "",
        ),
    ):
        observation_type = key[0]
        page = key[1]
        scope_bucket = key[2] if len(key) > 2 else None
        member_ids = tuple(o.observation_id for o in pool)
        provider_counts: dict[tuple[str, str], int] = {}
        for obs in pool:
            key = (obs.provider_id, obs.provider_version)
            provider_counts[key] = provider_counts.get(key, 0) + 1
        edges: list[tuple[str, str, float]] = []
        affinity_records: list[AffinityEdge] = []
        for i in range(len(pool)):
            for j in range(i + 1, len(pool)):
                # basis_code (Article 26) is recorded on AffinityEdge below; it changes no scoring
                # or union decision here -- clustering's own behavior is unchanged by Article 26/27.
                score, basis, basis_code = _pairwise_affinity(
                    pool[i],
                    pool[j],
                    i,
                    j,
                    evidence_by_id,
                    provider_counts,
                    policy.enforce_scope_compatible_alignment,
                )
                score = round(score, policy.rounding_ndigits)
                edges.append((pool[i].observation_id, pool[j].observation_id, score))
                affinity_records.append(
                    AffinityEdge(
                        observation_id_a=pool[i].observation_id,
                        observation_id_b=pool[j].observation_id,
                        score=score,
                        basis=basis,
                        basis_code=basis_code,
                    )
                )

        components = _connected_components(member_ids, edges, policy.cluster_join_threshold)
        for component in components:
            component_edges = tuple(
                edge
                for edge in affinity_records
                if edge.observation_id_a in component and edge.observation_id_b in component
            )
            basis = (
                f"type={observation_type.value}, page={page}, "
                f"cluster_join_threshold={policy.cluster_join_threshold} "
                f"(Policy v{policy.policy_version}), {len(component)} member(s)"
            )
            if scope_bucket is not None:
                basis += f", scope={scope_bucket}"
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
