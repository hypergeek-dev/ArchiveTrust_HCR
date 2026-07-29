"""Phase C -- Observation Matching (MILESTONE4_COMPARISON_ENGINE.md S3, Constitution Article 18).

Establishes each participating provider's cardinality of contribution to a fixed cluster. Cardinality
is geometric/structural (inherited from clustering), never text-based -- "do the values agree" is
Phase D's question, not this phase's (S3: "the 'approximate' question ... is Phase D, deliberately
separated").

**Known, documented simplification:** `ONE_TO_MANY` (the shared-contribution case, C11) is defined
in `MatchCardinality` and the domain model already supports representing it
(`CanonicalObservation.contributing_observations` allows one Observation to be referenced by more
than one Canonical Observation, tested in Milestone 1). This module's `match_cluster`, however,
only classifies within one fixed cluster produced by a strictly-partitioning clustering phase
(`clustering.py` assigns each Observation to exactly one cluster) -- it cannot itself detect "this
one Observation actually spans two slots," since that requires clustering to place an Observation
in more than one cluster, which this implementation's clustering does not yet do (see
`clustering.py`'s module docstring). `ONE_TO_MANY` is therefore never produced automatically today;
recorded here rather than silently omitted from the enum.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict

from archivetrust.domain.comparison.capability_matrix import Capability, CapabilityMatrix
from archivetrust.domain.comparison.clustering import Cluster
from archivetrust.domain.ontology.base import Observation


class MatchCardinality(str, Enum):
    ONE_TO_ONE = "one_to_one"
    MANY_TO_ONE = "many_to_one"
    ONE_TO_MANY = "one_to_many"
    MISSING = "missing"
    NOT_ATTEMPTED = "not_attempted"


class ProviderContribution(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider_id: str
    provider_version: str
    cardinality: MatchCardinality
    observation_ids: tuple[str, ...]


def match_cluster(
    cluster: Cluster,
    observations_by_id: dict[str, Observation],
    participating_providers: tuple[tuple[str, str], ...],
    capability_matrix: CapabilityMatrix,
) -> tuple[ProviderContribution, ...]:
    """`participating_providers` is every `(provider_id, provider_version)` that produced a
    `ProviderObservationGraph` for this Archive Object -- i.e., was actually invoked, regardless
    of whether it contributed to *this* cluster. A provider never invoked at all for this document
    does not appear here and is not classified (Constitution Article 18's "never invoked" case is
    a fact about the whole document, established upstream by `ProviderObservationAttempted`
    telemetry -- not something this per-cluster function re-derives).
    """
    members = [observations_by_id[oid] for oid in cluster.member_observation_ids]

    contributions: list[ProviderContribution] = []
    for provider_id, provider_version in sorted(participating_providers):
        provider_members = tuple(
            m.observation_id
            for m in members
            if m.provider_id == provider_id and m.provider_version == provider_version
        )
        if len(provider_members) == 1:
            cardinality = MatchCardinality.ONE_TO_ONE
        elif len(provider_members) > 1:
            cardinality = MatchCardinality.MANY_TO_ONE
        else:
            capability = capability_matrix.capability_for(
                provider_id=provider_id,
                provider_version=provider_version,
                observation_type=cluster.observation_type,
            )
            cardinality = (
                MatchCardinality.NOT_ATTEMPTED
                if capability == Capability.NO
                else MatchCardinality.MISSING
            )
        contributions.append(
            ProviderContribution(
                provider_id=provider_id,
                provider_version=provider_version,
                cardinality=cardinality,
                observation_ids=provider_members,
            )
        )
    return tuple(contributions)
