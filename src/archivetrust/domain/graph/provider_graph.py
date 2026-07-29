"""ProviderObservationGraph (MILESTONE1_DOMAIN_MODEL.md S2).

A single provider's own claimed internal structure, for one invocation. Raw, single-source
structural signal -- an input to comparison, never a conclusion, and never sufficient by itself to
feed anything downstream of the Comparison Engine (S2.6).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_validator

from archivetrust.domain.graph._acyclic import assert_acyclic
from archivetrust.domain.ontology.base import Observation


class ProviderObservationGraph(BaseModel):
    """Parent/child/related edges only among Observations sharing this graph's `provider_id`,
    `provider_version`, and `invocation_id` (S2.3) -- the structural guarantee that prevents one
    provider's claims from silently becoming cross-provider structure by default.
    """

    model_config = ConfigDict(frozen=True)

    provider_id: str
    provider_version: str
    invocation_id: str
    observations: tuple[Observation, ...]

    @model_validator(mode="after")
    def _validate(self) -> "ProviderObservationGraph":
        if not self.observations:
            raise ValueError("ProviderObservationGraph.observations must be non-empty")

        ids = {obs.observation_id for obs in self.observations}
        if len(ids) != len(self.observations):
            raise ValueError("ProviderObservationGraph.observations contains duplicate ids")

        for obs in self.observations:
            if obs.provider_id != self.provider_id or obs.provider_version != self.provider_version:
                raise ValueError(
                    f"Observation {obs.observation_id} belongs to provider "
                    f"{obs.provider_id}/{obs.provider_version}, not this graph's "
                    f"{self.provider_id}/{self.provider_version} (S2.3: a ProviderObservationGraph "
                    "must never link Observations from two different providers)"
                )
            referenced = (
                *obs.parent_observations,
                *obs.child_observations,
                *(link.target_observation_id for link in obs.related_observations),
            )
            for target in referenced:
                if target not in ids:
                    raise ValueError(
                        f"Observation {obs.observation_id} references {target}, which is not "
                        "part of this ProviderObservationGraph"
                    )

        assert_acyclic({obs.observation_id: obs.child_observations for obs in self.observations})
        return self
