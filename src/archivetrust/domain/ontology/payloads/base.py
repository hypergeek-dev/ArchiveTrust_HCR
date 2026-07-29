"""Base type for Observation payloads, and the Graph-Reference Rule it enforces.

MILESTONE1_DOMAIN_MODEL.md S5.1 (the Graph-Reference Rule, Constitution Article 7): a payload may
hold semantic *content* but must never hold a field whose sole purpose is to reference another
Observation and name the relationship -- that belongs on a typed graph edge
(`related_observations`, S2-S3), never duplicated as payload content. Concretely, no payload class
in this package may define a field that is itself an Observation id (a `..._observation_id`
field) -- such a reference belongs on `Observation.related_observations`
(`archivetrust.domain.ontology.base.RelatedObservationLink`) instead.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict

from archivetrust.domain.ontology.types import ObservationType


class ObservationPayload(BaseModel):
    """Abstract base for all Observation payload types.

    Subclasses must set `observation_type` to the `ObservationType` member they implement,
    matched against `Observation.observation_type` at construction time (base.py).
    """

    model_config = ConfigDict(frozen=True)

    observation_type: ClassVar[ObservationType]

    # Historically set True only by RelationshipPayload (MILESTONE1_DOMAIN_MODEL.md S5.10's
    # explicit line between Relationship-as-Observation and relationship-as-graph-edge), deleted in
    # docs/htr-migration-plan.md Stage 5 (EXECUTED) along with `ObservationType.RELATIONSHIP` --
    # NER-style entity/relationship extraction is out of scope for HTR research. No current payload
    # sets this `True`; the seam is retained (not removed) for a future payload type that
    # legitimately needs to carry an Observation reference as content, not just as a graph edge.
    _allows_observation_references: ClassVar[bool] = False

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if cls._allows_observation_references:
            return
        declared_fields = cls.__dict__.get("__annotations__", {})
        for field_name in declared_fields:
            if field_name.endswith("_observation_id") or field_name.endswith("_observation_ids"):
                raise TypeError(
                    f"{cls.__name__}.{field_name} violates the Graph-Reference Rule "
                    "(MILESTONE1_DOMAIN_MODEL.md S5.1, Constitution Article 7): a payload may "
                    "not hold a field that is only a reference to another Observation. Express "
                    "this as a related_observations graph edge instead."
                )
