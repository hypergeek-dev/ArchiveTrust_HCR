"""ReconciledObservationGraph (MILESTONE1_DOMAIN_MODEL.md S3).

The single, cross-provider, authoritative structural view of a document. It **is** the aggregate
of all Canonical Observations for one document plus their mutual edges (S3.2) -- not a third,
independent structure. Constructible only from Comparison Engine output (Milestone 4); this type
only enforces the shape and invariants Milestone 1 owns.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_validator

from archivetrust.domain.canonical.observation import CanonicalObservation
from archivetrust.domain.graph._acyclic import assert_acyclic


class ReconciledObservationGraph(BaseModel):
    """Contains only Canonical Observations as nodes -- never raw Observations (S3.5). Versioned
    per reconciliation pass via `reconciliation_sequence`, the same axis introduced for Canonical
    Observation (S1.8); prior versions are retained elsewhere for replay, never overwritten here.
    """

    model_config = ConfigDict(frozen=True)

    reconciliation_sequence: int
    canonical_observations: tuple[CanonicalObservation, ...]

    @model_validator(mode="after")
    def _validate(self) -> "ReconciledObservationGraph":
        if not self.canonical_observations:
            raise ValueError("ReconciledObservationGraph.canonical_observations must be non-empty")

        ids = {node.canonical_observation_id for node in self.canonical_observations}
        if len(ids) != len(self.canonical_observations):
            raise ValueError("ReconciledObservationGraph.canonical_observations contains duplicate ids")

        for node in self.canonical_observations:
            referenced = (
                *node.parent_ids,
                *node.child_ids,
                *(link.target_canonical_observation_id for link in node.related_observations),
            )
            for target in referenced:
                if target not in ids:
                    raise ValueError(
                        f"CanonicalObservation {node.canonical_observation_id} references "
                        f"{target}, which is not part of this ReconciledObservationGraph"
                    )

        assert_acyclic(
            {node.canonical_observation_id: node.child_ids for node in self.canonical_observations}
        )
        return self

    def roots(self) -> tuple[CanonicalObservation, ...]:
        """Canonical Observations with no parent within this graph -- the document-hierarchy
        roots (S5.6: Document Hierarchy is a derived graph property, not a payload type)."""
        return tuple(node for node in self.canonical_observations if not node.parent_observations)
