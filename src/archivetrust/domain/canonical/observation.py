"""Canonical Observation (MILESTONE1_DOMAIN_MODEL.md S1).

ArchiveTrust's own best-supported claim about one semantic slot in a document, after
reconciliation across whatever Observations were available for that slot. Not a copy of any
single provider's Observation; the atomic unit the Canonical Document is assembled from (S1.1).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, SerializeAsAny, model_validator

from archivetrust.domain.confidence.models import CanonicalConfidence, ComparisonConfidence
from archivetrust.domain.ontology.base import Observation
from archivetrust.domain.ontology.payloads import PAYLOAD_TYPE_BY_OBSERVATION_TYPE
from archivetrust.domain.ontology.payloads.base import ObservationPayload
from archivetrust.domain.ontology.types import ObservationType
from archivetrust.domain.shared.ids import new_id
from archivetrust.domain.shared.versioning import CURRENT_ONTOLOGY_VERSION, CURRENT_SCHEMA_VERSION


class ContributingObservationRef(BaseModel):
    """A reference (never an embedded copy, S1.4) to one Observation that contributed to a
    Canonical Observation. `aspect` names which part of the Observation's payload was actually
    used -- required whenever the same Observation also contributes to a *different* Canonical
    Observation (S1.2's shared-contribution exception), so reuse is distinguishable from
    independent corroboration.
    """

    model_config = ConfigDict(frozen=True)

    observation_id: str
    aspect: str | None = None


class CanonicalGraphEdge(BaseModel):
    """A parent/child edge between two Canonical Observations, carrying its own edge-level
    Comparison Confidence -- distinct from either endpoint's confidence
    (`MILESTONE1_DOMAIN_MODEL.md` S3.4: "ArchiveTrust may be highly confident that both X and Y
    exist ... while being much less confident about the specific parent/child relationship").

    Added during Milestone 4 implementation per `MILESTONE4_COMPARISON_ENGINE.md` S18 item 1,
    which pre-licenses exactly this change as an implementation detail already anticipated by
    S3.4, not a new architectural concept: "a graph link is `{target_id, relation_type,
    comparison_confidence?, agreement?}`" rather than a bare id. Replaces the bare
    `tuple[str, ...]` this milestone's `parent_observations`/`child_observations` fields
    originally shipped with (see IMPLEMENTATION_STATUS.md for the amendment record).
    """

    model_config = ConfigDict(frozen=True)

    target_canonical_observation_id: str
    comparison_confidence: ComparisonConfidence | None = None


class RelatedCanonicalObservationLink(BaseModel):
    """The Canonical-Observation-to-Canonical-Observation analogue of
    `archivetrust.domain.ontology.base.RelatedObservationLink`, with the same S18-item-1
    edge-confidence addition as `CanonicalGraphEdge`.
    """

    model_config = ConfigDict(frozen=True)

    target_canonical_observation_id: str
    relation_type: str
    comparison_confidence: ComparisonConfidence | None = None


class CanonicalObservation(BaseModel):
    """See module docstring. Invariants enforced here are exactly S1.2's list; anything not
    enforceable from this type's own fields alone (e.g. true cross-slot shared-contribution
    bookkeeping) is the ReconciledObservationGraph's responsibility (S3), not this type's.
    """

    model_config = ConfigDict(frozen=True)

    canonical_observation_id: str
    semantic_slot_id: str
    observation_type: ObservationType
    payload: SerializeAsAny[ObservationPayload]
    contributing_observations: tuple[ContributingObservationRef, ...]
    comparison_confidence: ComparisonConfidence
    canonical_confidence: CanonicalConfidence | None = None
    ontology_version: int
    schema_version: int
    reconciliation_sequence: int
    clustering_basis: str
    reconciliation_basis: str
    reconciliation_basis_code: str | None = None
    """Structured counterpart to `reconciliation_basis` (Constitution Article 26) -- e.g.
    `ReconciliationBasisCode.TEXT_MAJORITY_VOTE_CONSENSUS.value`. Kept as a plain `str` at this
    layer, not the `ReconciliationBasisCode` enum itself, so this module stays independent of
    `domain.comparison` (that module already depends on this one; the reverse would be circular).
    `None` only for callers that predate this field (never backfilled) -- never a guessed value."""
    parent_observations: tuple[CanonicalGraphEdge, ...] = ()
    child_observations: tuple[CanonicalGraphEdge, ...] = ()
    related_observations: tuple[RelatedCanonicalObservationLink, ...] = ()
    supersedes: str | None = None
    # Never set by this object on itself -- CanonicalObservation is frozen (Article 15:
    # supersession, never erasure). A predecessor's `superseded_by` is a derived fact, discovered
    # by the journal (Milestone 2) by finding whichever later version's `supersedes` points back
    # to it, not by mutating the stored predecessor.
    superseded_by: str | None = None
    human_correction_ref: str | None = None
    rationale: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_polymorphic_payload(cls, data: Any) -> Any:
        """Same purpose as Observation._coerce_polymorphic_payload -- see that docstring."""
        if isinstance(data, dict):
            payload = data.get("payload")
            observation_type = data.get("observation_type")
            if isinstance(payload, dict) and observation_type is not None:
                payload_cls = PAYLOAD_TYPE_BY_OBSERVATION_TYPE[ObservationType(observation_type)]
                data = {**data, "payload": payload_cls.model_validate(payload)}
        return data

    @model_validator(mode="after")
    def _validate(self) -> "CanonicalObservation":
        if not self.contributing_observations:
            raise ValueError(
                "CanonicalObservation.contributing_observations must be non-empty (S1.2: "
                "non-empty provenance -- there is no canonical fact without evidentiary basis)"
            )
        if self.payload.observation_type != self.observation_type:
            raise ValueError(
                "CanonicalObservation.observation_type does not match its payload's type"
            )
        if self.reconciliation_sequence < 0:
            raise ValueError("CanonicalObservation.reconciliation_sequence must be >= 0")
        return self

    @property
    def parent_ids(self) -> tuple[str, ...]:
        return tuple(edge.target_canonical_observation_id for edge in self.parent_observations)

    @property
    def child_ids(self) -> tuple[str, ...]:
        return tuple(edge.target_canonical_observation_id for edge in self.child_observations)

    @classmethod
    def reconcile(
        cls,
        *,
        semantic_slot_id: str,
        payload: ObservationPayload,
        contributing_observations: tuple[Observation, ...],
        comparison_confidence: ComparisonConfidence,
        clustering_basis: str,
        reconciliation_basis: str,
        reconciliation_sequence: int,
        reconciliation_basis_code: str | None = None,
        contribution_aspects: dict[str, str] | None = None,
        canonical_confidence: CanonicalConfidence | None = None,
        parent_observations: tuple[CanonicalGraphEdge, ...] = (),
        child_observations: tuple[CanonicalGraphEdge, ...] = (),
        related_observations: tuple[RelatedCanonicalObservationLink, ...] = (),
        supersedes: str | None = None,
        human_correction_ref: str | None = None,
        rationale: str | None = None,
        ontology_version: int = CURRENT_ONTOLOGY_VERSION,
        schema_version: int = CURRENT_SCHEMA_VERSION,
    ) -> "CanonicalObservation":
        """The only supported construction path. Takes the actual contributing Observation
        objects (never just their ids) so type homogeneity (S1.2) can be validated against real
        data, even though only references are retained on the resulting object.
        """
        if not contributing_observations:
            raise ValueError("reconcile() requires at least one contributing Observation")
        distinct_types = {obs.observation_type for obs in contributing_observations}
        if distinct_types != {payload.observation_type}:
            raise ValueError(
                "All contributing_observations must share the same observation_type as the "
                f"reconciled payload ({payload.observation_type}); got {distinct_types} (S1.2: "
                "a reconciled Heading is built only from Heading Observations, never mixed)"
            )

        aspects = contribution_aspects or {}
        refs = tuple(
            ContributingObservationRef(
                observation_id=obs.observation_id, aspect=aspects.get(obs.observation_id)
            )
            for obs in contributing_observations
        )

        return cls(
            canonical_observation_id=new_id("canonical_observation"),
            semantic_slot_id=semantic_slot_id,
            observation_type=payload.observation_type,
            payload=payload,
            contributing_observations=refs,
            comparison_confidence=comparison_confidence,
            canonical_confidence=canonical_confidence,
            ontology_version=ontology_version,
            schema_version=schema_version,
            reconciliation_sequence=reconciliation_sequence,
            clustering_basis=clustering_basis,
            reconciliation_basis=reconciliation_basis,
            reconciliation_basis_code=reconciliation_basis_code,
            parent_observations=parent_observations,
            child_observations=child_observations,
            related_observations=related_observations,
            supersedes=supersedes,
            human_correction_ref=human_correction_ref,
            rationale=rationale,
        )

    def supersede(
        self,
        *,
        payload: ObservationPayload,
        contributing_observations: tuple[Observation, ...],
        comparison_confidence: ComparisonConfidence,
        clustering_basis: str,
        reconciliation_basis: str,
        reconciliation_basis_code: str | None = None,
        contribution_aspects: dict[str, str] | None = None,
        canonical_confidence: CanonicalConfidence | None = None,
        human_correction_ref: str | None = None,
        rationale: str | None = None,
    ) -> "CanonicalObservation":
        """Produces a new version for this same semantic slot (Constitution Article 15:
        supersession, never erasure). The predecessor is returned unmodified by the caller's
        responsibility to retain it -- this method never mutates `self`.
        """
        new_version = CanonicalObservation.reconcile(
            semantic_slot_id=self.semantic_slot_id,
            payload=payload,
            contributing_observations=contributing_observations,
            comparison_confidence=comparison_confidence,
            clustering_basis=clustering_basis,
            reconciliation_basis=reconciliation_basis,
            reconciliation_basis_code=reconciliation_basis_code,
            reconciliation_sequence=self.reconciliation_sequence + 1,
            contribution_aspects=contribution_aspects,
            canonical_confidence=canonical_confidence,
            parent_observations=self.parent_observations,
            child_observations=self.child_observations,
            related_observations=self.related_observations,
            supersedes=self.canonical_observation_id,
            human_correction_ref=human_correction_ref,
            rationale=rationale,
            ontology_version=self.ontology_version,
            schema_version=self.schema_version,
        )
        return new_version
