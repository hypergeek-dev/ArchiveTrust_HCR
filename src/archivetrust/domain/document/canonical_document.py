"""Canonical Document (ROADMAP.md S5.11, MILESTONE1_DOMAIN_MODEL.md S6, Constitution Article 12).

An aggregate over Canonical Observations -- a named, versioned view of which reconciled facts
constitute "this document." Asserts nothing a Canonical Observation doesn't already assert; holds
no independent payload content and computes no independent confidence. The single legal source is
the ReconciledObservationGraph (Article 12) -- enforced here by having no public constructor that
accepts raw provider data, only `assemble()`, which requires a ReconciledObservationGraph.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_validator

from archivetrust.domain.graph.reconciled_graph import ReconciledObservationGraph
from archivetrust.domain.shared.ids import new_id


class CanonicalDocument(BaseModel):
    model_config = ConfigDict(frozen=True)

    logical_document_id: str
    document_snapshot_id: str
    archive_object_ref: str
    contained_observations: tuple[str, ...]
    ontology_version: int
    document_version: int
    reassembly_trigger: str
    supersedes: str | None = None
    # Derived by the journal from later snapshots' `supersedes`, never self-set -- same rule as
    # CanonicalObservation.superseded_by.
    superseded_by: str | None = None

    @model_validator(mode="after")
    def _validate(self) -> "CanonicalDocument":
        if not self.contained_observations:
            raise ValueError("CanonicalDocument.contained_observations must be non-empty")
        if self.document_version < 0:
            raise ValueError("CanonicalDocument.document_version must be >= 0")
        return self

    @classmethod
    def assemble(
        cls,
        *,
        reconciled_graph: ReconciledObservationGraph,
        archive_object_ref: str,
        reassembly_trigger: str,
        logical_document_id: str | None = None,
        supersedes: str | None = None,
    ) -> "CanonicalDocument":
        """The only supported construction path (Article 12: exactly one legal source). Root
        Canonical Observations (S5.6: hierarchy is a graph property) become `contained_observations`,
        referenced by id -- never embedded by copy, and never a shortcut around the Canonical
        Observation -> Observation -> Evidence traceability chain (S5.11's Evidence-reference ban).
        """
        roots = reconciled_graph.roots()
        return cls(
            logical_document_id=logical_document_id or new_id("document"),
            document_snapshot_id=new_id("document_snapshot"),
            archive_object_ref=archive_object_ref,
            contained_observations=tuple(node.canonical_observation_id for node in roots),
            ontology_version=max(node.ontology_version for node in reconciled_graph.canonical_observations),
            document_version=reconciled_graph.reconciliation_sequence,
            reassembly_trigger=reassembly_trigger,
            supersedes=supersedes,
        )
