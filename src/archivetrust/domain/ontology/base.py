"""The Observation type (ROADMAP.md S5.9, S5.6): a typed, ontology-mapped claim, referencing
Evidence by id, positioned in a ProviderObservationGraph via parent/child/related links.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, SerializeAsAny, model_validator

from archivetrust.domain.confidence.models import ProviderConfidence
from archivetrust.domain.evidence.models import Evidence
from archivetrust.domain.ontology.payloads import PAYLOAD_TYPE_BY_OBSERVATION_TYPE
from archivetrust.domain.ontology.payloads.base import ObservationPayload
from archivetrust.domain.ontology.scope import ObservationScopeMeasurement, measure_observation_scope
from archivetrust.domain.ontology.types import ObservationType
from archivetrust.domain.shared.ids import new_id
from archivetrust.domain.shared.versioning import CURRENT_ONTOLOGY_VERSION, CURRENT_SCHEMA_VERSION


class RelatedObservationLink(BaseModel):
    """A typed, non-hierarchical edge to another Observation (e.g. "captions", "annotates",
    "spans"). Parent/child edges are hierarchical and carried on their own fields instead.
    """

    model_config = ConfigDict(frozen=True)

    target_observation_id: str
    relation_type: str


class Observation(BaseModel):
    """A single provider's typed, ontology-mapped claim about one semantic slot.

    `provider_confidence` is read through from referenced Evidence and must never be invented
    (ROADMAP.md S5.9) -- the only supported construction path is `Observation.from_evidence`,
    which derives it rather than accepting it directly.
    """

    model_config = ConfigDict(frozen=True)

    observation_id: str
    provider_id: str
    provider_version: str
    observation_type: ObservationType
    payload: SerializeAsAny[ObservationPayload]
    evidence_ids: tuple[str, ...]
    provider_confidence: ProviderConfidence | None
    ontology_version: int
    schema_version: int
    parent_observations: tuple[str, ...] = ()
    child_observations: tuple[str, ...] = ()
    related_observations: tuple[RelatedObservationLink, ...] = ()
    graph_source: str | None = None
    scope: ObservationScopeMeasurement | None = None
    """How much semantic content this Observation's claim covers (Phase 19/20,
    `domain/ontology/scope.py`) -- optional and defaulted so historical telemetry recorded before
    this field existed still deserializes unchanged (no migration). Populated automatically by
    `from_evidence`; never asserted by a caller."""

    @model_validator(mode="before")
    @classmethod
    def _coerce_polymorphic_payload(cls, data: Any) -> Any:
        """Reconstructs the correct payload subclass from a plain dict (e.g. deserialized JSON),
        keyed by `observation_type`. Pydantic cannot infer this on its own because `payload` is
        typed as the abstract `ObservationPayload` base -- without this, round-trip
        deserialization would silently lose every subclass-specific field.
        """
        if isinstance(data, dict):
            payload = data.get("payload")
            observation_type = data.get("observation_type")
            if isinstance(payload, dict) and observation_type is not None:
                payload_cls = PAYLOAD_TYPE_BY_OBSERVATION_TYPE[ObservationType(observation_type)]
                data = {**data, "payload": payload_cls.model_validate(payload)}
        return data

    @model_validator(mode="after")
    def _validate(self) -> "Observation":
        if not self.evidence_ids:
            raise ValueError(
                "Observation.evidence_ids must be non-empty -- every Observation must trace to "
                "at least one Evidence record (Constitution Article 4)"
            )
        if self.payload.observation_type != self.observation_type:
            raise ValueError(
                f"Observation.observation_type ({self.observation_type}) does not match its "
                f"payload's type ({self.payload.observation_type})"
            )
        return self

    @classmethod
    def from_evidence(
        cls,
        *,
        provider_id: str,
        provider_version: str,
        payload: ObservationPayload,
        evidence: tuple[Evidence, ...],
        parent_observations: tuple[str, ...] = (),
        child_observations: tuple[str, ...] = (),
        related_observations: tuple[RelatedObservationLink, ...] = (),
        graph_source: str | None = None,
        ontology_version: int = CURRENT_ONTOLOGY_VERSION,
        schema_version: int = CURRENT_SCHEMA_VERSION,
    ) -> "Observation":
        """The only supported construction path. Derives `provider_confidence` from the
        referenced Evidence rather than accepting it as a caller-supplied value, so it can never
        be invented independently of what the provider actually reported.

        If the referenced Evidence records disagree on `provider_confidence` (possible when one
        Observation is corroborated by two Evidence records from the same provider at different
        processing stages), `provider_confidence` is left unset (None) rather than picking one
        arbitrarily or averaging them -- averaging would fabricate a value no single Evidence
        record actually asserts.
        """
        if not evidence:
            raise ValueError("Observation.from_evidence requires at least one Evidence record")
        for record in evidence:
            if record.provider != provider_id or record.provider_version != provider_version:
                raise ValueError(
                    "All Evidence passed to Observation.from_evidence must share this "
                    "Observation's provider_id and provider_version"
                )

        distinct_confidences = {
            record.provider_confidence for record in evidence if record.provider_confidence is not None
        }
        provider_confidence = None
        if len(distinct_confidences) == 1:
            (value,) = distinct_confidences
            provider_confidence = ProviderConfidence(
                provider=provider_id, provider_version=provider_version, value=value
            )

        return cls(
            observation_id=new_id("observation"),
            provider_id=provider_id,
            provider_version=provider_version,
            observation_type=payload.observation_type,
            payload=payload,
            evidence_ids=tuple(record.evidence_id for record in evidence),
            provider_confidence=provider_confidence,
            ontology_version=ontology_version,
            schema_version=schema_version,
            parent_observations=parent_observations,
            child_observations=child_observations,
            related_observations=related_observations,
            graph_source=graph_source,
            scope=measure_observation_scope(payload, evidence),
        )
