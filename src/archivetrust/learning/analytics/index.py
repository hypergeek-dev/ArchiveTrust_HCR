"""A read-only, derived index over the Trust Engine telemetry stream (ROADMAP_V2.md S13).

The Quality Center, Provider Health, and Evolution Center analyzers all need the same cross-event
joins -- "which providers contributed to this Canonical Observation," "which observation-types
were corrected," "how many independent sources attempted this slot." `TelemetryIndex` computes
those joins once, as an append-only derived view, so each analyzer is a small pure function over
the index rather than re-scanning raw events.

This is the analytics analogue of `application.journal.JournalState`, but cross-document (the
Journal is per-document) and strictly read-only over Trust Engine data: it copies references to
frozen domain objects, never mutates or reconstructs them. It never imports a provider or a write
path (ROADMAP_V2.md S5.2, Guiding Principle 2).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from archivetrust.domain.canonical.observation import CanonicalObservation
from archivetrust.domain.ontology.types import ObservationType
from archivetrust.domain.telemetry.events import (
    CanonicalDecisionCreated,
    EvidenceRejected,
    HumanCorrectionApplied,
    HumanCorrectionSubmitted,
    ObservationCreated,
    ProviderObservationAttempted,
)
from archivetrust.learning.source import TelemetrySource


@dataclass(frozen=True)
class ProviderRef:
    """A (provider_id, provider_version) pair. Kept whole because Provider Confidence and any
    future calibration are only ever meaningful per exact version, never per bare provider name
    (ROADMAP.md S5.3.1) -- collapsing the version away here would quietly enable the cross-version
    blending that invariant forbids.
    """

    provider_id: str
    provider_version: str


@dataclass
class TelemetryIndex:
    """Derived, cross-document view over one `TelemetrySource`. Built once via `build`; all fields
    are read-only query surfaces for the analyzers.
    """

    provider_by_observation: dict[str, ProviderRef] = field(default_factory=dict)
    observation_type_by_observation: dict[str, ObservationType] = field(default_factory=dict)

    canonical_by_id: dict[str, CanonicalObservation] = field(default_factory=dict)
    # Latest Canonical Observation per semantic slot, by highest reconciliation_sequence seen.
    latest_canonical_by_slot: dict[str, CanonicalObservation] = field(default_factory=dict)
    # semantic_slot_id -> document_ref (Milestone 11: adaptive sampling needs to reopen a review
    # packet for a corpus-wide-sampled slot, which requires the document it lives in -- the one
    # join CanonicalObservation itself doesn't carry).
    document_ref_by_slot: dict[str, str] = field(default_factory=dict)

    corrections_submitted: list[HumanCorrectionSubmitted] = field(default_factory=list)
    corrections_applied: list[HumanCorrectionApplied] = field(default_factory=list)

    provider_attempts: list[ProviderObservationAttempted] = field(default_factory=list)
    evidence_rejections: list[EvidenceRejected] = field(default_factory=list)

    @classmethod
    def build(cls, source: TelemetrySource) -> "TelemetryIndex":
        index = cls()
        for event in source.all_events():
            if isinstance(event, ObservationCreated):
                obs = event.observation
                index.provider_by_observation[obs.observation_id] = ProviderRef(
                    provider_id=obs.provider_id, provider_version=obs.provider_version
                )
                index.observation_type_by_observation[obs.observation_id] = obs.observation_type
            elif isinstance(event, CanonicalDecisionCreated):
                index._record_canonical(event.canonical_observation)
                index.document_ref_by_slot[event.canonical_observation.semantic_slot_id] = (
                    event.document_ref
                )
            elif isinstance(event, HumanCorrectionApplied):
                index.corrections_applied.append(event)
                # A correction produces a superseding Canonical Observation -- record it exactly as
                # a machine re-reconciliation would be, so "latest per slot" reflects the corrected
                # value (Constitution Article 15; mirrors JournalState's handling).
                index._record_canonical(event.resulting_canonical_observation)
                index.document_ref_by_slot[
                    event.resulting_canonical_observation.semantic_slot_id
                ] = event.document_ref
            elif isinstance(event, HumanCorrectionSubmitted):
                index.corrections_submitted.append(event)
            elif isinstance(event, ProviderObservationAttempted):
                index.provider_attempts.append(event)
            elif isinstance(event, EvidenceRejected):
                index.evidence_rejections.append(event)
        return index

    def _record_canonical(self, canonical: CanonicalObservation) -> None:
        self.canonical_by_id[canonical.canonical_observation_id] = canonical
        current = self.latest_canonical_by_slot.get(canonical.semantic_slot_id)
        if current is None or canonical.reconciliation_sequence >= current.reconciliation_sequence:
            self.latest_canonical_by_slot[canonical.semantic_slot_id] = canonical

    def contributing_providers(self, canonical: CanonicalObservation) -> frozenset[ProviderRef]:
        """The distinct providers whose Observations contributed to a Canonical Observation.

        Resolved via the observation->provider index built from `ObservationCreated`; a
        contributing Observation with no recorded creation event (possible if telemetry is
        incomplete) is skipped rather than guessed at -- the Learning Platform never invents
        provenance the Trust Engine didn't record (ROADMAP.md Guiding Principle 5).
        """
        providers = {
            self.provider_by_observation[ref.observation_id]
            for ref in canonical.contributing_observations
            if ref.observation_id in self.provider_by_observation
        }
        return frozenset(providers)

    def latest_canonicals(self) -> tuple[CanonicalObservation, ...]:
        """The current (latest-per-slot) Canonical Observations -- the state the archive is in
        now, deduplicated across supersession chains."""
        return tuple(self.latest_canonical_by_slot.values())

    def corrections_by_target(self) -> dict[str, list[HumanCorrectionSubmitted]]:
        """Submitted corrections grouped by the Canonical Observation they targeted."""
        grouped: dict[str, list[HumanCorrectionSubmitted]] = defaultdict(list)
        for correction in self.corrections_submitted:
            grouped[correction.target_canonical_observation_id].append(correction)
        return dict(grouped)
