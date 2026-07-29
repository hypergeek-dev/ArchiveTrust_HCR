"""Authoritative current canonical-state projection.

This module is the single domain-owned interpretation of *current* document truth.  It consumes
append-only telemetry facts and projects them without mutating history.  UI, export, replay,
evaluation, and service facades must consume this projection instead of independently choosing a
latest canonical observation or document snapshot.

The projection deliberately distinguishes the latest recorded ``CanonicalDocument`` snapshot
from the current observation graph.  Historical workspaces can contain a correction after their
latest snapshot; ``requires_reassembly`` makes that inconsistency explicit and prevents export
from silently presenting the stale snapshot as current.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict

from archivetrust.domain.canonical.observation import CanonicalObservation
from archivetrust.domain.confidence.models import ComparisonClassification
from archivetrust.domain.document.canonical_document import CanonicalDocument
from archivetrust.domain.evidence.models import Evidence
from archivetrust.domain.ontology.base import Observation
from archivetrust.domain.telemetry.events import (
    CanonicalDecisionCreated,
    CanonicalDocumentCreated,
    EvidenceCreated,
    HumanCorrectionApplied,
    ObservationCreated,
    ProvenanceContextEstablished,
    ReviewOutcomeRecorded,
    ReviewPacketClosed,
    ReviewPacketDispatched,
    ReviewPacketOpened,
    TelemetryEvent,
)


class CurrentPolicyVersions(BaseModel):
    model_config = ConfigDict(frozen=True)

    ontology_version: int | None = None
    alignment_version: int | None = None
    reconciliation_version: int | None = None
    capability_matrix_version: int | None = None
    confidence_version: int | None = None
    feedback_version: int | None = None
    configuration_hash: str | None = None
    provenance_context_id: str | None = None


class CurrentCanonicalSlot(BaseModel):
    """Current truth and complete lineage for one semantic slot."""

    model_config = ConfigDict(frozen=True)

    semantic_slot_id: str
    current: CanonicalObservation
    history: tuple[CanonicalObservation, ...]
    contributing_observations: tuple[Observation, ...]
    evidence: tuple[Evidence, ...]
    review_outcomes: tuple[ReviewOutcomeRecorded, ...]
    review_status: str
    packet_id: str | None = None

    @property
    def classification(self) -> ComparisonClassification:
        return self.current.comparison_confidence.classification

    @property
    def contested(self) -> bool:
        return self.classification is ComparisonClassification.CONTESTED

    @property
    def corrected(self) -> bool:
        return self.current.human_correction_ref is not None

    @property
    def export_eligible(self) -> bool:
        return True

    @property
    def evaluation_eligible(self) -> bool:
        return True


class CurrentDocumentState(BaseModel):
    """The one authoritative current-state answer for a document."""

    model_config = ConfigDict(frozen=True)

    document_ref: str
    archive_object_ref: str
    latest_canonical_document: CanonicalDocument | None
    canonical_document_history: tuple[CanonicalDocument, ...]
    slots: tuple[CurrentCanonicalSlot, ...]
    review_outcomes: tuple[ReviewOutcomeRecorded, ...]
    effective_contained_observations: tuple[str, ...]
    requires_reassembly: bool
    export_eligible: bool
    evaluation_eligible: bool
    policy_versions: CurrentPolicyVersions
    integrity_status: str
    latest_snapshot_event_id: str | None = None
    latest_snapshot_recorded_at: str | None = None
    latest_snapshot_schema_version: int | None = None

    def slot(self, semantic_slot_id: str) -> CurrentCanonicalSlot:
        for item in self.slots:
            if item.semantic_slot_id == semantic_slot_id:
                return item
        raise KeyError(semantic_slot_id)

    def canonical_observation(self, canonical_observation_id: str) -> CanonicalObservation:
        for slot in self.slots:
            for observation in slot.history:
                if observation.canonical_observation_id == canonical_observation_id:
                    return observation
        raise KeyError(canonical_observation_id)

    @property
    def current_canonical_observations(self) -> tuple[CanonicalObservation, ...]:
        return tuple(slot.current for slot in self.slots)

    @property
    def unresolved_count(self) -> int:
        return sum(slot.contested for slot in self.slots)


def project_current_document(
    events: Iterable[TelemetryEvent], *, integrity_status: str = "not_checked"
) -> CurrentDocumentState:
    """Project one document's append-only telemetry into its authoritative current state."""

    event_list = tuple(events)
    document_refs = {event.document_ref for event in event_list}
    if not document_refs:
        raise ValueError("current-state projection requires at least one telemetry event")
    if len(document_refs) != 1:
        raise ValueError("current-state projection accepts events for exactly one document")
    document_ref = next(iter(document_refs))

    evidence_by_id: dict[str, Evidence] = {}
    observations_by_id: dict[str, Observation] = {}
    histories: dict[str, list[CanonicalObservation]] = defaultdict(list)
    slot_order: list[str] = []
    outcomes: dict[str, list[ReviewOutcomeRecorded]] = defaultdict(list)
    all_outcomes: list[ReviewOutcomeRecorded] = []
    dispatches: dict[str, ReviewPacketDispatched] = {}
    closures: dict[str, ReviewPacketClosed] = {}
    openings: dict[str, ReviewPacketOpened] = {}
    snapshots: list[CanonicalDocument] = []
    latest_snapshot_event: CanonicalDocumentCreated | None = None
    latest_snapshot_index = -1
    latest_canonical_index = -1
    latest_context: ProvenanceContextEstablished | None = None

    for index, event in enumerate(event_list):
        if isinstance(event, EvidenceCreated):
            evidence_by_id[event.evidence.evidence_id] = event.evidence
        elif isinstance(event, ObservationCreated):
            observations_by_id[event.observation.observation_id] = event.observation
        elif isinstance(event, (CanonicalDecisionCreated, HumanCorrectionApplied)):
            canonical = (
                event.canonical_observation
                if isinstance(event, CanonicalDecisionCreated)
                else event.resulting_canonical_observation
            )
            if canonical.semantic_slot_id not in histories:
                slot_order.append(canonical.semantic_slot_id)
            histories[canonical.semantic_slot_id].append(canonical)
            latest_canonical_index = index
        elif isinstance(event, CanonicalDocumentCreated):
            snapshots.append(event.canonical_document)
            latest_snapshot_event = event
            latest_snapshot_index = index
        elif isinstance(event, ReviewOutcomeRecorded):
            outcomes[event.semantic_slot_id].append(event)
            all_outcomes.append(event)
        elif isinstance(event, ReviewPacketDispatched):
            dispatches[event.semantic_slot_id] = event
        elif isinstance(event, ReviewPacketClosed):
            closures[event.packet_id] = event
        elif isinstance(event, ReviewPacketOpened):
            openings[event.packet_id] = event
        elif isinstance(event, ProvenanceContextEstablished):
            latest_context = event

    current_by_slot = {slot: versions[-1] for slot, versions in histories.items()}
    canonical_id_to_slot = {
        canonical.canonical_observation_id: slot
        for slot, versions in histories.items()
        for canonical in versions
    }
    latest_snapshot = snapshots[-1] if snapshots else None
    effective_ids: tuple[str, ...] = ()
    if latest_snapshot is not None:
        effective_ids = tuple(
            current_by_slot[canonical_id_to_slot[canonical_id]].canonical_observation_id
            if canonical_id in canonical_id_to_slot
            else canonical_id
            for canonical_id in latest_snapshot.contained_observations
        )

    slot_states: list[CurrentCanonicalSlot] = []
    for slot_id in slot_order:
        current = current_by_slot[slot_id]
        contributing = tuple(
            observations_by_id[ref.observation_id]
            for ref in current.contributing_observations
            if ref.observation_id in observations_by_id
        )
        slot_evidence: list[Evidence] = []
        seen_evidence: set[str] = set()
        for observation in contributing:
            for evidence_id in observation.evidence_ids:
                evidence = evidence_by_id.get(evidence_id)
                if evidence is not None and evidence_id not in seen_evidence:
                    slot_evidence.append(evidence)
                    seen_evidence.add(evidence_id)

        dispatch = dispatches.get(slot_id)
        closure = closures.get(dispatch.packet_id) if dispatch is not None else None
        slot_outcomes = tuple(outcomes.get(slot_id, ()))
        if closure is not None:
            review_status = f"closed:{closure.closure_kind.value}"
        elif dispatch is not None and dispatch.packet_id in openings:
            review_status = "in_progress"
        elif dispatch is not None:
            review_status = "dispatched"
        elif slot_outcomes:
            review_status = "legacy_incomplete"
        elif current.comparison_confidence.classification is ComparisonClassification.CONTESTED:
            review_status = "pending"
        else:
            review_status = "not_required"

        slot_states.append(
            CurrentCanonicalSlot(
                semantic_slot_id=slot_id,
                current=current,
                history=tuple(histories[slot_id]),
                contributing_observations=contributing,
                evidence=tuple(slot_evidence),
                review_outcomes=slot_outcomes,
                review_status=review_status,
                packet_id=dispatch.packet_id if dispatch is not None else None,
            )
        )

    requires_reassembly = bool(
        latest_snapshot is not None and latest_canonical_index > latest_snapshot_index
    )
    archive_object_ref = (
        latest_snapshot.archive_object_ref if latest_snapshot is not None else document_ref
    )
    versions = _policy_versions(event_list, latest_context, slot_states)
    return CurrentDocumentState(
        document_ref=document_ref,
        archive_object_ref=archive_object_ref,
        latest_canonical_document=latest_snapshot,
        canonical_document_history=tuple(snapshots),
        slots=tuple(slot_states),
        review_outcomes=tuple(all_outcomes),
        effective_contained_observations=effective_ids,
        requires_reassembly=requires_reassembly,
        export_eligible=latest_snapshot is not None and not requires_reassembly,
        evaluation_eligible=bool(slot_states),
        policy_versions=versions,
        integrity_status=integrity_status,
        latest_snapshot_event_id=(
            latest_snapshot_event.event_id if latest_snapshot_event is not None else None
        ),
        latest_snapshot_recorded_at=(
            latest_snapshot_event.recorded_at if latest_snapshot_event is not None else None
        ),
        latest_snapshot_schema_version=(
            latest_snapshot_event.schema_version if latest_snapshot_event is not None else None
        ),
    )


def _policy_versions(
    events: tuple[TelemetryEvent, ...],
    context: ProvenanceContextEstablished | None,
    slots: list[CurrentCanonicalSlot],
) -> CurrentPolicyVersions:
    ontology = max((slot.current.ontology_version for slot in slots), default=None)
    def latest_version(field: str) -> int | None:
        return next(
            (
                value
                for event in reversed(events)
                if (value := getattr(event, field, None)) is not None
            ),
            None,
        )

    return CurrentPolicyVersions(
        ontology_version=ontology or (context.ontology_version if context is not None else None),
        alignment_version=(
            latest_version("alignment_algorithm_version")
            or (context.alignment_algorithm_version if context is not None else None)
        ),
        reconciliation_version=(
            latest_version("reconciliation_policy_version")
            or (context.reconciliation_policy_version if context is not None else None)
        ),
        capability_matrix_version=(
            latest_version("capability_matrix_version")
            or (context.capability_matrix_version if context is not None else None)
        ),
        confidence_version=(
            latest_version("confidence_policy_version")
            or (context.confidence_policy_version if context is not None else None)
        ),
        feedback_version=(
            latest_version("feedback_policy_version")
            or (context.feedback_policy_version if context is not None else None)
        ),
        configuration_hash=context.configuration_hash if context is not None else None,
        provenance_context_id=context.context_id if context is not None else None,
    )
