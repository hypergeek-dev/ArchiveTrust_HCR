"""The Replayable Processing Journal (ROADMAP.md S5.10, S8 Milestone 2; Constitution Article 17).

Reconstructs the full knowledge-evolution graph -- Evidence, both Observation Graphs, every
Comparison Decision, the full Confidence Evolution, Canonical Document snapshots, structural
exclusions (Constitution Article 27), and semantic mapping provenance (Article 28) -- from stored
telemetry alone, never by re-running a provider. Applies registered ontology migrations
(`archivetrust.domain.ontology.migrations.MigrationRegistry`) to any Observation recorded under an
older `ontology_version` before it enters the reconstructed state, per ROADMAP.md S5.4.

This module is `application`, not `domain` (S5.8's module boundaries): it orchestrates domain
types but is not itself one -- `JournalState` is a read-only query surface over plain domain
objects, not a new domain concept.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from archivetrust.domain.canonical.observation import CanonicalObservation
from archivetrust.domain.document.canonical_document import CanonicalDocument
from archivetrust.domain.evidence.models import Evidence
from archivetrust.domain.graph.provider_graph import ProviderObservationGraph
from archivetrust.domain.graph.reconciled_graph import ReconciledObservationGraph
from archivetrust.domain.ontology.base import Observation
from archivetrust.domain.ontology.migrations import MigrationRegistry
from archivetrust.domain.telemetry.events import (
    AgreementCalculated,
    AlignmentAttempted,
    CandidateExcluded,
    CandidateExcludedBatch,
    CanonicalDecisionCreated,
    CanonicalDocumentCreated,
    ConfidenceChanged,
    EvidenceCreated,
    EvidenceRejected,
    HumanCorrectionApplied,
    ObservationAligned,
    ObservationCompared,
    ObservationCreated,
    ObservationLeftUnaligned,
    ObservationMapped,
    ProviderObservationAttempted,
    TelemetryEvent,
)
from archivetrust.providers.mapping_registry import CURRENT_MAPPING_TABLES


class UnknownSemanticSlotError(KeyError):
    """Raised when a caller asks for a semantic slot the journal has no CanonicalDecisionCreated
    history for. Distinct from an empty result so "we never reconciled this slot" is never
    silently confused with "we reconciled it and there's nothing here" (the same Article-18
    discipline applied to the canonical layer).
    """


class JournalState:
    """The result of replaying one document's telemetry. Read-only from the caller's perspective
    -- all mutation happens inside `Journal.replay`, before this object is handed back.
    """

    def __init__(self) -> None:
        self._evidence_by_id: dict[str, Evidence] = {}
        self._observations_by_id: dict[str, Observation] = {}
        self._observation_invocation: dict[str, tuple[str, str, str]] = {}
        self._provider_observation_attempts: list[ProviderObservationAttempted] = []
        self._evidence_rejections: list[EvidenceRejected] = []
        self._canonical_observations_by_id: dict[str, CanonicalObservation] = {}
        self._canonical_observation_chain_by_slot: dict[str, list[str]] = defaultdict(list)
        self._canonical_documents_by_id: dict[str, CanonicalDocument] = {}
        self._document_snapshot_chain: dict[str, list[str]] = defaultdict(list)
        self._confidence_events_by_subject: dict[str, list[ConfidenceChanged]] = defaultdict(list)
        self._observation_compared_by_slot: dict[str, list[ObservationCompared]] = defaultdict(list)
        self._agreement_calculated_by_slot: dict[str, list[AgreementCalculated]] = defaultdict(list)
        self._alignment_attempts_by_group: dict[str, AlignmentAttempted] = {}
        self._alignment_state_by_observation: dict[
            str, ObservationAligned | ObservationLeftUnaligned
        ] = {}
        self._excluded_candidates: list[CandidateExcluded] = []
        self._observation_mapped_by_observation_id: dict[str, ObservationMapped] = {}

    # -- Evidence -------------------------------------------------------------------------

    def evidence(self, evidence_id: str) -> Evidence:
        return self._evidence_by_id[evidence_id]

    def all_evidence(self) -> tuple[Evidence, ...]:
        return tuple(self._evidence_by_id.values())

    def provider_observation_attempts(self) -> tuple[ProviderObservationAttempted, ...]:
        """Article 18: what was attempted, whether or not it produced anything."""
        return tuple(self._provider_observation_attempts)

    def evidence_rejections(self) -> tuple[EvidenceRejected, ...]:
        return tuple(self._evidence_rejections)

    # -- Observations and ProviderObservationGraph -----------------------------------------

    def observation(self, observation_id: str) -> Observation:
        return self._observations_by_id[observation_id]

    def all_observations(self) -> tuple[Observation, ...]:
        return tuple(self._observations_by_id.values())

    def provider_observation_graph(
        self, provider_id: str, provider_version: str, invocation_id: str
    ) -> ProviderObservationGraph:
        key = (provider_id, provider_version, invocation_id)
        matching = tuple(
            self._observations_by_id[observation_id]
            for observation_id, observation_key in self._observation_invocation.items()
            if observation_key == key
        )
        return ProviderObservationGraph(
            provider_id=provider_id,
            provider_version=provider_version,
            invocation_id=invocation_id,
            observations=matching,
        )

    # -- Alignment Observability (ROADMAP.md S5.12, Article 24) -----------------------------

    def alignment_attempt(self, comparison_group_id: str) -> AlignmentAttempted | None:
        """The recorded grouping decision for one comparison group -- the durable answer to "why
        were these Observations grouped together?", reconstructed from telemetry alone. `None` if
        no alignment was recorded for that group (distinct from an empty grouping).
        """
        return self._alignment_attempts_by_group.get(comparison_group_id)

    def alignment_attempts(self) -> tuple[AlignmentAttempted, ...]:
        return tuple(self._alignment_attempts_by_group.values())

    def alignment_state(
        self, observation_id: str
    ) -> ObservationAligned | ObservationLeftUnaligned | None:
        """One Observation's recorded terminal alignment state (Aligned/Unaligned). `None` only if
        the Observation was never put through alignment -- itself a distinguishable fact, never
        conflated with "aligned with nothing" (which is a recorded `ObservationLeftUnaligned`).
        """
        return self._alignment_state_by_observation.get(observation_id)

    def unaligned_observations(self) -> tuple[str, ...]:
        """Every Observation explicitly recorded as Unaligned -- proof that no Observation silently
        disappeared from the comparison (Constitution Article 18, Article 24).
        """
        return tuple(
            observation_id
            for observation_id, state in self._alignment_state_by_observation.items()
            if isinstance(state, ObservationLeftUnaligned)
        )

    def excluded_candidate_pairs(self) -> tuple[CandidateExcluded, ...]:
        """Every structurally-excluded candidate pair recorded for this document (Constitution
        Article 27) -- the exclusion-side symmetry `alignment_attempts()` alone does not provide.
        """
        return tuple(self._excluded_candidates)

    # -- Semantic Lineage (Constitution Article 28) -----------------------------------------

    def mapping_used_for(self, observation_id: str) -> ObservationMapped | None:
        """The recorded `ObservationMapped` event for one Observation, if any -- `None` for
        historical data recorded before this event kind was wired (`application/pipeline.py`),
        never inferred (Article 18's discipline, applied here).
        """
        return self._observation_mapped_by_observation_id.get(observation_id)

    def mapping_table_is_stale(self, observation_id: str) -> bool | None:
        """Whether this Observation was mapped under a `MappingTable` version that has since been
        superseded by a newer one currently registered for the same provider
        (`providers/mapping_registry.py`). `None` -- never guessed -- when this cannot be
        established: no recorded mapping, no recorded version, or the provider isn't in the
        current registry (a historical/removed provider).
        """
        mapping = self.mapping_used_for(observation_id)
        if mapping is None or mapping.mapping_table_version is None:
            return None
        observation = self._observations_by_id.get(observation_id)
        if observation is None:
            return None
        current_table = CURRENT_MAPPING_TABLES.get(observation.provider_id)
        if current_table is None:
            return None
        return mapping.mapping_table_version < current_table.table_version

    # -- Canonical Observations and Confidence Evolution -----------------------------------

    def canonical_observation(self, canonical_observation_id: str) -> CanonicalObservation:
        return self._canonical_observations_by_id[canonical_observation_id]

    def canonical_observation_history(self, semantic_slot_id: str) -> tuple[CanonicalObservation, ...]:
        """Every version recorded for this semantic slot, oldest to newest -- this *is* Confidence
        Evolution's concrete mechanism at the Canonical Observation layer
        (MILESTONE1_DOMAIN_MODEL.md S1.4).
        """
        if semantic_slot_id not in self._canonical_observation_chain_by_slot:
            raise UnknownSemanticSlotError(semantic_slot_id)
        ids = self._canonical_observation_chain_by_slot[semantic_slot_id]
        versions = [self._canonical_observations_by_id[i] for i in ids]
        return tuple(sorted(versions, key=lambda co: co.reconciliation_sequence))

    def canonical_observation_as_of(
        self, semantic_slot_id: str, reconciliation_sequence: int
    ) -> CanonicalObservation | None:
        """"What did we believe about this slot at time T" (ROADMAP.md S5.10) -- the latest
        version whose reconciliation_sequence does not exceed the requested point. None if the
        slot did not yet exist at that point.
        """
        candidates = [
            co
            for co in self.canonical_observation_history(semantic_slot_id)
            if co.reconciliation_sequence <= reconciliation_sequence
        ]
        return candidates[-1] if candidates else None

    def known_semantic_slots(self) -> tuple[str, ...]:
        return tuple(self._canonical_observation_chain_by_slot)

    def reconciled_observation_graph_as_of(
        self, reconciliation_sequence: int
    ) -> ReconciledObservationGraph:
        """Reconstructs the ReconciledObservationGraph as it stood at a given reconciliation
        pass, by taking each known slot's latest version not exceeding that sequence.

        Known limitation (documented in IMPLEMENTATION_STATUS.md): this assumes a reconciliation
        pass updates its member Canonical Observations' graph-position fields consistently with
        each other, which is true for any single Comparison Engine pass (Milestone 4) but is not
        independently re-verified here -- if two slots' histories were produced by unrelated,
        inconsistent processes referencing each other's stale ids, graph construction below will
        raise rather than silently produce an inconsistent graph (ReconciledObservationGraph's own
        dangling-reference validator), which is the correct failure mode either way.
        """
        nodes = tuple(
            co
            for co in (
                self.canonical_observation_as_of(slot, reconciliation_sequence)
                for slot in self.known_semantic_slots()
            )
            if co is not None
        )
        return ReconciledObservationGraph(
            reconciliation_sequence=reconciliation_sequence, canonical_observations=nodes
        )

    def confidence_evolution(self, subject_id: str) -> tuple[ConfidenceChanged, ...]:
        return tuple(self._confidence_events_by_subject.get(subject_id, ()))

    def comparison_decisions(self, semantic_slot_id: str) -> tuple[ObservationCompared, ...]:
        """Every clustering decision recorded for this semantic slot -- what was compared and
        why (`clustering_basis`), per ROADMAP.md S5.10's "every Comparison Decision" replay
        requirement.
        """
        return tuple(self._observation_compared_by_slot.get(semantic_slot_id, ()))

    def agreement_calculations(self, semantic_slot_id: str) -> tuple[AgreementCalculated, ...]:
        """Every agreement/Comparison-Confidence calculation recorded for this semantic slot --
        the other half of "every Comparison Decision" (S5.10): what was compared
        (`comparison_decisions`) and what was concluded about agreement (this).
        """
        return tuple(self._agreement_calculated_by_slot.get(semantic_slot_id, ()))

    # -- Canonical Document -------------------------------------------------------------------

    def canonical_document(self, document_snapshot_id: str) -> CanonicalDocument:
        return self._canonical_documents_by_id[document_snapshot_id]

    def canonical_document_history(self, logical_document_id: str) -> tuple[CanonicalDocument, ...]:
        ids = self._document_snapshot_chain.get(logical_document_id, [])
        snapshots = [self._canonical_documents_by_id[i] for i in ids]
        return tuple(sorted(snapshots, key=lambda doc: doc.document_version))

    def latest_canonical_document(self, logical_document_id: str) -> CanonicalDocument:
        history = self.canonical_document_history(logical_document_id)
        if not history:
            raise KeyError(logical_document_id)
        return history[-1]

    # -- internal, used only by Journal.replay ----------------------------------------------

    def _record_evidence(self, evidence: Evidence) -> None:
        self._evidence_by_id[evidence.evidence_id] = evidence

    def _record_observation(self, observation: Observation, invocation_id: str) -> None:
        self._observations_by_id[observation.observation_id] = observation
        self._observation_invocation[observation.observation_id] = (
            observation.provider_id,
            observation.provider_version,
            invocation_id,
        )

    def _record_canonical_observation(self, canonical_observation: CanonicalObservation) -> None:
        self._canonical_observations_by_id[canonical_observation.canonical_observation_id] = (
            canonical_observation
        )
        self._canonical_observation_chain_by_slot[canonical_observation.semantic_slot_id].append(
            canonical_observation.canonical_observation_id
        )

    def _record_canonical_document(self, canonical_document: CanonicalDocument) -> None:
        self._canonical_documents_by_id[canonical_document.document_snapshot_id] = canonical_document
        self._document_snapshot_chain[canonical_document.logical_document_id].append(
            canonical_document.document_snapshot_id
        )

    def _record_observation_compared(self, event: ObservationCompared) -> None:
        self._observation_compared_by_slot[event.semantic_slot_id].append(event)

    def _record_agreement_calculated(self, event: AgreementCalculated) -> None:
        self._agreement_calculated_by_slot[event.semantic_slot_id].append(event)


class Journal:
    """Replays a stream of `TelemetryEvent`s into a `JournalState`. Stateless across calls --
    each `replay()` starts a fresh `JournalState`, matching the requirement that replay must be
    reproducible purely from stored events.
    """

    def __init__(self, migration_registry: MigrationRegistry | None = None) -> None:
        self._migration_registry = migration_registry or MigrationRegistry()

    def replay(self, events: Iterable[TelemetryEvent]) -> JournalState:
        state = JournalState()
        for event in events:
            self._apply(state, event)
        return state

    def _apply(self, state: JournalState, event: TelemetryEvent) -> None:
        if isinstance(event, ProviderObservationAttempted):
            state._provider_observation_attempts.append(event)
        elif isinstance(event, EvidenceRejected):
            state._evidence_rejections.append(event)
        elif isinstance(event, EvidenceCreated):
            state._record_evidence(event.evidence)
        elif isinstance(event, ObservationCreated):
            observation = self._migration_registry.migrate(event.observation)
            state._record_observation(observation, event.invocation_id)
        elif isinstance(event, CanonicalDecisionCreated):
            state._record_canonical_observation(event.canonical_observation)
        elif isinstance(event, HumanCorrectionApplied):
            # A human correction produces a new superseding CanonicalObservation version, exactly
            # like a machine re-reconciliation -- Constitution Article 15 draws no distinction
            # between the two at the storage layer, only at the *reason* for supersession.
            state._record_canonical_observation(event.resulting_canonical_observation)
        elif isinstance(event, CanonicalDocumentCreated):
            state._record_canonical_document(event.canonical_document)
        elif isinstance(event, ConfidenceChanged):
            state._confidence_events_by_subject[event.subject_id].append(event)
        elif isinstance(event, ObservationCompared):
            state._record_observation_compared(event)
        elif isinstance(event, AgreementCalculated):
            state._record_agreement_calculated(event)
        elif isinstance(event, AlignmentAttempted):
            # The grouping hypothesis, made replayable (S5.12): keyed by comparison_group_id so
            # "why were these grouped?" is answerable months later without re-running providers.
            state._alignment_attempts_by_group[event.comparison_group_id] = event
        elif isinstance(event, (ObservationAligned, ObservationLeftUnaligned)):
            # Every Observation's terminal alignment state, so none can silently disappear
            # (Article 18, Article 24).
            state._alignment_state_by_observation[event.observation_id] = event
        elif isinstance(event, CandidateExcluded):
            # Constitution Article 27: the exclusion-side symmetry AlignmentAttempted alone
            # doesn't provide -- see excluded_candidate_pairs().
            state._excluded_candidates.append(event)
        elif isinstance(event, CandidateExcludedBatch):
            # F3 encoding repair: compact storage, same replay query surface.
            for exclusion in event.exclusions:
                state._excluded_candidates.append(
                    CandidateExcluded(
                        event_id=event.event_id,
                        document_ref=event.document_ref,
                        schema_version=event.schema_version,
                        recorded_at=event.recorded_at,
                        reconciliation_policy_version=event.reconciliation_policy_version,
                        capability_matrix_version=event.capability_matrix_version,
                        confidence_policy_version=event.confidence_policy_version,
                        feedback_policy_version=event.feedback_policy_version,
                        alignment_algorithm_version=event.alignment_algorithm_version,
                        candidate_observation_id=exclusion.candidate_observation_id,
                        compared_against_observation_id=exclusion.compared_against_observation_id,
                        excluding_mechanism=exclusion.excluding_mechanism,
                        basis_code=exclusion.basis_code,
                        structural=exclusion.structural,
                    )
                )
        elif isinstance(event, ObservationMapped):
            # Constitution Article 28: which MappingTableEntry produced this Observation, so
            # mapping_table_is_stale() can answer "was this processed under a mapping-table
            # version that has since been superseded" from telemetry alone.
            state._observation_mapped_by_observation_id[event.observation_id] = event
        # ObservationAccepted, ObservationRejected, ObservationMerged, KnowledgeMerged,
        # KnowledgeDiscarded, HumanCorrectionSubmitted, DatasetCandidateCreated: these are
        # legitimate telemetry with no further state to reconstruct beyond what's captured above --
        # CanonicalDecisionCreated already carries the accepted clustering_basis/
        # reconciliation_basis, and ConfidenceChanged already carries Confidence Evolution.
        # Replaying them is a no-op for JournalState today; nothing here silently drops
        # information, since the events themselves remain available to any
        # caller via the TelemetrySink they were read from.
