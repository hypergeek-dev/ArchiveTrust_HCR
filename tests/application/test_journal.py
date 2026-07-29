"""Full-replay tests (ROADMAP.md S5.10, S8 Milestone 2): the journal must reconstruct Evidence,
both Observation Graphs, Comparison Decisions, Confidence Evolution, and the current Canonical
Document snapshot purely from stored telemetry -- never by re-running a provider.
"""

from __future__ import annotations

from archivetrust.application.journal import Journal, UnknownSemanticSlotError
from archivetrust.domain.canonical.observation import CanonicalObservation
from archivetrust.domain.confidence.models import ComparisonClassification, ComparisonConfidence
from archivetrust.domain.document.canonical_document import CanonicalDocument
from archivetrust.domain.evidence.models import Evidence, ProcessingStage
from archivetrust.domain.graph.reconciled_graph import ReconciledObservationGraph
from archivetrust.domain.ontology.base import Observation
from archivetrust.domain.ontology.migrations import MigrationRegistry
from archivetrust.domain.ontology.payloads import HeadingPayload, ParagraphPayload
from archivetrust.domain.ontology.types import ObservationType
from archivetrust.domain.telemetry.events import (
    AgreementCalculated,
    CanonicalDecisionCreated,
    CanonicalDocumentCreated,
    ConfidenceChanged,
    ConfidenceLevel,
    EvidenceCreated,
    HumanCorrectionApplied,
    ObservationCompared,
    ObservationCreated,
    ProviderObservationAttempted,
)
from archivetrust.infrastructure.storage.telemetry_sink import InMemoryTelemetrySink

DOCUMENT_REF = "archive-object-1"


def _single_source() -> ComparisonConfidence:
    return ComparisonConfidence(
        classification=ComparisonClassification.UNCORROBORATED_SINGLE_SOURCE, magnitude=None, basis="n/a"
    )


def _build_full_history() -> tuple[InMemoryTelemetrySink, Observation, CanonicalObservation, CanonicalDocument]:
    sink = InMemoryTelemetrySink()

    sink.append(
        ProviderObservationAttempted(
            event_id="event_1",
            document_ref=DOCUMENT_REF,
            provider_id="docling",
            provider_version="1.0",
            invocation_id="invocation_1",
        )
    )

    evidence = Evidence.create(
        provider="docling",
        provider_version="1.0",
        raw_output="Chapter 1: Origins",
        processing_stage=ProcessingStage.LAYOUT_ANALYSIS,
        provider_confidence=0.9,
    )
    sink.append(
        EvidenceCreated(
            event_id="event_2", document_ref=DOCUMENT_REF, invocation_id="invocation_1", evidence=evidence
        )
    )

    observation = Observation.from_evidence(
        provider_id="docling",
        provider_version="1.0",
        payload=HeadingPayload(text="Chapter 1: Origins", level=1),
        evidence=(evidence,),
    )
    sink.append(
        ObservationCreated(
            event_id="event_3",
            document_ref=DOCUMENT_REF,
            invocation_id="invocation_1",
            observation=observation,
        )
    )

    canonical = CanonicalObservation.reconcile(
        semantic_slot_id="slot-heading-1",
        payload=observation.payload,
        contributing_observations=(observation,),
        comparison_confidence=_single_source(),
        clustering_basis="sole observation in slot",
        reconciliation_basis="single-source acceptance",
        reconciliation_sequence=0,
    )
    sink.append(
        CanonicalDecisionCreated(
            event_id="event_4", document_ref=DOCUMENT_REF, canonical_observation=canonical
        )
    )
    sink.append(
        ConfidenceChanged(
            event_id="event_5",
            document_ref=DOCUMENT_REF,
            subject_id=canonical.canonical_observation_id,
            level=ConfidenceLevel.COMPARISON,
            previous_value=None,
            new_value=None,
            reason="single-source: not applicable",
        )
    )

    graph = ReconciledObservationGraph(reconciliation_sequence=0, canonical_observations=(canonical,))
    document = CanonicalDocument.assemble(
        reconciled_graph=graph, archive_object_ref=DOCUMENT_REF, reassembly_trigger="initial_assembly"
    )
    sink.append(
        CanonicalDocumentCreated(event_id="event_6", document_ref=DOCUMENT_REF, canonical_document=document)
    )

    return sink, observation, canonical, document


def test_replay_reconstructs_evidence():
    sink, _observation, _canonical, _document = _build_full_history()
    state = Journal().replay(sink.events_for_document(DOCUMENT_REF))
    assert len(state.all_evidence()) == 1
    assert state.all_evidence()[0].raw_output == "Chapter 1: Origins"


def test_replay_reconstructs_provider_observation_attempts():
    sink, _observation, _canonical, _document = _build_full_history()
    state = Journal().replay(sink.events_for_document(DOCUMENT_REF))
    attempts = state.provider_observation_attempts()
    assert len(attempts) == 1
    assert attempts[0].provider_id == "docling"


def test_replay_reconstructs_provider_observation_graph():
    sink, observation, _canonical, _document = _build_full_history()
    state = Journal().replay(sink.events_for_document(DOCUMENT_REF))
    graph = state.provider_observation_graph("docling", "1.0", "invocation_1")
    assert graph.observations == (observation,)


def test_replay_reconstructs_canonical_observation_history():
    sink, _observation, canonical, _document = _build_full_history()
    state = Journal().replay(sink.events_for_document(DOCUMENT_REF))
    history = state.canonical_observation_history("slot-heading-1")
    assert history == (canonical,)


def test_replay_raises_for_unknown_semantic_slot():
    sink, _observation, _canonical, _document = _build_full_history()
    state = Journal().replay(sink.events_for_document(DOCUMENT_REF))
    try:
        state.canonical_observation_history("slot-does-not-exist")
        raised = False
    except UnknownSemanticSlotError:
        raised = True
    assert raised


def test_replay_reconstructs_reconciled_observation_graph():
    sink, _observation, canonical, _document = _build_full_history()
    state = Journal().replay(sink.events_for_document(DOCUMENT_REF))
    graph = state.reconciled_observation_graph_as_of(0)
    assert graph.canonical_observations == (canonical,)


def test_replay_reconstructs_confidence_evolution():
    sink, _observation, canonical, _document = _build_full_history()
    state = Journal().replay(sink.events_for_document(DOCUMENT_REF))
    evolution = state.confidence_evolution(canonical.canonical_observation_id)
    assert len(evolution) == 1
    assert evolution[0].level == ConfidenceLevel.COMPARISON


def test_replay_reconstructs_current_canonical_document_snapshot():
    sink, _observation, _canonical, document = _build_full_history()
    state = Journal().replay(sink.events_for_document(DOCUMENT_REF))
    latest = state.latest_canonical_document(document.logical_document_id)
    assert latest == document


def test_replay_reconstructs_supersession_chain_after_human_correction():
    sink, observation, canonical, document = _build_full_history()

    revised_payload = HeadingPayload(text="Chapter One: Origins", level=1)
    revised = canonical.supersede(
        payload=revised_payload,
        contributing_observations=(observation,),
        comparison_confidence=_single_source(),
        clustering_basis="sole observation in slot",
        reconciliation_basis="human correction applied",
        human_correction_ref="correction_1",
    )
    sink.append(
        HumanCorrectionApplied(
            event_id="event_7",
            document_ref=DOCUMENT_REF,
            correction_id="correction_1",
            resulting_canonical_observation=revised,
        )
    )

    state = Journal().replay(sink.events_for_document(DOCUMENT_REF))
    history = state.canonical_observation_history("slot-heading-1")
    assert history == (canonical, revised)
    assert state.canonical_observation_as_of("slot-heading-1", 0) == canonical
    assert state.canonical_observation_as_of("slot-heading-1", 1) == revised


def test_replay_reconstructs_comparison_decisions():
    sink, _observation, canonical, _document = _build_full_history()
    sink.append(
        ObservationCompared(
            event_id="event_compared",
            document_ref=DOCUMENT_REF,
            semantic_slot_id="slot-heading-1",
            compared_observation_ids=("observation_a",),
            clustering_basis="sole observation in slot",
        )
    )
    sink.append(
        AgreementCalculated(
            event_id="event_agreement",
            document_ref=DOCUMENT_REF,
            semantic_slot_id="slot-heading-1",
            comparison_confidence=_single_source(),
        )
    )

    state = Journal().replay(sink.events_for_document(DOCUMENT_REF))
    decisions = state.comparison_decisions("slot-heading-1")
    agreements = state.agreement_calculations("slot-heading-1")
    assert len(decisions) == 1
    assert decisions[0].clustering_basis == "sole observation in slot"
    assert len(agreements) == 1
    assert agreements[0].comparison_confidence.classification == ComparisonClassification.UNCORROBORATED_SINGLE_SOURCE


def test_replay_is_scoped_to_one_document():
    sink, _observation, _canonical, _document = _build_full_history()
    other_evidence = Evidence.create(
        provider="tesseract",
        provider_version="5.3",
        raw_output="unrelated document text",
        processing_stage=ProcessingStage.OCR,
    )
    sink.append(
        EvidenceCreated(
            event_id="event_other",
            document_ref="archive-object-2",
            invocation_id="invocation_2",
            evidence=other_evidence,
        )
    )

    state = Journal().replay(sink.events_for_document(DOCUMENT_REF))
    assert other_evidence not in state.all_evidence()


def test_replay_applies_registered_ontology_migration():
    evidence = Evidence.create(
        provider="tesseract", provider_version="5.3", raw_output="body", processing_stage=ProcessingStage.OCR
    )
    old_observation = Observation.from_evidence(
        provider_id="tesseract",
        provider_version="5.3",
        payload=ParagraphPayload(text="body"),
        evidence=(evidence,),
        ontology_version=0,
    )
    sink = InMemoryTelemetrySink()
    sink.append(
        EvidenceCreated(
            event_id="event_1", document_ref=DOCUMENT_REF, invocation_id="invocation_1", evidence=evidence
        )
    )
    sink.append(
        ObservationCreated(
            event_id="event_2",
            document_ref=DOCUMENT_REF,
            invocation_id="invocation_1",
            observation=old_observation,
        )
    )

    registry = MigrationRegistry()
    registry.register(
        ObservationType.PARAGRAPH, 0, lambda obs: obs.model_copy(update={"ontology_version": 1})
    )

    state = Journal(migration_registry=registry).replay(sink.events_for_document(DOCUMENT_REF))
    migrated = state.observation(old_observation.observation_id)
    assert migrated.ontology_version == 1
