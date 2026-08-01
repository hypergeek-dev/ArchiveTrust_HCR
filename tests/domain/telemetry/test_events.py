from __future__ import annotations

import pytest
from pydantic import ValidationError

from archivetrust.domain.evidence.models import Evidence, ProcessingStage
from archivetrust.domain.telemetry.events import (
    EVENT_TYPE_BY_KIND,
    EvidenceCreated,
    ProviderObservationAttempted,
    TelemetryEventKind,
    parse_event,
)


def test_all_canonical_event_kinds_are_registered():
    # ROADMAP.md S12's canonical set is exhaustive by design. 18 original event kinds + 3 added by
    # the Alignment Observability refinement (Revision 4: AlignmentAttempted, ObservationAligned,
    # ObservationLeftUnaligned) + 3 added by the Telemetry Architecture Standard (Constitution
    # Article 32: ProvenanceContextEstablished; Article 27: CandidateExcluded; Article 30:
    # ReviewOutcomeRecorded) = 24, plus F3's CandidateExcludedBatch compact encoding = 25, plus
    # F4's created/dispatched/opened/closed lifecycle events = 29, plus docs/htr-migration-plan.md
    # Stage 3's 5 HTR event kinds (SegmentationRunCompleted, MethodRunCompleted,
    # ReviewSubmissionRecorded, AdjudicationRecorded, CanonicalResultCreated) = 34, plus the HTR
    # research-persistence kinds (docs/architecture/htr-event-model.md §3: 30 enumerated there, plus
    # the 4 additions its §3 list omitted -- CollectionCreated, ReviewedResultRecorded,
    # MetricDefinitionRegistered, GroundTruthTextRecorded -- each justified in its own enum-member
    # docstring and in docs/architecture/htr-telemetry.md §6) = 68, plus the research-question
    # feedback loop's 2 (ResearchQuestionRaised, ExperimentDraftedFromQuestion -- the edge *back* from
    # a knowledge record to a new experiment, which §3's one-way layer 10-12 list does not cover; each
    # justified in its own enum-member docstring) = 70, plus the RGB-normalization preprocessing
    # stage's 4 (ImageNormalizationStarted, ImageNormalizationCompleted, ImageNormalizationFailed,
    # DerivedImageArtifactCreated -- Transkribus Swedish Lion I page-level preprocessing provenance)
    # = 74, plus the active-method transition / Loghi integration's 9
    # (docs/loghi-integration-audit.md: MethodResearchStatusChanged, LoghiEnvironmentValidated,
    # LoghiPipelineStarted, LoghiStageStarted, LoghiStageCompleted, LoghiStageFailed,
    # LoghiPageXmlGenerated, DomainRelationshipRecorded, CrossDomainComparisonCreated -- each
    # justified in its own enum-member docstring and in docs/architecture/htr-telemetry.md §6) = 83,
    # plus the Swedish Loghi fine-tuning pilot's 3 (TrainingSessionStarted,
    # TrainingSessionCheckpointed, TrainingSessionCompleted -- a training-session lifecycle is
    # materially different from an inference run's, so it is not shoehorned into the existing
    # HTR-run events; each justified in its own enum-member docstring) = 86, plus the read-only
    # training dashboard's 2 (TrainingSessionFailed -- a real process/container failure, distinct
    # from any clean stop_reason `TrainingSessionCompleted` covers; RunWarningRecorded -- a
    # `run_health.py` finding made durable evidence rather than only a live-computed value) = 88.
    assert len(TelemetryEventKind) == 88
    assert set(EVENT_TYPE_BY_KIND) == set(TelemetryEventKind)


def test_event_is_immutable():
    event = ProviderObservationAttempted(
        event_id="event_1",
        document_ref="doc-1",
        provider_id="docling",
        provider_version="1.0",
        invocation_id="invocation_1",
    )
    with pytest.raises(ValidationError):
        event.provider_id = "someone-else"  # type: ignore[misc]


def test_round_trip_serialization_of_a_nested_domain_object():
    evidence = Evidence.create(
        provider="docling", provider_version="1.0", raw_output="x", processing_stage=ProcessingStage.OCR
    )
    event = EvidenceCreated(
        event_id="event_1", document_ref="doc-1", invocation_id="invocation_1", evidence=evidence
    )
    restored = EvidenceCreated.model_validate(event.model_dump())
    assert restored == event


def test_event_can_carry_policy_and_matrix_versions():
    event = ProviderObservationAttempted(
        event_id="event_1",
        document_ref="doc-1",
        provider_id="docling",
        provider_version="1.0",
        invocation_id="invocation_1",
        reconciliation_policy_version=3,
        capability_matrix_version=2,
    )
    assert event.reconciliation_policy_version == 3
    assert event.capability_matrix_version == 2


def test_event_policy_and_matrix_versions_default_to_none():
    event = ProviderObservationAttempted(
        event_id="event_1",
        document_ref="doc-1",
        provider_id="docling",
        provider_version="1.0",
        invocation_id="invocation_1",
    )
    assert event.reconciliation_policy_version is None
    assert event.capability_matrix_version is None


def test_parse_event_dispatches_on_kind():
    evidence = Evidence.create(
        provider="docling", provider_version="1.0", raw_output="x", processing_stage=ProcessingStage.OCR
    )
    event = EvidenceCreated(
        event_id="event_1", document_ref="doc-1", invocation_id="invocation_1", evidence=evidence
    )
    parsed = parse_event(event.model_dump())
    assert isinstance(parsed, EvidenceCreated)
    assert parsed == event
