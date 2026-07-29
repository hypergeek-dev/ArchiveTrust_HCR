from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from xml.etree import ElementTree as ET

from archivetrust.application.current_state import CurrentStateService
from archivetrust.application.export.interop import write_interop_package
from archivetrust.application.export.json_export import export_workspace_json
from archivetrust.application.journal import Journal
from archivetrust.core.service import HeadlessCoreService
from archivetrust.domain.confidence.models import ComparisonClassification
from archivetrust.evaluation.evaluate import evaluate
from archivetrust.evaluation.ground_truth import (
    FileGroundTruthStore,
    GroundTruthAnnotation,
    VerificationStatus,
)
from archivetrust.infrastructure.storage.telemetry_sink import InMemoryTelemetrySink
from archivetrust.learning.review.sink import InMemoryReviewInteractionSink
from archivetrust.presentation.evidence_explorer_viewmodel import EvidenceExplorerViewModel
from archivetrust.presentation.read_model import ReadModel
from archivetrust.review.service import ReviewAction, ReviewService

from tests.review._helpers import emit_document_snapshot, emit_slot, heading


@dataclass(frozen=True)
class _Workspace:
    id: str = "workspace-1"
    name: str = "Consistency"
    description: str = ""


def test_corrected_identity_and_value_are_consistent_across_supported_surfaces(tmp_path) -> None:
    sink = InMemoryTelemetrySink()
    original = emit_slot(
        sink,
        document_ref="doc1",
        canonical_payload=heading("Chapter I"),
        provider_payloads=(
            ("source_a", heading("Chapter I")),
            ("source_b", heading("Chapter l")),
        ),
        classification=ComparisonClassification.CONTESTED,
    )
    emit_document_snapshot(sink, document_ref="doc1", archive_object_ref="archive1")
    review = ReviewService(
        telemetry_source=sink,
        telemetry_sink=sink,
        interaction_sink=InMemoryReviewInteractionSink(),
    )
    (packet,) = review.open_document(document_ref="doc1", archive_object_ref="archive1")
    result = review.submit_decision(
        packet=packet,
        action=ReviewAction.MANUAL_EDIT,
        corrected_output="Chapter 1",
    )
    current_id = result.resulting_canonical_observation.canonical_observation_id

    authoritative = CurrentStateService(sink).document("doc1")
    assert authoritative.slot(original.semantic_slot_id).current.payload.text == "Chapter 1"
    assert authoritative.effective_contained_observations == (current_id,)

    replayed = Journal().replay(sink.events_for_document("doc1"))
    assert replayed.canonical_observation_history(original.semantic_slot_id)[-1].canonical_observation_id == current_id

    ui_trace = EvidenceExplorerViewModel(sink).document_trace("doc1").canonical_traces[0]
    assert (ui_trace.canonical_observation_id, ui_trace.value) == (current_id, "Chapter 1")

    workspace = _Workspace()
    json_payload = json.loads(export_workspace_json(workspace=workspace, telemetry_source=sink))
    json_observation = json_payload["documents"][0]["canonical_observations"][0]
    assert (json_observation["canonical_observation_id"], json_observation["payload"]["text"]) == (
        current_id,
        "Chapter 1",
    )

    package = write_interop_package(
        tmp_path / "interop", workspace=workspace, telemetry_source=sink
    )
    page_value = ET.parse(package.documents[0].page_xml).findtext(
        ".//{http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15}Unicode"
    )
    alto_value = ET.parse(package.documents[0].alto_xml).find(
        ".//{http://www.loc.gov/standards/alto/ns-v4#}String"
    ).attrib["CONTENT"]
    mets_observation = ET.parse(package.mets_xml).find(
        ".//ArchiveTrustExport/Document/CanonicalObservation"
    )
    assert page_value == alto_value == mets_observation.attrib["value"] == "Chapter 1"
    assert mets_observation.attrib["canonicalObservationId"] == current_id

    store = FileGroundTruthStore(tmp_path / "annotations.jsonl")
    store.append(
        GroundTruthAnnotation(
            annotation_id="annotation-1",
            archive_object_ref="doc1",
            content_hash="test-hash",
            page=1,
            field="heading",
            observation_type="heading",
            text="Chapter 1",
            annotator="independent-reviewer",
            method="visual transcription",
                source="consistency-test",
                created_at=datetime.now(timezone.utc).isoformat(),
                verification_status=VerificationStatus.VERIFIED,
                legal_basis="test campaign authorization",
                sampling_basis="complete fixture population",
            )
    )
    evaluation = evaluate(store=store, telemetry_source=sink)
    evaluated = next(score for score in evaluation.field_scores if score.system == "canonical")
    assert evaluated.best_candidate == "Chapter 1"

    read_model = ReadModel(telemetry_source=sink)
    service = HeadlessCoreService(
        workspace=workspace,
        telemetry_source=sink,
        read_model=read_model,
        processing_viewmodel_factory=lambda: None,
        acquisition_manager=None,
    )
    service_current = service.current_document("doc1").slot(original.semantic_slot_id).current
    assert (service_current.canonical_observation_id, service_current.payload.text) == (
        current_id,
        "Chapter 1",
    )
