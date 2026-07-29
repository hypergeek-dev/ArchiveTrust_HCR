"""Versioned JSON export for canonical document snapshots."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from archivetrust.application.current_state import CurrentStateService
from archivetrust.domain.current_state import CurrentDocumentState
from archivetrust.infrastructure.storage.integrity import write_file_digest_manifest

from datetime import datetime, timezone

EXPORT_SCHEMA = "archivetrust.canonical_document.v2"


class WorkspaceInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    name: str


class ExportProvenance(BaseModel):
    model_config = ConfigDict(frozen=True)

    event_id: str
    recorded_at: str | None
    schema_version: int
    reconciliation_policy_version: int | None
    capability_matrix_version: int | None
    confidence_policy_version: int | None
    feedback_policy_version: int | None
    alignment_algorithm_version: int | None


class CanonicalDocumentExport(BaseModel):
    model_config = ConfigDict(frozen=True)

    logical_document_id: str
    document_snapshot_id: str
    archive_object_ref: str
    contained_observations: tuple[str, ...]
    ontology_version: int
    document_version: int
    reassembly_trigger: str
    supersedes: str | None
    superseded_by: str | None


class CanonicalObservationExport(BaseModel):
    model_config = ConfigDict(frozen=True)

    canonical_observation_id: str
    semantic_slot_id: str
    observation_type: str
    payload: dict
    classification: str
    contested: bool
    corrected: bool
    comparison_confidence: dict
    canonical_confidence: dict | None
    contributing_observation_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    provider_versions: tuple[str, ...]
    supersedes: str | None
    superseded_by: str | None
    human_correction_ref: str | None


class ExportReviewOutcome(BaseModel):
    model_config = ConfigDict(frozen=True)

    outcome_id: str
    semantic_slot_id: str
    canonical_observation_id: str
    action: str
    outcome: str
    correction_id: str | None
    recorded_at: str | None


class ExportedDocument(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_ref: str
    display_name: str
    canonical_document: CanonicalDocumentExport
    canonical_observations: tuple[CanonicalObservationExport, ...]
    contested_slots: tuple[str, ...]
    provenance: ExportProvenance
    review_outcomes: tuple[ExportReviewOutcome, ...]
    integrity_status: str
    archive_object_hash: str | None
    unavailable_or_unresolved_fields: tuple[str, ...]


class WorkspaceExport(BaseModel):
    model_config = ConfigDict(frozen=True)

    export_schema: str
    exported_at: str
    workspace: WorkspaceInfo
    documents: tuple[ExportedDocument, ...]
    operational_review_data_included: bool = True
    evaluation_reference_data: str = "excluded_separate_data_plane"


def workspace_export(
    *, workspace, telemetry_source, acquisition_manager=None, current_state_service=None
) -> WorkspaceExport:
    states = (current_state_service or CurrentStateService(telemetry_source)).documents()
    documents = [
        _exported_document(
            state=state,
            acquisition_manager=acquisition_manager,
        )
        for state in states
        if state.export_eligible
    ]
    documents.sort(key=lambda doc: (doc.display_name.lower(), doc.document_ref))
    return WorkspaceExport(
        export_schema=EXPORT_SCHEMA,
        exported_at=datetime.now(timezone.utc).isoformat(),
        workspace=WorkspaceInfo(id=workspace.id, name=workspace.name),
        documents=tuple(documents),
    )


def export_workspace_json(
    *, workspace, telemetry_source, acquisition_manager=None, current_state_service=None
) -> str:
    payload = workspace_export(
        workspace=workspace,
        telemetry_source=telemetry_source,
        acquisition_manager=acquisition_manager,
        current_state_service=current_state_service,
    )
    return json.dumps(payload.model_dump(mode="json"), indent=2, sort_keys=True)


def write_workspace_json(
    path: str | Path,
    *,
    workspace,
    telemetry_source,
    acquisition_manager=None,
    integrity_sidecar: bool = False,
    current_state_service=None,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        export_workspace_json(
            workspace=workspace,
            telemetry_source=telemetry_source,
            acquisition_manager=acquisition_manager,
            current_state_service=current_state_service,
        ),
        encoding="utf-8",
    )
    if integrity_sidecar:
        write_file_digest_manifest(output_path)
    return output_path


def exportable_document_count(telemetry_source) -> int:
    return sum(
        state.export_eligible for state in CurrentStateService(telemetry_source).documents()
    )


def _exported_document(
    *,
    state: CurrentDocumentState,
    acquisition_manager,
) -> ExportedDocument:
    document = state.latest_canonical_document
    if document is None or not state.export_eligible:
        raise ValueError(f"Document {state.document_ref} is not export eligible")
    archive_object = (
        acquisition_manager.archive_object_by_ref(document.archive_object_ref)
        if acquisition_manager is not None
        else None
    )
    return ExportedDocument(
        document_ref=state.document_ref,
        display_name=_document_label(state.document_ref, archive_object=archive_object),
        canonical_document=CanonicalDocumentExport(
            logical_document_id=document.logical_document_id,
            document_snapshot_id=document.document_snapshot_id,
            archive_object_ref=document.archive_object_ref,
            contained_observations=document.contained_observations,
            ontology_version=document.ontology_version,
            document_version=document.document_version,
            reassembly_trigger=document.reassembly_trigger,
            supersedes=document.supersedes,
            superseded_by=document.superseded_by,
        ),
        canonical_observations=tuple(
            CanonicalObservationExport(
                canonical_observation_id=slot.current.canonical_observation_id,
                semantic_slot_id=slot.semantic_slot_id,
                observation_type=slot.current.observation_type.value,
                payload=slot.current.payload.model_dump(mode="json"),
                classification=slot.classification.value,
                contested=slot.contested,
                corrected=slot.corrected,
                comparison_confidence=slot.current.comparison_confidence.model_dump(mode="json"),
                canonical_confidence=(
                    slot.current.canonical_confidence.model_dump(mode="json")
                    if slot.current.canonical_confidence is not None
                    else None
                ),
                contributing_observation_ids=tuple(
                    observation.observation_id for observation in slot.contributing_observations
                ),
                evidence_ids=tuple(evidence.evidence_id for evidence in slot.evidence),
                provider_versions=tuple(
                    sorted(
                        {
                            f"{observation.provider_id}@{observation.provider_version}"
                            for observation in slot.contributing_observations
                        }
                    )
                ),
                supersedes=slot.current.supersedes,
                superseded_by=slot.current.superseded_by,
                human_correction_ref=slot.current.human_correction_ref,
            )
            for slot in state.slots
        ),
        contested_slots=tuple(slot.semantic_slot_id for slot in state.slots if slot.contested),
        provenance=ExportProvenance(
            event_id=state.latest_snapshot_event_id or "unavailable",
            recorded_at=state.latest_snapshot_recorded_at,
            schema_version=state.latest_snapshot_schema_version or 1,
            reconciliation_policy_version=state.policy_versions.reconciliation_version,
            capability_matrix_version=state.policy_versions.capability_matrix_version,
            confidence_policy_version=state.policy_versions.confidence_version,
            feedback_policy_version=state.policy_versions.feedback_version,
            alignment_algorithm_version=state.policy_versions.alignment_version,
        ),
        review_outcomes=tuple(
            ExportReviewOutcome(
                outcome_id=outcome.outcome_id,
                semantic_slot_id=outcome.semantic_slot_id,
                canonical_observation_id=outcome.canonical_observation_id,
                action=outcome.action,
                outcome=outcome.outcome.value,
                correction_id=outcome.correction_id,
                recorded_at=outcome.recorded_at,
            )
            for outcome in state.review_outcomes
        ),
        integrity_status=state.integrity_status,
        archive_object_hash=getattr(archive_object, "content_hash", None),
        unavailable_or_unresolved_fields=tuple(
            field
            for field, unavailable in (
                ("archive_object_hash", archive_object is None),
                ("integrity_status", state.integrity_status == "not_checked"),
                ("contested_slots", bool(state.unresolved_count)),
            )
            if unavailable
        ),
    )


def _document_label(document_ref: str, *, archive_object) -> str:
    original_filename = getattr(archive_object, "original_filename", None)
    if isinstance(original_filename, str) and original_filename.strip():
        return original_filename.strip()
    if len(document_ref) <= 10:
        return document_ref
    return f"{document_ref[:10]}..."
