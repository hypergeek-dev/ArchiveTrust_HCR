"""Typed headless service facade for ArchiveTrust core projections."""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel, ConfigDict

from archivetrust.application.export.json_export import export_workspace_json
from archivetrust.learning.analytics.provider_health import ProviderHealth
from archivetrust.learning.analytics.quality import HumanEffortSummary, ObservationTypeQuality
from archivetrust.learning.source import TelemetrySource
from archivetrust.presentation.document_lifecycle import (
    DocumentLifecycleDetail,
    ProcessingRunSummary,
    document_lifecycle_details,
    processing_run_history,
)
from archivetrust.presentation.operations_viewmodel import ProcessingCenterViewModel, ProcessingOverview, QueueItem
from archivetrust.presentation.read_model import ReadModel
from archivetrust.presentation.read_model.projections.quality import (
    ConfidenceBucket,
    CorrectionBreakdown,
    ReviewDecisionStats,
)
from archivetrust.domain.current_state import CurrentDocumentState


class WorkspaceSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    workspace_id: str
    name: str
    description: str


class HeadlessCoreService:
    """Read-only service API over one workspace's replayable projections."""

    def __init__(
        self,
        *,
        workspace,
        telemetry_source: TelemetrySource,
        read_model: ReadModel,
        processing_viewmodel_factory: Callable[[], ProcessingCenterViewModel],
        acquisition_manager,
        processing_progress=None,
    ) -> None:
        self._workspace = workspace
        self._telemetry_source = telemetry_source
        self._read_model = read_model
        self._processing_viewmodel_factory = processing_viewmodel_factory
        self._acquisition_manager = acquisition_manager
        self._processing_progress = processing_progress

    def workspace(self) -> WorkspaceSummary:
        return WorkspaceSummary(
            workspace_id=self._workspace.id,
            name=self._workspace.name,
            description=self._workspace.description,
        )

    def processing_overview(self) -> ProcessingOverview:
        return self._processing().overview()

    def review_queue(self) -> tuple[QueueItem, ...]:
        return self._processing().review_queue()

    def documents(self) -> tuple[DocumentLifecycleDetail, ...]:
        return document_lifecycle_details(
            processing=self._processing(),
            acquisition_manager=self._acquisition_manager,
            telemetry_source=self._telemetry_source,
            processing_progress=self._processing_progress,
        )

    def processing_runs(self) -> tuple[ProcessingRunSummary, ...]:
        return processing_run_history(self._processing_progress)

    def provider_execution(self) -> tuple[ProviderHealth, ...]:
        return self._read_model.provider_execution()

    def observation_survival(self) -> tuple[ObservationTypeQuality, ...]:
        return self._read_model.observation_survival()

    def review_session_activity(self) -> HumanEffortSummary | None:
        return self._read_model.review_session_activity()

    def confidence_distribution(self) -> tuple[ConfidenceBucket, ...]:
        return self._read_model.confidence_distribution()

    def correction_breakdown(self) -> CorrectionBreakdown:
        return self._read_model.correction_breakdown()

    def review_decision_stats(self) -> ReviewDecisionStats:
        return self._read_model.review_decision_stats()

    def current_document(self, document_ref: str) -> CurrentDocumentState:
        return self._read_model.current_document(document_ref)

    def current_documents(self) -> tuple[CurrentDocumentState, ...]:
        return self._read_model.current_documents()

    def export_workspace_json(self) -> str:
        return export_workspace_json(
            workspace=self._workspace,
            telemetry_source=self._telemetry_source,
            acquisition_manager=self._acquisition_manager,
            current_state_service=self._read_model.current_state,
        )

    def _processing(self) -> ProcessingCenterViewModel:
        return self._processing_viewmodel_factory()
