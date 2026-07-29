"""Read-model facade shared by presentation ViewModels."""

from __future__ import annotations

from archivetrust.learning.analytics.provider_health import ProviderHealth, provider_health
from archivetrust.application.current_state import CurrentStateService
from archivetrust.domain.current_state import CurrentDocumentState
from archivetrust.learning.analytics.quality import (
    HumanEffortSummary,
    ObservationTypeQuality,
    human_effort_summary,
    observation_type_quality,
)
from archivetrust.learning.review.session import sessions_from_interactions
from archivetrust.learning.review.sink import ReviewInteractionSink
from archivetrust.learning.source import TelemetrySource
from archivetrust.presentation.read_model.core import CoreAggregate
from archivetrust.presentation.read_model.projections.quality import (
    ConfidenceBucket,
    CorrectionBreakdown,
    ReviewDecisionStats,
    confidence_distribution,
    correction_breakdown,
    review_decision_stats,
)


class ReadModel:
    """One per Workspace, owning shared aggregates and cached analytics projections."""

    def __init__(
        self,
        *,
        telemetry_source: TelemetrySource,
        review_interactions: ReviewInteractionSink | None = None,
        core: CoreAggregate | None = None,
    ) -> None:
        self._telemetry_source = telemetry_source
        self._review_interactions = review_interactions
        self.core = core if core is not None else CoreAggregate()
        self.current_state = CurrentStateService(telemetry_source)
        self._provider_health_cache: tuple[int, tuple[ProviderHealth, ...]] | None = None
        self._observation_quality_cache: tuple[int, tuple[ObservationTypeQuality, ...]] | None = None

    def refresh(self) -> None:
        # No tuple() materialization (release WS5): `all_events()` is already a snapshot-
        # consistent Sequence; `CoreAggregate.update` slices it from its cursor, so a refresh
        # parses only events recorded since the last one and holds none of them afterwards.
        self.core.update(self._telemetry_source.all_events())

    def provider_execution(self) -> tuple[ProviderHealth, ...]:
        self.refresh()
        cursor = self.core.cursor
        if self._provider_health_cache is None or self._provider_health_cache[0] != cursor:
            self._provider_health_cache = (cursor, tuple(provider_health(self._telemetry_source)))
        return self._provider_health_cache[1]

    def observation_survival(self) -> tuple[ObservationTypeQuality, ...]:
        self.refresh()
        cursor = self.core.cursor
        if self._observation_quality_cache is None or self._observation_quality_cache[0] != cursor:
            self._observation_quality_cache = (cursor, tuple(observation_type_quality(self._telemetry_source)))
        return self._observation_quality_cache[1]

    def review_session_activity(self) -> HumanEffortSummary | None:
        if self._review_interactions is None:
            return None
        sessions = sessions_from_interactions(tuple(self._review_interactions.all_interactions()))
        return human_effort_summary(sessions)

    def confidence_distribution(self) -> tuple[ConfidenceBucket, ...]:
        self.refresh()
        return confidence_distribution(self.core)

    def correction_breakdown(self) -> CorrectionBreakdown:
        self.refresh()
        return correction_breakdown(self.core)

    def review_decision_stats(self) -> ReviewDecisionStats:
        self.refresh()
        return review_decision_stats(self.core)

    def current_document(self, document_ref: str) -> CurrentDocumentState:
        return self.current_state.document(document_ref)

    def current_documents(self) -> tuple[CurrentDocumentState, ...]:
        return self.current_state.documents()
