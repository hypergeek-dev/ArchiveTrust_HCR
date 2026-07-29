"""Quality Center ViewModel (ROADMAP_V2.md S10)."""

from __future__ import annotations

from archivetrust.learning.analytics.provider_health import ProviderHealth
from archivetrust.learning.analytics.quality import HumanEffortSummary, ObservationTypeQuality
from archivetrust.learning.review.session import sessions_from_interactions
from archivetrust.learning.review.sink import ReviewInteractionSink
from archivetrust.learning.source import TelemetrySource
from archivetrust.presentation.read_model import ReadModel
from archivetrust.presentation.read_model.projections.quality import (
    ConfidenceBucket,
    CorrectionBreakdown,
    ReviewDecisionStats,
)


class QualityCenterViewModel:
    def __init__(
        self,
        source: TelemetrySource,
        interaction_sink: ReviewInteractionSink,
        *,
        read_model: ReadModel | None = None,
    ) -> None:
        self._source = source
        self._interactions = interaction_sink
        self._read_model = read_model or ReadModel(
            telemetry_source=source,
            review_interactions=interaction_sink,
        )

    def observation_survival(self) -> tuple[ObservationTypeQuality, ...]:
        return self._read_model.observation_survival()

    def provider_accuracy(self) -> tuple[ProviderHealth, ...]:
        return self._read_model.provider_execution()

    def human_effort(self) -> HumanEffortSummary:
        summary = self._read_model.review_session_activity()
        if summary is not None:
            return summary

        from archivetrust.learning.analytics.quality import human_effort_summary

        sessions = sessions_from_interactions(tuple(self._interactions.all_interactions()))
        return human_effort_summary(sessions)

    def confidence_distribution(self) -> tuple[ConfidenceBucket, ...]:
        return self._read_model.confidence_distribution()

    def correction_breakdown(self) -> CorrectionBreakdown:
        return self._read_model.correction_breakdown()

    def review_decision_stats(self) -> ReviewDecisionStats:
        return self._read_model.review_decision_stats()
