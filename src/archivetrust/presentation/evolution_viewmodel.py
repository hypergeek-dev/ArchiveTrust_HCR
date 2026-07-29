"""Evolution Center ViewModel (ROADMAP_V2.md §11).

Framework-independent. Reuses the existing Evolution Center analytics (`detect_correction_hotspots`,
`detect_single_source_concepts`) and surfaces the Alignment Observability record (ROADMAP.md §5.12)
and provider drift, all read from the same telemetry stream. Everything here is advisory evidence
for a human decision — nothing changes production (LP-5). No backend module is modified.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from archivetrust.domain.telemetry.events import (
    AlignmentAttempted,
    ObservationAligned,
    ObservationLeftUnaligned,
)
from archivetrust.learning.analytics.evolution import (
    EvolutionCandidate,
    detect_correction_hotspots,
    detect_single_source_concepts,
)
from archivetrust.learning.analytics.provider_health import provider_health
from archivetrust.learning.source import TelemetrySource


class AlignmentSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    attempts: int
    aligned_observations: int
    unaligned_observations: int
    unaligned_examples: tuple[str, ...]
    """A few rationales for why observations were left unaligned — the concrete signal a future
    'this grouping is wrong' review would act on."""


class ProviderDrift(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider_id: str
    provider_version: str
    correction_rate: float
    contributing_slots: int


class EvolutionCenterViewModel:
    def __init__(self, source: TelemetrySource) -> None:
        self._source = source

    def candidates(self) -> tuple[EvolutionCandidate, ...]:
        """All evidence-backed evolution candidates, most pressing first. Advisory only (LP-5)."""
        hotspots = detect_correction_hotspots(self._source)
        single_source = detect_single_source_concepts(self._source)
        return tuple(sorted(hotspots + single_source, key=lambda c: (-c.magnitude, c.subject)))

    def alignment_summary(self) -> AlignmentSummary:
        attempts = 0
        aligned = 0
        unaligned = 0
        examples: list[str] = []
        for event in self._source.all_events():
            if isinstance(event, AlignmentAttempted):
                attempts += 1
            elif isinstance(event, ObservationAligned):
                aligned += 1
            elif isinstance(event, ObservationLeftUnaligned):
                unaligned += 1
                if event.reason not in examples:
                    examples.append(event.reason)
        return AlignmentSummary(
            attempts=attempts,
            aligned_observations=aligned,
            unaligned_observations=unaligned,
            unaligned_examples=tuple(examples[:4]),
        )

    def provider_drift(self) -> tuple[ProviderDrift, ...]:
        # Provider drift is surfaced as behavior (downstream correction rate), never as blended
        # confidence (LP-6). Providers whose contributions are corrected most are flagged first.
        rows = [
            ProviderDrift(
                provider_id=h.provider_id,
                provider_version=h.provider_version,
                correction_rate=h.correction_rate if h.correction_rate is not None else 0.0,
                contributing_slots=h.contributing_slot_count,
            )
            for h in provider_health(self._source)
        ]
        rows.sort(key=lambda r: (-r.correction_rate, r.provider_id))
        return tuple(rows)
