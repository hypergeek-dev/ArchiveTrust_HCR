"""Coverage analysis: which evidence patterns never accumulate calibration evidence on their own
(Milestone 11, Phase 7).

Detects blind spots and recommends a statistically justified sampling strategy to close them --
it never enqueues anything itself (spec: "Do not automatically begin reviewing them"). A pattern
is a blind spot exactly when it has zero *calibration*-intent evidence (Phase 4/5's
`CalibrationMaturity.NO_EVIDENCE`); patterns already past that stage are, by definition, not blind
spots even if they were slow to get there.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum

from pydantic import BaseModel, ConfigDict

from archivetrust.domain.calibration.pattern import CalibrationPattern
from archivetrust.domain.calibration.progress import CalibrationMaturity, CalibrationProgress


class BlindSpotReason(str, Enum):
    NEVER_QUEUED_OPERATIONALLY = "never_queued_operationally"
    """`queued_for_review == 0` under the current `TriagePolicy` -- normal operation will never
    surface a single item of this pattern, calibration or otherwise, without a deliberate sampling
    decision."""

    NO_CALIBRATION_EVIDENCE = "no_calibration_evidence"
    """The pattern is reachable operationally (some items are queued), but no
    `ReviewIntent.CALIBRATION` review has ever been recorded for it -- operational corrections
    alone cannot close this gap (see `domain/calibration/progress.py` module docstring)."""


class BlindSpot(BaseModel):
    model_config = ConfigDict(frozen=True)

    pattern: str
    population: int
    corpus_share: float
    reason: BlindSpotReason
    recommended_sampling_strategy: str
    rationale: str


def find_blind_spots(
    patterns: tuple[CalibrationPattern, ...],
    progress_by_pattern: Mapping[str, CalibrationProgress],
    *,
    total_population: int,
) -> tuple[BlindSpot, ...]:
    """Every pattern with `CalibrationMaturity.NO_EVIDENCE`, in descending order of corpus share
    (the patterns whose blind-spot status matters most, surfaced first).
    """
    blind_spots: list[BlindSpot] = []
    for pattern in patterns:
        progress = progress_by_pattern.get(pattern.name)
        if progress is None or progress.maturity != CalibrationMaturity.NO_EVIDENCE:
            continue
        share = pattern.population / total_population if total_population else 0.0
        never_queued = pattern.queued_for_review == 0
        reason = (
            BlindSpotReason.NEVER_QUEUED_OPERATIONALLY
            if never_queued
            else BlindSpotReason.NO_CALIBRATION_EVIDENCE
        )
        strategy = "stratified_sampling" if never_queued else "random_sampling"
        rationale = (
            f"{pattern.population} slot(s) ({share:.2%} of corpus); "
            + (
                "never queued under the current triage policy, so normal operation will never "
                "generate evidence for it -- reaching it requires deliberately sampling outside "
                "the operational review flow."
                if never_queued
                else "queued operationally, but no statistically-sampled calibration review has "
                "been recorded for it yet -- operational corrections are a biased subsample and "
                "cannot substitute for calibration evidence."
            )
        )
        blind_spots.append(
            BlindSpot(
                pattern=pattern.name,
                population=pattern.population,
                corpus_share=round(share, 4),
                reason=reason,
                recommended_sampling_strategy=strategy,
                rationale=rationale,
            )
        )
    blind_spots.sort(key=lambda b: -b.corpus_share)
    return tuple(blind_spots)
