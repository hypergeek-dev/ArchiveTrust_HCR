"""Quality projections over the shared core aggregate."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from archivetrust.presentation.read_model.core import CoreAggregate


class ConfidenceBucket(BaseModel):
    model_config = ConfigDict(frozen=True)

    label: str
    count: int


class CorrectionBreakdown(BaseModel):
    model_config = ConfigDict(frozen=True)

    by_category: tuple[tuple[str, int], ...]
    by_action: tuple[tuple[str, int], ...]
    total: int


class ReviewDecisionStats(BaseModel):
    model_config = ConfigDict(frozen=True)

    terminal_decisions_recorded: int
    by_outcome: tuple[tuple[str, int], ...]
    corrections_submitted: int
    packets_dispatched: int = 0
    packets_closed: int = 0
    open_dispatched_packets: int = 0
    packet_closures_by_kind: tuple[tuple[str, int], ...] = ()


def confidence_distribution(stats: CoreAggregate) -> tuple[ConfidenceBucket, ...]:
    buckets = {
        "High (>=0.90)": 0,
        "Good (0.70-0.90)": 0,
        "Fair (0.50-0.70)": 0,
        "Low (<0.50)": 0,
        "Not yet scored": 0,
    }
    for canonical in stats.latest_canonicals.values():
        confidence = canonical.canonical_confidence
        if confidence is None:
            buckets["Not yet scored"] += 1
        elif confidence.value >= 0.9:
            buckets["High (>=0.90)"] += 1
        elif confidence.value >= 0.7:
            buckets["Good (0.70-0.90)"] += 1
        elif confidence.value >= 0.5:
            buckets["Fair (0.50-0.70)"] += 1
        else:
            buckets["Low (<0.50)"] += 1
    return tuple(ConfidenceBucket(label=k, count=v) for k, v in buckets.items())


def correction_breakdown(stats: CoreAggregate) -> CorrectionBreakdown:
    by_category: dict[str, int] = {}
    by_action: dict[str, int] = {}
    for (category, action), count in stats.corrections_by_category_action.items():
        by_category[category] = by_category.get(category, 0) + count
        by_action[action] = by_action.get(action, 0) + count
    return CorrectionBreakdown(
        by_category=tuple(sorted(by_category.items(), key=lambda kv: (-kv[1], kv[0]))),
        by_action=tuple(sorted(by_action.items(), key=lambda kv: (-kv[1], kv[0]))),
        total=stats.corrections_submitted,
    )


def review_decision_stats(stats: CoreAggregate) -> ReviewDecisionStats:
    return ReviewDecisionStats(
        terminal_decisions_recorded=stats.review_outcomes_recorded,
        by_outcome=tuple(sorted(stats.review_outcomes_by_outcome.items())),
        corrections_submitted=stats.corrections_submitted,
        packets_dispatched=stats.review_packets_dispatched,
        packets_closed=stats.review_packets_closed,
        open_dispatched_packets=max(0, stats.review_packets_dispatched - stats.review_packets_closed),
        packet_closures_by_kind=tuple(sorted(stats.review_packet_closures_by_kind.items())),
    )
