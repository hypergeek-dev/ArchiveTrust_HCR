"""Reviewer-label agreement statistics for calibration campaigns."""

from __future__ import annotations

from collections import Counter, defaultdict
from enum import Enum

from pydantic import BaseModel, ConfigDict


class AdjudicationReason(str, Enum):
    TOO_FEW_LABELS = "too_few_labels"
    REVIEWER_DISAGREEMENT = "reviewer_disagreement"
    NON_FIXED_PANEL = "non_fixed_panel"


class ReviewerLabel(BaseModel):
    model_config = ConfigDict(frozen=True)

    item_id: str
    reviewer_ref: str
    label: str


class AgreementMetric(BaseModel):
    model_config = ConfigDict(frozen=True)

    value: float | None
    item_count: int
    label_count: int
    reason: str | None = None


class AdjudicationItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    item_id: str
    reasons: tuple[AdjudicationReason, ...]
    labels: tuple[ReviewerLabel, ...]


class AgreementReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    item_count: int
    reviewer_count: int
    label_count: int
    categories: tuple[str, ...]
    percent_agreement: AgreementMetric
    cohens_kappa: AgreementMetric
    fleiss_kappa: AgreementMetric
    adjudication_items: tuple[AdjudicationItem, ...]

    @property
    def adjudication_item_ids(self) -> tuple[str, ...]:
        return tuple(item.item_id for item in self.adjudication_items)


def agreement_report(labels: tuple[ReviewerLabel, ...] | list[ReviewerLabel]) -> AgreementReport:
    normalized = _normalize_labels(labels)
    grouped = _labels_by_item(normalized)
    reviewers = tuple(sorted({label.reviewer_ref for label in normalized}))
    categories = tuple(sorted({label.label for label in normalized}))
    return AgreementReport(
        item_count=len(grouped),
        reviewer_count=len(reviewers),
        label_count=len(normalized),
        categories=categories,
        percent_agreement=percent_agreement(normalized),
        cohens_kappa=cohens_kappa(normalized),
        fleiss_kappa=fleiss_kappa(normalized),
        adjudication_items=items_requiring_adjudication(normalized),
    )


def percent_agreement(labels: tuple[ReviewerLabel, ...] | list[ReviewerLabel]) -> AgreementMetric:
    normalized = _normalize_labels(labels)
    grouped = _labels_by_item(normalized)
    compared = 0
    agreed = 0
    for item_labels in grouped.values():
        if len(item_labels) < 2:
            continue
        compared += 1
        if len({label.label for label in item_labels}) == 1:
            agreed += 1
    if compared == 0:
        return AgreementMetric(
            value=None,
            item_count=0,
            label_count=len(normalized),
            reason="no_items_with_two_or_more_labels",
        )
    return AgreementMetric(value=agreed / compared, item_count=compared, label_count=len(normalized))


def cohens_kappa(labels: tuple[ReviewerLabel, ...] | list[ReviewerLabel]) -> AgreementMetric:
    normalized = _normalize_labels(labels)
    grouped = _labels_by_item(normalized)
    paired_items = [item_labels for item_labels in grouped.values() if len(item_labels) == 2]
    if not paired_items:
        return AgreementMetric(
            value=None,
            item_count=0,
            label_count=len(normalized),
            reason="cohens_kappa_requires_two_labels_per_item",
        )

    observed = sum(1 for item_labels in paired_items if item_labels[0].label == item_labels[1].label)
    total = len(paired_items)
    observed_agreement = observed / total

    first_marginal: Counter[str] = Counter()
    second_marginal: Counter[str] = Counter()
    for item_labels in paired_items:
        ordered = sorted(item_labels, key=lambda label: label.reviewer_ref)
        first_marginal[ordered[0].label] += 1
        second_marginal[ordered[1].label] += 1

    categories = set(first_marginal) | set(second_marginal)
    expected_agreement = sum(
        (first_marginal[category] / total) * (second_marginal[category] / total)
        for category in categories
    )
    value = _kappa(observed_agreement, expected_agreement)
    return AgreementMetric(value=value, item_count=total, label_count=2 * total)


def fleiss_kappa(labels: tuple[ReviewerLabel, ...] | list[ReviewerLabel]) -> AgreementMetric:
    normalized = _normalize_labels(labels)
    grouped = _labels_by_item(normalized)
    if not grouped:
        return AgreementMetric(value=None, item_count=0, label_count=0, reason="no_labels")

    panel_sizes = {len(item_labels) for item_labels in grouped.values()}
    if len(panel_sizes) != 1:
        return AgreementMetric(
            value=None,
            item_count=len(grouped),
            label_count=len(normalized),
            reason="fleiss_kappa_requires_fixed_panel_size",
        )
    panel_size = panel_sizes.pop()
    if panel_size < 2:
        return AgreementMetric(
            value=None,
            item_count=len(grouped),
            label_count=len(normalized),
            reason="fleiss_kappa_requires_two_or_more_reviewers",
        )

    categories = tuple(sorted({label.label for label in normalized}))
    label_totals: Counter[str] = Counter()
    per_item_agreement = []
    for item_labels in grouped.values():
        counts = Counter(label.label for label in item_labels)
        label_totals.update(counts)
        per_item_agreement.append(
            sum(count * (count - 1) for count in counts.values()) / (panel_size * (panel_size - 1))
        )

    mean_item_agreement = sum(per_item_agreement) / len(per_item_agreement)
    total_assignments = len(grouped) * panel_size
    expected_agreement = sum((label_totals[category] / total_assignments) ** 2 for category in categories)
    value = _kappa(mean_item_agreement, expected_agreement)
    return AgreementMetric(value=value, item_count=len(grouped), label_count=len(normalized))


def items_requiring_adjudication(
    labels: tuple[ReviewerLabel, ...] | list[ReviewerLabel],
    *,
    minimum_labels: int = 2,
) -> tuple[AdjudicationItem, ...]:
    normalized = _normalize_labels(labels)
    grouped = _labels_by_item(normalized)
    required: list[AdjudicationItem] = []
    for item_id, item_labels in sorted(grouped.items()):
        reasons: list[AdjudicationReason] = []
        if len(item_labels) < minimum_labels:
            reasons.append(AdjudicationReason.TOO_FEW_LABELS)
        if len({label.label for label in item_labels}) > 1:
            reasons.append(AdjudicationReason.REVIEWER_DISAGREEMENT)
        if reasons:
            required.append(
                AdjudicationItem(
                    item_id=item_id,
                    reasons=tuple(reasons),
                    labels=tuple(sorted(item_labels, key=lambda label: label.reviewer_ref)),
                )
            )
    return tuple(required)


def _normalize_labels(labels: tuple[ReviewerLabel, ...] | list[ReviewerLabel]) -> tuple[ReviewerLabel, ...]:
    normalized = tuple(ReviewerLabel.model_validate(label) for label in labels)
    seen: set[tuple[str, str]] = set()
    for label in normalized:
        key = (label.item_id, label.reviewer_ref)
        if key in seen:
            raise ValueError(
                f"Reviewer {label.reviewer_ref!r} supplied more than one label for item {label.item_id!r}"
            )
        seen.add(key)
    return normalized


def _labels_by_item(labels: tuple[ReviewerLabel, ...]) -> dict[str, tuple[ReviewerLabel, ...]]:
    grouped: dict[str, list[ReviewerLabel]] = defaultdict(list)
    for label in labels:
        grouped[label.item_id].append(label)
    return {item_id: tuple(item_labels) for item_id, item_labels in grouped.items()}


def _kappa(observed_agreement: float, expected_agreement: float) -> float | None:
    denominator = 1 - expected_agreement
    if denominator == 0:
        return None
    return (observed_agreement - expected_agreement) / denominator
