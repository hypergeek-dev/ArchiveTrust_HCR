from __future__ import annotations

import pytest

from archivetrust.domain.calibration.agreement import (
    AdjudicationReason,
    ReviewerLabel,
    agreement_report,
    cohens_kappa,
    fleiss_kappa,
    items_requiring_adjudication,
    percent_agreement,
)


def label(item: str, reviewer: str, value: str) -> ReviewerLabel:
    return ReviewerLabel(item_id=item, reviewer_ref=reviewer, label=value)


def test_duplicate_reviewer_item_label_fails_clearly() -> None:
    labels = [label("slot-1", "reviewer-a", "accept"), label("slot-1", "reviewer-a", "reject")]

    with pytest.raises(ValueError, match="more than one label"):
        cohens_kappa(labels)


def test_percent_agreement_is_undefined_without_double_reviewed_items() -> None:
    metric = percent_agreement([label("slot-1", "reviewer-a", "accept")])

    assert metric.value is None
    assert metric.reason == "no_items_with_two_or_more_labels"


def test_cohens_kappa_perfect_agreement_is_one() -> None:
    labels = [
        label("slot-1", "reviewer-a", "accept"),
        label("slot-1", "reviewer-b", "accept"),
        label("slot-2", "reviewer-a", "reject"),
        label("slot-2", "reviewer-b", "reject"),
    ]

    metric = cohens_kappa(labels)

    assert metric.value == 1.0
    assert metric.item_count == 2


def test_cohens_kappa_chance_level_example_is_zero() -> None:
    labels = [
        label("slot-1", "reviewer-a", "yes"),
        label("slot-1", "reviewer-b", "yes"),
        label("slot-2", "reviewer-a", "yes"),
        label("slot-2", "reviewer-b", "no"),
        label("slot-3", "reviewer-a", "no"),
        label("slot-3", "reviewer-b", "yes"),
        label("slot-4", "reviewer-a", "no"),
        label("slot-4", "reviewer-b", "no"),
    ]

    assert cohens_kappa(labels).value == 0.0


def test_cohens_kappa_can_be_negative_for_systematic_disagreement() -> None:
    labels = [
        label("slot-1", "reviewer-a", "yes"),
        label("slot-1", "reviewer-b", "no"),
        label("slot-2", "reviewer-a", "yes"),
        label("slot-2", "reviewer-b", "no"),
        label("slot-3", "reviewer-a", "no"),
        label("slot-3", "reviewer-b", "yes"),
        label("slot-4", "reviewer-a", "no"),
        label("slot-4", "reviewer-b", "yes"),
    ]

    assert cohens_kappa(labels).value == -1.0


def test_fleiss_kappa_perfect_agreement_is_one() -> None:
    labels = [
        label("slot-1", "reviewer-a", "accept"),
        label("slot-1", "reviewer-b", "accept"),
        label("slot-1", "reviewer-c", "accept"),
        label("slot-2", "reviewer-a", "reject"),
        label("slot-2", "reviewer-b", "reject"),
        label("slot-2", "reviewer-c", "reject"),
    ]

    assert fleiss_kappa(labels).value == 1.0


def test_fleiss_kappa_is_undefined_for_non_fixed_panel() -> None:
    labels = [
        label("slot-1", "reviewer-a", "accept"),
        label("slot-1", "reviewer-b", "accept"),
        label("slot-2", "reviewer-a", "reject"),
        label("slot-2", "reviewer-b", "reject"),
        label("slot-2", "reviewer-c", "reject"),
    ]

    metric = fleiss_kappa(labels)

    assert metric.value is None
    assert metric.reason == "fleiss_kappa_requires_fixed_panel_size"


def test_items_requiring_adjudication_identifies_under_reviewed_and_split_slots() -> None:
    labels = [
        label("slot-1", "reviewer-a", "accept"),
        label("slot-1", "reviewer-b", "accept"),
        label("slot-2", "reviewer-a", "accept"),
        label("slot-2", "reviewer-b", "reject"),
        label("slot-3", "reviewer-a", "accept"),
    ]

    items = items_requiring_adjudication(labels)

    assert tuple(item.item_id for item in items) == ("slot-2", "slot-3")
    assert items[0].reasons == (AdjudicationReason.REVIEWER_DISAGREEMENT,)
    assert items[1].reasons == (AdjudicationReason.TOO_FEW_LABELS,)


def test_agreement_report_is_json_serializable_and_surfaces_adjudication_ids() -> None:
    labels = [
        label("slot-1", "reviewer-a", "accept"),
        label("slot-1", "reviewer-b", "accept"),
        label("slot-2", "reviewer-a", "accept"),
        label("slot-2", "reviewer-b", "reject"),
    ]

    report = agreement_report(labels)
    payload = report.model_dump(mode="json")

    assert payload["item_count"] == 2
    assert payload["categories"] == ["accept", "reject"]
    assert report.percent_agreement.value == 0.5
    assert report.adjudication_item_ids == ("slot-2",)
