"""Exclusion from the benchmark, only with a recorded reason (task brief item 6): success and
rejection paths."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from archivetrust.review.blind_review.exclusion import ExclusionRecord, exclude_from_benchmark
from archivetrust.review.blind_review.store import BlindReviewStore


def test_exclusion_succeeds_with_a_real_reason():
    store = BlindReviewStore()
    record = exclude_from_benchmark(
        store=store,
        target_ref="ground_truth_annotation_1",
        reason="Source image is illegible due to water damage across the whole line.",
        excluded_by="curator-1",
        excluded_at="2026-01-01T08:00:00Z",
    )
    assert record.target_ref == "ground_truth_annotation_1"
    assert record.reason.startswith("Source image is illegible")
    assert store.exclusion_for("ground_truth_annotation_1") is record


def test_exclusion_rejects_empty_reason():
    store = BlindReviewStore()
    with pytest.raises(ValidationError):
        exclude_from_benchmark(
            store=store,
            target_ref="ground_truth_annotation_1",
            reason="",
            excluded_by="curator-1",
            excluded_at="2026-01-01T08:00:00Z",
        )
    assert store.exclusion_for("ground_truth_annotation_1") is None


def test_exclusion_rejects_whitespace_only_reason():
    store = BlindReviewStore()
    with pytest.raises(ValidationError):
        exclude_from_benchmark(
            store=store,
            target_ref="ground_truth_annotation_1",
            reason="   ",
            excluded_by="curator-1",
            excluded_at="2026-01-01T08:00:00Z",
        )


def test_exclusion_record_construction_itself_rejects_missing_reason():
    with pytest.raises(ValidationError):
        ExclusionRecord.create(
            target_ref="ground_truth_annotation_1",
            reason=None,  # type: ignore[arg-type]
            excluded_by="curator-1",
            excluded_at="2026-01-01T08:00:00Z",
        )
