from __future__ import annotations

import pytest
from pydantic import ValidationError

from archivetrust.domain.evidence.models import (
    BoundingBox,
    Evidence,
    Precision,
    ProcessingStage,
)


def test_identical_raw_output_content_addresses_to_same_id():
    a = Evidence.create(
        provider="docling",
        provider_version="1.0",
        raw_output="same text",
        processing_stage=ProcessingStage.OCR,
    )
    b = Evidence.create(
        provider="docling",
        provider_version="1.0",
        raw_output="same text",
        processing_stage=ProcessingStage.OCR,
    )
    assert a.evidence_id == b.evidence_id


def test_different_raw_output_content_addresses_differently():
    a = Evidence.create(
        provider="docling", provider_version="1.0", raw_output="a", processing_stage=ProcessingStage.OCR
    )
    b = Evidence.create(
        provider="docling", provider_version="1.0", raw_output="b", processing_stage=ProcessingStage.OCR
    )
    assert a.evidence_id != b.evidence_id


def test_hand_set_evidence_id_that_does_not_match_content_is_rejected():
    with pytest.raises(ValidationError):
        Evidence(
            evidence_id="evidence_bogus",
            provider="docling",
            provider_version="1.0",
            raw_output="x",
            processing_stage=ProcessingStage.OCR,
        )


def test_evidence_is_immutable():
    ev = Evidence.create(
        provider="docling", provider_version="1.0", raw_output="x", processing_stage=ProcessingStage.OCR
    )
    with pytest.raises(ValidationError):
        ev.raw_output = "y"  # type: ignore[misc]


def test_bounding_box_requires_non_negative_extent():
    with pytest.raises(ValidationError):
        BoundingBox(x0=10, y0=0, x1=0, y1=10, precision=Precision.PIXEL_ACCURATE)


def test_coarse_and_pixel_accurate_boxes_are_distinguishable():
    coarse = BoundingBox(x0=0, y0=0, x1=1, y1=1, precision=Precision.COARSE_ESTIMATE)
    exact = BoundingBox(x0=0, y0=0, x1=1, y1=1, precision=Precision.PIXEL_ACCURATE)
    assert coarse.precision != exact.precision


def test_round_trip_serialization():
    ev = Evidence.create(
        provider="qwen2.5-vl",
        provider_version="2.5",
        raw_output="{\"heading\": \"Ch 1\"}",
        processing_stage=ProcessingStage.VLM_INFERENCE,
        prompt="Extract headings",
        bounding_box=BoundingBox(x0=0, y0=0, x1=10, y1=10, precision=Precision.COARSE_ESTIMATE),
    )
    restored = Evidence.model_validate(ev.model_dump())
    assert restored == ev
    restored_json = Evidence.model_validate_json(ev.model_dump_json())
    assert restored_json == ev
