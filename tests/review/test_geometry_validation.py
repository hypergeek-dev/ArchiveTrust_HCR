from __future__ import annotations

from pathlib import Path

import pytest

from archivetrust.domain.confidence.models import ComparisonClassification, ComparisonConfidence
from archivetrust.domain.evidence.models import BoundingBox, Precision
from archivetrust.domain.ontology.types import ObservationType
from archivetrust.review.geometry_validation import (
    GEOMETRY_VALIDATION_POLICY_VERSION,
    GeometryValidationStatus,
    validate_evidence_geometry,
)
from archivetrust.review.packet import (
    AgreementView,
    CandidateView,
    DisclosureTier,
    EvidenceView,
    ReviewPacket,
    ReviewReason,
)


def _write_visible_pdf(path: Path) -> None:
    stream = b"""BT /F1 18 Tf 72 720 Td (Municipal Record Heading) Tj ET
BT /F1 12 Tf 72 660 Td (First paragraph anchors the review evidence.) Tj ET
BT /F1 12 Tf 72 620 Td (Second paragraph gives a separate visible line.) Tj ET
BT /F1 10 Tf 72 72 Td (Page 1) Tj ET
"""
    objects = [
        b"<</Type/Catalog/Pages 2 0 R>>",
        b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
        b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Resources<</Font<</F1 4 0 R>>>>/Contents 5 0 R>>",
        b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>",
        f"<</Length {len(stream)}>>\nstream\n".encode() + stream + b"endstream",
    ]
    body = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(body))
        body.extend(f"{index} 0 obj\n".encode())
        body.extend(obj)
        body.extend(b"\nendobj\n")
    xref_offset = len(body)
    body.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    body.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        body.extend(f"{offset:010d} 00000 n \n".encode())
    body.extend(f"trailer<</Size {len(objects) + 1}/Root 1 0 R>>\nstartxref\n{xref_offset}\n%%EOF".encode())
    path.write_bytes(bytes(body))


def _evidence(
    *,
    box: BoundingBox | None,
    text: str = "Municipal Record Heading",
    page: int | None = 1,
    metadata: dict | None = None,
) -> EvidenceView:
    return EvidenceView(
        evidence_id="evidence-1",
        provider_id="docling",
        provider_version="2.x",
        page=page,
        bounding_box=box,
        raw_output=text,
        provider_confidence=0.91,
        coordinate_metadata=metadata or {
            "coordinate_units": "pdf_points",
            "coordinate_origin": "bottom_left",
            "page_width": 612.0,
            "page_height": 792.0,
        },
    )


def _candidate(evidence: EvidenceView, *, value: str = "Municipal Record Heading") -> CandidateView:
    return CandidateView(
        observation_id="observation-1",
        provider_id="docling",
        provider_version="2.x",
        value=value,
        provider_confidence=0.91,
        evidence=(evidence,),
    )


def _box(
    x0: float = 72.0,
    y0: float = 706.0,
    x1: float = 330.0,
    y1: float = 740.0,
    *,
    precision: Precision = Precision.PIXEL_ACCURATE,
) -> BoundingBox:
    return BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1, precision=precision)


@pytest.fixture
def visible_pdf(tmp_path) -> Path:
    path = tmp_path / "visible.pdf"
    _write_visible_pdf(path)
    return path


def _validate(evidence: EvidenceView, archive_path: Path | None, **kwargs):
    return validate_evidence_geometry(candidate=_candidate(evidence), evidence=evidence, archive_path=archive_path, **kwargs)


def test_correct_top_left_page_local_geometry(visible_pdf: Path) -> None:
    evidence = _evidence(
        box=_box(72, 52, 330, 86),
        metadata={
            "coordinate_units": "pdf_points",
            "coordinate_origin": "top_left",
            "page_width": 612.0,
            "page_height": 792.0,
        },
    )
    result = _validate(evidence, visible_pdf)
    assert result.status is GeometryValidationStatus.PRECISE_VALID
    assert result.transformed_rectangle == pytest.approx((144.0, 104.0, 516.0, 68.0))
    assert result.may_draw_overlay is True


def test_correct_bottom_left_pdf_geometry(visible_pdf: Path) -> None:
    result = _validate(_evidence(box=_box()), visible_pdf)
    assert result.status is GeometryValidationStatus.PRECISE_VALID
    assert result.transformed_rectangle == pytest.approx((144.0, 104.0, 516.0, 68.0))


def test_correct_top_left_pixel_geometry_uses_rendered_pixel_page_bounds(visible_pdf: Path) -> None:
    evidence = _evidence(
        box=_box(144, 104, 660, 172),
        metadata={
            "coordinate_units": "pixels",
            "coordinate_origin": "top_left",
            "page_width": 1224,
            "page_height": 1584,
        },
    )
    result = _validate(evidence, visible_pdf)
    assert result.status is GeometryValidationStatus.PRECISE_VALID
    assert result.transformed_rectangle == pytest.approx((144.0, 104.0, 516.0, 68.0))


def test_crop_local_geometry_rejected_without_page_transform_metadata(visible_pdf: Path) -> None:
    evidence = _evidence(
        box=_box(0, 0, 100, 100),
        metadata={"coordinate_units": "crop_pixels", "coordinate_origin": "top_left", "page_width": 100, "page_height": 100},
    )
    result = _validate(evidence, visible_pdf)
    assert result.status is GeometryValidationStatus.INCOMPATIBLE_COORDINATE_SPACE
    assert result.reason_code == "CROP_LOCAL_WITHOUT_PAGE_TRANSFORM"
    assert result.may_draw_overlay is False


def test_page_mismatch(visible_pdf: Path) -> None:
    result = _validate(_evidence(box=_box(), page=2), visible_pdf)
    assert result.status is GeometryValidationStatus.PAGE_MISMATCH
    assert result.may_draw_overlay is False


def test_out_of_bounds_box(visible_pdf: Path) -> None:
    result = _validate(_evidence(box=_box(72, 706, 900, 740)), visible_pdf)
    assert result.status is GeometryValidationStatus.OUT_OF_BOUNDS
    assert result.may_draw_overlay is False


def test_invalid_source_dimensions(visible_pdf: Path) -> None:
    evidence = _evidence(
        box=_box(),
        metadata={"coordinate_units": "pdf_points", "coordinate_origin": "bottom_left", "page_width": 0, "page_height": 792},
    )
    result = _validate(evidence, visible_pdf)
    assert result.status is GeometryValidationStatus.INVALID_DIMENSIONS
    assert result.may_draw_overlay is False


def test_missing_geometry(visible_pdf: Path) -> None:
    result = _validate(_evidence(box=None, metadata={}), visible_pdf)
    assert result.status is GeometryValidationStatus.MISSING
    assert result.may_draw_overlay is False


def test_whole_page_evidence(visible_pdf: Path) -> None:
    result = _validate(_evidence(box=None, metadata={"whole_page": True}), visible_pdf)
    assert result.status is GeometryValidationStatus.WHOLE_PAGE
    assert result.may_draw_overlay is False


def test_coarse_geometry(visible_pdf: Path) -> None:
    result = _validate(_evidence(box=_box(precision=Precision.COARSE_ESTIMATE)), visible_pdf)
    assert result.status is GeometryValidationStatus.COARSE_VALID
    assert result.may_draw_overlay is True


def test_candidate_to_evidence_association_mismatch(visible_pdf: Path) -> None:
    evidence = _evidence(
        box=_box(),
        metadata={
            "coordinate_units": "pdf_points",
            "coordinate_origin": "bottom_left",
            "page_width": 612,
            "page_height": 792,
            "observation_id": "other-observation",
        },
    )
    result = _validate(evidence, visible_pdf)
    assert result.status is GeometryValidationStatus.INCOMPATIBLE_COORDINATE_SPACE
    assert result.reason_code == "ASSOCIATION_OBSERVATION_MISMATCH"
    assert result.may_draw_overlay is False


def test_valid_structural_geometry_with_strong_spatial_text_support(visible_pdf: Path) -> None:
    result = _validate(_evidence(box=_box()), visible_pdf, check_spatial_support=True)
    assert result.status is GeometryValidationStatus.PRECISE_VALID
    assert result.diagnostics["spatial_support"]["status"] == "supported"
    assert result.diagnostics["spatial_support"]["score"] > 0.9


def test_structurally_valid_but_spatially_implausible_geometry(visible_pdf: Path) -> None:
    evidence = _evidence(box=_box(72, 70, 120, 90))
    result = _validate(evidence, visible_pdf, check_spatial_support=True)
    assert result.status is GeometryValidationStatus.SPATIALLY_IMPLAUSIBLE
    assert result.may_draw_overlay is False


def test_unavailable_pdf_text_extraction_returns_unverified() -> None:
    result = _validate(_evidence(box=_box()), archive_path=None, check_spatial_support=True)
    assert result.status is GeometryValidationStatus.UNVERIFIED
    assert result.may_draw_overlay is True


def test_invalid_geometry_is_preserved_but_not_drawn(visible_pdf: Path) -> None:
    result = _validate(_evidence(box=_box(72, 706, 900, 740)), visible_pdf)
    assert result.original_geometry is not None
    assert result.original_geometry["x1"] == 900
    assert result.status is GeometryValidationStatus.OUT_OF_BOUNDS
    assert result.may_draw_overlay is False


def test_validation_result_is_replayable_and_policy_versioned(visible_pdf: Path) -> None:
    first = _validate(_evidence(box=_box()), visible_pdf)
    second = _validate(_evidence(box=_box()), visible_pdf)
    assert first == second
    assert first.validation_policy_version == GEOMETRY_VALIDATION_POLICY_VERSION
    assert first.model_dump()["validation_policy_version"] == GEOMETRY_VALIDATION_POLICY_VERSION


def test_real_failing_batch_pattern_crop_local_coordinates_regression(visible_pdf: Path) -> None:
    evidence = _evidence(
        box=_box(10, 10, 80, 40),
        metadata={
            "coordinate_units": "pixels",
            "coordinate_origin": "top_left",
            "coordinate_space": "crop_local",
            "page_width": 100,
            "page_height": 60,
        },
    )
    result = _validate(evidence, visible_pdf)
    assert result.status is GeometryValidationStatus.INCOMPATIBLE_COORDINATE_SPACE
    assert result.reason_code == "CROP_LOCAL_WITHOUT_PAGE_TRANSFORM"


def test_packet_evidence_remains_immutable_after_validation(visible_pdf: Path) -> None:
    evidence = _evidence(box=_box())
    packet = ReviewPacket(
        semantic_slot_id="slot-1",
        document_ref="doc-1",
        canonical_observation_id="canonical-1",
        observation_type=ObservationType.HEADING,
        archive_object_ref="archive-1",
        current_value="Municipal Record Heading",
        candidates=(_candidate(evidence),),
        agreement=AgreementView(
            classification=ComparisonClassification.UNCORROBORATED_SINGLE_SOURCE,
            independent_source_count=1,
        ),
        comparison_confidence=ComparisonConfidence(
            classification=ComparisonClassification.UNCORROBORATED_SINGLE_SOURCE,
            magnitude=None,
            basis="one provider attempted this slot",
        ),
        canonical_confidence=None,
        disclosure_tier=DisclosureTier.TIER_1_SIMPLE,
        review_reason=ReviewReason.SINGLE_SOURCE,
    )
    before = packet.model_dump()
    _validate(evidence, visible_pdf)
    assert packet.model_dump() == before
