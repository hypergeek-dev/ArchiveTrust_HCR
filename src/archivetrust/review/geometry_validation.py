"""Deterministic validation for review evidence geometry.

Validation is a derived, replayable artifact. It never mutates Evidence and it never repairs a
bounding box. The review viewer uses the result to decide whether a box is safe to draw.
"""

from __future__ import annotations

import math
import re
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from archivetrust.domain.evidence.models import BoundingBox, Precision
from archivetrust.infrastructure.rendering.pdf_renderer import (
    DEFAULT_SCALE,
    PdfRenderError,
    pdf_page_count,
    render_pdf_page,
)
from archivetrust.review.packet import CandidateView, EvidenceView

GEOMETRY_VALIDATION_POLICY_VERSION = 1
_KNOWN_ORIGINS = {"top_left", "bottom_left"}
_KNOWN_UNITS = {"pdf_points", "pixels"}


class GeometryValidationStatus(str, Enum):
    PRECISE_VALID = "precise_valid"
    COARSE_VALID = "coarse_valid"
    MISSING = "missing"
    WHOLE_PAGE = "whole_page"
    PAGE_MISMATCH = "page_mismatch"
    OUT_OF_BOUNDS = "out_of_bounds"
    INVALID_DIMENSIONS = "invalid_dimensions"
    INCOMPATIBLE_COORDINATE_SPACE = "incompatible_coordinate_space"
    SPATIALLY_IMPLAUSIBLE = "spatially_implausible"
    UNVERIFIED = "unverified"


class GeometryValidationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: GeometryValidationStatus
    reason_code: str
    explanation: str
    original_geometry: dict[str, Any] | None
    source_dimensions: tuple[float, float] | None
    rendered_page_dimensions: tuple[float, float] | None
    transformed_rectangle: tuple[float, float, float, float] | None
    validation_policy_version: int = GEOMETRY_VALIDATION_POLICY_VERSION
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    may_draw_overlay: bool


def validate_evidence_geometry(
    *,
    candidate: CandidateView,
    evidence: EvidenceView,
    archive_path: Path | None,
    current_page: int | None = None,
    check_spatial_support: bool = False,
) -> GeometryValidationResult:
    box = evidence.bounding_box
    metadata = evidence.coordinate_metadata or {}
    if box is None:
        if metadata.get("whole_page") is True or metadata.get("geometry_scope") == "whole_page":
            return _result(
                GeometryValidationStatus.WHOLE_PAGE,
                "GEOMETRY_WHOLE_PAGE",
                "This reading was reported for the whole page, not for a located region.",
                evidence=evidence,
                box=None,
                source_dimensions=None,
                rendered_dimensions=None,
                transformed=None,
                may_draw=False,
            )
        return _result(
            GeometryValidationStatus.MISSING,
            "GEOMETRY_MISSING",
            "No source position was recorded for this reading.",
            evidence=evidence,
            box=None,
            source_dimensions=None,
            rendered_dimensions=None,
            transformed=None,
            may_draw=False,
        )

    if evidence.provider_id != candidate.provider_id:
        return _invalid(
            "ASSOCIATION_PROVIDER_MISMATCH",
            "The geometry provider does not match the candidate provider.",
            evidence=evidence,
            box=box,
        )
    recorded_observation = metadata.get("observation_id")
    if recorded_observation is not None and recorded_observation != candidate.observation_id:
        return _invalid(
            "ASSOCIATION_OBSERVATION_MISMATCH",
            "The geometry is linked to a different observation than the candidate being reviewed.",
            evidence=evidence,
            box=box,
            diagnostics={"recorded_observation_id": recorded_observation, "candidate_observation_id": candidate.observation_id},
        )

    if evidence.page is None:
        return _invalid("PAGE_MISSING", "The geometry does not name a source page.", evidence=evidence, box=box)
    if current_page is not None and evidence.page != current_page:
        return _page_mismatch(evidence=evidence, box=box, diagnostics={"current_page": current_page})

    rendered_dimensions = None
    if archive_path is not None and archive_path.exists():
        try:
            page_count = pdf_page_count(archive_path)
        except PdfRenderError as exc:
            return _unverified(
                "SOURCE_PAGE_COUNT_UNAVAILABLE",
                "The source page count could not be verified.",
                evidence=evidence,
                box=box,
                diagnostics={"error": str(exc)},
            )
        if not (1 <= evidence.page <= page_count):
            return _page_mismatch(
                evidence=evidence,
                box=box,
                diagnostics={"page_count": page_count},
            )
        try:
            rendered = render_pdf_page(archive_path, evidence.page)
        except PdfRenderError as exc:
            return _unverified(
                "RENDERED_PAGE_UNAVAILABLE",
                "The source page could not be rendered for geometry validation.",
                evidence=evidence,
                box=box,
                diagnostics={"error": str(exc)},
            )
        rendered_dimensions = (rendered.page_width_pt, rendered.page_height_pt)

    if not _finite_box(box):
        return _invalid("BOX_NON_FINITE", "The recorded source position contains a non-finite coordinate.", evidence=evidence, box=box)
    if box.x0 >= box.x1 or box.y0 >= box.y1:
        return _invalid("BOX_INVALID_EXTENT", "The recorded source position has no positive area.", evidence=evidence, box=box)

    if _is_crop_local(metadata):
        return _result(
            GeometryValidationStatus.INCOMPATIBLE_COORDINATE_SPACE,
            "CROP_LOCAL_WITHOUT_PAGE_TRANSFORM",
            "The recorded position is crop-local and has no transform back to page coordinates.",
            evidence=evidence,
            box=box,
            source_dimensions=_source_dimensions(metadata, rendered_dimensions),
            rendered_dimensions=rendered_dimensions,
            transformed=None,
            may_draw=False,
        )

    origin = metadata.get("coordinate_origin")
    if origin is None:
        origin = "top_left"
        origin_inferred = True
    else:
        origin_inferred = False
    if origin not in _KNOWN_ORIGINS:
        return _result(
            GeometryValidationStatus.INCOMPATIBLE_COORDINATE_SPACE,
            "UNKNOWN_COORDINATE_ORIGIN",
            "The coordinate origin is not known.",
            evidence=evidence,
            box=box,
            source_dimensions=_source_dimensions(metadata, rendered_dimensions),
            rendered_dimensions=rendered_dimensions,
            transformed=None,
            may_draw=False,
        )

    units = metadata.get("coordinate_units")
    units_inferred = False
    if units is None:
        units = "pdf_points" if rendered_dimensions is not None else "pixels"
        units_inferred = True
    if units not in _KNOWN_UNITS:
        return _result(
            GeometryValidationStatus.INCOMPATIBLE_COORDINATE_SPACE,
            "UNKNOWN_COORDINATE_UNITS",
            "The coordinate units are not known.",
            evidence=evidence,
            box=box,
            source_dimensions=_source_dimensions(metadata, rendered_dimensions),
            rendered_dimensions=rendered_dimensions,
            transformed=None,
            may_draw=False,
        )

    source_dimensions = _source_dimensions(metadata, rendered_dimensions)
    if source_dimensions is None:
        return _result(
            GeometryValidationStatus.INVALID_DIMENSIONS,
            "SOURCE_DIMENSIONS_MISSING",
            "The source coordinate dimensions are missing.",
            evidence=evidence,
            box=box,
            source_dimensions=None,
            rendered_dimensions=rendered_dimensions,
            transformed=None,
            may_draw=False,
        )
    source_width, source_height = source_dimensions
    if not (_positive_finite(source_width) and _positive_finite(source_height)):
        return _result(
            GeometryValidationStatus.INVALID_DIMENSIONS,
            "SOURCE_DIMENSIONS_INVALID",
            "The source coordinate dimensions are not positive finite values.",
            evidence=evidence,
            box=box,
            source_dimensions=source_dimensions,
            rendered_dimensions=rendered_dimensions,
            transformed=None,
            may_draw=False,
        )
    if box.x0 < 0 or box.y0 < 0 or box.x1 > source_width or box.y1 > source_height:
        return _result(
            GeometryValidationStatus.OUT_OF_BOUNDS,
            "BOX_OUTSIDE_SOURCE_DIMENSIONS",
            "The recorded source position falls outside the declared source page dimensions.",
            evidence=evidence,
            box=box,
            source_dimensions=source_dimensions,
            rendered_dimensions=rendered_dimensions,
            transformed=None,
            may_draw=False,
        )

    transformed = _transform_box(
        box,
        origin=origin,
        source_dimensions=source_dimensions,
        rendered_dimensions=rendered_dimensions,
        units=units,
    )
    if rendered_dimensions is not None and transformed is not None:
        x, y, width, height = transformed
        rendered_width, rendered_height = rendered_dimensions
        rendered_width *= DEFAULT_SCALE
        rendered_height *= DEFAULT_SCALE
        if x < 0 or y < 0 or x + width > rendered_width or y + height > rendered_height:
            return _result(
                GeometryValidationStatus.OUT_OF_BOUNDS,
                "TRANSFORMED_RECTANGLE_OUTSIDE_RENDERED_PAGE",
                "The transformed source position falls outside the rendered page.",
                evidence=evidence,
                box=box,
                source_dimensions=source_dimensions,
                rendered_dimensions=rendered_dimensions,
                transformed=transformed,
                may_draw=False,
            )

    diagnostics = {
        "coordinate_origin": origin,
        "coordinate_origin_inferred": origin_inferred,
        "coordinate_units": units,
        "coordinate_units_inferred": units_inferred,
    }
    if check_spatial_support:
        spatial = _spatial_support(
            archive_path=archive_path,
            evidence=evidence,
            candidate_text=candidate.value,
            box=box,
            origin=origin,
            units=units,
            source_dimensions=source_dimensions,
        )
        diagnostics["spatial_support"] = spatial
        if spatial["status"] == "implausible":
            return _result(
                GeometryValidationStatus.SPATIALLY_IMPLAUSIBLE,
                "TEXT_NOT_SUPPORTED_IN_REGION",
                "The recorded source position contains no meaningful textual support for this reading.",
                evidence=evidence,
                box=box,
                source_dimensions=source_dimensions,
                rendered_dimensions=rendered_dimensions,
                transformed=transformed,
                may_draw=False,
                diagnostics=diagnostics,
            )
        if spatial["status"] == "unverified":
            return _result(
                GeometryValidationStatus.UNVERIFIED,
                spatial["reason_code"],
                "Structural geometry is valid, but spatial text support could not be verified.",
                evidence=evidence,
                box=box,
                source_dimensions=source_dimensions,
                rendered_dimensions=rendered_dimensions,
                transformed=transformed,
                may_draw=True,
                diagnostics=diagnostics,
            )

    status = GeometryValidationStatus.COARSE_VALID if box.precision is Precision.COARSE_ESTIMATE else GeometryValidationStatus.PRECISE_VALID
    reason = "COARSE_GEOMETRY_VALID" if status is GeometryValidationStatus.COARSE_VALID else "PRECISE_GEOMETRY_VALID"
    explanation = (
        "The recorded source position is structurally valid but approximate."
        if status is GeometryValidationStatus.COARSE_VALID
        else "The recorded source position is structurally valid."
    )
    return _result(
        status,
        reason,
        explanation,
        evidence=evidence,
        box=box,
        source_dimensions=source_dimensions,
        rendered_dimensions=rendered_dimensions,
        transformed=transformed,
        may_draw=True,
        diagnostics=diagnostics,
    )


def _transform_box(
    box: BoundingBox,
    *,
    origin: str,
    source_dimensions: tuple[float, float],
    rendered_dimensions: tuple[float, float] | None,
    units: str,
) -> tuple[float, float, float, float]:
    source_width, source_height = source_dimensions
    if origin == "bottom_left":
        y0 = source_height - box.y1
        y1 = source_height - box.y0
    else:
        y0, y1 = box.y0, box.y1
    scale = _render_scale(units=units, source_dimensions=source_dimensions, rendered_dimensions=rendered_dimensions)
    return (
        box.x0 * scale,
        y0 * scale,
        (box.x1 - box.x0) * scale,
        (y1 - y0) * scale,
    )


def _render_scale(
    *,
    units: str,
    source_dimensions: tuple[float, float],
    rendered_dimensions: tuple[float, float] | None,
) -> float:
    if rendered_dimensions is None:
        return 1.0
    source_width, _source_height = source_dimensions
    rendered_width, _rendered_height = rendered_dimensions
    if units == "pdf_points":
        return DEFAULT_SCALE
    if source_width:
        return (rendered_width * DEFAULT_SCALE) / source_width
    return 1.0


def _spatial_support(
    *,
    archive_path: Path | None,
    evidence: EvidenceView,
    candidate_text: str | None,
    box: BoundingBox,
    origin: str,
    units: str,
    source_dimensions: tuple[float, float],
) -> dict[str, Any]:
    if archive_path is None or not archive_path.exists():
        return {"status": "unverified", "reason_code": "SOURCE_DOCUMENT_UNAVAILABLE"}
    if not candidate_text:
        return {"status": "unverified", "reason_code": "CANDIDATE_TEXT_UNAVAILABLE"}
    if units != "pdf_points":
        return {"status": "unverified", "reason_code": "TEXT_EXTRACTION_REQUIRES_PDF_POINT_GEOMETRY"}
    try:
        import pypdfium2 as pdfium
    except ModuleNotFoundError:
        return {"status": "unverified", "reason_code": "PDF_TEXT_EXTRACTION_UNAVAILABLE"}
    try:
        pdf = pdfium.PdfDocument(str(archive_path))
        try:
            if evidence.page is None or not (1 <= evidence.page <= len(pdf)):
                return {"status": "unverified", "reason_code": "TEXT_EXTRACTION_PAGE_UNAVAILABLE"}
            page = pdf[evidence.page - 1]
            try:
                textpage = page.get_textpage()
                try:
                    left, bottom, right, top = _pdf_bounded_rect(box, origin=origin, source_dimensions=source_dimensions)
                    region_text = textpage.get_text_bounded(left, bottom, right, top)
                finally:
                    textpage.close()
            finally:
                page.close()
        finally:
            pdf.close()
    except Exception as exc:  # noqa: BLE001 -- diagnostic only.
        return {"status": "unverified", "reason_code": "PDF_TEXT_EXTRACTION_FAILED", "error": str(exc)}
    score = _text_support_score(candidate_text, region_text)
    status = "supported" if score >= 0.35 else "implausible"
    return {
        "status": status,
        "reason_code": "TEXT_SUPPORT_SCORE",
        "score": score,
        "candidate_text": candidate_text,
        "region_text": region_text,
    }


def _pdf_bounded_rect(
    box: BoundingBox, *, origin: str, source_dimensions: tuple[float, float]
) -> tuple[float, float, float, float]:
    _width, height = source_dimensions
    if origin == "bottom_left":
        return box.x0, box.y0, box.x1, box.y1
    return box.x0, height - box.y1, box.x1, height - box.y0


def _text_support_score(candidate_text: str, region_text: str) -> float:
    candidate_tokens = set(_tokens(candidate_text))
    region_tokens = set(_tokens(region_text))
    if not candidate_tokens or not region_tokens:
        return 0.0
    return len(candidate_tokens & region_tokens) / len(candidate_tokens)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _source_dimensions(
    metadata: dict[str, Any], rendered_dimensions: tuple[float, float] | None
) -> tuple[float, float] | None:
    width = metadata.get("page_width", metadata.get("pixel_width"))
    height = metadata.get("page_height", metadata.get("pixel_height"))
    if width is None or height is None:
        return rendered_dimensions
    try:
        return float(width), float(height)
    except (TypeError, ValueError):
        return None


def _finite_box(box: BoundingBox) -> bool:
    return all(math.isfinite(value) for value in (box.x0, box.y0, box.x1, box.y1))


def _positive_finite(value: float) -> bool:
    return math.isfinite(value) and value > 0


def _is_crop_local(metadata: dict[str, Any]) -> bool:
    space = metadata.get("coordinate_space") or metadata.get("coordinate_units")
    if space in {"crop_local", "crop_pixels"}:
        return not metadata.get("page_transform")
    return bool(metadata.get("crop_local")) and not metadata.get("page_transform")


def _invalid(reason_code: str, explanation: str, *, evidence: EvidenceView, box: BoundingBox, diagnostics: dict[str, Any] | None = None) -> GeometryValidationResult:
    return _result(
        GeometryValidationStatus.INCOMPATIBLE_COORDINATE_SPACE,
        reason_code,
        explanation,
        evidence=evidence,
        box=box,
        source_dimensions=None,
        rendered_dimensions=None,
        transformed=None,
        may_draw=False,
        diagnostics=diagnostics,
    )


def _page_mismatch(*, evidence: EvidenceView, box: BoundingBox, diagnostics: dict[str, Any]) -> GeometryValidationResult:
    return _result(
        GeometryValidationStatus.PAGE_MISMATCH,
        "PAGE_MISMATCH",
        "The recorded source position refers to a different or unavailable page.",
        evidence=evidence,
        box=box,
        source_dimensions=None,
        rendered_dimensions=None,
        transformed=None,
        may_draw=False,
        diagnostics=diagnostics,
    )


def _unverified(reason_code: str, explanation: str, *, evidence: EvidenceView, box: BoundingBox, diagnostics: dict[str, Any]) -> GeometryValidationResult:
    return _result(
        GeometryValidationStatus.UNVERIFIED,
        reason_code,
        explanation,
        evidence=evidence,
        box=box,
        source_dimensions=None,
        rendered_dimensions=None,
        transformed=None,
        may_draw=True,
        diagnostics=diagnostics,
    )


def _result(
    status: GeometryValidationStatus,
    reason_code: str,
    explanation: str,
    *,
    evidence: EvidenceView,
    box: BoundingBox | None,
    source_dimensions: tuple[float, float] | None,
    rendered_dimensions: tuple[float, float] | None,
    transformed: tuple[float, float, float, float] | None,
    may_draw: bool,
    diagnostics: dict[str, Any] | None = None,
) -> GeometryValidationResult:
    original = None
    if box is not None:
        original = {
            "x0": box.x0,
            "y0": box.y0,
            "x1": box.x1,
            "y1": box.y1,
            "precision": box.precision.value,
            "page": evidence.page,
            "coordinate_metadata": dict(evidence.coordinate_metadata or {}),
        }
    return GeometryValidationResult(
        status=status,
        reason_code=reason_code,
        explanation=explanation,
        original_geometry=original,
        source_dimensions=source_dimensions,
        rendered_page_dimensions=rendered_dimensions,
        transformed_rectangle=transformed,
        diagnostics=diagnostics or {},
        may_draw_overlay=may_draw,
    )
