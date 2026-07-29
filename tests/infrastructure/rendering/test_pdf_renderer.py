"""Regression tests for the shared, Qt-free PDF renderer (Multi-Provider Activation milestone)."""

from __future__ import annotations

import pytest

from archivetrust.infrastructure.rendering.pdf_renderer import (
    PdfRenderError,
    pdf_page_count,
    render_pdf_page,
    renderer_version,
)
from tests.infrastructure.rendering._minimal_pdf import write_minimal_pdf


def test_pdf_page_count_matches_document(tmp_path) -> None:
    pdf = tmp_path / "doc.pdf"
    write_minimal_pdf(pdf, num_pages=3)
    assert pdf_page_count(pdf) == 3


def test_render_pdf_page_returns_pixels_and_point_dimensions(tmp_path) -> None:
    pdf = tmp_path / "doc.pdf"
    write_minimal_pdf(pdf, num_pages=1, width=200, height=300)

    rendered = render_pdf_page(pdf, 1, scale=2.0)

    assert rendered.page_width_pt == 200.0
    assert rendered.page_height_pt == 300.0
    assert rendered.pixel_width == 400  # 200pt * scale 2.0
    assert rendered.pixel_height == 600
    assert rendered.scale == 2.0
    assert rendered.renderer_id == "pypdfium2"
    assert rendered.renderer_version == renderer_version()


def test_render_pdf_page_arbitrary_page_number(tmp_path) -> None:
    pdf = tmp_path / "doc.pdf"
    write_minimal_pdf(pdf, num_pages=5)

    rendered = render_pdf_page(pdf, 4, scale=1.0)

    assert rendered.page_no == 4


def test_render_pdf_page_out_of_range_raises(tmp_path) -> None:
    pdf = tmp_path / "doc.pdf"
    write_minimal_pdf(pdf, num_pages=2)

    with pytest.raises(PdfRenderError):
        render_pdf_page(pdf, 5)


def test_render_pdf_page_unreadable_file_raises(tmp_path) -> None:
    not_a_pdf = tmp_path / "not_a_pdf.pdf"
    not_a_pdf.write_bytes(b"this is not a pdf")

    with pytest.raises(PdfRenderError):
        render_pdf_page(not_a_pdf, 1)


def test_render_pdf_page_is_deterministic(tmp_path) -> None:
    pdf = tmp_path / "doc.pdf"
    write_minimal_pdf(pdf, num_pages=1)

    first = render_pdf_page(pdf, 1, scale=2.0)
    second = render_pdf_page(pdf, 1, scale=2.0)

    assert first.image.tobytes() == second.image.tobytes()
