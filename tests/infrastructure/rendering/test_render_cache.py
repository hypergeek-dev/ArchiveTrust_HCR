"""Regression tests for `PageRenderCache` -- rendered pages as reproducible derived artifacts
(Multi-Provider Activation milestone)."""

from __future__ import annotations

from archivetrust.infrastructure.rendering.pdf_renderer import PdfRenderError
from archivetrust.infrastructure.rendering.render_cache import PageRenderCache, read_render_sidecar
from tests.infrastructure.rendering._minimal_pdf import write_minimal_pdf


def test_get_or_render_writes_a_cached_png_and_sidecar(tmp_path) -> None:
    pdf = tmp_path / "doc.pdf"
    write_minimal_pdf(pdf, num_pages=1, width=200, height=300)
    cache = PageRenderCache(tmp_path / "derived")

    artifact = cache.get_or_render(content_hash="abc123", pdf_path=pdf, page_no=1, scale=2.0)

    assert artifact.path.exists()
    assert artifact.path.with_suffix(".json").exists()
    assert artifact.cache_hit is False
    assert artifact.page_width_pt == 200.0
    assert artifact.pixel_width == 400


def test_get_or_render_reuses_the_cached_artifact(tmp_path) -> None:
    pdf = tmp_path / "doc.pdf"
    write_minimal_pdf(pdf, num_pages=1)
    cache = PageRenderCache(tmp_path / "derived")

    first = cache.get_or_render(content_hash="abc123", pdf_path=pdf, page_no=1)
    assert first.cache_hit is False

    second = cache.get_or_render(content_hash="abc123", pdf_path=pdf, page_no=1)
    assert second.cache_hit is True
    assert second.path == first.path
    assert second.page_width_pt == first.page_width_pt


def test_different_pages_and_scales_get_distinct_paths(tmp_path) -> None:
    pdf = tmp_path / "doc.pdf"
    write_minimal_pdf(pdf, num_pages=2)
    cache = PageRenderCache(tmp_path / "derived")

    page1 = cache.get_or_render(content_hash="abc123", pdf_path=pdf, page_no=1, scale=2.0)
    page2 = cache.get_or_render(content_hash="abc123", pdf_path=pdf, page_no=2, scale=2.0)
    page1_other_scale = cache.get_or_render(content_hash="abc123", pdf_path=pdf, page_no=1, scale=1.0)

    assert page1.path != page2.path
    assert page1.path != page1_other_scale.path


def test_page_filename_lets_page_number_be_recovered_from_the_stem(tmp_path) -> None:
    """The naming discipline the Vision Provider Activation investigation prescribed: the
    *filename* is always exactly "page-N.png" -- renderer id/version/scale tags live only in the
    parent directory name, never in the filename itself -- so a provider's
    `Path(source).stem.rsplit("-", 1)[-1]` recovers the page number unambiguously regardless of
    what those directory-name tags (or, as this test's own tmp_path proves, an ancestor directory
    supplied by the surrounding environment) contain. Parsing the *stem* (filename only, via
    `Path(...).stem`) rather than the raw path string is exactly what makes this robust -- any
    hyphens elsewhere in the full path are irrelevant to `.stem`.
    """
    pdf = tmp_path / "doc.pdf"
    write_minimal_pdf(pdf, num_pages=7)
    cache = PageRenderCache(tmp_path / "derived")

    artifact = cache.get_or_render(content_hash="abc123", pdf_path=pdf, page_no=7, scale=2.0)

    stem = artifact.path.stem
    assert stem == "page-7"
    assert stem.rsplit("-", 1)[-1] == "7"


def test_read_render_sidecar_returns_renderer_provenance(tmp_path) -> None:
    pdf = tmp_path / "doc.pdf"
    write_minimal_pdf(pdf, num_pages=1)
    cache = PageRenderCache(tmp_path / "derived")

    artifact = cache.get_or_render(content_hash="abc123", pdf_path=pdf, page_no=1, scale=2.0)
    sidecar = read_render_sidecar(artifact.path)

    assert sidecar["renderer_id"] == "pypdfium2"
    assert sidecar["scale"] == 2.0
    assert sidecar["page_no"] == 1


def test_read_render_sidecar_missing_returns_empty_dict(tmp_path) -> None:
    image_path = tmp_path / "not_cached.png"
    image_path.write_bytes(b"fake")

    assert read_render_sidecar(image_path) == {}


def test_get_or_render_propagates_render_errors(tmp_path) -> None:
    pdf = tmp_path / "doc.pdf"
    write_minimal_pdf(pdf, num_pages=1)
    cache = PageRenderCache(tmp_path / "derived")

    try:
        cache.get_or_render(content_hash="abc123", pdf_path=pdf, page_no=99)
        assert False, "expected PdfRenderError"
    except PdfRenderError:
        pass
