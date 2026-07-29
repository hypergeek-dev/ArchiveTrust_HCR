"""The shared PDF-page rasterization core (Multi-Provider Activation milestone).

`pypdfium2 -> PIL.Image` only -- no Qt anywhere in this module, so it is importable from
`providers/*`, `acquisition/*`, and any headless context (verified: this is exactly the path the
Vision Provider Activation investigation validated headless, independent of the desktop client).
The desktop's `clients/desktop/pdf_render.py` wraps this module's output in a `QImage`; it does not
duplicate the rasterization logic.

**Determinism note**, restated from the investigation: rendering is deterministic for a pinned
(pypdfium2 version, scale) -- not guaranteed bit-identical across pdfium *versions*. That is exactly
why `renderer_version()` is recorded as provenance wherever a render feeds a provider (see
`render_cache.py`), rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from PIL import Image

RENDERER_ID = "pypdfium2"
DEFAULT_SCALE = 2.0
"""Pixels per PDF point -- high enough that overlay text and OCR both stay legible; matches the
desktop Review Center's prior `RENDER_SCALE` constant."""


class PdfRenderError(Exception):
    """Raised when a page could not be rasterized or a page count could not be determined --
    callers fall back to an honestly-labeled degraded state (never a silent blank/empty result;
    Constitution Article 18 applied to rendering)."""


@dataclass(frozen=True)
class RenderedPage:
    """One rasterized page: the image plus everything needed to map a provider's bounding boxes
    back onto it, and to record renderer provenance on the Evidence that cites it."""

    image: Image.Image
    page_no: int
    page_width_pt: float
    page_height_pt: float
    scale: float
    renderer_id: str
    renderer_version: str

    @property
    def pixel_width(self) -> int:
        return self.image.width

    @property
    def pixel_height(self) -> int:
        return self.image.height


def renderer_version() -> str:
    try:
        return version("pypdfium2")
    except PackageNotFoundError:  # pragma: no cover -- pypdfium2 is a pinned dependency
        return "unknown"


def _open_document(path: Path) -> Any:
    try:
        import pypdfium2 as pdfium
    except ModuleNotFoundError as exc:
        raise PdfRenderError("pypdfium2 is not installed") from exc
    try:
        return pdfium.PdfDocument(str(path))
    except Exception as exc:  # noqa: BLE001 -- any pdfium open failure becomes one honest error type
        raise PdfRenderError(f"could not open {path.name} as a PDF: {exc}") from exc


def pdf_page_count(path: Path) -> int:
    document = _open_document(path)
    try:
        return len(document)
    finally:
        document.close()


def render_pdf_page(path: Path, page_no: int, *, scale: float = DEFAULT_SCALE) -> RenderedPage:
    """Rasterizes one 1-indexed PDF page. Raises `PdfRenderError` for any failure -- a missing
    file, an unreadable PDF, or an out-of-range page -- never returns a partial/blank result.
    """
    document = _open_document(path)
    try:
        if not (1 <= page_no <= len(document)):
            raise PdfRenderError(f"page {page_no} out of range for {path.name} ({len(document)} pages)")
        page = document[page_no - 1]
        try:
            width_pt, height_pt = page.get_size()
            bitmap = page.render(scale=scale)
            try:
                pil_image = bitmap.to_pil().convert("RGB")
            finally:
                bitmap.close()
        finally:
            page.close()
    except PdfRenderError:
        raise
    except Exception as exc:  # noqa: BLE001 -- any other pdfium failure becomes the same honest error
        raise PdfRenderError(f"could not render {path.name} page {page_no}: {exc}") from exc
    finally:
        document.close()

    return RenderedPage(
        image=pil_image,
        page_no=page_no,
        page_width_pt=float(width_pt),
        page_height_pt=float(height_pt),
        scale=scale,
        renderer_id=RENDERER_ID,
        renderer_version=renderer_version(),
    )
