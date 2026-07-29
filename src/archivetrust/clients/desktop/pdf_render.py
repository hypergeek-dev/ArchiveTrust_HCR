"""Desktop-only `QImage` wrapper over the shared PDF renderer (Multi-Provider Activation milestone).

The actual rasterization (`pypdfium2` -> `PIL.Image`) lives in
`infrastructure.rendering.pdf_renderer` -- Qt-free, headless, shared with the provider adapters
that now also render pages. This module's only job is the final `PIL.Image` -> `QImage` conversion
the Review Center needs; it duplicates no rendering logic.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QImage

from archivetrust.infrastructure.rendering.pdf_renderer import (
    DEFAULT_SCALE as RENDER_SCALE,
    PdfRenderError,
    render_pdf_page as _render_pdf_page,
)

__all__ = ["RENDER_SCALE", "PageRenderError", "render_pdf_page"]

PageRenderError = PdfRenderError
"""Kept as a distinct name for backward compatibility with existing Review Center imports."""


def render_pdf_page(path: Path, page_no: int, *, scale: float = RENDER_SCALE) -> tuple[QImage, float, float]:
    """Returns `(image, page_width_points, page_height_points)` for one 1-indexed PDF page --
    same signature as before this module delegated to the shared renderer, so callers
    (`ReviewCenterView`) need no changes.
    """
    rendered = _render_pdf_page(path, page_no, scale=scale)
    data = rendered.image.tobytes("raw", "RGB")
    image = QImage(data, rendered.pixel_width, rendered.pixel_height, rendered.pixel_width * 3, QImage.Format.Format_RGB888)
    image = image.copy()  # own the buffer -- `data`/the PIL image go out of scope after return
    return image, rendered.page_width_pt, rendered.page_height_pt
