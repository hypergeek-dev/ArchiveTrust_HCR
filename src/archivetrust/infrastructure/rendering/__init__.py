"""Shared, Qt-free PDF page rendering (Operational Hardening milestone, Priority: multi-provider
activation). Extracted from `clients/desktop/pdf_render.py` so image-input providers
(Tesseract+LayoutParser, Qwen2.5-VL) and the desktop Review Center share one rasterization core --
never two renderers producing potentially different pixels for the same page.
"""

from __future__ import annotations

from archivetrust.infrastructure.rendering.pdf_renderer import (
    RENDERER_ID,
    PdfRenderError,
    RenderedPage,
    pdf_page_count,
    render_pdf_page,
    renderer_version,
)
from archivetrust.infrastructure.rendering.render_cache import (
    RenderedPageArtifact,
    PageRenderCache,
    read_render_sidecar,
)

__all__ = [
    "RENDERER_ID",
    "PdfRenderError",
    "RenderedPage",
    "pdf_page_count",
    "render_pdf_page",
    "renderer_version",
    "RenderedPageArtifact",
    "PageRenderCache",
    "read_render_sidecar",
]
