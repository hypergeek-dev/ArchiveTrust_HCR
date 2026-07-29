"""Rendered pages as reproducible derived artifacts (Multi-Provider Activation milestone).

A rendered page is a deterministic function of (Archive Object content hash, page number, render
scale, renderer id+version) -- the canonical case `WorkspaceLayout.derived_dir` already existed for
and had, until now, never been used (`storage_usage().derived_bytes` was always 0). Caching here
means the Review Center and every image-input provider share one render per page instead of each
re-rasterizing; deleting `derived/` and reprocessing reconstructs it bit-identically for a pinned
renderer version -- never a second source of truth, always reproducible from the immutable archive.

**Path naming, deliberately hyphen-disciplined**: the cache key (content hash, scale, renderer)
lives in the *directory* name, using only alphanumerics and underscores; the *filename* is always
exactly `page-{page_no}.png` (plus a `.json` sidecar). This is what lets a provider recover its page
number from `Path(source).stem.rsplit("-", 1)[-1]` unambiguously -- the only hyphen anywhere in the
full path is the one in "page-N", regardless of what a renderer id/version string might contain. A
renderer upgrade changes the directory name, so old cached pages for a prior renderer version are
never overwritten or reinterpreted; they simply age out of use.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from archivetrust.infrastructure.rendering.pdf_renderer import (
    DEFAULT_SCALE,
    PdfRenderError,
    render_pdf_page,
)


class RenderedPageArtifact(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    path: Path
    page_no: int
    scale: float
    renderer_id: str
    renderer_version: str
    page_width_pt: float
    page_height_pt: float
    pixel_width: int
    pixel_height: int
    cache_hit: bool


def _safe_tag(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value)


def _scale_tag(scale: float) -> str:
    return _safe_tag(f"{scale:.2f}")


def _page_dir(derived_dir: Path, content_hash: str, *, scale: float, renderer_id: str, renderer_version: str) -> Path:
    renderer_tag = f"{_safe_tag(renderer_id)}_{_safe_tag(renderer_version)}"
    return derived_dir / content_hash / f"r_{renderer_tag}_s_{_scale_tag(scale)}"


def read_render_sidecar(image_path: Path) -> dict[str, Any]:
    """Reads the renderer-provenance sidecar next to a rendered page image, if one exists --
    what a provider adapter merges into its Evidence's `supporting_metadata` (Priority: rendering
    provenance) without needing to know anything about how the image was produced. `{}` when no
    sidecar exists (the image came from some other source, or wasn't produced by this cache).
    """
    sidecar = image_path.with_suffix(".json")
    if not sidecar.exists():
        return {}
    try:
        return json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


class PageRenderCache:
    """Bound to one Workspace's `derived/` directory. `get_or_render` is the only entry point:
    reuses an existing cached page when present, else rasterizes and caches it.
    """

    def __init__(self, derived_dir: Path) -> None:
        self._derived_dir = derived_dir

    def get_or_render(
        self, *, content_hash: str, pdf_path: Path, page_no: int, scale: float = DEFAULT_SCALE
    ) -> RenderedPageArtifact:
        from archivetrust.infrastructure.rendering.pdf_renderer import RENDERER_ID, renderer_version

        version = renderer_version()
        page_dir = _page_dir(
            self._derived_dir, content_hash, scale=scale, renderer_id=RENDERER_ID, renderer_version=version
        )
        png_path = page_dir / f"page-{page_no}.png"
        json_path = png_path.with_suffix(".json")

        if png_path.exists() and json_path.exists():
            metadata = json.loads(json_path.read_text(encoding="utf-8"))
            return RenderedPageArtifact(path=png_path, cache_hit=True, **metadata)

        rendered = render_pdf_page(pdf_path, page_no, scale=scale)
        page_dir.mkdir(parents=True, exist_ok=True)
        rendered.image.save(png_path)
        metadata = {
            "page_no": rendered.page_no,
            "scale": rendered.scale,
            "renderer_id": rendered.renderer_id,
            "renderer_version": rendered.renderer_version,
            "page_width_pt": rendered.page_width_pt,
            "page_height_pt": rendered.page_height_pt,
            "pixel_width": rendered.pixel_width,
            "pixel_height": rendered.pixel_height,
        }
        json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return RenderedPageArtifact(path=png_path, cache_hit=False, **metadata)
