"""Image probing, perceptual hashing and deterministic line cropping for the benchmark.

Cropping follows the project's existing policy (`florence2_line_detector.crop_lines`): cut from the
full-resolution page as decoded, clamp to the raster, PNG with pinned encoder settings. The page is
cropped from its *stored* raster -- EXIF orientation is never applied silently; a page with a
non-identity orientation tag is flagged at inspection so a human decides which raster the
coordinates refer to.
"""

from __future__ import annotations

import io
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageStat

from archivetrust.htr.benchmark.contract import sha256_bytes

EXIF_ORIENTATION_TAG = 274
PNG_SAVE_KWARGS = {"format": "PNG", "optimize": False, "compress_level": 6}
"""Same pins as `rgb_normalization` and `crop_lines`, so a Pillow default change cannot alter hashes."""

BLANK_STDDEV_THRESHOLD = 2.0
"""Grayscale standard deviation below which an image is flagged as (near-)blank for review."""

PASSTHROUGH_FORMATS = frozenset({"PNG", "JPEG"})
"""Supplied line images in these formats are copied byte-identical; both model stacks decode them."""

LOSSLESS_CONVERTIBLE_MODES = frozenset({"1", "L", "LA", "P", "RGB", "RGBA"})
"""8-bit modes whose conversion to L/RGB PNG loses nothing a recognizer reads (opaque alpha only).
16-bit, CMYK, float and multi-frame images need a human decision."""


@dataclass(frozen=True)
class ImageProbe:
    ok: bool
    sha256: str
    size_bytes: int
    error: str | None = None
    format: str | None = None
    mode: str | None = None
    width: int | None = None
    height: int | None = None
    exif_orientation: int | None = None
    has_alpha: bool = False
    alpha_all_opaque: bool | None = None
    frames: int = 1
    gray_stddev: float | None = None
    dhash: str | None = None

    @property
    def blank(self) -> bool:
        return self.gray_stddev is not None and self.gray_stddev < BLANK_STDDEV_THRESHOLD


def dhash(image: Image.Image, size: int = 8) -> str:
    """64-bit difference hash (hex). Robust to re-encoding and mild rescaling; used only to *flag*
    near-duplicates for review, never to decide identity."""
    small = image.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    pixels = small.tobytes()  # mode L: one byte per pixel
    bits = 0
    for row in range(size):
        for col in range(size):
            left = pixels[row * (size + 1) + col]
            right = pixels[row * (size + 1) + col + 1]
            bits = (bits << 1) | (1 if left > right else 0)
    return f"{bits:0{size * size // 4}x}"


def hamming(a: str, b: str) -> int:
    return (int(a, 16) ^ int(b, 16)).bit_count()


def near_duplicate_pairs(hashes: Sequence[tuple[str, str]], *, max_distance: int = 4) -> list[tuple[str, str, int]]:
    """All pairs of `(key, dhash64)` within `max_distance` bits, found exactly without O(n^2) work:
    split each hash into `max_distance + 1` bands; by pigeonhole any pair within the distance agrees
    on at least one band."""
    bands = max_distance + 1
    width = 64 // bands
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    values = [int(h, 16) for _, h in hashes]
    for index, value in enumerate(values):
        for band in range(bands):
            shift = band * width
            span = width if band < bands - 1 else 64 - shift
            buckets[(band, (value >> shift) & ((1 << span) - 1))].append(index)
    found: dict[tuple[int, int], int] = {}
    for members in buckets.values():
        for i_pos, i in enumerate(members):
            for j in members[i_pos + 1 :]:
                pair = (min(i, j), max(i, j))
                if pair not in found:
                    distance = (values[i] ^ values[j]).bit_count()
                    if distance <= max_distance:
                        found[pair] = distance
    return sorted((hashes[i][0], hashes[j][0], d) for (i, j), d in found.items())


def probe_image(path: Path) -> ImageProbe:
    return probe_image_bytes(path.read_bytes())


def probe_image_bytes(data: bytes) -> ImageProbe:
    digest = sha256_bytes(data)
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
        with Image.open(io.BytesIO(data)) as image:
            fmt, mode, (width, height) = image.format, image.mode, image.size
            frames = getattr(image, "n_frames", 1)
            orientation = image.getexif().get(EXIF_ORIENTATION_TAG)
            image.load()
            has_alpha = "A" in image.getbands() or (mode == "P" and "transparency" in image.info)
            alpha_opaque = None
            if has_alpha:
                alpha = image.convert("RGBA").getchannel("A")
                alpha_opaque = alpha.getextrema()[0] == 255
            gray = image.convert("L")
            gray.thumbnail((512, 512))
            stddev = ImageStat.Stat(gray).stddev[0]
            return ImageProbe(
                ok=True, sha256=digest, size_bytes=len(data), format=fmt, mode=mode, width=width, height=height,
                exif_orientation=orientation, has_alpha=has_alpha, alpha_all_opaque=alpha_opaque, frames=frames,
                gray_stddev=round(stddev, 3), dhash=dhash(image),
            )
    except Image.DecompressionBombError as exc:
        return ImageProbe(ok=False, sha256=digest, size_bytes=len(data), error=f"too_large: {exc}")
    except Exception as exc:  # noqa: BLE001 -- any decode failure is a finding, never a crash
        return ImageProbe(ok=False, sha256=digest, size_bytes=len(data), error=f"{type(exc).__name__}: {exc}")


def encode_png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, **PNG_SAVE_KWARGS)
    return buffer.getvalue()


def open_page_rgb(data: bytes) -> Image.Image:
    """Same decoding as `PageImage.open()`: load fully, convert to RGB, no EXIF transpose."""
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        return image.convert("RGB")


def polygon_bbox(points: Iterable[tuple[float, float]], *, width: int, height: int) -> tuple[int, int, int, int] | None:
    """Rounded, raster-clamped bounding box (x1/y1 exclusive), or None when it is empty after clamping --
    the same rounding and clamping as `crop_lines`."""
    pts = list(points)
    if not pts:
        return None
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    left, top = max(0, int(round(min(xs)))), max(0, int(round(min(ys))))
    right, bottom = min(width, int(round(max(xs)))), min(height, int(round(max(ys))))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def crop_line(page: Image.Image, bbox: tuple[int, int, int, int], *, polygon: Sequence[tuple[float, float]] | None = None,
              mask_polygon: bool = False) -> bytes:
    """`bbox_v1` crop, or `polygon_mask_v1` (outside-polygon pixels set to white) when `mask_polygon`."""
    cut = page.crop(bbox)
    if mask_polygon:
        if not polygon:
            raise ValueError("polygon_mask_v1 needs a polygon")
        x0, y0 = bbox[0], bbox[1]
        mask = Image.new("L", cut.size, 0)
        ImageDraw.Draw(mask).polygon([(x - x0, y - y0) for x, y in polygon], fill=255)
        white = Image.new(cut.mode, cut.size, (255, 255, 255) if cut.mode == "RGB" else 255)
        cut = Image.composite(cut, white, mask)
    return encode_png(cut)


def prepare_supplied_line_image(data: bytes, probe: ImageProbe) -> tuple[bytes, str, str | None]:
    """Returns `(bytes, extension, conversion)`. PNG/JPEG without meaningful alpha are passed through
    byte-identical (`conversion=None`). Anything else is re-encoded losslessly as PNG and the conversion
    is named, so the manifest records it. Images with non-opaque alpha are refused: how transparency
    is flattened would differ between the model stacks, so a human must decide."""
    if not probe.ok:
        raise ValueError(f"unreadable image: {probe.error}")
    if probe.has_alpha and not probe.alpha_all_opaque:
        raise ValueError("image has non-opaque alpha; flattening must be decided by a human")
    if probe.mode not in LOSSLESS_CONVERTIBLE_MODES or probe.frames != 1:
        raise ValueError(f"mode {probe.mode} / {probe.frames} frames has no lossless 8-bit conversion; a human must decide")
    if probe.format in PASSTHROUGH_FORMATS and not probe.has_alpha and probe.mode in {"L", "RGB"}:
        return data, ".jpg" if probe.format == "JPEG" else ".png", None
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        target = "L" if image.mode in {"1", "L", "LA"} else "RGB"
        return encode_png(image.convert(target)), ".png", f"{probe.format}/{probe.mode}->PNG/{target}"
