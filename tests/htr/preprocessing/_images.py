"""Synthetic multi-mode test images for the RGB-normalization stage.

**Provenance: every image built here is synthesized in-process by this module.** None is a real
archival scan, and none is copied from anywhere. They exist to exercise *colour-representation*
handling -- one image per Pillow input mode the stage must support -- which needs no real handwriting:
what is under test is that a CMYK page becomes RGB with its geometry intact, not that anything can be
read off it.

The one real image in play is `tests/fixtures/htr/trolldomskommissionen_sample_line.jpg` (a genuine
line crop from Riksarkivet's trolldomskommissionen_lines dataset -- see `tests/fixtures/htr/README.md`
for its provenance), used by `test_demonstration_run.py` as a stand-in *page*. It is a line crop, not
a page, and that test says so; it is used because it is the only real historical-document image this
repository contains, and demonstrating the stage on a real archival scan is more honest than
demonstrating it only on synthetic swatches.
"""

from __future__ import annotations

import io

from PIL import Image

WIDTH = 12
HEIGHT = 8


def encode(image: Image.Image, image_format: str = "PNG", **params) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=image_format, **params)
    return buffer.getvalue()


def bilevel() -> bytes:
    """Mode `1`: 1 bit per pixel. The classic scanned-microfilm representation."""
    image = Image.new("1", (WIDTH, HEIGHT), 0)
    for x in range(WIDTH // 2):
        image.putpixel((x, 0), 1)
    return encode(image)


def grayscale() -> bytes:
    """Mode `L`: 8-bit grayscale."""
    image = Image.new("L", (WIDTH, HEIGHT), 128)
    image.putpixel((0, 0), 0)
    return encode(image)


def grayscale_alpha() -> bytes:
    """Mode `LA`: grayscale plus a real alpha channel, fully transparent so compositing is visible."""
    return encode(Image.new("LA", (WIDTH, HEIGHT), (64, 0)))


def palette() -> bytes:
    """Mode `P`: palette, opaque. Palette entry 1 is a saturated red."""
    image = Image.new("P", (WIDTH, HEIGHT), 1)
    image.putpalette([0, 0, 0, 220, 30, 40] + [0, 0, 0] * 254)
    return encode(image)


def palette_with_transparency() -> bytes:
    """Mode `P` carrying a `transparency` index rather than an alpha channel -- transparency that a
    naive `convert("RGB")` silently discards, turning transparent pixels into whatever palette colour
    happened to sit at that index."""
    image = Image.new("P", (WIDTH, HEIGHT), 0)
    image.putpalette([17, 220, 33] + [0, 0, 0] * 255)
    return encode(image, transparency=0)


def cmyk() -> bytes:
    """Mode `CMYK`: four channels, as produced by print-oriented scanning workflows. Written as TIFF
    because PNG cannot hold CMYK -- which is itself part of what makes this case worth testing."""
    return encode(Image.new("CMYK", (WIDTH, HEIGHT), (0, 80, 80, 0)), "TIFF")


def rgb() -> bytes:
    """Mode `RGB`: already conforming. Still re-encoded to canonical form by the stage."""
    return encode(Image.new("RGB", (WIDTH, HEIGHT), (10, 20, 30)))


def rgba() -> bytes:
    """Mode `RGBA`: a fully transparent red, so a white composite is unmistakable (a wrong composite
    over black, or a discarded alpha, both produce visibly different pixels)."""
    return encode(Image.new("RGBA", (WIDTH, HEIGHT), (200, 0, 0, 0)))


def grayscale_16bit() -> bytes:
    """Mode `I;16`: 16-bit grayscale, as produced by high-bit-depth archival scanners.

    Deliberately filled with mid-to-high values *above 255*. Pillow's own `convert("L")` clips rather
    than scales, so on this input a naive implementation turns 4096, 32768 and 65535 all into 255 --
    most of the page becomes pure white. See `test_input_modes.py::test_16_bit_grayscale_is_scaled...`.
    """
    image = Image.new("I;16", (WIDTH, HEIGHT))
    image.putdata([0, 4096, 32768, 65535] * ((WIDTH * HEIGHT) // 4))
    return encode(image)


def rgb_with_icc_profile() -> bytes:
    """An RGB image carrying an embedded ICC profile, to exercise record-then-strip."""
    return encode(Image.new("RGB", (WIDTH, HEIGHT), (90, 90, 90)), icc_profile=_MINIMAL_ICC)


def rgb_with_exif_orientation(orientation: int) -> bytes:
    """An RGB image whose EXIF `Orientation` tag is set, written as JPEG (the format that actually
    carries EXIF in practice).

    Deliberately non-square (`WIDTH != HEIGHT`) so an orientation that swaps axes produces a genuinely
    different geometry -- a square test image would let an unapplied rotation pass unnoticed.
    """
    image = Image.new("RGB", (WIDTH, HEIGHT), (200, 100, 50))
    exif = image.getexif()
    exif[274] = orientation
    return encode(image, "JPEG", exif=exif, quality=95)


def undecodable() -> bytes:
    """Bytes that are not an image in any format -- the `UNDECODABLE_IMAGE` failure case."""
    return b"this is not an image, it is a sentence about not being one"


def _build_minimal_icc() -> bytes:
    """A structurally valid, minimal ICC v2 profile with a readable `desc` tag.

    Hand-built rather than shipped as a binary fixture: it needs to be inspectable, and its only job
    is to be *present and identifiable* so the stage can record its hash and description before
    stripping it. It is not a colorimetrically meaningful profile and is never applied to pixels
    (`IccProfilePolicy.RECORD_AND_STRIP_WITHOUT_APPLYING`), so its transform tables are irrelevant.
    """
    description = b"ArchiveTrust synthetic test profile\x00"
    desc_tag = b"desc" + b"\x00" * 4 + len(description).to_bytes(4, "big") + description
    tag_count = 1
    header_size = 128
    table_size = 4 + tag_count * 12
    desc_offset = header_size + table_size
    total = desc_offset + len(desc_tag)

    header = bytearray(b"\x00" * header_size)
    header[0:4] = total.to_bytes(4, "big")
    header[4:8] = b"ATst"  # preferred CMM
    header[8:12] = (0x02100000).to_bytes(4, "big")  # profile version 2.1
    header[12:16] = b"mntr"
    header[16:20] = b"RGB "
    header[20:24] = b"XYZ "
    header[36:40] = b"acsp"

    table = tag_count.to_bytes(4, "big") + (
        b"desc" + desc_offset.to_bytes(4, "big") + len(desc_tag).to_bytes(4, "big")
    )
    return bytes(header) + table + desc_tag


_MINIMAL_ICC = _build_minimal_icc()

ALL_MODE_BUILDERS = {
    "1": bilevel,
    "L": grayscale,
    "LA": grayscale_alpha,
    "P": palette,
    "P_transparency": palette_with_transparency,
    "CMYK": cmyk,
    "RGB": rgb,
    "RGBA": rgba,
    "I;16": grayscale_16bit,
}
"""Every input mode the stage is required to handle, keyed by a readable case name. Used by the
parametrized conversion test so adding a mode here without a passing conversion is impossible."""
