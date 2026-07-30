"""Every input colour mode the stage must handle becomes 8-bit RGB, with geometry preserved.

One passing test per mode the specification enumerates: bilevel, grayscale, grayscale+alpha, palette,
palette+transparency, CMYK, RGB, RGBA, 16-bit grayscale. Parametrized over
`_images.ALL_MODE_BUILDERS`, so a mode cannot be added to that table without a conversion assertion
running over it.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image

from archivetrust.htr.preprocessing.models import (
    AlphaCompositingPolicy,
    RgbNormalizationConfig,
)
from archivetrust.htr.preprocessing.rgb_normalization import normalize_page_image
from tests.htr.preprocessing import _images


def _decode(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes))


@pytest.mark.parametrize("case_name", sorted(_images.ALL_MODE_BUILDERS))
def test_every_supported_input_mode_becomes_8_bit_rgb(case_name):
    """The core conversion guarantee, for all nine required input modes."""
    result = normalize_page_image(_images.ALL_MODE_BUILDERS[case_name]())

    output = _decode(result.image_bytes)
    assert output.mode == "RGB", f"{case_name} did not become RGB"
    assert len(output.getbands()) == 3, f"{case_name} did not produce exactly 3 channels"
    assert output.getbands() == ("R", "G", "B"), f"{case_name} channel order is not R-G-B"
    assert output.format == "PNG", f"{case_name} was not encoded as lossless PNG"
    # 8 bits per channel: every sample is an int in 0..255, which is what mode RGB guarantees and
    # what `Image.getextrema()` reports per band.
    for low, high in output.getextrema():
        assert 0 <= low <= 255 and 0 <= high <= 255


@pytest.mark.parametrize("case_name", sorted(_images.ALL_MODE_BUILDERS))
def test_every_supported_input_mode_preserves_dimensions(case_name):
    """No resize, no crop, no padding -- for any mode, in or out."""
    result = normalize_page_image(_images.ALL_MODE_BUILDERS[case_name]())

    assert (result.output_width, result.output_height) == (_images.WIDTH, _images.HEIGHT)
    assert (result.source.width, result.source.height) == (_images.WIDTH, _images.HEIGHT)
    assert result.dimensions_changed_by_orientation is False
    output = _decode(result.image_bytes)
    assert output.size == (_images.WIDTH, _images.HEIGHT)


@pytest.mark.parametrize("case_name", sorted(_images.ALL_MODE_BUILDERS))
def test_every_supported_input_mode_records_its_true_source_mode(case_name):
    """The provenance record reports what the source *was*, not what it became."""
    result = normalize_page_image(_images.ALL_MODE_BUILDERS[case_name]())

    expected_mode = _decode(_images.ALL_MODE_BUILDERS[case_name]()).mode
    assert result.source.color_mode == expected_mode
    assert result.source.color_mode != "RGB" or case_name == "RGB"


def test_bilevel_source_bit_depth_is_recorded_as_1_not_8():
    """A bilevel source is 1 bit per pixel and the record must say so -- reporting 8 would erase the
    single most relevant fact about a microfilm scan."""
    result = normalize_page_image(_images.bilevel())

    assert result.source.color_mode == "1"
    assert result.source.bit_depth == 1
    assert result.source.channel_count == 1


def test_16_bit_grayscale_is_scaled_not_clipped():
    """The regression this stage's existence most concretely protects against.

    Pillow 10.4.0's `Image.convert("L")` on mode `I;16` **clips**: source samples 4096, 32768 and
    65535 all become 255. Verified directly below, so this test documents the upstream behaviour
    rather than merely asserting ours differs from a remembered claim. The stage scales linearly
    instead, so distinct source tones stay distinct.
    """
    source = Image.new("I;16", (4, 1))
    source.putdata([0, 4096, 32768, 65535])
    assert list(source.convert("L").getdata()) == [0, 255, 255, 255], (
        "Pillow's clipping behaviour changed; the scaling rationale in rgb_normalization.py "
        "needs revisiting"
    )

    result = normalize_page_image(_images.grayscale_16bit())

    assert result.source.color_mode == "I;16"
    assert result.source.bit_depth == 16
    tones = {pixel[0] for pixel in _decode(result.image_bytes).getdata()}
    assert tones == {0, 15, 127, 255}, (
        f"expected linearly scaled tones, got {sorted(tones)} -- values collapsing to 255 means "
        "the clipping conversion was used"
    )


def test_rgba_transparency_is_composited_over_white():
    """A fully transparent red must become white, not red and not black."""
    result = normalize_page_image(_images.rgba())

    assert result.source.alpha_present is True
    assert result.alpha_compositing_applied is True
    assert result.compositing_background == "#FFFFFF"
    assert set(_decode(result.image_bytes).getdata()) == {(255, 255, 255)}


def test_grayscale_alpha_transparency_is_composited_over_white():
    """`LA` carries a real alpha channel and must be composited, not merely dropped."""
    result = normalize_page_image(_images.grayscale_alpha())

    assert result.source.color_mode == "LA"
    assert result.alpha_compositing_applied is True
    assert set(_decode(result.image_bytes).getdata()) == {(255, 255, 255)}


def test_palette_transparency_is_composited_rather_than_silently_discarded():
    """A `P` image's `transparency` index is transparency, even though it is not an alpha channel.

    The failure mode this guards: `convert("RGB")` straight from `P` ignores the transparency entry
    and emits the palette colour at that index -- here a vivid green -- so transparent page margins
    would come out green instead of white.
    """
    result = normalize_page_image(_images.palette_with_transparency())

    assert result.source.color_mode == "P"
    assert result.source.alpha_present is True, "a palette transparency index is transparency"
    assert result.alpha_compositing_applied is True
    pixels = set(_decode(result.image_bytes).getdata())
    assert pixels == {(255, 255, 255)}
    assert (17, 220, 33) not in pixels, "the palette colour leaked through instead of compositing"


def test_opaque_palette_is_converted_without_claiming_compositing_happened():
    """An opaque `P` image has nothing to composite, and the record must not imply otherwise."""
    result = normalize_page_image(_images.palette())

    assert result.source.alpha_present is False
    assert result.alpha_compositing_applied is False
    assert set(_decode(result.image_bytes).getdata()) == {(220, 30, 40)}


def test_cmyk_becomes_rgb():
    """Four channels in, three out."""
    result = normalize_page_image(_images.cmyk())

    assert result.source.color_mode == "CMYK"
    assert result.source.channel_count == 4
    assert result.source.image_format == "TIFF"
    assert _decode(result.image_bytes).mode == "RGB"


def test_already_conforming_rgb_keeps_its_pixel_values():
    """Normalizing an already-RGB image must not alter a single pixel. This is the strongest
    available check that no tonal/contrast/colour operation is applied: any enhancement would move
    these values."""
    result = normalize_page_image(_images.rgb())

    assert set(_decode(result.image_bytes).getdata()) == {(10, 20, 30)}
    assert result.alpha_compositing_applied is False
    assert result.exif_orientation_applied is False


def test_black_background_policy_composites_over_black():
    """The alpha policy is genuinely a policy, not a hardcoded white."""
    config = RgbNormalizationConfig(alpha_policy=AlphaCompositingPolicy.BLACK_BACKGROUND)

    result = normalize_page_image(_images.rgba(), config=config)

    assert result.compositing_background == "#000000"
    assert set(_decode(result.image_bytes).getdata()) == {(0, 0, 0)}
