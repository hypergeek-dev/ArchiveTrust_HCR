"""Determinism, idempotence, ICC handling, EXIF orientation, and hash distinctness."""

from __future__ import annotations

import io

import pytest
from PIL import Image

from archivetrust.htr.preprocessing.models import (
    AlphaCompositingPolicy,
    NormalizedPageArtifact,
    PageImageArtifact,
    RgbNormalizationConfig,
)
from archivetrust.htr.preprocessing.rgb_normalization import normalize_page_image
from tests.htr.preprocessing import _images


@pytest.mark.parametrize("case_name", sorted(_images.ALL_MODE_BUILDERS))
def test_normalize_is_idempotent_by_output_content_hash(case_name):
    """`normalize(normalize(x)) == normalize(x)`, compared by output content hash.

    The literal assertion the specification asks for, run over every supported input mode: feeding the
    stage's own output back in must produce byte-identical output, i.e. the canonical form is a genuine
    fixed point rather than something that drifts each pass.
    """
    once = normalize_page_image(_images.ALL_MODE_BUILDERS[case_name]())
    twice = normalize_page_image(once.image_bytes)

    assert twice.content_hash == once.content_hash
    assert twice.image_bytes == once.image_bytes


@pytest.mark.parametrize("case_name", sorted(_images.ALL_MODE_BUILDERS))
def test_normalization_is_deterministic_across_repeated_runs(case_name):
    """Two independent runs over identical input bytes produce identical output bytes -- no
    timestamp, no random seed, no environment leaking into the raster."""
    source = _images.ALL_MODE_BUILDERS[case_name]()

    first = normalize_page_image(source)
    second = normalize_page_image(source)

    assert first.content_hash == second.content_hash


def test_third_pass_is_also_stable():
    """Idempotence at the second pass could in principle hide a two-cycle. It does not."""
    first = normalize_page_image(_images.cmyk())
    second = normalize_page_image(first.image_bytes)
    third = normalize_page_image(second.image_bytes)

    assert first.content_hash == second.content_hash == third.content_hash


@pytest.mark.parametrize(
    "case_name", ["CMYK", "P", "P_transparency", "LA", "RGBA", "I;16", "1"]
)
def test_original_and_normalized_hashes_are_distinct_for_non_conforming_sources(case_name):
    """For any source not already in canonical form, the derived artifact's digest differs from the
    original's -- so the two are distinguishable by content, not merely by prefix.

    Parametrized over the genuinely non-conforming modes, which is every realistic archival input.
    The already-conforming case is covered separately below, because its behaviour is different and
    deliberately so."""
    source_bytes = _images.ALL_MODE_BUILDERS[case_name]()
    result = normalize_page_image(source_bytes)

    source_hash = PageImageArtifact.compute_hash(source_bytes)
    normalized_hash = NormalizedPageArtifact.compute_hash(result.image_bytes)

    assert source_hash[len("page_image_") :] != normalized_hash[len("normalized_page_") :]


def test_hashes_are_always_distinct_strings_even_when_the_bytes_coincide():
    """The prefixes differ, so an original and a derived artifact can never be confused in a lookup
    even in the edge case where normalization changed nothing.

    This documents a real finding rather than a hypothetical: a plain 12x8 RGB PNG written by Pillow
    already *is* this stage's canonical form, so it normalizes to byte-identical output. An earlier
    revision of `NormalizedPageArtifact` rejected that as invalid; this test pins the corrected
    behaviour so it cannot regress back into a validation error.
    """
    source_bytes = _images.rgb()
    result = normalize_page_image(source_bytes)

    source_hash = PageImageArtifact.compute_hash(source_bytes)
    normalized_hash = NormalizedPageArtifact.compute_hash(result.image_bytes)

    assert source_hash != normalized_hash
    assert source_hash.startswith("page_image_")
    assert normalized_hash.startswith("normalized_page_")
    # The underlying digests genuinely coincide here, and that is allowed.
    assert source_hash[len("page_image_") :] == normalized_hash[len("normalized_page_") :]


def test_an_already_canonical_image_normalizes_to_identical_bytes():
    """Conforming input is re-encoded, not passed through -- and for input that already matches the
    canonical form exactly, re-encoding is a no-op at the byte level. Pixels and geometry are
    untouched either way."""
    source_bytes = _images.rgb()

    result = normalize_page_image(source_bytes)

    assert result.image_bytes == source_bytes
    assert Image.open(io.BytesIO(result.image_bytes)).size == Image.open(
        io.BytesIO(source_bytes)
    ).size


def test_a_non_canonical_source_really_is_rewritten():
    """The complement of the above: a JPEG-with-EXIF source is genuinely rewritten, so "re-encoded to
    canonical form" is not a claim that only holds vacuously."""
    source_bytes = _images.rgb_with_exif_orientation(6)

    result = normalize_page_image(source_bytes)

    assert result.image_bytes != source_bytes
    assert Image.open(io.BytesIO(result.image_bytes)).format == "PNG"


def test_a_different_config_produces_a_different_configuration_hash():
    """The configuration hash is what pipeline versioning keys on, so it must actually move."""
    default = RgbNormalizationConfig()
    black = RgbNormalizationConfig(alpha_policy=AlphaCompositingPolicy.BLACK_BACKGROUND)

    assert default.configuration_hash != black.configuration_hash
    assert default.configuration_hash == RgbNormalizationConfig().configuration_hash


def test_configuration_hash_is_stable_across_field_declaration_order():
    """Two configs built with the same values hash identically regardless of kwarg order -- the hash
    depends on values, never on how the object was constructed."""
    first = RgbNormalizationConfig(output_format="PNG", target_mode="RGB")
    second = RgbNormalizationConfig(target_mode="RGB", output_format="PNG")

    assert first.configuration_hash == second.configuration_hash


# -- ICC profile handling -----------------------------------------------------------------------


def test_icc_profile_is_recorded_then_stripped_without_being_applied():
    """The documented policy, end to end: presence recorded, bytes hashed, description read, profile
    absent from the output."""
    result = normalize_page_image(_images.rgb_with_icc_profile())

    assert result.source.icc_profile_present is True
    assert result.source.icc_profile_hash is not None
    assert len(result.source.icc_profile_hash) == 64, "expected a raw sha256 hex digest"
    assert result.source.icc_profile_description == "ArchiveTrust synthetic test profile"

    output = Image.open(io.BytesIO(result.image_bytes))
    assert output.info.get("icc_profile") is None, "the profile was not stripped from the output"


def test_icc_profile_is_not_applied_to_pixel_values():
    """Record-and-strip means pixels are untouched. If the profile were applied, these grays would
    move; they do not."""
    result = normalize_page_image(_images.rgb_with_icc_profile())

    assert set(Image.open(io.BytesIO(result.image_bytes)).getdata()) == {(90, 90, 90)}


def test_absent_icc_profile_is_recorded_as_absent_never_as_a_placeholder():
    result = normalize_page_image(_images.rgb())

    assert result.source.icc_profile_present is False
    assert result.source.icc_profile_hash is None
    assert result.source.icc_profile_description is None


def test_icc_bearing_input_is_still_idempotent():
    """Once stripped, a second pass sees no profile and must produce the same bytes."""
    once = normalize_page_image(_images.rgb_with_icc_profile())
    twice = normalize_page_image(once.image_bytes)

    assert twice.content_hash == once.content_hash


# -- EXIF orientation ---------------------------------------------------------------------------


def test_no_exif_orientation_tag_is_recorded_as_none_not_as_orientation_1():
    """`None` means "no tag", which is a different fact from "explicitly upright"."""
    result = normalize_page_image(_images.rgb())

    assert result.source.exif_orientation is None
    assert result.exif_orientation_applied is False


def test_orientation_1_is_recorded_but_no_transpose_is_performed():
    result = normalize_page_image(_images.rgb_with_exif_orientation(1))

    assert result.source.exif_orientation == 1
    assert result.exif_orientation_applied is False
    assert (result.output_width, result.output_height) == (_images.WIDTH, _images.HEIGHT)


@pytest.mark.parametrize("orientation", [5, 6, 7, 8])
def test_axis_swapping_orientations_are_physically_applied_and_flagged(orientation):
    """Orientations 5-8 involve a 90/270-degree rotation, so width and height legitimately swap. The
    change is recorded via `dimensions_changed_by_orientation` so it is never an unexplained
    difference -- and `preserve_dimensions` does not reject it."""
    result = normalize_page_image(_images.rgb_with_exif_orientation(orientation))

    assert result.source.exif_orientation == orientation
    assert result.exif_orientation_applied is True
    assert result.dimensions_changed_by_orientation is True
    assert (result.output_width, result.output_height) == (_images.HEIGHT, _images.WIDTH)


@pytest.mark.parametrize("orientation", [2, 3, 4])
def test_non_axis_swapping_orientations_are_applied_without_changing_geometry(orientation):
    """Orientations 2, 3 and 4 are flips/180-degree rotations: pixels move, dimensions do not."""
    result = normalize_page_image(_images.rgb_with_exif_orientation(orientation))

    assert result.exif_orientation_applied is True
    assert result.dimensions_changed_by_orientation is False
    assert (result.output_width, result.output_height) == (_images.WIDTH, _images.HEIGHT)


def test_exif_metadata_is_stripped_from_the_output():
    """Orientation is applied *to the pixels* and then the metadata is normalized away, so no
    downstream reader can rotate a second time."""
    result = normalize_page_image(_images.rgb_with_exif_orientation(6))

    output = Image.open(io.BytesIO(result.image_bytes))
    assert output.getexif().get(274) is None
    assert not output.info.get("exif")


def test_orientation_can_be_disabled_and_then_geometry_is_untouched():
    """`apply_exif_orientation=False` is a real configuration, and it changes the output hash --
    which is exactly why it is part of the configuration hash."""
    applied = normalize_page_image(_images.rgb_with_exif_orientation(6))
    ignored = normalize_page_image(
        _images.rgb_with_exif_orientation(6),
        config=RgbNormalizationConfig(apply_exif_orientation=False),
    )

    assert applied.dimensions_changed_by_orientation is True
    assert ignored.exif_orientation_applied is False
    assert (ignored.output_width, ignored.output_height) == (_images.WIDTH, _images.HEIGHT)
    assert ignored.source.exif_orientation == 6, "the tag is still recorded even when not applied"
    assert applied.content_hash != ignored.content_hash
