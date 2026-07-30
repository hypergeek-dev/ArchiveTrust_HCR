"""The no-enhancement contract, checked by construction rather than asserted in prose.

The stage's hard constraint is that it performs *no* contrast adjustment, sharpening, denoising,
thresholding, resizing, deskewing, cropping, colour balancing, or lossy compression. A test that
merely normalized a few images and checked the pixels would not establish that: it would establish
that no enhancement happened *to those images*.

These tests instead parse `rgb_normalization.py`'s AST and assert properties of the implementation:

* every Pillow attribute it touches is in its own declared `ALLOWED_PILLOW_OPERATIONS` allowlist;
* no enhancement-capable Pillow module is imported anywhere in the package;
* the one function that changes pixel values is short enough to verify by eye;
* the configuration model structurally cannot express an enhancement.

Adding `ImageEnhance`, `ImageFilter`, `.resize`, `.rotate`, or `.point`-with-a-lookup-table to that
module makes this file fail.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from archivetrust.htr.preprocessing import models, rgb_normalization
from archivetrust.htr.preprocessing.models import RgbNormalizationConfig
from archivetrust.htr.preprocessing.rgb_normalization import ALLOWED_PILLOW_OPERATIONS

_MODULE_PATH = Path(inspect.getfile(rgb_normalization))
_MODULE_SOURCE = _MODULE_PATH.read_text(encoding="utf-8")
_MODULE_AST = ast.parse(_MODULE_SOURCE)

FORBIDDEN_OPERATION_NAMES = frozenset(
    {
        # Geometry
        "resize", "thumbnail", "crop", "rotate", "transform", "reduce", "resized",
        # Tone / colour
        "autocontrast", "equalize", "colorize", "posterize", "solarize", "invert",
        "brightness", "contrast", "color", "sharpness", "gamma", "level",
        # Filtering
        "filter", "gaussian_blur", "unsharp_mask", "box_blur", "medianfilter", "effect_spread",
        # Binarization / quantization
        "quantize", "threshold", "posterise", "dither",
    }
)
"""Operation names that would each individually breach the contract. Checked as a belt-and-braces
complement to the allowlist: the allowlist is authoritative, but naming the specific forbidden
operations makes a failure message say *what* was introduced."""

FORBIDDEN_MODULES = frozenset(
    {"ImageEnhance", "ImageFilter", "ImageMath", "ImageChops", "ImageStat", "cv2", "skimage", "scipy"}
)
"""Modules whose entire purpose is image enhancement or arithmetic. `ImageOps` is deliberately *not*
here -- only its `exif_transpose` is used, and the allowlist is what constrains that."""


def _attribute_names_accessed() -> set[str]:
    """Every attribute name accessed anywhere in the module (`x.foo` -> `"foo"`)."""
    return {
        node.attr for node in ast.walk(_MODULE_AST) if isinstance(node, ast.Attribute)
    }


def _imported_names() -> set[str]:
    names: set[str] = set()
    for node in ast.walk(_MODULE_AST):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.update(alias.name for alias in node.names)
            if node.module:
                names.add(node.module.split(".")[0])
    return names


def test_no_forbidden_enhancement_operation_appears_in_the_module():
    """The direct, named check: none of the enhancement operations is referenced at all."""
    offenders = sorted(_attribute_names_accessed() & FORBIDDEN_OPERATION_NAMES)

    assert not offenders, (
        f"{_MODULE_PATH.name} references forbidden enhancement operation(s) {offenders} -- this "
        "stage must perform no contrast, sharpening, denoising, thresholding, resizing, deskewing, "
        "cropping, or colour balancing"
    )


def test_no_enhancement_capable_module_is_imported():
    offenders = sorted(_imported_names() & FORBIDDEN_MODULES)

    assert not offenders, (
        f"{_MODULE_PATH.name} imports {offenders}, whose purpose is image enhancement or arithmetic"
    )


def test_every_pillow_attribute_used_is_declared_in_the_allowlist():
    """The authoritative check. Any Pillow (or image-object) attribute the module touches must appear
    in `ALLOWED_PILLOW_OPERATIONS`, so widening the module's capabilities requires editing that
    allowlist -- a visible, reviewable act -- rather than quietly adding a call.

    Attributes belonging to this project's own models (`config.alpha_policy`, `source.width`, ...) are
    excluded by name, since they are not Pillow surface.
    """
    accessed = _attribute_names_accessed()

    own_model_fields = (
        set(RgbNormalizationConfig.model_fields)
        | set(models.NormalizedPageArtifact.model_fields)
        | set(models.PageImageArtifact.model_fields)
        | set(models.NormalizationFailure.model_fields)
        | {
            # This module's own dataclass fields, helpers, enum members and stdlib attributes.
            "image_bytes", "source", "output_width", "output_height", "compositing_background",
            "alpha_compositing_applied", "exif_orientation_applied",
            "dimensions_changed_by_orientation", "warnings", "content_hash", "color_mode",
            "bit_depth", "channel_count", "image_format", "icc_profile_present", "icc_profile_hash",
            "icc_profile_description", "alpha_present", "exif_orientation", "failure",
            "compute_hash", "create", "configuration_hash", "value", "append", "get", "getvalue",
            "sha256", "hexdigest", "decode", "split", "BytesIO", "RECORD_AND_STRIP_WITHOUT_APPLYING",
            "UNDECODABLE_IMAGE", "RGB_OUTPUT_NOT_PRODUCIBLE", "INVALID_OUTPUT_DIMENSIONS",
            "ENCODE_FAILED", "HASH_NOT_COMPUTABLE", "ARTIFACT_NOT_PERSISTED", "Image",
            "from_bytes", "int", "strip",
            # Enum members of this project's own policy enums, plus the exception-type name used in
            # failure messages. None is Pillow surface.
            "WHITE_BACKGROUND", "BLACK_BACKGROUND", "__name__",
        }
    )

    undeclared = sorted(accessed - ALLOWED_PILLOW_OPERATIONS - own_model_fields)

    assert not undeclared, (
        f"{_MODULE_PATH.name} accesses attribute(s) {undeclared} that are neither in "
        "ALLOWED_PILLOW_OPERATIONS nor a known model field. If one is a legitimate Pillow "
        "operation, add it to that allowlist deliberately and justify it there; if it is an "
        "enhancement, it must not be here at all."
    )


def test_the_pixel_changing_function_is_short_enough_to_verify_by_eye():
    """`_to_rgb` is the only function that changes pixel values, and the contract's credibility rests
    on a reader being able to check it by inspection. A generous ceiling that still forbids it quietly
    growing into a pipeline."""
    function = next(
        node
        for node in ast.walk(_MODULE_AST)
        if isinstance(node, ast.FunctionDef) and node.name == "_to_rgb"
    )
    body_lines = function.end_lineno - function.lineno

    assert body_lines <= 40, (
        f"_to_rgb has grown to {body_lines} lines. It is the one function that changes pixels and "
        "must stay eyeball-verifiable; if it genuinely needs to be longer, the no-enhancement "
        "contract needs a stronger check than inspection."
    )


def test_the_allowlist_itself_contains_no_enhancement_operation():
    """Guards the guard: someone could satisfy the allowlist test by adding `resize` to the
    allowlist. This fails if they do."""
    offenders = sorted(ALLOWED_PILLOW_OPERATIONS & FORBIDDEN_OPERATION_NAMES)

    assert not offenders, f"ALLOWED_PILLOW_OPERATIONS has been widened to include {offenders}"


# -- The configuration model cannot express an enhancement --------------------------------------


def test_enhancement_field_is_structurally_pinned_to_none():
    with pytest.raises(ValueError, match="enhancement must be 'none'"):
        RgbNormalizationConfig(enhancement="autocontrast")


def test_a_lossy_output_format_is_refused():
    """A normalization stage that re-encoded to JPEG would be discarding archival detail while
    claiming to normalize."""
    with pytest.raises(ValueError, match="output_format must be one of"):
        RgbNormalizationConfig(output_format="JPEG")


def test_dimension_preservation_cannot_be_switched_off():
    with pytest.raises(ValueError, match="preserve_dimensions must be True"):
        RgbNormalizationConfig(preserve_dimensions=False)


@pytest.mark.parametrize(
    "field,value",
    [("target_mode", "L"), ("bits_per_channel", 16), ("channel_count", 4), ("channel_order", "BGR")],
)
def test_the_output_representation_cannot_be_reconfigured_away_from_8_bit_rgb(field, value):
    """The target representation is fixed by this stage's definition. A config that could ask for
    16-bit BGR would make "normalized" mean nothing in particular."""
    with pytest.raises(ValueError):
        RgbNormalizationConfig(**{field: value})


def test_the_docstring_states_the_contract():
    """A docstring-level contract test: the module's own documentation must name what it does not do,
    so the guarantee is discoverable by a reader and not only by this test file."""
    docstring = rgb_normalization.__doc__ or ""

    for forbidden in ("contrast", "sharpen", "denois", "threshold", "resiz", "deskew", "crop",
                      "colour balancing", "lossy"):
        assert forbidden in docstring.lower(), (
            f"rgb_normalization's module docstring no longer states that it performs no "
            f"{forbidden!r} operation"
        )
