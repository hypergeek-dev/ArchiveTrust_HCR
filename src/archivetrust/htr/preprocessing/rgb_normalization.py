"""The versioned RGB-normalization preprocessing stage.

An **independent pipeline stage**, in exactly the sense `docs/htr-domain-design.md` §7 establishes for
segmentation: it is not buried inside one method's adapter, it is named and versioned, its output is
content-addressed, and any method that needs it consumes its artifact rather than re-deriving it. It
exists because Transkribus Swedish Lion I page-level workflows require a page image whose colour
representation is a stated fact rather than whatever a library default produced -- see
`docs/methods/transkribus-swedish-lion-1.md`.

## The contract

Given raw image bytes and an `RgbNormalizationConfig`, produce bytes that are **exactly**: RGB,
3 channels, R-G-B order, 8 bits per channel, no alpha, no embedded ICC profile, no EXIF/XMP metadata,
lossless PNG, and pixel geometry unchanged except for a physically-applied EXIF orientation.

**No enhancement, by construction.** This stage performs no contrast adjustment, no autocontrast or
level stretching, no sharpening, no denoising, no thresholding or binarization, no resizing, no
deskewing, no cropping, no padding, no colour balancing, no gamma adjustment, and no lossy
compression. That is not a promise made in prose and hoped for: `ALLOWED_PILLOW_OPERATIONS` below
enumerates every Pillow call this module is permitted to make, and
`tests/htr/preprocessing/test_no_enhancement_contract.py` AST-scans this file and fails if any other
Pillow attribute is referenced anywhere in it. Adding `ImageEnhance`, `ImageFilter`,
`ImageOps.autocontrast`, `.resize`, `.thresh`, or `.rotate` to this module makes the test suite red.

The two operations that *do* touch pixel values are both bit-depth or channel normalizations that the
target representation mathematically requires, not aesthetic choices:

* **Alpha compositing.** RGB has no alpha channel, so transparency must be resolved against a
  background. The background is a versioned, recorded policy value, defaulting to white.
* **16-bit to 8-bit reduction.** An 8-bit-per-channel target cannot hold 16-bit samples. This is
  done by explicit linear scaling (`value * 255/65535`), *never* by Pillow's own `convert("L")`,
  which was verified on Pillow 10.4.0 to **clip** rather than scale: source samples 4096, 32768 and
  65535 all became 255, turning most of a 16-bit archival scan into pure white. Linear scaling is
  determinate and preserves relative tone; it is emphatically not a min/max contrast stretch (which
  would depend on image content and would be an enhancement).

## Determinism and idempotence

Identical input bytes and an identical configuration always produce identical output bytes -- no
timestamps, no random seeds, no content-dependent branching beyond the declared mode handling, and
PNG encoding is pinned to fixed parameters. Consequently
`normalize(normalize(x)).content_hash == normalize(x).content_hash`, asserted literally in
`tests/htr/preprocessing/test_idempotence.py`.

**An already-conforming image is still rewritten -- and may come out byte-identical.** Output is
always re-encoded to this stage's canonical PNG form (metadata stripped, profile removed, fixed
encoder parameters) rather than passed through, so the normalized artifact's identity is "the
canonical form produced by version V of this stage", not "unchanged bytes". For most real archival
input (CMYK, palette, 16-bit, JPEG, anything carrying an ICC profile or EXIF) that re-encode produces
a different digest from the source. For a source that already *is* the canonical form, it produces the
same digest -- which is correct and is not treated as an error: see `NormalizedPageArtifact._validate`
for why an earlier revision's "digests must differ" check was removed. The two hashes remain distinct
strings regardless, because their prefixes differ.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field

from PIL import Image, ImageOps, UnidentifiedImageError

from archivetrust.htr.preprocessing.models import (
    RGB_NORMALIZATION_VERSION,
    AlphaCompositingPolicy,
    IccProfilePolicy,
    NormalizationError,
    NormalizationFailure,
    NormalizationFailureCategory,
    NormalizedPageArtifact,
    PageImageArtifact,
    RgbNormalizationConfig,
)
from archivetrust.domain.shared.ids import new_id

ALLOWED_PILLOW_OPERATIONS = frozenset(
    {
        # -- Decoding and encoding -------------------------------------------------------------
        "open",  # Image.open
        "new",  # Image.new -- the compositing background canvas
        "save",  # encode to the configured lossless format
        "load",  # force decode so a truncated file fails here, not later
        # -- Mode / channel normalization ------------------------------------------------------
        "convert",  # the mode conversions themselves
        "point",  # explicit, linear 16-bit -> 8-bit scaling (see module docstring)
        "split",  # extracting an alpha band to use as a compositing mask
        "getchannel",  # ditto, by name
        "paste",  # compositing the source over the background through its alpha mask
        # -- Orientation -----------------------------------------------------------------------
        "exif_transpose",  # ImageOps.exif_transpose -- physically applies EXIF orientation
        "getexif",  # reading the Orientation tag in order to record it
        # -- Read-only inspection --------------------------------------------------------------
        "mode",
        "size",
        "width",
        "height",
        "format",
        "info",
        "getdata",
        "close",
    }
)
"""Every Pillow attribute this module is allowed to touch, enumerated so the no-enhancement contract
is machine-checkable rather than a claim in a docstring.

`tests/htr/preprocessing/test_no_enhancement_contract.py` parses this file's AST, collects every
attribute accessed on `Image`/`ImageOps`/an image object, and asserts the set is a subset of this one.
Nothing here can crop, scale, filter, threshold, or adjust tone. Widening this set is therefore a
deliberate, reviewable act that shows up in a diff next to this comment."""

_BACKGROUND_RGB = {
    AlphaCompositingPolicy.WHITE_BACKGROUND: (255, 255, 255),
    AlphaCompositingPolicy.BLACK_BACKGROUND: (0, 0, 0),
}

_BACKGROUND_HEX = {
    AlphaCompositingPolicy.WHITE_BACKGROUND: "#FFFFFF",
    AlphaCompositingPolicy.BLACK_BACKGROUND: "#000000",
}

_SIXTEEN_BIT_MODES = frozenset({"I", "I;16", "I;16B", "I;16L", "I;16N"})
"""Modes whose samples are wider than 8 bits and therefore need explicit scaling rather than
Pillow's clipping `convert`. `I` is 32-bit-signed in Pillow but is what 16-bit TIFFs commonly decode
to, and it clips identically, so it is handled on the same path."""

_ALPHA_MODES = frozenset({"LA", "RGBA", "PA", "La", "RGBa"})

_EXIF_ORIENTATION_TAG = 274
"""EXIF `Orientation`. Named rather than inlined so the record of *why* 274 is read is in the code."""

_ORIENTATIONS_THAT_SWAP_AXES = frozenset({5, 6, 7, 8})
"""The four EXIF orientations involving a 90/270-degree rotation, which legitimately exchange width
and height. Recorded so a dimension difference is never unexplained."""


@dataclass(frozen=True)
class SourceImageProperties:
    """What was observed about the source *before* anything was done to it."""

    color_mode: str
    bit_depth: int
    channel_count: int
    width: int
    height: int
    image_format: str | None
    icc_profile_present: bool
    icc_profile_hash: str | None
    icc_profile_description: str | None
    alpha_present: bool
    exif_orientation: int | None


@dataclass(frozen=True)
class NormalizationResult:
    """The emitted bytes plus everything needed to provenance them."""

    image_bytes: bytes
    source: SourceImageProperties
    output_width: int
    output_height: int
    compositing_background: str
    alpha_compositing_applied: bool
    exif_orientation_applied: bool
    dimensions_changed_by_orientation: bool
    warnings: tuple[str, ...] = field(default=())

    @property
    def content_hash(self) -> str:
        """The normalized artifact's content address -- the value the idempotence test compares."""
        return NormalizedPageArtifact.compute_hash(self.image_bytes)


def _channel_count(mode: str) -> int:
    return {"1": 1, "L": 1, "La": 2, "LA": 2, "P": 1, "PA": 2, "I": 1, "F": 1, "CMYK": 4}.get(
        mode, 4 if mode in ("RGBA", "RGBa") else 3 if mode == "RGB" else 1
    )


def _bit_depth(mode: str) -> int:
    if mode == "1":
        return 1
    if mode in _SIXTEEN_BIT_MODES:
        return 16
    return 8


def _icc_description(profile: bytes) -> str | None:
    """Reads an ICC profile's `desc` tag without a colour-management library.

    Deliberately minimal and failure-tolerant: this is recorded as evidence about a profile that is
    about to be stripped, so an unreadable header must produce `None` plus a warning, never an
    exception and never a fabricated name. The profile's own bytes are hashed regardless, so identity
    survives even when the description does not.
    """
    try:
        count = int.from_bytes(profile[128:132], "big")
        for index in range(count):
            entry = 132 + index * 12
            signature = profile[entry : entry + 4]
            if signature != b"desc":
                continue
            offset = int.from_bytes(profile[entry + 4 : entry + 8], "big")
            size = int.from_bytes(profile[entry + 8 : entry + 12], "big")
            body = profile[offset : offset + size]
            text = body[12:].split(b"\x00", 1)[0]
            return text.decode("ascii", errors="replace") or None
    except Exception:  # noqa: BLE001 -- an unreadable profile is recorded, never raised
        return None
    return None


def _observe(image: Image.Image, *, warnings: list[str]) -> SourceImageProperties:
    """Records the source's honest, pre-transformation properties."""
    profile = image.info.get("icc_profile")
    description = None
    if profile:
        description = _icc_description(profile)
        if description is None:
            warnings.append(
                "embedded ICC profile present but its 'desc' tag could not be read; the profile "
                "is still recorded by SHA-256"
            )

    orientation: int | None = None
    try:
        orientation = image.getexif().get(_EXIF_ORIENTATION_TAG)
    except Exception:  # noqa: BLE001 -- unreadable EXIF is an absence, not a failure
        warnings.append("EXIF block present but unreadable; no orientation was applied")

    return SourceImageProperties(
        color_mode=image.mode,
        bit_depth=_bit_depth(image.mode),
        channel_count=_channel_count(image.mode),
        width=image.width,
        height=image.height,
        image_format=image.format,
        icc_profile_present=bool(profile),
        icc_profile_hash=hashlib.sha256(profile).hexdigest() if profile else None,
        icc_profile_description=description,
        alpha_present=image.mode in _ALPHA_MODES or "transparency" in image.info,
        exif_orientation=int(orientation) if isinstance(orientation, int) else None,
    )


def _to_rgb(image: Image.Image, *, config: RgbNormalizationConfig) -> tuple[Image.Image, bool]:
    """Converts any supported input mode to 8-bit RGB. Returns `(rgb_image, alpha_was_composited)`.

    Short on purpose: this is the one function whose every line changes pixels, so it is kept
    eyeball-verifiable. Read top to bottom, it is four cases and nothing else -- widen it and the
    no-enhancement contract stops being checkable by inspection.
    """
    mode = image.mode

    # 1. Wider-than-8-bit samples: explicit linear scaling, never Pillow's clipping convert.
    if mode in _SIXTEEN_BIT_MODES:
        return image.point(lambda value: value * (255.0 / 65535.0), "L").convert("RGB"), False

    # 2. Palette images: expand through RGBA when a transparency entry exists, else straight to RGB.
    if mode in ("P", "PA"):
        image = image.convert("RGBA" if "transparency" in image.info or mode == "PA" else "RGB")
        mode = image.mode

    # 3. Transparency of any kind must be resolved against the versioned background colour.
    if mode in _ALPHA_MODES:
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, _BACKGROUND_RGB[config.alpha_policy])
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background, True

    # 4. Everything else (`1`, `L`, `CMYK`, `RGB`) is a direct, lossless-or-required conversion.
    return (image if mode == "RGB" else image.convert("RGB")), False


def normalize_page_image(
    image_bytes: bytes, *, config: RgbNormalizationConfig | None = None
) -> NormalizationResult:
    """Normalizes raw page-image bytes to this stage's canonical RGB form.

    Pure: no filesystem, no telemetry, no clock. The caller owns persistence and event emission
    (`normalization_service.py`), which is what makes this function unit-testable against every input
    mode without a store, and what keeps the transform itself free of anything that could make two
    runs differ.

    Raises `NormalizationError` -- carrying a durable `NormalizationFailure` -- for every failure the
    specification requires to be caught *before* export. It never returns the original bytes as a
    fallback; there is no code path here that can.
    """
    config = config or RgbNormalizationConfig()
    warnings: list[str] = []

    def fail(category: NormalizationFailureCategory, reason: str) -> NormalizationError:
        return NormalizationError(
            NormalizationFailure.create(
                page_id="",
                category=category,
                reason=reason,
                configuration_hash=config.configuration_hash,
                occurred_at="",
                source_content_hash=(
                    PageImageArtifact.compute_hash(image_bytes) if image_bytes else None
                ),
            )
        )

    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise fail(
            NormalizationFailureCategory.UNDECODABLE_IMAGE, f"{type(exc).__name__}: {exc}"
        ) from exc

    source = _observe(image, warnings=warnings)

    # EXIF orientation is applied *first*, so every later step sees the pixels a reader would.
    oriented = image
    orientation_applied = False
    if config.apply_exif_orientation and source.exif_orientation not in (None, 1):
        oriented = ImageOps.exif_transpose(image)
        orientation_applied = True
    axes_swapped = orientation_applied and source.exif_orientation in _ORIENTATIONS_THAT_SWAP_AXES

    try:
        rgb, composited = _to_rgb(oriented, config=config)
    except (OSError, ValueError) as exc:
        raise fail(
            NormalizationFailureCategory.RGB_OUTPUT_NOT_PRODUCIBLE,
            f"cannot produce RGB from mode {source.color_mode!r}: {type(exc).__name__}: {exc}",
        ) from exc

    if rgb.mode != "RGB":
        raise fail(
            NormalizationFailureCategory.RGB_OUTPUT_NOT_PRODUCIBLE,
            f"conversion from mode {source.color_mode!r} yielded mode {rgb.mode!r}, not 'RGB'",
        )

    expected = (
        (source.height, source.width) if axes_swapped else (source.width, source.height)
    )
    if rgb.width < 1 or rgb.height < 1:
        raise fail(
            NormalizationFailureCategory.INVALID_OUTPUT_DIMENSIONS,
            f"output dimensions {rgb.width}x{rgb.height} are not a valid raster",
        )
    if config.preserve_dimensions and (rgb.width, rgb.height) != expected:
        raise fail(
            NormalizationFailureCategory.INVALID_OUTPUT_DIMENSIONS,
            f"geometry changed from {source.width}x{source.height} to {rgb.width}x{rgb.height} "
            f"for a reason other than an applied EXIF orientation",
        )

    # Encode. `optimize=False`/`compress_level=6` are pinned rather than left to the library default
    # so an encoder-default change in a future Pillow cannot silently alter output hashes.
    #
    # `icc_profile=None` and `exif=None` are passed **explicitly**, and that is load-bearing rather
    # than decorative: Pillow propagates `Image.info["icc_profile"]` into the encoder automatically
    # when the key is merely absent from the save arguments, so simply *not* mentioning the profile
    # silently re-embeds it in the output. Verified against Pillow 10.4.0 and covered by
    # `tests/htr/preprocessing/test_idempotence.py::
    # test_icc_profile_is_recorded_then_stripped_without_being_applied`, which failed before these two
    # arguments were added. The profile is already recorded on the artifact by hash and description
    # (IccProfilePolicy.RECORD_AND_STRIP_WITHOUT_APPLYING), so stripping it loses no evidence.
    if config.icc_profile_policy is not IccProfilePolicy.RECORD_AND_STRIP_WITHOUT_APPLYING:
        raise fail(
            NormalizationFailureCategory.ENCODE_FAILED,
            f"unsupported icc_profile_policy {config.icc_profile_policy!r} for normalization "
            f"version {RGB_NORMALIZATION_VERSION}",
        )
    buffer = io.BytesIO()
    try:
        rgb.save(
            buffer,
            format=config.output_format,
            optimize=False,
            compress_level=6,
            icc_profile=None,
            exif=None,
        )
    except (OSError, ValueError, KeyError) as exc:
        raise fail(
            NormalizationFailureCategory.ENCODE_FAILED,
            f"cannot encode to {config.output_format}: {type(exc).__name__}: {exc}",
        ) from exc

    return NormalizationResult(
        image_bytes=buffer.getvalue(),
        source=source,
        output_width=rgb.width,
        output_height=rgb.height,
        compositing_background=_BACKGROUND_HEX[config.alpha_policy],
        alpha_compositing_applied=composited,
        exif_orientation_applied=orientation_applied,
        dimensions_changed_by_orientation=axes_swapped,
        warnings=tuple(warnings),
    )


def build_normalized_artifact(
    result: NormalizationResult,
    *,
    source_artifact: PageImageArtifact,
    storage_path: str,
    executed_at: str,
    config: RgbNormalizationConfig | None = None,
) -> NormalizedPageArtifact:
    """Assembles the provenance record for a completed normalization.

    Separate from `normalize_page_image` because the transform must stay clock-free: `executed_at` is
    supplied by the caller, never read from `datetime.now()` inside the transform, so two runs of the
    transform over the same bytes are byte-identical and hash-identical.
    """
    config = config or RgbNormalizationConfig()
    source = result.source
    try:
        normalized_hash = NormalizedPageArtifact.compute_hash(result.image_bytes)
    except Exception as exc:  # noqa: BLE001 -- an uncomputable hash is a real, reportable failure
        raise NormalizationError(
            NormalizationFailure.create(
                page_id=source_artifact.page_id,
                category=NormalizationFailureCategory.HASH_NOT_COMPUTABLE,
                reason=f"{type(exc).__name__}: {exc}",
                configuration_hash=config.configuration_hash,
                occurred_at=executed_at,
                source_content_hash=source_artifact.content_hash,
            )
        ) from exc

    return NormalizedPageArtifact(
        normalized_artifact_id=new_id("normalized_page_artifact"),
        source_artifact_id=source_artifact.artifact_id,
        source_content_hash=source_artifact.content_hash,
        normalized_content_hash=normalized_hash,
        page_id=source_artifact.page_id,
        storage_path=storage_path,
        byte_size=len(result.image_bytes),
        source_color_mode=source.color_mode,
        source_bit_depth=source.bit_depth,
        source_channel_count=source.channel_count,
        source_width=source.width,
        source_height=source.height,
        source_format=source.image_format,
        source_icc_profile_present=source.icc_profile_present,
        source_icc_profile_hash=source.icc_profile_hash,
        source_icc_profile_description=source.icc_profile_description,
        source_alpha_present=source.alpha_present,
        source_exif_orientation=source.exif_orientation,
        output_color_mode="RGB",
        output_bit_depth=8,
        output_channel_count=3,
        output_width=result.output_width,
        output_height=result.output_height,
        output_format=config.output_format,
        output_icc_profile_present=False,
        compositing_background=result.compositing_background,
        alpha_compositing_applied=result.alpha_compositing_applied,
        exif_orientation_applied=result.exif_orientation_applied,
        dimensions_changed_by_orientation=result.dimensions_changed_by_orientation,
        configuration_hash=config.configuration_hash,
        executed_at=executed_at,
        warnings=result.warnings,
    )
