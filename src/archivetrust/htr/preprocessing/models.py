"""Preprocessing entities: the versioned RGB-normalization stage's configuration and provenance.

**Pure frozen Pydantic + stdlib `hashlib`, deliberately.** `domain/telemetry/events.py` imports this
module directly so its events can carry the *full* domain object they announce (that module's own
docstring requires it: "replay must reconstruct [state] from telemetry alone"). That import is only
legitimate while this module stays free of third-party dependencies beyond pydantic -- ROADMAP.md S11
and `tests/domain/test_dependency_direction.py`. Pillow therefore lives in `rgb_normalization.py`,
never here, and `htr/preprocessing/__init__.py` re-exports this module only. Exactly the arrangement
`htr/corpus/models.py` already has with `InputCrop`.

## Why two artifact models rather than fields on `Page`

`PageImageArtifact` (the original) and `NormalizedPageArtifact` (the derived) are new models; the
existing `htr/corpus/models.py::Page` is left untouched. That was a decision, not an oversight.

`Page` is a geometric/ordinal record -- `page_number`, `width`, `height` -- and is constructed in
places where **no page image exists at all**. The committed baseline is the proof: its Transkribus
`MethodRun` has `input_crop_id=None` and its `PageRegistered` events describe pages of a
hand-authored PAGE XML fixture with no raw image anywhere in the repository
(`tests/fixtures/transkribus/` contains only `.xml`/`.txt`). Putting `image_hash`/`image_storage_path`
on `Page` would make every one of those constructions assert an image it does not have, and would
make "this page has no image" indistinguishable from "this page's image was never hashed". A separate
artifact model says the true thing: a page *may* have zero or more image artifacts, each
content-addressed, each independently provenanced.

The link is by id, never embedded (Constitution Article 7, docs/htr-domain-design.md §4): a
`NormalizedPageArtifact` carries `source_artifact_id` and `source_content_hash`, not a copy of the
`PageImageArtifact`.
"""

from __future__ import annotations

import hashlib
from enum import Enum

from pydantic import BaseModel, ConfigDict, model_validator

from archivetrust.domain.shared.ids import content_address, new_id

RGB_NORMALIZATION_STAGE = "image_color_normalization"
"""The stage's canonical name. The one string a pipeline configuration names it by (see
`htr/experiment/baseline_template.py::ImageColorNormalizationRequirement`) and the one this
module's config model records, so a configuration and an artifact cannot disagree about which
stage produced what."""

RGB_NORMALIZATION_VERSION = "1.0.0"
"""The *implementation* version of the normalization transform, bumped whenever a change could
alter output bytes for any input. Distinct from `RgbNormalizationConfig.version`, which versions the
configuration *schema*: two runs may share a config version and differ in implementation version, and
either difference must force a new `ExperimentVersion` (docs/methods/transkribus-swedish-lion-1.md
§6). Both are recorded on every `NormalizedPageArtifact`."""

RGB_NORMALIZATION_IMPLEMENTATION = "archivetrust.htr.preprocessing.rgb_normalization"
"""The implementation's dotted module name, recorded on every artifact so a provenance record names
the code that produced it rather than merely a version number that could belong to anything."""


class AlphaCompositingPolicy(str, Enum):
    """How transparency is removed. An RGB output has no alpha channel, so transparent pixels must
    be resolved against *something*; leaving that implicit is the anti-pattern this stage exists to
    remove, so it is a versioned, recorded enum value rather than a library default.
    """

    WHITE_BACKGROUND = "white_background"
    """Composite over opaque white, `(255, 255, 255)`. The default -- see
    `docs/DATA_HANDLING_POLICY.md` §7 for the reasoning and for the confirmed absence of any prior
    archival-image colour policy in this repository that would dictate otherwise."""

    BLACK_BACKGROUND = "black_background"
    """Composite over opaque black. Offered so the policy is genuinely a choice rather than a
    hardcoded constant wearing an enum's clothes; no configuration in this repository selects it."""


class IccProfilePolicy(str, Enum):
    """What happens to an embedded ICC profile.

    **`RECORD_AND_STRIP_WITHOUT_APPLYING` is the default, and is a reasoned departure from the two
    options the specification offered** ("strip after applying, or apply-then-record"). Both of those
    *apply* the profile, i.e. transform pixel values through a colour-management engine. That was
    rejected for two independent reasons:

    1. **It is an enhancement.** Applying an ICC transform rewrites pixel values to change how the
       image looks. This stage's hard constraint is that it performs no colour balancing, no
       contrast, no tonal adjustment of any kind (`rgb_normalization.py`'s contract). An ICC
       conversion is exactly a colour transform, so applying one would breach the stage's own
       guarantee while claiming to uphold it.
    2. **It is not deterministic in the sense required.** Pillow's ICC transform is performed by
       LittleCMS, whose output depends on the linked liblcms version and its rendering-intent
       implementation. Identical input bytes on two machines could produce different output bytes --
       and "must be deterministic" is a stated requirement of this stage, not a preference.

    Nothing is lost: the profile is *recorded* (presence, byte size, and its own SHA-256, plus the
    profile's declared description where readable) before being stripped, so the colour intent
    remains recoverable evidence attached to the artifact. Because the policy is a versioned enum
    field, a future normalization version may legitimately choose `APPLY_THEN_STRIP` -- it would
    simply be a different configuration hash and therefore a different pipeline configuration, which
    is the whole point of versioning it.
    """

    RECORD_AND_STRIP_WITHOUT_APPLYING = "record_and_strip_without_applying"
    """Record the profile's identity, do not transform pixels through it, emit output with no
    embedded profile."""


class NormalizationFailureCategory(str, Enum):
    """Which of the specification's enumerated pre-export failure conditions occurred. A closed set,
    each value set only where the code has a structural reason to believe it -- never inferred from a
    message string, matching `ProviderFailureCategory`'s discipline (Constitution Article 18).
    """

    UNDECODABLE_IMAGE = "undecodable_image"
    """The bytes could not be decoded as an image at all."""

    RGB_OUTPUT_NOT_PRODUCIBLE = "rgb_output_not_producible"
    """Decoding succeeded but no 3-channel 8-bit RGB raster could be produced from it."""

    INVALID_OUTPUT_DIMENSIONS = "invalid_output_dimensions"
    """A zero/negative dimension, or -- with `preserve_dimensions` -- geometry that changed for a
    reason other than an applied EXIF orientation."""

    ENCODE_FAILED = "encode_failed"
    """The normalized raster could not be encoded to the configured output format."""

    HASH_NOT_COMPUTABLE = "hash_not_computable"
    """The output artifact's content hash could not be computed."""

    ARTIFACT_NOT_PERSISTED = "artifact_not_persisted"
    """The artifact could not be written to the blob store or the export package."""

    MANIFEST_NOT_WRITTEN = "manifest_not_written"
    """The export package's manifest could not be written."""


class RgbNormalizationConfig(BaseModel):
    """The versioned, content-addressable configuration of one normalization stage.

    Every field is an *explicit* statement of something a library default would otherwise decide
    silently. `configuration_hash` is what a pipeline configuration and an `ExperimentVersion`
    record, so changing any field below necessarily changes the pipeline configuration
    (docs/methods/transkribus-swedish-lion-1.md §6).
    """

    model_config = ConfigDict(frozen=True)

    stage: str = RGB_NORMALIZATION_STAGE
    version: str = "1.0.0"
    """The configuration *schema* version -- see `RGB_NORMALIZATION_VERSION` for why the
    implementation version is a separate thing."""
    target_mode: str = "RGB"
    bits_per_channel: int = 8
    channel_count: int = 3
    channel_order: str = "RGB"
    """R-G-B, stated rather than assumed: 'RGB' as a Pillow mode name already implies the order, but
    a provenance record read by a human years later should not have to know that."""
    output_format: str = "PNG"
    """Lossless by default. `_validate` refuses any format this stage cannot guarantee losslessness
    for -- a normalization stage that quietly re-encoded to JPEG would be an enhancement stage."""
    alpha_policy: AlphaCompositingPolicy = AlphaCompositingPolicy.WHITE_BACKGROUND
    icc_profile_policy: IccProfilePolicy = IccProfilePolicy.RECORD_AND_STRIP_WITHOUT_APPLYING
    apply_exif_orientation: bool = True
    """Physically rotate/flip per EXIF `Orientation`, then normalize the metadata away, so the stored
    pixels are the pixels a reader sees. The one operation permitted to change geometry."""
    strip_metadata: bool = True
    """Emit no EXIF/XMP/text chunks. Metadata that survived into a derived artifact would make the
    derived artifact claim provenance facts about the original."""
    preserve_dimensions: bool = True
    """Width/height must be unchanged, except for an orientation-driven transpose. No resize, no
    crop, no padding, ever."""
    enhancement: str = "none"
    """Structurally pinned to `"none"` by `_validate`. Not a knob -- a declaration, present so that
    the absence of enhancement is an asserted, hashed part of the configuration rather than something
    a reader has to infer from the absence of other fields."""

    @model_validator(mode="after")
    def _validate(self) -> "RgbNormalizationConfig":
        if self.enhancement != "none":
            raise ValueError(
                "RgbNormalizationConfig.enhancement must be 'none' -- this stage performs no "
                "contrast, sharpening, denoising, thresholding, resizing, deskewing, cropping, "
                "colour balancing, or lossy compression (see rgb_normalization.py's contract)"
            )
        if self.target_mode != "RGB":
            raise ValueError("RgbNormalizationConfig.target_mode must be 'RGB'")
        if self.bits_per_channel != 8:
            raise ValueError("RgbNormalizationConfig.bits_per_channel must be 8")
        if self.channel_count != 3:
            raise ValueError("RgbNormalizationConfig.channel_count must be 3")
        if self.channel_order != "RGB":
            raise ValueError("RgbNormalizationConfig.channel_order must be 'RGB'")
        if self.output_format not in _LOSSLESS_OUTPUT_FORMATS:
            raise ValueError(
                f"RgbNormalizationConfig.output_format must be one of "
                f"{sorted(_LOSSLESS_OUTPUT_FORMATS)} -- a lossy re-encode is an enhancement, "
                f"not a normalization; got {self.output_format!r}"
            )
        if not self.preserve_dimensions:
            raise ValueError(
                "RgbNormalizationConfig.preserve_dimensions must be True -- geometry changes other "
                "than an applied EXIF orientation are out of scope for this stage"
            )
        return self

    @property
    def configuration_hash(self) -> str:
        """A deterministic content address over every field, via the domain's existing
        `content_address` helper (`domain/shared/ids.py`) with its own prefix -- no second hashing
        scheme is introduced here, matching `InputCrop`'s and `ProvenanceContextEstablished`'s
        precedent. Field order is fixed by sorting, so the hash depends on values only.
        """
        dumped = self.model_dump(mode="json")
        parts = [f"{key}={dumped[key]}" for key in sorted(dumped)]
        return content_address(*parts, prefix="normalization_config")


_LOSSLESS_OUTPUT_FORMATS = frozenset({"PNG"})
"""The output formats this stage can guarantee are lossless. PNG only, today: it is lossless, it
carries 8-bit RGB natively, and Pillow encodes it deterministically for a fixed input raster."""


class PageImageArtifact(BaseModel):
    """The *original*, un-normalized page image, content-addressed. The `source_*` half of a
    normalization's provenance -- see the module docstring for why this is its own model and not
    fields on `Page`.

    Bytes live out of line (`storage_path`, mirroring `InputCrop.storage_path`); `content_hash` is
    what identifies the content, not where it sits.
    """

    model_config = ConfigDict(frozen=True)

    artifact_id: str
    content_hash: str
    """`page_image_<sha256>` -- computed by `compute_hash`/`create`, never hand-set."""
    page_id: str
    storage_path: str
    byte_size: int

    @model_validator(mode="after")
    def _validate(self) -> "PageImageArtifact":
        if not self.content_hash.startswith("page_image_"):
            raise ValueError(
                "PageImageArtifact.content_hash must be content-addressed via "
                "PageImageArtifact.compute_hash / .create, never hand-set"
            )
        if self.byte_size < 0:
            raise ValueError("PageImageArtifact.byte_size must be >= 0")
        return self

    @staticmethod
    def compute_hash(image_bytes: bytes) -> str:
        return f"page_image_{hashlib.sha256(image_bytes).hexdigest()}"

    @classmethod
    def create(
        cls, *, image_bytes: bytes, page_id: str, storage_path: str
    ) -> "PageImageArtifact":
        return cls(
            artifact_id=new_id("page_image_artifact"),
            content_hash=cls.compute_hash(image_bytes),
            page_id=page_id,
            storage_path=storage_path,
            byte_size=len(image_bytes),
        )


class NormalizedPageArtifact(BaseModel):
    """The complete provenance record of one successful normalization.

    Records the source's observed properties, the output's properties, the policy decisions taken,
    and the exact implementation/version/configuration that took them -- so the transformation is
    reconstructible from the record without re-reading either image, and so a reader can tell what
    was true of the original from what this stage made true of the derived copy.

    The link to the original is `source_artifact_id` + `source_content_hash`: an id reference and a
    content address, never an embedded `PageImageArtifact` (Constitution Article 7).
    """

    model_config = ConfigDict(frozen=True)

    normalized_artifact_id: str
    source_artifact_id: str
    source_content_hash: str
    normalized_content_hash: str
    """`normalized_page_<sha256>` of the emitted bytes."""
    page_id: str
    storage_path: str
    byte_size: int

    # -- Observed properties of the source, before anything was done to it --------------------
    source_color_mode: str
    """Pillow's raw mode string as decoded (`"1"`, `"L"`, `"LA"`, `"P"`, `"CMYK"`, `"RGB"`,
    `"RGBA"`, `"I"`, `"I;16"`, ...) -- the honest observed value, not a normalized label."""
    source_bit_depth: int
    """Bits *per channel* as decoded: 1 for bilevel, 16 for `I;16`, 8 otherwise."""
    source_channel_count: int
    source_width: int
    source_height: int
    source_format: str | None = None
    """The container format Pillow identified (`"PNG"`, `"TIFF"`, ...). `None` when the decoder
    reported none -- never guessed from a filename."""
    source_icc_profile_present: bool = False
    source_icc_profile_hash: str | None = None
    """SHA-256 of the embedded profile's bytes, so the stripped profile stays identifiable. `None`
    exactly when no profile was present."""
    source_icc_profile_description: str | None = None
    """The profile's own declared description, when readable from its header. `None` when absent or
    unparseable -- never a placeholder."""
    source_alpha_present: bool = False
    """True for a real alpha channel *or* a palette `transparency` entry -- both are transparency
    that the compositing policy had to resolve."""
    source_exif_orientation: int | None = None
    """The raw EXIF `Orientation` tag value (1-8), or `None` when the image carried none. `None`
    means "no orientation tag", never "orientation 1 assumed"."""

    # -- Properties of what this stage actually emitted ---------------------------------------
    output_color_mode: str
    output_bit_depth: int
    output_channel_count: int
    output_width: int
    output_height: int
    output_format: str
    output_icc_profile_present: bool = False

    # -- The policy decisions and the code that took them ------------------------------------
    compositing_background: str
    """The literal background the alpha policy composited over, e.g. `"#FFFFFF"` -- the resolved
    value, not the policy name, so the record does not require the enum to interpret."""
    alpha_compositing_applied: bool
    """Whether compositing actually happened. False for a source with no transparency: the policy
    still applied, it simply had nothing to resolve, and saying so is more useful than a record
    implying every image was composited."""
    exif_orientation_applied: bool
    """Whether pixels were physically transposed. False when there was no orientation tag, or the
    tag was 1 (identity)."""
    dimensions_changed_by_orientation: bool
    """True only for the 90/270-degree orientations that legitimately swap width and height -- the
    single sanctioned geometry change, flagged so a dimension difference never looks unexplained."""

    normalization_implementation: str = RGB_NORMALIZATION_IMPLEMENTATION
    normalization_version: str = RGB_NORMALIZATION_VERSION
    configuration_hash: str
    executed_at: str
    warnings: tuple[str, ...] = ()
    """Real, non-fatal observations (e.g. an unparseable ICC header that was still recorded by
    hash). Empty when there were none -- never populated with reassurance."""

    @model_validator(mode="after")
    def _validate(self) -> "NormalizedPageArtifact":
        if not self.normalized_content_hash.startswith("normalized_page_"):
            raise ValueError(
                "NormalizedPageArtifact.normalized_content_hash must be content-addressed via "
                "NormalizedPageArtifact.compute_hash, never hand-set"
            )
        if not self.source_content_hash.startswith("page_image_"):
            raise ValueError(
                "NormalizedPageArtifact.source_content_hash must be a PageImageArtifact content "
                "address"
            )
        # There is deliberately **no** check that the source and normalized digests differ.
        #
        # An earlier revision of this model rejected equal digests, on the reasoning that a derived
        # artifact must be distinguishable from its original. That was wrong, and
        # `tests/htr/preprocessing/test_idempotence.py` caught it: a source that is *already* in this
        # stage's exact canonical form (8-bit RGB PNG, no profile, no metadata, same encoder
        # parameters) legitimately re-encodes to byte-identical output, so its digest legitimately
        # coincides. Rejecting that would have turned "this page needed no changes" -- a true and
        # useful fact -- into a validation error, and would have made the stage unable to record its
        # own output being fed back through it.
        #
        # The two *hashes* are still always distinct strings, because their prefixes differ
        # (`page_image_...` vs `normalized_page_...`), so no lookup can confuse an original with a
        # derived artifact even when the underlying bytes are the same.
        if self.output_color_mode != "RGB":
            raise ValueError("NormalizedPageArtifact.output_color_mode must be 'RGB'")
        if self.output_channel_count != 3 or self.output_bit_depth != 8:
            raise ValueError(
                "NormalizedPageArtifact output must be 3 channels at 8 bits per channel"
            )
        if self.output_width < 1 or self.output_height < 1:
            raise ValueError("NormalizedPageArtifact output dimensions must be >= 1")
        return self

    @staticmethod
    def compute_hash(image_bytes: bytes) -> str:
        return f"normalized_page_{hashlib.sha256(image_bytes).hexdigest()}"


class NormalizationFailure(BaseModel):
    """A normalization attempt that could not complete -- preserved as durable evidence, never a
    swallowed exception (docs/htr-domain-design.md §1's "FailureRecord ... preserved, never
    excluded", applied at this stage's granularity).

    Its existence is the mechanism that makes "never fall back to the unnormalized original" real:
    the export path has a recorded failure to refuse on, rather than an absence it might read as
    success.
    """

    model_config = ConfigDict(frozen=True)

    failure_id: str
    page_id: str
    source_content_hash: str | None = None
    """`None` when the bytes could not even be hashed; otherwise the original's content address,
    so a failed attempt is still traceable to the input that caused it."""
    category: NormalizationFailureCategory
    reason: str
    configuration_hash: str
    normalization_version: str = RGB_NORMALIZATION_VERSION
    occurred_at: str

    @model_validator(mode="after")
    def _validate(self) -> "NormalizationFailure":
        if not self.reason.strip():
            raise ValueError("NormalizationFailure.reason must be non-empty")
        return self

    @classmethod
    def create(
        cls,
        *,
        page_id: str,
        category: NormalizationFailureCategory,
        reason: str,
        configuration_hash: str,
        occurred_at: str,
        source_content_hash: str | None = None,
    ) -> "NormalizationFailure":
        return cls(
            failure_id=new_id("normalization_failure"),
            page_id=page_id,
            source_content_hash=source_content_hash,
            category=category,
            reason=reason,
            configuration_hash=configuration_hash,
            occurred_at=occurred_at,
        )


class NormalizationError(RuntimeError):
    """Raised by `rgb_normalization.normalize_page_image` when normalization fails.

    Carries the `NormalizationFailure` record so a caller cannot handle the error without having the
    durable evidence in hand -- the same discipline `associate_external_import` applies by taking a
    real `ExternalImport` rather than an id.
    """

    def __init__(self, failure: NormalizationFailure) -> None:
        super().__init__(f"{failure.category.value}: {failure.reason}")
        self.failure = failure
