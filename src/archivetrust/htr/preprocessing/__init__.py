"""Image preprocessing as an independent, versioned pipeline stage.

Sibling to `htr/corpus/`, `htr/experiment/` and `htr/knowledge/`, following this codebase's
one-package-per-concern convention, and following `docs/htr-domain-design.md` §7's principle that a
pipeline stage belongs to the pipeline rather than to whichever method happens to need it first.

**This `__init__` deliberately re-exports `models` only, never `rgb_normalization`.**
`domain/telemetry/events.py` imports `htr.preprocessing.models` so its events can carry the full
typed artifact (as `InputCropCreated` already carries an `InputCrop`), and importing this package is
what executes this file. Re-exporting `rgb_normalization` here would therefore drag Pillow into every
`domain.telemetry` import and break ROADMAP.md S11's "domain layer has zero third-party runtime
dependencies beyond schema/validation library". Callers that need the transform import
`archivetrust.htr.preprocessing.rgb_normalization` explicitly -- one extra line at four call sites,
in exchange for a domain layer that stays honest about what it depends on.
"""

from __future__ import annotations

from archivetrust.htr.preprocessing.models import (
    RGB_NORMALIZATION_IMPLEMENTATION,
    RGB_NORMALIZATION_STAGE,
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

__all__ = [
    "RGB_NORMALIZATION_IMPLEMENTATION",
    "RGB_NORMALIZATION_STAGE",
    "RGB_NORMALIZATION_VERSION",
    "AlphaCompositingPolicy",
    "IccProfilePolicy",
    "NormalizationError",
    "NormalizationFailure",
    "NormalizationFailureCategory",
    "NormalizedPageArtifact",
    "PageImageArtifact",
    "RgbNormalizationConfig",
]
