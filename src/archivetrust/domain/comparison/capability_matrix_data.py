"""The real, ratified Capability Matrix (Operational Hardening milestone, Priority 11).

Transcribes `docs/investigation/CAPABILITY_MATRIX.md` (Milestone 0's deliverable, "Complete,
supersedes the illustrative matrix in ROADMAP.md §9") into the machine-readable
`CapabilityMatrix` the Comparison Engine actually reads. The desktop composition root previously
passed `CapabilityMatrix(matrix_version=1, entries=())` -- an empty matrix defaults every
provider/type pair to `Capability.NO` (weight 0.0), which meant no structural edge could ever be
elected (`edge_accept_threshold=0.5` against zero support) regardless of what Docling or Tesseract
actually claim. This module is the fix: the same investigation work, wired in, not a new capability
model (no new concept beyond what `CAPABILITY_MATRIX.md` already rated).

**Provider versions pinned to what the desktop composition root actually registers**
(`clients/desktop/composition.py`: `DoclingAdapter(..., provider_version="2.x")`,
`TesseractLayoutParserAdapter(..., provider_version="5.x")`) -- `CapabilityMatrixEntry` is keyed on
`(provider_id, provider_version)` by design (a version bump requires an explicit new entry, never
silent inheritance), so entries must match those exact strings or they silently default to `NO`.

**Qwen2.5-VL is not included.** Its adapter reports `provider_version=raw_response.model_version`
(`providers/qwen_vl/adapter.py`) -- a value read from the runtime at invocation time, not a fixed
string this static matrix can pin ahead of time. An uncovered `(provider_id, provider_version)`
pair already defaults to `NO` (`CapabilityMatrix.capability_for`'s own documented behavior) rather
than silently assuming capability, so this omission is honest, not a special case: Qwen2.5-VL's
structural/coverage claims are conservatively rated `NO` until a versioned entry can be added for
whatever `model_version` a deployment actually pins.
"""

from __future__ import annotations

from archivetrust.domain.comparison.capability_matrix import (
    Capability,
    CapabilityMatrix,
    CapabilityMatrixEntry,
)
from archivetrust.domain.ontology.types import ObservationType

CAPABILITY_MATRIX_VERSION = 2

DOCLING_PROVIDER_VERSION = "2.x"
TESSERACT_LAYOUTPARSER_PROVIDER_VERSION = "5.x"

# Native/Derived/Partial/No per CAPABILITY_MATRIX.md's table. Types the matrix doesn't rate at all
# for a provider (e.g. Metadata for Tesseract+LayoutParser: "No") are recorded explicitly below
# rather than left to the default -- an entry that says "No" is a different, audited fact from no
# entry existing (Article 6: Full Exposure applies to the rating itself, not only to Observations).
_DOCLING_RATINGS: dict[ObservationType, Capability] = {
    ObservationType.PAGE: Capability.NATIVE,
    ObservationType.HEADING: Capability.NATIVE,
    ObservationType.SECTION: Capability.NATIVE,
    ObservationType.PARAGRAPH: Capability.NATIVE,
    ObservationType.METADATA: Capability.PARTIAL,
    ObservationType.ARCHIVE_BOUNDARY: Capability.NO,
    ObservationType.TABLE: Capability.NATIVE,
    ObservationType.CAPTION: Capability.PARTIAL,
    ObservationType.IMAGE: Capability.NATIVE,
    ObservationType.FOOTNOTE: Capability.NATIVE,
    ObservationType.LAYOUT_REGION: Capability.NATIVE,
}

_TESSERACT_LAYOUTPARSER_RATINGS: dict[ObservationType, Capability] = {
    ObservationType.PAGE: Capability.NATIVE,
    ObservationType.HEADING: Capability.PARTIAL,
    ObservationType.SECTION: Capability.DERIVED,
    ObservationType.PARAGRAPH: Capability.NATIVE,
    ObservationType.METADATA: Capability.NO,
    ObservationType.ARCHIVE_BOUNDARY: Capability.NO,
    ObservationType.TABLE: Capability.PARTIAL,
    ObservationType.CAPTION: Capability.NO,
    ObservationType.IMAGE: Capability.NATIVE,
    ObservationType.FOOTNOTE: Capability.NO,
    ObservationType.LAYOUT_REGION: Capability.NATIVE,
}


def _entries_for(
    provider_id: str, provider_version: str, ratings: dict[ObservationType, Capability]
) -> tuple[CapabilityMatrixEntry, ...]:
    return tuple(
        CapabilityMatrixEntry(
            provider_id=provider_id,
            provider_version=provider_version,
            observation_type=observation_type,
            capability=capability,
        )
        for observation_type, capability in ratings.items()
    )


def production_capability_matrix() -> CapabilityMatrix:
    """The `CapabilityMatrix` the desktop composition root wires into
    `WorkspaceProcessingService`/`run_comparison_engine` -- every entry backed by
    `docs/investigation/CAPABILITY_MATRIX.md`, none invented for this milestone."""
    entries = _entries_for("docling", DOCLING_PROVIDER_VERSION, _DOCLING_RATINGS) + _entries_for(
        "tesseract_layoutparser", TESSERACT_LAYOUTPARSER_PROVIDER_VERSION, _TESSERACT_LAYOUTPARSER_RATINGS
    )
    return CapabilityMatrix(matrix_version=CAPABILITY_MATRIX_VERSION, entries=entries)
