"""Transkribus page preparation ViewModel: normalization and manual-handover status per page.

Framework-independent, like every other ViewModel in this package -- it constructs plain frozen
Pydantic rows and touches no Qt. A focused extension of the Transkribus-facing presentation surface
(`presentation/htr_methods_viewmodel.py` covers the *method*; this covers a *page*'s journey through
the manual round trip), not a page rebuild.

**The state labels are fixed and load-bearing.** `PreparationState`'s five values are exactly the
labels specified for this surface, and `.label` returns them verbatim. They are an enum rather than
free strings so a UI cannot invent a sixth state, and so the "did ArchiveTrust upload this?" question
has exactly one answer everywhere it is asked.

**Nothing here may imply ArchiveTrust performed an upload.** `UPLOADED_EXTERNALLY` means a human
recorded that *they* uploaded it, and `upload_actor` says who. This reuses the established
"external service, manual only" framing from `providers/transkribus/README.md`
(quoted in `EXTERNAL_UPLOAD_DISCLAIMER`) rather than paraphrasing it into wording that could drift
into implying automation. ArchiveTrust has no upload code path at all -- see
`htr/preprocessing/export_package.py`'s module docstring.

**Reads a store, never a filesystem or a network.** Every fact below comes from the
`HtrResearchStore` projection (which is itself rebuilt from telemetry), so this surface cannot show a
normalization that has no durable event behind it.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict

from archivetrust.htr.preprocessing.models import (
    NormalizationFailure,
    NormalizedPageArtifact,
)
from archivetrust.providers.transkribus.external_import import (
    CorrespondenceBasis,
    ImportAssociation,
    ImportTargetKind,
)

EXTERNAL_UPLOAD_DISCLAIMER = (
    "Transkribus is an external service. ArchiveTrust never uploads anything to it: a researcher "
    "must open Transkribus themselves, upload the prepared images, run Swedish Lion I, and export "
    "the result back to local disk manually."
)
"""Shown alongside any upload-related state. Deliberately the same claim as
`providers/transkribus/README.md` §"What 'manual import mode' means, concretely" and
`htr/preprocessing/export_package.py::EXTERNAL_PROCESSING_BOUNDARY`."""


class PreparationState(str, Enum):
    """Where one page is in the normalize -> hand over -> import -> map round trip.

    Ordered by progression, so `>=` comparisons on the members' index are meaningful to a caller that
    wants "at least uploaded". Each value's label is the exact specified wording.
    """

    NORMALIZATION_REQUIRED = "normalization_required"
    READY_FOR_EXTERNAL_UPLOAD = "ready_for_external_upload"
    UPLOADED_EXTERNALLY = "uploaded_externally"
    RESULT_IMPORTED = "result_imported"
    RESULT_MAPPING_VERIFIED = "result_mapping_verified"

    @property
    def label(self) -> str:
        """The exact display label. Verbatim, not title-cased from the value, so the specified
        wording cannot drift through a formatting helper."""
        return _STATE_LABELS[self]


_STATE_LABELS = {
    PreparationState.NORMALIZATION_REQUIRED: "Normalization required",
    PreparationState.READY_FOR_EXTERNAL_UPLOAD: "Ready for external upload",
    PreparationState.UPLOADED_EXTERNALLY: "Uploaded externally",
    PreparationState.RESULT_IMPORTED: "Result imported",
    PreparationState.RESULT_MAPPING_VERIFIED: "Result mapping verified",
}

_STATE_ORDER = tuple(PreparationState)


class PagePreparationRow(BaseModel):
    """Everything the Transkribus preparation surface shows for one page."""

    model_config = ConfigDict(frozen=True)

    page_id: str
    state: PreparationState
    state_label: str
    """`state.label`, denormalized onto the row so a template needs no enum lookup."""

    rgb_normalization_required: bool
    """Whether this page's pipeline configuration requires normalization. Always shown, including
    when `True` and unsatisfied -- that combination is the whole point of the first state."""
    normalization_version: str | None = None
    configuration_hash: str | None = None

    source_image_mode: str | None = None
    """The source image's observed colour mode (`"CMYK"`, `"P"`, `"I;16"`, ...). `None` when this page
    has not been normalized, meaning "not yet observed" -- never defaulted to `"RGB"`, which would be
    a guess about the very thing this stage exists to establish."""
    source_image_bit_depth: int | None = None

    normalization_status: str
    """A short human phrase: `"Not normalized"`, `"Normalized"`, or `"Failed: <category>"`. Distinct
    from `state`, which describes the *round trip*; a page can be `Normalized` and still be at
    `Ready for external upload`."""
    normalized_artifact_hash: str | None = None
    normalization_failure_category: str | None = None
    normalization_failure_reason: str | None = None

    preparation_package_status: str
    """`"Not prepared"` or `"Prepared: <package_id>"`."""
    export_package_id: str | None = None

    external_upload_status: str
    """`"Not uploaded (manual, external)"` or `"Uploaded externally by <actor>"`. Never phrased as an
    action ArchiveTrust took."""
    upload_actor: str | None = None
    external_upload_disclaimer: str = EXTERNAL_UPLOAD_DISCLAIMER

    imported_result_status: str
    """`"No result imported"` or `"Result imported"`."""
    external_import_id: str | None = None

    mapping_status: str
    """`"Not mapped"`, or `"Mapped (researcher-confirmed, not cryptographic proof)"`. The caveat is
    part of the label because a bare "Verified" next to two content hashes would read as a hash
    comparison that never happened."""
    correspondence_basis: str | None = None


class TranskribusPreparationViewModel:
    """Projects a page's normalization + manual-handover status into `PagePreparationRow`s.

    Inputs are passed in by the caller (the composition root), never discovered here -- the same rule
    `MethodOverviewViewModel` follows, and what keeps this constructible in a test with a plain
    `HtrResearchStore`.

    `uploaded_page_ids` and `package_ids_by_page` are supplied rather than read from a store because
    neither fact is ArchiveTrust's to know on its own: an external upload is something a human tells
    the system happened, and which package a page went out in is known by whoever built it. Modelling
    them as inputs keeps this ViewModel from implying it observed an upload.
    """

    def __init__(
        self,
        store,
        *,
        normalization_required_page_ids: frozenset[str] | set[str] = frozenset(),
        package_ids_by_page: dict[str, str] | None = None,
        uploaded_page_ids: dict[str, str] | None = None,
        associations: tuple[ImportAssociation, ...] = (),
    ) -> None:
        self._store = store
        self._required = frozenset(normalization_required_page_ids)
        self._packages = package_ids_by_page or {}
        self._uploaded = uploaded_page_ids or {}
        """`page_id -> the researcher who uploaded it`. A name, not a boolean, because
        "Uploaded externally" is meaningless without knowing who did it."""
        self._associations = associations

    def page_rows(self, page_ids: tuple[str, ...]) -> tuple[PagePreparationRow, ...]:
        return tuple(self._row(page_id) for page_id in page_ids)

    def _row(self, page_id: str) -> PagePreparationRow:
        artifacts = self._store.normalized_page_artifacts(page_id=page_id)
        failures = self._store.normalization_failures(page_id=page_id)
        artifact: NormalizedPageArtifact | None = artifacts[-1] if artifacts else None
        failure: NormalizationFailure | None = failures[-1] if failures else None

        association = self._association_for(artifact)
        package_id = self._packages.get(page_id)
        upload_actor = self._uploaded.get(page_id)

        # State progression. Each step requires the previous one's evidence to exist, so a state can
        # never be reached by assertion alone -- e.g. "Result mapping verified" requires a real
        # ImportAssociation whose target is the real normalized artifact.
        state = PreparationState.NORMALIZATION_REQUIRED
        if artifact is not None:
            state = PreparationState.READY_FOR_EXTERNAL_UPLOAD
            if upload_actor is not None:
                state = PreparationState.UPLOADED_EXTERNALLY
            if association is not None:
                state = (
                    PreparationState.RESULT_MAPPING_VERIFIED
                    if association.correspondence_basis is not None
                    else PreparationState.RESULT_IMPORTED
                )

        if artifact is not None:
            normalization_status = "Normalized"
        elif failure is not None:
            normalization_status = f"Failed: {failure.category.value}"
        else:
            normalization_status = "Not normalized"

        return PagePreparationRow(
            page_id=page_id,
            state=state,
            state_label=state.label,
            rgb_normalization_required=page_id in self._required,
            normalization_version=artifact.normalization_version if artifact else None,
            configuration_hash=(
                artifact.configuration_hash
                if artifact
                else (failure.configuration_hash if failure else None)
            ),
            source_image_mode=artifact.source_color_mode if artifact else None,
            source_image_bit_depth=artifact.source_bit_depth if artifact else None,
            normalization_status=normalization_status,
            normalized_artifact_hash=(
                artifact.normalized_content_hash if artifact else None
            ),
            normalization_failure_category=(
                failure.category.value if artifact is None and failure else None
            ),
            normalization_failure_reason=(
                failure.reason if artifact is None and failure else None
            ),
            preparation_package_status=(
                f"Prepared: {package_id}" if package_id else "Not prepared"
            ),
            export_package_id=package_id,
            external_upload_status=(
                f"Uploaded externally by {upload_actor}"
                if upload_actor
                else "Not uploaded (manual, external)"
            ),
            upload_actor=upload_actor,
            imported_result_status=(
                "Result imported" if association is not None else "No result imported"
            ),
            external_import_id=(
                association.external_import_id if association is not None else None
            ),
            mapping_status=(
                "Mapped (researcher-confirmed, not cryptographic proof)"
                if association is not None
                and association.correspondence_basis is CorrespondenceBasis.RESEARCHER_CONFIRMED
                else "Mapped (hash-verified)"
                if association is not None
                and association.correspondence_basis is CorrespondenceBasis.HASH_VERIFIED
                else "Not mapped"
            ),
            correspondence_basis=(
                association.correspondence_basis.value
                if association is not None and association.correspondence_basis is not None
                else None
            ),
        )

    def _association_for(
        self, artifact: NormalizedPageArtifact | None
    ) -> ImportAssociation | None:
        """The association pointing at this page's normalized artifact, if any.

        Matched on the artifact's own id, not on the page id: an association records which *bytes* a
        result came from, and a page with two normalized artifacts (a re-normalization under a new
        version) must not show its old artifact's mapping against its new one.
        """
        if artifact is None:
            return None
        for association in self._associations:
            if (
                association.target_kind is ImportTargetKind.NORMALIZED_PAGE_ARTIFACT
                and association.target_id == artifact.normalized_artifact_id
            ):
                return association
        return None
