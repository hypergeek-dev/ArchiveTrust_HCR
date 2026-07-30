"""`ExternalImport` (docs/htr-domain-design.md §1, §5): Transkribus manual-import provenance.

Stage 1 (docs/htr-migration-plan.md Stage 1 item 6) added the bare fields. This module is now
extended for Stage 8 (the full adapter: PAGE/ALTO/plain-text parsing, no-auto-fetch enforcement,
association-to-corpus-entity). This module must never make a network call -- it has none to make,
in this phase or any planned future one covered by manual-import mode; API mode (if ever built) is
explicitly out of scope here (see `providers/transkribus/README.md`).

**Stage 8 extensions to `ExternalImport` (documented per the task's "extend it, document what you
added and why" instruction, rather than creating a second competing model):**

- `ExternalImportFormat.PLAIN_TEXT` added alongside `PAGE_XML`/`ALTO_XML` -- Stage 8 explicitly
  requires plain-text as a third supported manual-import format.
- `vendor_reported_accuracy: float | None` added -- when a Transkribus export states its own
  accuracy/CER figure (rare, but real PAGE XML `MetadataItem` custom fields can carry one), it
  must be recorded, but *never* merged into or compared directly against ArchiveTrust's own
  CER/WER metrics (the task brief's explicit caveat). Keeping it as its own distinctly-named field
  on `ExternalImport`, separate from anything `evaluation/metrics.py` produces, is what enforces
  that separation structurally rather than just by convention.
- `ImportTargetKind` + `ImportAssociation` + `associate_external_import` added below -- the
  "associate a Transkribus result with an existing ArchiveTrust document/page/region/experiment"
  mechanism required by Stage 8 item 4.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, model_validator

from archivetrust.domain.shared.ids import new_id


class ExternalImportFormat(str, Enum):
    """Which manual-export format the imported file is in."""

    PAGE_XML = "page_xml"
    ALTO_XML = "alto_xml"
    PLAIN_TEXT = "plain_text"


class ExternalImport(BaseModel):
    """One manually-imported Transkribus export, wrapping the `MethodRun` it produced (§1:
    "ExternalImport -- Transkribus manual-import provenance wrapper around a MethodRun"). No
    network call is ever made without explicit user action -- this type only records that a
    human already exported and supplied a file; it has no fetch capability itself."""

    model_config = ConfigDict(frozen=True)

    external_import_id: str
    method_run_id: str
    source_file_path: str
    export_format: ExternalImportFormat
    imported_by: str
    imported_at: str
    transkribus_document_id: str | None = None
    """Transkribus's own document/collection identifier, when present in the export, kept as
    audit-only, non-branchable metadata (mirroring `LayoutRegionPayload.native_label`'s
    discipline -- Constitution Article 20: no downstream logic may special-case a provider)."""
    vendor_reported_accuracy: float | None = None
    """A vendor-reported accuracy/CER figure, only when the export file itself states one -- see
    module docstring. Never fabricated when absent, never merged into ArchiveTrust's own metrics."""

    @classmethod
    def create(
        cls,
        *,
        method_run_id: str,
        source_file_path: str,
        export_format: ExternalImportFormat,
        imported_by: str,
        imported_at: str,
        transkribus_document_id: str | None = None,
        vendor_reported_accuracy: float | None = None,
    ) -> "ExternalImport":
        return cls(
            external_import_id=new_id("external_import"),
            method_run_id=method_run_id,
            source_file_path=source_file_path,
            export_format=export_format,
            imported_by=imported_by,
            imported_at=imported_at,
            transkribus_document_id=transkribus_document_id,
            vendor_reported_accuracy=vendor_reported_accuracy,
        )


class ImportTargetKind(str, Enum):
    """Which kind of existing ArchiveTrust entity a Transkribus `ExternalImport` is being
    associated with (Stage 8 item 4: "associate a Transkribus result with an existing ArchiveTrust
    document, page, region, or experiment")."""

    DOCUMENT = "document"
    PAGE = "page"
    REGION = "region"
    EXPERIMENT = "experiment"
    NORMALIZED_PAGE_ARTIFACT = "normalized_page_artifact"
    """Added with the RGB-normalization stage (`htr/preprocessing/`). The target is a specific
    `NormalizedPageArtifact` -- the exact derived image that was handed to Transkribus -- identified
    by its content hash. Distinct from `PAGE` on purpose: a page may have several normalized
    artifacts over time (a new normalization version produces a new one), and "this result came from
    that page" is a weaker, less useful claim than "this result came from those exact bytes"."""


class CorrespondenceBasis(str, Enum):
    """*How well established* an association is -- the honest epistemic status of the link.

    Added with the RGB-normalization stage because the export/import round trip creates a link that
    genuinely cannot be proven, and the type system should say so rather than let a reader assume a
    content hash on both sides means the hashes were compared.
    """

    RESEARCHER_CONFIRMED = "researcher_confirmed"
    """A human asserts this imported result was produced from this normalized artifact, on the basis
    of their own record of which file they uploaded.

    **This is not cryptographic proof and must never be presented as such.** Transkribus does not
    preserve, echo, or return ArchiveTrust's content hashes, so nothing in the imported PAGE/ALTO
    file can be checked against the normalized image's hash. The export manifest
    (`htr/preprocessing/export_package.py`) records which normalized hash was written to which
    filename, which is what makes the researcher's claim *auditable* -- but the final link is still
    testimony, not verification."""

    HASH_VERIFIED = "hash_verified"
    """The imported artifact itself carries a content hash that was recomputed and matched. No
    Transkribus import can currently use this value -- it exists so that `RESEARCHER_CONFIRMED` is a
    meaningful, contrastive choice rather than the only option the enum offers, and so a future
    import path that *can* verify has somewhere honest to record it."""


class ImportAssociation(BaseModel):
    """The linkage record produced by associating one `ExternalImport` with one existing
    ArchiveTrust entity. A separate, id-referencing record -- not a mutation of `ExternalImport`
    (which stays frozen and import-file-scoped) and not an embedded copy of the target entity
    (Constitution Article 7: Graph Structure Is Not Duplicated as Payload Content) -- consistent
    with docs/htr-domain-design.md §4's "implemented as a linked sequence of content-addressed ids
    rather than foreign keys" discipline. One `ExternalImport` may be associated with more than one
    target over time (e.g. first a `Page`, later also the `Experiment` it was folded into); each
    call to `associate_external_import` produces one additional, independent `ImportAssociation`,
    never overwrites a prior one -- append-only, matching the rest of this codebase's
    supersede/append-not-overwrite discipline.
    """

    model_config = ConfigDict(frozen=True)

    association_id: str
    external_import_id: str
    target_kind: ImportTargetKind
    target_id: str
    associated_at: str
    associated_by: str | None = None
    correspondence_basis: CorrespondenceBasis | None = None
    """How well established this link is -- see `CorrespondenceBasis`. Added with the
    RGB-normalization stage and additive: `None` on every association recorded before this field
    existed, which honestly means "the basis was not recorded", never a retroactive claim that one was
    verified. Always set for a `NORMALIZED_PAGE_ARTIFACT` association, enforced below."""
    source_content_hash: str | None = None
    """The normalized artifact's content hash this result is claimed to have come from. Recorded
    alongside `target_id` because the hash is what the export manifest gave the researcher and
    therefore what they actually confirmed against."""
    export_package_id: str | None = None
    """Which export package the uploaded file came from, when known -- the audit trail from an
    imported result back to the handover it belongs to."""

    @model_validator(mode="after")
    def _validate(self) -> "ImportAssociation":
        if not self.target_id:
            raise ValueError("ImportAssociation.target_id must be non-empty")
        if (
            self.target_kind is ImportTargetKind.NORMALIZED_PAGE_ARTIFACT
            and self.correspondence_basis is None
        ):
            raise ValueError(
                "ImportAssociation.correspondence_basis is required when target_kind is "
                "NORMALIZED_PAGE_ARTIFACT -- an unqualified link to specific image bytes would "
                "read as verified when it is researcher-confirmed testimony"
            )
        return self


def associate_external_import(
    external_import: ExternalImport,
    *,
    target_kind: ImportTargetKind,
    target_id: str,
    associated_at: str,
    associated_by: str | None = None,
) -> ImportAssociation:
    """Associates a parsed Transkribus `ExternalImport` with an existing ArchiveTrust document,
    page, region, or experiment (Stage 8 item 4). This is the concrete "association mechanism" the
    task requires: given an `ExternalImport` (already produced by `adapter.py::recognize` +
    parsing) and a target reference, produce the linkage as a new `ImportAssociation` record.

    Deliberately takes the already-constructed `ExternalImport`, not just its id: forces the
    caller to have a real `ExternalImport` in hand (evidence-chain traceability, per
    docs/htr-domain-design.md §4), not merely a string that happens to look like one.
    """
    return ImportAssociation(
        association_id=new_id("import_association"),
        external_import_id=external_import.external_import_id,
        target_kind=target_kind,
        target_id=target_id,
        associated_at=associated_at,
        associated_by=associated_by,
    )


def associate_imported_result_with_normalized_page(
    external_import: ExternalImport,
    *,
    normalized_artifact_id: str,
    normalized_content_hash: str,
    confirmed_by: str,
    associated_at: str,
    export_package_id: str | None = None,
) -> ImportAssociation:
    """Records that an imported Transkribus result corresponds to a specific normalized page image.

    The closing step of the export/import round trip
    (`htr/preprocessing/export_package.py` -> external, manual Transkribus processing -> import).
    Returns an `ImportAssociation` whose `correspondence_basis` is fixed to
    `RESEARCHER_CONFIRMED` -- there is no parameter to override it, because no Transkribus import can
    honestly claim anything stronger: Transkribus never sees or returns ArchiveTrust's content hashes,
    so the link rests on the researcher's own record of which file they uploaded, cross-checkable
    against the export manifest but not verifiable from the imported file itself.

    `confirmed_by` is required, not optional. An unattributed "researcher-confirmed" correspondence
    would be a claim with no claimant, which is worse than no record: the whole value of this
    association is knowing *who* vouched for it.
    """
    if not confirmed_by.strip():
        raise ValueError(
            "associate_imported_result_with_normalized_page requires confirmed_by -- a "
            "researcher-confirmed correspondence must name the researcher who confirmed it"
        )
    if not normalized_content_hash.startswith("normalized_page_"):
        raise ValueError(
            "normalized_content_hash must be a NormalizedPageArtifact content address "
            f"('normalized_page_...'); got {normalized_content_hash!r}"
        )
    return ImportAssociation(
        association_id=new_id("import_association"),
        external_import_id=external_import.external_import_id,
        target_kind=ImportTargetKind.NORMALIZED_PAGE_ARTIFACT,
        target_id=normalized_artifact_id,
        associated_at=associated_at,
        associated_by=confirmed_by,
        correspondence_basis=CorrespondenceBasis.RESEARCHER_CONFIRMED,
        source_content_hash=normalized_content_hash,
        export_package_id=export_package_id,
    )
