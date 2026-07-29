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

    @model_validator(mode="after")
    def _validate(self) -> "ImportAssociation":
        if not self.target_id:
            raise ValueError("ImportAssociation.target_id must be non-empty")
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
