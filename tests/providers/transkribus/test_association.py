"""Tests for the "associate a Transkribus result with an existing ArchiveTrust document, page,
region, or experiment" mechanism (docs/htr-migration-plan.md Stage 8 item 4)."""

from __future__ import annotations

import pytest

from archivetrust.providers.transkribus.external_import import (
    ExternalImport,
    ExternalImportFormat,
    ImportAssociation,
    ImportTargetKind,
    associate_external_import,
)


def _sample_import() -> ExternalImport:
    return ExternalImport.create(
        method_run_id="method_run_1",
        source_file_path="/imports/volume_12_folio_003r.page.xml",
        export_format=ExternalImportFormat.PAGE_XML,
        imported_by="curator-1",
        imported_at="2026-06-12T09:50:00Z",
        vendor_reported_accuracy=0.912,
    )


def test_associate_with_a_document():
    external_import = _sample_import()
    association = associate_external_import(
        external_import,
        target_kind=ImportTargetKind.DOCUMENT,
        target_id="archive_object_abc123",
        associated_at="2026-06-12T10:00:00Z",
        associated_by="curator-1",
    )
    assert isinstance(association, ImportAssociation)
    assert association.external_import_id == external_import.external_import_id
    assert association.target_kind is ImportTargetKind.DOCUMENT
    assert association.target_id == "archive_object_abc123"
    assert association.associated_by == "curator-1"


@pytest.mark.parametrize(
    "target_kind",
    [ImportTargetKind.DOCUMENT, ImportTargetKind.PAGE, ImportTargetKind.REGION, ImportTargetKind.EXPERIMENT],
)
def test_associate_with_every_target_kind(target_kind):
    external_import = _sample_import()
    association = associate_external_import(
        external_import,
        target_kind=target_kind,
        target_id="some_target_id",
        associated_at="2026-06-12T10:00:00Z",
    )
    assert association.target_kind is target_kind


def test_association_references_by_id_never_embeds_the_import():
    external_import = _sample_import()
    association = associate_external_import(
        external_import,
        target_kind=ImportTargetKind.PAGE,
        target_id="page_1",
        associated_at="2026-06-12T10:00:00Z",
    )
    dumped = association.model_dump()
    # Graph-Reference Rule (Constitution Article 7): a reference by id, never an inline copy.
    assert dumped["external_import_id"] == external_import.external_import_id
    assert "source_file_path" not in dumped
    assert "export_format" not in dumped


def test_two_associations_for_the_same_import_are_independent_append_only_records():
    external_import = _sample_import()
    first = associate_external_import(
        external_import,
        target_kind=ImportTargetKind.PAGE,
        target_id="page_1",
        associated_at="2026-06-12T10:00:00Z",
    )
    second = associate_external_import(
        external_import,
        target_kind=ImportTargetKind.EXPERIMENT,
        target_id="experiment_1",
        associated_at="2026-06-12T10:05:00Z",
    )
    assert first.association_id != second.association_id
    assert first.external_import_id == second.external_import_id == external_import.external_import_id


def test_empty_target_id_is_rejected():
    external_import = _sample_import()
    with pytest.raises(ValueError):
        associate_external_import(
            external_import,
            target_kind=ImportTargetKind.DOCUMENT,
            target_id="",
            associated_at="2026-06-12T10:00:00Z",
        )


def test_vendor_reported_accuracy_is_kept_on_the_import_never_folded_into_the_association():
    external_import = _sample_import()
    assert external_import.vendor_reported_accuracy == pytest.approx(0.912)
    association = associate_external_import(
        external_import,
        target_kind=ImportTargetKind.DOCUMENT,
        target_id="archive_object_abc123",
        associated_at="2026-06-12T10:00:00Z",
    )
    assert not hasattr(association, "vendor_reported_accuracy")
