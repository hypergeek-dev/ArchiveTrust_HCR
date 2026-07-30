"""Export package + manifest, the researcher-confirmed import association, and the ViewModel."""

from __future__ import annotations

import json

import pytest

from archivetrust.htr.persistence import DurableHtrResearchStore
from archivetrust.htr.preprocessing.export_package import (
    MANIFEST_FILENAME,
    ExportPackageManifest,
    PageImageSelection,
    build_export_package,
    read_manifest,
)
from archivetrust.htr.preprocessing.models import NormalizationError
from archivetrust.htr.preprocessing.normalization_service import NormalizationService
from archivetrust.infrastructure.storage.blob_store import ContentAddressedBlobStore
from archivetrust.infrastructure.storage.telemetry_sink import InMemoryTelemetrySink
from archivetrust.presentation.htr_transkribus_preparation_viewmodel import (
    PreparationState,
    TranskribusPreparationViewModel,
)
from archivetrust.providers.transkribus.external_import import (
    CorrespondenceBasis,
    ExternalImport,
    ExternalImportFormat,
    ImportTargetKind,
    associate_imported_result_with_normalized_page,
)
from tests.htr.preprocessing import _images

AT = "2026-07-30T09:00:00+00:00"


@pytest.fixture()
def service(tmp_path):
    store = DurableHtrResearchStore(InMemoryTelemetrySink())
    return NormalizationService(
        blob_store=ContentAddressedBlobStore(tmp_path / "blobs"), store=store
    ), store


def _selections():
    return (
        PageImageSelection(
            page_id="page_1",
            image_bytes=_images.cmyk(),
            archive_object_ref="archive_object_lion",
            page_number=1,
            original_filename="folio_001.tif",
        ),
        PageImageSelection(
            page_id="page_2",
            image_bytes=_images.grayscale_16bit(),
            archive_object_ref="archive_object_lion",
            page_number=2,
        ),
    )


# -- Export package -----------------------------------------------------------------------------


def test_export_package_writes_normalized_images_and_a_manifest(tmp_path, service):
    svc, _ = service

    package = build_export_package(_selections(), destination=tmp_path / "out", service=svc)

    assert package.directory.is_dir()
    assert (package.directory / MANIFEST_FILENAME).is_file()
    written = sorted(p.name for p in package.directory.iterdir())
    assert written == [
        "archive_object_lion_p0001.png",
        "archive_object_lion_p0002.png",
        MANIFEST_FILENAME,
    ]
    # The written files are the normalized bytes, read back through the content-addressed store.
    for entry in package.manifest.entries:
        data = (package.directory / entry.output_filename).read_bytes()
        assert data.startswith(b"\x89PNG")


def test_the_manifest_records_every_required_field(tmp_path, service):
    svc, _ = service

    package = build_export_package(_selections(), destination=tmp_path / "out", service=svc)
    manifest = read_manifest(package.directory)

    assert manifest.package_id == package.package_id
    assert manifest.stage == "image_color_normalization"
    assert manifest.normalization_version == "1.0.0"
    assert manifest.configuration_hash == svc.config.configuration_hash
    assert manifest.created_at
    assert len(manifest.entries) == 2

    first = manifest.entries[0]
    assert first.page_id == "page_1"
    assert first.archive_object_ref == "archive_object_lion"
    assert first.page_number == 1
    assert first.original_filename == "folio_001.tif"
    assert first.original_content_hash.startswith("page_image_")
    assert first.normalized_content_hash.startswith("normalized_page_")
    assert first.output_filename == "archive_object_lion_p0001.png"
    assert (first.width, first.height) == (_images.WIDTH, _images.HEIGHT)
    assert first.source_color_mode == "CMYK"
    assert manifest.entries[1].original_filename is None, "an absent filename is not invented"


def test_the_manifest_states_the_external_processing_boundary_and_the_hash_caveat(tmp_path, service):
    """The package carries its own honesty with it: ArchiveTrust uploaded nothing, and the
    correspondence a researcher will later confirm is not cryptographic proof."""
    svc, _ = service

    manifest = build_export_package(
        _selections(), destination=tmp_path / "out", service=svc
    ).manifest

    assert "ArchiveTrust performs no upload" in manifest.external_processing_boundary
    assert "manually upload" in manifest.external_processing_boundary
    assert "researcher-confirmed" in manifest.correspondence_caveat
    assert "not cryptographic proof" in manifest.correspondence_caveat


def test_the_manifest_is_valid_round_trippable_json(tmp_path, service):
    svc, _ = service

    package = build_export_package(_selections(), destination=tmp_path / "out", service=svc)
    raw = json.loads(package.manifest_path.read_text(encoding="utf-8"))

    assert ExportPackageManifest.model_validate(raw) == package.manifest


def test_every_packaged_page_has_real_telemetry_behind_it(tmp_path, service):
    svc, store = service

    package = build_export_package(_selections(), destination=tmp_path / "out", service=svc)

    for entry in package.manifest.entries:
        artifact = store.normalized_artifact_by_hash(entry.normalized_content_hash)
        assert artifact is not None, "a manifest entry with no recorded artifact"
        assert artifact.page_id == entry.page_id


def test_a_failing_page_prevents_the_package_from_being_written(tmp_path, service):
    """The structural guarantee: an un-normalized original can never reach a package directory."""
    svc, store = service
    selections = (
        PageImageSelection(
            page_id="page_ok",
            image_bytes=_images.rgb(),
            archive_object_ref="archive_object_lion",
            page_number=1,
        ),
        PageImageSelection(
            page_id="page_broken",
            image_bytes=_images.undecodable(),
            archive_object_ref="archive_object_lion",
            page_number=2,
        ),
    )

    with pytest.raises(NormalizationError):
        build_export_package(selections, destination=tmp_path / "out", service=svc)

    # The failure is durable evidence...
    assert len(store.normalization_failures(page_id="page_broken")) == 1
    # ...and no manifest was written, so nothing claims a valid handover.
    packages = list((tmp_path / "out").iterdir())
    assert all(not (p / MANIFEST_FILENAME).exists() for p in packages)


def test_an_empty_selection_is_refused(tmp_path, service):
    svc, _ = service

    with pytest.raises(ValueError, match="at least one selected page"):
        build_export_package((), destination=tmp_path / "out", service=svc)


def test_a_manifest_with_no_entries_is_refused():
    with pytest.raises(ValueError, match="entries must be non-empty"):
        ExportPackageManifest(
            package_id="export_package_x",
            normalization_version="1.0.0",
            configuration_hash="normalization_config_x",
            created_at=AT,
            entries=(),
        )


# -- Import association -------------------------------------------------------------------------


def _external_import():
    return ExternalImport.create(
        method_run_id="method_run_lion_1",
        source_file_path="/researcher/exports/lion_page1.xml",
        export_format=ExternalImportFormat.PAGE_XML,
        imported_by="dennis",
        imported_at=AT,
    )


def test_an_imported_result_is_associated_with_a_normalized_page_as_researcher_confirmed(
    tmp_path, service
):
    """The closing step of the round trip, and the honesty requirement at its centre."""
    svc, _ = service
    package = build_export_package(_selections(), destination=tmp_path / "out", service=svc)
    artifact = package.normalized_artifacts[0]

    association = associate_imported_result_with_normalized_page(
        _external_import(),
        normalized_artifact_id=artifact.normalized_artifact_id,
        normalized_content_hash=artifact.normalized_content_hash,
        confirmed_by="dennis",
        associated_at=AT,
        export_package_id=package.package_id,
    )

    assert association.target_kind is ImportTargetKind.NORMALIZED_PAGE_ARTIFACT
    assert association.target_id == artifact.normalized_artifact_id
    assert association.source_content_hash == artifact.normalized_content_hash
    assert association.correspondence_basis is CorrespondenceBasis.RESEARCHER_CONFIRMED
    assert association.correspondence_basis is not CorrespondenceBasis.HASH_VERIFIED
    assert association.associated_by == "dennis"
    assert association.export_package_id == package.package_id


def test_the_association_cannot_claim_hash_verification(tmp_path, service):
    """There is no parameter to override the basis -- the function's signature is the enforcement."""
    import inspect

    signature = inspect.signature(associate_imported_result_with_normalized_page)

    assert "correspondence_basis" not in signature.parameters
    assert "basis" not in signature.parameters


def test_an_unattributed_confirmation_is_refused():
    """A researcher-confirmed correspondence must name the researcher."""
    with pytest.raises(ValueError, match="must name the researcher"):
        associate_imported_result_with_normalized_page(
            _external_import(),
            normalized_artifact_id="normalized_page_artifact_x",
            normalized_content_hash="normalized_page_" + "a" * 64,
            confirmed_by="   ",
            associated_at=AT,
        )


def test_a_non_normalized_hash_is_refused():
    with pytest.raises(ValueError, match="must be a NormalizedPageArtifact content address"):
        associate_imported_result_with_normalized_page(
            _external_import(),
            normalized_artifact_id="normalized_page_artifact_x",
            normalized_content_hash="page_image_" + "a" * 64,
            confirmed_by="dennis",
            associated_at=AT,
        )


def test_a_normalized_artifact_association_requires_a_recorded_basis():
    """Guards the model directly: an unqualified link to specific bytes would read as verified."""
    from archivetrust.providers.transkribus.external_import import ImportAssociation

    with pytest.raises(ValueError, match="correspondence_basis is required"):
        ImportAssociation(
            association_id="import_association_x",
            external_import_id="external_import_x",
            target_kind=ImportTargetKind.NORMALIZED_PAGE_ARTIFACT,
            target_id="normalized_page_artifact_x",
            associated_at=AT,
        )


def test_pre_existing_association_kinds_still_work_without_a_basis():
    """Additive, not breaking: the Stage 8 association path is unchanged and still records `None`,
    which honestly means "the basis was not recorded"."""
    from archivetrust.providers.transkribus.external_import import associate_external_import

    association = associate_external_import(
        _external_import(),
        target_kind=ImportTargetKind.PAGE,
        target_id="page_1",
        associated_at=AT,
    )

    assert association.correspondence_basis is None
    assert association.source_content_hash is None


# -- ViewModel ----------------------------------------------------------------------------------


def test_the_five_state_labels_are_exactly_as_specified():
    assert [state.label for state in PreparationState] == [
        "Normalization required",
        "Ready for external upload",
        "Uploaded externally",
        "Result imported",
        "Result mapping verified",
    ]


def test_a_page_with_no_normalization_shows_normalization_required(tmp_path, service):
    _, store = service
    view = TranskribusPreparationViewModel(
        store, normalization_required_page_ids={"page_1"}
    )

    row = view.page_rows(("page_1",))[0]

    assert row.state is PreparationState.NORMALIZATION_REQUIRED
    assert row.state_label == "Normalization required"
    assert row.rgb_normalization_required is True
    assert row.normalization_status == "Not normalized"
    assert row.normalized_artifact_hash is None
    assert row.source_image_mode is None, "an unobserved source mode must not default to RGB"


def test_a_normalized_page_is_ready_for_external_upload(tmp_path, service):
    svc, store = service
    package = build_export_package(_selections(), destination=tmp_path / "out", service=svc)
    view = TranskribusPreparationViewModel(
        store,
        normalization_required_page_ids={"page_1"},
        package_ids_by_page={"page_1": package.package_id},
    )

    row = view.page_rows(("page_1",))[0]

    assert row.state is PreparationState.READY_FOR_EXTERNAL_UPLOAD
    assert row.state_label == "Ready for external upload"
    assert row.normalization_status == "Normalized"
    assert row.source_image_mode == "CMYK"
    assert row.source_image_bit_depth == 8
    assert row.normalized_artifact_hash == package.normalized_artifacts[0].normalized_content_hash
    assert row.normalization_version == "1.0.0"
    assert row.preparation_package_status == f"Prepared: {package.package_id}"
    assert row.external_upload_status == "Not uploaded (manual, external)"


def test_an_externally_uploaded_page_names_the_researcher_and_never_implies_archivetrust_uploaded(
    tmp_path, service
):
    svc, store = service
    build_export_package(_selections(), destination=tmp_path / "out", service=svc)
    view = TranskribusPreparationViewModel(store, uploaded_page_ids={"page_1": "dennis"})

    row = view.page_rows(("page_1",))[0]

    assert row.state is PreparationState.UPLOADED_EXTERNALLY
    assert row.state_label == "Uploaded externally"
    assert row.external_upload_status == "Uploaded externally by dennis"
    assert row.upload_actor == "dennis"
    assert "ArchiveTrust never uploads" in row.external_upload_disclaimer
    assert "researcher must open Transkribus themselves" in row.external_upload_disclaimer


def test_a_mapped_result_shows_verified_mapping_with_the_non_proof_caveat(tmp_path, service):
    svc, store = service
    package = build_export_package(_selections(), destination=tmp_path / "out", service=svc)
    artifact = package.normalized_artifacts[0]
    association = associate_imported_result_with_normalized_page(
        _external_import(),
        normalized_artifact_id=artifact.normalized_artifact_id,
        normalized_content_hash=artifact.normalized_content_hash,
        confirmed_by="dennis",
        associated_at=AT,
        export_package_id=package.package_id,
    )
    view = TranskribusPreparationViewModel(
        store, uploaded_page_ids={"page_1": "dennis"}, associations=(association,)
    )

    row = view.page_rows(("page_1",))[0]

    assert row.state is PreparationState.RESULT_MAPPING_VERIFIED
    assert row.state_label == "Result mapping verified"
    assert row.imported_result_status == "Result imported"
    assert row.mapping_status == "Mapped (researcher-confirmed, not cryptographic proof)"
    assert row.correspondence_basis == "researcher_confirmed"
    assert row.external_import_id == association.external_import_id


def test_a_failed_normalization_is_surfaced_with_its_category(tmp_path, service):
    svc, store = service
    with pytest.raises(NormalizationError):
        svc.normalize_and_record(_images.undecodable(), page_id="page_broken")
    view = TranskribusPreparationViewModel(
        store, normalization_required_page_ids={"page_broken"}
    )

    row = view.page_rows(("page_broken",))[0]

    assert row.state is PreparationState.NORMALIZATION_REQUIRED
    assert row.normalization_status == "Failed: undecodable_image"
    assert row.normalization_failure_category == "undecodable_image"
    assert row.normalization_failure_reason
    assert row.normalized_artifact_hash is None


def test_an_association_for_a_different_artifact_is_not_shown_against_this_one(tmp_path, service):
    """A page re-normalized under a new version must not inherit its old artifact's mapping."""
    svc, store = service
    package = build_export_package(_selections(), destination=tmp_path / "out", service=svc)
    other = package.normalized_artifacts[1]
    association = associate_imported_result_with_normalized_page(
        _external_import(),
        normalized_artifact_id=other.normalized_artifact_id,
        normalized_content_hash=other.normalized_content_hash,
        confirmed_by="dennis",
        associated_at=AT,
    )
    view = TranskribusPreparationViewModel(store, associations=(association,))

    row = view.page_rows(("page_1",))[0]

    assert row.mapping_status == "Not mapped"
    assert row.imported_result_status == "No result imported"
