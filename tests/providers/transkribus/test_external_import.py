from __future__ import annotations

from archivetrust.providers.transkribus import ExternalImport, ExternalImportFormat


def test_external_import_round_trips():
    record = ExternalImport.create(
        method_run_id="method_run_1",
        source_file_path="/tmp/export.xml",
        export_format=ExternalImportFormat.PAGE_XML,
        imported_by="curator-1",
        imported_at="2026-01-01T00:00:00Z",
    )
    restored = ExternalImport.model_validate(record.model_dump())
    assert restored == record
    assert record.export_format is ExternalImportFormat.PAGE_XML
