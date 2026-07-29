"""`ResearchReport` JSON/CSV export: real round-trip, honest refusal of HTML/PDF."""

from __future__ import annotations

import csv
import io
import json

import pytest

from archivetrust.research.reports.export import (
    CSV_COLUMNS,
    REPORT_EXPORT_SCHEMA,
    UNSUPPORTED_FORMATS,
    UnsupportedExportFormatError,
    report_from_json,
    report_to_csv,
    report_to_json,
    write_report,
)
from archivetrust.research.reports.models import ResearchReport


def _report() -> ResearchReport:
    return ResearchReport.create(
        title="Swedish Historical HTR Baseline Comparison",
        generated_at="2026-07-29T12:00:00Z",
        experiment_run_ids=("experiment_run_1", "experiment_run_2"),
        methodology="Controlled comparison over shared input crops.",
        dataset_description="Trolldomskommissionen, 17th-century court records.",
        results_summary="SATRN reached the lowest normalized CER; Florence-2 failed on one line.",
        limitations="Sample of two lines; not a statistically powered comparison.",
        reproducibility_manifest_refs=("manifest_1",),
        legacy_experiments_included=False,
        metadata={"hardware": {"gpu": "RTX 3070"}, "seed": 1},
    )


def test_json_round_trip_reconstructs_an_identical_report() -> None:
    original = _report()
    restored = report_from_json(report_to_json(original))

    assert restored == original
    assert restored.metadata == {"hardware": {"gpu": "RTX 3070"}, "seed": 1}


def test_json_export_states_its_schema_version() -> None:
    payload = json.loads(report_to_json(_report()))
    assert payload["export_schema"] == REPORT_EXPORT_SCHEMA
    assert payload["report"]["title"] == "Swedish Historical HTR Baseline Comparison"


def test_json_export_is_deterministic_for_the_same_report() -> None:
    report = _report()
    assert report_to_json(report) == report_to_json(report)


def test_a_payload_from_another_schema_version_is_refused_not_guessed_at() -> None:
    payload = json.loads(report_to_json(_report()))
    payload["export_schema"] = "archivetrust.research_report.v99"

    with pytest.raises(ValueError, match="Unsupported research report export schema"):
        report_from_json(json.dumps(payload))


def test_malformed_json_payloads_are_rejected_with_a_reason() -> None:
    with pytest.raises(ValueError, match="must be a JSON object"):
        report_from_json("[1, 2, 3]")
    with pytest.raises(ValueError, match="must carry a 'report' object"):
        report_from_json(json.dumps({"export_schema": REPORT_EXPORT_SCHEMA}))


def test_csv_has_one_row_per_experiment_run_with_the_declared_columns() -> None:
    rows = list(csv.reader(io.StringIO(report_to_csv(_report()))))

    assert tuple(rows[0]) == CSV_COLUMNS
    assert len(rows) == 3  # header + one row per experiment run
    assert rows[1][CSV_COLUMNS.index("experiment_run_id")] == "experiment_run_1"
    assert rows[2][CSV_COLUMNS.index("experiment_run_id")] == "experiment_run_2"
    assert rows[1][CSV_COLUMNS.index("title")] == "Swedish Historical HTR Baseline Comparison"
    assert rows[1][CSV_COLUMNS.index("legacy_experiments_included")] == "false"


def test_csv_uses_a_platform_independent_line_terminator() -> None:
    """Exported artifacts that differ only by platform would defeat hash comparison."""
    text = report_to_csv(_report())
    assert "\r\n" not in text
    assert text.endswith("\n")


def test_csv_deliberately_omits_metadata_and_is_not_claimed_to_round_trip() -> None:
    assert "metadata" not in CSV_COLUMNS
    assert "RTX 3070" not in report_to_csv(_report())


def test_a_report_with_no_optional_prose_still_exports_cleanly() -> None:
    minimal = ResearchReport.create(
        title="Minimal",
        generated_at="2026-07-29T12:00:00Z",
        experiment_run_ids=("experiment_run_1",),
    )
    rows = list(csv.reader(io.StringIO(report_to_csv(minimal))))
    assert rows[1][CSV_COLUMNS.index("methodology")] == ""
    assert report_from_json(report_to_json(minimal)) == minimal


def test_legacy_inclusion_is_surfaced_in_both_formats() -> None:
    """`docs/htr-domain-design.md` §8: legacy inclusion is never silently blended in."""
    legacy = ResearchReport.create(
        title="With legacy",
        generated_at="2026-07-29T12:00:00Z",
        experiment_run_ids=("experiment_run_1",),
        legacy_experiments_included=True,
    )
    rows = list(csv.reader(io.StringIO(report_to_csv(legacy))))
    assert rows[1][CSV_COLUMNS.index("legacy_experiments_included")] == "true"
    assert report_from_json(report_to_json(legacy)).legacy_experiments_included is True


def test_html_and_pdf_are_refused_with_the_reason_not_silently_missing() -> None:
    assert UNSUPPORTED_FORMATS == ("html", "pdf")
    for unsupported in UNSUPPORTED_FORMATS:
        with pytest.raises(UnsupportedExportFormatError, match="not implemented"):
            write_report(_report(), __import__("pathlib").Path("out"), export_format=unsupported)


def test_an_unknown_format_names_what_is_supported() -> None:
    from pathlib import Path

    with pytest.raises(UnsupportedExportFormatError, match="supported: json, csv"):
        write_report(_report(), Path("out.xyz"), export_format="xyz")


def test_write_report_writes_utf8_and_creates_parent_directories(tmp_path) -> None:
    report = _report()
    json_path = write_report(
        report, tmp_path / "nested" / "report.json", export_format="json"
    )
    csv_path = write_report(report, tmp_path / "nested" / "report.csv", export_format="csv")

    assert json_path.exists() and csv_path.exists()
    assert report_from_json(json_path.read_text(encoding="utf-8")) == report
    assert not json_path.read_bytes().startswith(b"\xef\xbb\xbf")  # no BOM
    assert csv_path.read_text(encoding="utf-8").startswith(",".join(CSV_COLUMNS))


def test_format_argument_is_case_and_whitespace_tolerant(tmp_path) -> None:
    path = write_report(_report(), tmp_path / "r.json", export_format="  JSON ")
    assert path.exists()
