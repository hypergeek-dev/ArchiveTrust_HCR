"""`ResearchReport` export: JSON and CSV (Stage 11, brief's "Support JSON and CSV first").

**HTML and PDF are deliberately not implemented.** The brief permits them "only if the architecture
already supports it cleanly". It does not: there is no template engine, no HTML renderer, and no PDF
writer anywhere in this project's dependency set (`pyproject.toml`'s runtime dependency is
`pydantic`; `pdf_render.py` in the desktop client *reads* PDFs via Qt for display, it does not write
them). Adding either would mean adding a rendering dependency to a research tool as a side effect of
an export feature. `UNSUPPORTED_FORMATS` names them so a caller asking for one gets a clear refusal
rather than a silently missing option.

**Round-trip is real for JSON, deliberately one-way for CSV.** `report_to_json`/`report_from_json`
reconstruct an identical `ResearchReport` (tested). CSV cannot: `metadata` is an arbitrary nested
dict and the whole point of the CSV shape is a flat, spreadsheet-openable table of one row per
experiment run. `report_to_csv` therefore documents itself as a lossy view, not a serialization
format -- calling it a round-trip format would be the misleading claim.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from archivetrust.research.reports.models import ResearchReport

REPORT_EXPORT_SCHEMA = "archivetrust.research_report.v1"
"""Versioned like the existing `archivetrust.canonical_document.v2` export schema
(`application/export/json_export.py`) -- an exported file states which schema produced it."""

UNSUPPORTED_FORMATS = ("html", "pdf")
"""See module docstring. Named rather than silently absent."""


class UnsupportedExportFormatError(ValueError):
    """Raised for an export format this project deliberately does not implement."""


def report_to_json(report: ResearchReport, *, indent: int | None = 2) -> str:
    """Serializes one report. `indent=None` produces the compact form; the default is indented
    because a research report is read by humans at least as often as by machines."""
    payload = {
        "export_schema": REPORT_EXPORT_SCHEMA,
        "report": report.model_dump(mode="json"),
    }
    return json.dumps(payload, indent=indent, sort_keys=True, ensure_ascii=False)


def report_from_json(text: str) -> ResearchReport:
    """Reconstructs a `ResearchReport` from `report_to_json` output. Rejects a payload written by a
    different schema version rather than best-effort-parsing it into something subtly wrong."""
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("A research report export must be a JSON object")
    schema = payload.get("export_schema")
    if schema != REPORT_EXPORT_SCHEMA:
        raise ValueError(
            f"Unsupported research report export schema {schema!r} "
            f"(this build reads {REPORT_EXPORT_SCHEMA!r})"
        )
    body = payload.get("report")
    if not isinstance(body, dict):
        raise ValueError("A research report export must carry a 'report' object")
    return ResearchReport.model_validate(body)


CSV_COLUMNS = (
    "report_id",
    "title",
    "generated_at",
    "experiment_run_id",
    "reproducibility_manifest_refs",
    "legacy_experiments_included",
    "methodology",
    "dataset_description",
    "results_summary",
    "limitations",
)
"""One row per `experiment_run_id`, with the report-level fields repeated on each row -- the shape a
spreadsheet filter and a pivot table both want. `metadata` is excluded: an arbitrary nested dict has
no honest flat-column representation, and flattening it would invent column names."""


def report_to_csv(report: ResearchReport) -> str:
    """A flat, one-row-per-experiment-run view. Lossy by design (see module docstring and
    `CSV_COLUMNS`): use `report_to_json` when the report must survive a round trip.

    Line terminator is `\\n` explicitly, so output is byte-identical across platforms rather than
    picking up `\\r\\n` from `csv`'s default on Windows -- exported research artifacts that differ
    only by platform would defeat the hash-comparison discipline used elsewhere in this codebase.
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(CSV_COLUMNS)
    for experiment_run_id in report.experiment_run_ids:
        writer.writerow(
            [
                report.report_id,
                report.title,
                report.generated_at,
                experiment_run_id,
                " ".join(report.reproducibility_manifest_refs),
                "true" if report.legacy_experiments_included else "false",
                report.methodology or "",
                report.dataset_description or "",
                report.results_summary or "",
                report.limitations or "",
            ]
        )
    return buffer.getvalue()


def write_report(report: ResearchReport, path: Path, *, export_format: str) -> Path:
    """Writes one report to `path` in `export_format` (`"json"` or `"csv"`). Raises
    `UnsupportedExportFormatError` for `"html"`/`"pdf"` with the reason, and for anything else.

    UTF-8 without a BOM, matching every other writer in this codebase.
    """
    normalized = export_format.strip().lower()
    if normalized in UNSUPPORTED_FORMATS:
        raise UnsupportedExportFormatError(
            f"{normalized!r} export is not implemented: this project has no template engine or "
            "PDF writer, and adding a rendering dependency for an export format was judged out of "
            "scope. Export JSON or CSV instead."
        )
    if normalized == "json":
        text = report_to_json(report)
    elif normalized == "csv":
        text = report_to_csv(report)
    else:
        raise UnsupportedExportFormatError(
            f"Unknown research report export format {export_format!r}; supported: json, csv"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
