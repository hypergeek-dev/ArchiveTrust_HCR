"""Tabular deliveries: CSV / TSV / JSONL rows pointing at line images, headerless `path<TAB>text`
lists (the Loghi list format), and Parquet with embedded images (the Hugging Face `Image` feature,
as in Riksarkivet's published line datasets).

Columns are recognized by common names; anything ambiguous is a blocker naming the columns found,
never a guess.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path, PurePosixPath

from archivetrust.htr.benchmark.findings import Finding
from archivetrust.htr.benchmark.sources.base import CandidateLine, Extraction, is_image, read_text_file

TABLE_SUFFIXES = (".csv", ".tsv", ".jsonl", ".parquet")
IMAGE_COLUMNS = ("image", "image_path", "line_image", "img", "file", "filename", "file_name", "path")
TEXT_COLUMNS = ("text", "transcription", "gt", "ground_truth", "groundtruth", "label", "line_text", "sentence")
OPTIONAL_COLUMNS = {
    "document": ("document_id", "document", "doc_id", "doc", "volume"),
    "page": ("page_id", "page", "page_name"),
    "line": ("line_id", "id", "line"),
    "collection": ("collection", "dataset", "subset", "source"),
    "writer": ("writer_id", "writer", "hand", "scribe"),
}


def _pick(columns: list[str], wanted: tuple[str, ...]) -> list[str]:
    lowered = {c.lower(): c for c in columns}
    return [lowered[w] for w in wanted if w in lowered]


def _resolve(root: Path, table_rel: str, value: str, names: set[str]) -> str | None:
    value = value.replace("\\", "/")
    table_dir = PurePosixPath(table_rel).parent
    for candidate in (table_dir / value, PurePosixPath(value)):
        if str(candidate) in names:
            return str(candidate)
    return None


class TabularAdapter:
    adapter_id = "tabular"
    version = "1"

    def detect(self, root: Path, files: list[str]) -> float:
        tables = [f for f in files if f.lower().endswith(TABLE_SUFFIXES)]
        if not tables:
            return 0.0
        rows, _ = self._read_rows(root, tables[0])
        if rows and _pick(list(rows[0]), TEXT_COLUMNS):
            return 0.9
        return 0.3

    def _read_rows(self, root: Path, table_rel: str) -> tuple[list[dict], list[Finding]]:
        suffix = PurePosixPath(table_rel).suffix.lower()
        if suffix == ".parquet":
            try:
                import pyarrow.parquet as pq  # noqa: PLC0415 -- optional, only for parquet deliveries
            except ImportError:
                return [], [Finding(severity="blocker", code="tabular.pyarrow_missing", path=table_rel,
                                    message="reading Parquet needs pyarrow (pip install pyarrow)")]
            return pq.read_table(root / table_rel).to_pylist(), []
        text, findings = read_text_file(root, table_rel)
        if text is None:
            return [], findings
        if suffix == ".jsonl":
            rows = []
            for number, raw in enumerate(text.splitlines(), start=1):
                if raw.strip():
                    try:
                        rows.append(json.loads(raw))
                    except json.JSONDecodeError as exc:
                        findings.append(Finding(severity="blocker", code="tabular.bad_json", path=f"{table_rel}#row={number}",
                                                message=f"row {number}: {exc}"))
            return rows, findings
        delimiter = "\t" if suffix == ".tsv" else ","
        reader = list(csv.reader(io.StringIO(text, newline=""), delimiter=delimiter))
        if not reader:
            return [], findings
        header = reader[0]
        if _pick(header, TEXT_COLUMNS) or _pick(header, IMAGE_COLUMNS):
            return [dict(zip(header, row, strict=False)) | {"__row__": n} for n, row in enumerate(reader[1:], start=2)], findings
        findings.append(Finding(severity="warning", code="tabular.headerless", path=table_rel,
                                message="no recognizable header; reading as headerless <image><sep><text> rows"))
        return [{"image": row[0], "text": delimiter.join(row[1:]), "__row__": n} for n, row in enumerate(reader, start=1) if row], findings

    def extract(self, root: Path, files: list[str]) -> Extraction:
        out = Extraction(adapter_id=self.adapter_id, adapter_version=self.version)
        names = set(files)
        for table_rel in [f for f in files if f.lower().endswith(TABLE_SUFFIXES)]:
            out.consumed_files.add(table_rel)
            rows, findings = self._read_rows(root, table_rel)
            out.findings.extend(findings)
            if not rows:
                continue
            columns = sorted({k for row in rows[:50] for k in row if k != "__row__"})
            image_cols, text_cols = _pick(columns, IMAGE_COLUMNS), _pick(columns, TEXT_COLUMNS)
            if len(text_cols) != 1 or not image_cols:
                out.findings.append(Finding(severity="blocker", code="tabular.columns_unresolved", path=table_rel,
                                            message=f"need exactly one text column and an image column; found columns {columns}",
                                            detail={"text_candidates": text_cols, "image_candidates": image_cols}))
                continue
            image_col, text_col = image_cols[0], text_cols[0]
            optional = {k: (_pick(columns, v) or [None])[0] for k, v in OPTIONAL_COLUMNS.items()}
            self._extract_rows(root, table_rel, rows, image_col, text_col, optional, names, out)
        return out

    def _extract_rows(self, root, table_rel, rows, image_col, text_col, optional, names, out: Extraction) -> None:
        counters: dict[tuple[str, str], int] = {}
        for index, row in enumerate(rows):
            row_ref = f"row={row.get('__row__', index)}"
            image_value = row.get(image_col)
            image_bytes, image_rel, original = None, None, None
            if isinstance(image_value, dict):  # Hugging Face Image feature: {"bytes": ..., "path": ...}
                image_bytes, original = image_value.get("bytes"), image_value.get("path")
                image_rel = f"{table_rel}#{row_ref}"
            elif isinstance(image_value, (bytes, bytearray)):
                image_bytes, image_rel = bytes(image_value), f"{table_rel}#{row_ref}"
            elif isinstance(image_value, str) and image_value:
                original = image_value
                image_rel = _resolve(root, table_rel, image_value, names)
                if image_rel is None:
                    out.findings.append(Finding(severity="blocker", code="image.missing", path=f"{table_rel}#{row_ref}",
                                                message=f"{row_ref}: image {image_value!r} not found"))
                    continue
                out.consumed_files.add(image_rel)
            if image_rel is None or ("#" in image_rel and not image_bytes) or ("#" not in image_rel and not is_image(image_rel)):
                out.findings.append(Finding(severity="blocker", code="image.missing", path=f"{table_rel}#{row_ref}", message=f"{row_ref}: no image"))
                continue

            text = row.get(text_col)
            if text is not None and not isinstance(text, str):
                out.findings.append(Finding(severity="needs_review", code="gt.not_text", path=image_rel,
                                            message=f"{row_ref}: text cell is {type(text).__name__}, not a string"))
                text = None
            document = str(row.get(optional["document"]) or "") if optional["document"] else ""
            if not document:
                parent = str(PurePosixPath(original or "").parent) if original and "#" not in image_rel else "."
                document = parent if parent not in (".", "") else PurePosixPath(table_rel).stem
            page = str(row.get(optional["page"]) or "lines") if optional["page"] else "lines"
            line = str(row.get(optional["line"]) or "") if optional["line"] else ""
            if not line:
                line = PurePosixPath(original).stem if original else f"row{index:06d}"
            order = counters.get((document, page), 0)
            counters[(document, page)] = order + 1
            if text is None:
                out.findings.append(Finding(severity="needs_review", code="gt.missing", path=image_rel, message=f"{row_ref}: no transcription"))
            out.lines.append(CandidateLine(
                document_raw=document, page_raw=page, line_raw=line, line_order=order, image_kind="line",
                image_path=image_rel, gt_path=table_rel, gt_source=text, source_line_ref=row_ref,
                collection=str(row.get(optional["collection"])) if optional["collection"] and row.get(optional["collection"]) else None,
                writer_id=str(row.get(optional["writer"])) if optional["writer"] and row.get(optional["writer"]) else None,
                metadata={"original_image_name": original} if original else {}, image_bytes=image_bytes,
            ))
