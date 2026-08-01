"""Builds a complete, real source inventory of the Swedish training data
(`F:\\huggingface_dataset`, Riksarkivet's "Training data for Swedish Lion Libre" HuggingFace
collection -- see docs/methods/loghi-swedish-finetuning.md for this dataset's own provenance record).

**What the source data actually contains, established by reading it, not assumed**: each row is
`{image: {bytes, path}, transcription: str}`. `image.path` is `None` on every row inspected -- there
is no archive/volume/writer/document identifier below the collection (top-level folder) granularity.
Every field this module cannot populate from a real source value is `None` with
`unavailable_field_reason` stated on the record, never fabricated.

Every row gets a stable `line_id` = `f"{collection}:{parquet_filename}:{row_index}"` -- real,
reproducible from the row's own location, not invented.

**Streaming, not whole-file-in-memory.** An earlier version of this module read each Parquet file's
`image`+`transcription` columns via `pq.read_table(...).to_pylist()` for the *whole file at once* --
which decodes every row's raw image bytes into Python objects simultaneously. For the largest source
file (tens of thousands of line crops) that pushed the process to 13.7GB resident on a 64GB machine
with only ~11GB free at the time, and it was killed mid-run rather than risk OOM alongside Docker
Desktop and, later, GPU training. This version streams `ParquetFile.iter_batches()` in bounded batches
and writes the inventory incrementally via `pq.ParquetWriter`, so peak memory is one batch, not one
file and not the whole 565K-row inventory.
"""

from __future__ import annotations

import hashlib
import io
import unicodedata
from pathlib import Path
from typing import Iterator

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict

TRANSCRIPTION_SOURCE = "huggingface_parquet_transcription_column"
DEFAULT_BATCH_SIZE = 512

_CONTROL_CHAR_ALLOWED = {"\n"}
"""Only newline is treated as a legitimate structural character in a transcription (a "line" that
wraps). Tab was allowed here in an earlier version of this module and was wrong: 3 of the 565,146 real
rows carry a literal U+0009 in their transcription, which is not in the generic Loghi charlist (455
printable characters -- no control characters at all) and has no plausible reading as something a
17th/18th-century scribe wrote. Character-compatibility checking (`character_inventory.py`) caught this
as the *only* unsupported character across the entire 10,999-line pilot -- a formatting artifact worth
excluding as `control_characters_present`, not a real Swedish letterform worth extending the model's
vocabulary for."""

_UNAVAILABLE_FIELD_REASON = (
    "archive_or_fonds_id/volume_or_document_id/writer_or_hand_id: the HuggingFace parquet schema "
    "carries only {image: {bytes, path}, transcription}; image.path is None on every row inspected, "
    "so no identifier below collection granularity is available in the source data."
)


def _has_control_characters(text: str) -> bool:
    return any(
        unicodedata.category(ch) in ("Cc", "Cf") and ch not in _CONTROL_CHAR_ALLOWED for ch in text
    )


def _unique_characters(text: str) -> str:
    return "".join(sorted(set(text)))


class SourceLineRecord(BaseModel):
    """One candidate training line, exactly as WP4 specifies -- every field either a real value read
    from the source or an honestly `None` absence with a stated reason."""

    model_config = ConfigDict(frozen=True)

    collection: str
    archive_or_fonds_id: str | None = None
    volume_or_document_id: str | None = None
    writer_or_hand_id: str | None = None
    line_id: str
    source_parquet_file: str
    row_index: int
    image_path: str | None
    transcription_source: str
    image_content_hash: str | None
    image_width: int | None
    image_height: int | None
    image_format: str | None
    transcription: str | None
    transcription_length: int | None
    unique_characters: str
    control_characters_present: bool
    nfc_normalization_differs: bool
    unsupported_characters: tuple[str, ...]
    """Characters not present in the supplied charlist -- empty when no charlist was supplied, or when
    every character is supported."""
    duplicate_status: str
    """`"unique"` or `"duplicate"` -- first occurrence of an (image_hash) or (image_hash,
    transcription) pair is `"unique"`; every later occurrence is `"duplicate"`, never excluded from
    the record (preserved as evidence), only excluded from `valid`."""
    duplicate_of_line_id: str | None
    valid: bool
    exclusion_reason: str | None
    unavailable_field_reason: str | None = _UNAVAILABLE_FIELD_REASON

    def as_row_dict(self) -> dict:
        return {**self.model_dump(), "unsupported_characters": list(self.unsupported_characters)}


INVENTORY_ARROW_SCHEMA = pa.schema(
    [
        ("collection", pa.string()),
        ("archive_or_fonds_id", pa.string()),
        ("volume_or_document_id", pa.string()),
        ("writer_or_hand_id", pa.string()),
        ("line_id", pa.string()),
        ("source_parquet_file", pa.string()),
        ("row_index", pa.int64()),
        ("image_path", pa.string()),
        ("transcription_source", pa.string()),
        ("image_content_hash", pa.string()),
        ("image_width", pa.int64()),
        ("image_height", pa.int64()),
        ("image_format", pa.string()),
        ("transcription", pa.string()),
        ("transcription_length", pa.int64()),
        ("unique_characters", pa.string()),
        ("control_characters_present", pa.bool_()),
        ("nfc_normalization_differs", pa.bool_()),
        ("unsupported_characters", pa.list_(pa.string())),
        ("duplicate_status", pa.string()),
        ("duplicate_of_line_id", pa.string()),
        ("valid", pa.bool_()),
        ("exclusion_reason", pa.string()),
        ("unavailable_field_reason", pa.string()),
    ]
)


def _load_charlist(charlist_path: str | Path | None) -> set[str] | None:
    if charlist_path is None:
        return None
    text = Path(charlist_path).read_text(encoding="utf-8")
    return set(text)


def iter_source_rows(
    dataset_root: str | Path, *, batch_size: int = DEFAULT_BATCH_SIZE
) -> Iterator[tuple[str, str, int, dict]]:
    """Yields `(collection, parquet_filename, row_index, row_dict)` for every row under
    `dataset_root/*_lines/*.parquet`, in a deterministic (sorted) order -- the same order every
    process sees, which is what makes `line_id` reproducible run to run.

    Streams via `ParquetFile.iter_batches(batch_size=...)` rather than reading a whole file's worth of
    decoded image bytes into memory at once -- see module docstring.
    """
    root = Path(dataset_root)
    for collection_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for parquet_path in sorted(collection_dir.glob("*.parquet")):
            parquet_file = pq.ParquetFile(parquet_path)
            row_index = 0
            for batch in parquet_file.iter_batches(
                batch_size=batch_size, columns=["image", "transcription"]
            ):
                for row in batch.to_pylist():
                    yield collection_dir.name, parquet_path.name, row_index, row
                    row_index += 1


def _build_record(
    *, collection: str, parquet_filename: str, row_index: int, row: dict, charlist: set[str] | None,
    seen_image_hashes: dict[str, str], seen_pairs: dict[tuple[str, str], str],
    seen_image_transcriptions: dict[str, str | None],
) -> SourceLineRecord:
    line_id = f"{collection}:{parquet_filename}:{row_index}"
    image_field = row.get("image") or {}
    image_bytes = image_field.get("bytes")
    image_path = image_field.get("path")
    transcription = row.get("transcription")

    exclusion_reasons: list[str] = []

    image_hash: str | None = None
    width: int | None = None
    height: int | None = None
    image_format: str | None = None

    if not image_bytes:
        exclusion_reasons.append("missing_image")
    else:
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                width, height = img.size
                image_format = img.format
        except (UnidentifiedImageError, OSError):
            exclusion_reasons.append("corrupt_image")

    if transcription is None:
        exclusion_reasons.append("missing_transcription")
    elif not transcription.strip():
        exclusion_reasons.append("empty_transcription")

    unique_chars = _unique_characters(transcription) if transcription else ""
    control_chars_present = _has_control_characters(transcription) if transcription else False
    if control_chars_present:
        exclusion_reasons.append("control_characters_present")

    nfc_differs = False
    if transcription:
        nfc_differs = transcription != unicodedata.normalize("NFC", transcription)
        if nfc_differs:
            exclusion_reasons.append("nfc_normalization_differs")

    unsupported_chars: tuple[str, ...] = ()
    if charlist is not None and transcription:
        unsupported = sorted(set(transcription) - charlist - _CONTROL_CHAR_ALLOWED)
        unsupported_chars = tuple(unsupported)
        if unsupported_chars:
            exclusion_reasons.append("unsupported_characters")

    if width is not None and height is not None:
        if width <= 0 or height <= 0:
            exclusion_reasons.append("extreme_line_dimensions")
        elif width / max(height, 1) > 200 or height / max(width, 1) > 50:
            exclusion_reasons.append("extreme_line_dimensions")

    # Three genuinely distinct cases, checked independently -- an earlier version of this function
    # guarded the pair-check with `duplicate_of_line_id is None`, which made
    # "duplicate_image_transcription_pair" unreachable dead code: since a pair is keyed on
    # `(image_hash, transcription)`, a duplicate image whose transcription also matches the first
    # occurrence is *always* also a duplicate pair, so skipping the pair check whenever the image
    # check already fired meant the pair reason could never be recorded on its own. Checked separately
    # instead, so a single row can legitimately carry both reasons when both are true, and so the
    # "same image, different transcription" case -- a real, distinct data-quality signal the brief
    # calls "suspicious transcription mismatch," not a duplicate at all -- is recorded honestly rather
    # than folded into "duplicate_image" and never mentioned again.
    duplicate_status = "unique"
    duplicate_of_line_id: str | None = None
    if image_hash is not None:
        if image_hash in seen_image_hashes:
            duplicate_status = "duplicate"
            duplicate_of_line_id = seen_image_hashes[image_hash]
            exclusion_reasons.append("duplicate_image")
            if transcription != seen_image_transcriptions.get(image_hash):
                exclusion_reasons.append("suspicious_transcription_mismatch")
        else:
            seen_image_hashes[image_hash] = line_id
            seen_image_transcriptions[image_hash] = transcription

        pair_key = (image_hash, transcription or "")
        if pair_key in seen_pairs:
            duplicate_status = "duplicate"
            if duplicate_of_line_id is None:
                duplicate_of_line_id = seen_pairs[pair_key]
            exclusion_reasons.append("duplicate_image_transcription_pair")
        else:
            seen_pairs[pair_key] = line_id

    valid = not exclusion_reasons
    exclusion_reason = ";".join(exclusion_reasons) if exclusion_reasons else None

    return SourceLineRecord(
        collection=collection,
        line_id=line_id,
        source_parquet_file=parquet_filename,
        row_index=row_index,
        image_path=image_path,
        transcription_source=TRANSCRIPTION_SOURCE,
        image_content_hash=image_hash,
        image_width=width,
        image_height=height,
        image_format=image_format,
        transcription=transcription,
        transcription_length=len(transcription) if transcription is not None else None,
        unique_characters=unique_chars,
        control_characters_present=control_chars_present,
        nfc_normalization_differs=nfc_differs,
        unsupported_characters=unsupported_chars,
        duplicate_status=duplicate_status,
        duplicate_of_line_id=duplicate_of_line_id,
        valid=valid,
        exclusion_reason=exclusion_reason,
    )


def build_source_inventory(
    *,
    dataset_root: str | Path,
    output_path: str | Path,
    charlist_path: str | Path | None = None,
    max_rows: int | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    progress_every: int | None = None,
) -> dict:
    """Streams the complete inventory to `output_path` (Parquet, written incrementally) and returns a
    real, computed summary -- never holds more than `batch_size` rows' worth of records plus the
    (small) hash-dedup dicts in memory at once. `max_rows` exists only for tests/smoke runs over a
    bounded prefix.
    """
    charlist = _load_charlist(charlist_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seen_image_hashes: dict[str, str] = {}
    seen_pairs: dict[tuple[str, str], str] = {}
    seen_image_transcriptions: dict[str, str | None] = {}

    by_collection: dict[str, dict[str, int]] = {}
    exclusion_counts: dict[str, int] = {}
    total = 0
    valid_total = 0

    writer = pq.ParquetWriter(output_path, INVENTORY_ARROW_SCHEMA)
    try:
        batch_rows: list[dict] = []

        def flush() -> None:
            if not batch_rows:
                return
            table = pa.Table.from_pylist(batch_rows, schema=INVENTORY_ARROW_SCHEMA)
            writer.write_table(table)
            batch_rows.clear()

        for count, (collection, parquet_filename, row_index, row) in enumerate(
            iter_source_rows(dataset_root, batch_size=batch_size)
        ):
            if max_rows is not None and count >= max_rows:
                break

            record = _build_record(
                collection=collection,
                parquet_filename=parquet_filename,
                row_index=row_index,
                row=row,
                charlist=charlist,
                seen_image_hashes=seen_image_hashes,
                seen_pairs=seen_pairs,
                seen_image_transcriptions=seen_image_transcriptions,
            )
            batch_rows.append(record.as_row_dict())

            total += 1
            bucket = by_collection.setdefault(collection, {"total": 0, "valid": 0})
            bucket["total"] += 1
            if record.valid:
                valid_total += 1
                bucket["valid"] += 1
            if record.exclusion_reason:
                for reason in record.exclusion_reason.split(";"):
                    exclusion_counts[reason] = exclusion_counts.get(reason, 0) + 1

            if len(batch_rows) >= batch_size:
                flush()
            if progress_every and total % progress_every == 0:
                print(f"  ...{total} rows processed ({valid_total} valid)", flush=True)

        flush()
    finally:
        writer.close()

    return {
        "total_rows": total,
        "valid_rows": valid_total,
        "excluded_rows": total - valid_total,
        "by_collection": by_collection,
        "exclusion_counts": exclusion_counts,
    }


def read_inventory(
    inventory_path: str | Path, *, columns: list[str] | None = None, valid_only: bool = False
) -> pa.Table:
    """Reads the written inventory back -- the interface `pilot_split.py`/tests use, never a second
    in-memory list built by re-running `build_source_inventory`."""
    read_columns = columns
    drop_valid_column = False
    if valid_only and columns is not None and "valid" not in columns:
        read_columns = [*columns, "valid"]
        drop_valid_column = True

    table = pq.read_table(inventory_path, columns=read_columns)
    if valid_only:
        import pyarrow.compute as pc

        table = table.filter(pc.field("valid"))
        if drop_valid_column:
            table = table.drop_columns(["valid"])
    return table
