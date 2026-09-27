"""Contamination check: does the benchmark overlap the material the models were trained on?

Both models were trained on Riksarkivet's public Hugging Face line collections (the Loghi model on a
subset whose exact split manifests no longer exist; Lion on the collections themselves), so the whole
HF corpus is indexed as the conservative "possibly seen" set.

Signals, strongest first:
- identical image bytes (only catches unmodified re-distribution);
- near-identical image (dHash <= 4 bits) AND identical normalized text -- strong;
- identical transcription, exact or normalized, for lines of >= `MIN_TEXT_CHARS` characters (short
  lines like "Anno 1723" repeat naturally and are not evidence);
- near-identical image alone -- weak (text lines look alike at 9x8 pixels), reported, not concluded.

No match is NOT proof of no overlap: re-cropped, re-scanned or re-transcribed material evades every
signal here. The report says so.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from archivetrust.evaluation.metrics import normalize_text
from archivetrust.htr.benchmark.build import verify_frozen
from archivetrust.htr.benchmark.contract import sha256_bytes, sha256_file, sha256_text
from archivetrust.htr.benchmark.imaging import dhash, open_page_rgb
from archivetrust.htr.benchmark.provenance import utc_now

OVERLAP_VERSION = "1"
MIN_TEXT_CHARS = 20
DHASH_MAX_DISTANCE = 4
_BANDS = DHASH_MAX_DISTANCE + 1
_BAND_WIDTH = 64 // _BANDS


def _bands(hex_hash: str) -> list[int]:
    value = int(hex_hash, 16)
    out = []
    for band in range(_BANDS):
        shift = band * _BAND_WIDTH
        span = _BAND_WIDTH if band < _BANDS - 1 else 64 - shift
        out.append((value >> shift) & ((1 << span) - 1))
    return out


def text_key(text: str) -> str:
    return normalize_text(text).casefold()


def _image_dhash(data: bytes) -> str | None:
    try:
        return dhash(open_page_rgb(data))
    except Exception:  # noqa: BLE001 -- an undecodable training image is skipped, and counted
        return None


def _connect(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
    band_cols = ", ".join(f"b{i} INTEGER" for i in range(_BANDS))
    db.execute(f"CREATE TABLE IF NOT EXISTS lines (source TEXT, collection TEXT, image_sha256 TEXT, dhash TEXT, "
               f"text_sha256 TEXT, text_norm_sha256 TEXT, text_len INTEGER, {band_cols})")
    return db


def build_training_index(parquet_files: Iterable[Path], index_path: Path, *, batch_size: int = 512) -> dict:
    """Streams HF parquet shards (image struct + transcription column) into an SQLite index.
    Collection = the shard's parent directory name. Refuses to overwrite an existing index."""
    import pyarrow.parquet as pq  # noqa: PLC0415 -- optional dependency, only for this command

    from archivetrust.htr.benchmark.sources.tabular import IMAGE_COLUMNS, TEXT_COLUMNS, _pick  # noqa: PLC0415

    index_path = Path(index_path)
    if index_path.exists():
        raise FileExistsError(f"{index_path} exists; indexes are not rebuilt in place")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    db = _connect(index_path)
    counts: Counter[str] = Counter()
    files = sorted(Path(p) for p in parquet_files)
    for shard in files:
        parquet = pq.ParquetFile(shard)
        names = parquet.schema_arrow.names
        image_cols, text_cols = _pick(names, IMAGE_COLUMNS), _pick(names, TEXT_COLUMNS)
        if not image_cols or len(text_cols) != 1:
            counts["shards_skipped_unrecognized_columns"] += 1
            continue
        row = 0
        for batch in parquet.iter_batches(batch_size=batch_size, columns=[image_cols[0], text_cols[0]]):
            rows = []
            for record in batch.to_pylist():
                image, text = record[image_cols[0]], record[text_cols[0]] or ""
                data = image.get("bytes") if isinstance(image, dict) else image
                digest = sha256_bytes(data) if data else None
                hashed = _image_dhash(data) if data else None
                counts["rows"] += 1
                counts["rows_without_decodable_image"] += hashed is None
                rows.append((f"{shard.name}#row={row}", shard.parent.name, digest, hashed, sha256_text(text),
                             sha256_text(text_key(text)), len(text), *(_bands(hashed) if hashed else [None] * _BANDS)))
                row += 1
            db.executemany(f"INSERT INTO lines VALUES ({', '.join('?' * (7 + _BANDS))})", rows)
        db.commit()
    for column in ("image_sha256", "text_sha256", "text_norm_sha256", *(f"b{i}" for i in range(_BANDS))):
        db.execute(f"CREATE INDEX IF NOT EXISTS idx_{column} ON lines ({column})")
    meta = {"overlap_version": OVERLAP_VERSION, "built_at_utc": utc_now(), "shards": len(files),
            "shard_names_sha256": sha256_text("\n".join(f.name for f in files)), "counts": dict(counts)}
    db.execute("INSERT OR REPLACE INTO meta VALUES ('index', ?)", (json.dumps(meta),))
    db.commit()
    db.close()
    return meta


def check_overlap(frozen_dir: Path, index_path: Path, out_dir: Path) -> dict:
    record, lines, findings = verify_frozen(frozen_dir)
    if findings:
        raise RuntimeError(f"frozen benchmark failed verification: {[f.code for f in findings][:5]}")
    db = sqlite3.connect(f"file:{Path(index_path).as_posix()}?mode=ro", uri=True)
    meta = json.loads(db.execute("SELECT value FROM meta WHERE key='index'").fetchone()[0])
    matches: list[dict] = []
    summary: Counter[str] = Counter()
    for line in lines:
        data = (Path(frozen_dir) / line.line_image_path).read_bytes()
        hit: dict = {"line_id": line.line_id, "signals": []}
        exact_image = db.execute("SELECT source FROM lines WHERE image_sha256=? LIMIT 5", (line.image_sha256,)).fetchall()
        if exact_image:
            hit["signals"].append({"signal": "identical_image", "sources": [r[0] for r in exact_image]})
        norm_key = sha256_text(text_key(line.gt_canonical))
        long_enough = len(line.gt_canonical) >= MIN_TEXT_CHARS
        if long_enough:
            exact_text = db.execute("SELECT source FROM lines WHERE text_sha256=? LIMIT 5", (sha256_text(line.gt_canonical),)).fetchall()
            if exact_text:
                hit["signals"].append({"signal": "identical_text", "sources": [r[0] for r in exact_text]})
            else:
                norm_text = db.execute("SELECT source FROM lines WHERE text_norm_sha256=? LIMIT 5", (norm_key,)).fetchall()
                if norm_text:
                    hit["signals"].append({"signal": "identical_normalized_text", "sources": [r[0] for r in norm_text]})
        hashed = _image_dhash(data)
        if hashed:
            where = " OR ".join(f"b{i}=?" for i in range(_BANDS))
            near = [(src, h, t) for src, h, t in db.execute(
                f"SELECT source, dhash, text_norm_sha256 FROM lines WHERE {where} LIMIT 2000", _bands(hashed))
                if h and (int(h, 16) ^ int(hashed, 16)).bit_count() <= DHASH_MAX_DISTANCE]
            strong = [src for src, _, t in near if t == norm_key]
            if strong:
                hit["signals"].append({"signal": "near_image_and_same_text", "sources": strong[:5]})
            elif near:
                hit["signals"].append({"signal": "near_image_only_weak", "sources": [s for s, _, _ in near[:5]]})
        for signal in {s["signal"] for s in hit["signals"]}:
            summary[signal] += 1
        if hit["signals"]:
            matches.append(hit)
    db.close()
    strong_signals = {"identical_image", "identical_text", "identical_normalized_text", "near_image_and_same_text"}
    result = {
        "overlap_version": OVERLAP_VERSION,
        "benchmark_id": record["benchmark_id"],
        "manifest_sha256": record["manifest_sha256"],
        "index": {"path": str(index_path), "sha256": sha256_file(Path(index_path)), "meta": meta},
        "lines_checked": len(lines),
        "lines_with_strong_signal": sum(1 for m in matches if any(s["signal"] in strong_signals for s in m["signals"])),
        "signal_counts": dict(summary),
        "min_text_chars": MIN_TEXT_CHARS,
        "lines_too_short_for_text_check": sum(1 for line in lines if len(line.gt_canonical) < MIN_TEXT_CHARS),
        "caveat": "No match is not proof of no overlap: re-cropped, re-scanned or re-transcribed material evades these checks.",
        "checked_at_utc": utc_now(),
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "overlap.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "overlap_matches.jsonl").write_text("".join(json.dumps(m, ensure_ascii=False) + "\n" for m in matches), encoding="utf-8")
    return result
