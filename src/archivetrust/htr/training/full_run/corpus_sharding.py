"""Deterministic full-corpus sharding -- one full-run "epoch" = one shard = one container invocation,
reusing `training_session.py`'s existing one-epoch-per-invocation engine completely unmodified for the
mechanics. Built here, at the ArchiveTrust level, specifically to avoid the pinned `loghi-htr` commit's
own `--steps_per_epoch` flag, which is upstream-documented broken (`modes/training.py:81`,
`# FIXME: steps_per_epoch is not working properly`) -- confirmed by reading the pinned source directly,
not assumed.

**The pilot's train/val/test split stays honored, not re-litigated.** Shards are built from every
valid inventory line *except* the pilot's own `val_manifest`/`test_reserved_manifest` line IDs -- the
pilot's train portion legitimately becomes part of the full corpus (the brief's explicit instruction:
"do not exclude it merely because it was used during the pilot"), while its validation and reserved
test lines stay held out, exactly as they always were.

**Wrapping**: once every usable line has appeared in exactly one shard (one "lap"), a new lap begins
with a freshly re-shuffled order (a seed derived from the original), so a long-running full-corpus
training session can keep producing shards indefinitely rather than exhausting after one pass.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict

from archivetrust.htr.training.swedish_dataset_inventory import read_inventory

SHARD_MANIFEST_COLUMNS = [
    "line_id", "collection", "source_parquet_file", "row_index", "image_content_hash",
    "image_width", "image_height", "transcription_length",
]


class ShardInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    shard_index: int
    lap: int
    """Which full pass over the corpus this shard belongs to -- `0` for the first lap, `1` for the
    re-shuffled wrap-around, etc."""
    line_count: int
    manifest_path: str


class CorpusShardingSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    seed: int
    shard_line_count: int
    total_valid_lines: int
    excluded_line_count: int
    usable_line_count: int
    shards: tuple[ShardInfo, ...]
    excluded_manifest_paths: tuple[str, ...]
    line_id_set_hash: str
    """sha256 over the sorted, real usable line-ID set -- reproducibility evidence, the same
    discipline `pilot_split.py`'s own manifest hashes already use."""


def _excluded_line_ids(manifest_paths: tuple[str | Path, ...]) -> set[str]:
    excluded: set[str] = set()
    for path in manifest_paths:
        path = Path(path)
        if not path.exists():
            continue
        table = pq.read_table(path, columns=["line_id"])
        excluded.update(table.column("line_id").to_pylist())
    return excluded


def build_full_corpus_shards(
    *,
    inventory_path: str | Path,
    output_dir: str | Path,
    exclude_manifest_paths: tuple[str | Path, ...] = (),
    shard_line_count: int,
    seed: int,
    target_shard_count: int,
) -> CorpusShardingSummary:
    """Refuses to overwrite an existing `output_dir` with content in it -- the same "never regenerate
    an existing split" discipline `pilot_split.py::build_pilot_split` already established, so a
    resumed full run always reuses the exact same shard manifests it started with.
    """
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Shard manifests already exist under {output_dir} -- this function never regenerates an "
            "existing shard set (a resumed full run must reuse the exact same shards). Delete them "
            "explicitly first if a genuinely new sharding is intended."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    table = read_inventory(inventory_path, columns=SHARD_MANIFEST_COLUMNS, valid_only=True)
    total_valid_lines = table.num_rows

    excluded = _excluded_line_ids(exclude_manifest_paths)
    excluded_count = 0
    if excluded:
        keep_mask = [line_id not in excluded for line_id in table.column("line_id").to_pylist()]
        excluded_count = keep_mask.count(False)
        table = table.filter(pa.array(keep_mask))

    usable_line_count = table.num_rows
    line_ids_sorted = sorted(table.column("line_id").to_pylist())
    line_id_set_hash = hashlib.sha256(json.dumps(line_ids_sorted).encode("utf-8")).hexdigest()

    shards: list[ShardInfo] = []
    shard_index = 0
    lap = 0
    positions = list(range(usable_line_count))
    rng = random.Random(seed)
    rng.shuffle(positions)
    cursor = 0

    while shard_index < target_shard_count and usable_line_count > 0:
        if cursor >= usable_line_count:
            lap += 1
            rng = random.Random(seed + lap)
            positions = list(range(usable_line_count))
            rng.shuffle(positions)
            cursor = 0

        take = positions[cursor : cursor + shard_line_count]
        cursor += len(take)

        subset = table.take(pa.array(sorted(take), type=pa.int64()))
        manifest_path = output_dir / f"shard_{shard_index:05d}.parquet"
        pq.write_table(subset, manifest_path)
        shards.append(
            ShardInfo(shard_index=shard_index, lap=lap, line_count=subset.num_rows, manifest_path=str(manifest_path))
        )
        shard_index += 1

    summary = CorpusShardingSummary(
        seed=seed,
        shard_line_count=shard_line_count,
        total_valid_lines=total_valid_lines,
        excluded_line_count=excluded_count,
        usable_line_count=usable_line_count,
        shards=tuple(shards),
        excluded_manifest_paths=tuple(str(p) for p in exclude_manifest_paths),
        line_id_set_hash=line_id_set_hash,
    )
    _write_summary(summary, output_dir / "sharding_summary.json")
    return summary


def _write_summary(summary: CorpusShardingSummary, path: Path) -> None:
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-sharding-summary-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(summary.model_dump_json(indent=2))
        os.replace(tmp_path, str(path))
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def load_sharding_summary(output_dir: str | Path) -> CorpusShardingSummary:
    path = Path(output_dir) / "sharding_summary.json"
    return CorpusShardingSummary.model_validate_json(path.read_text(encoding="utf-8"))
