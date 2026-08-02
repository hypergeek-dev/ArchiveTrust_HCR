"""Turns full-corpus Parquet shard manifests into what the pinned `loghi-htr` container actually
consumes: extracted PNG line images plus UTF-8 `<container_path>\\t<ground_truth>` list files.

**Why this module exists.** `corpus_sharding.py` produces Parquet manifests -- correct, hashable,
auditable descriptions of *which lines* each shard contains. The container cannot read Parquet:
`data/manager.py:257` does `open(file_path, encoding="utf-8")` and iterates lines. The pilot bridged
this gap with `loghi_training_data.prepare_loghi_training_data()`; the full-run path never called it,
so `cmd_prepare` produced a run that could not train. That was caught by a real Gate 4 shard attempt
failing in 53s (`stop_reason=epoch_failed`, empty `epoch_output/epoch_1/`). This module is the missing
call, plus the two things the pilot never had to solve because it had a single 9,999-line split.

**Trap 1: laps triplicate the images.** The 171-shard plan is 3 laps over the *same* 562,123 usable
lines (`sharding_summary.json`: lap 0/1/2 each cover the identical set, reshuffled). Calling the
adapter once per shard would extract every image once per lap -- ~1.7M files, ~250GB, instead of
562,123 files and ~82GB. So images are extracted exactly once, from the union of lap 0, and every
shard's list file references that one shared pool.

**Trap 2: the container mount forces a flat layout.** `container_epoch_runner._build_argv` mounts
`Path(train_list_path).parent` at `/lists`, and image paths inside a list file are
`/lists/images/<split>/<name>.png`. A shard list file must therefore be a *sibling* of `images/`, not
in a subdirectory -- otherwise `/lists` resolves to the subdirectory and `/lists/images` does not
exist. Hence the flat pool layout below, rather than a per-run `prepared-data/` tree.

**Why the pool is shared rather than per-run.** The code-revision guard invalidates a prepared run on
every commit, so runs are re-prepared often (six times during the audits alone). Re-extracting 82GB
each time is untenable. The pool is keyed by the facts that determine its contents -- the dataset
hash and the shard-plan hash -- so reuse is verifiable rather than assumed, and a changed dataset or
shard plan builds a distinct pool instead of silently reusing a stale one. Image filenames are
derived from `line_id` by the adapter itself, so the pool is content-addressed by construction.

Layout (flat, by necessity):

    training/_prepared_data/<dataset8>_<plan8>/
        images/train/<line_id>.png        562,123 files, written once
        images/val/<line_id>.png            1,000 files
        train_list.txt                      full union list (adapter output)
        val_list.txt                        validation list (adapter output)
        shard_00000_list.txt ...            one per shard, derived from train_list.txt
        preparation_report.json             adapter's own report
        pool_manifest.json                  what this pool is keyed to, for verifiable reuse
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict

from archivetrust.htr.training.full_run.corpus_sharding import CorpusShardingSummary
from archivetrust.htr.training.loghi_training_data import (
    _line_id_to_filename,
    prepare_loghi_training_data,
)

POOL_MANIFEST_NAME = "pool_manifest.json"
SHARD_LIST_TEMPLATE = "shard_{index:05d}_list.txt"


class ShardTrainingDataPaths(BaseModel):
    """Where the trainer-consumable artifacts actually live, resolved once at prepare time so the
    orchestrator never has to re-derive them."""

    model_config = ConfigDict(frozen=True)

    pool_dir: str
    validation_list_path: str
    shard_list_paths: tuple[str, ...]
    """Index-aligned with `CorpusShardingSummary.shards` -- `shard_list_paths[i]` is what
    `--train_list` receives for `shards[i]`."""
    train_image_count: int
    validation_image_count: int
    reused_existing_pool: bool
    dataset_hash: str
    training_manifest_hash: str


def pool_directory(*, training_root: str | Path, dataset_hash: str, training_manifest_hash: str) -> Path:
    """Keyed by both hashes: a changed corpus *or* a changed shard plan yields a different pool, so
    reuse can never silently serve data that does not match the run being prepared."""
    return Path(training_root) / "_prepared_data" / f"{dataset_hash[:8]}_{training_manifest_hash[:8]}"


def _union_manifest_of_first_lap(summary: CorpusShardingSummary, destination: Path) -> Path:
    """Every usable line exactly once. Lap 0 *is* the usable set by construction (verified against the
    real plan: lap 0 = 562,123 rows, 562,123 unique, equal to `usable_line_count`), so concatenating
    its shards is both the complete set and duplicate-free -- no dedup pass required."""
    lap0 = [s for s in summary.shards if s.lap == 0]
    if not lap0:
        raise ValueError("Sharding summary contains no lap-0 shards -- cannot build a union manifest.")
    tables = [pq.read_table(s.manifest_path) for s in lap0]
    union = pa.concat_tables(tables)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(union, destination)
    return destination


def _write_shard_list_files(
    *, summary: CorpusShardingSummary, pool_dir: Path, train_list_path: Path
) -> tuple[str, ...]:
    """Derives one list file per shard from the already-written union list, without re-reading the
    source Parquet. Safe because the adapter's own `_line_id_to_filename` is a pure function of
    `line_id`, so a shard's lines can be looked up rather than re-extracted."""
    by_filename: dict[str, str] = {}
    with train_list_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            container_path = line.split("\t", 1)[0]
            by_filename[Path(container_path).stem] = line

    written: list[str] = []
    for shard in summary.shards:
        line_ids = pq.read_table(shard.manifest_path, columns=["line_id"]).column("line_id").to_pylist()
        target = pool_dir / SHARD_LIST_TEMPLATE.format(index=shard.shard_index)
        missing = 0
        with target.open("w", encoding="utf-8") as out:
            for line_id in line_ids:
                entry = by_filename.get(_line_id_to_filename(line_id))
                if entry is None:
                    # A manifest row whose image the adapter could not extract (counted in the
                    # adapter's own excluded_count). Skipped here rather than written as a dangling
                    # path the container would fail on -- and surfaced, never silent.
                    missing += 1
                    continue
                out.write(entry + "\n")
        if missing:
            raise RuntimeError(
                f"Shard {shard.shard_index} references {missing} line(s) absent from the extracted "
                f"image pool at {pool_dir}. The pool is incomplete or does not match this shard plan; "
                "refusing to write a list file the container would fail on."
            )
        written.append(str(target))
    return tuple(written)


def _read_pool_manifest(pool_dir: Path) -> dict | None:
    path = pool_dir / POOL_MANIFEST_NAME
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_pool_manifest(pool_dir: Path, payload: dict) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(pool_dir), prefix=".tmp-pool-manifest-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, str(pool_dir / POOL_MANIFEST_NAME))
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def build_or_reuse_shard_training_data(
    *,
    training_root: str | Path,
    dataset_root: str | Path,
    sharding_summary: CorpusShardingSummary,
    validation_manifest_path: str | Path,
    dataset_hash: str,
    training_manifest_hash: str,
    force_rebuild: bool = False,
) -> ShardTrainingDataPaths:
    """Extracts images once and writes every list file the run needs. Idempotent: an existing pool
    whose `pool_manifest.json` matches both hashes and whose expected files are all present is reused
    as-is."""
    pool_dir = pool_directory(
        training_root=training_root, dataset_hash=dataset_hash, training_manifest_hash=training_manifest_hash
    )
    expected_shard_lists = [
        pool_dir / SHARD_LIST_TEMPLATE.format(index=s.shard_index) for s in sharding_summary.shards
    ]
    val_list_path = pool_dir / "val_list.txt"
    train_list_path = pool_dir / "train_list.txt"

    existing = _read_pool_manifest(pool_dir)
    complete = (
        existing is not None
        and existing.get("dataset_hash") == dataset_hash
        and existing.get("training_manifest_hash") == training_manifest_hash
        and existing.get("shard_count") == len(sharding_summary.shards)
        and val_list_path.exists()
        and train_list_path.exists()
        and all(p.exists() for p in expected_shard_lists)
    )
    if complete and not force_rebuild:
        return ShardTrainingDataPaths(
            pool_dir=str(pool_dir),
            validation_list_path=str(val_list_path),
            shard_list_paths=tuple(str(p) for p in expected_shard_lists),
            train_image_count=existing.get("train_image_count", 0),
            validation_image_count=existing.get("validation_image_count", 0),
            reused_existing_pool=True,
            dataset_hash=dataset_hash,
            training_manifest_hash=training_manifest_hash,
        )

    pool_dir.mkdir(parents=True, exist_ok=True)
    union_manifest = _union_manifest_of_first_lap(sharding_summary, pool_dir / ".union_manifest.parquet")
    try:
        report = prepare_loghi_training_data(
            dataset_root=dataset_root,
            manifest_paths={"train": union_manifest, "val": Path(validation_manifest_path)},
            output_dir=pool_dir,
        )
    finally:
        union_manifest.unlink(missing_ok=True)

    counts = {s.split_name: s.sample_count for s in report.splits}
    shard_list_paths = _write_shard_list_files(
        summary=sharding_summary, pool_dir=pool_dir, train_list_path=train_list_path
    )

    _write_pool_manifest(
        pool_dir,
        {
            "dataset_hash": dataset_hash,
            "training_manifest_hash": training_manifest_hash,
            "shard_count": len(sharding_summary.shards),
            "train_image_count": counts.get("train", 0),
            "validation_image_count": counts.get("val", 0),
            "adapter_excluded_count": report.excluded_count,
            "dataset_root": str(dataset_root),
        },
    )

    return ShardTrainingDataPaths(
        pool_dir=str(pool_dir),
        validation_list_path=str(val_list_path),
        shard_list_paths=shard_list_paths,
        train_image_count=counts.get("train", 0),
        validation_image_count=counts.get("val", 0),
        reused_existing_pool=False,
        dataset_hash=dataset_hash,
        training_manifest_hash=training_manifest_hash,
    )
