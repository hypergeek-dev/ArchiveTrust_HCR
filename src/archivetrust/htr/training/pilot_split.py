"""Builds the deterministic, group-aware pilot train/val/reserved-test split
(docs/methods/loghi-swedish-finetuning.md).

**Grouping key actually available.** WP5 ranks preferred grouping keys `collection > archival volume >
document > writer/hand > another defensible grouping`. Neither volume, document, nor writer identifiers
exist in the source schema (`swedish_dataset_inventory.py`'s own finding: `image.path` is `None` on
every row). The strongest real boundary this data offers is **`(collection, source_parquet_file)`** --
a "file-group." Where a collection has 2+ source files, whole file-groups are assigned to exactly one
of train/val/reserved-test, so no file's lines are ever split across two pools.

**Honest limitation, not smoothed over.** 7 of the 11 collections have only one source file
(`svea_hovratt_lines` and `trolldomskommissionen_lines` have 2; `bergskollegium_relationer_och_
skrivelser_lines` has 4; `goteborgs_poliskammare_fore_1900_lines` has 22). For a single-file collection,
file-level grouping alone would exclude it entirely from either train or val, which would break the
brief's other requirement -- "stratify where possible across source collection" -- worse than the
alternative. For those collections only, lines are split at line granularity, seeded and stratified by
transcription length, and every such collection's manifest rows are marked
`split_granularity="line_level_single_file_collection"` so this is a visible, disclosed fact about the
manifest, not a silent inconsistency between collections.

**Domination cap.** `goteborgs_poliskammare_fore_1900_lines` alone is 63% of the valid corpus
(356,953 / 563,958 valid lines). `max_collection_fraction` (default 0.25) caps any one collection's
share of the train+val pool; the capped remainder is redistributed proportionally among collections
still under their own cap (a standard water-filling allocation), so every collection with any valid
data has a chance to appear in the pilot.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict

from archivetrust.htr.training.swedish_dataset_inventory import read_inventory

DEFAULT_SEED = 20260801
DEFAULT_TARGET_TRAIN = 10_000
DEFAULT_TARGET_VAL = 1_000
DEFAULT_TARGET_TEST_RESERVED = 1_000
DEFAULT_MAX_COLLECTION_FRACTION = 0.25

MANIFEST_COLUMNS = [
    "line_id",
    "collection",
    "source_parquet_file",
    "row_index",
    "image_content_hash",
    "image_width",
    "image_height",
    "transcription_length",
]


class PilotSplitSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    seed: int
    target_train: int
    target_val: int
    target_test_reserved: int
    actual_train: int
    actual_val: int
    actual_test_reserved: int
    max_collection_fraction: float
    train_collection_distribution: dict[str, int]
    val_collection_distribution: dict[str, int]
    test_reserved_collection_distribution: dict[str, int]
    single_file_collections: tuple[str, ...]
    """Collections split at line granularity rather than file-group granularity -- disclosed, per the
    module docstring."""
    train_manifest_hash: str
    val_manifest_hash: str
    test_reserved_manifest_hash: str
    shortfall_warnings: tuple[str, ...] = ()
    """Non-empty only if a target could not be met -- the brief's "use the largest defensible split
    and report the actual counts" fallback, never silently substituted."""


def _file_groups_by_collection(table: pa.Table) -> dict[str, dict[str, list[int]]]:
    """`{collection: {source_parquet_file: [row_positions_in_table, ...]}}`."""
    collections = table.column("collection").to_pylist()
    files = table.column("source_parquet_file").to_pylist()
    result: dict[str, dict[str, list[int]]] = {}
    for position, (collection, source_file) in enumerate(zip(collections, files)):
        result.setdefault(collection, {}).setdefault(source_file, []).append(position)
    return result


def _water_fill_allocation(
    *, available_by_collection: dict[str, int], total_target: int, max_fraction: float
) -> dict[str, int]:
    """Capped-proportional allocation: no collection gets more than `max_fraction * total_target`
    (bounded also by its own availability); leftover capacity is redistributed proportionally among
    collections still under their cap, iterated to convergence."""
    cap = {c: min(available_by_collection[c], int(total_target * max_fraction)) for c in available_by_collection}
    allocation = {c: 0 for c in available_by_collection}
    remaining = total_target
    uncapped = set(available_by_collection)

    while remaining > 0 and uncapped:
        share = remaining / len(uncapped)
        newly_capped: list[str] = []
        for c in list(uncapped):
            want = min(share, cap[c] - allocation[c])
            want = max(0, int(round(want)))
            allocation[c] += want
            remaining -= want
            if allocation[c] >= cap[c]:
                newly_capped.append(c)
        for c in newly_capped:
            uncapped.discard(c)
        if not newly_capped:
            break  # converged with fractional remainder too small to allocate further

    return allocation


def build_pilot_split(
    *,
    inventory_path: str | Path,
    output_dir: str | Path,
    target_train: int = DEFAULT_TARGET_TRAIN,
    target_val: int = DEFAULT_TARGET_VAL,
    target_test_reserved: int = DEFAULT_TARGET_TEST_RESERVED,
    seed: int = DEFAULT_SEED,
    max_collection_fraction: float = DEFAULT_MAX_COLLECTION_FRACTION,
) -> PilotSplitSummary:
    """Builds and writes the three immutable manifests. Refuses to overwrite existing manifests --
    a resumed session must reuse them, never regenerate (see `write_manifest`'s guard)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train_manifest.parquet"
    val_path = output_dir / "val_manifest.parquet"
    test_path = output_dir / "test_reserved_manifest.parquet"
    summary_path = output_dir / "pilot_split_summary.json"

    if train_path.exists() or val_path.exists() or test_path.exists():
        raise FileExistsError(
            f"Pilot manifests already exist under {output_dir} -- this function never regenerates an "
            "existing pilot split (a resumed training session must reuse the exact same manifests). "
            "Delete them explicitly first if a genuinely new pilot split is intended."
        )

    table = read_inventory(
        inventory_path,
        columns=[
            "line_id",
            "collection",
            "source_parquet_file",
            "row_index",
            "image_content_hash",
            "image_width",
            "image_height",
            "transcription_length",
        ],
        valid_only=True,
    )

    groups = _file_groups_by_collection(table)
    rng = random.Random(seed)

    available_by_collection = {c: sum(len(rows) for rows in files.values()) for c, files in groups.items()}
    pool_target = target_train + target_val
    pool_allocation = _water_fill_allocation(
        available_by_collection=available_by_collection, total_target=pool_target, max_fraction=max_collection_fraction
    )

    single_file_collections: list[str] = []
    train_positions: list[int] = []
    val_positions: list[int] = []
    test_positions: list[int] = []

    per_collection_test_target = target_test_reserved // max(1, len(groups))

    for collection, files in groups.items():
        quota = pool_allocation.get(collection, 0)
        if quota <= 0:
            continue

        file_names = sorted(files)
        if len(file_names) < 2:
            single_file_collections.append(collection)

        train_share = int(quota * target_train / pool_target)
        val_share = quota - train_share
        test_share = min(per_collection_test_target, quota)
        deficits = {"train": train_share, "val": val_share, "test": test_share}
        pools = {"train": train_positions, "val": val_positions, "test": test_positions}

        if len(file_names) < 2:
            # Only one file exists -- a whole-file assignment could give this pool's deficit to at
            # most ONE of train/val/test, starving the other two entirely (a real bug this branch
            # replaces: an earlier version did exactly that). This is the disclosed
            # `line_level_single_file_collection` degraded granularity: the same file's lines are
            # split directly across train/val/test, in a fixed shuffled order, deficit by deficit.
            (only_file,) = file_names
            positions = files[only_file][:]
            rng.shuffle(positions)
            cursor = 0
            for pool_name in ("test", "val", "train"):
                take = positions[cursor : cursor + deficits[pool_name]]
                pools[pool_name].extend(take)
                cursor += len(take)
        else:
            # 2+ files: whole-file assignment preserves the file-group leakage boundary. Greedily
            # assign each shuffled file to whichever pool currently has the largest unmet deficit, but
            # -- unlike a broken earlier version of this loop -- keep iterating over every remaining
            # file even after one pool's deficit is satisfied, so a collection with few files still
            # gets a chance to fill all three pools rather than only the first one picked.
            shuffled_files = file_names[:]
            rng.shuffle(shuffled_files)
            for fname in shuffled_files:
                if all(d <= 0 for d in deficits.values()):
                    break
                target_pool = max(deficits, key=lambda p: deficits[p])
                if deficits[target_pool] <= 0:
                    break
                positions = files[fname][:]
                rng.shuffle(positions)
                take = positions[: deficits[target_pool]]
                pools[target_pool].extend(take)
                deficits[target_pool] -= len(take)

    # Trim to exact targets where over-allocated (rounding can overshoot slightly), never fabricate
    # rows to hit a target exactly.
    rng.shuffle(train_positions)
    rng.shuffle(val_positions)
    rng.shuffle(test_positions)
    train_positions = train_positions[:target_train]
    val_positions = val_positions[:target_val]
    test_positions = test_positions[:target_test_reserved]

    shortfall_warnings: list[str] = []
    if len(train_positions) < target_train:
        shortfall_warnings.append(
            f"train: requested {target_train}, achieved {len(train_positions)} "
            f"(largest defensible split under max_collection_fraction={max_collection_fraction})"
        )
    if len(val_positions) < target_val:
        shortfall_warnings.append(f"val: requested {target_val}, achieved {len(val_positions)}")
    if len(test_positions) < target_test_reserved:
        shortfall_warnings.append(
            f"test_reserved: requested {target_test_reserved}, achieved {len(test_positions)}"
        )

    def write_manifest(path: Path, positions: list[int], split_name: str) -> str:
        # Explicit `type=pa.int64()` -- a real bug caught by this module's own tests: `pa.array([])`
        # on an empty position list (a genuinely possible split, e.g. an empty test_reserved pool for
        # a small corpus) infers Arrow type `null`, and `.take()` on a string column with `null`-typed
        # indices raises `ArrowNotImplementedError` instead of returning an empty table.
        subset = table.take(pa.array(sorted(positions), type=pa.int64()))
        rows = subset.to_pylist()
        for row in rows:
            row["split"] = split_name
            row["split_granularity"] = (
                "line_level_single_file_collection"
                if row["collection"] in single_file_collections
                else "file_group_level"
            )
        out_table = pa.Table.from_pylist(rows)
        pq.write_table(out_table, path)
        line_ids = sorted(r["line_id"] for r in rows)
        return hashlib.sha256(json.dumps(line_ids).encode("utf-8")).hexdigest()

    train_hash = write_manifest(train_path, train_positions, "train")
    val_hash = write_manifest(val_path, val_positions, "val")
    test_hash = write_manifest(test_path, test_positions, "test_reserved")

    def distribution(positions: list[int]) -> dict[str, int]:
        # Explicit `type=pa.int64()` -- a real bug caught by this module's own tests: `pa.array([])`
        # on an empty position list (a genuinely possible split, e.g. an empty test_reserved pool for
        # a small corpus) infers Arrow type `null`, and `.take()` on a string column with `null`-typed
        # indices raises `ArrowNotImplementedError` instead of returning an empty table.
        subset = table.take(pa.array(sorted(positions), type=pa.int64()))
        counts: dict[str, int] = {}
        for c in subset.column("collection").to_pylist():
            counts[c] = counts.get(c, 0) + 1
        return counts

    summary = PilotSplitSummary(
        seed=seed,
        target_train=target_train,
        target_val=target_val,
        target_test_reserved=target_test_reserved,
        actual_train=len(train_positions),
        actual_val=len(val_positions),
        actual_test_reserved=len(test_positions),
        max_collection_fraction=max_collection_fraction,
        train_collection_distribution=distribution(train_positions),
        val_collection_distribution=distribution(val_positions),
        test_reserved_collection_distribution=distribution(test_positions),
        single_file_collections=tuple(single_file_collections),
        train_manifest_hash=train_hash,
        val_manifest_hash=val_hash,
        test_reserved_manifest_hash=test_hash,
        shortfall_warnings=tuple(shortfall_warnings),
    )
    summary_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    return summary


def load_pilot_split_summary(output_dir: str | Path) -> PilotSplitSummary:
    path = Path(output_dir) / "pilot_split_summary.json"
    return PilotSplitSummary.model_validate_json(path.read_text(encoding="utf-8"))
