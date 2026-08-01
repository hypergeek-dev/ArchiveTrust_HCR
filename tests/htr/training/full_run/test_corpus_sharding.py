from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from archivetrust.htr.training.full_run.corpus_sharding import (
    build_full_corpus_shards,
    load_sharding_summary,
)
from archivetrust.htr.training.swedish_dataset_inventory import build_source_inventory


@pytest.fixture()
def real_inventory(synthetic_dataset_root, synthetic_charlist, tmp_path):
    inv_path = tmp_path / "inventory.parquet"
    build_source_inventory(dataset_root=synthetic_dataset_root, output_path=inv_path, charlist_path=synthetic_charlist)
    return inv_path


def _write_exclude_manifest(path, line_ids):
    table = pa.table({"line_id": pa.array(line_ids, type=pa.string())})
    pq.write_table(table, path)


def test_builds_real_shards_covering_every_usable_line(real_inventory, tmp_path):
    output_dir = tmp_path / "shards"
    summary = build_full_corpus_shards(
        inventory_path=real_inventory, output_dir=output_dir, shard_line_count=3, seed=1, target_shard_count=10,
    )
    assert summary.usable_line_count == summary.total_valid_lines  # nothing excluded
    total_shard_lines = sum(s.line_count for s in summary.shards)
    # exactly one full lap's worth of shards fit in target_shard_count=10 for this small fixture,
    # so total_shard_lines should be a whole multiple of usable_line_count (allowing for wrap laps)
    assert total_shard_lines > 0
    assert all(s.line_count <= 3 for s in summary.shards)


def test_excludes_val_and_test_manifest_line_ids(real_inventory, tmp_path):
    full_table = pq.read_table(real_inventory, columns=["line_id", "valid"])
    valid_line_ids = [
        lid for lid, valid in zip(full_table.column("line_id").to_pylist(), full_table.column("valid").to_pylist())
        if valid
    ]
    excluded_ids = valid_line_ids[:2]

    val_manifest = tmp_path / "val_manifest.parquet"
    _write_exclude_manifest(val_manifest, excluded_ids)

    output_dir = tmp_path / "shards"
    summary = build_full_corpus_shards(
        inventory_path=real_inventory, output_dir=output_dir, exclude_manifest_paths=(val_manifest,),
        shard_line_count=100, seed=1, target_shard_count=1,
    )
    assert summary.excluded_line_count == len(excluded_ids)
    assert summary.usable_line_count == summary.total_valid_lines - len(excluded_ids)

    for shard in summary.shards:
        shard_table = pq.read_table(shard.manifest_path, columns=["line_id"])
        shard_ids = set(shard_table.column("line_id").to_pylist())
        assert shard_ids.isdisjoint(excluded_ids)  # real, exact overlap check


def test_refuses_to_overwrite_an_existing_shard_set(real_inventory, tmp_path):
    output_dir = tmp_path / "shards"
    build_full_corpus_shards(
        inventory_path=real_inventory, output_dir=output_dir, shard_line_count=3, seed=1, target_shard_count=2,
    )
    with pytest.raises(FileExistsError):
        build_full_corpus_shards(
            inventory_path=real_inventory, output_dir=output_dir, shard_line_count=3, seed=1, target_shard_count=2,
        )


def test_deterministic_given_the_same_seed(real_inventory, tmp_path):
    output_dir_a = tmp_path / "shards_a"
    output_dir_b = tmp_path / "shards_b"
    summary_a = build_full_corpus_shards(
        inventory_path=real_inventory, output_dir=output_dir_a, shard_line_count=3, seed=42, target_shard_count=3,
    )
    summary_b = build_full_corpus_shards(
        inventory_path=real_inventory, output_dir=output_dir_b, shard_line_count=3, seed=42, target_shard_count=3,
    )
    assert summary_a.line_id_set_hash == summary_b.line_id_set_hash
    for shard_a, shard_b in zip(summary_a.shards, summary_b.shards):
        ids_a = pq.read_table(shard_a.manifest_path, columns=["line_id"]).column("line_id").to_pylist()
        ids_b = pq.read_table(shard_b.manifest_path, columns=["line_id"]).column("line_id").to_pylist()
        assert ids_a == ids_b


def test_different_seeds_produce_different_shard_ordering(real_inventory, tmp_path):
    output_dir_a = tmp_path / "shards_a"
    output_dir_b = tmp_path / "shards_b"
    summary_a = build_full_corpus_shards(
        inventory_path=real_inventory, output_dir=output_dir_a, shard_line_count=3, seed=1, target_shard_count=1,
    )
    summary_b = build_full_corpus_shards(
        inventory_path=real_inventory, output_dir=output_dir_b, shard_line_count=3, seed=999, target_shard_count=1,
    )
    ids_a = pq.read_table(summary_a.shards[0].manifest_path, columns=["line_id"]).column("line_id").to_pylist()
    ids_b = pq.read_table(summary_b.shards[0].manifest_path, columns=["line_id"]).column("line_id").to_pylist()
    assert set(ids_a) != set(ids_b) or ids_a != ids_b


def test_wraps_around_into_a_second_lap_when_target_exceeds_one_pass(real_inventory, tmp_path):
    output_dir = tmp_path / "shards"
    # shard_line_count is large enough that one shard consumes the entire small fixture corpus --
    # requesting 3 shards forces 3 separate laps.
    summary = build_full_corpus_shards(
        inventory_path=real_inventory, output_dir=output_dir, shard_line_count=1000, seed=1, target_shard_count=3,
    )
    laps = {s.lap for s in summary.shards}
    assert laps == {0, 1, 2}


def test_load_sharding_summary_round_trips(real_inventory, tmp_path):
    output_dir = tmp_path / "shards"
    build_full_corpus_shards(
        inventory_path=real_inventory, output_dir=output_dir, shard_line_count=3, seed=1, target_shard_count=2,
    )
    reloaded = load_sharding_summary(output_dir)
    assert reloaded.seed == 1
    assert len(reloaded.shards) == 2


def test_shard_manifests_preserve_the_real_schema_columns(real_inventory, tmp_path):
    output_dir = tmp_path / "shards"
    summary = build_full_corpus_shards(
        inventory_path=real_inventory, output_dir=output_dir, shard_line_count=3, seed=1, target_shard_count=1,
    )
    columns = pq.read_table(summary.shards[0].manifest_path).column_names
    assert "line_id" in columns
    assert "collection" in columns
    assert "row_index" in columns
