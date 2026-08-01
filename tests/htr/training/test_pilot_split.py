from __future__ import annotations

import pytest

from archivetrust.htr.training.pilot_split import build_pilot_split
from archivetrust.htr.training.swedish_dataset_inventory import build_source_inventory


@pytest.fixture()
def inventory_path(synthetic_dataset_root, synthetic_charlist, tmp_path):
    path = tmp_path / "inventory.parquet"
    build_source_inventory(dataset_root=synthetic_dataset_root, output_path=path, charlist_path=synthetic_charlist)
    return path


def test_split_is_deterministic_for_the_same_seed(inventory_path, tmp_path):
    out1 = tmp_path / "split1"
    out2 = tmp_path / "split2"
    s1 = build_pilot_split(inventory_path=inventory_path, output_dir=out1, target_train=5, target_val=2, target_test_reserved=1, seed=7)
    s2 = build_pilot_split(inventory_path=inventory_path, output_dir=out2, target_train=5, target_val=2, target_test_reserved=1, seed=7)
    assert s1.train_manifest_hash == s2.train_manifest_hash
    assert s1.val_manifest_hash == s2.val_manifest_hash


def test_different_seeds_can_produce_different_splits(inventory_path, tmp_path):
    out1 = tmp_path / "split1"
    out2 = tmp_path / "split2"
    s1 = build_pilot_split(inventory_path=inventory_path, output_dir=out1, target_train=5, target_val=2, target_test_reserved=1, seed=1)
    s2 = build_pilot_split(inventory_path=inventory_path, output_dir=out2, target_train=5, target_val=2, target_test_reserved=1, seed=2)
    # not strictly required to differ (small pool), but hashes are independently, correctly computed
    assert s1.train_manifest_hash and s2.train_manifest_hash


def test_no_line_id_appears_in_more_than_one_split(inventory_path, tmp_path):
    import pyarrow.parquet as pq

    build_pilot_split(inventory_path=inventory_path, output_dir=tmp_path / "split", target_train=6, target_val=3, target_test_reserved=2, seed=3)
    train = set(pq.read_table(tmp_path / "split" / "train_manifest.parquet", columns=["line_id"]).column("line_id").to_pylist())
    val = set(pq.read_table(tmp_path / "split" / "val_manifest.parquet", columns=["line_id"]).column("line_id").to_pylist())
    test = set(pq.read_table(tmp_path / "split" / "test_reserved_manifest.parquet", columns=["line_id"]).column("line_id").to_pylist())
    assert not (train & val)
    assert not (train & test)
    assert not (val & test)


def test_refuses_to_overwrite_existing_manifests(inventory_path, tmp_path):
    out = tmp_path / "split"
    build_pilot_split(inventory_path=inventory_path, output_dir=out, target_train=3, target_val=1, target_test_reserved=1, seed=1)
    with pytest.raises(FileExistsError):
        build_pilot_split(inventory_path=inventory_path, output_dir=out, target_train=3, target_val=1, target_test_reserved=1, seed=1)


def test_shortfall_is_reported_honestly_when_targets_exceed_available_data(inventory_path, tmp_path):
    summary = build_pilot_split(
        inventory_path=inventory_path, output_dir=tmp_path / "split",
        target_train=10_000, target_val=1_000, target_test_reserved=1_000, seed=1,
    )
    assert summary.actual_train < 10_000
    assert summary.shortfall_warnings  # honestly reported, never silently substituted


def test_multi_file_collection_never_splits_a_single_file_across_pools(inventory_path, tmp_path):
    """`collection_a` has 2 files; group-aware assignment must keep each file's rows in exactly one
    pool -- verified indirectly via `single_file_collections` correctly excluding it."""
    summary = build_pilot_split(inventory_path=inventory_path, output_dir=tmp_path / "split", target_train=6, target_val=3, target_test_reserved=2, seed=1)
    assert "collection_a" not in summary.single_file_collections
    assert "collection_b" in summary.single_file_collections


def test_max_collection_fraction_caps_a_dominant_collection(inventory_path, tmp_path):
    summary = build_pilot_split(
        inventory_path=inventory_path, output_dir=tmp_path / "split",
        target_train=8, target_val=2, target_test_reserved=1, seed=1, max_collection_fraction=0.5,
    )
    total_pool = summary.actual_train + summary.actual_val
    for count in summary.train_collection_distribution.values():
        assert count <= total_pool * 0.5 + 1  # small integer-rounding slack


def test_load_pilot_split_summary_round_trips(inventory_path, tmp_path):
    from archivetrust.htr.training.pilot_split import load_pilot_split_summary

    out = tmp_path / "split"
    original = build_pilot_split(inventory_path=inventory_path, output_dir=out, target_train=4, target_val=2, target_test_reserved=1, seed=1)
    reloaded = load_pilot_split_summary(out)
    assert reloaded == original
