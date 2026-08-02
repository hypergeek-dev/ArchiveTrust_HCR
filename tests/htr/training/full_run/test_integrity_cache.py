from __future__ import annotations

import os
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from archivetrust.htr.training.full_run.integrity_cache import (
    build_cache,
    cache_is_valid,
    deterministic_sample_indices,
    load_cache,
    sampled_shard_hash,
    save_cache,
)


def _shard(path: Path, ids):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"line_id": pa.array(ids, type=pa.string())}), path)
    return path


@pytest.fixture()
def setup(tmp_path):
    run_dir = tmp_path / "run"
    shards = [_shard(run_dir / "shards" / f"s{i}.parquet", [f"l{i}_{j}" for j in range(5)]) for i in range(6)]
    ident = dict(dataset_hash="d" * 64, training_manifest_hash="t" * 64,
                 validation_manifest_hash="v" * 64, configuration_hash="c" * 64)
    cache = build_cache(shard_paths=shards, validation_overlap=0, test_overlap=0, **ident)
    save_cache(run_dir, cache)
    return {"run_dir": run_dir, "shards": shards, "ident": ident}


def test_cache_round_trips(setup):
    loaded = load_cache(setup["run_dir"])
    assert loaded is not None
    assert loaded.shard_count == 6
    assert loaded.validation_overlap == 0


def test_unchanged_inputs_validate_from_cache(setup):
    ok, reason = cache_is_valid(load_cache(setup["run_dir"]), shard_paths=setup["shards"], **setup["ident"])
    assert ok is True
    assert "fingerprint matches" in reason


def test_missing_cache_forces_full_validation(setup):
    ok, reason = cache_is_valid(None, shard_paths=setup["shards"], **setup["ident"])
    assert ok is False
    assert "no cached" in reason


@pytest.mark.parametrize("field,label", [
    ("dataset_hash", "dataset hash"),
    ("training_manifest_hash", "training manifest hash"),
    ("validation_manifest_hash", "validation manifest hash"),
    ("configuration_hash", "configuration hash"),
])
def test_any_identity_change_invalidates_the_cache(setup, field, label):
    ident = dict(setup["ident"]); ident[field] = "z" * 64
    ok, reason = cache_is_valid(load_cache(setup["run_dir"]), shard_paths=setup["shards"], **ident)
    assert ok is False
    assert label in reason


def test_changed_shard_content_invalidates_via_size_or_mtime(setup):
    """A rewritten shard must invalidate -- this is the tampering case the cache must never mask."""
    target = setup["shards"][2]
    _shard(target, [f"tampered_{j}" for j in range(40)])  # different content AND size
    ok, reason = cache_is_valid(load_cache(setup["run_dir"]), shard_paths=setup["shards"], **setup["ident"])
    assert ok is False
    assert "size changed" in reason or "modification time changed" in reason


def test_touched_shard_invalidates_even_when_size_is_identical(setup):
    """Same bytes, new mtime -- still invalidates, because the cache cannot prove content is
    unchanged from size alone."""
    target = setup["shards"][1]
    st = target.stat()
    os.utime(target, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    ok, reason = cache_is_valid(load_cache(setup["run_dir"]), shard_paths=setup["shards"], **setup["ident"])
    assert ok is False
    assert "modification time changed" in reason


def test_missing_shard_file_invalidates(setup):
    setup["shards"][0].unlink()
    ok, reason = cache_is_valid(load_cache(setup["run_dir"]), shard_paths=setup["shards"], **setup["ident"])
    assert ok is False
    assert "missing" in reason or "size changed" in reason


def test_added_shard_invalidates_on_count(setup, tmp_path):
    extra = _shard(setup["run_dir"] / "shards" / "s99.parquet", ["x1"])
    ok, reason = cache_is_valid(
        load_cache(setup["run_dir"]), shard_paths=setup["shards"] + [extra], **setup["ident"])
    assert ok is False
    assert "shard count changed" in reason


def test_sample_is_deterministic_and_bounded(setup):
    a = deterministic_sample_indices(manifest_hash="a1b2c3d4e5f60718", shard_count=171)
    b = deterministic_sample_indices(manifest_hash="a1b2c3d4e5f60718", shard_count=171)
    assert a == b, "sample must be reproducible for auditability"
    assert len(a) == 8
    assert all(0 <= i < 171 for i in a)


def test_sample_covers_everything_when_the_set_is_small(setup):
    assert deterministic_sample_indices(manifest_hash="ff" * 8, shard_count=3) == [0, 1, 2]


def test_sampled_hash_actually_reads_content(setup):
    """The cache hit path must still re-read real bytes, not trust metadata alone."""
    before = sampled_shard_hash(setup["shards"], [0, 1])
    _shard(setup["shards"][1], ["completely", "different", "content"])
    after = sampled_shard_hash(setup["shards"], [0, 1])
    assert before != after
