from __future__ import annotations

import pytest

from archivetrust.htr.training.full_run.corpus_sharding import ShardInfo
from archivetrust.htr.training.full_run.epoch_accounting import (
    compute_epoch_position,
    shards_per_epoch_from_plan,
)

SHARDS_PER_EPOCH = 57
TOTAL = 171


def _plan(shards_per_epoch=SHARDS_PER_EPOCH, laps=3):
    """Mirrors the real plan: `laps` passes over the same line set, `shards_per_epoch` shards each."""
    out = []
    idx = 0
    for lap in range(laps):
        for _ in range(shards_per_epoch):
            out.append(ShardInfo(shard_index=idx, lap=lap, line_count=9999, manifest_path=f"s{idx}.parquet"))
            idx += 1
    return tuple(out)


def test_shards_per_epoch_is_derived_from_lap_zero_not_assumed():
    assert shards_per_epoch_from_plan(_plan()) == SHARDS_PER_EPOCH


def test_one_shard_does_not_increment_epochs_completed():
    """The core terminology defect: completing one shard is 1/57th of an epoch, not an epoch."""
    pos = compute_epoch_position(global_shards_completed=1, shards=_plan(), total_shards_planned=TOTAL)
    assert pos.epochs_completed == 0
    assert pos.shards_completed_in_current_epoch == 1
    assert pos.epoch_progress == pytest.approx(1 / 57)
    assert pos.global_shards_completed == 1
    assert pos.is_epoch_boundary is False


def test_zero_shards_is_a_clean_start_not_an_epoch_boundary():
    pos = compute_epoch_position(global_shards_completed=0, shards=_plan(), total_shards_planned=TOTAL)
    assert pos.epochs_completed == 0
    assert pos.shards_completed_in_current_epoch == 0
    assert pos.epoch_progress == 0.0
    assert pos.is_epoch_boundary is False


def test_shard_56_is_still_inside_epoch_1():
    pos = compute_epoch_position(global_shards_completed=56, shards=_plan(), total_shards_planned=TOTAL)
    assert pos.epochs_completed == 0
    assert pos.shards_completed_in_current_epoch == 56
    assert pos.is_epoch_boundary is False


def test_shard_57_completes_epoch_1():
    pos = compute_epoch_position(global_shards_completed=57, shards=_plan(), total_shards_planned=TOTAL)
    assert pos.epochs_completed == 1
    assert pos.shards_completed_in_current_epoch == 0
    assert pos.epoch_progress == 0.0
    assert pos.is_epoch_boundary is True


def test_shard_58_is_shard_1_of_epoch_2():
    pos = compute_epoch_position(global_shards_completed=58, shards=_plan(), total_shards_planned=TOTAL)
    assert pos.epochs_completed == 1
    assert pos.shards_completed_in_current_epoch == 1
    assert pos.is_epoch_boundary is False


def test_shard_171_completes_all_three_epochs():
    pos = compute_epoch_position(global_shards_completed=171, shards=_plan(), total_shards_planned=TOTAL)
    assert pos.epochs_completed == 3
    assert pos.is_epoch_boundary is True
    assert pos.total_epochs_planned == 3


def test_epoch_boundary_fires_once_per_epoch_only():
    boundaries = [
        n for n in range(1, TOTAL + 1)
        if compute_epoch_position(global_shards_completed=n, shards=_plan(), total_shards_planned=TOTAL).is_epoch_boundary
    ]
    assert boundaries == [57, 114, 171]


def test_summary_line_never_calls_a_shard_an_epoch():
    line = compute_epoch_position(global_shards_completed=1, shards=_plan(), total_shards_planned=TOTAL).summary_line()
    assert "epochs_completed: 0" in line
    assert "shards_completed_in_current_epoch: 1/57" in line
    assert "global_shards_completed: 1/171" in line


def test_degenerate_single_lap_plan_does_not_divide_by_zero():
    single = tuple(ShardInfo(shard_index=i, lap=0, line_count=10, manifest_path=f"s{i}") for i in range(4))
    pos = compute_epoch_position(global_shards_completed=4, shards=single, total_shards_planned=4)
    assert pos.shards_per_epoch == 4
    assert pos.epochs_completed == 1
