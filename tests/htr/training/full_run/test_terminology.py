"""Guards against reintroducing the shard-called-an-epoch terminology.

One shard is 1/57th of an epoch at full-corpus scale. Reporting shards as epochs overstates progress
by 57x and invites treating a single shard's validation result as epoch-level evidence.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path

import pytest

import archivetrust.htr.training.full_run.cli as cli
from archivetrust.htr.training.full_run.run_state import (
    create_initial_run_state,
    heartbeat,
    mark_running,
    save_run_state,
)

FULL_RUN_PKG = Path(cli.__file__).resolve().parent


def test_status_output_reports_shards_and_epochs_distinctly(tmp_path, capsys):
    """`status` must never present a shard count as an epoch count."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from archivetrust.htr.training.full_run.corpus_sharding import CorpusShardingSummary, ShardInfo

    run_dir = tmp_path / "run"
    shards_dir = run_dir / "shards"
    shards_dir.mkdir(parents=True)
    infos = []
    for i in range(6):
        p = shards_dir / f"s{i}.parquet"
        pq.write_table(pa.table({"line_id": pa.array([f"l{i}"], type=pa.string())}), p)
        infos.append(ShardInfo(shard_index=i, lap=i // 3, line_count=1, manifest_path=str(p)))
    (shards_dir / "sharding_summary.json").write_text(
        CorpusShardingSummary(
            seed=1, shard_line_count=1, total_valid_lines=3, excluded_line_count=0, usable_line_count=3,
            shards=tuple(infos), excluded_manifest_paths=(), line_id_set_hash="h" * 64,
        ).model_dump_json(indent=2), encoding="utf-8")

    state = heartbeat(
        mark_running(create_initial_run_state(run_id="r1", configuration_hash="h1")),
        current_epoch=2, global_shards_completed=2, epochs_completed=0,
        shards_completed_in_current_epoch=2, shards_per_epoch=3, epoch_progress=2 / 3,
    )
    save_run_state(run_dir / "run-state", state)

    assert cli.main(["status", "--run", str(run_dir)]) == 0
    out = capsys.readouterr().out
    assert "shards_completed_in_current_epoch: 2 / 3" in out
    assert "epochs_completed: 0" in out
    assert "global_shards_completed: 2 / 6" in out
    assert "epoch_progress: 0.6667" in out


def test_run_state_exposes_honest_epoch_fields():
    state = create_initial_run_state(run_id="r", configuration_hash="h")
    for field in ("global_shards_completed", "epochs_completed",
                  "shards_completed_in_current_epoch", "shards_per_epoch", "epoch_progress"):
        assert hasattr(state, field), f"run_state must expose {field}"


def test_legacy_current_epoch_field_is_documented_as_a_shard_counter():
    """The field is kept for backward-compatible reads of existing run_state.json, but its docstring
    must warn that it counts shards -- otherwise the next reader repeats the same mistake."""
    import archivetrust.htr.training.full_run.run_state as rs

    source = Path(rs.__file__).read_text(encoding="utf-8")
    idx = source.index("current_epoch: int = 0")
    following = source[idx: idx + 600]
    assert "counts SHARDS" in following
    assert "57" in following


def test_prepare_output_does_not_call_shard_executions_epochs():
    """`Configured maximum epochs (shards)` was self-contradictory -- it labelled a shard ceiling as
    an epoch ceiling."""
    source = Path(cli.__file__).read_text(encoding="utf-8")
    assert "maximum epochs (shards)" not in source
    assert "maximum SHARD executions" in source


def test_epoch_accounting_module_documents_the_57_to_1_relationship():
    import archivetrust.htr.training.full_run.epoch_accounting as ea

    doc = (ea.__doc__ or "")
    assert "57" in doc
    assert "shard" in doc.lower() and "epoch" in doc.lower()


def test_gui_status_panel_has_no_shard_labelled_as_epoch():
    gui = (FULL_RUN_PKG / "gui" / "app.py").read_text(encoding="utf-8")
    assert '"Current epoch (shard):"' not in gui, "self-contradictory label must be gone"
    assert "epoch_position" in gui
