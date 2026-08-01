from __future__ import annotations

from archivetrust.htr.training.full_run.launch_manifest import (
    PREPARED_MARKER_NAME,
    build_launch_manifest,
    load_launch_manifest,
    write_launch_manifest,
    write_prepared_marker,
)


def _manifest(**overrides):
    base = dict(
        run_id="r1", original_loghi_base_checkpoint_path="/base", base_checkpoint_hash="bh",
        known_pilot_run_dirs=(), training_manifest_dir="/shards", training_manifest_hash="th",
        validation_manifest_path="/val.parquet", validation_manifest_hash="vh", dataset_hash="dh",
        train_line_count=100, validation_line_count=10, excluded_test_line_count=5,
        code_commit_hash="c1", container_image_name="img:latest", container_image_digest=None,
        batch_size=16, optimizer="adam", scheduler="constant", learning_rate=0.0001, random_seed=42,
        max_epochs=100, max_epochs_basis="test", early_stopping_patience=5, early_stopping_basis="test",
        wall_clock_policy="test", shard_count=10, checkpoint_policy="test", dashboard_config="test",
        exact_launch_command="python -m x start --confirm-full-corpus-run",
    )
    base.update(overrides)
    return build_launch_manifest(**base)


def test_pilot_checkpoint_used_as_parent_is_always_false():
    manifest = _manifest()
    assert manifest.pilot_checkpoint_used_as_parent is False


def test_rejection_evidence_names_the_checked_directory():
    manifest = _manifest(original_loghi_base_checkpoint_path="/base/original", known_pilot_run_dirs=("/pilot1", "/pilot2"))
    assert "/base/original" in manifest.pilot_checkpoint_rejection_evidence
    assert "/pilot1" in manifest.pilot_checkpoint_rejection_evidence


def test_telemetry_schema_version_is_recorded():
    manifest = _manifest()
    assert manifest.telemetry_schema_version


def test_preparation_timestamp_is_a_real_utc_timestamp():
    manifest = _manifest()
    assert manifest.preparation_timestamp.endswith("Z")


def test_write_and_load_round_trips(tmp_path):
    manifest = _manifest()
    output_path = tmp_path / "launch_manifest.json"
    write_launch_manifest(manifest, output_path)
    assert output_path.exists()
    loaded = load_launch_manifest(output_path)
    assert loaded == manifest


def test_write_is_atomic_no_partial_file_left_on_success(tmp_path):
    manifest = _manifest()
    output_path = tmp_path / "nested" / "launch_manifest.json"
    write_launch_manifest(manifest, output_path)
    leftover_tmp_files = list(output_path.parent.glob(".tmp-launch-manifest-*"))
    assert leftover_tmp_files == []


def test_write_prepared_marker_creates_a_readable_file(tmp_path):
    run_dir = tmp_path / "run"
    marker_path = write_prepared_marker(run_dir)
    assert marker_path.name == PREPARED_MARKER_NAME
    assert marker_path.exists()
    text = marker_path.read_text(encoding="utf-8")
    assert "PREPARED" in text
    assert "NOT" in text
