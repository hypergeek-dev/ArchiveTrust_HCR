from __future__ import annotations

from pathlib import Path

import pytest

from archivetrust.htr.training.full_run.corpus_sharding import build_full_corpus_shards
from archivetrust.htr.training.full_run.shard_training_data import (
    build_or_reuse_shard_training_data,
    pool_directory,
)
from archivetrust.htr.training.swedish_dataset_inventory import build_source_inventory
from tests.htr.training.full_run.conftest import write_pilot_manifests_from_inventory


@pytest.fixture()
def corpus(tmp_path, synthetic_dataset_root, synthetic_charlist, pilot_fixture_dir):
    """A real (tiny) end-to-end corpus: inventory -> pilot val/test manifests -> shard plan whose
    shard count forces more than one lap, so lap behaviour is genuinely exercised."""
    inventory_path = tmp_path / "inventory.parquet"
    build_source_inventory(
        dataset_root=synthetic_dataset_root, output_path=inventory_path, charlist_path=synthetic_charlist
    )
    manifests = write_pilot_manifests_from_inventory(pilot_fixture_dir, inventory_path, val_count=2, test_count=1)
    shards_dir = tmp_path / "run" / "shards"
    summary = build_full_corpus_shards(
        inventory_path=inventory_path,
        output_dir=shards_dir,
        exclude_manifest_paths=(manifests["val_manifest"], manifests["test_reserved_manifest"]),
        shard_line_count=2,
        seed=1,
        target_shard_count=6,
    )
    return {
        "training_root": tmp_path / "training",
        "dataset_root": synthetic_dataset_root,
        "summary": summary,
        "val_manifest": manifests["val_manifest"],
    }


def _build(corpus, **overrides):
    kwargs = dict(
        training_root=corpus["training_root"],
        dataset_root=corpus["dataset_root"],
        sharding_summary=corpus["summary"],
        validation_manifest_path=corpus["val_manifest"],
        dataset_hash="d" * 64,
        training_manifest_hash=corpus["summary"].line_id_set_hash,
    )
    kwargs.update(overrides)
    return build_or_reuse_shard_training_data(**kwargs)


def test_produces_one_list_file_per_shard(corpus):
    paths = _build(corpus)
    assert len(paths.shard_list_paths) == len(corpus["summary"].shards)
    for p in paths.shard_list_paths:
        assert Path(p).exists()


def test_list_files_are_utf8_text_in_the_container_format(corpus):
    """The exact defect that failed Gate 4: the trainer needs `<path>\\t<text>`, not Parquet."""
    paths = _build(corpus)
    first = Path(paths.shard_list_paths[0])
    assert not first.suffix == ".parquet"
    content = first.read_text(encoding="utf-8")  # must decode as UTF-8, unlike a Parquet file
    assert content.strip(), "list file is empty"
    for line in content.strip().split("\n"):
        assert "\t" in line, f"line is not tab-separated: {line!r}"
        image_path, ground_truth = line.split("\t", 1)
        assert image_path.startswith("/lists/images/"), "path must be container-relative, not a host path"
        assert image_path.endswith(".png")
        assert ground_truth != ""


def test_referenced_images_actually_exist_on_disk(corpus):
    """A list file pointing at images that were never extracted would fail inside the container
    rather than here -- so resolve every container path back to the host and check."""
    paths = _build(corpus)
    pool = Path(paths.pool_dir)
    for shard_list in paths.shard_list_paths:
        for line in Path(shard_list).read_text(encoding="utf-8").strip().split("\n"):
            container_path = line.split("\t", 1)[0]
            host_path = pool / container_path[len("/lists/"):]
            assert host_path.exists(), f"{container_path} referenced but not extracted"
            assert host_path.stat().st_size > 0


def test_images_are_extracted_once_not_once_per_lap(corpus):
    """The lap trap: 3 laps over the same lines would triplicate extraction (~250GB instead of
    ~82GB at real scale). Images must be written once and shared by every lap's shards."""
    summary = corpus["summary"]
    laps = {s.lap for s in summary.shards}
    assert len(laps) > 1, "fixture must span multiple laps for this test to mean anything"

    paths = _build(corpus)
    extracted = list((Path(paths.pool_dir) / "images" / "train").glob("*.png"))
    assert len(extracted) == summary.usable_line_count
    # Total referenced lines across all shards far exceeds the number of files on disk.
    total_referenced = sum(
        len(Path(p).read_text(encoding="utf-8").strip().split("\n")) for p in paths.shard_list_paths
    )
    assert total_referenced > len(extracted)


def test_later_laps_reference_the_same_image_files(corpus):
    paths = _build(corpus)
    by_lap: dict[int, set[str]] = {}
    for shard, list_path in zip(corpus["summary"].shards, paths.shard_list_paths):
        refs = {
            line.split("\t", 1)[0]
            for line in Path(list_path).read_text(encoding="utf-8").strip().split("\n")
        }
        by_lap.setdefault(shard.lap, set()).update(refs)
    laps = sorted(by_lap)
    # A later lap draws from the same universe of extracted images, but may be truncated mid-lap by
    # `target_shard_count` -- so the invariant is containment, not equality. What matters is that it
    # reuses existing files rather than introducing newly-extracted ones.
    assert by_lap[laps[1]] <= by_lap[laps[0]], "a later lap must reuse images already extracted for lap 0"
    assert by_lap[laps[1]], "the later lap referenced nothing at all"


def test_validation_list_is_written_and_separate_from_training(corpus):
    paths = _build(corpus)
    val = Path(paths.validation_list_path)
    assert val.exists() and val.name == "val_list.txt"
    val_refs = {line.split("\t", 1)[0] for line in val.read_text(encoding="utf-8").strip().split("\n")}
    train_refs = set()
    for p in paths.shard_list_paths:
        train_refs |= {line.split("\t", 1)[0] for line in Path(p).read_text(encoding="utf-8").strip().split("\n")}
    assert val_refs, "validation list is empty"
    assert not (val_refs & train_refs), "validation images leaked into the training lists"


def test_pool_is_reused_when_hashes_match(corpus):
    first = _build(corpus)
    assert first.reused_existing_pool is False
    second = _build(corpus)
    assert second.reused_existing_pool is True
    assert second.pool_dir == first.pool_dir


def test_a_changed_shard_plan_hash_builds_a_distinct_pool(corpus):
    """Reuse must be keyed to what the pool actually contains -- a different plan must never be
    silently served from an existing pool."""
    first = _build(corpus)
    second = _build(corpus, training_manifest_hash="f" * 64)
    assert second.pool_dir != first.pool_dir
    assert second.reused_existing_pool is False


def test_a_changed_dataset_hash_builds_a_distinct_pool(corpus):
    first = _build(corpus)
    second = _build(corpus, dataset_hash="e" * 64)
    assert second.pool_dir != first.pool_dir


def test_pool_directory_is_deterministic_for_the_same_inputs(corpus):
    a = pool_directory(training_root=corpus["training_root"], dataset_hash="a" * 64, training_manifest_hash="b" * 64)
    b = pool_directory(training_root=corpus["training_root"], dataset_hash="a" * 64, training_manifest_hash="b" * 64)
    assert a == b


def test_force_rebuild_ignores_an_existing_pool(corpus):
    _build(corpus)
    rebuilt = _build(corpus, force_rebuild=True)
    assert rebuilt.reused_existing_pool is False


def test_reported_counts_match_what_was_written(corpus):
    paths = _build(corpus)
    on_disk = len(list((Path(paths.pool_dir) / "images" / "train").glob("*.png")))
    assert paths.train_image_count == on_disk
    val_on_disk = len(list((Path(paths.pool_dir) / "images" / "val").glob("*.png")))
    assert paths.validation_image_count == val_on_disk
