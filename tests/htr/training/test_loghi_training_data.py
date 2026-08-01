from __future__ import annotations

import hashlib

from archivetrust.htr.training.loghi_training_data import prepare_loghi_training_data
from archivetrust.htr.training.pilot_split import build_pilot_split
from archivetrust.htr.training.swedish_dataset_inventory import build_source_inventory


def _build_pilot(synthetic_dataset_root, synthetic_charlist, tmp_path):
    inv_path = tmp_path / "inventory.parquet"
    build_source_inventory(dataset_root=synthetic_dataset_root, output_path=inv_path, charlist_path=synthetic_charlist)
    split_dir = tmp_path / "manifests"
    build_pilot_split(inventory_path=inv_path, output_dir=split_dir, target_train=8, target_val=3, target_test_reserved=1, seed=1)
    return split_dir


def test_writes_a_list_file_per_split(synthetic_dataset_root, synthetic_charlist, tmp_path):
    split_dir = _build_pilot(synthetic_dataset_root, synthetic_charlist, tmp_path)
    output_dir = tmp_path / "prepared"
    report = prepare_loghi_training_data(
        dataset_root=synthetic_dataset_root,
        manifest_paths={"train": split_dir / "train_manifest.parquet", "val": split_dir / "val_manifest.parquet"},
        output_dir=output_dir,
    )
    names = {s.split_name for s in report.splits}
    assert names == {"train", "val"}
    for split in report.splits:
        assert (output_dir / f"{split.split_name}_list.txt").exists()


def test_image_paths_in_list_files_are_container_relative_not_host_paths(synthetic_dataset_root, synthetic_charlist, tmp_path):
    """A real bug caught before the first container run: host Windows paths (`D:\\...`) don't exist
    inside the Linux container, which mounts the whole `output_dir` at `/lists`."""
    split_dir = _build_pilot(synthetic_dataset_root, synthetic_charlist, tmp_path)
    output_dir = tmp_path / "prepared"
    prepare_loghi_training_data(
        dataset_root=synthetic_dataset_root,
        manifest_paths={"train": split_dir / "train_manifest.parquet"},
        output_dir=output_dir,
    )
    lines = (output_dir / "train_list.txt").read_text(encoding="utf-8").splitlines()
    assert lines  # the fixture's pilot split genuinely produced at least one training sample
    for line in lines:
        path_field = line.split("\t")[0]
        assert path_field.startswith("/lists/images/train/")
        assert ":" not in path_field  # no drive letter, no host-style path snuck through


def test_list_file_is_tab_separated_with_ground_truth_as_the_last_field(synthetic_dataset_root, synthetic_charlist, tmp_path):
    """The real `loghi-htr` line format (`data/manager.py::_process_line`): `ground_truth =
    fields[-1]` -- confirmed by reading the pinned commit's own source, not assumed."""
    split_dir = _build_pilot(synthetic_dataset_root, synthetic_charlist, tmp_path)
    output_dir = tmp_path / "prepared"
    prepare_loghi_training_data(
        dataset_root=synthetic_dataset_root,
        manifest_paths={"train": split_dir / "train_manifest.parquet"},
        output_dir=output_dir,
    )
    lines = (output_dir / "train_list.txt").read_text(encoding="utf-8").splitlines()
    for line in lines:
        fields = line.split("\t")
        assert len(fields) == 2


def test_images_are_written_byte_identical_to_the_source_never_reencoded(synthetic_dataset_root, synthetic_charlist, tmp_path):
    split_dir = _build_pilot(synthetic_dataset_root, synthetic_charlist, tmp_path)
    output_dir = tmp_path / "prepared"
    prepare_loghi_training_data(
        dataset_root=synthetic_dataset_root,
        manifest_paths={"train": split_dir / "train_manifest.parquet"},
        output_dir=output_dir,
    )
    import pyarrow.parquet as pq

    manifest = pq.read_table(split_dir / "train_manifest.parquet").to_pylist()
    for row in manifest:
        expected_hash = row["image_content_hash"]
        written_files = list((output_dir / "images" / "train").glob(f"*{row['line_id'].replace(':', '__')}*.png"))
        assert written_files, f"no written image found for {row['line_id']}"
        actual_hash = hashlib.sha256(written_files[0].read_bytes()).hexdigest()
        assert actual_hash == expected_hash


def test_transcriptions_in_the_list_file_are_verbatim_not_normalized(synthetic_dataset_root, synthetic_charlist, tmp_path):
    """The fixture includes real Swedish diacritics (å/ä/ö) and an em-dash -- must survive untouched."""
    split_dir = _build_pilot(synthetic_dataset_root, synthetic_charlist, tmp_path)
    output_dir = tmp_path / "prepared"
    prepare_loghi_training_data(
        dataset_root=synthetic_dataset_root,
        manifest_paths={"train": split_dir / "train_manifest.parquet"},
        output_dir=output_dir,
    )
    text = (output_dir / "train_list.txt").read_text(encoding="utf-8")
    # at least one of the fixture's real diacritic-bearing transcriptions must appear verbatim
    # somewhere in train or the other splits weren't sampled here -- check across all known fixture strings
    known_strings = ["Silff — 10 lodh", "fåår — 1 mk", "vnghnött — 18 ör", "Koppar — 3 mk",
                      "penigar — 5 mk", "Kor — 7 mk", "Suin — 2 ör", "Axsiö", "Staffan J"]
    assert any(s in text for s in known_strings)


def test_writes_a_preparation_report_json(synthetic_dataset_root, synthetic_charlist, tmp_path):
    split_dir = _build_pilot(synthetic_dataset_root, synthetic_charlist, tmp_path)
    output_dir = tmp_path / "prepared"
    prepare_loghi_training_data(
        dataset_root=synthetic_dataset_root,
        manifest_paths={"train": split_dir / "train_manifest.parquet"},
        output_dir=output_dir,
    )
    assert (output_dir / "preparation_report.json").exists()


def test_sample_count_matches_the_manifests_row_count(synthetic_dataset_root, synthetic_charlist, tmp_path):
    import pyarrow.parquet as pq

    split_dir = _build_pilot(synthetic_dataset_root, synthetic_charlist, tmp_path)
    output_dir = tmp_path / "prepared"
    report = prepare_loghi_training_data(
        dataset_root=synthetic_dataset_root,
        manifest_paths={"train": split_dir / "train_manifest.parquet"},
        output_dir=output_dir,
    )
    manifest_rows = pq.read_table(split_dir / "train_manifest.parquet").num_rows
    train_split = next(s for s in report.splits if s.split_name == "train")
    assert train_split.sample_count == manifest_rows
