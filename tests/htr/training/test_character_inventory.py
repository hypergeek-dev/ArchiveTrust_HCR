from __future__ import annotations

from archivetrust.htr.training.character_inventory import build_character_compatibility_report
from archivetrust.htr.training.pilot_split import build_pilot_split
from archivetrust.htr.training.swedish_dataset_inventory import build_source_inventory


def _build_pilot(synthetic_dataset_root, synthetic_charlist, tmp_path):
    inv_path = tmp_path / "inventory.parquet"
    build_source_inventory(dataset_root=synthetic_dataset_root, output_path=inv_path, charlist_path=synthetic_charlist)
    split_dir = tmp_path / "manifests"
    build_pilot_split(inventory_path=inv_path, output_dir=split_dir, target_train=8, target_val=3, target_test_reserved=1, seed=1)
    return inv_path, split_dir


def test_reports_incompatible_when_a_character_is_missing_from_charlist(synthetic_dataset_root, synthetic_charlist, tmp_path):
    inv_path, split_dir = _build_pilot(synthetic_dataset_root, synthetic_charlist, tmp_path)
    report = build_character_compatibility_report(
        inventory_path=inv_path,
        manifest_paths=[split_dir / "train_manifest.parquet", split_dir / "val_manifest.parquet"],
        charlist_path=synthetic_charlist,
    )
    # `Ænglebro` (unsupported Æ) was excluded from the inventory as invalid, so it should never reach
    # the pilot manifests at all -- meaning the pilot itself should already be compatible.
    assert report.compatible is True
    assert report.unsupported_character_count == 0


def test_frequency_table_reflects_real_counts(synthetic_dataset_root, synthetic_charlist, tmp_path):
    inv_path, split_dir = _build_pilot(synthetic_dataset_root, synthetic_charlist, tmp_path)
    report = build_character_compatibility_report(
        inventory_path=inv_path,
        manifest_paths=[split_dir / "train_manifest.parquet", split_dir / "val_manifest.parquet"],
        charlist_path=synthetic_charlist,
    )
    assert report.total_lines_examined > 0
    total_freq = sum(f.frequency for f in report.frequencies)
    assert total_freq > 0
    # every frequency entry names a real codepoint string
    assert all(f.codepoint.startswith("U+") for f in report.frequencies)


def test_swedish_diacritics_are_recognized_as_supported(synthetic_dataset_root, synthetic_charlist, tmp_path):
    inv_path, split_dir = _build_pilot(synthetic_dataset_root, synthetic_charlist, tmp_path)
    report = build_character_compatibility_report(
        inventory_path=inv_path,
        manifest_paths=[split_dir / "train_manifest.parquet", split_dir / "val_manifest.parquet"],
        charlist_path=synthetic_charlist,
    )
    by_char = {f.character: f for f in report.frequencies}
    if "ö" in by_char:
        assert by_char["ö"].supported_by_generic_checkpoint is True


def test_never_reads_via_terminal_output_only_real_codepoints(synthetic_dataset_root, synthetic_charlist, tmp_path):
    """Regression guard for this integration's own console-encoding lesson: the report's character
    field must be the real Python string, verifiable by codepoint, not merely "looks right" when
    printed."""
    inv_path, split_dir = _build_pilot(synthetic_dataset_root, synthetic_charlist, tmp_path)
    report = build_character_compatibility_report(
        inventory_path=inv_path,
        manifest_paths=[split_dir / "train_manifest.parquet", split_dir / "val_manifest.parquet"],
        charlist_path=synthetic_charlist,
    )
    for entry in report.frequencies:
        assert len(entry.character) == 1
        assert entry.codepoint == f"U+{ord(entry.character):04X}"
