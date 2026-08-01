from __future__ import annotations

from archivetrust.htr.training.swedish_dataset_inventory import build_source_inventory, read_inventory


def test_inventory_covers_every_row_across_every_collection_and_file(synthetic_dataset_root, synthetic_charlist, tmp_path):
    summary = build_source_inventory(
        dataset_root=synthetic_dataset_root,
        output_path=tmp_path / "inventory.parquet",
        charlist_path=synthetic_charlist,
    )
    assert summary["total_rows"] == 15  # 6 + 4 + 5


def test_detects_missing_image_and_empty_transcription(synthetic_dataset_root, synthetic_charlist, tmp_path):
    summary = build_source_inventory(
        dataset_root=synthetic_dataset_root,
        output_path=tmp_path / "inventory.parquet",
        charlist_path=synthetic_charlist,
    )
    assert summary["exclusion_counts"]["missing_image"] == 1
    assert summary["exclusion_counts"]["empty_transcription"] == 1


def test_detects_duplicate_image_and_pair(synthetic_dataset_root, synthetic_charlist, tmp_path):
    summary = build_source_inventory(
        dataset_root=synthetic_dataset_root,
        output_path=tmp_path / "inventory.parquet",
        charlist_path=synthetic_charlist,
    )
    assert summary["exclusion_counts"]["duplicate_image"] == 1
    assert summary["exclusion_counts"]["duplicate_image_transcription_pair"] == 1
    table = read_inventory(tmp_path / "inventory.parquet")
    rows = {r["transcription"]: r for r in table.to_pylist() if r["transcription"] == "duplicate line"}
    duplicates = [r for r in table.to_pylist() if r["transcription"] == "duplicate line"]
    assert len(duplicates) == 2
    statuses = sorted(r["duplicate_status"] for r in duplicates)
    assert statuses == ["duplicate", "unique"]


def test_detects_unsupported_character(synthetic_dataset_root, synthetic_charlist, tmp_path):
    """`Ænglebro` contains U+00C6, deliberately absent from the fixture charlist."""
    summary = build_source_inventory(
        dataset_root=synthetic_dataset_root,
        output_path=tmp_path / "inventory.parquet",
        charlist_path=synthetic_charlist,
    )
    assert summary["exclusion_counts"]["unsupported_characters"] == 1
    table = read_inventory(tmp_path / "inventory.parquet")
    row = next(r for r in table.to_pylist() if r["transcription"] == "Ænglebro")
    assert row["unsupported_characters"] == ["Æ"]
    assert row["valid"] is False


def test_transcription_preserved_verbatim_never_modernized(synthetic_dataset_root, synthetic_charlist, tmp_path):
    build_source_inventory(
        dataset_root=synthetic_dataset_root,
        output_path=tmp_path / "inventory.parquet",
        charlist_path=synthetic_charlist,
    )
    table = read_inventory(tmp_path / "inventory.parquet")
    texts = {r["transcription"] for r in table.to_pylist()}
    assert "vnghnött — 18 ör" in texts  # historical spelling untouched, em-dash preserved verbatim


def test_valid_rows_are_exactly_total_minus_excluded(synthetic_dataset_root, synthetic_charlist, tmp_path):
    summary = build_source_inventory(
        dataset_root=synthetic_dataset_root,
        output_path=tmp_path / "inventory.parquet",
        charlist_path=synthetic_charlist,
    )
    assert summary["valid_rows"] == summary["total_rows"] - summary["excluded_rows"]
    assert summary["excluded_rows"] > 0


def test_line_id_is_reproducible_and_reflects_real_source_location(synthetic_dataset_root, synthetic_charlist, tmp_path):
    build_source_inventory(
        dataset_root=synthetic_dataset_root,
        output_path=tmp_path / "inventory.parquet",
        charlist_path=synthetic_charlist,
    )
    table = read_inventory(tmp_path / "inventory.parquet")
    line_ids = table.column("line_id").to_pylist()
    assert "collection_a:collection_a_1.parquet:0" in line_ids
    assert len(line_ids) == len(set(line_ids))  # every line_id unique


def test_archive_volume_writer_identifiers_are_honestly_none(synthetic_dataset_root, synthetic_charlist, tmp_path):
    build_source_inventory(
        dataset_root=synthetic_dataset_root,
        output_path=tmp_path / "inventory.parquet",
        charlist_path=synthetic_charlist,
    )
    table = read_inventory(tmp_path / "inventory.parquet", columns=["archive_or_fonds_id", "volume_or_document_id", "writer_or_hand_id"])
    for row in table.to_pylist():
        assert row["archive_or_fonds_id"] is None
        assert row["volume_or_document_id"] is None
        assert row["writer_or_hand_id"] is None


def test_read_inventory_valid_only_filters_correctly(synthetic_dataset_root, synthetic_charlist, tmp_path):
    summary = build_source_inventory(
        dataset_root=synthetic_dataset_root,
        output_path=tmp_path / "inventory.parquet",
        charlist_path=synthetic_charlist,
    )
    valid_table = read_inventory(tmp_path / "inventory.parquet", valid_only=True)
    assert valid_table.num_rows == summary["valid_rows"]


def test_same_image_different_transcription_is_a_suspicious_mismatch_not_a_plain_duplicate(
    synthetic_dataset_root, synthetic_charlist, tmp_path
):
    """Regression test for a real bug found while building this module: the pair-duplicate check was
    unreachable dead code, and "same image, different transcription" (a real data-quality signal
    distinct from a harmless duplicate) was silently folded into `duplicate_image` with no separate
    flag at all."""
    import io

    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image

    root = synthetic_dataset_root / "collection_c"
    root.mkdir()
    img = Image.new("RGB", (30, 10), color=(50, 60, 70))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    same_bytes = buf.getvalue()

    table = pa.table(
        {
            "image": pa.array(
                [{"bytes": same_bytes, "path": None}, {"bytes": same_bytes, "path": None}],
                type=pa.struct([("bytes", pa.binary()), ("path", pa.string())]),
            ),
            "transcription": pa.array(["first reading", "conflicting reading"], type=pa.string()),
        }
    )
    pq.write_table(table, root / "collection_c_1.parquet")

    from archivetrust.htr.training.swedish_dataset_inventory import build_source_inventory, read_inventory

    summary = build_source_inventory(
        dataset_root=synthetic_dataset_root,
        output_path=tmp_path / "inventory.parquet",
        charlist_path=synthetic_charlist,
    )
    assert summary["exclusion_counts"].get("suspicious_transcription_mismatch") == 1
    # the same-image-different-transcription row must NOT also be counted as a plain
    # duplicate_image_transcription_pair, since the pair (image_hash, transcription) is genuinely new
    table = read_inventory(tmp_path / "inventory.parquet")
    row = next(r for r in table.to_pylist() if r["transcription"] == "conflicting reading")
    assert "suspicious_transcription_mismatch" in row["exclusion_reason"]
    assert "duplicate_image_transcription_pair" not in row["exclusion_reason"]


def test_max_rows_bounds_the_build(synthetic_dataset_root, synthetic_charlist, tmp_path):
    summary = build_source_inventory(
        dataset_root=synthetic_dataset_root,
        output_path=tmp_path / "inventory.parquet",
        charlist_path=synthetic_charlist,
        max_rows=3,
    )
    assert summary["total_rows"] == 3
