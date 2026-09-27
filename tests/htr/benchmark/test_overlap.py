import pytest

from archivetrust.htr.benchmark.build import build_candidate, freeze
from archivetrust.htr.benchmark.overlap import build_training_index, check_overlap
from tests.htr.benchmark._fixtures import line_png, write

LONG = "Kongl. Maj:ts nådiga resolution uppå"


def test_overlap_detects_shared_images_and_long_texts_only(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    src = tmp_path / "incoming" / "s"
    write(src, "d/a.png", line_png("abc", seed=1))
    write(src, "d/a.gt.txt", "Anno 1723")  # short: never text-matched
    write(src, "d/b.png", line_png("abcdef", seed=2))
    write(src, "d/b.gt.txt", LONG)
    write(src, "d/c.png", line_png("xyz", seed=3))
    write(src, "d/c.gt.txt", "helt ny rad som inte finns i träningsdata")
    build_candidate(src, tmp_path / "cand", dataset_id="ds", source_id="s")
    freeze(tmp_path / "cand", tmp_path / "bench", benchmark_id="b1")

    shard = tmp_path / "hf" / "collection_x" / "train-0.parquet"
    shard.parent.mkdir(parents=True)
    pq.write_table(pa.table({
        "image": [{"bytes": line_png("abc", seed=1), "path": "p"}, {"bytes": line_png("qq", seed=9), "path": "q"}],
        "transcription": ["Anno 1723", LONG.upper()],
    }), shard)
    meta = build_training_index([shard], tmp_path / "idx.sqlite")
    assert meta["counts"]["rows"] == 2
    with pytest.raises(FileExistsError):
        build_training_index([shard], tmp_path / "idx.sqlite")

    result = check_overlap(tmp_path / "bench", tmp_path / "idx.sqlite", tmp_path / "out")
    assert result["signal_counts"]["identical_image"] == 1
    assert result["signal_counts"]["identical_normalized_text"] == 1
    assert "identical_text" not in result["signal_counts"]
    assert result["lines_with_strong_signal"] == 2 and "not proof" in result["caveat"]
