"""Shared fixtures for `htr/training/` tests: a small, real, synthetic Parquet dataset mirroring the
real `F:\\huggingface_dataset` schema (`image: {bytes, path}`, `transcription: str`), so tests never
depend on the real 87GB dataset being present."""

from __future__ import annotations

import io
import itertools

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image


_next_color = itertools.count(1)
"""Unbounded (not `range(1, 256)`) -- a single grayscale channel's 255 values were exhausted for
real once enough test files across the `htr/training/` suite shared this one session-level counter
(`StopIteration` on the 256th call). Spreading the counter across all 3 RGB channels below gives a
~16M-value space instead."""


def _make_png_bytes(*, width: int = 40, height: int = 10) -> bytes:
    """Every call (with no explicit `color`) produces a genuinely distinct PNG -- a solid-color
    image whose color increments per call, so two "different" fixture rows are never accidentally
    byte-identical (PNG-encoding an identical solid color is fully deterministic, which silently
    made every row in an earlier version of this fixture collide as a false "duplicate")."""
    n = next(_next_color)
    color = (n % 256, (n // 256) % 256, (n // 65536) % 256)
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def synthetic_dataset_root(tmp_path):
    """Two collections: `collection_a` (2 files, 6+4 rows) and `collection_b` (1 file, 5 rows),
    mirroring the real dataset's mix of multi-file and single-file collections. Includes one row with
    a missing image, one with an empty transcription, and one duplicate pair -- real detection targets.
    """
    root = tmp_path / "dataset_root"

    def _write(collection: str, filename: str, rows: list[dict]) -> None:
        d = root / collection
        d.mkdir(parents=True, exist_ok=True)
        table = pa.table(
            {
                "image": pa.array(
                    [{"bytes": r["image_bytes"], "path": None} for r in rows],
                    type=pa.struct([("bytes", pa.binary()), ("path", pa.string())]),
                ),
                "transcription": pa.array([r["transcription"] for r in rows], type=pa.string()),
            }
        )
        pq.write_table(table, d / filename)

    shared_png = _make_png_bytes()
    _write(
        "collection_a",
        "collection_a_1.parquet",
        [
            {"image_bytes": _make_png_bytes(), "transcription": "Silff — 10 lodh"},
            {"image_bytes": _make_png_bytes(), "transcription": "fåår — 1 mk"},
            {"image_bytes": _make_png_bytes(), "transcription": "vnghnött — 18 ör"},
            {"image_bytes": shared_png, "transcription": "duplicate line"},
            {"image_bytes": shared_png, "transcription": "duplicate line"},  # real duplicate pair
            {"image_bytes": b"", "transcription": "missing image"},  # missing image
        ],
    )
    _write(
        "collection_a",
        "collection_a_2.parquet",
        [
            {"image_bytes": _make_png_bytes(), "transcription": "Koppar — 3 mk"},
            {"image_bytes": _make_png_bytes(), "transcription": "penigar — 5 mk"},
            {"image_bytes": _make_png_bytes(), "transcription": ""},  # empty transcription
            {"image_bytes": _make_png_bytes(), "transcription": "Kor — 7 mk"},
        ],
    )
    _write(
        "collection_b",
        "collection_b_1.parquet",
        [
            {"image_bytes": _make_png_bytes(), "transcription": "Suin — 2 ör"},
            {"image_bytes": _make_png_bytes(), "transcription": "Axsiö"},
            {"image_bytes": _make_png_bytes(), "transcription": "Ænglebro"},  # Æ, a real vocabulary edge case
            {"image_bytes": _make_png_bytes(), "transcription": "vnghnött"},
            {"image_bytes": _make_png_bytes(), "transcription": "Staffan J"},
        ],
    )
    return root


@pytest.fixture()
def synthetic_charlist(tmp_path):
    """A charlist covering everything in `synthetic_dataset_root` except `Æ` -- the deliberate
    vocabulary gap `character_inventory.py`'s tests exercise."""
    chars = set(" —abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZåäöÅÄÖ0123456789")
    path = tmp_path / "charlist.txt"
    path.write_text("".join(sorted(chars)), encoding="utf-8")
    return path
