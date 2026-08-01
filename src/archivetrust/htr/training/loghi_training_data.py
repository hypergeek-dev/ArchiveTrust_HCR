"""Converts the pilot manifests into the real, pinned `loghi-htr` training-list input format.

**The real format, read from the pinned `loghi-htr` commit's own `data/manager.py::_process_line` /
`_get_ground_truth`** (`.loghi-upstream/loghi-htr`, commit recorded in `providers/loghi/
pinned_versions.py`) -- not assumed from upstream documentation, which does not specify it precisely:

```
<image_path>\\t<ground_truth_text>
```

One line per sample in a plain UTF-8 text file; `ground_truth = fields[-1]` (the *last* tab-separated
field), so a 2-column `path\\ttext` line is valid without an explicit sample-weight column. `config[
"train_list"]`/`config["validation_list"]` are themselves space-separated *paths to these text files*
(`generic-2023-02-15`'s own shipped config lists 17 such files for `train_list`) -- this module writes
one such list file per pilot split (`train_list.txt`, `val_list.txt`).

**`<image_path>` is container-mount-relative (`/lists/images/<split>/<file>.png`), never a host path.**
A real bug caught before the first container run: `container_epoch_runner.py` mounts this module's
whole `output_dir` at `/lists` inside the container, so every path written into a list file must
already be relative to that mount point -- a Windows host path (`D:\\ArchiveTrust_HCR\\...`) does not
exist inside the Linux container at all. The two are coupled by this fixed convention, documented here
and in that module, not by a shared constant (the mount point is Docker CLI syntax, not Python).

**Images are written byte-identical to the source Parquet, never re-encoded.** Every sampled row's
`image.format` was confirmed `PNG` at inventory time; `tf.io.decode_image` (what `loghi-htr`'s own
`DataLoader` calls) reads PNG directly, so no transcoding step exists between the source bytes and what
Loghi actually reads -- one fewer place for a silent, unrecorded transform to happen. Every written
image's filename embeds its own `image_content_hash`, so a prepared file's provenance is verifiable by
re-hashing it, not just by trusting a manifest row.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict

LOGHI_TRAINING_DATA_ADAPTER_VERSION = "1.0.0"


class PreparedSplit(BaseModel):
    model_config = ConfigDict(frozen=True)

    split_name: str
    sample_count: int
    list_file_path: str
    images_dir: str


class TrainingDataPreparationReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    adapter_version: str
    source_inventory_path: str
    splits: tuple[PreparedSplit, ...]
    excluded_count: int
    """Rows present in the manifest but not written (e.g. a source image byte-decode failure at
    prep time genuinely different from the inventory-time check) -- 0 in the expected case, never
    silently dropped without being counted here."""


def _line_id_to_filename(line_id: str) -> str:
    """A filesystem-safe filename derived from the real `line_id` -- reversible enough to trace back
    (the manifest still carries the authoritative `line_id` -> path mapping), never a bare integer
    counter that would lose the connection to its source row."""
    safe = line_id.replace(":", "__").replace("/", "_").replace("\\", "_")
    return safe


def prepare_loghi_training_data(
    *,
    dataset_root: str | Path,
    manifest_paths: dict[str, str | Path],
    output_dir: str | Path,
) -> TrainingDataPreparationReport:
    """`manifest_paths`: `{"train": train_manifest.parquet, "val": val_manifest.parquet, ...}`.
    Writes `output_dir/images/<split>/<line_id>__<image_content_hash short>.png` and
    `output_dir/<split>_list.txt` (the real `loghi-htr` two-column TSV format) for each split.
    """
    output_dir = Path(output_dir)
    images_root = output_dir / "images"

    splits: list[PreparedSplit] = []
    excluded_count = 0

    for split_name, manifest_path in manifest_paths.items():
        manifest_table = pq.read_table(
            manifest_path, columns=["line_id", "collection", "source_parquet_file", "row_index", "image_content_hash"]
        )
        wanted = {
            (row["collection"], row["source_parquet_file"], row["row_index"]): row
            for row in manifest_table.to_pylist()
        }

        split_images_dir = images_root / split_name
        split_images_dir.mkdir(parents=True, exist_ok=True)
        list_file_path = output_dir / f"{split_name}_list.txt"

        written = 0
        with list_file_path.open("w", encoding="utf-8") as list_file:
            root = Path(dataset_root)
            for collection_dir in sorted(p for p in root.iterdir() if p.is_dir()):
                if not any(key[0] == collection_dir.name for key in wanted):
                    continue
                for parquet_path in sorted(collection_dir.glob("*.parquet")):
                    parquet_file = pq.ParquetFile(parquet_path)
                    row_index = 0
                    for batch in parquet_file.iter_batches(batch_size=512, columns=["image", "transcription"]):
                        for row in batch.to_pylist():
                            key = (collection_dir.name, parquet_path.name, row_index)
                            row_index += 1
                            if key not in wanted:
                                continue
                            manifest_row = wanted[key]
                            image_bytes = (row.get("image") or {}).get("bytes")
                            transcription = row.get("transcription")
                            if not image_bytes or transcription is None:
                                excluded_count += 1
                                continue

                            filename = _line_id_to_filename(manifest_row["line_id"])
                            image_path = split_images_dir / f"{filename}.png"
                            image_path.write_bytes(image_bytes)

                            # Container-mount-relative path, NOT the host path -- a real bug caught
                            # before the first real container run: `container_epoch_runner.py` mounts
                            # this whole `output_dir` (the `prepared-data` root) at `/lists` inside the
                            # container, so a path the container can actually open must be relative to
                            # that mount point, never a Windows host path like
                            # `D:\ArchiveTrust_HCR\...\images\train\x.png`, which does not exist inside
                            # the Linux container at all. `loghi-htr`'s own line format
                            # (data/manager.py::_process_line): tab-separated, ground truth is the LAST
                            # field, so "path\ttext" is valid without an explicit weight column.
                            container_path = f"/lists/images/{split_name}/{filename}.png"
                            list_file.write(f"{container_path}\t{transcription}\n")
                            written += 1

        splits.append(
            PreparedSplit(
                split_name=split_name,
                sample_count=written,
                list_file_path=str(list_file_path),
                images_dir=str(split_images_dir),
            )
        )

    report = TrainingDataPreparationReport(
        adapter_version=LOGHI_TRAINING_DATA_ADAPTER_VERSION,
        source_inventory_path=str(dataset_root),
        splits=tuple(splits),
        excluded_count=excluded_count,
    )
    (output_dir / "preparation_report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return report
