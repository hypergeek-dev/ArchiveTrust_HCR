"""Exact diff between two benchmark manifests (read-only), e.g. a superseded freeze and its cleaned successor.

    .venv\\Scripts\\python.exe scripts/benchmark_manifest_diff.py OLD_DIR NEW_DIR OUT_DIR

OLD_DIR / NEW_DIR are frozen benchmark folders or candidate folders (each holding manifest.jsonl and the line crops).
Writes to OUT_DIR: removed_lines.csv, added_lines.csv, changed_lines.csv (retained lines whose manifest record or crop
bytes differ), and diff_summary.json (counts, removed pages/documents, characters and words, input hashes).
Characters are NFC code points of canonical GT including inner spaces; words are whitespace-split, as the scorer does.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

from archivetrust.htr.benchmark.contract import read_manifest, sha256_file


def chars(text: str) -> int:
    return len(unicodedata.normalize("NFC", text))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("old_dir", type=Path)
    ap.add_argument("new_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    args = ap.parse_args()
    old = {l.line_id: l for l in read_manifest(args.old_dir / "manifest.jsonl")}
    new = {l.line_id: l for l in read_manifest(args.new_dir / "manifest.jsonl")}
    removed = sorted(old.keys() - new.keys())
    added = sorted(new.keys() - old.keys())
    changed = []
    for key in sorted(old.keys() & new.keys()):
        a, b = old[key], new[key]
        record_same = json.dumps(a.model_dump(mode="json"), sort_keys=True) == json.dumps(b.model_dump(mode="json"), sort_keys=True)
        crop_same = sha256_file(args.old_dir / a.line_image_path) == sha256_file(args.new_dir / b.line_image_path)
        if not (record_same and crop_same):
            changed.append([key, record_same, crop_same, a.gt_canonical, b.gt_canonical])

    args.out_dir.mkdir(parents=True, exist_ok=True)

    def write(name: str, header: list[str], rows: list[list]) -> None:
        with (args.out_dir / name).open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(header)
            w.writerows(rows)

    line_header = ["line_id", "document_id", "page_id", "source_line_ref", "gt_source_relative_path", "gt_canonical"]

    def line_row(l):
        return [l.line_id, l.document_id, l.page_id, l.source_line_ref, l.gt_source_relative_path, l.gt_canonical]

    write("removed_lines.csv", line_header, [line_row(old[k]) for k in removed])
    write("added_lines.csv", line_header, [line_row(new[k]) for k in added])
    write("changed_lines.csv", ["line_id", "record_identical", "crop_identical", "old_gt", "new_gt"], changed)

    def pages(lines):
        return {(l.document_id, l.page_id) for l in lines}

    def totals(lines):
        lines = list(lines)
        return {"documents": len({l.document_id for l in lines}), "pages": len(pages(lines)), "lines": len(lines),
                "characters": sum(chars(l.gt_canonical) for l in lines),
                "words": sum(len(l.gt_canonical.split()) for l in lines)}

    old_pages, new_pages = pages(old.values()), pages(new.values())
    removed_pages = sorted(old_pages - new_pages)
    partly = sorted(p for p in old_pages & new_pages
                    if sum(1 for l in old.values() if (l.document_id, l.page_id) == p)
                    != sum(1 for l in new.values() if (l.document_id, l.page_id) == p))
    t_old, t_new = totals(old.values()), totals(new.values())
    summary = {
        "old": {"dir": args.old_dir.as_posix(), "manifest_sha256": sha256_file(args.old_dir / "manifest.jsonl"), **t_old},
        "new": {"dir": args.new_dir.as_posix(), "manifest_sha256": sha256_file(args.new_dir / "manifest.jsonl"), **t_new},
        "difference": {k: t_new[k] - t_old[k] for k in t_old},
        "removed_lines": len(removed), "added_lines": len(added), "changed_retained_lines": len(changed),
        "retained_lines_identical": len(old.keys() & new.keys()) - len(changed),
        "removed_characters": sum(chars(old[k].gt_canonical) for k in removed),
        "removed_words": sum(len(old[k].gt_canonical.split()) for k in removed),
        "removed_lines_by_document": dict(sorted(Counter(old[k].document_id for k in removed).items())),
        "removed_pages": [f"{d}/{p}" for d, p in removed_pages],
        "pages_partly_removed": [f"{d}/{p}" for d, p in partly],
        "documents_removed_entirely": sorted({l.document_id for l in old.values()} - {l.document_id for l in new.values()}),
    }
    (args.out_dir / "diff_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("removed_pages",)}, indent=1, ensure_ascii=False))
    return 0 if not changed else 1


if __name__ == "__main__":
    sys.exit(main())
