"""Deterministic, stratified image-review sample for the completeness finding (read-only).

    .venv\\Scripts\\python.exe scripts/benchmark_image_review_sample.py svea-hovratt-2026-09 svea-hovratt-2026-09-primary

Question for the reviewer: does the supplied GT visibly correspond to the handwriting, or does it look like raw,
uncorrected recognition output? The sample is chosen by the fixed rules below from the completeness audit's
recognition_provenance.csv (never from model predictions) and written with line crops and page-image links to
benchmark-data/work/<source>/image-review/. Assessments are recorded by hand in a separate file; this script never
writes them, never writes GT and never runs a model.

Strata (per collection unless noted; pages already selected are skipped):
  A1 suspect  first, middle and last page of the T2 save cluster (T2 pages sorted by last save)
  A2 suspect  the T2/T3-no-convention page with the most line-final '¬' (ties: lowest page number)
  A3 suspect  the T2/T3-no-convention page with the highest out-of-vocabulary rate (ties: lowest page number)
  A4 suspect  one seeded page among the T3 pages without student convention
  A5 suspect  every T4 page (mixed line-end marks): lines before and after the first '¬' line end
  B1 control  one seeded page classed "edited" (no T2/T3/T4 signal)
  B2 control  one seeded page, over all collections, among the fast-save pages that do show student edits
Lines: LINES_PER_PAGE seeded lines per page among retained lines with >= MIN_LINE_CHARS non-space characters
(A5: LINES_PER_SIDE on each side of the first '¬').
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import random
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote

from archivetrust.htr.benchmark.contract import read_manifest, sha256_file

SEED = 20260927
LINES_PER_PAGE = 3
LINES_PER_SIDE = 2
MIN_LINE_CHARS = 20
ROOT = Path("benchmark-data")
P = "{http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15}"
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
ASSESSMENTS = ("CORRECTED / HUMAN-LIKE", "LIKELY UNCORRECTED MODEL OUTPUT", "AMBIGUOUS")


def nonspace(text: str) -> int:
    return sum(not c.isspace() for c in text)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source_id")
    ap.add_argument("benchmark_id")
    ap.add_argument("--provenance-csv", type=Path,
                    help="recognition_provenance.csv (default: work/<source>/completeness-audit/)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    source = ROOT / "incoming" / args.source_id
    work = ROOT / "work" / args.source_id
    frozen = ROOT / "benchmark" / args.benchmark_id
    prov_csv = args.provenance_csv or work / "completeness-audit" / "recognition_provenance.csv"
    out = work / "image-review"
    if (out / "sample.csv").exists() and not args.overwrite:
        print(f"{out / 'sample.csv'} exists; pass --overwrite to regenerate", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)

    with prov_csv.open(encoding="utf-8", newline="") as f:
        prov = list(csv.DictReader(f))
    for r in prov:
        r["page_nr"] = int(r["page_nr"])
        r["oov"] = float(r["oov_rate_vs_training_vocabulary"] or 0)
        r["neg"] = int(r["line_final_not_sign"])
    lines_by_page = defaultdict(list)
    for line in read_manifest(frozen / "manifest.jsonl"):
        lines_by_page[line.gt_source_relative_path].append(line)
    for ls in lines_by_page.values():
        ls.sort(key=lambda l: l.line_order)

    rng = random.Random(SEED)
    chosen: dict[str, str] = {}  # page_xml -> stratum

    def take(rows: list[dict], stratum: str) -> None:
        for r in rows:
            if r["page_xml"] not in chosen:
                chosen[r["page_xml"]] = stratum
                return

    def seeded(rows: list[dict]) -> list[dict]:
        rows = sorted(rows, key=lambda r: r["page_xml"])
        rng.shuffle(rows)
        return rows

    by_coll = defaultdict(list)
    for r in sorted(prov, key=lambda r: (r["collection"], r["page_nr"])):
        by_coll[r["collection"]].append(r)
    for coll, rows in sorted(by_coll.items()):
        t2 = sorted((r for r in rows if r["class"].startswith("likely_uncorrected")), key=lambda r: (r["last_change"], r["page_xml"]))
        t3_nc = [r for r in rows if r["class"].startswith("ambiguous_no_student_convention")]
        if t2:
            for label, r in (("A1_cluster_first", t2[0]), ("A1_cluster_middle", t2[len(t2) // 2]), ("A1_cluster_last", t2[-1])):
                take([r], label)
        suspects = t2 + t3_nc
        take(sorted(suspects, key=lambda r: (-r["neg"], r["page_nr"])), "A2_most_not_sign_line_ends")
        take(sorted(suspects, key=lambda r: (-r["oov"], r["page_nr"])), "A3_highest_oov_rate")
        take(seeded(t3_nc), "A4_T3_no_student_convention")
        for r in rows:
            if "mixed" in r["class"]:
                take([r], "A5_T4_mixed_line_ends")
        take(seeded([r for r in rows if r["class"] == "edited"]), "B1_control_edited")
    take(seeded([r for r in prov if r["class"].startswith("ambiguous_machine_speed_save_with_student_edits")]),
         "B2_control_fast_save_with_edits")

    info = {r["page_xml"]: r for r in prov}
    sample = []
    for rel, stratum in sorted(chosen.items(), key=lambda kv: (info[kv[0]]["collection"], info[kv[0]]["page_nr"])):
        r = info[rel]
        ls = lines_by_page[rel]
        eligible = [l for l in ls if nonspace(l.gt_canonical) >= MIN_LINE_CHARS]
        if stratum.startswith("A5"):
            cut = next((i for i, l in enumerate(ls) if l.gt_canonical.rstrip().endswith("¬")), len(ls))
            before = [l for l in eligible if l.line_order < ls[cut].line_order] if cut < len(ls) else eligible
            after = [l for l in eligible if cut < len(ls) and l.line_order >= ls[cut].line_order]
            picked = rng.sample(before, min(LINES_PER_SIDE, len(before))) + rng.sample(after, min(LINES_PER_SIDE, len(after)))
            side = {id(l): ("before_first_not_sign" if l in before else "from_first_not_sign") for l in picked}
        else:
            picked = rng.sample(eligible, min(LINES_PER_PAGE, len(eligible)))
            side = {}
        page_el = ET.parse(source / rel).getroot().find(f"{P}Page")
        page_image = (Path(rel).parent.parent / page_el.get("imageFilename")).as_posix()
        for l in sorted(picked, key=lambda l: l.line_order):
            sample.append({
                "stratum": stratum, "collection": r["collection"], "page": r["page"], "page_nr": r["page_nr"],
                "audit_class": r["class"], "page_signals": f"{r['sec_per_line_since_previous_save']} s/line; "
                f"line-end marks {r['line_final_mark_sequence'] or '-'}; oov {r['oov']}",
                "line_order": l.line_order, "source_line_ref": l.source_line_ref, "line_id": l.line_id,
                "position": side.get(id(l), ""), "gt": l.gt_canonical, "line_crop": l.line_image_path,
                "page_image": page_image})

    header = list(sample[0])
    with (out / "sample.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, header, lineterminator="\n")
        w.writeheader()
        w.writerows(sample)
    rel_root = Path("../../..")
    rows_html = []
    for s in sample:
        crop = (rel_root / "benchmark" / args.benchmark_id / s["line_crop"]).as_posix()
        page = (rel_root / "incoming" / args.source_id / s["page_image"]).as_posix()
        rows_html.append(
            f"<tr><td>{html.escape(s['stratum'])}<br>{s['collection']} p{s['page_nr']:02d} {html.escape(s['page'])}"
            f"<br>{html.escape(s['source_line_ref'])} {html.escape(s['position'])}<br><small>{html.escape(s['page_signals'])}"
            f"</small></td><td><img loading='lazy' src='{quote(crop)}' style='max-width:900px;width:100%'>"
            f"<div style='font-size:18px'>{html.escape(s['gt'])}</div></td>"
            f"<td><a href='{quote(page)}'>page image</a></td></tr>")
    (out / "sample.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Image review sample</title>"
        "<style>body{font:14px sans-serif;margin:16px}td{border-top:1px solid #ccc;vertical-align:top;padding:6px}</style>"
        f"<h1>Image review sample ({len(chosen)} pages, {len(sample)} lines, seed {SEED})</h1>"
        f"<p>Assess each line as one of: {' / '.join(ASSESSMENTS)}. Compare with the image only; never with model "
        "predictions. Do not change GT.</p><table>" + "".join(rows_html) + "</table>\n", encoding="utf-8")
    record = {
        "script": "scripts/benchmark_image_review_sample.py", "script_sha256": sha256_file(Path(__file__)),
        "benchmark_id": args.benchmark_id, "manifest_sha256": sha256_file(frozen / "manifest.jsonl"),
        "provenance_csv": prov_csv.as_posix(), "provenance_csv_sha256": sha256_file(prov_csv),
        "constants": {"SEED": SEED, "LINES_PER_PAGE": LINES_PER_PAGE, "LINES_PER_SIDE": LINES_PER_SIDE,
                      "MIN_LINE_CHARS": MIN_LINE_CHARS},
        "pages": len(chosen), "lines": len(sample),
        "strata": {k: sum(v == k for v in chosen.values()) for k in sorted(set(chosen.values()))},
        "assessment_values": list(ASSESSMENTS),
    }
    (out / "sample_record.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for p in (out / "sample.csv", out / "sample.html", out / "sample_record.json"):
        if EMAIL.search(p.read_text(encoding="utf-8")):
            print(f"e-mail address found in {p}; refusing to keep it", file=sys.stderr)
            p.unlink()
            return 1
    print(json.dumps(record["strata"], indent=1))
    print(f"{len(chosen)} pages, {len(sample)} lines -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
