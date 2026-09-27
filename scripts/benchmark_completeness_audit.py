"""Deterministic transcription-completeness audit of a benchmark (read-only).

    .venv\\Scripts\\python.exe scripts/benchmark_completeness_audit.py svea-hovratt-2026-09 svea-hovratt-2026-09-primary

Question: does the retained benchmark behave like an incomplete transcription, or like a complete one whose
Transkribus status (IN_PROGRESS) was never updated? Reads the frozen manifest, the decisions' exclusion list, every
PAGE-XML and ALTO file of the retained pages and every page image of the delivery. Writes CSV/JSON tables, a
findings list, a stratified (fixed-seed) manual spot-check queue and an HTML view to
benchmark-data/work/<source>/completeness-audit/. Never writes GT; never runs a model.

Labels: DETERMINISTIC (exact count of a stated rule), HEURISTIC (a stated rule approximating a concept; every
threshold is a constant below and is copied into run_record.json), MANUAL SPOT-CHECK ONLY (queued, not concluded).
Personal metadata (user ids, e-mail addresses in Creator strings) is never written; outputs are scanned for e-mail
addresses and the run fails if one is found.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import random
import re
import statistics
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from archivetrust.htr.benchmark.contract import read_manifest, sha256_file

# ---- explicit rules and thresholds ------------------------------------------------------------
SHORT_MAX_CHARS = 3  # a transcribed line with <= this many non-space chars is "short"
MIN_TAIL_LINES = 2  # a trailing run of >= this many lines without GT (reading order) is an "unfinished tail"
ROBUST_Z = 4.0  # |robust z| of log(chars per crop-height unit of width) >= this is a crop/text outlier
WIDE_ASPECT = 6.0  # crop width/height >= this ...
WIDE_MAX_CHARS = 2  # ... with <= this many GT chars is "wide crop, tiny text"
NARROW_REGION_WIDTH_FRAC = 0.25  # a region narrower than this fraction of the page width is "narrow" (marginal/heading)
REGION_BOTTOM_GAP_LINE_HEIGHTS = 3.0  # unsegmented space below the last line of a region >= this x median line height
BATCH_MAX_SEC_PER_LINE = 1.0  # a page's last save <= this many seconds per line after the previous save = machine speed
ALTO_CHAR_DIFF_FRAC = 0.02  # PAGE vs ALTO non-space character totals differing by more than this fraction = major
SEED = 20260927  # spot-check sampling seed
NORMAL_PAGES_PER_COLLECTION = 2
MAX_OUTLIER_LINES_IN_QUEUE = 40
OOV_MIN_LETTERS = 3  # T7: word tokens of >= this many letters are checked against the training vocabulary
OOV_TOKEN = re.compile(rf"[^\W\d_]{{{OOV_MIN_LETTERS},}}")
HUMAN_MARKER = re.compile(r"\[|\?{2,}")  # editorial marks a recognition model does not emit (D4b material)
PLACEHOLDER_RULES = {  # rule -> (label, regex); counted on retained and protocol-excluded source text
    "P01_repeated_question_marks": ("DETERMINISTIC", re.compile(r"\?{2,}")),
    "P02_repeated_dots": ("DETERMINISTIC", re.compile(r"\.{2,}")),
    "P03_ellipsis_char": ("DETERMINISTIC", re.compile("…")),
    "P04_underscore": ("DETERMINISTIC", re.compile(r"_")),
    "P05_repeated_hyphens_or_dashes": ("DETERMINISTIC", re.compile(r"[-‐‑–—¬]{2,}")),
    "P06_todo_like": ("DETERMINISTIC", re.compile(r"(?i)\b(todo|fixme|tbd)\b")),
    "P07_xxx": ("DETERMINISTIC", re.compile(r"(?i)x{3,}")),
    "P08_angle_brackets": ("DETERMINISTIC", re.compile(r"<[^>]*>|[<>]")),
    "P09_square_brackets": ("DETERMINISTIC", re.compile(r"\[[^\]]*\]|[\[\]]")),
    "P10_empty_brackets": ("DETERMINISTIC", re.compile(r"\[\s*\]|\(\s*\)|\{\s*\}")),
    "P11_xml_escape_debris": ("DETERMINISTIC", re.compile(r"&(amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);")),
    "P12_replacement_char": ("DETERMINISTIC", re.compile("�")),
    "P13_object_replacement_char": ("DETERMINISTIC", re.compile("￼")),
    "P14_single_question_mark": ("HEURISTIC", re.compile(r"(?<!\?)\?(?!\?)")),  # normal punctuation in most cases
    "P15_double_space": ("HEURISTIC", re.compile(r"\S {2,}\S")),
    "P16_dangling_open_bracket_at_end": ("DETERMINISTIC", re.compile(r"[\[(<{]\s*$")),
}
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
# -------------------------------------------------------------------------------------------------

ROOT = Path("benchmark-data")
P = "{http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15}"
A = "{http://www.loc.gov/standards/alto/ns-v4#}"
RO_INDEX = re.compile(r"readingOrder\s*\{\s*index:(\d+)")


def job_of(path: str) -> str:
    match = re.search(r"export_job_(\d+)", path)
    return match.group(1) if match else "(unknown)"


def bbox(points: str) -> tuple[int, int, int, int]:
    xy = [tuple(int(float(v)) for v in pt.split(",")) for pt in points.split()]
    xs, ys = [p[0] for p in xy], [p[1] for p in xy]
    return min(xs), min(ys), max(xs), max(ys)


def ro(el) -> int:
    match = RO_INDEX.search(el.get("custom", ""))
    return int(match.group(1)) if match else 10**6


def nonspace(text: str) -> int:
    return sum(not c.isspace() for c in text)


def pct(n, d):
    return round(100 * n / d, 3) if d else None


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def tool_names(creator: str) -> list[str]:
    """Software names from a Transkribus Creator string; personal parts (names, e-mails) are dropped."""
    return sorted({m.split("::")[0].strip() for m in re.findall(r"name=([^:\n]+(?:::[^\n]*)?)", creator)})


def parse_page(path: Path) -> dict:
    root = ET.parse(path).getroot()
    meta = root.find(f"{P}Metadata")
    tm = meta.find(f"{P}TranskribusMetadata")
    page = root.find(f"{P}Page")
    order = {r.get("regionRef"): int(r.get("index")) for r in page.iter(f"{P}RegionRefIndexed")}
    regions = []
    for r_i, region in enumerate(page.iter(f"{P}TextRegion")):
        coords = region.find(f"{P}Coords")
        lines = []
        for tl in region.findall(f"{P}TextLine"):
            u = tl.find(f"{P}TextEquiv/{P}Unicode")
            tl_coords = tl.find(f"{P}Coords")
            lines.append({"id": tl.get("id"), "ro": ro(tl), "text": (u.text or "") if u is not None else None,
                          "text_equiv_count": len(tl.findall(f"{P}TextEquiv")),
                          "bbox": bbox(tl_coords.get("points")) if tl_coords is not None else None})
        lines.sort(key=lambda l: l["ro"])
        ru = region.find(f"{P}TextEquiv/{P}Unicode")
        regions.append({"id": region.get("id"), "order": order.get(region.get("id"), 10**6 + r_i),
                        "bbox": bbox(coords.get("points")) if coords is not None else None,
                        "type": region.get("type") or "", "lines": lines,
                        "region_text": (ru.text or "") if ru is not None else None})
    regions.sort(key=lambda r: r["order"])
    return {
        "width": int(page.get("imageWidth")), "height": int(page.get("imageHeight")),
        "image_filename": page.get("imageFilename"),
        "created": meta.findtext(f"{P}Created"), "last_change": meta.findtext(f"{P}LastChange"),
        "status": tm.get("status") if tm is not None else None,
        "page_nr": int(tm.get("pageNr")) if tm is not None and tm.get("pageNr") else None,
        "tsid": int(tm.get("tsid")) if tm is not None and tm.get("tsid") else None,
        "transkribus_page_id": tm.get("pageId") if tm is not None else None,
        "transkribus_image_id": tm.get("imageId") if tm is not None else None,
        "user_present": bool(tm is not None and tm.get("userId")),
        "tools": tool_names(meta.findtext(f"{P}Creator") or ""),
        "regions": regions,
    }


def parse_alto(path: Path) -> dict:
    root = ET.parse(path).getroot()
    blocks = list(root.iter(f"{A}TextBlock"))
    lines = {}
    for tl in root.iter(f"{A}TextLine"):
        lines[tl.get("ID")] = " ".join(s.get("CONTENT", "") for s in tl.findall(f"{A}String"))
    return {"blocks": len(blocks), "lines": lines}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source_id")
    ap.add_argument("benchmark_id")
    ap.add_argument("--training-parquet-dir", type=Path, help="vocabulary source for the T7 out-of-vocabulary signal")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    source = ROOT / "incoming" / args.source_id
    work = ROOT / "work" / args.source_id
    frozen = ROOT / "benchmark" / args.benchmark_id
    out = work / "completeness-audit"
    if out.exists() and not args.overwrite:
        print(f"{out} exists; pass --overwrite to regenerate", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)

    manifest = read_manifest(frozen / "manifest.jsonl")
    retained = {(l.gt_source_relative_path, l.source_line_ref): l for l in manifest}
    excluded = {json.loads(x)["target"]: json.loads(x)["reason"]
                for x in (frozen / "excluded_by_decision.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()}
    page_files = sorted({l.gt_source_relative_path for l in manifest})
    vocab: set[str] = set()
    training_files = []
    if args.training_parquet_dir:
        import pyarrow.parquet as pq  # noqa: PLC0415

        for f in sorted(args.training_parquet_dir.glob("*.parquet")):
            texts = pq.read_table(f, columns=["transcription"]).column(0).to_pylist()
            vocab.update(w.lower() for t in texts for w in OOV_TOKEN.findall(t or ""))
            training_files.append({"name": f.name, "bytes": f.stat().st_size, "rows": len(texts)})

    def oov_rate(text: str) -> float | None:
        words = [w.lower() for w in OOV_TOKEN.findall(text)]
        return round(sum(w not in vocab for w in words) / len(words), 4) if vocab and words else None
    jobs = sorted({job_of(p) for p in page_files})
    findings: list[dict] = []
    queue: list[list] = []

    def finding(fid, label, rule, count, denom, collections, pages, examples, output, limits):
        findings.append({"id": fid, "label": label, "rule": rule, "count": count, "denominator": denom,
                         "percent": pct(count, denom) if isinstance(count, int) and denom else None,
                         "collections": collections, "pages_affected": pages, "examples": examples[:8],
                         "output": output, "interpretation_limits": limits})

    # ---- parse every retained page ----------------------------------------------------------------
    pages: dict[str, dict] = {}
    line_rows: list[dict] = []
    for rel in page_files:
        info = parse_page(source / rel)
        info["rel"], info["job"], info["stem"] = rel, job_of(rel), Path(rel).stem
        doc_dir = Path(rel).parent.parent.as_posix()
        seq = 0
        for region in info["regions"]:
            for line in region["lines"]:
                key = (rel, line["id"])
                text = line["text"]
                if key in retained:
                    status = "retained"
                elif f"{doc_dir}/{info['stem']}/{line['id']}" in excluded:
                    reason = excluded[f"{doc_dir}/{info['stem']}/{line['id']}"]
                    status = "missing_gt" if text is None or not text.strip() else f"protocol_excluded:{reason.split(':')[0].split(' (')[0]}"
                else:
                    status = "unaccounted"
                line_rows.append({"page": rel, "job": info["job"], "stem": info["stem"], "region": region["id"],
                                  "line": line["id"], "seq": seq, "text": text or "", "status": status,
                                  "has_gt": bool(text and text.strip()), "bbox": line["bbox"],
                                  "manifest": retained.get(key)})
                seq += 1
        pages[rel] = info
    unaccounted = [r for r in line_rows if r["status"] == "unaccounted"]
    missing_from_xml = len(retained) - sum(r["status"] == "retained" for r in line_rows)
    finding("S0", "DETERMINISTIC", "every source TextLine of the retained pages is either retained, excluded by a "
            "recorded decision, or unaccounted; every manifest line is found in the PAGE-XML",
            len(unaccounted) + missing_from_xml, len(line_rows), jobs, len({r["page"] for r in unaccounted}),
            [f"{r['stem']}/{r['line']}" for r in unaccounted], "page_completeness.csv",
            "a non-zero count would mean the audit cannot map source lines to the benchmark")
    by_page = defaultdict(list)
    for r in line_rows:
        by_page[r["page"]].append(r)

    # ---- 1-3. page completeness, unfinished tails, internal gaps --------------------------------------
    page_rows, tail_rows, gap_rows = [], [], []
    for rel in page_files:
        info, rows = pages[rel], by_page[rel]
        with_gt = [r for r in rows if r["has_gt"]]
        chars = [nonspace(r["text"]) for r in with_gt]
        flags = "".join("G" if r["has_gt"] else "E" for r in rows)
        tail = len(flags) - len(flags.rstrip("E"))
        head_done = len(flags.rstrip("E"))
        if tail >= MIN_TAIL_LINES:
            tail_rows.append([info["job"], rel, info["stem"], len(rows), head_done, tail, pct(head_done, len(rows)),
                              rows[head_done]["line"]])
        for m in re.finditer(r"(?<=G)E+(?=G)", flags):
            gap = rows[m.start():m.end()]
            gap_rows.append([info["job"], rel, info["stem"], m.start(), m.end() - m.start(),
                             ";".join(g["line"] for g in gap), ";".join(sorted({g["region"] for g in gap}))])
        retained_rows = [r for r in rows if r["status"] == "retained"]
        page_rows.append([
            info["job"], rel, info["stem"], info["page_nr"], len(info["regions"]), len(rows), len(with_gt),
            len(rows) - len(with_gt), pct(len(with_gt), len(rows)), len(retained_rows),
            sum(r["status"].startswith("protocol_excluded") for r in rows), sum(r["status"] == "missing_gt" for r in rows),
            sum(len(r["text"].strip()) for r in retained_rows), sum(len(r["text"].split()) for r in retained_rows),
            statistics.median(chars) if chars else None, round(statistics.mean(chars), 2) if chars else None,
            sum(c <= SHORT_MAX_CHARS for c in chars), pct(sum(c <= SHORT_MAX_CHARS for c in chars), len(chars)),
            tail, flags.count("E") - tail,
        ])
    write_csv(out / "page_completeness.csv", [
        "collection", "page_xml", "page", "page_nr", "text_regions", "source_lines_geometric", "source_lines_with_gt",
        "source_lines_empty_gt", "pct_source_lines_with_gt", "retained_lines", "protocol_excluded_lines",
        "missing_gt_lines", "retained_chars", "retained_words", "median_nonspace_chars_per_gt_line",
        "mean_nonspace_chars_per_gt_line", f"short_lines_le{SHORT_MAX_CHARS}", "pct_short_lines",
        "unfinished_tail_lines", "internal_empty_lines"], page_rows)
    write_csv(out / "unfinished_tails.csv", ["collection", "page_xml", "page", "source_lines", "lines_before_cutoff",
              "tail_empty_lines", "pct_completed_before_cutoff", "first_empty_line_id"], tail_rows)
    write_csv(out / "internal_gaps.csv", ["collection", "page_xml", "page", "gap_start_seq", "gap_length", "line_ids",
              "regions"], gap_rows)
    n_src = len(line_rows)
    n_gt = sum(r["has_gt"] for r in line_rows)
    empty = [r for r in line_rows if not r["has_gt"]]
    finding("C1", "DETERMINISTIC", "source TextLines (geometry) on retained pages carrying non-empty GT", n_gt, n_src, jobs,
            len(page_files), [f"empty: {r['stem']}/{r['line']} ({r['job']})" for r in empty], "page_completeness.csv",
            "Every page was run through a recognition model (Creator: Jaemtlands_domsagas_M1), which fills every "
            "line. Line coverage therefore cannot distinguish corrected from uncorrected text (see T1-T4).")
    full = sum(1 for r in page_rows if r[7] == 0)
    finding("C2", "DETERMINISTIC", "retained pages where every source line carries GT", full, len(page_rows), jobs,
            len(page_rows) - full, [f"{r[2]} ({r[0]}): {r[7]} empty" for r in page_rows if r[7]], "page_completeness.csv",
            "same limit as C1")
    finding("C3", "DETERMINISTIC", f"pages whose reading-order sequence ends in >= {MIN_TAIL_LINES} lines without GT",
            len(tail_rows), len(page_rows), sorted({r[0] for r in tail_rows}), len(tail_rows),
            [f"{r[2]}: {r[5]} empty after {r[4]}" for r in tail_rows], "unfinished_tails.csv", "same limit as C1")
    finding("C4", "DETERMINISTIC", "runs of lines without GT between lines with GT (internal gaps)", len(gap_rows),
            len(page_rows), sorted({r[0] for r in gap_rows}), len({r[1] for r in gap_rows}),
            [f"{r[2]}: len {r[4]} {r[5]}" for r in gap_rows], "internal_gaps.csv", "same limit as C1")
    short_total = sum(r[16] for r in page_rows)
    finding("C5", "HEURISTIC", f"GT lines with <= {SHORT_MAX_CHARS} non-space chars", short_total, n_gt, jobs,
            sum(1 for r in page_rows if r[16]), [], "page_completeness.csv",
            "short lines are usually genuine (numbers, catchwords, 'd.', single words); they are listed, not concluded")

    # ---- 4. region coverage ------------------------------------------------------------------------
    region_rows = []
    for rel in page_files:
        info = pages[rel]
        for region in info["regions"]:
            lines = region["lines"]
            with_gt = [l for l in lines if l["text"] and l["text"].strip()]
            x0, y0, x1, y1 = region["bbox"] or (0, 0, 0, 0)
            heights = [l["bbox"][3] - l["bbox"][1] for l in lines if l["bbox"]]
            med_h = statistics.median(heights) if heights else None
            last_bottom = max((l["bbox"][3] for l in lines if l["bbox"]), default=None)
            bottom_gap = (y1 - last_bottom) if last_bottom is not None else None
            joined = "\n".join((l["text"] or "") for l in lines)
            region_rows.append([
                info["job"], rel, info["stem"], region["id"], region["order"], region["type"] or "(none)",
                len(lines), len(with_gt), pct(len(with_gt), len(lines)),
                round((x1 - x0) / info["width"], 3), round(((x0 + x1) / 2) / info["width"], 3),
                (x1 - x0) / info["width"] < NARROW_REGION_WIDTH_FRAC,
                bottom_gap, med_h,
                bool(med_h and bottom_gap is not None and bottom_gap >= REGION_BOTTOM_GAP_LINE_HEIGHTS * med_h),
                None if region["region_text"] is None else region["region_text"].replace("\r\n", "\n") == joined,
            ])
    write_csv(out / "region_coverage.csv", [
        "collection", "page_xml", "page", "region_id", "reading_order", "region_type", "lines", "lines_with_gt",
        "pct_lines_with_gt", "width_frac_of_page", "x_center_frac", "narrow_region", "unsegmented_px_below_last_line",
        "median_line_height_px", "large_unsegmented_space_below", "region_textequiv_equals_joined_lines"], region_rows)
    empty_regions = [r for r in region_rows if r[6] == 0 or r[7] == 0]
    finding("R1", "DETERMINISTIC", "text regions with no lines or no line carrying GT", len(empty_regions),
            len(region_rows), sorted({r[0] for r in empty_regions}), len({r[1] for r in empty_regions}),
            [f"{r[2]}/{r[3]} lines={r[6]}" for r in empty_regions], "region_coverage.csv",
            "a region without lines may be a layout-analysis artefact, not an untranscribed text block")
    typed = Counter(r[5] for r in region_rows)
    finding("R2", "DETERMINISTIC", "region type labels (PAGE @type or structure tag)", len(region_rows), len(region_rows),
            jobs, len(page_files), [f"{k}: {v}" for k, v in typed.items()], "region_coverage.csv",
            "without type labels, marginalia/headings can only be approximated geometrically (R3)")
    narrow = [r for r in region_rows if r[11]]
    finding("R3", "HEURISTIC", f"regions narrower than {NARROW_REGION_WIDTH_FRAC} of the page width (possible "
            "marginalia/headings) and their GT coverage", len(narrow), len(region_rows), sorted({r[0] for r in narrow}),
            len({r[1] for r in narrow}),
            [f"{r[2]}/{r[3]}: {r[7]}/{r[6]} lines with GT" for r in narrow[:8]], "region_coverage.csv",
            "narrow regions carry GT at the same rate as main regions if coverage is ~100%; no omitted region type")
    gapped = [r for r in region_rows if r[14]]
    finding("R4", "HEURISTIC", f"regions with unsegmented space below their last line >= {REGION_BOTTOM_GAP_LINE_HEIGHTS} "
            "x median line height (text that may have no line geometry at all)", len(gapped), len(region_rows),
            sorted({r[0] for r in gapped}), len({r[1] for r in gapped}),
            [f"{r[2]}/{r[3]}: {r[12]} px (line h {r[13]})" for r in gapped], "region_coverage.csv + manual queue",
            "region polygons are often drawn generously; the page image decides")
    for r in gapped:
        queue.append(["R4_unsegmented_space_below_region", r[0], r[2], "", "", f"{r[12]} px below last line", r[1]])
    mismatch = [r for r in region_rows if r[15] is False]
    finding("T5", "DETERMINISTIC", "region-level TextEquiv differs from the region's line texts joined by newlines",
            len(mismatch), sum(r[15] is not None for r in region_rows), sorted({r[0] for r in mismatch}),
            len({r[1] for r in mismatch}), [f"{r[2]}/{r[3]}" for r in mismatch], "region_coverage.csv",
            "Transkribus normally keeps both in sync; a mismatch shows an alternative text version, not which is newer")

    # ---- 5. crop vs text outliers ------------------------------------------------------------------
    crop_rows = []
    by_job_feat = defaultdict(list)
    feats = []
    for r in line_rows:
        m = r["manifest"]
        if m is None:
            continue
        w, h = m.image_width, m.image_height
        chars = nonspace(m.gt_canonical)
        val = chars / (w / h) if w and h else None
        feats.append((r, m, w, h, chars, val))
        if val:
            by_job_feat[r["job"]].append(math.log(val))

    stats = {}
    for j, vals in by_job_feat.items():
        med = statistics.median(vals)
        mad = statistics.median(abs(v - med) for v in vals) * 1.4826
        stats[j] = (med, mad)
    for r, m, w, h, chars, val in feats:
        med, mad = stats[r["job"]]
        z = (math.log(val) - med) / mad if val and mad else None
        wide_tiny = w / h >= WIDE_ASPECT and chars <= WIDE_MAX_CHARS
        if (z is not None and abs(z) >= ROBUST_Z) or wide_tiny:
            crop_rows.append([r["job"], m.line_id, r["stem"], w, h, round(w / h, 2), chars, len(m.gt_canonical.split()),
                              round(val, 3), None if z is None else round(z, 2),
                              "wide_crop_tiny_text" if wide_tiny else ("text_long_for_crop" if z > 0 else "text_short_for_crop"),
                              m.gt_canonical, m.line_image_path])
    crop_rows.sort(key=lambda x: (-abs(x[9] or 99), x[1]))
    write_csv(out / "crop_text_outliers.csv", ["collection", "line_id", "page", "crop_w", "crop_h", "aspect", "gt_nonspace_chars",
              "gt_words", "chars_per_height_unit", "robust_z_log", "kind", "gt", "line_image_path"], crop_rows)
    per_job_stats = {j: {"median_chars_per_height_unit": round(math.exp(s[0]), 3), "mad_log": round(s[1], 4)}
                     for j, s in sorted(stats.items())}
    finding("O1", "HEURISTIC", f"retained lines with |robust z| >= {ROBUST_Z} of log(non-space chars / (crop w/h)) within "
            f"their collection, or crop aspect >= {WIDE_ASPECT} with <= {WIDE_MAX_CHARS} chars", len(crop_rows), len(feats),
            sorted({r[0] for r in crop_rows}), len({r[2] for r in crop_rows}),
            [f"{r[1].split('/')[-2]}/{r[1].split('/')[-1]} {r[10]} z={r[9]} '{r[11][:30]}'" for r in crop_rows],
            "crop_text_outliers.csv", f"per-collection medians: {per_job_stats}. Short catchwords and numbers on "
            "wide polygons are normal; long text on short crops can be a baseline/polygon issue, not a GT issue.")
    for r in crop_rows[:MAX_OUTLIER_LINES_IN_QUEUE]:
        queue.append(["O1_crop_text_outlier", r[0], r[2], r[1], r[11], f"{r[10]} z={r[9]} aspect={r[5]}", r[12]])

    # ---- 6-7. placeholder inventory and truncation signals --------------------------------------------
    ph_rows, ph_counts = [], defaultdict(Counter)
    for r in line_rows:
        scope = "retained" if r["status"] == "retained" else r["status"]
        for rule, (label, rx) in PLACEHOLDER_RULES.items():
            for m in rx.finditer(r["text"]):
                ph_rows.append([rule, label, scope, r["job"], r["stem"], r["line"], m.group(0), r["text"]])
                ph_counts[(rule, scope)][r["job"]] += 1
    ph_rows.sort(key=lambda x: (x[0], x[2], x[3], x[4], x[5], x[6]))
    write_csv(out / "placeholder_inventory.csv", ["rule", "label", "scope", "collection", "page", "line_id", "match", "text"],
              ph_rows)
    retained_ph_lines = {(x[4], x[5]) for x in ph_rows if x[2] == "retained" and x[1] == "DETERMINISTIC"}
    for rule, (label, _) in PLACEHOLDER_RULES.items():
        c = ph_counts.get((rule, "retained"), Counter())
        ex = [f"{x[4]}/{x[5]}: {x[6]!r} in '{x[7][:40]}'" for x in ph_rows if x[0] == rule and x[2] == "retained"]
        finding(f"PH_{rule}", label, f"{rule} in retained lines", sum(c.values()), len(feats), sorted(c),
                len({(x[4]) for x in ph_rows if x[0] == rule and x[2] == "retained"}), ex, "placeholder_inventory.csv",
                "a match is a candidate marker, not proof of an unfinished line; context is in the CSV")
    te_multi = sum(1 for p in pages.values() for rg in p["regions"] for l in rg["lines"] if l["text_equiv_count"] > 1)
    finding("T6", "DETERMINISTIC", "TextLines with more than one TextEquiv (alternative text versions)", te_multi, n_src,
            [], 0, [], "page_completeness.csv", "only the exported transcript version is present in the delivery")

    # ---- 8. PAGE vs ALTO -------------------------------------------------------------------------------
    alto_rows = []
    line_text_diff = []
    for rel in page_files:
        info, rows = pages[rel], by_page[rel]
        alto_path = source / Path(rel).parent.parent / "alto" / Path(rel).name
        if not alto_path.is_file():
            alto_rows.append([info["job"], rel, info["stem"], False] + [None] * 11)
            continue
        alto = parse_alto(alto_path)
        page_text = {r["line"]: r["text"] for r in rows}
        same_ids = set(page_text) == set(alto["lines"])
        diffs = [lid for lid in page_text if lid in alto["lines"]
                 and " ".join(page_text[lid].split()) != " ".join(alto["lines"][lid].split())]
        line_text_diff += [(info["job"], info["stem"], lid, page_text[lid], alto["lines"][lid]) for lid in diffs]
        pc = sum(nonspace(t) for t in page_text.values())
        ac = sum(nonspace(t) for t in alto["lines"].values())
        major = (not same_ids) or (pc and abs(pc - ac) / pc > ALTO_CHAR_DIFF_FRAC)
        alto_rows.append([info["job"], rel, info["stem"], True, len(info["regions"]), alto["blocks"], len(rows),
                          len(alto["lines"]), sum(bool(t.strip()) for t in page_text.values()),
                          sum(bool(t.strip()) for t in alto["lines"].values()), pc, ac,
                          sum(len(t.split()) for t in page_text.values()), sum(len(t.split()) for t in alto["lines"].values()),
                          f"ids_equal={same_ids};line_text_diffs={len(diffs)};major={bool(major)}"])
    write_csv(out / "page_alto_comparison.csv", ["collection", "page_xml", "page", "alto_present", "page_regions",
              "alto_blocks", "page_lines", "alto_lines", "page_lines_with_text", "alto_lines_with_text",
              "page_nonspace_chars", "alto_nonspace_chars", "page_words", "alto_words", "verdict"], alto_rows)
    alto_major = [r for r in alto_rows if not r[3] or "major=True" in r[14]]
    finding("X1", "DETERMINISTIC", f"pages where PAGE and ALTO differ in line ids or by > {ALTO_CHAR_DIFF_FRAC:.0%} non-space "
            "characters (or ALTO missing)", len(alto_major), len(alto_rows), sorted({r[0] for r in alto_major}),
            len(alto_major), [r[2] for r in alto_major], "page_alto_comparison.csv",
            "ALTO was exported from the same Transkribus transcript (2023); agreement shows export consistency, "
            "not transcription correctness")
    finding("X2", "DETERMINISTIC", "lines whose whitespace-collapsed text differs between PAGE and ALTO", len(line_text_diff),
            n_src, sorted({d[0] for d in line_text_diff}), len({d[1] for d in line_text_diff}),
            [f"{d[1]}/{d[2]}: '{d[3][:25]}' vs '{d[4][:25]}'" for d in line_text_diff], "page_alto_comparison.csv",
            "same limit as X1")
    for d in line_text_diff:
        queue.append(["X2_page_alto_line_text_differs", d[0], d[1], d[2], d[3], f"ALTO: {d[4]}", ""])

    # ---- 9. versions / duplicates across the whole delivery ------------------------------------------
    version_rows = []
    all_xml = sorted(p for p in source.rglob("page/*.xml") if "__MACOSX" not in p.parts)
    img_hash, tk_page, tk_image = defaultdict(list), defaultdict(list), defaultdict(list)
    for x in all_xml:
        rel = x.relative_to(source).as_posix()
        root = ET.parse(x).getroot()
        tm = root.find(f"{P}Metadata/{P}TranskribusMetadata")
        pg = root.find(f"{P}Page")
        img = x.parent.parent / pg.get("imageFilename")
        if img.is_file():
            img_hash[sha256_file(img)].append(rel)
        if tm is not None:
            tk_page[tm.get("pageId")].append(rel)
            tk_image[tm.get("imageId")].append(rel)
    retained_set = set(page_files)
    for kind, groups in (("identical_page_image_sha256", img_hash), ("same_transkribus_pageId", tk_page),
                         ("same_transkribus_imageId", tk_image)):
        for key, rels in sorted(groups.items()):
            if len(rels) > 1:
                texts = {r: "\n".join((u.text or "") for u in ET.parse(source / r).getroot().iter(f"{P}Unicode")) for r in rels}
                version_rows.append([kind, key, len(rels), sum(r in retained_set for r in rels),
                                     len(set(texts.values())), " | ".join(sorted(job_of(r) for r in rels)), " | ".join(sorted(rels))])
    names = Counter((job_of(p), Path(p).name) for p in page_files)
    for (j, name), n in sorted(names.items()):
        if n > 1:
            version_rows.append(["same_filename_within_collection", f"{j}/{name}", n, n, None, j, ""])
    write_csv(out / "version_conflicts.csv", ["kind", "key", "files", "retained_files", "distinct_texts", "collections",
              "paths"], version_rows)
    finding("V1", "DETERMINISTIC", "duplicate page images / Transkribus pageIds / imageIds across all 14 export jobs, and "
            "duplicate filenames within a retained collection", len(version_rows), len(all_xml),
            sorted({c for r in version_rows for c in r[5].split(" | ")}), sum(r[3] for r in version_rows),
            [f"{r[0]} x{r[2]} retained={r[3]} distinct_texts={r[4]} [{r[5]}]" for r in version_rows],
            "version_conflicts.csv", "only one transcript version per page is exported; older versions are not visible")

    # ---- T. recognition provenance and edit timing (the IN_PROGRESS question) ------------------------
    htr_rows = []
    by_doc = defaultdict(list)
    for rel in page_files:
        by_doc[Path(rel).parent.parent.as_posix()].append(rel)
    for doc, rels in sorted(by_doc.items()):
        # the previous save is taken over EVERY PAGE-XML of the source document, retained or not, so that the
        # signal of a page does not change when other pages of its document are excluded from a benchmark
        saves = sorted((ET.parse(p).getroot().find(f"{P}Metadata").findtext(f"{P}LastChange"), p.name)
                       for p in (source / doc / "page").glob("*.xml"))
        for rel in sorted(rels):
            info, rows = pages[rel], by_page[rel]
            me = (info["last_change"], Path(rel).name)
            earlier = [s for s in saves if s < me]
            last = datetime.fromisoformat(info["last_change"])
            gap = (last - datetime.fromisoformat(earlier[-1][0])).total_seconds() if earlier else None
            texts = [r["text"] for r in rows]
            markers = sum(bool(HUMAN_MARKER.search(t)) for t in texts)
            hyph = sum(t.rstrip().endswith("-") for t in texts)
            neg = sum(t.rstrip().endswith("¬") for t in texts)
            order_marks = [("-" if t.rstrip().endswith("-") else "¬") for t in texts if t.rstrip().endswith(("-", "¬"))]
            switch = bool(order_marks) and "-" in order_marks and "¬" in order_marks
            spl = round(gap / len(rows), 3) if gap is not None and rows else None
            no_student = hyph == 0 and markers == 0
            machine_speed = spl is not None and spl <= BATCH_MAX_SEC_PER_LINE
            cls = ("likely_uncorrected_recognition_output" if no_student and machine_speed else
                   "ambiguous_no_student_convention" if no_student else
                   "ambiguous_machine_speed_save_with_student_edits" if machine_speed else "edited")
            if switch:
                cls += "+mixed_line_end_convention"
            htr_rows.append([info["job"], rel, info["stem"], info["page_nr"], info["status"], info["created"], info["last_change"],
                             info["tsid"], None if gap is None else round(gap, 1), len(rows), spl, markers, hyph, neg,
                             "".join(order_marks), " | ".join(info["tools"]), info["user_present"], cls,
                             oov_rate(" ".join(r["text"] for r in rows if r["status"] == "retained"))])
    htr_rows.sort(key=lambda r: (r[0], r[3] or 0, r[1]))
    write_csv(out / "recognition_provenance.csv", [
        "collection", "page_xml", "page", "page_nr", "transkribus_status", "created", "last_change", "tsid",
        "sec_since_previous_save_in_document", "source_lines", "sec_per_line_since_previous_save",
        "lines_with_human_editorial_marks", "line_final_hyphen", "line_final_not_sign", "line_final_mark_sequence",
        "recognition_tools", "user_id_present_redacted", "class", "oov_rate_vs_training_vocabulary"], htr_rows)
    tools = Counter(r[15] for r in htr_rows)
    finding("T1", "DETERMINISTIC", "PAGE-XML Creator lists an HTR model and layout-analysis tools (names only)",
            len(htr_rows), len(htr_rows), jobs, len(htr_rows), [f"{k} x{v}" for k, v in tools.items()],
            "recognition_provenance.csv", "shows the pages were machine-recognised at some point, not whether the "
            "text was corrected afterwards")
    unc = [r for r in htr_rows if r[17].startswith("likely_uncorrected")]
    unc_lines = sum(1 for r in line_rows if r["status"] == "retained" and r["page"] in {x[1] for x in unc})
    finding("T2", "HEURISTIC", "pages whose last save followed the previous save in the same document within "
            f"<= {BATCH_MAX_SEC_PER_LINE} s per line (machine speed) AND that carry no student convention (0 line-final "
            "'-' and 0 lines with [ or ??): likely recognition output that was never corrected",
            len(unc), len(htr_rows), sorted({r[0] for r in unc}), len(unc),
            [f"{r[0]} p{r[3]:02d} {r[2]}: {r[10]} s/line, ¬{r[13]} -{r[12]}" for r in unc], "recognition_provenance.csv",
            f"{unc_lines} retained lines are on these pages. The timing shows only the LAST save; the absence of '-' "
            "and editorial marks is indirect. The page images must confirm (manual queue).")
    amb = [r for r in htr_rows if r[17].startswith("ambiguous")]
    finding("T3", "MANUAL SPOT-CHECK ONLY", "pages with exactly one of the two T2 signals", len(amb), len(htr_rows),
            sorted({r[0] for r in amb}), len(amb), [f"{r[0]} p{r[3]:02d} {r[17]}" for r in amb],
            "recognition_provenance.csv + manual queue", "a fast final save after correction is possible; a page "
            "can also genuinely contain no hyphenated line-ends")
    mixed = [r for r in htr_rows if "mixed" in r[17]]
    finding("T4", "HEURISTIC", "pages mixing line-final '-' and '¬' (correction possibly stopped partway down the page)",
            len(mixed), len(htr_rows), sorted({r[0] for r in mixed}), len(mixed),
            [f"{r[0]} p{r[3]:02d}: {r[14]}" for r in mixed], "recognition_provenance.csv",
            "a student may also have used both marks deliberately")
    for r in unc + amb + [m for m in mixed if m not in unc + amb]:
        queue.append([f"T_{r[17]}", r[0], r[2], "", "", f"{r[10]} s/line; ¬{r[13]} -{r[12]} marks{r[11]}", r[1]])
    htr_by_page = {r[1]: r[17] for r in htr_rows}
    if vocab:
        groups = defaultdict(list)
        for r in htr_rows:
            groups[(r[0], r[17].split("+")[0])].append(r[18])
        finding("T7", "HEURISTIC", f"share of word tokens (>= {OOV_MIN_LETTERS} letters, lowercased) on a page that never "
                f"occur in the training-reference vocabulary ({len(vocab)} types); median per collection and T2 class",
                None, None, jobs, len(htr_rows),
                [f"{j} {c}: median {statistics.median(v):.3f} (n={len(v)})" for (j, c), v in sorted(groups.items())],
                "recognition_provenance.csv", "uncorrected recognition output should raise the rate, but so do "
                "names, places and the spelling of an earlier year; compare within a document, not across")

    # ---- 11. per-collection profile ------------------------------------------------------------------
    coll_rows = []
    for j in jobs:
        rows_j = [r for r in line_rows if r["job"] == j]
        ret_j = [r for r in rows_j if r["status"] == "retained"]
        pr = [r for r in page_rows if r[0] == j]
        rr = [r for r in region_rows if r[0] == j]
        hr = [r for r in htr_rows if r[0] == j]
        coll_rows.append([
            j, len(pr), len(rows_j), pct(sum(r["has_gt"] for r in rows_j), len(rows_j)), len(ret_j),
            round(statistics.mean(len(r["text"].strip()) for r in ret_j), 2),
            round(statistics.mean(len(r["text"].split()) for r in ret_j), 2),
            sum(1 for r in ret_j if (r["stem"], r["line"]) in retained_ph_lines),
            sum(r[16] for r in pr), sum(1 for r in crop_rows if r[0] == j),
            pct(sum(r[7] for r in rr), sum(r[6] for r in rr)), sum(1 for g in gap_rows if g[0] == j),
            sum(1 for t in tail_rows if t[0] == j),
            sum(r["text"].rstrip().endswith("-") for r in ret_j), sum(r["text"].rstrip().endswith("¬") for r in ret_j),
            sum(1 for r in hr if r[17].startswith("likely_uncorrected")),
            sum(1 for r in ret_j if htr_by_page[r["page"]].startswith("likely_uncorrected")),
            sum(1 for r in hr if r[17].startswith("ambiguous")), sum(1 for r in hr if "mixed" in r[17]),
            sum(1 for r in alto_rows if r[0] == j and (not r[3] or "major=True" in r[14])),
        ])
    write_csv(out / "collection_completeness.csv", [
        "collection", "pages", "source_lines", "pct_source_lines_with_gt", "retained_lines", "mean_chars_per_line",
        "mean_words_per_line", "retained_lines_with_placeholder_candidates", f"short_lines_le{SHORT_MAX_CHARS}",
        "crop_text_outliers", "pct_region_lines_with_gt", "internal_gaps", "unfinished_tail_pages", "line_final_hyphen",
        "line_final_not_sign", "pages_likely_uncorrected", "retained_lines_on_likely_uncorrected_pages",
        "pages_ambiguous", "pages_mixed_line_end_convention", "page_alto_major_disagreements"], coll_rows)

    # ---- 12. stratified manual spot-check queue --------------------------------------------------------
    rng = random.Random(SEED)
    flagged_pages = {q[6] for q in queue if q[6]} | {t[1] for t in tail_rows} | {g[1] for g in gap_rows}
    for j in jobs:
        normal = sorted(r[1] for r in page_rows if r[0] == j and r[1] not in flagged_pages)
        for rel in rng.sample(normal, min(NORMAL_PAGES_PER_COLLECTION, len(normal))):
            queue.append(["S_random_normal_page", j, pages[rel]["stem"], "", "", f"seed {SEED}", rel])
    for t in tail_rows:
        queue.append(["C3_unfinished_tail", t[0], t[2], t[7], "", f"{t[5]} empty after {t[4]}", t[1]])
    for g in gap_rows:
        queue.append(["C4_internal_gap", g[0], g[2], g[5], "", f"length {g[4]}", g[1]])
    for r in empty:
        queue.append(["C1_source_line_without_gt", r["job"], r["stem"], r["line"], "", "empty GT", r["page"]])
    dens = sorted(((sum(1 for x in ph_rows if x[2] == "retained" and x[1] == "DETERMINISTIC" and x[4] == pages[rel]["stem"])
                    / max(1, len(by_page[rel])), rel) for rel in page_files), key=lambda x: (-x[0], x[1]))
    for d, rel in dens[:3]:
        if d > 0:
            queue.append(["PH_high_placeholder_density", pages[rel]["job"], pages[rel]["stem"], "", "",
                          f"{d:.3f} deterministic placeholder matches per line", rel])
    cov_sorted = sorted(page_rows, key=lambda r: (r[8], r[1]))
    for tag, r in (("S_min_coverage_page", cov_sorted[0]), ("S_max_coverage_page", cov_sorted[-1])):
        queue.append([tag, r[0], r[2], "", "", f"{r[8]}% lines with GT", r[1]])
    for r in alto_major:
        queue.append(["X1_page_alto_major_disagreement", r[0], r[2], "", "", r[14], r[1]])
    seen, final = set(), []
    for q in queue:
        key = (q[0], q[2], q[3])
        if key not in seen:
            seen.add(key)
            final.append(q)
    final.sort(key=lambda q: (q[0], q[1], q[2], q[3]))
    write_csv(out / "manual_spotcheck_queue.csv", ["reason", "collection", "page", "line_id", "gt", "metric",
              "image_ref"], final)
    rel_root = Path("../../..")
    rows_html = []
    for q in final:
        ref = q[6]
        if ref.endswith(".png"):
            src = (rel_root / "benchmark" / args.benchmark_id / ref).as_posix()
        elif ref.endswith(".xml"):
            p = pages.get(ref)
            src = (rel_root / "incoming" / args.source_id / Path(ref).parent.parent / p["image_filename"]).as_posix() if p else ""
        else:
            src = ""
        img = f'<img loading="lazy" src="{quote(src)}" style="max-width:100%;max-height:420px">' if src else ""
        rows_html.append(f"<tr><td>{html.escape(q[0])}</td><td>{q[1]}</td><td>{html.escape(q[2])}<br>{html.escape(q[3])}</td>"
                         f"<td>{html.escape(q[4])}</td><td>{html.escape(q[5])}</td><td>{img}</td></tr>")
    (out / "manual_spotcheck.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Completeness spot-check</title>"
        "<style>body{font:14px sans-serif;margin:16px}td{border-top:1px solid #ccc;vertical-align:top;padding:4px}</style>"
        f"<h1>Completeness spot-check ({len(final)} items, seed {SEED})</h1><p>Read-only. Do not change GT from this view.</p>"
        "<table><tr><th>reason</th><th>coll.</th><th>page / line</th><th>GT</th><th>metric</th><th>image</th></tr>"
        + "".join(rows_html) + "</table>\n", encoding="utf-8")

    # ---- summary metrics, findings, run record ---------------------------------------------------------
    summary = {
        "pct_source_line_regions_with_gt": pct(n_gt, n_src),
        "pct_retained_pages_with_full_eligible_gt_coverage": pct(full, len(page_rows)),
        "pages_with_unfinished_tails": len(tail_rows), "pages_with_internal_gaps": len({g[1] for g in gap_rows}),
        "pages_with_whole_untranscribed_regions": len({r[1] for r in empty_regions}),
        "retained_lines_with_deterministic_placeholder_candidates": len(retained_ph_lines),
        "crop_text_extreme_outliers": len(crop_rows), "page_alto_major_disagreements": len(alto_major),
        "duplicate_or_version_conflicts": len(version_rows),
        "pages_likely_uncorrected_recognition_output": len(unc),
        "retained_lines_on_likely_uncorrected_pages": unc_lines,
        "pages_ambiguous_recognition_signals": len(amb), "pages_mixed_line_end_convention": len(mixed),
        "manual_spotcheck_items": len(final),
    }
    (out / "completeness_findings.json").write_text(json.dumps({"summary": summary, "findings": findings},
                                                               ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for path in sorted(out.iterdir()):
        if path.suffix in (".csv", ".json", ".html") and EMAIL.search(path.read_text(encoding="utf-8")):
            print(f"refusing: {path.name} contains an e-mail address", file=sys.stderr)
            return 1
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "scripts", "src"], capture_output=True, text=True).stdout
    constants = {k: (v if isinstance(v, (str, int, float)) else repr(v)) for k, v in globals().items()
                 if k.isupper() and k not in ("ROOT", "P", "A", "RO_INDEX", "EMAIL")}
    run = {"source_id": args.source_id, "benchmark_id": args.benchmark_id,
           "frozen_json_sha256": sha256_file(frozen / "FROZEN.json"), "manifest_sha256": sha256_file(frozen / "manifest.jsonl"),
           "excluded_by_decision_sha256": sha256_file(frozen / "excluded_by_decision.jsonl"),
           "decisions_sha256": sha256_file(frozen / "decisions.jsonl"),
           "retained_pages": len(page_files), "source_lines": n_src, "retained_lines": len(feats),
           "training_vocabulary_files": training_files,
           "script": "scripts/benchmark_completeness_audit.py",
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "git_commit": commit, "git_code_dirty_paths": dirty.splitlines(), "constants": constants,
           "outputs": sorted(p.name for p in out.iterdir() if p.name != "run_record.json"),
           "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    (out / "run_record.json").write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))
    for f in findings:
        if not f["id"].startswith("PH_") or f["count"]:
            print(f"[{f['label']}] {f['id']}: {f['count']}/{f['denominator']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
