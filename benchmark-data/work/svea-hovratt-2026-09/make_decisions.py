"""Writes decisions.jsonl for delivery svea-hovratt-2026-09 from the decisions the project owner
approved on 2026-09-27 (D1-D4, D4b extended to unbracketed '??'/'???', D6 probable uncorrected recognition output;
see docs/BENCHMARK_PROTOCOL.md). Earlier outputs are kept: decisions.v1.jsonl / editorial_markup_review.v1.jsonl (before
the D4b extension) and decisions.primary-v1.jsonl (the decisions frozen with svea-hovratt-2026-09-primary, before D6).
Deterministic: it reads only the delivery (via the harness's own PAGE-XML extraction), the inspection findings and the
completeness audit's recognition_provenance.csv (pinned by SHA-256), never model output.

Run from the repository root:  .venv\\Scripts\\python.exe benchmark-data/work/svea-hovratt-2026-09/make_decisions.py
Refuses to overwrite an existing decisions.jsonl.
"""

import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

from archivetrust.htr.benchmark.build import line_target
from archivetrust.htr.benchmark.inspection import inspect_source

SOURCE_ID = "svea-hovratt-2026-09"
ROOT = Path("benchmark-data")
SOURCE = ROOT / "incoming" / SOURCE_ID
WORK = ROOT / "work" / SOURCE_ID
REVIEWER = "hypergeek-dev (approved 2026-09-27)"

KEEP = {"4502442", "4502443", "4502444", "4502445", "4502446"}  # transcribed Svea Hovrätt collections
HELD_BACK = {"4502439", "4502440"}  # TRAINING_VALIDATION_SET_*
SAMPLES = {"4502447", "4502448", "4502449", "4502450"}  # Transkribus sample documents
INSUFFICIENT_GT = {"4502437", "4502438", "4502441"}  # no lines / no GT / 78 of 2705 lines transcribed
NO_GT_CODES = {"gt.missing", "gt.empty", "gt.whitespace_only"}
UNREADABLE_PLACEHOLDER = re.compile(r"\?{2,}")  # D4b extension (approved 2026-09-27): 2+ consecutive '?', anywhere

# D6 (2026-09-27): page-level rule over the completeness audit's signals (scripts/benchmark_completeness_audit.py,
# commit cdf8307). A page is excluded when its source document has at least one T2 page (machine-speed last save AND
# no student convention) and the page itself either carries no student convention (T2, or T3 without convention) or
# mixes line-final '-' and '¬' (T4). The resulting page set must equal the set confirmed by the image review.
D6_PROVENANCE_CSV = WORK / "completeness-audit" / "recognition_provenance.csv"
D6_PROVENANCE_SHA256 = "4144b3a5d6c530654cd227ec3725174db06a515e62073926c6bcd3786b1135e7"
D6_CONFIRMED_PAGES = {("4502442", n) for n in range(1, 32)} | {("4502443", n) for n in range(9, 13)}

REASONS = {
    "samples": "D1: Transkribus sample document, not Svea Hovratt material; outside the primary benchmark",
    "insufficient": "D1: collection has no or insufficient GT; outside the primary benchmark",
    "held_back": "D2: TRAINING_VALIDATION_SET_* collection held back until provenance is confirmed",
    "markup": "D4b: editorial markup (e.g. [???]); convention unconfirmed, held in review queue, excluded from primary",
    "placeholder": "D4b (extended 2026-09-27): unbracketed unreadable-text placeholder, regex \\?{2,} (two or more consecutive "
                   "'?', also embedded, e.g. 'oppbur???'); a single '?' is not a placeholder; GT text unchanged; held in "
                   "review queue, excluded from primary",
    "no_gt": "D1: line in a kept collection has no usable transcription (missing/empty/whitespace-only)",
    "trim": "D4a: deterministic outer-whitespace trim (gt = gt_source.strip()); inner whitespace untouched",
    "uncorrected": "D6: probable uncorrected Transkribus recognition output; whole page excluded (source document has a "
                   "page with machine-speed save and no student convention, and this page has no student convention or "
                   "mixes line-final '-' and '¬'); confirmed by image review; GT text unchanged",
}


SCORING_NOTE = (
    "# D5 (scoring, pre-registered 2026-09-27 before any model run; not a GT change): primary score = raw. "
    "One labelled sensitivity score 'line_end_hyphen_harmonized' (scoring.py SCORING_VERSION 2): a line-final '¬' "
    "counts as '-' in reference and prediction. Internal hyphens, 'ß', ';' and all other characters untouched. "
    "Both raw and sensitivity CER/WER are reported.\n"
)
PLACEHOLDER_NOTE = (
    "# D4b extension (approved 2026-09-27, before any model run, after the deterministic convention audit): a line "
    "whose source GT matches the regex \\?{2,} (two or more consecutive '?', anywhere, including embedded forms such "
    "as 'oppbur???' and '21???') is an unreadable-text placeholder and is excluded from primary and held in "
    "editorial_markup_review.jsonl. A single '?' is not affected. No GT text is modified.\n"
)


UNCORRECTED_NOTE = (
    "# D6 (approved 2026-09-27, before any model run, after the completeness audit and a seeded image review): 35 "
    "whole pages are excluded from primary as probable uncorrected Transkribus recognition output: export job 4502442 "
    "pages 1-31 and 4502443 pages 9-12. Rule: the source document has at least one page whose last save followed "
    "the previous save within <= 1.0 s per line AND that has no student convention (0 line-final '-', 0 lines with "
    "'[' or '??'); every page of that document with no student convention, or mixing line-final '-' and '¬', "
    "is excluded. 4502443 p9 (corrected top, uncorrected from the first '¬') is excluded whole: no objective "
    "boundary exists. Lines already excluded by D1-D4b keep their earlier decision. No GT text is modified.\n"
)


def d6_pages() -> dict[str, dict]:
    """Page XML (relative to the delivery) -> audit row, for the pages D6 excludes."""
    if hashlib.sha256(D6_PROVENANCE_CSV.read_bytes()).hexdigest() != D6_PROVENANCE_SHA256:
        raise SystemExit(f"{D6_PROVENANCE_CSV} is not the pinned audit output; refusing to derive D6 from it")
    with D6_PROVENANCE_CSV.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    no_convention = lambda r: int(r["line_final_hyphen"]) == 0 and int(r["lines_with_human_editorial_marks"]) == 0  # noqa: E731
    docs_with_t2 = {r["page_xml"].rsplit("/page/", 1)[0] for r in rows if r["class"].startswith("likely_uncorrected")}
    pages = {r["page_xml"]: r for r in rows if r["page_xml"].rsplit("/page/", 1)[0] in docs_with_t2
             and (no_convention(r) or "mixed_line_end_convention" in r["class"])}
    got = {(r["collection"], int(r["page_nr"])) for r in pages.values()}
    if got != D6_CONFIRMED_PAGES:
        raise SystemExit(f"D6 rule selects {sorted(got ^ D6_CONFIRMED_PAGES)} differently from the confirmed set")
    return pages


def job_of(path: str) -> str:
    match = re.search(r"export_job_(\d+)", path.replace("\\", "/"))
    if not match:
        raise SystemExit(f"no export job in {path!r}")
    return match.group(1)


def main() -> int:
    out = WORK / "decisions.jsonl"
    if out.exists():
        print(f"{out} exists; refusing to overwrite", file=sys.stderr)
        return 1
    result = inspect_source(SOURCE, adapter_id="page_xml")
    assert result.extraction is not None and result.extraction.adapter_id == "page_xml"
    findings_by_line: dict[str, list] = {}
    for finding in result.findings:
        if finding.line_id:
            findings_by_line.setdefault(finding.line_id, []).append(finding)

    decisions: list[dict] = []
    markup_queue: list[dict] = []
    d6 = d6_pages()
    d6_archive: list[dict] = []
    d6_record: list[dict] = []
    excluded_pages: set[str] = set()
    tally = Counter()
    for line in sorted(result.extraction.lines, key=lambda c: (c.gt_path or "", c.line_order)):
        job = job_of(line.gt_path or line.image_path)
        if job not in KEEP:
            category = ("samples" if job in SAMPLES else "held_back" if job in HELD_BACK
                        else "insufficient" if job in INSUFFICIENT_GT else None)
            if category is None:
                raise SystemExit(f"export job {job} is not classified")
            if line.gt_path not in excluded_pages:
                excluded_pages.add(line.gt_path)
                decisions.append({"target": f"file:{line.gt_path}", "action": "exclude", "reason": REASONS[category],
                                  "reviewer": REVIEWER})
                tally[f"exclude page ({category})"] += 1
            continue
        key = line_target(line)
        codes = {f.code for f in findings_by_line.get(key, [])}
        if "gt.possible_editorial_markup" in codes:
            decisions.append({"target": key, "action": "exclude", "reason": REASONS["markup"], "reviewer": REVIEWER})
            markup = next(f for f in findings_by_line[key] if f.code == "gt.possible_editorial_markup")
            markup_queue.append({"target": key, "gt_path": line.gt_path, "image_path": line.image_path,
                                 "gt_source": line.gt_source, "spans": markup.detail})
            tally["exclude line (markup review)"] += 1
        elif line.gt_source is not None and UNREADABLE_PLACEHOLDER.search(line.gt_source):
            decisions.append({"target": key, "action": "exclude", "reason": REASONS["placeholder"], "reviewer": REVIEWER})
            markup_queue.append({"target": key, "gt_path": line.gt_path, "image_path": line.image_path,
                                 "gt_source": line.gt_source, "rule": "D4b_unbracketed_placeholder",
                                 "spans": UNREADABLE_PLACEHOLDER.findall(line.gt_source)})
            tally["exclude line (unbracketed placeholder review)"] += 1
        elif codes & NO_GT_CODES or line.gt_source is None or not line.gt_source.strip():
            decisions.append({"target": key, "action": "exclude", "reason": REASONS["no_gt"], "reviewer": REVIEWER})
            tally["exclude line (no GT)"] += 1
        elif "gt.outer_whitespace" in codes:
            trimmed = line.gt_source.strip()
            assert trimmed and trimmed == line.gt_source.strip() and trimmed in line.gt_source
            decisions.append({"target": key, "action": "set_gt", "gt": trimmed, "reason": REASONS["trim"],
                              "reviewer": REVIEWER})
            tally["set_gt (outer-whitespace trim)"] += 1
        if line.gt_path in d6:
            d6_archive.append({"target": key, "gt_path": line.gt_path, "image_path": line.image_path,
                               "gt_source": line.gt_source, "page_class": d6[line.gt_path]["class"],
                               "earlier_decision": decisions[-1]["reason"].split(":")[0]
                               if decisions and decisions[-1]["target"] == key else None})

    for gt_path, row in sorted(d6.items()):
        label = f"{row['collection']} p{int(row['page_nr']):02d} {row['page']}"
        decisions.append({"target": f"file:{gt_path}", "action": "exclude",
                          "reason": f"{REASONS['uncorrected']} [{label}]", "reviewer": REVIEWER})
        d6_record.append({"page": label, "gt_path": gt_path, "class": row["class"],
                          "sec_per_line_since_previous_save": row["sec_per_line_since_previous_save"],
                          "line_final_marks": row["line_final_mark_sequence"],
                          "lines_with_human_editorial_marks": int(row["lines_with_human_editorial_marks"]),
                          "source_lines": int(row["source_lines"])})
        tally["exclude page (D6 probable uncorrected recognition output)"] += 1
    if {d["gt_path"] for d in d6_archive} != set(d6):
        raise SystemExit("a D6 page was not found in the delivery extraction")

    out.write_text("".join(json.dumps(d, ensure_ascii=False, sort_keys=True) + "\n" for d in decisions)
                   + SCORING_NOTE + PLACEHOLDER_NOTE + UNCORRECTED_NOTE,
                   encoding="utf-8")
    (WORK / "editorial_markup_review.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in markup_queue), encoding="utf-8")
    (WORK / "d6_pages.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in d6_record), encoding="utf-8")
    (WORK / "d6_excluded_lines.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in d6_archive), encoding="utf-8")
    for name, count in sorted(tally.items()):
        print(f"{count:6}  {name}")
    print(f"{len(decisions):6}  decisions -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
