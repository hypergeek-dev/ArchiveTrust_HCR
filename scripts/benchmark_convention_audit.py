"""Deterministic transcription-convention audit of a benchmark candidate (read-only).

    .venv\\Scripts\\python.exe scripts/benchmark_convention_audit.py svea-hovratt-2026-09 \\
        --training-parquet-dir benchmark-data/training-reference

Reads the candidate manifest (canonical GT, per-line Transkribus `custom` attribute), the delivery's
PAGE-XML / metadata.xml files, the editorial-markup review queue and, optionally, the transcription
column of training parquet files as a comparison corpus. Writes CSV/JSON tables, a findings list and
a manual-review queue to benchmark-data/work/<source>/convention-audit/. Never writes GT.

Every heuristic constant is defined below; every finding carries its rule, label and output file.
Labels: DETERMINISTIC (exact count of a stated rule), HEURISTIC (a stated rule that approximates a
concept), MANUAL REVIEW REQUIRED (cases the rules cannot resolve; listed in manual_review_queue.csv).
Personal fields in the delivery metadata (uploader, user names/ids, e-mail) are never copied.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from archivetrust.htr.benchmark.contract import BenchmarkLine, read_manifest, sha256_file
from archivetrust.htr.benchmark.report import default_loghi_charset

# ---- explicit rules and thresholds ------------------------------------------------------------
HYPHEN_LIKE = "-¬=⸗‐‑–—"  # characters counted as possible line-end hyphenation marks
SENTENCE_END = ".!?"  # a token after one of these (ignoring spaces) is "after sentence punctuation"
TOKEN = re.compile(r"[^\W\d_]+")  # a word token = a run of letters
MAX_ABBR_LETTERS = 4  # A1: 1..4 letters directly followed by '.' or ':' is an abbreviation candidate
ABBR_TOKEN = re.compile(rf"(?<![^\W\d_])([^\W\d_]{{1,{MAX_ABBR_LETTERS}}})([.:])(?=\s|$|[,;)\]])")
ABBR_MIN_COUNT = 3  # A1 candidates occurring fewer times are still listed but not counted as a pattern
EDITORIAL_RULES = {  # rule id -> (label, regex)
    "E1_square_brackets": ("DETERMINISTIC", re.compile(r"\[[^\]]*\]")),
    "E2_curly_braces": ("DETERMINISTIC", re.compile(r"\{[^}]*\}")),
    "E3_angle_brackets": ("DETERMINISTIC", re.compile(r"<[^>]*>")),
    "E4_parentheses": ("HEURISTIC", re.compile(r"\([^)]*\)")),  # may be original text, not editorial
    "E5_question_mark_touching_letter": ("HEURISTIC", re.compile(r"[^\W\d_]\?|\?[^\W\d_]")),
    "E6_ellipsis": ("DETERMINISTIC", re.compile(r"…|\.\.\.")),
    "E7_unbalanced_bracket": ("DETERMINISTIC", re.compile(r"^[^\[]*\]|\[[^\]]*$")),
    "E9_repeated_question_marks": ("DETERMINISTIC", re.compile(r"\?{2,}")),  # uncertainty marker without brackets
}
EDITORIAL_TAGS = ("unclear", "gap", "sic", "comment", "supplied", "del", "add")  # Transkribus custom tags
ABBR_TAGS = ("abbrev",)
HISTORICAL_SPELLING = {  # marker -> regex over lowercased text (not casefold: it maps ß to ss); frequency per 1000 words (HEURISTIC)
    "hw- (hwar, hwilken)": re.compile(r"\bhw"),
    "qw- (qwinna)": re.compile(r"\bqw"),
    "dh (medh, godh)": re.compile(r"dh"),
    "fw (hafwa, af)": re.compile(r"fw"),
    "ff (aff, ...)": re.compile(r"ff"),
    "ck word-final": re.compile(r"ck\b"),
    "æ": re.compile("æ"),
    "ß": re.compile("ß"),
    "ſ long s": re.compile("ſ"),
}
PERSONAL_KEYS = {"uploader", "uploaderid", "username", "userid", "user", "email", "owner", "creator_email"}
TRAINING_TEXT_COLUMN = "transcription"
# -------------------------------------------------------------------------------------------------

ROOT = Path("benchmark-data")
NS = re.compile(r"\{[^}]*\}")
CUSTOM_TAG = re.compile(r"(\w+)\s*\{([^}]*)\}")


def job_of(path: str) -> str:
    match = re.search(r"export_job_(\d+)", path.replace("\\", "/"))
    return match.group(1) if match else "(unknown)"


def custom_tags(custom: str | None) -> list[tuple[str, dict]]:
    out = []
    for name, body in CUSTOM_TAG.findall(custom or ""):
        props = dict(p.split(":", 1) for p in (x.strip() for x in body.split(";")) if ":" in p)
        out.append((name, {k.strip(): v.strip() for k, v in props.items()}))
    return out


def tag_span(text: str, props: dict) -> str | None:
    try:
        start, length = int(props["offset"]), int(props["length"])
    except (KeyError, ValueError):
        return None
    return text[start:start + length]


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def per(n: int, d: int, k: int) -> float | None:
    return round(n * k / d, 3) if d else None


def load_training_texts(parquet_dir: Path | None) -> tuple[list[str], list[dict]]:
    if parquet_dir is None:
        return [], []
    import pyarrow.parquet as pq  # noqa: PLC0415

    texts, files = [], []
    for f in sorted(parquet_dir.glob("*.parquet")):
        column = pq.read_table(f, columns=[TRAINING_TEXT_COLUMN]).column(0).to_pylist()
        texts += [unicodedata.normalize("NFC", t or "").strip() for t in column]
        files.append({"name": f.name, "bytes": f.stat().st_size, "rows": len(column)})
    return texts, files


def token_positions(text: str) -> list[tuple[str, str]]:
    """(token, position) with position in {line_initial, after_sentence_end, mid}."""
    out = []
    for i, m in enumerate(TOKEN.finditer(text)):
        before = text[:m.start()].rstrip()
        if i == 0 and not before:
            pos = "line_initial"
        elif before and before[-1] in SENTENCE_END:
            pos = "after_sentence_end"
        else:
            pos = "mid"
        out.append((m.group(0), pos))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source_id")
    ap.add_argument("--training-parquet-dir", type=Path)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--out-dir", type=Path, help="output folder (default: work/<source>/convention-audit)")
    args = ap.parse_args()

    work = ROOT / "work" / args.source_id
    source = ROOT / "incoming" / args.source_id
    out = args.out_dir or work / "convention-audit"
    if out.exists() and not args.overwrite:
        print(f"{out} exists; pass --overwrite to regenerate", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = work / "candidate" / "manifest.jsonl"
    lines = sorted(read_manifest(manifest_path), key=lambda l: (l.document_id, l.page_id, l.line_order))
    jobs = sorted({job_of(l.source_relative_path) for l in lines})
    job_line = {l.line_id: job_of(l.source_relative_path) for l in lines}
    training, training_files = load_training_texts(args.training_parquet_dir)
    charset = default_loghi_charset()
    if charset is None:
        print("Loghi charset not found (frozen checkpoint missing?)", file=sys.stderr)
        return 1
    findings: list[dict] = []
    review: list[list] = []

    def finding(fid, label, rule, matches, collections, examples, output):
        findings.append({"id": fid, "label": label, "rule": rule, "matches": matches, "collections": collections,
                         "examples": examples[:5], "output": output, "script": "scripts/benchmark_convention_audit.py"})

    by_job: dict[str, list] = defaultdict(list)
    for l in lines:
        by_job[job_of(l.source_relative_path)].append(l)
    groups = {j: [l.gt_canonical for l in by_job[j]] for j in jobs}
    groups["candidate_total"] = [l.gt_canonical for l in lines]
    if training:
        groups["training_ref"] = training
    cols = list(groups)
    nchars = {g: sum(len(t) for t in ts) for g, ts in groups.items()}
    nwords = {g: sum(len(t.split()) for t in ts) for g, ts in groups.items()}

    # ---- 1. metadata_summary.json (XML field extraction) ------------------------------------------
    meta: dict = {"collections": {}}
    for j in jobs:
        page_files = sorted({l.gt_source_relative_path for l in by_job[j]})
        tags, status, creators, conf_lines, custom_names = Counter(), Counter(), Counter(), 0, Counter()
        for rel in page_files:
            root = ET.parse(source / rel).getroot()
            for el in root.iter():
                tag = NS.sub("", el.tag)
                tags[tag] += 1
                if tag == "TranskribusMetadata":
                    status[el.get("status", "(none)")] += 1
                if tag == "Creator" and el.text:
                    creators["<redacted: contains @>" if "@" in el.text else el.text.strip()] += 1
                if tag == "TextLine":
                    for te in el:
                        if NS.sub("", te.tag) == "TextEquiv" and te.get("conf") is not None:
                            conf_lines += 1
        for l in by_job[j]:
            for name, _ in custom_tags(l.source_metadata.get("custom")):
                custom_names[name] += 1
        coll_dir = source / Path(page_files[0]).parent.parent
        doc_meta = {}
        meta_file = coll_dir / "metadata.xml"
        if meta_file.is_file():
            for el in ET.parse(meta_file).getroot().iter():
                key = NS.sub("", el.tag)
                if el.text and el.text.strip() and len(list(el)) == 0:
                    doc_meta[key] = "<redacted: present>" if key.lower() in PERSONAL_KEYS or "@" in el.text else el.text.strip()[:200]
        meta["collections"][j] = {
            "collection_name": unicodedata.normalize("NFC", coll_dir.name), "pages": len(page_files),
            "lines_in_candidate": len(by_job[j]), "page_xml_element_counts": dict(sorted(tags.items())),
            "transkribus_page_status": dict(status), "page_xml_creator": dict(creators),
            "textlines_with_textequiv_conf": conf_lines, "line_custom_tag_counts": dict(sorted(custom_names.items())),
            "document_metadata_xml": dict(sorted(doc_meta.items())),
        }
    (out / "metadata_summary.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                                               encoding="utf-8")
    statuses = {j: m["transkribus_page_status"] for j, m in meta["collections"].items()}
    finding("M1", "DETERMINISTIC", "count TranskribusMetadata@status per page", sum(sum(s.values()) for s in statuses.values()),
            jobs, [f"{j}: {s}" for j, s in statuses.items()], "metadata_summary.json")
    tagged = {j: m["line_custom_tag_counts"] for j, m in meta["collections"].items()}
    finding("M2", "DETERMINISTIC", "Transkribus custom tags on candidate TextLines (name counts)",
            sum(sum(v for k, v in t.items() if k != "readingOrder") for t in tagged.values()), jobs,
            [f"{j}: {t}" for j, t in tagged.items()], "metadata_summary.json")
    conf = {j: m["textlines_with_textequiv_conf"] for j, m in meta["collections"].items()}
    finding("M3", "DETERMINISTIC", "TextLine/TextEquiv elements carrying a @conf attribute (machine-recognition trace)",
            sum(conf.values()), [j for j, v in conf.items() if v], [f"{j}: {v}" for j, v in conf.items()], "metadata_summary.json")

    # ---- 2. unicode_inventory.csv -----------------------------------------------------------------
    counts = {g: Counter("".join(ts)) for g, ts in groups.items()}
    chars = sorted(set().union(*(counts[g] for g in cols)))
    rows = []
    for c in chars:
        rows.append([c if c.isprintable() else "", f"U+{ord(c):04X}", unicodedata.name(c, "(unnamed)"),
                     unicodedata.category(c), c in charset] + [counts[g][c] for g in cols]
                    + [per(counts[g][c], nchars[g], 10000) for g in cols])
    write_csv(out / "unicode_inventory.csv", ["char", "codepoint", "name", "category", "in_loghi_charset"]
              + [f"count_{g}" for g in cols] + [f"per10k_{g}" for g in cols], rows)
    oov = {c: counts["candidate_total"][c] for c in chars if counts["candidate_total"][c] and c not in charset and c != " "}
    finding("U1", "DETERMINISTIC", "candidate GT code points not in the Loghi charset", sum(oov.values()),
            [j for j in jobs if any(counts[j][c] for c in oov)], [f"{c!r} x{n}" for c, n in oov.items()], "unicode_inventory.csv")
    odd = {c: n for c, n in counts["candidate_total"].items()
           if unicodedata.category(c) in ("Mn", "Cc", "Cf", "Co", "Zs") and c != " "}
    finding("U2", "DETERMINISTIC", "combining marks, control/format/private-use chars or non-ASCII spaces in candidate GT",
            sum(odd.values()), [j for j in jobs if any(counts[j][c] for c in odd)],
            [f"U+{ord(c):04X} {unicodedata.name(c, '?')} x{n}" for c, n in odd.items()], "unicode_inventory.csv")
    if training:
        only_train = sorted((c for c in counts["training_ref"] if not counts["candidate_total"][c] and counts["training_ref"][c] >= 50),
                            key=lambda c: -counts["training_ref"][c])
        finding("U3", "DETERMINISTIC", "chars with >= 50 occurrences in training_ref and 0 in candidate",
                len(only_train), ["training_ref"], [f"{c!r} x{counts['training_ref'][c]}" for c in only_train], "unicode_inventory.csv")

    # ---- 3. punctuation_profile.csv ---------------------------------------------------------------
    punct = [c for c in chars if unicodedata.category(c)[0] in "PS"]
    final = {g: Counter(t[-1] for t in ts if t) for g, ts in groups.items()}
    initial = {g: Counter(t[0] for t in ts if t) for g, ts in groups.items()}
    rows = [[c, f"U+{ord(c):04X}", unicodedata.name(c, "?")] + [counts[g][c] for g in cols]
            + [per(counts[g][c], nchars[g], 10000) for g in cols] + [final[g][c] for g in cols] + [initial[g][c] for g in cols]
            for c in punct]
    write_csv(out / "punctuation_profile.csv", ["char", "codepoint", "name"] + [f"count_{g}" for g in cols]
              + [f"per10k_{g}" for g in cols] + [f"line_final_{g}" for g in cols] + [f"line_initial_{g}" for g in cols], rows)

    # ---- 4. hyphenation_analysis.csv (line-final marks; next-line heuristic on the candidate) -------
    nxt: dict[str, BenchmarkLine] = {}
    for a, b in zip(lines, lines[1:]):
        if (a.document_id, a.page_id) == (b.document_id, b.page_id) and b.line_order == a.line_order + 1:
            nxt[a.line_id] = b
    rows, ambiguous = [], []
    for g in cols:
        for mark in HYPHEN_LIKE:
            texts = groups[g]
            lf = sum(1 for t in texts if t.endswith(mark))
            internal_compound = sum(len(re.findall(rf"[^\W\d_]{re.escape(mark)}[^\W\d_]", t)) for t in texts)
            internal_spaced = sum(len(re.findall(rf"\s{re.escape(mark)}\s", t)) for t in texts)
            li = sum(1 for t in texts if t.startswith(mark))
            nl = Counter()
            if g != "training_ref":
                for l in (by_job[g] if g in by_job else lines):
                    if not l.gt_canonical.endswith(mark):
                        continue
                    b = nxt.get(l.line_id)
                    first = b.gt_canonical.lstrip()[:1] if b else ""
                    kind = ("next_absent" if b is None else "next_lower" if first.islower()
                            else "next_upper" if first.isupper() else "next_other")
                    nl[kind] += 1
                    if g == "candidate_total" and kind in ("next_upper", "next_other"):
                        ambiguous.append((l, mark, kind, b))
            rows.append([g, mark, f"U+{ord(mark):04X}", lf, per(lf, len(texts), 100), internal_compound, internal_spaced, li,
                         nl["next_lower"], nl["next_upper"], nl["next_other"], nl["next_absent"]])
    write_csv(out / "hyphenation_analysis.csv", ["group", "mark", "codepoint", "line_final", "line_final_pct_of_lines",
              "internal_letter_mark_letter", "internal_space_mark_space", "line_initial", "next_line_starts_lower",
              "next_line_starts_upper", "next_line_starts_other", "next_line_not_in_candidate"], rows)
    for mark in "-¬":
        per_job = {j: sum(1 for t in groups[j] if t.endswith(mark)) for j in jobs}
        finding(f"H{'1' if mark == '-' else '2'}", "DETERMINISTIC", f"lines whose canonical GT ends with {mark!r}",
                sum(per_job.values()), [j for j, v in per_job.items() if v], [f"{j}: {v}" for j, v in per_job.items()],
                "hyphenation_analysis.csv")
    lower = sum(r[8] for r in rows if r[0] == "candidate_total" and r[1] in "-¬")
    finding("H3", "HEURISTIC", "line ends with '-' or '¬' and the next line (same page, next reading order) starts lowercase "
            "= word continues (hyphenation)", lower, jobs, [], "hyphenation_analysis.csv")
    for l, mark, kind, b in ambiguous:
        review.append(["H4_line_end_mark_next_not_lowercase", job_line[l.line_id], l.line_id, l.gt_canonical,
                       b.gt_canonical if b else "", l.line_image_path])
    finding("H4", "MANUAL REVIEW REQUIRED", "line ends with a hyphen-like mark but the next line starts uppercase/other: "
            "hyphenation or dash?", len(ambiguous), sorted({job_line[a[0].line_id] for a in ambiguous}),
            [f"{a[0].gt_canonical[-30:]} || {a[3].gt_canonical[:20] if a[3] else ''}" for a in ambiguous], "manual_review_queue.csv")
    mixed = {j: (sum(1 for t in groups[j] if t.endswith("-")), sum(1 for t in groups[j] if t.endswith("¬"))) for j in jobs}
    both = [j for j, (h, n) in mixed.items() if h and n]
    finding("H5", "DETERMINISTIC", "collections using both '-' and '¬' line-finally", len(both), both,
            [f"{j}: '-' {h}, '¬' {n}" for j, (h, n) in mixed.items()], "hyphenation_analysis.csv")
    for l in lines:
        if l.gt_canonical.endswith("¬"):
            review.append(["H6_line_final_not_sign_in_candidate", job_line[l.line_id], l.line_id, l.gt_canonical,
                           nxt[l.line_id].gt_canonical if l.line_id in nxt else "", l.line_image_path])

    # ---- 5. capitalization_profile.csv ------------------------------------------------------------
    rows = []
    cap_mid = {}
    for g in cols:
        c = Counter()
        for t in groups[g]:
            for tok, pos in token_positions(t):
                c[f"{pos}_tokens"] += 1
                c[f"{pos}_capitalized"] += tok[0].isupper()
                c["all_caps_len2plus"] += len(tok) >= 2 and tok.isupper()
                c["tokens"] += 1
        cap_mid[g] = per(c["mid_capitalized"], c["mid_tokens"], 100)
        rows.append([g, c["tokens"], c["line_initial_tokens"], per(c["line_initial_capitalized"], c["line_initial_tokens"], 100),
                     c["after_sentence_end_tokens"], per(c["after_sentence_end_capitalized"], c["after_sentence_end_tokens"], 100),
                     c["mid_tokens"], cap_mid[g], c["all_caps_len2plus"]])
    write_csv(out / "capitalization_profile.csv", ["group", "tokens", "line_initial_tokens", "line_initial_capitalized_pct",
              "after_sentence_end_tokens", "after_sentence_end_capitalized_pct", "mid_sentence_tokens",
              "mid_sentence_capitalized_pct", "all_caps_tokens_len2plus"], rows)
    finding("C1", "HEURISTIC", "share of mid-sentence letter tokens starting uppercase (token not line-initial and not after "
            f"'{SENTENCE_END}'); high values fit unnormalized early-modern capitalization", None, cols,
            [f"{g}: {v}%" for g, v in cap_mid.items()], "capitalization_profile.csv")

    # ---- 6. abbreviation_candidates.csv -----------------------------------------------------------
    abbr = defaultdict(lambda: {"count": Counter(), "examples": []})
    for g in [*jobs, "training_ref"] if training else jobs:
        src = [(l.line_id, l.gt_canonical) for l in by_job[g]] if g in by_job else [(None, t) for t in training]
        for line_id, t in src:
            for m in ABBR_TOKEN.finditer(t):
                key = ("A1_short_token_dot_colon", m.group(1) + m.group(2))
                abbr[key]["count"][g] += 1
                if line_id and len(abbr[key]["examples"]) < 3:
                    abbr[key]["examples"].append(f"{line_id} :: {t}")
    for l in lines:
        for name, props in custom_tags(l.source_metadata.get("custom")):
            if name in ABBR_TAGS:
                key = ("A2_transkribus_abbrev_tag", f"{tag_span(l.gt_source, props)} -> {props.get('expansion', '')}")
                abbr[key]["count"][job_line[l.line_id]] += 1
                if len(abbr[key]["examples"]) < 3:
                    abbr[key]["examples"].append(f"{l.line_id} :: {l.gt_source}")
        for ch in l.gt_canonical:
            if unicodedata.category(ch) in ("Lm", "Mn") or ch in "ↄꝛ÷⁊ꝰ":
                key = ("A3_abbreviation_sign_char", f"{ch} U+{ord(ch):04X}")
                abbr[key]["count"][job_line[l.line_id]] += 1
                if len(abbr[key]["examples"]) < 3:
                    abbr[key]["examples"].append(f"{l.line_id} :: {l.gt_canonical}")
    agroups = [*jobs, "training_ref"] if training else jobs
    rows = sorted(([rule, cand, sum(v["count"][j] for j in jobs)] + [v["count"][g] for g in agroups] + [" | ".join(v["examples"])]
                   for (rule, cand), v in abbr.items()), key=lambda r: (r[0], -r[2], r[1]))
    write_csv(out / "abbreviation_candidates.csv", ["rule", "candidate", "count_candidate_total"]
              + [f"count_{g}" for g in agroups] + ["examples"], rows)
    a1 = [r for r in rows if r[0] == "A1_short_token_dot_colon" and r[2] >= ABBR_MIN_COUNT]
    finding("A1", "HEURISTIC", f"tokens of 1..{MAX_ABBR_LETTERS} letters followed by '.' or ':' occurring >= {ABBR_MIN_COUNT} "
            "times in the candidate (abbreviations kept unexpanded)", sum(r[2] for r in a1), jobs,
            [f"{r[1]} x{r[2]}" for r in a1[:10]], "abbreviation_candidates.csv")
    a2 = [r for r in rows if r[0] == "A2_transkribus_abbrev_tag"]
    finding("A2", "DETERMINISTIC", "Transkribus 'abbrev' tags on candidate lines", sum(r[2] for r in a2), jobs,
            [f"{r[1]} x{r[2]}" for r in a2[:10]], "abbreviation_candidates.csv")

    # ---- 7. editorial_markup.csv ------------------------------------------------------------------
    queue_path = work / "editorial_markup_review.jsonl"
    held = [json.loads(x) for x in queue_path.read_text(encoding="utf-8").splitlines()] if queue_path.is_file() else []
    scope = [(l.line_id, job_line[l.line_id], l.gt_source, "candidate", l.source_metadata.get("custom")) for l in lines]
    scope += [(h["target"], job_of(h["gt_path"]), h["gt_source"], "excluded_markup_review", None) for h in held]
    rows, per_rule = [], defaultdict(Counter)
    for line_id, j, text, where, custom in scope:
        for rule, (label, rx) in EDITORIAL_RULES.items():
            for m in rx.finditer(text or ""):
                rows.append([rule, label, where, j, line_id, m.group(0), text])
                per_rule[(rule, where)][j] += 1
        for name, props in custom_tags(custom):
            if name in EDITORIAL_TAGS:
                rows.append([f"E8_transkribus_{name}_tag", "DETERMINISTIC", where, j, line_id, tag_span(text, props) or "", text])
                per_rule[(f"E8_transkribus_{name}_tag", where)][j] += 1
    rows.sort(key=lambda r: (r[0], r[2], r[3], r[4], r[5]))
    write_csv(out / "editorial_markup.csv", ["rule", "label", "scope", "collection", "line_id", "match", "gt_source"], rows)
    for (rule, where), c in sorted(per_rule.items()):
        label = next((lab for r, (lab, _) in EDITORIAL_RULES.items() if r == rule), "DETERMINISTIC")
        examples = [r[5] for r in rows if r[0] == rule and r[2] == where]
        finding(f"{rule}@{where}", label, f"{rule} in {where} lines", sum(c.values()), sorted(c),
                [f"{e!r} x{n}" for e, n in Counter(examples).most_common(5)], "editorial_markup.csv")
        if where == "candidate" and label != "DETERMINISTIC" or (where == "candidate" and rule.startswith(("E1", "E2", "E3", "E7", "E9"))):
            for r in rows:
                if r[0] == rule and r[2] == "candidate":
                    lm = next(l for l in lines if l.line_id == r[4])
                    review.append([f"{rule}_in_candidate", r[3], r[4], r[6], "", lm.line_image_path])

    # ---- 8. collection_comparison.csv -------------------------------------------------------------
    metrics: list[tuple[str, dict]] = [
        ("lines", {g: len(groups[g]) for g in cols}),
        ("characters", nchars), ("words", nwords),
        ("mean_chars_per_line", {g: round(nchars[g] / len(groups[g]), 2) if groups[g] else None for g in cols}),
        ("pct_lines_ending_-", {g: per(sum(t.endswith("-") for t in groups[g]), len(groups[g]), 100) for g in cols}),
        ("pct_lines_ending_¬", {g: per(sum(t.endswith("¬") for t in groups[g]), len(groups[g]), 100) for g in cols}),
        ("mid_sentence_capitalized_pct", cap_mid),
        ("digits_per10k_chars", {g: per(sum(ch.isdigit() for t in groups[g] for ch in t), nchars[g], 10000) for g in cols}),
    ]
    for c in ";:,.()&§½/'\"":
        metrics.append((f"{c}_per10k_chars", {g: per(counts[g][c], nchars[g], 10000) for g in cols}))
    for name, rx in HISTORICAL_SPELLING.items():
        metrics.append((f"hist_{name}_per1000_words",
                        {g: per(sum(len(rx.findall(t.lower())) for t in groups[g]), nwords[g], 1000) for g in cols}))
    metrics.append((f"A1_abbr_candidates_per1000_words",
                    {g: per(sum(len(ABBR_TOKEN.findall(t)) for t in groups[g]), nwords[g], 1000) for g in cols}))
    write_csv(out / "collection_comparison.csv", ["metric"] + cols, [[m] + [v.get(g) for g in cols] for m, v in metrics])
    hist = {name: {g: v for g, v in vals.items()} for name, vals in metrics if name.startswith("hist_")}
    finding("S1", "HEURISTIC", "historical spelling markers per 1000 words (regexes in HISTORICAL_SPELLING); similar rates in "
            "candidate and training_ref fit diplomatic (unmodernized) spelling in both", None, cols,
            [f"{k}: cand {v.get('candidate_total')} / train {v.get('training_ref')}" for k, v in hist.items()],
            "collection_comparison.csv")

    # ---- review queue, findings, run record ---------------------------------------------------------
    review = sorted({tuple(r) for r in review}, key=lambda r: (r[0], r[1], r[2]))  # one row per line and reason
    write_csv(out / "manual_review_queue.csv", ["queue_reason", "collection", "line_id", "gt_canonical_or_source",
              "next_line_gt", "line_image_path"], review)
    (out / "audit_findings.json").write_text(json.dumps(findings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    constants = {k: (v if isinstance(v, (str, int, tuple)) else repr(v)) for k, v in globals().items()
                 if k.isupper() and k not in ("ROOT", "NS", "CUSTOM_TAG")}
    run = {"source_id": args.source_id, "manifest_sha256": sha256_file(manifest_path), "candidate_lines": len(lines),
           "build_json_sha256": sha256_file(work / "candidate" / "build.json"), "training_files": training_files,
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "constants": constants,
           "outputs": sorted(p.name for p in out.iterdir()), "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    (out / "run_record.json").write_text(json.dumps(run, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    for f in findings:
        print(f"[{f['label']}] {f['id']}: {f['matches']}  -> {f['output']}")
    print(f"manual review queue: {len(review)} rows; outputs in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
