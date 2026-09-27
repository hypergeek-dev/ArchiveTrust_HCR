"""Writes dataset_card.json for svea-hovratt-2026-09-primary-v2: model-independent facts and caveats embedded in
FROZEN.json by `freeze --dataset-card`. The card of the superseded svea-hovratt-2026-09-primary is kept as
dataset_card.primary-v1.json (and inside that benchmark's FROZEN.json). Counts are derived from the candidate, the decisions and the convention audit, never typed.
No personal names or e-mail addresses. Run from the repository root:
    .venv\\Scripts\\python.exe benchmark-data/work/svea-hovratt-2026-09/make_dataset_card.py
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

from archivetrust.htr.benchmark.contract import read_manifest, sha256_file
from archivetrust.htr.benchmark.report import default_loghi_charset
from archivetrust.htr.benchmark.scoring import SENSITIVITY_RULES

WORK = Path("benchmark-data/work/svea-hovratt-2026-09")
BENCHMARK_ID = "svea-hovratt-2026-09-primary-v2"
SUPERSEDED = Path("benchmark-data/benchmark/svea-hovratt-2026-09-primary")
CONVENTION = WORK / "convention-audit-primary-v2"
COMPLETENESS_V1 = WORK / "completeness-audit"
COMPLETENESS = WORK / "completeness-audit-primary-v2"
REVIEW = WORK / "image-review"
PLACEHOLDER = r"\?{2,}"
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def job_of(path: str) -> str:
    return re.search(r"export_job_(\d+)", path).group(1)


def main() -> int:
    build = json.loads((WORK / "candidate/build.json").read_text(encoding="utf-8"))
    lines = read_manifest(WORK / "candidate/manifest.jsonl")
    excluded = [json.loads(x) for x in (WORK / "candidate/excluded.jsonl").read_text(encoding="utf-8").splitlines()]
    queue = [json.loads(x) for x in (WORK / "editorial_markup_review.jsonl").read_text(encoding="utf-8").splitlines()]
    meta = json.loads((CONVENTION / "metadata_summary.json").read_text(encoding="utf-8"))
    audit_run = json.loads((CONVENTION / "run_record.json").read_text(encoding="utf-8"))
    completeness_run = json.loads((COMPLETENESS / "run_record.json").read_text(encoding="utf-8"))
    completeness = json.loads((COMPLETENESS / "completeness_findings.json").read_text(encoding="utf-8"))["summary"]
    diff = json.loads((WORK / "v1-v2-diff/diff_summary.json").read_text(encoding="utf-8"))
    if not (audit_run["manifest_sha256"] == completeness_run["manifest_sha256"] == diff["new"]["manifest_sha256"]
            == build["manifest_sha256"]):
        print("an audit or the v1/v2 diff was not run on the current candidate", file=sys.stderr)
        return 1
    if completeness["pages_likely_uncorrected_recognition_output"] or completeness["pages_mixed_line_end_convention"]:
        print("the completeness audit still finds probable uncorrected pages", file=sys.stderr)
        return 1
    charset = default_loghi_charset()
    reason_tag = lambda r: r.split(":")[0].split(" (")[0]  # noqa: E731 -- "D4b (extended ...)" -> "D4b"
    placeholder_rows = [q for q in queue if q.get("rule") == "D4b_unbracketed_placeholder"]
    oov = Counter(c for line in lines for c in line.gt_canonical if c not in charset and c != " ")
    status = Counter()
    for c in meta["collections"].values():
        status.update(c["transkribus_page_status"])

    card = {
        "card_version": "2",
        "benchmark_id": BENCHMARK_ID,
        "supersedes": {
            "benchmark_id": SUPERSEDED.name,
            "frozen_json_sha256": sha256_file(SUPERSEDED / "FROZEN.json"),
            "manifest_sha256": sha256_file(SUPERSEDED / "manifest.jsonl"),
            "status": "SUPERSEDED - DO NOT USE FOR PRIMARY ACCURACY BENCHMARK (kept unchanged as a historical snapshot)",
            "reason": "Completeness audit found evidence that a substantial subset of pages contains uncorrected "
                      "Transkribus recognition output rather than independently corrected reference transcription; "
                      "found before any model inference.",
            "difference": {k: diff[k] for k in ("difference", "removed_lines", "added_lines", "changed_retained_lines",
                                                "retained_lines_identical", "documents_removed_entirely")},
        },
        "measures": "agreement with the supplied reference transcription (accuracy against the supplied reference), "
                    "not accuracy against an independently adjudicated or formally certified diplomatic ground truth",
        "provenance": {
            "status": "PROVENANCE UNAVAILABLE / CANNOT BE RESOLVED FROM SOURCE",
            "transcribed_by": "students, in Transkribus",
            "supplied_by": "the teacher of the students (the supplier), as a Transkribus export delivered 2026-09-27",
            "facts": [
                "The supplier has no complete documented transcription guideline and no detailed provenance trail "
                "for the individual student transcriptions.",
                "The supplier provided the best information available.",
                "No organized mechanism exists to obtain authoritative answers from the original transcribers.",
                "Undocumented conventions are not inferred; the GT is used as delivered, subject only to the "
                "pre-registered exclusions and transformations below.",
            ],
            "answered": {
                "Q1 used to train/validate/test Lion, Transkribus or Riksarkivet-derived models": "no (supplier, 2026-09-27)",
                "Q4 sample documents intended": "no (project owner, 2026-09-27); excluded by D1",
            },
            "unanswerable": [
                "Q2 what the two TRAINING_VALIDATION_SET_* collections are and which models used them "
                "(excluded regardless, D2)",
                "Q3 meaning of [???] and of unbracketed ??/??? (excluded, D4b)",
                "Q5 diplomatic vs normalized transcription; line-end hyphenation (- vs ¬); ß",
                "whether the collections were transcribed under different guidelines (audit: ':' as abbreviation "
                "mark and 'dh' spellings differ by collection; line-final ¬ turned out to be uncorrected "
                "recognition output, removed by D6)",
            ],
            "training_overlap_check": [
                "Partial: against the Svea Hovratt training subset only (40,983 lines), not the whole training corpus.",
                "No identical line images; no near-identical images (dHash threshold 4, nearest >= 8).",
                "No exact or normalized transcription overlap of 20+ characters.",
                "Word 4-gram overlap 0.1%, consistent with common legal phrasing.",
                "Same court and era, different series.",
            ],
        },
        "source": {
            "delivery_id": build["source_id"],
            "delivery_tree": build["source_tree"],
            "format": "PAGE-XML (adapter page_xml chosen explicitly, D3; ALTO files unused)",
            "adapter": build["adapter"],
            "export_jobs_in_primary": sorted({job_of(line.source_relative_path) for line in lines}),
            "transkribus_page_status": dict(status),
        },
        "counts": {
            "documents": len({line.document_id for line in lines}),
            "pages": len({(line.document_id, line.page_id) for line in lines}),
            "lines": len(lines),
            "characters": sum(len(line.gt_canonical) for line in lines),
            "words": sum(len(line.gt_canonical.split()) for line in lines),
            "lines_per_export_job": dict(sorted(Counter(job_of(line.source_relative_path) for line in lines).items())),
            "excluded_lines_by_rule": dict(sorted(Counter(reason_tag(e["reason"]) for e in excluded).items())),
            "editorial_review_queue_lines": len(queue),
            "d4b_unbracketed_placeholder_lines": len(placeholder_rows),
            "crop_clamped_lines": sum(bool(line.source_metadata.get("crop_clamped")) for line in lines),
        },
        "eligibility_rules": {
            "D1": "keep only export jobs 4502442-4502446 (transcribed Svea Hovratt); samples and collections with "
                  "no/insufficient GT excluded by page; lines without GT excluded",
            "D2": "TRAINING_VALIDATION_SET_* collections (4502439, 4502440) excluded completely",
            "D3": "PAGE-XML is authoritative",
            "D4a": "outer whitespace trimmed (gt = gt_source.strip()); nothing else changed; gt_source preserved",
            "D4b": {
                "rule": "lines with editorial markup such as [???], and lines whose delivered GT matches the regex "
                        "below (two or more consecutive '?', anywhere, including embedded forms), are excluded "
                        "from primary and held in the editorial review queue with their text unchanged",
                "regex": PLACEHOLDER,
                "examples": ["???", "??", "oppbur???", "21???"],
                "single_question_mark": "retained",
            },
            "D6": {
                "rule": "whole pages excluded as probable uncorrected Transkribus recognition output: in a source "
                        "document with at least one page whose last save followed the previous save within <= 1.0 s "
                        "per line and that carries no student convention (0 line-final '-', 0 lines with '[' or "
                        "'??'), every page with no student convention or mixing line-final '-' and '¬' is excluded",
                "pages": "export job 4502442 pages 1-31; 4502443 pages 9-12 (35 pages)",
                "lines_excluded": sum(1 for e in excluded if reason_tag(e["reason"]) == "D6"),
                "evidence": "completeness audit (scripts/benchmark_completeness_audit.py): signals T2/T3/T4",
                "confirmation": "seeded image review of 15 pages / 46 lines (image-review/REVIEW.md): suspect "
                                "lines 11 likely uncorrected, 18 ambiguous, 2 corrected (4502443 p9 above its first "
                                "¬); control lines 15 corrected, 0 otherwise",
                "partial_pages": "none; 4502443 p9 is mixed but excluded whole because no objective boundary exists",
                "gt_changed": False,
            },
            "kept_as_supplied": ["characters outside Loghi's charset", "identical transcriptions on different "
                                 "images", "crops clamped to the page raster (flagged crop_clamped)"],
            "spot_check_lines": "Convention-audit spot-check lines (single '?', unusual line-end hyphenation) "
                                "stay in primary unchanged; they are not tuned and are not reviewed "
                                "against model predictions before the primary run.",
        },
        "scoring": {
            "primary": "raw CER/WER against canonical GT",
            "D5_sensitivity": SENSITIVITY_RULES,
            "not_normalized": "ß, ';', spelling, punctuation and all other characters",
        },
        "caveats": {
            "source_status": f"All {sum(status.values())} pages are marked {', '.join(status)} in Transkribus; none "
                             "is marked GT/FINAL. The status was not changed. On the retained pages the completeness "
                             "audit finds no material deterministic evidence of incomplete transcription (inference; "
                             "not a certification).",
            "correction_status": "Every page was prefilled by a Transkribus HTR model. Pages whose signals and image "
                                 "review indicate uncorrected output are excluded (D6). "
                                 f"{completeness['pages_ambiguous_recognition_signals']} retained pages were saved at "
                                 "machine speed but carry student edits; they are retained (one was reviewed and "
                                 "reads as corrected). Residual uncorrected lines cannot be ruled out.",
            "duplicate_pages": "5 retained pages also occur, as later recognition runs, in the excluded "
                               "TRAINING_VALIDATION_SET_* export jobs; the benchmark uses the earlier, edited version.",
            "transcription_convention": "The student transcription guidelines are unavailable. The deterministic "
                                        "convention audit shows convention differences between collections. The "
                                        "supplied GT is the reference as delivered.",
            "editorial_uncertainty": f"D4b excludes lines matching {PLACEHOLDER}; a single '?' is retained; excluded "
                                     "GT text is unchanged and preserved outside the primary set.",
            "hyphenation": "Raw scoring is primary. A separately labelled secondary sensitivity score treats only "
                           "line-final ¬ and line-final - as equivalent; internal ¬ or - remain different. After D6 "
                           "no reference line ends in ¬; the rule still applies to predictions.",
            "charset": "Characters outside Loghi's output charset remain and count normally as errors if Loghi "
                       f"cannot emit them: {dict(oov)}.",
        },
        "reconstruction": {
            "decisions_file": "decisions.jsonl (copied into the frozen benchmark; sha256 = decisions_sha256)",
            "decision_generator_sha256": sha256_file(WORK / "make_decisions.py"),
            "superseded_decisions_sha256": {"decisions.v1.jsonl (before the D4b extension)":
                                            sha256_file(WORK / "decisions.v1.jsonl"),
                                            "decisions.primary-v1.jsonl (frozen with the superseded benchmark)":
                                            sha256_file(WORK / "decisions.primary-v1.jsonl")},
            "d6_page_signals_sha256": sha256_file(WORK / "d6_pages.jsonl"),
            "d6_excluded_lines_archive_sha256": sha256_file(WORK / "d6_excluded_lines.jsonl"),
            "completeness_audit": {
                "script": "scripts/benchmark_completeness_audit.py",
                "superseded_benchmark": {
                    "script_sha256": json.loads((COMPLETENESS_V1 / "run_record.json").read_text(encoding="utf-8"))["script_sha256"],
                    "recognition_provenance_sha256": sha256_file(COMPLETENESS_V1 / "recognition_provenance.csv"),
                    "findings_sha256": sha256_file(COMPLETENESS_V1 / "completeness_findings.json")},
                "this_benchmark": {"script_sha256": completeness_run["script_sha256"],
                                   "findings_sha256": sha256_file(COMPLETENESS / "completeness_findings.json"),
                                   "summary": completeness}},
            "image_review": {"script": "scripts/benchmark_image_review_sample.py",
                             "sample_sha256": sha256_file(REVIEW / "sample.csv"),
                             "assessments_sha256": sha256_file(REVIEW / "assessments.csv"),
                             "reviewer": "Claude (AI assistant), visual comparison of crops with GT; not a trained "
                                         "palaeographer"},
            "v1_v2_diff": {"script": "scripts/benchmark_manifest_diff.py",
                           "summary_sha256": sha256_file(WORK / "v1-v2-diff/diff_summary.json"),
                           "removed_lines_sha256": sha256_file(WORK / "v1-v2-diff/removed_lines.csv")},
            "editorial_review_queue_sha256": sha256_file(WORK / "editorial_markup_review.jsonl"),
            "convention_audit": {"script": "scripts/benchmark_convention_audit.py",
                                 "script_sha256": audit_run["script_sha256"],
                                 "findings_sha256": sha256_file(CONVENTION / "audit_findings.json")},
        },
        "model_runs_before_freeze": "none",
    }
    text = json.dumps(card, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if EMAIL.search(text):
        print("refusing: the card contains an e-mail address", file=sys.stderr)
        return 1
    (WORK / "dataset_card.json").write_text(text, encoding="utf-8")
    print(json.dumps(card["counts"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
