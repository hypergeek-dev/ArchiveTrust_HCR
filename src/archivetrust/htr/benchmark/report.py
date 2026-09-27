"""Score a completed run (both prediction files) and write the comparison report.

    reports/<run_id>/predictions/{loghi,lion}.jsonl  ->  reports/<run_id>/scores.json,
                                                         line_scores_<model>.jsonl, report.md
"""

from __future__ import annotations

import json
from pathlib import Path

from archivetrust.evaluation.metrics import METRICS_VERSION
from archivetrust.htr.benchmark.build import verify_frozen
from archivetrust.htr.benchmark.contract import sha256_file
from archivetrust.htr.benchmark.inference import read_predictions
from archivetrust.htr.benchmark.model_registry import LOGHI, load_loghi_charset
from archivetrust.htr.benchmark.normalization import protocol_record
from archivetrust.htr.benchmark.pairwise import compare
from archivetrust.htr.benchmark.provenance import label, provenance, utc_now
from archivetrust.htr.benchmark.scoring import (
    SCORING_VERSION,
    ScoringError,
    aggregate,
    bootstrap,
    breakdown,
    check_alignment,
    line_score_rows,
    score_line,
)


def score_run(frozen_dir: Path, report_dir: Path, *, prediction_files: dict[str, str] | None = None, official: bool = False,
              loghi_charset: set[str] | None = None) -> dict:
    """`prediction_files` maps a short name to a file in `report_dir/predictions/` (default: loghi.jsonl
    and lion.jsonl). Refuses to score anything that does not cover the frozen manifest exactly."""
    frozen_dir, report_dir = Path(frozen_dir), Path(report_dir)
    record, lines, findings = verify_frozen(frozen_dir)
    if findings:
        raise ScoringError(f"frozen benchmark failed verification: {[f.code for f in findings][:5]}")
    files = prediction_files or {"loghi": "loghi.jsonl", "lion": "lion.jsonl"}
    by_id = {line.line_id: line for line in lines}
    per_model, runs = {}, {}
    for name, filename in files.items():
        path = report_dir / "predictions" / filename
        predictions = read_predictions(path)
        check_alignment(lines, predictions, manifest_sha256=record["manifest_sha256"])
        per_model[name] = [score_line(by_id[p.line_id], p) for p in sorted(predictions, key=lambda p: p.line_id)]
        run_path = path.with_name(path.stem + ".run.json")
        runs[name] = json.loads(run_path.read_text(encoding="utf-8")) if run_path.is_file() else None
        if runs[name] and runs[name]["predictions_sha256"] != sha256_file(path):
            raise ScoringError(f"{path} does not match the hash in its run record")
        (report_dir / f"line_scores_{name}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in line_score_rows(per_model[name])), encoding="utf-8")

    scores = {
        "scoring_version": SCORING_VERSION,
        "metrics_version": METRICS_VERSION,
        "normalization": protocol_record(),
        "benchmark": {k: record[k] for k in ("benchmark_id", "dataset_id", "manifest_sha256", "lines", "documents", "pages",
                                             "reference_characters", "crop_policies", "excluded_by_decision",
                                             "excluded_unresolved_at_freeze", "frozen_at_utc")},
        "benchmark_frozen_official": record["provenance"]["official"],
        "prediction_files": {name: {"file": files[name], "sha256": sha256_file(report_dir / "predictions" / files[name])} for name in files},
        "runs": runs,
        "corpus": {name: aggregate(s) for name, s in per_model.items()},
        "bootstrap": bootstrap(per_model, lines),
        "by_collection": {name: breakdown(s, "collection") for name, s in per_model.items()},
        "by_document": {name: breakdown(s, "document_id") for name, s in per_model.items()},
        "scoring_provenance": provenance(official=official),
        "scored_at_utc": utc_now(),
    }
    names = sorted(per_model)
    if len(names) == 2:
        scores["pairwise"] = compare(names[0], per_model[names[0]], names[1], per_model[names[1]])
    if loghi_charset is not None:
        chars = [c for line in lines for c in line.gt_canonical]
        oov = [c for c in chars if c not in loghi_charset]
        scores["loghi_out_of_vocabulary"] = {"occurrences": len(oov), "rate": len(oov) / len(chars) if chars else None,
                                             "characters": sorted(set(oov))}
    (report_dir / "scores.json").write_text(json.dumps(scores, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (report_dir / "report.md").write_text(render_report(scores), encoding="utf-8")
    return scores


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2%}"


def _ci(interval: list[float] | None) -> str:
    return "n/a" if not interval else f"[{interval[0]:.2%}, {interval[1]:.2%}]"


def render_report(scores: dict) -> str:
    b = scores["benchmark"]
    runs = scores["runs"]
    boot = scores["bootstrap"]
    corpus = scores["corpus"]
    official = scores["scoring_provenance"]["official"] and scores["benchmark_frozen_official"] and all(
        (r or {}).get("provenance", {}).get("official") for r in runs.values())
    out = [
        f"# HTR benchmark: {' vs '.join(sorted(corpus))} on `{b['benchmark_id']}`",
        "",
        f"**Status: {'OFFICIAL' if official else label(scores['scoring_provenance']) + ' -- not all stages were official runs'}.** "
        "Pre-adaptation results: the benchmark was frozen before any model output was seen.",
        "",
        "## Benchmark identity",
        "",
        f"- Dataset `{b['dataset_id']}`, {b['lines']} lines, {b['pages']} pages, {b['documents']} documents, "
        f"{b['reference_characters']} reference characters; crop policy {b['crop_policies']}.",
        f"- Manifest SHA-256 `{b['manifest_sha256']}`; frozen {b['frozen_at_utc']}.",
        f"- Excluded before freeze: {b['excluded_by_decision']} by recorded decision, {b['excluded_unresolved_at_freeze']} unresolved at freeze.",
        f"- GT normalization `{scores['normalization']['protocol_id']}` (rules: {', '.join(scores['normalization']['rules'])}); "
        f"predictions: {', '.join(scores['normalization']['prediction_rules'])}. Nothing else.",
        "",
        "## Models",
        "",
    ]
    for name, run in sorted(runs.items()):
        if run:
            m = run["model"]
            out.append(f"- **{name}**: `{m['model_id']}`, decoding profile `{m['decoding_profile']}` {json.dumps(m['decoding'])}; "
                       f"predictions `{run['predictions_sha256'][:16]}...`, status {run['status_counts']}, {label(run['provenance'])}.")
        else:
            out.append(f"- **{name}**: no run record found (predictions file hash {scores['prediction_files'][name]['sha256'][:16]}...).")
    out += [
        "",
        "## Headline (corpus-level; every line counted, failures scored as empty output)",
        "",
        f"Confidence intervals: {boot['interval']} cluster bootstrap over {boot['clusters']} {boot['cluster_unit']}s, "
        f"{boot['samples']} resamples, seed {boot['seed']}.",
        "",
        "| Model | CER | 95% CI | WER | 95% CI | CER (ws-norm) | Exact lines | S / I / D chars | Failed+missing | Empty |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, c in sorted(corpus.items()):
        ci = boot["per_model"][name]
        out.append(f"| {name} | {_pct(c['cer'])} | {_ci(ci['cer'])} | {_pct(c['wer'])} | {_ci(ci['wer'])} | "
                   f"{_pct(c['cer_whitespace_normalized'])} | {_pct(c['exact_line_accuracy'])} | "
                   f"{c['char_substitutions']} / {c['char_insertions']} / {c['char_deletions']} | "
                   f"{c['failed_or_missing_lines']} | {c['status_counts'].get('empty', 0)} |")
    if boot.get("warning"):
        out += ["", f"> Warning: {boot['warning']}"]
    for pair, entry in boot["paired_differences"].items():
        out += ["", f"**Paired difference ({entry['cer']['difference']})**: CER 95% CI {_ci(entry['cer']['ci95'])}, "
                f"WER 95% CI {_ci(entry['wer']['ci95'])}. An interval that contains 0 means the benchmark does not "
                "separate the models at this size."]
    if "pairwise" in scores:
        p = scores["pairwise"]
        out += ["", "## Line-level comparison", "", "| Category | Lines |", "|---|---|"]
        out += [f"| {k} | {v} |" for k, v in p["categories"].items()]
        out += ["", f"Catastrophic lines (failed/missing or line CER >= 50%): {p['catastrophic']}.", "",
                "### Error types (character edits)", "", "| Tag | " + " | ".join(p["error_profile"]) + " |",
                "|---|" + "---|" * len(p["error_profile"])]
        tags = sorted({t for prof in p["error_profile"].values() for t in prof["tags"]})
        out += [f"| {t} | " + " | ".join(str(prof["tags"].get(t, 0)) for prof in p["error_profile"].values()) + " |" for t in tags]
        for name, prof in p["error_profile"].items():
            out += ["", f"Top confusions, {name}: " + ", ".join(f"`{c}` x{n}" for c, n in prof["top_confusions"][:15])]
        for title, rows in p["disagreement_sample"].items():
            out += ["", f"### Largest disagreements: {title} (top 10 of {len(rows)} in scores.json)", ""]
            for r in rows[:10]:
                out.append(f"- `{r['line_id']}` REF `{r['reference']}` | " + " | ".join(
                    f"{k} `{v}`" for k, v in r.items() if k not in ("line_id", "reference")))
    if len(scores["by_collection"].get(next(iter(corpus)), {})) > 1:
        out += ["", "## By collection", "", "| Collection | " + " | ".join(f"{n} CER" for n in sorted(corpus)) + " | Lines |",
                "|---|" + "---|" * (len(corpus) + 1)]
        first = sorted(corpus)[0]
        for coll, agg in scores["by_collection"][first].items():
            out.append(f"| {coll} | " + " | ".join(_pct(scores["by_collection"][n][coll]["cer"]) for n in sorted(corpus)) + f" | {agg['lines']} |")
    oov = scores.get("loghi_out_of_vocabulary")
    out += [
        "",
        "## Caveats that bound these numbers",
        "",
        f"- Loghi's reference validation CER ({LOGHI.pinned['reference_validation']['cer']:.2%}) was measured in-distribution; it is not "
        "comparable to this external benchmark.",
        "- Both models were trained on Riksarkivet's public HF line collections. Overlap between this benchmark and that material "
        "must be checked with the `overlap` command; no detected overlap is not proof of none.",
        f"- Loghi cannot emit characters outside its 124-character set: "
        + ("not computed." if not oov else f"{oov['occurrences']} reference characters ({_pct(oov['rate'])}) are out of its vocabulary: "
           f"{''.join(oov['characters'])}."),
        "- Loghi durations are batch-amortized; Lion's are per line. Timing is not a like-for-like comparison.",
        "- Only Loghi reports a per-line confidence.",
    ]
    return "\n".join(out) + "\n"


def default_loghi_charset() -> set[str] | None:
    try:
        return load_loghi_charset()
    except OSError:
        return None
