"""Phase 17 -- Containment Signal Validation (2026-07-15).

Measurement-only, per this phase's own charter: no clustering threshold, comparison rule,
confidence logic, or Review Center behavior is touched. Read-only over the complete regenerated
review corpus (`benchmarks/calibration_corpus_1.jsonl`, 522 packets, every provider pair, not just
the docling/tesseract high-IoU population Phases 14-16 focused on) plus a targeted evidence lookup
for clean candidate text (see Phase 16's erratum: `raw_output` is a JSON payload string and must be
unwrapped via `_clean_text()`, never read directly).

Run: `python scripts/phase17_containment_signal_validation.py`
Writes: `benchmarks/phase17_containment_signal_validation.json`,
        `benchmarks/phase17_containment_features.csv`
"""

from __future__ import annotations

import difflib
import itertools
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = REPO_ROOT / "benchmarks" / "calibration_corpus_1.jsonl"
TELEMETRY_PATH = (
    REPO_ROOT / "archivetrust_data" / "workspaces" / "46e94f15d73347c2be79f1527e646c11"
    / "telemetry" / "events.jsonl"
)
OUTPUT_JSON = REPO_ROOT / "benchmarks" / "phase17_containment_signal_validation.json"
OUTPUT_CSV = REPO_ROOT / "benchmarks" / "phase17_containment_features.csv"

_WORD_RE = re.compile(r"\w+", re.UNICODE)
_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")


def _clean_text(raw_output: str) -> str:
    """See Phase 16 erratum: `raw_output` is the provider's whole JSON payload, not clean text."""
    try:
        return json.loads(raw_output).get("text") or ""
    except (json.JSONDecodeError, AttributeError):
        return raw_output or ""


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    return " ".join(text.split()).lower()


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(_norm(text))


def _sentences(text: str) -> list[str]:
    return [_norm(p) for p in _SENTENCE_SPLIT_RE.split(text or "") if p.strip()]


def _lcs_len(a: str, b: str) -> int:
    matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
    match = matcher.find_longest_match(0, len(a), 0, len(b))
    return match.size


def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _pair_metrics(text_a: str, text_b: str) -> dict:
    na, nb = _norm(text_a), _norm(text_b)
    lcs = _lcs_len(na, nb)
    forward = lcs / len(na) if na else 0.0   # fraction of A found (as one run) in B
    reverse = lcs / len(nb) if nb else 0.0   # fraction of B found (as one run) in A
    wa, wb = _words(text_a), _words(text_b)
    sa, sb = _sentences(text_a), _sentences(text_b)
    sent_forward = (sum(1 for s in sa if s in sb) / len(sa)) if sa else 0.0
    sent_reverse = (sum(1 for s in sb if s in sa) / len(sb)) if sb else 0.0
    return {
        "forward_containment": round(forward, 4),
        "reverse_containment": round(reverse, 4),
        "lcs_ratio_max": round(lcs / max(len(na), len(nb)), 4) if max(len(na), len(nb)) else 0.0,
        "shared_token_jaccard": round(_jaccard(wa, wb), 4),
        "sentence_containment_forward": round(sent_forward, 4),
        "sentence_containment_reverse": round(sent_reverse, 4),
        "paragraph_similarity_ratio": round(difflib.SequenceMatcher(None, na, nb).ratio(), 4),
    }


def _fetch_evidence(evidence_ids: set[str]) -> dict[str, dict]:
    found: dict[str, dict] = {}
    with TELEMETRY_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            if '"kind":"EvidenceCreated"' not in line:
                continue
            if not any(eid in line for eid in evidence_ids):
                continue
            obj = json.loads(line)
            evidence = obj.get("evidence")
            if not evidence:
                continue
            eid = evidence.get("evidence_id")
            if eid in evidence_ids:
                found[eid] = evidence
            if len(found) == len(evidence_ids):
                break
    return found


def build_dataset() -> tuple[pd.DataFrame, dict]:
    corpus = [json.loads(line) for line in CORPUS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    evidence_ids = {eid for r in corpus for eid in r["evidence_ids"]}
    evidence = _fetch_evidence(evidence_ids)

    rows = []
    multi_candidate_skipped = 0
    for r in corpus:
        ev_records = [evidence[eid] for eid in r["evidence_ids"] if eid in evidence]
        by_provider: dict[str, str] = {}
        for e in ev_records:
            by_provider.setdefault(e["provider"], _clean_text(e.get("raw_output") or ""))
        providers = sorted(by_provider)
        if len(providers) < 2:
            multi_candidate_skipped += 1
            continue

        # For >2 distinct providers, compute every pairwise combination and take the
        # maximum-forward-containment pair as the packet-level signal (documented, not hidden):
        # the question this phase asks is "does containment exist between *any* two candidates,"
        # since that is the case a reconciliation rule would need to detect.
        best = None
        for p1, p2 in itertools.combinations(providers, 2):
            m = _pair_metrics(by_provider[p1], by_provider[p2])
            m["pair"] = f"{p1}/{p2}"
            if best is None or max(m["forward_containment"], m["reverse_containment"]) > max(
                best["forward_containment"], best["reverse_containment"]
            ):
                best = m

        rows.append({
            "semantic_slot_id": r["semantic_slot_id"],
            "document_ref": r["document_ref"],
            "category": r["category"],
            "num_candidates": r["num_candidates"],
            "num_distinct_providers": len(providers),
            "provider_pair": best["pair"],
            "iou": r["pairwise_iou"],
            "comparison_magnitude": r["comparison_magnitude"],
            "canonical_confidence": r["canonical_confidence"],
            "observation_type": r["observation_type"],
            "candidate_lengths": r["candidate_lengths"],
            "length_ratio": (
                max(r["candidate_lengths"]) / min(r["candidate_lengths"])
                if r["candidate_lengths"] and min(r["candidate_lengths"]) > 0 else None
            ),
            "containment_ratio_stored_phase14": r["containment_ratio"],
            **best,
        })

    df = pd.DataFrame(rows)
    meta = {
        "corpus_total": len(corpus),
        "rows_built": len(df),
        "multi_candidate_or_missing_evidence_skipped": multi_candidate_skipped,
        "provider_pair_distribution": dict(Counter(df["provider_pair"])),
        "note_multi_provider_packets": (
            "Packets with 3 distinct providers use the pairwise combination with the highest "
            "max(forward, reverse) containment as the packet-level signal, not an average -- "
            "documented, not hidden (see build_dataset())."
        ),
    }
    return df, meta


def containment_statistics(df: pd.DataFrame) -> dict:
    max_containment = df[["forward_containment", "reverse_containment"]].max(axis=1)
    df = df.assign(max_containment=max_containment)
    bins = [0, 0.10, 0.30, 0.50, 0.70, 0.90, 1.0001]
    labels = ["0-10%", "10-30%", "30-50%", "50-70%", "70-90%", "90-100%"]
    df = df.assign(containment_bin=pd.cut(df["max_containment"], bins=bins, labels=labels, right=False))
    bin_counts = df["containment_bin"].value_counts().reindex(labels).to_dict()
    return {
        "n": len(df),
        "forward_containment_describe": df["forward_containment"].describe().round(4).to_dict(),
        "reverse_containment_describe": df["reverse_containment"].describe().round(4).to_dict(),
        "max_containment_bin_counts": {k: int(v) for k, v in bin_counts.items()},
        "sentence_containment_forward_describe": df["sentence_containment_forward"].describe().round(4).to_dict(),
        "paragraph_similarity_ratio_describe": df["paragraph_similarity_ratio"].describe().round(4).to_dict(),
    }, df


def relationship_analysis(df: pd.DataFrame) -> dict:
    df = df.assign(max_containment=df[["forward_containment", "reverse_containment"]].max(axis=1))
    numeric_pairs = [
        ("max_containment", "iou"),
        ("max_containment", "length_ratio"),
        ("max_containment", "comparison_magnitude"),
        ("max_containment", "canonical_confidence"),
    ]
    results = {}
    for x, y in numeric_pairs:
        sub = df[[x, y]].dropna()
        if len(sub) < 3:
            results[f"{x}__vs__{y}"] = {"n": len(sub), "note": "insufficient data"}
            continue
        pr, pp = stats.pearsonr(sub[x], sub[y])
        sr, sp = stats.spearmanr(sub[x], sub[y])
        results[f"{x}__vs__{y}"] = {
            "n": len(sub), "pearson_r": round(float(pr), 4), "pearson_p": round(float(pp), 6),
            "spearman_r": round(float(sr), 4), "spearman_p": round(float(sp), 6),
        }

    # By provider pair -- interaction effect check.
    by_pair = df.groupby("provider_pair")["max_containment"].agg(["count", "mean", "median"]).round(4)
    results["max_containment_by_provider_pair"] = by_pair.to_dict(orient="index")

    # By observation type (all are 'paragraph' in this corpus -- reported honestly).
    results["observation_types_present"] = dict(Counter(df["observation_type"]))
    return results


def information_gain(df: pd.DataFrame) -> dict:
    """Compares three feature sets' ability to predict the engine's own 'similar content'
    proxy (`comparison_magnitude >= 0.5`, the same cutoff `classify_packet` already uses) via
    5-fold cross-validated logistic regression AUC -- not because AUC is the final word on
    "usefulness," but because it directly answers Step 6's question: does containment add
    independent predictive value beyond geometry and length alone?
    """
    df = df.assign(max_containment=df[["forward_containment", "reverse_containment"]].max(axis=1))
    sub = df[["iou", "length_ratio", "max_containment", "comparison_magnitude"]].dropna()

    target_note = "comparison_magnitude >= 0.5 (classify_packet's own 'similar content' cutoff)"
    y = (sub["comparison_magnitude"] >= 0.5).astype(int)
    if int(y.sum()) < 20:
        # Only 3 of 392 packets in this iou-available subset ever reach the engine's own 0.5
        # 'similar content' cutoff -- far too few for a cross-validated comparison. Fall back to
        # a population-derived threshold (top quartile of this subset's own magnitude
        # distribution) so the positive class is well populated without inventing a new,
        # unmotivated number.
        threshold = float(sub["comparison_magnitude"].quantile(0.75))
        target_note = f"comparison_magnitude >= {threshold:.4f} (top quartile of this subset, since only " \
            f"{int((sub['comparison_magnitude'] >= 0.5).sum())} packets reach the engine's 0.5 cutoff)"
        y = (sub["comparison_magnitude"] >= threshold).astype(int)

    if y.sum() < 5 or y.nunique() < 2:
        return {
            "note": f"positive class too small (n={int(y.sum())}) for cross-validated AUC even after "
            "fallback; reporting positive/negative feature means instead",
            "target_definition": target_note,
            "n": len(sub),
            "positive_class_count": int(y.sum()),
            "positive_means": sub[y == 1][["iou", "length_ratio", "max_containment"]].mean().round(4).to_dict(),
            "negative_means": sub[y == 0][["iou", "length_ratio", "max_containment"]].mean().round(4).to_dict(),
        }

    feature_sets = {
        "geometry_only": ["iou"],
        "geometry_plus_length": ["iou", "length_ratio"],
        "geometry_plus_length_plus_containment": ["iou", "length_ratio", "max_containment"],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {
        "n": len(sub),
        "target_definition": target_note,
        "positive_class_count": int(y.sum()),
        "positive_class_pct": round(100 * float(y.mean()), 2),
    }
    for name, cols in feature_sets.items():
        X = sub[cols].values
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        try:
            scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
            results[name] = {"mean_auc": round(float(np.mean(scores)), 4), "std_auc": round(float(np.std(scores)), 4)}
        except ValueError as exc:
            results[name] = {"error": str(exc)}
    return results


def main() -> None:
    df, meta = build_dataset()
    stats_result, df_binned = containment_statistics(df)
    results = {
        "meta": meta,
        "step1_containment_statistics": stats_result,
        "step5_relationship_analysis": relationship_analysis(df),
        "step6_information_gain": information_gain(df),
    }
    OUTPUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    df_binned.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_CSV}")
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print(json.dumps(stats_result["max_containment_bin_counts"], indent=2))
    print(json.dumps(results["step6_information_gain"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
