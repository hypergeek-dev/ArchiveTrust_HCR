"""Phase 15 -- Provider Segmentation Relationship Study (2026-07-15).

Measurement-only, per this phase's own charter: no clustering threshold, IoU threshold,
containment logic, Confidence Engine, or Review Center behavior is touched. Read-only over
already-persisted artifacts:

- `benchmarks/calibration_corpus_1.jsonl` (Phase 14's per-packet dataset) -- filtered here to the
  exact `(docling, tesseract_layoutparser)` provider pair, across every category, not just
  `low_iou_or_partial_capture_geometric_merge`.
- A single streaming pass over `events.jsonl` to fetch `EvidenceCreated.raw_output` /
  `bounding_box` / `page` for every evidence id these packets reference -- never a replay, never a
  provider invocation.

No embeddings or external semantic-similarity model are introduced (there is no existing semantic-
similarity capability in ArchiveTrust -- `domain/comparison/text_reconciliation.py` deliberately
has none, by Constitution). Wherever this study would want "semantic similarity," it is reported
as *unavailable*, never approximated by a model this codebase doesn't have.

Run: `python scripts/phase15_provider_segmentation_study.py`
Writes: `benchmarks/phase15_provider_segmentation_study.json`
"""

from __future__ import annotations

import difflib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = REPO_ROOT / "benchmarks" / "calibration_corpus_1.jsonl"
TELEMETRY_PATH = (
    REPO_ROOT / "archivetrust_data" / "workspaces" / "46e94f15d73347c2be79f1527e646c11"
    / "telemetry" / "events.jsonl"
)
OUTPUT_PATH = REPO_ROOT / "benchmarks" / "phase15_provider_segmentation_study.json"

PAIR = ("docling", "tesseract_layoutparser")


# -- evidence lookup (read-only, targeted) ----------------------------------------------------

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


# -- text features ------------------------------------------------------------------------------

_WORD_RE = re.compile(r"\w+", re.UNICODE)
_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")


def _clean_text(raw_output: str) -> str:
    """`EvidenceCreated.raw_output` is the provider's *entire* JSON payload as a string (e.g.
    `{"native_label": "Text", "text": "...", "table_data": null}`), not clean extracted text --
    discovered as a bug in this function during Phase 16's visual audit (2026-07-15; see
    `docs/PHASE_16_HIGH_IOU_DIVERGENCE_INVESTIGATION_2026-07-15.md` erratum). Every length/word/
    sentence/token feature in this script must go through this extractor, never raw `raw_output`
    directly, or JSON boilerplate (`text`, `null`, `table_data`, `label`/`native_label`) silently
    contaminates every text-similarity feature.
    """
    try:
        return json.loads(raw_output).get("text") or ""
    except (json.JSONDecodeError, AttributeError):
        return raw_output or ""


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    return " ".join(text.split()).lower()


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(_norm(text))


def _sentence_count(text: str) -> int:
    parts = [p for p in _SENTENCE_SPLIT_RE.split(text or "") if p.strip()]
    return max(len(parts), 1 if (text or "").strip() else 0)


def _lcs_len(a: str, b: str) -> int:
    matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
    match = matcher.find_longest_match(0, len(a), 0, len(b))
    return match.size


def _jaccard(words_a: list[str], words_b: list[str]) -> float:
    set_a, set_b = set(words_a), set(words_b)
    if not set_a and not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _bbox_area(bbox: dict) -> float:
    return max(0.0, bbox["x1"] - bbox["x0"]) * max(0.0, bbox["y1"] - bbox["y0"])


def _bbox_iou_parts(a: dict, b: dict) -> tuple[float, float, float]:
    """(intersection_area, union_area, iou) -- mirrors
    `domain/comparison/geometry.py::intersection_over_union` exactly, reimplemented here only
    because this script works from raw evidence dicts, not reconstructed `BoundingBox` domain
    objects (no pipeline/domain-model machinery is invoked in this phase)."""
    x0, y0 = max(a["x0"], b["x0"]), max(a["y0"], b["y0"])
    x1, y1 = min(a["x1"], b["x1"]), min(a["y1"], b["y1"])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_a, area_b = _bbox_area(a), _bbox_area(b)
    union = area_a + area_b - intersection
    iou = intersection / union if union > 0 else 0.0
    return intersection, union, iou


# -- dataset construction (Objective 1) ----------------------------------------------------------

def build_dataset() -> tuple[pd.DataFrame, dict]:
    corpus = [json.loads(line) for line in CORPUS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    pair_records = [r for r in corpus if tuple(sorted(r["providers"])) == PAIR]
    two_candidate = [r for r in pair_records if r["num_candidates"] == 2]
    excluded_multi = [r for r in pair_records if r["num_candidates"] != 2]

    evidence_ids = {eid for r in two_candidate for eid in r["evidence_ids"]}
    evidence = _fetch_evidence(evidence_ids)

    rows = []
    skipped_no_evidence = 0
    for r in two_candidate:
        ev_records = [evidence[eid] for eid in r["evidence_ids"] if eid in evidence]
        if len(ev_records) != 2:
            skipped_no_evidence += 1
            continue
        # Order deterministically: docling first, tesseract second.
        by_provider = {e["provider"]: e for e in ev_records}
        if set(by_provider) != set(PAIR):
            skipped_no_evidence += 1
            continue
        docling_ev, tess_ev = by_provider["docling"], by_provider["tesseract_layoutparser"]
        text_a = _clean_text(docling_ev.get("raw_output") or "")
        text_b = _clean_text(tess_ev.get("raw_output") or "")
        words_a, words_b = _words(text_a), _words(text_b)
        len_a, len_b = len(text_a), len(text_b)

        bbox_a, bbox_b = docling_ev.get("bounding_box"), tess_ev.get("bounding_box")
        geom = {}
        if bbox_a and bbox_b:
            inter, union, iou = _bbox_iou_parts(bbox_a, bbox_b)
            geom = {
                "intersection_area": inter,
                "union_area": union,
                "iou_recomputed": iou,
                "area_docling": _bbox_area(bbox_a),
                "area_tesseract": _bbox_area(bbox_b),
                "width_docling": bbox_a["x1"] - bbox_a["x0"],
                "height_docling": bbox_a["y1"] - bbox_a["y0"],
                "width_tesseract": bbox_b["x1"] - bbox_b["x0"],
                "height_tesseract": bbox_b["y1"] - bbox_b["y0"],
                "centroid_y_docling": (bbox_a["y0"] + bbox_a["y1"]) / 2,
                "centroid_y_tesseract": (bbox_b["y0"] + bbox_b["y1"]) / 2,
            }

        magnitude = r["comparison_magnitude"]
        rows.append({
            "semantic_slot_id": r["semantic_slot_id"],
            "document_ref": r["document_ref"],
            "category": r["category"],
            "page": (r["pages"] or [None])[0],
            "iou": r["pairwise_iou"],
            "containment_ratio_stored": r["containment_ratio"],
            "comparison_magnitude": magnitude,
            "normalized_edit_distance": (1 - magnitude) if magnitude is not None else None,
            "length_docling": len_a,
            "length_tesseract": len_b,
            "length_ratio": (max(len_a, len_b) / min(len_a, len_b)) if min(len_a, len_b) > 0 else None,
            "docling_is_longer": len_a >= len_b,
            "word_count_docling": len(words_a),
            "word_count_tesseract": len(words_b),
            "word_count_ratio": (
                max(len(words_a), len(words_b)) / min(len(words_a), len(words_b))
                if min(len(words_a), len(words_b)) > 0 else None
            ),
            "sentence_count_docling": _sentence_count(text_a),
            "sentence_count_tesseract": _sentence_count(text_b),
            "lcs_len": _lcs_len(_norm(text_a), _norm(text_b)),
            "lcs_ratio_max": (
                _lcs_len(_norm(text_a), _norm(text_b)) / max(len(_norm(text_a)), len(_norm(text_b)))
                if max(len(_norm(text_a)), len(_norm(text_b))) > 0 else None
            ),
            "shared_token_jaccard": _jaccard(words_a, words_b),
            "semantic_similarity": None,  # explicitly unavailable -- see module docstring
            **geom,
        })

    df = pd.DataFrame(rows)
    meta = {
        "total_docling_tesseract_records_all_num_candidates": len(pair_records),
        "two_candidate_records": len(two_candidate),
        "excluded_multi_candidate_records": len(excluded_multi),
        "excluded_multi_candidate_num_candidates_distribution": dict(Counter(r["num_candidates"] for r in excluded_multi)),
        "rows_with_full_evidence_and_geometry": len(df),
        "rows_skipped_missing_evidence_or_mixed_provider": skipped_no_evidence,
        "semantic_similarity_capability": "UNAVAILABLE -- no embedding/semantic-similarity model "
        "exists anywhere in ArchiveTrust (domain/comparison/text_reconciliation.py deliberately "
        "has none: 'No semantic/embedding similarity anywhere in this alignment -> deterministic "
        "consensus selection'). Not approximated here; reported as a genuine gap (Objective 7).",
    }
    return df, meta


# -- Objective 2: relationship analysis -----------------------------------------------------------

def relationship_analysis(df: pd.DataFrame) -> dict:
    pairs = [
        ("iou", "comparison_magnitude"),
        ("iou", "normalized_edit_distance"),
        ("containment_ratio_stored", "comparison_magnitude"),
        ("length_ratio", "comparison_magnitude"),
        ("word_count_ratio", "comparison_magnitude"),
        ("shared_token_jaccard", "comparison_magnitude"),
        ("lcs_ratio_max", "comparison_magnitude"),
        ("iou", "length_ratio"),
    ]
    results = {}
    for x, y in pairs:
        sub = df[[x, y]].dropna()
        if len(sub) < 3:
            results[f"{x}__vs__{y}"] = {"n": len(sub), "note": "insufficient data"}
            continue
        pearson_r, pearson_p = stats.pearsonr(sub[x], sub[y])
        spearman_r, spearman_p = stats.spearmanr(sub[x], sub[y])
        results[f"{x}__vs__{y}"] = {
            "n": len(sub),
            "pearson_r": round(float(pearson_r), 4),
            "pearson_p": round(float(pearson_p), 6),
            "spearman_r": round(float(spearman_r), 4),
            "spearman_p": round(float(spearman_p), 6),
            "linear_vs_monotonic_gap": round(abs(float(spearman_r) - float(pearson_r)), 4),
        }
    return results


# -- Objective 3: population discovery --------------------------------------------------------

def population_discovery(df: pd.DataFrame) -> dict:
    feature_cols = ["iou", "containment_ratio_stored", "comparison_magnitude", "length_ratio", "shared_token_jaccard"]
    sub = df[feature_cols].dropna()
    X = StandardScaler().fit_transform(sub.values)

    silhouettes = {}
    for k in range(2, 7):
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        score = silhouette_score(X, km.labels_)
        silhouettes[k] = round(float(score), 4)
    best_k = max(silhouettes, key=silhouettes.get)

    kmeans_final = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(X)
    sub_labeled = sub.copy()
    sub_labeled["cluster"] = kmeans_final.labels_
    cluster_profiles = {
        int(c): {
            "n": int((sub_labeled["cluster"] == c).sum()),
            **{col: round(float(sub_labeled[sub_labeled["cluster"] == c][col].mean()), 4) for col in feature_cols},
        }
        for c in sorted(sub_labeled["cluster"].unique())
    }

    agglo = AgglomerativeClustering(n_clusters=best_k).fit(X)
    agglo_silhouette = round(float(silhouette_score(X, agglo.labels_)), 4)

    dbscan = DBSCAN(eps=1.0, min_samples=10).fit(X)
    n_dbscan_clusters = len(set(dbscan.labels_) - {-1})
    n_dbscan_noise = int((dbscan.labels_ == -1).sum())

    return {
        "n_rows_used": len(sub),
        "features_used": feature_cols,
        "kmeans_silhouette_by_k": silhouettes,
        "best_k_by_silhouette": int(best_k),
        "kmeans_cluster_profiles": cluster_profiles,
        "agglomerative_silhouette_at_best_k": agglo_silhouette,
        "dbscan_eps1.0_min_samples10": {
            "n_clusters_found": n_dbscan_clusters,
            "n_noise_points": n_dbscan_noise,
            "pct_noise": round(100 * n_dbscan_noise / len(sub), 1),
        },
        "interpretation": (
            f"Silhouette scores are low across all k ({min(silhouettes.values())}-{max(silhouettes.values())}, "
            "on a [-1,1] scale where >0.5 indicates strong separation) -- there is no strongly "
            "separated multi-cluster structure in this feature space; k-means finds a best-fit "
            f"partition at k={best_k} but the clusters are not cleanly divided. DBSCAN's high "
            "noise fraction (see above) corroborates a lack of dense, well-separated populations. "
            "Interpret cluster profiles as gradients along which the population continuously "
            "varies, not as sharply distinct populations."
        ),
    }


# -- Objective 4: provider behaviour ----------------------------------------------------------

def provider_behaviour(df: pd.DataFrame) -> dict:
    n = len(df)
    docling_longer = int(df["docling_is_longer"].sum())
    return {
        "n_pairs": n,
        "docling_is_longer_candidate": docling_longer,
        "docling_is_longer_pct": round(100 * docling_longer / n, 1),
        "tesseract_is_longer_candidate": n - docling_longer,
        "tesseract_is_longer_pct": round(100 * (n - docling_longer) / n, 1),
        "mean_length_docling": round(float(df["length_docling"].mean()), 1),
        "mean_length_tesseract": round(float(df["length_tesseract"].mean()), 1),
        "median_length_docling": float(df["length_docling"].median()),
        "median_length_tesseract": float(df["length_tesseract"].median()),
        "mean_word_count_docling": round(float(df["word_count_docling"].mean()), 1),
        "mean_word_count_tesseract": round(float(df["word_count_tesseract"].mean()), 1),
        "mean_sentence_count_docling": round(float(df["sentence_count_docling"].mean()), 2),
        "mean_sentence_count_tesseract": round(float(df["sentence_count_tesseract"].mean()), 2),
        "mean_area_docling": round(float(df["area_docling"].mean()), 1) if "area_docling" in df else None,
        "mean_area_tesseract": round(float(df["area_tesseract"].mean()), 1) if "area_tesseract" in df else None,
        "interpretation": (
            "docling is the longer/larger candidate in the large majority of pairs (see "
            "docling_is_longer_pct) with a higher mean character/word/sentence count -- consistent "
            "with docling producing more aggressively merged, multi-sentence paragraph blocks, "
            "while tesseract_layoutparser produces more granular, shorter per-line/per-sentence "
            "blocks. This is a distributional tendency, not a rule that holds for every pair."
        ),
    }


# -- Objective 5: candidate decision boundaries -------------------------------------------------

def decision_boundaries(df: pd.DataFrame) -> dict:
    """A shallow, interpretable decision tree is used purely as a boundary-search tool (it finds
    axis-aligned thresholds directly from the data) -- not as a proposed policy. Target label is a
    proxy for "likely same content" derived from the engine's own comparison_magnitude at its
    documented 0.5 similar-content cutoff (`classify_packet`), the only threshold already in
    production use; no new threshold is invented to build the target itself.
    """
    feature_cols = ["iou", "containment_ratio_stored", "length_ratio", "shared_token_jaccard", "lcs_ratio_max"]
    sub = df[feature_cols + ["comparison_magnitude"]].dropna()
    y = (sub["comparison_magnitude"] >= 0.5).astype(int)
    X = sub[feature_cols]

    if y.nunique() < 2:
        return {"note": "target label has only one class in this population; no boundary search possible"}

    positive_n = int(y.sum())
    base = {
        "n": len(sub),
        "positive_class_definition": "comparison_magnitude >= 0.5 (the engine's own existing "
        "'similar_content' cutoff, per classify_packet) -- a proxy target, not a new threshold",
        "positive_class_count": positive_n,
        "positive_class_pct": round(100 * float(y.mean()), 1),
    }

    if positive_n < 10:
        # A tree with a sane min_samples_leaf cannot isolate a positive class this rare -- it
        # would either collapse to a trivial always-negative predictor (spuriously high accuracy
        # via class imbalance) or overfit on single points if min_samples_leaf were dropped to
        # match. Neither is a meaningful "boundary" -- report the positive examples directly
        # instead of manufacturing a tree.
        positive_rows = sub[y == 1][feature_cols].round(4).to_dict(orient="records")
        negative_means = {col: round(float(sub[y == 0][col].mean()), 4) for col in feature_cols}
        base.update({
            "note": (
                f"Positive class too rare (n={positive_n}) for a statistically meaningful decision "
                "tree at any reasonable min_samples_leaf -- a tree would either trivially predict "
                "the majority class (spuriously high accuracy from class imbalance) or overfit "
                "individual points. Reporting the positive examples directly instead."
            ),
            "positive_examples": positive_rows,
            "negative_class_means": negative_means,
        })
        return base

    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=42)
    tree.fit(X, y)
    train_accuracy = round(float(tree.score(X, y)), 4)
    base.update({
        "tree_train_accuracy": train_accuracy,
        "tree_rules": export_text(tree, feature_names=feature_cols),
        "feature_importances": {
            col: round(float(imp), 4) for col, imp in zip(feature_cols, tree.feature_importances_)
        },
        "interpretation": (
            f"Train accuracy of {train_accuracy} on the tree's own training data (no held-out "
            "split -- population is too small and this is boundary discovery, not a validated "
            "classifier) shows how separable the classes are given these features. See "
            "tree_rules for the actual candidate thresholds found."
        ),
    })
    return base


def containment_boundary_search(df: pd.DataFrame) -> dict:
    """A second, better-populated boundary search: can geometry + length signals (deliberately
    excluding containment/LCS themselves) predict a "likely partial-capture" pair (containment
    >= 0.3, matching the same magnitude the clustering engine already treats as its join
    threshold, applied here to a different metric only for a like-for-like comparison point) --
    i.e. is partial capture geometrically/length-wise predictable without computing containment?
    """
    feature_cols = ["iou", "length_ratio", "word_count_ratio", "shared_token_jaccard"]
    sub = df[feature_cols + ["containment_ratio_stored"]].dropna()
    y = (sub["containment_ratio_stored"] >= 0.3).astype(int)
    X = sub[feature_cols]
    positive_n = int(y.sum())
    if positive_n < 10 or y.nunique() < 2:
        return {"note": f"positive class too small (n={positive_n}) for this boundary search"}

    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=8, random_state=42)
    tree.fit(X, y)
    return {
        "n": len(sub),
        "positive_class_definition": "containment_ratio_stored >= 0.3 (likely partial-capture pair)",
        "positive_class_count": positive_n,
        "positive_class_pct": round(100 * float(y.mean()), 1),
        "tree_train_accuracy": round(float(tree.score(X, y)), 4),
        "tree_rules": export_text(tree, feature_names=feature_cols),
        "feature_importances": {
            col: round(float(imp), 4) for col, imp in zip(feature_cols, tree.feature_importances_)
        },
    }


def main() -> None:
    df, meta = build_dataset()
    results = {
        "meta": meta,
        "objective2_relationship_analysis": relationship_analysis(df),
        "objective3_population_discovery": population_discovery(df),
        "objective4_provider_behaviour": provider_behaviour(df),
        "objective5_decision_boundaries": decision_boundaries(df),
        "objective5_containment_boundary_search": containment_boundary_search(df),
    }
    OUTPUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    df.to_csv(REPO_ROOT / "benchmarks" / "phase15_feature_dataset.csv", index=False)
    print(f"Wrote {OUTPUT_PATH}")
    print(f"Wrote {REPO_ROOT / 'benchmarks' / 'phase15_feature_dataset.csv'}")
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print(json.dumps(results["objective4_provider_behaviour"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
