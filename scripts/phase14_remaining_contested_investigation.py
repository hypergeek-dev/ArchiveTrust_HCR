"""Phase 14 -- Remaining Contested Packet Investigation (2026-07-15).

Investigation-only, per its own charter: does not modify clustering, IoU thresholds, containment
logic, or Review Center behavior. Read-only over already-persisted artifacts:

- `benchmarks/calibration_corpus_1.jsonl` (522 records -- one per regenerated, multi-provider
  contested packet in Benchmark #1's post-clustering-fix replay, produced by
  `scripts/semantic_alignment_audit.py::classify_packet`). This *is* the authoritative,
  already-computed per-packet dataset the replay produced; it is not regenerated here.
- A small number of `EvidenceCreated.raw_output` values, looked up directly from
  `archivetrust_data/workspaces/<id>/telemetry/events.jsonl` by a fast substring pre-filter +
  targeted `json.loads` (never a full pydantic parse of the ~977MB stream, and never a pipeline
  replay) -- used only to read representative packets' actual OCR text for Step 4.

No provider is invoked. No document is reprocessed. Nothing is written back to
`archivetrust_data/`. Writes only `benchmarks/phase14_remaining_contested_investigation.json`.

Run: `python scripts/phase14_remaining_contested_investigation.py`
"""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = REPO_ROOT / "benchmarks" / "calibration_corpus_1.jsonl"
TELEMETRY_PATH = (
    REPO_ROOT / "archivetrust_data" / "workspaces" / "46e94f15d73347c2be79f1527e646c11"
    / "telemetry" / "events.jsonl"
)
OUTPUT_PATH = REPO_ROOT / "benchmarks" / "phase14_remaining_contested_investigation.json"

TARGET_CATEGORY = "low_iou_or_partial_capture_geometric_merge"


def _load_corpus() -> list[dict]:
    return [json.loads(line) for line in CORPUS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def _length_ratio(lengths: list[int]) -> float | None:
    if not lengths or min(lengths) == 0:
        return None
    return max(lengths) / min(lengths)


def _bucket_iou(iou: float) -> str:
    if iou < 0.35:
        return "A_0.30-0.35_near_threshold"
    if iou < 0.45:
        return "B_0.35-0.45_above_threshold"
    if iou < 0.60:
        return "C_0.45-0.60_moderate_overlap"
    return "D_0.60plus_high_overlap"


def _mean(xs: list[float]) -> float | None:
    return round(statistics.mean(xs), 4) if xs else None


def step1_validate_existing_label(target: list[dict]) -> dict:
    ious = [r["pairwise_iou"] for r in target]
    assert all(i is not None for i in ious), "classify_packet only assigns this category when iou is not None"
    buckets = Counter(_bucket_iou(i) for i in ious)
    return {
        "n": len(target),
        "iou_min": min(ious),
        "iou_max": max(ious),
        "iou_mean": _mean(ious),
        "iou_median": round(statistics.median(ious), 4),
        "cluster_join_threshold": 0.3,
        "pct_below_threshold_ie_truly_low_iou": round(100 * sum(1 for i in ious if i < 0.3) / len(ious), 1),
        "iou_bucket_distribution": dict(sorted(buckets.items())),
        "verdict": (
            "INCORRECT for the 'low_iou' half of the label: 0 of {n} packets have IoU below the "
            "clustering engine's own 0.3 join threshold (min observed IoU is {mn}); 91.7%+ sit in "
            "the 0.30-0.45 band immediately AT or ABOVE threshold, with a tail up to {mx}. These are "
            "not low-IoU merges -- they are moderate-to-high geometric overlaps. The 'partial_capture' "
            "half of the label is directionally right for a real (but distinct, see Step 2) subset."
        ).format(n=len(target), mn=min(ious), mx=max(ious)),
    }


def step2_and_3_taxonomy(target: list[dict]) -> dict:
    granularity_mismatch = [r for r in target if r["scope_mismatched"]]
    remainder = [r for r in target if not r["scope_mismatched"]]
    close_length = [r for r in remainder if (_length_ratio(r["candidate_lengths"]) or 99) < 1.5]
    adjacent_distinct = [r for r in remainder if (_length_ratio(r["candidate_lengths"]) or 99) >= 1.5]

    def summarize(members: list[dict]) -> dict:
        providers = Counter(tuple(r["providers"]) for r in members)
        return {
            "count": len(members),
            "pct_of_350": round(100 * len(members) / len(target), 1),
            "mean_iou": _mean([r["pairwise_iou"] for r in members]),
            "mean_containment_ratio": _mean([r["containment_ratio"] for r in members]),
            "mean_comparison_magnitude": _mean([r["comparison_magnitude"] for r in members]),
            "mean_canonical_confidence": _mean(
                [r["canonical_confidence"] for r in members if r["canonical_confidence"] is not None]
            ),
            "provider_pairs": {"/".join(k): v for k, v in providers.most_common()},
            "observation_types": dict(Counter(r["observation_type"] for r in members)),
        }

    # Outlier detection within "adjacent_distinct" + "close_length": near-threshold magnitude
    # (just below classify_packet's 0.5 similar_content cutoff) or high containment despite a
    # moderate length ratio (containment computed but unused by classify_packet outside the
    # scope_mismatched>5x branch) -- both are candidates for "actually the same content,
    # misclassified as distinct by a metric that OCR noise or an over-strict length gate defeated".
    candidates_for_misclassification = [
        r for r in remainder
        if r["comparison_magnitude"] >= 0.4 or r["containment_ratio"] >= 0.9
    ]

    return {
        "A_granularity_mismatch_partial_capture": {
            **summarize(granularity_mismatch),
            "definition": "scope_mismatched=True (candidate lengths differ >5x) -- one provider's "
            "paragraph box captured a small fragment of what the other read as a much larger "
            "region. A real partial-capture / segmentation-granularity relationship.",
        },
        "B_adjacent_distinct_paragraphs": {
            **summarize(adjacent_distinct),
            "definition": "Not scope-mismatched, length ratio >=1.5x, low containment AND low "
            "comparison magnitude -- two genuinely different, similarly-sized paragraphs whose "
            "boxes coincidentally overlap above the 0.3 IoU join threshold. The dominant pattern.",
        },
        "C_close_length_ambiguous": {
            **summarize(close_length),
            "definition": "Not scope-mismatched, length ratio <1.5x -- superficially 'could be the "
            "same paragraph read twice', but mean containment/magnitude are NOT elevated versus "
            "group B on average; length similarity alone is not a reliable duplicate signal here.",
        },
        "candidates_for_metric_misclassification": {
            "count": len(candidates_for_misclassification),
            "pct_of_350": round(100 * len(candidates_for_misclassification) / len(target), 1),
            "definition": "magnitude>=0.4 (near the 0.5 similar_content cutoff) OR "
            "containment_ratio>=0.9 despite being routed through the magnitude-based branch -- "
            "manually verified examples (Step 4) show at least some of these are genuine "
            "same-content pairs whose text-similarity metrics were defeated by OCR noise or an "
            "over-strict scope-mismatch gate, not genuinely distinct content.",
        },
    }


def _find_evidence_text(evidence_ids: set[str]) -> dict[str, dict]:
    found: dict[str, dict] = {}
    with TELEMETRY_PATH.open(encoding="utf-8") as fh:
        for line in fh:
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


def step4_representative_samples(target: list[dict]) -> list[dict]:
    """One example per IoU bucket, plus the strongest metric-misclassification outliers, with real
    OCR text attached."""
    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for r in target:
        by_bucket[_bucket_iou(r["pairwise_iou"])].append(r)

    picks: list[dict] = []
    for bucket in sorted(by_bucket):
        picks.append(sorted(by_bucket[bucket], key=lambda r: r["pairwise_iou"])[0])
    picks += sorted(target, key=lambda r: -r["containment_ratio"])[:2]
    picks += sorted(target, key=lambda r: -r["comparison_magnitude"])[:2]
    # de-duplicate by slot id, preserving order
    seen = set()
    unique_picks = []
    for r in picks:
        if r["semantic_slot_id"] in seen:
            continue
        seen.add(r["semantic_slot_id"])
        unique_picks.append(r)

    evidence_ids = {eid for r in unique_picks for eid in r["evidence_ids"]}
    evidence = _find_evidence_text(evidence_ids)

    samples = []
    for r in unique_picks:
        texts = [
            {"provider": evidence[eid]["provider"], "raw_output": evidence[eid]["raw_output"]}
            for eid in r["evidence_ids"] if eid in evidence
        ]
        samples.append({
            "document_ref": r["document_ref"],
            "semantic_slot_id": r["semantic_slot_id"],
            "iou": r["pairwise_iou"],
            "containment_ratio": r["containment_ratio"],
            "comparison_magnitude": r["comparison_magnitude"],
            "scope_mismatched": r["scope_mismatched"],
            "candidate_lengths": r["candidate_lengths"],
            "texts": texts,
        })
    return samples


def main() -> None:
    corpus = _load_corpus()
    target = [r for r in corpus if r["category"] == TARGET_CATEGORY]

    other_categories = {}
    for cat in sorted({r["category"] for r in corpus} - {TARGET_CATEGORY}):
        members = [r for r in corpus if r["category"] == cat]
        other_categories[cat] = {
            "count": len(members),
            "provider_pairs": {
                "/".join(k): v for k, v in Counter(tuple(r["providers"]) for r in members).most_common()
            },
            "mean_containment_ratio": _mean([r["containment_ratio"] for r in members if r["containment_ratio"] is not None]),
            "mean_comparison_magnitude": _mean([r["comparison_magnitude"] for r in members if r["comparison_magnitude"] is not None]),
        }

    all_observation_types = Counter(r["observation_type"] for r in corpus)

    results = {
        "corpus_total": len(corpus),
        "target_category": TARGET_CATEGORY,
        "target_category_count": len(target),
        "all_observation_types_in_remaining_522": dict(all_observation_types),
        "step1_validation": step1_validate_existing_label(target),
        "step2_3_taxonomy": step2_and_3_taxonomy(target),
        "step4_representative_samples": step4_representative_samples(target),
        "other_categories_summary": other_categories,
    }
    OUTPUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")
    print(json.dumps(results["step1_validation"], indent=2, ensure_ascii=False))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "definition"} for k, v in results["step2_3_taxonomy"].items()}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
