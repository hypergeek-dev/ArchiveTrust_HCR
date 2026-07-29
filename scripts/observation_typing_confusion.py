"""Phase 3: confusion matrix (native_label=='Text' -> mapped ObservationType.PARAGRAPH -> actual
semantic object), from the Phase 2 manually-labeled stratified sample. Reuses
domain.calibration.statistics.wilson_ci (already built for this exact purpose -- no new interval
formula introduced).
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from archivetrust.domain.calibration.statistics import wilson_ci  # noqa: E402

SAMPLE = REPO_ROOT / "benchmarks" / "observation_typing_sample.jsonl"
LABELS = REPO_ROOT / "benchmarks" / "observation_typing_sample_labels.json"
OUT = REPO_ROOT / "benchmarks" / "observation_typing_confusion.json"


def main():
    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    labels.pop("_methodology", None)

    records = [json.loads(line) for line in SAMPLE.read_text(encoding="utf-8").splitlines() if line.strip()]
    missing = [r["observation_id"] for r in records if r["observation_id"] not in labels]
    if missing:
        raise SystemExit(f"Missing manual labels for {len(missing)} sampled observations: {missing[:5]}...")

    by_stratum: dict[str, Counter] = defaultdict(Counter)
    overall = Counter()
    for r in records:
        actual = labels[r["observation_id"]]
        by_stratum[r["stratum"]][actual] += 1
        overall[actual] += 1

    print("=== CONFUSION MATRIX: native_label=='Text' -> mapped=PARAGRAPH -> actual semantic object ===\n")
    result = {"by_stratum": {}, "overall": {}}
    for stratum in sorted(by_stratum, key=lambda s: -sum(by_stratum[s].values())):
        counts = by_stratum[stratum]
        n = sum(counts.values())
        non_paragraph = n - counts.get("paragraph", 0)
        ci = wilson_ci(non_paragraph, n)
        print(f"[{stratum}] n={n}")
        for actual, c in counts.most_common():
            print(f"    actual={actual}: {c} ({100*c/n:.1f}%)")
        ci_str = f"[{ci[0]*100:.1f}%, {ci[1]*100:.1f}%]" if ci else "n/a"
        print(f"    -> mistyped (actual != paragraph): {non_paragraph}/{n} = {100*non_paragraph/n:.1f}%  95% CI {ci_str}\n")
        result["by_stratum"][stratum] = {
            "n": n,
            "counts": dict(counts),
            "mistyped": non_paragraph,
            "mistyped_pct": 100 * non_paragraph / n,
            "mistyped_95ci": list(ci) if ci else None,
        }

    n_total = sum(overall.values())
    non_paragraph_total = n_total - overall.get("paragraph", 0)
    ci_total = wilson_ci(non_paragraph_total, n_total)
    print(f"=== OVERALL (pooled across strata, n={n_total}) ===")
    for actual, c in overall.most_common():
        print(f"    actual={actual}: {c} ({100*c/n_total:.1f}%)")
    print(
        f"    -> mistyped (actual != paragraph): {non_paragraph_total}/{n_total} = "
        f"{100*non_paragraph_total/n_total:.1f}%  95% CI [{ci_total[0]*100:.1f}%, {ci_total[1]*100:.1f}%]"
    )
    result["overall"] = {
        "n": n_total,
        "counts": dict(overall),
        "mistyped": non_paragraph_total,
        "mistyped_pct": 100 * non_paragraph_total / n_total,
        "mistyped_95ci": list(ci_total) if ci_total else None,
    }

    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
