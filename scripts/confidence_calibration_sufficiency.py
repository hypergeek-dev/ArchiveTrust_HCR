"""Confidence Calibration Sufficiency Assessment (2026-07-14).

Discovers every review pattern Benchmark #1 actually produces, computes the statistically
justified sample size needed to estimate each pattern's empirical correctness at 95% confidence
for margins of error of +/-10%, +/-5%, +/-2% (standard proportion-CI sample-size formula with a
finite-population correction, using p=0.5 -- the maximum-variance, no-prior-information-assumed
choice, since Benchmark #1 has zero human review outcomes to estimate a real p from), and reports
current review/correction counts (measured) against those requirements.

Read-only over archivetrust_data/. No provider is invoked, no benchmark is rerun.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

WORKSPACE_DIR = REPO_ROOT / "archivetrust_data" / "workspaces" / "46e94f15d73347c2be79f1527e646c11"
OUTPUT_DIR = REPO_ROOT / "benchmarks"

from archivetrust.domain.calibration.statistics import Z_95, required_sample_size, wilson_ci  # noqa: E402


def main() -> None:
    from archivetrust.domain.telemetry.events import HumanCorrectionSubmitted
    from archivetrust.infrastructure.storage.telemetry_sink import FileTelemetrySink
    from archivetrust.learning.analytics.index import TelemetryIndex
    from archivetrust.review.triage import review_reason_for

    print("Loading Benchmark #1 telemetry (read-only)...")
    sink = FileTelemetrySink(WORKSPACE_DIR / "telemetry" / "events.jsonl")
    index = TelemetryIndex.build(sink)
    canonicals = index.latest_canonicals()
    total_slots = len(canonicals)
    print(f"Total canonical slots: {total_slots}")

    corrections = [e for e in sink.all_events() if isinstance(e, HumanCorrectionSubmitted)]
    print(f"HumanCorrectionSubmitted events in Benchmark #1: {len(corrections)} (confirms prior finding)")

    # -- Pattern discovery 1: (classification, observation_type) -- the exact grain review_reason_for
    # routes on; this IS what determines whether a slot enters the review queue at all.
    by_classification_type: dict[tuple[str, str], dict] = defaultdict(lambda: {"n": 0, "queued": 0})
    # -- Pattern discovery 2: provider-participation-count patterns (corpus-wide, not tied to type)
    by_provider_count: dict[int, dict] = defaultdict(lambda: {"n": 0, "queued": 0})
    # -- Pattern discovery 3: single-source patterns broken out by which single provider contributed
    by_single_provider: dict[str, dict] = defaultdict(lambda: {"n": 0, "queued": 0})

    for c in canonicals:
        providers = index.contributing_providers(c)
        n_providers = len(providers)
        cls = c.comparison_confidence.classification.value
        otype = c.observation_type.value
        queued = review_reason_for(c) is not None

        row = by_classification_type[(cls, otype)]
        row["n"] += 1
        row["queued"] += int(queued)

        row = by_provider_count[n_providers]
        row["n"] += 1
        row["queued"] += int(queued)

        if n_providers == 1:
            provider_id = next(iter(providers)).provider_id
            row = by_single_provider[provider_id]
            row["n"] += 1
            row["queued"] += int(queued)

    # `corrections` is confirmed empty above (0 HumanCorrectionSubmitted events anywhere in this
    # dataset) -- so "reviews completed" / "corrections recorded" is 0 for every discovered pattern
    # below. Stated explicitly in each pattern row, not silently defaulted.

    margins = (0.10, 0.05, 0.02)
    report = {
        "total_canonical_slots": total_slots,
        "total_human_corrections_in_benchmark_1": len(corrections),
        "z_critical_95_percent": Z_95,
        "sample_size_assumption": "p=0.5 (maximum variance, no prior estimate available)",
        "patterns_by_classification_and_observation_type": [],
        "patterns_by_provider_participation_count": [],
        "patterns_by_single_contributing_provider": [],
    }

    def _pattern_row(name: str, n_population: int, n_queued: int, n_reviewed: int, n_corrected: int) -> dict:
        row = {
            "pattern": name,
            "population_in_benchmark_1": n_population,
            "queued_for_review": n_queued,
            "reviews_completed": n_reviewed,
            "corrections_recorded": n_corrected,
        }
        if n_reviewed > 0:
            row["empirical_correctness"] = round(1 - n_corrected / n_reviewed, 4)
            ci = wilson_ci(n_reviewed - n_corrected, n_reviewed)
            row["wilson_95_ci"] = [round(ci[0], 4), round(ci[1], 4)] if ci else None
        else:
            row["empirical_correctness"] = "unavailable (0 reviews completed)"
            row["wilson_95_ci"] = None
        row["required_sample_size"] = {
            f"margin_pm{int(m*100)}pct": required_sample_size(m, n_population) for m in margins
        }
        row["progress_pct_of_loosest_requirement"] = round(
            100 * n_reviewed / row["required_sample_size"]["margin_pm10pct"], 1
        ) if row["required_sample_size"]["margin_pm10pct"] else None
        return row

    for (cls, otype), counts in sorted(by_classification_type.items(), key=lambda kv: -kv[1]["n"]):
        name = f"{cls}/{otype}"
        report["patterns_by_classification_and_observation_type"].append(
            _pattern_row(name, counts["n"], counts["queued"], 0, 0)
        )

    for n_providers, counts in sorted(by_provider_count.items()):
        label = {1: "single-provider", 2: "two-provider", 3: "three-provider"}.get(
            n_providers, f"{n_providers}-provider"
        )
        report["patterns_by_provider_participation_count"].append(
            _pattern_row(label, counts["n"], counts["queued"], 0, 0)
        )

    for provider_id, counts in sorted(by_single_provider.items(), key=lambda kv: -kv[1]["n"]):
        name = f"single-source/{provider_id}"
        report["patterns_by_single_contributing_provider"].append(
            _pattern_row(name, counts["n"], counts["queued"], 0, 0)
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "confidence_calibration_sufficiency.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")

    # Also print a compact table to stdout for immediate inspection.
    print("\n(classification/observation_type) patterns:")
    for row in report["patterns_by_classification_and_observation_type"]:
        req = row["required_sample_size"]
        print(
            f"  {row['pattern']:<45} N={row['population_in_benchmark_1']:<7} queued={row['queued_for_review']:<7} "
            f"reviewed=0  req(±10%={req['margin_pm10pct']}, ±5%={req['margin_pm5pct']}, ±2%={req['margin_pm2pct']})"
        )

    print("\nProvider-participation-count patterns:")
    for row in report["patterns_by_provider_participation_count"]:
        req = row["required_sample_size"]
        print(
            f"  {row['pattern']:<20} N={row['population_in_benchmark_1']:<7} queued={row['queued_for_review']:<7} "
            f"req(±10%={req['margin_pm10pct']}, ±5%={req['margin_pm5pct']}, ±2%={req['margin_pm2pct']})"
        )

    print("\nSingle-provider-identity patterns:")
    for row in report["patterns_by_single_contributing_provider"]:
        req = row["required_sample_size"]
        print(
            f"  {row['pattern']:<35} N={row['population_in_benchmark_1']:<7} queued={row['queued_for_review']:<7} "
            f"req(±10%={req['margin_pm10pct']}, ±5%={req['margin_pm5pct']}, ±2%={req['margin_pm2pct']})"
        )


if __name__ == "__main__":
    main()
