"""Calibration Dashboard (Milestone 11, Phase 8).

Read-only over `archivetrust_data/`, same discipline as `confidence_calibration_sufficiency.py`
and `derive_benchmark.py`: no provider is invoked, no document is reprocessed. This is additive to
the existing sufficiency report, not a replacement of it -- it renders live per-pattern
calibration progress (`domain.calibration.progress`), blind spots (`domain.calibration.coverage`),
and review priority (`domain.calibration.information_gain`) using the same underlying, now-shared
statistics module that report promoted out of its own one-off script.

Run: `python scripts/calibration_dashboard.py`. Writes `benchmarks/CALIBRATION_DASHBOARD.md` and
`benchmarks/calibration_dashboard.json`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

WORKSPACE_DIR = REPO_ROOT / "archivetrust_data" / "workspaces" / "46e94f15d73347c2be79f1527e646c11"
OUTPUT_DIR = REPO_ROOT / "benchmarks"


def _bar(pct: float, width: int = 10) -> str:
    """ASCII-only (not the sufficiency report's Unicode block glyphs) -- Windows console default
    encoding (cp1252) can't print `█`/`░`, and this script's stdout is a diagnostic
    convenience, not the artifact of record (the .md/.json files are)."""
    filled = round(min(max(pct, 0.0), 100.0) / 100.0 * width)
    return "#" * filled + "." * (width - filled)


def main() -> None:
    from archivetrust.domain.calibration.information_gain import expected_information_gain
    from archivetrust.domain.calibration.coverage import find_blind_spots
    from archivetrust.infrastructure.storage.telemetry_sink import FileTelemetrySink
    from archivetrust.learning.analytics.index import TelemetryIndex
    from archivetrust.review.sampling.discovery import by_classification_and_type
    from archivetrust.review.sampling.log import InMemorySamplingLogSink
    from archivetrust.review.sampling.progress import compute_calibration_progress_map

    print("Loading telemetry (read-only)...")
    sink = FileTelemetrySink(WORKSPACE_DIR / "telemetry" / "events.jsonl")
    index = TelemetryIndex.build(sink)

    # No calibration-intent review has ever run against this workspace (Milestone 11 is new); an
    # empty in-memory log is the honest starting state, not a fabricated one. A deployment running
    # real calibration review would pass a `FileSamplingLogSink` over its own persisted log here.
    sampling_log = InMemorySamplingLogSink()

    patterns = by_classification_and_type(index)
    progress_by_pattern = compute_calibration_progress_map(index, sampling_log)
    total_population = sum(p.population for p in patterns)
    blind_spots = find_blind_spots(patterns, progress_by_pattern, total_population=total_population)

    report: dict = {"total_canonical_slots": total_population, "patterns": [], "blind_spots": []}

    print("\nConfidence Calibration Progress (toward +/-10% margin, 95% confidence)\n")
    for pattern in patterns:
        progress = progress_by_pattern[pattern.name]
        share = pattern.population / total_population if total_population else 0.0
        gain = expected_information_gain(progress, corpus_share=share)
        pct = progress.progress_pct_of_loosest_requirement or 0.0
        req10 = progress.required_n["margin_pm10pct"]
        print(
            f"{pattern.name:<45} {_bar(pct)}  {progress.reviews_completed} / {req10} reviews  "
            f"{pct:.0f}%  [{progress.maturity.value}]"
        )
        report["patterns"].append(
            {
                "pattern": pattern.name,
                "population": pattern.population,
                "corpus_share": round(share, 4),
                "queued_for_review": pattern.queued_for_review,
                "reviews_completed": progress.reviews_completed,
                "corrections_recorded": progress.corrections_recorded,
                "empirical_correctness": progress.empirical_correctness,
                "wilson_95_ci": progress.wilson_95_ci,
                "used_worst_case_p": progress.used_worst_case_p,
                "required_n": progress.required_n,
                "maturity": progress.maturity.value,
                "progress_pct_of_loosest_requirement": progress.progress_pct_of_loosest_requirement,
                "expected_information_gain": gain,
            }
        )

    print(f"\n{len(blind_spots)} blind spot(s) ({sum(b.corpus_share for b in blind_spots):.1%} of corpus):\n")
    for spot in blind_spots:
        print(f"  {spot.pattern:<45} {spot.corpus_share:.2%}  reason={spot.reason.value}  "
              f"recommend={spot.recommended_sampling_strategy}")
        report["blind_spots"].append(spot.model_dump(mode="json"))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "calibration_dashboard.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_lines = [
        "# Calibration Dashboard",
        "",
        f"{total_population} canonical slots across {len(patterns)} pattern(s). "
        f"{len(blind_spots)} blind spot(s) ({sum(b.corpus_share for b in blind_spots):.1%} of corpus).",
        "",
        "| Pattern | Population | Reviews | Required (+/-10%) | Progress | Maturity |",
        "|---|---|---|---|---|---|",
    ]
    for row in report["patterns"]:
        pct = row["progress_pct_of_loosest_requirement"] or 0.0
        md_lines.append(
            f"| {row['pattern']} | {row['population']} | {row['reviews_completed']} | "
            f"{row['required_n']['margin_pm10pct']} | {pct:.0f}% | {row['maturity']} |"
        )
    md_lines += ["", "## Blind spots", "", "| Pattern | Corpus share | Reason | Recommended strategy |", "|---|---|---|---|"]
    for spot in report["blind_spots"]:
        md_lines.append(
            f"| {spot['pattern']} | {spot['corpus_share']:.2%} | {spot['reason']} | "
            f"{spot['recommended_sampling_strategy']} |"
        )
    (OUTPUT_DIR / "CALIBRATION_DASHBOARD.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"\nWrote {OUTPUT_DIR / 'calibration_dashboard.json'} and {OUTPUT_DIR / 'CALIBRATION_DASHBOARD.md'}")


if __name__ == "__main__":
    main()
