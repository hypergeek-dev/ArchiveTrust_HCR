#!/usr/bin/env python
"""Standalone entry point for the "Swedish Historical HTR Baseline Comparison" experiment
(docs/htr-migration-plan.md Stage 12; made durable 2026-07-30, docs/architecture/htr-telemetry.md).

Runs `archivetrust.htr.experiment.baseline_execution.run_baseline_comparison()` with the real
production adapters (SATRN via the isolated `.venv-satrn` subprocess, Florence-2 via this venv's
`transformers`, Transkribus via the hand-authored PAGE XML fixture), prints a real console summary,
and writes the real generated research report (JSON + CSV) under
`docs/experiments/baseline-comparison/`.

**What changed 2026-07-30.** This script previously let `run_baseline_comparison` construct a
throwaway in-memory `HtrResearchStore`, so the only thing that outlived the process was three export
files -- and `docs/htr-telemetry-knowledge-gap-analysis.md` §0 records that even those did not
survive to be committed. It now injects a `DurableHtrResearchStore` backed by a real
`FileTelemetrySink` at `docs/experiments/baseline-comparison/htr_research_events.jsonl`, so every
registration this run makes is appended to an append-only, hash-chained event log next to the
reports. That log -- not the reports, and not this process's memory -- is the run's source of truth:
`HtrJournal.replay` rebuilds the whole entity graph from it alone
(`tests/htr/persistence/test_real_baseline_reconstruction.py` proves it against this very file).

**Files written** (all under `OUTPUT_DIR`, all intended to be committed):

| File | What it is |
|---|---|
| `htr_research_events.jsonl` | the durable, append-only telemetry log -- the source of truth |
| `htr_research_events.jsonl.chain.jsonl` | `HashChainAppender`'s tamper-evidence sidecar |
| `blobs/` | `FileTelemetrySink`'s content-addressed store for any `raw_output` >4 KiB (legitimately empty at line scale) |
| `htr_coarse_entities.json` | the `WorkspaceStore`-idiom derived cache; deleting it loses nothing |
| `research_report.json` / `research_report_summary.csv` | the generated report, sourced from the log above |
| `metric_results.csv` | one row per (method, metric, value), likewise |

**One run per log.** A fresh `DurableHtrResearchStore` starts with an empty projection, so pointing a
second run at an existing log would silently append a second run's events to it rather than refusing.
Since the committed artifact is meant to be one run's evidence, this script refuses to start when the
log already exists unless `--fresh` (delete and start over) or `--append` (deliberately add another
run) is given. Refusing is the point: "which of the two runs in this file produced the committed
report?" is not a question the artifact should be able to raise.

Usage (from the repo root, main venv active):

    PYTHONPATH=src .venv/Scripts/python.exe scripts/run_baseline_comparison.py [--fresh|--append]

Requires: `.venv-satrn` set up per `src/archivetrust/providers/satrn/README.md`, and the
`transformers` extra installed for Florence-2 (see `src/archivetrust/providers/florence2_htr/
README.md`). Network access to huggingface.co is needed on a cold cache (first run only).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from archivetrust.htr.experiment.baseline_execution import (  # noqa: E402
    build_research_report,
    run_baseline_comparison,
    write_metric_results_csv,
)
from archivetrust.htr.persistence import (  # noqa: E402
    DurableHtrResearchStore,
    HtrCoarseEntitySnapshot,
)
from archivetrust.infrastructure.storage.telemetry_sink import FileTelemetrySink  # noqa: E402
from archivetrust.research.reports.export import write_report  # noqa: E402

OUTPUT_DIR = REPO_ROOT / "docs" / "experiments" / "baseline-comparison"
TELEMETRY_PATH = OUTPUT_DIR / "htr_research_events.jsonl"
CHAIN_PATH = OUTPUT_DIR / "htr_research_events.jsonl.chain.jsonl"
SNAPSHOT_PATH = OUTPUT_DIR / "htr_coarse_entities.json"


def _prepare_output_dir(*, fresh: bool, append: bool) -> None:
    """Applies the "one run per log" rule described in the module docstring."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not TELEMETRY_PATH.exists() or append:
        return
    if not fresh:
        raise SystemExit(
            f"Refusing to run: {TELEMETRY_PATH} already exists.\n"
            "  Pass --fresh to delete it and record a single clean run (what you want for a\n"
            "  committed artifact), or --append to deliberately add a second run's events to the\n"
            "  same log. See this script's module docstring for why this is not silent."
        )
    for path in (TELEMETRY_PATH, CHAIN_PATH, SNAPSHOT_PATH):
        if path.exists():
            path.unlink()
    print(f"      --fresh: removed the previous log at {TELEMETRY_PATH}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the Swedish Historical HTR Baseline Comparison for real."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--fresh", action="store_true", help="delete any existing telemetry log first")
    group.add_argument("--append", action="store_true", help="append this run to an existing log")
    args = parser.parse_args(argv)

    print("=" * 78)
    print("Swedish Historical HTR Baseline Comparison -- real execution, durable telemetry")
    print("=" * 78)

    print("\n[1/7] Preparing the durable telemetry destination")
    _prepare_output_dir(fresh=args.fresh, append=args.append)
    sink = FileTelemetrySink(TELEMETRY_PATH)
    store = DurableHtrResearchStore(
        sink,
        actor_id="scripts/run_baseline_comparison.py",
        snapshot=HtrCoarseEntitySnapshot(SNAPSHOT_PATH),
    )
    print(f"      telemetry log    = {TELEMETRY_PATH}")
    print(f"      store.is_durable = {store.is_durable}")
    if not store.is_durable:
        raise SystemExit("Refusing to run: the store did not resolve to a durable file-backed sink.")

    print("\n[2/7] Executing controlled + end-to-end comparison (real adapters, real GPU/CPU "
          "inference where applicable)...")
    result = run_baseline_comparison(store=store)
    print("      done.")

    print("\n[3/7] Shared InputCrop hash verification")
    print(f"      InputCrop.hash                     = {result.shared_crop.hash}")
    print(f"      SATRN MethodRun.input_crop_id      = {result.satrn.method_run.input_crop_id}")
    print(f"      Florence-2 MethodRun.input_crop_id = {result.florence2.method_run.input_crop_id}")
    assert result.satrn.method_run.input_crop_id == result.florence2.method_run.input_crop_id == result.shared_crop.crop_id
    print("      MATCH -- both local recognizers consumed byte-identical input crop bytes.")

    print("\n[4/7] Real recognition output")
    print(f"      Ground truth            : {result.ground_truth_text!r}")
    print(f"      SATRN raw output        : {result.satrn.transcript.raw_text!r}")
    if result.satrn.evidence is not None:
        print(f"        confidence={result.satrn.evidence.provider_confidence}, "
              f"device={result.satrn.evidence.execution_device}, "
              f"elapsed_ms={result.satrn.evidence.execution_time_ms}")
    print(f"      Florence-2 parsed output: {result.florence2.transcript.parsed_text!r}")
    if result.florence2.evidence is not None:
        print(f"        confidence_proxy={result.florence2.evidence.provider_confidence}, "
              f"device={result.florence2.evidence.execution_device}, "
              f"elapsed_ms={result.florence2.evidence.execution_time_ms}")
    print(f"      Transkribus parsed output (page-level-only, NOT controlled): "
          f"{result.transkribus.transcript.parsed_text!r}")

    print("\n[5/7] Real recognition metrics (against real ground truth)")
    if result.satrn.metrics is not None:
        m = result.satrn.metrics
        print(f"      SATRN      CER raw={m.character_error_rate_raw:.4f} "
              f"CER norm={m.character_error_rate_normalized:.4f} "
              f"WER raw={m.word_error_rate_raw:.4f} WER norm={m.word_error_rate_normalized:.4f}")
    if result.florence2.metrics is not None:
        m = result.florence2.metrics
        print(f"      Florence-2 CER raw={m.character_error_rate_raw:.4f} "
              f"CER norm={m.character_error_rate_normalized:.4f} "
              f"WER raw={m.word_error_rate_raw:.4f} WER norm={m.word_error_rate_normalized:.4f}")
    print("      Transkribus: no CER/WER computed -- page-level fixture has no corresponding "
          "ground truth (see baseline_template.py exclusion_criteria).")

    print("\n[6/7] Durable telemetry + reproducibility manifest (real, captured this run)")
    events = list(sink.all_events())
    kinds: dict[str, int] = {}
    for event in events:
        kinds[event.kind.value] = kinds.get(event.kind.value, 0) + 1
    print(f"      events appended      = {len(events)}")
    print(f"      distinct event kinds = {len(kinds)}")
    for kind in sorted(kinds):
        print(f"        {kind:38s} {kinds[kind]}")
    controlled_id = result.controlled_run.experiment_run_id
    end_to_end_id = result.end_to_end_run.experiment_run_id
    controlled_events = [e for e in events if e.correlation_id == controlled_id]
    end_to_end_events = [e for e in events if e.correlation_id == end_to_end_id]
    uncorrelated = [e for e in events if e.correlation_id is None]
    print(f"      controlled correlation_id       = {controlled_id}")
    print(f"      end-to-end correlation_id       = {end_to_end_id}")
    print(f"      events correlated to controlled = {len(controlled_events)}")
    print(f"      events correlated to end-to-end = {len(end_to_end_events)}")
    print(f"      events with an honest None      = {len(uncorrelated)}  (pre-run registrations)")
    assert controlled_id != end_to_end_id
    assert not ({e.event_id for e in controlled_events} & {e.event_id for e in end_to_end_events})
    print("      SEPARATE -- no event belongs to both runs' correlations.")
    print(f"      hash-chain sidecar   = {CHAIN_PATH.exists()}")
    print(f"      manifest_id  = {result.manifest.manifest_id}")
    print(f"      git_commit   = {result.manifest.git_commit}")
    print(f"      software_environment = {result.manifest.software_environment}")
    print(f"      hardware_environment = {result.manifest.hardware_environment}")

    print("\n[7/7] Generating the research report from those durable records")
    report = build_research_report(result)
    json_path = write_report(report, OUTPUT_DIR / "research_report.json", export_format="json")
    csv_path = write_report(report, OUTPUT_DIR / "research_report_summary.csv", export_format="csv")
    metrics_csv_path = write_metric_results_csv(result, OUTPUT_DIR / "metric_results.csv")
    print(f"      report source = {report.metadata['source']}")

    print("\nWrote:")
    for path in (TELEMETRY_PATH, CHAIN_PATH, SNAPSHOT_PATH, json_path, csv_path, metrics_csv_path):
        print(f"      {path}")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
