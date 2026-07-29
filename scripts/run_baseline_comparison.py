#!/usr/bin/env python
"""Standalone entry point for the "Swedish Historical HTR Baseline Comparison" experiment
(docs/htr-migration-plan.md Stage 12).

Runs `archivetrust.htr.experiment.baseline_execution.run_baseline_comparison()` with the real
production adapters (SATRN via the isolated `.venv-satrn` subprocess, Florence-2 via this venv's
`transformers`, Transkribus via the hand-authored PAGE XML fixture), prints a real console summary,
and writes the real generated research report (JSON + CSV) under
`docs/experiments/baseline-comparison/`.

Usage (from the repo root, main venv active):

    PYTHONPATH=src .venv/Scripts/python.exe scripts/run_baseline_comparison.py

Requires: `.venv-satrn` set up per `src/archivetrust/providers/satrn/README.md`, and the
`transformers` extra installed for Florence-2 (see `src/archivetrust/providers/florence2_htr/
README.md`). Network access to huggingface.co is needed on a cold cache (first run only).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from archivetrust.htr.experiment.baseline_execution import (  # noqa: E402
    build_research_report,
    run_baseline_comparison,
    write_metric_results_csv,
)
from archivetrust.research.reports.export import report_to_csv, write_report  # noqa: E402

OUTPUT_DIR = REPO_ROOT / "docs" / "experiments" / "baseline-comparison"


def main() -> int:
    print("=" * 78)
    print("Swedish Historical HTR Baseline Comparison -- real execution")
    print("=" * 78)

    print("\n[1/5] Executing controlled + end-to-end comparison (real adapters, real GPU/CPU "
          "inference where applicable)...")
    result = run_baseline_comparison()
    print("      done.")

    print("\n[2/5] Shared InputCrop hash verification")
    print(f"      InputCrop.hash              = {result.shared_crop.hash}")
    print(f"      SATRN MethodRun.input_crop_id     = {result.satrn.method_run.input_crop_id}")
    print(f"      Florence-2 MethodRun.input_crop_id = {result.florence2.method_run.input_crop_id}")
    assert result.satrn.method_run.input_crop_id == result.florence2.method_run.input_crop_id == result.shared_crop.crop_id
    print("      MATCH -- both local recognizers consumed byte-identical input crop bytes.")

    print("\n[3/5] Real recognition output")
    print(f"      Ground truth          : {result.ground_truth_text!r}")
    print(f"      SATRN raw output      : {result.satrn.transcript.raw_text!r}")
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

    print("\n[4/5] Real recognition metrics (against real ground truth)")
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

    print("\n[5/5] Reproducibility manifest (real, captured this run)")
    print(f"      manifest_id  = {result.manifest.manifest_id}")
    print(f"      git_commit   = {result.manifest.git_commit}")
    print(f"      software_environment = {result.manifest.software_environment}")
    print(f"      hardware_environment = {result.manifest.hardware_environment}")

    report = build_research_report(result)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = write_report(report, OUTPUT_DIR / "research_report.json", export_format="json")
    csv_path = write_report(report, OUTPUT_DIR / "research_report_summary.csv", export_format="csv")
    metrics_csv_path = write_metric_results_csv(result, OUTPUT_DIR / "metric_results.csv")

    print("\nWrote:")
    print(f"      {json_path}")
    print(f"      {csv_path}")
    print(f"      {metrics_csv_path}")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
