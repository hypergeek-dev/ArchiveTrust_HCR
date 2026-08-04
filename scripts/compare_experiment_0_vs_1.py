"""Side-by-side Experiment 0 (shard-based, optimizer reset every shard) vs. Experiment 1 (single
continuous full-corpus epoch) comparison. Pure read of artifacts both runs already produced -- writes
one new markdown report, modifies nothing belonging to either run.

Requires `scripts/evaluate_full_run_lap.py --run <dir> --checkpoint-kind best_val` to have already been
run against Experiment 1 (Experiment 0's own `lap1_best_val` evaluation already exists on disk).

Usage:
    python scripts/compare_experiment_0_vs_1.py --experiment-1-dir training/experiment-1-continuous-epoch-<stamp>Z
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

TRAINING_ROOT = REPO_ROOT / "training"
DEFAULT_EXPERIMENT_0_DIR = TRAINING_ROOT / "full-corpus-20260802T044250Z"


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _lap_evaluation(run_dir: Path, *, checkpoint_kind: str = "best_val") -> dict | None:
    candidates = sorted((run_dir / "lap-evaluation").glob(f"lap*_{checkpoint_kind}/lap_evaluation.json")) \
        if (run_dir / "lap-evaluation").exists() else []
    if not candidates:
        return None
    return _load_json(candidates[-1])


def _optimizer_trace_summary(run_dir: Path) -> dict:
    trace_paths = sorted(run_dir.glob("run-state/epoch_output/epoch_*/optimizer_trace.csv"))
    if not trace_paths:
        return {"present": False}
    rows = list(csv.DictReader(trace_paths[-1].open(encoding="utf-8")))
    if not rows:
        return {"present": True, "row_count": 0}
    iterations = [int(r["optimizer_iterations"]) for r in rows if r["optimizer_iterations"]]
    lrs = [float(r["learning_rate"]) for r in rows if r["learning_rate"]]
    monotonic = all(b >= a for a, b in zip(iterations, iterations[1:])) if len(iterations) > 1 else True
    return {
        "present": True,
        "row_count": len(rows),
        "first_iterations": iterations[0] if iterations else None,
        "last_iterations": iterations[-1] if iterations else None,
        "monotonic_nondecreasing": monotonic,
        "first_lr": lrs[0] if lrs else None,
        "last_lr": lrs[-1] if lrs else None,
        "path": str(trace_paths[-1]),
    }


def _gpu_telemetry_summary(run_dir: Path) -> dict:
    path = run_dir / "run-state" / "telemetry" / "gpu_samples.jsonl"
    if not path.exists():
        return {"present": False}
    utilizations, mem_used, temps = [], [], []
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue
            gpu = sample.get("gpu", {})
            if gpu.get("utilization_pct") is not None:
                utilizations.append(gpu["utilization_pct"])
            if gpu.get("memory_used_mb") is not None:
                mem_used.append(gpu["memory_used_mb"])
            if gpu.get("temperature_c") is not None:
                temps.append(gpu["temperature_c"])
    if not utilizations and not mem_used and not temps:
        return {"present": True, "sample_count": 0}
    return {
        "present": True,
        "sample_count": max(len(utilizations), len(mem_used), len(temps)),
        "mean_utilization_pct": round(statistics.mean(utilizations), 1) if utilizations else None,
        "max_utilization_pct": max(utilizations) if utilizations else None,
        "mean_memory_used_mb": round(statistics.mean(mem_used), 1) if mem_used else None,
        "max_memory_used_mb": max(mem_used) if mem_used else None,
        "max_temperature_c": max(temps) if temps else None,
    }


def _fmt(value) -> str:
    if value is None:
        return "(none)"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_report(*, exp0_dir: Path, exp1_dir: Path) -> str:
    exp0_eval = _lap_evaluation(exp0_dir)
    exp1_eval = _lap_evaluation(exp1_dir)
    exp0_state = _load_json(exp0_dir / "run-state" / "run_state.json") or {}
    exp1_state = _load_json(exp1_dir / "run-state" / "run_state.json") or {}
    exp0_manifest = _load_json(exp0_dir / "launch_manifest.json") or {}
    exp1_manifest = _load_json(exp1_dir / "launch_manifest.json") or {}
    exp1_trace = _optimizer_trace_summary(exp1_dir)
    exp1_gpu = _gpu_telemetry_summary(exp1_dir)

    lines: list[str] = []
    lines.append("# Experiment 0 (shard-based) vs. Experiment 1 (single continuous epoch)\n")
    lines.append(f"- Experiment 0 directory: `{exp0_dir}`")
    lines.append(f"- Experiment 1 directory: `{exp1_dir}`\n")

    lines.append("## Method\n")
    lines.append("| | Experiment 0 | Experiment 1 |")
    lines.append("|---|---|---|")
    lines.append(f"| Learning-rate policy | `{exp0_manifest.get('scheduler', '(unknown)')}` | `{exp1_manifest.get('scheduler', '(unknown)')}` |")
    lines.append(f"| Shard count | {exp0_manifest.get('shard_count', '(unknown)')} | {exp1_manifest.get('shard_count', '(unknown)')} |")
    lines.append(f"| Batch size | {exp0_manifest.get('batch_size', '(unknown)')} | {exp1_manifest.get('batch_size', '(unknown)')} |")
    lines.append(f"| Optimizer | {exp0_manifest.get('optimizer', '(unknown)')} | {exp1_manifest.get('optimizer', '(unknown)')} |")
    lines.append(f"| Learning rate | {exp0_manifest.get('learning_rate', '(unknown)')} | {exp1_manifest.get('learning_rate', '(unknown)')} |")
    lines.append(f"| Base checkpoint hash | `{exp0_manifest.get('base_checkpoint_hash', '(unknown)')[:16]}...` | `{exp1_manifest.get('base_checkpoint_hash', '(unknown)')[:16]}...` |\n")

    lines.append("## Run outcome\n")
    lines.append("| | Experiment 0 | Experiment 1 |")
    lines.append("|---|---|---|")
    lines.append(f"| status | {exp0_state.get('status', '(unknown)')} | {exp1_state.get('status', '(unknown)')} |")
    lines.append(f"| epochs_completed | {exp0_state.get('epochs_completed', '(unknown)')} | {exp1_state.get('epochs_completed', '(unknown)')} |")
    lines.append(f"| stop_reason | {exp0_state.get('stop_reason', '(unknown)')} | {exp1_state.get('stop_reason', '(unknown)')} |")
    lines.append(f"| best_metrics.val_cer (in-training) | {_fmt((exp0_state.get('best_metrics') or {}).get('val_cer'))} | {_fmt((exp1_state.get('best_metrics') or {}).get('val_cer'))} |\n")

    lines.append("## Held-out evaluation (`lap_evaluation.json`, best_val checkpoint, same code/metric implementation)\n")
    if exp0_eval and exp1_eval:
        o0, o1 = exp0_eval["overall"], exp1_eval["overall"]
        lines.append("| | Experiment 0 | Experiment 1 |")
        lines.append("|---|---|---|")
        lines.append(f"| corpus_cer | {_fmt(o0['corpus_cer'])} | {_fmt(o1['corpus_cer'])} |")
        lines.append(f"| corpus_wer | {_fmt(o0['corpus_wer'])} | {_fmt(o1['corpus_wer'])} |")
        lines.append(f"| line_error_rate | {_fmt(o0['line_error_rate'])} | {_fmt(o1['line_error_rate'])} |")
        lines.append(f"| mean_per_line_cer | {_fmt(o0['mean_per_line_cer'])} | {_fmt(o1['mean_per_line_cer'])} |")
        lines.append(f"| scored_line_count | {exp0_eval['scored_line_count']} | {exp1_eval['scored_line_count']} |")
        lines.append(f"| worst_collection | {exp0_eval['worst_collection']} | {exp1_eval['worst_collection']} |")
        lines.append(f"| best_collection | {exp0_eval['best_collection']} | {exp1_eval['best_collection']} |\n")

        lines.append("### Per-collection CER (Experiment 0 vs Experiment 1)\n")
        lines.append("| collection | Exp 0 corpus_cer | Exp 1 corpus_cer |")
        lines.append("|---|---|---|")
        exp1_by_collection = {c["collection"]: c for c in exp1_eval["per_collection"]}
        for c in exp0_eval["per_collection"]:
            other = exp1_by_collection.get(c["collection"])
            lines.append(
                f"| {c['collection']} | {_fmt(c['metrics']['corpus_cer'])} | "
                f"{_fmt(other['metrics']['corpus_cer']) if other else '(missing)'} |"
            )
        lines.append("")
    else:
        missing = []
        if not exp0_eval:
            missing.append("Experiment 0")
        if not exp1_eval:
            missing.append("Experiment 1")
        lines.append(f"Missing `lap_evaluation.json` for: {', '.join(missing)}. Run "
                      f"`scripts/evaluate_full_run_lap.py --run <dir> --checkpoint-kind best_val` first.\n")

    lines.append("## Experiment 1 optimizer/LR continuity proof\n")
    if exp1_trace.get("present"):
        lines.append(f"- Trace file: `{exp1_trace.get('path')}` ({exp1_trace.get('row_count')} rows)")
        lines.append(f"- optimizer.iterations: {exp1_trace.get('first_iterations')} -> {exp1_trace.get('last_iterations')}")
        lines.append(f"- monotonically non-decreasing across the whole epoch: **{exp1_trace.get('monotonic_nondecreasing')}**")
        lines.append(f"- learning_rate: {_fmt(exp1_trace.get('first_lr'))} -> {_fmt(exp1_trace.get('last_lr'))}\n")
    else:
        lines.append("No `optimizer_trace.csv` found under Experiment 1's `run-state/epoch_output/epoch_*/` -- "
                      "the mounted override may not have been picked up. Check the container ran with "
                      "`extra_volume_mounts` pointed at `training/experiment_1_overrides/"
                      "modes_training_with_optimizer_trace.py`.\n")

    lines.append("## Experiment 1 GPU/system telemetry\n")
    if exp1_gpu.get("present") and exp1_gpu.get("sample_count"):
        lines.append(f"- samples: {exp1_gpu['sample_count']}")
        lines.append(f"- utilization_pct: mean {_fmt(exp1_gpu.get('mean_utilization_pct'))}, max {_fmt(exp1_gpu.get('max_utilization_pct'))}")
        lines.append(f"- memory_used_mb: mean {_fmt(exp1_gpu.get('mean_memory_used_mb'))}, max {_fmt(exp1_gpu.get('max_memory_used_mb'))}")
        lines.append(f"- max temperature_c: {_fmt(exp1_gpu.get('max_temperature_c'))}\n")
    else:
        lines.append("No GPU telemetry samples found under Experiment 1's `run-state/telemetry/gpu_samples.jsonl`.\n")

    lines.append("## Caveat\n")
    lines.append("This isolates trainer continuity (one variable) -- corpus, validation set, architecture, "
                  "optimizer, and learning rate are held identical between the two runs by construction. "
                  "It is not a comparison against Swedish Lion or any other architecture.")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-0-dir", default=str(DEFAULT_EXPERIMENT_0_DIR))
    parser.add_argument("--experiment-1-dir", required=True)
    parser.add_argument("--output", default=None, help="defaults to <experiment-1-dir>/comparison_report.md")
    args = parser.parse_args()

    exp0_dir = Path(args.experiment_0_dir).resolve()
    exp1_dir = Path(args.experiment_1_dir).resolve()
    output_path = Path(args.output).resolve() if args.output else exp1_dir / "comparison_report.md"

    report = render_report(exp0_dir=exp0_dir, exp1_dir=exp1_dir)
    output_path.write_text(report, encoding="utf-8")
    print(f"written: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
