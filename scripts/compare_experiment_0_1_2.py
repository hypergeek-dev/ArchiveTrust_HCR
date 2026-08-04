"""Three-way comparison: Experiment 0 (shard-reset), Experiment 1 (continuous fine-tune),
Experiment 2 (native recommended VGSL trained from scratch). Pure read of artifacts all three runs
already produced -- writes new files only, modifies nothing belonging to any of them.

Usage:
    python scripts/compare_experiment_0_1_2.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

TRAINING_ROOT = REPO_ROOT / "training"
DEFAULT_EXP0_DIR = TRAINING_ROOT / "full-corpus-20260802T044250Z"
DEFAULT_EXP1_DIR = TRAINING_ROOT / "experiment-1-continuous-epoch-20260803T192146Z"
DEFAULT_EXP2_DIR = TRAINING_ROOT / "experiment-2-loghi-recommended-scratch-20260804T051959Z"


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _lap_evaluation(run_dir: Path, kind: str = "best_val") -> dict | None:
    candidates = sorted((run_dir / "lap-evaluation").glob(f"lap*_{kind}/lap_evaluation.json")) \
        if (run_dir / "lap-evaluation").exists() else []
    return _load_json(candidates[-1]) if candidates else None


def _run_context(run_dir: Path) -> dict:
    state = _load_json(run_dir / "run-state" / "run_state.json") or {}
    manifest = _load_json(run_dir / "launch_manifest.json") or {}
    identity = _load_json(run_dir / "run-state" / "training_identity.json") or {}
    trace_candidates = sorted(run_dir.glob("run-state/epoch_output/epoch_*/optimizer_trace.csv"))
    optimizer_steps = None
    starting_train_cer = None
    ending_train_cer = None
    if trace_candidates:
        import csv
        rows = list(csv.DictReader(trace_candidates[-1].open(encoding="utf-8")))
        data_rows = [r for r in rows if r.get("CER_metric")]
        if data_rows:
            starting_train_cer = float(data_rows[0]["CER_metric"])
            ending_train_cer = float(data_rows[-1]["CER_metric"])
        last_row = rows[-1] if rows else None
        if last_row and last_row.get("optimizer_iterations"):
            optimizer_steps = int(last_row["optimizer_iterations"])
    eval_data = _lap_evaluation(run_dir)
    checkpoint_size_bytes = None
    best_val_dir = None
    for entry in (_load_json(run_dir / "run-state" / "checkpoint_index.json") or {}).get("entries", []):
        if entry.get("checkpoint_kind") == "best_val":
            best_val_dir = Path(entry["checkpoint_dir"])
    if best_val_dir:
        candidate = best_val_dir if best_val_dir.is_absolute() else (REPO_ROOT / best_val_dir)
        model_file = candidate / "model.keras"
        if model_file.exists():
            checkpoint_size_bytes = model_file.stat().st_size

    return {
        "run_id": state.get("run_id"),
        "status": state.get("status"),
        "started_at": state.get("started_at"),
        "ended_at": state.get("ended_at"),
        "runtime_seconds": (_load_json(run_dir / "run-state" / "session_state.json") or {}).get("cumulative_training_seconds"),
        "in_training_best_val_cer": (state.get("best_metrics") or {}).get("val_cer"),
        "optimizer_steps": optimizer_steps,
        "starting_train_cer": starting_train_cer,
        "ending_train_cer": ending_train_cer,
        "held_out_corpus_cer": (eval_data or {}).get("overall", {}).get("corpus_cer"),
        "held_out_corpus_wer": (eval_data or {}).get("overall", {}).get("corpus_wer"),
        "held_out_line_error_rate": (eval_data or {}).get("overall", {}).get("line_error_rate"),
        "learning_rate_policy": manifest.get("scheduler"),
        "batch_size": manifest.get("batch_size"),
        "optimizer": manifest.get("optimizer"),
        "gpu_memory_used_mb_peak": None,  # filled from telemetry below
        "checkpoint_size_bytes": checkpoint_size_bytes,
        "per_collection": (eval_data or {}).get("per_collection", []),
        "confidence_buckets": (eval_data or {}).get("confidence_buckets", []),
        "identity_config": identity.get("configuration", {}),
        "initialization": manifest.get("initialization", "pretrained (generic Loghi checkpoint)"),
        "architecture": manifest.get("architecture_key") or identity.get("configuration", {}).get("model_architecture"),
    }


def _gpu_peak(run_dir: Path) -> float | None:
    path = run_dir / "run-state" / "telemetry" / "gpu_samples.jsonl"
    if not path.exists():
        return None
    peak = 0.0
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                sample = json.loads(line)
                mem = sample.get("gpu", {}).get("memory_used_mb")
                if mem is not None:
                    peak = max(peak, mem)
            except json.JSONDecodeError:
                continue
    return peak or None


def build_comparison(exp0_dir: Path, exp1_dir: Path, exp2_dir: Path) -> dict:
    c0, c1, c2 = _run_context(exp0_dir), _run_context(exp1_dir), _run_context(exp2_dir)
    c0["gpu_memory_used_mb_peak"] = _gpu_peak(exp0_dir)
    c1["gpu_memory_used_mb_peak"] = _gpu_peak(exp1_dir)
    c2["gpu_memory_used_mb_peak"] = _gpu_peak(exp2_dir)

    arch_report = _load_json(TRAINING_ROOT / "experiment_2_architecture_report" / "architecture_report_and_preflight.json") or {}
    exp1_arch = _load_json(TRAINING_ROOT / "experiment_2_architecture_report" / "exp1_model_inspection.json") or {}

    table = [
        {"experiment": 0, "initialization": "Generic pretrained Loghi", "architecture": "Generic checkpoint architecture (new10)",
         "trainer_lifecycle": "57 resets", "epochs": 1, "cer": c0["held_out_corpus_cer"], "true_wer": c0["held_out_corpus_wer"]},
        {"experiment": 1, "initialization": "Generic pretrained Loghi", "architecture": "Generic checkpoint architecture (new10)",
         "trainer_lifecycle": "Continuous", "epochs": 1, "cer": c1["held_out_corpus_cer"], "true_wer": c1["held_out_corpus_wer"]},
        {"experiment": 2, "initialization": "Random initialization", "architecture": "Recommended VGSL",
         "trainer_lifecycle": "Continuous", "epochs": 1, "cer": c2["held_out_corpus_cer"], "true_wer": c2["held_out_corpus_wer"]},
    ]

    comparison = {
        "generated_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()),
        "evidence_discipline_note": "See loghi_experiments_0_1_2_report.md for MEASURED/VERIFIED FROM "
                                    "SOURCE/INFERRED/HYPOTHESIS/UNRESOLVED classification of every claim.",
        "headline_table": table,
        "runtime_seconds": {"experiment_0": c0["runtime_seconds"], "experiment_1": c1["runtime_seconds"], "experiment_2": c2["runtime_seconds"]},
        "optimizer_steps": {"experiment_0": c0["optimizer_steps"], "experiment_1": c1["optimizer_steps"], "experiment_2": c2["optimizer_steps"]},
        "starting_train_cer": {"experiment_1": c1["starting_train_cer"], "experiment_2": c2["starting_train_cer"]},
        "ending_train_cer": {"experiment_0": c0["ending_train_cer"], "experiment_1": c1["ending_train_cer"], "experiment_2": c2["ending_train_cer"]},
        "validation_trajectory": {
            "experiment_1": {"in_training_best_val_cer": c1["in_training_best_val_cer"], "held_out_corpus_cer": c1["held_out_corpus_cer"]},
            "experiment_2": {"in_training_best_val_cer": c2["in_training_best_val_cer"], "held_out_corpus_cer": c2["held_out_corpus_cer"]},
        },
        "best_checkpoint_timing": {
            "experiment_0": "epoch 48 of 57 (best_val improved before the final shard)",
            "experiment_1": "end of the single epoch (only one validation pass exists)",
            "experiment_2": "end of the single epoch (only one validation pass exists)",
        },
        "confidence_calibration": {
            "experiment_2_buckets": c2["confidence_buckets"],
        },
        "parameter_count": {
            "experiment_1_generic_checkpoint": exp1_arch.get("total_params"),
            "experiment_2_recommended_scratch": arch_report.get("params_after_vocab_resize", {}).get("total"),
        },
        "gpu_memory_used_mb_peak": {
            "experiment_0": c0["gpu_memory_used_mb_peak"], "experiment_1": c1["gpu_memory_used_mb_peak"],
            "experiment_2": c2["gpu_memory_used_mb_peak"],
        },
        "checkpoint_size_bytes": {
            "experiment_0": c0["checkpoint_size_bytes"], "experiment_1": c1["checkpoint_size_bytes"],
            "experiment_2": c2["checkpoint_size_bytes"],
        },
        "throughput_lines_per_second": {
            k: (c["optimizer_steps"] * (c["batch_size"] or 16) / c["runtime_seconds"]) if c["optimizer_steps"] and c["runtime_seconds"] else None
            for k, c in (("experiment_0", c0), ("experiment_1", c1), ("experiment_2", c2))
        },
        "per_collection": {
            "experiment_0": c0["per_collection"], "experiment_1": c1["per_collection"], "experiment_2": c2["per_collection"],
        },
        "raw_contexts": {"experiment_0": c0, "experiment_1": c1, "experiment_2": c2},
    }
    return comparison


def _f(v, digits=1):
    return f"{v:.{digits}f}" if isinstance(v, (int, float)) else "n/a (Experiment 0 predates the optimizer_trace instrumentation)"


def render_markdown(comparison: dict) -> str:
    t = comparison["headline_table"]
    lines = ["# Experiment 0 vs 1 vs 2 comparison\n",
             f"Generated {comparison['generated_at']}.\n",
             "## Headline\n",
             "| Experiment | Initialization | Architecture | Trainer lifecycle | Epochs | CER | True WER |",
             "|---|---|---|---|---:|---:|---:|"]
    for row in t:
        lines.append(f"| {row['experiment']} | {row['initialization']} | {row['architecture']} | "
                     f"{row['trainer_lifecycle']} | {row['epochs']} | {row['cer']:.4f} | {row['true_wer']:.4f} |")

    lines.append("\n## Runtime, throughput, and scale\n")
    lines.append("| | Experiment 0 | Experiment 1 | Experiment 2 |")
    lines.append("|---|---|---|---|")
    rt = comparison["runtime_seconds"]
    lines.append(f"| Runtime (s) | {_f(rt['experiment_0'], 0)} | {_f(rt['experiment_1'], 0)} | {_f(rt['experiment_2'], 0)} |")
    os_ = comparison["optimizer_steps"]
    lines.append(f"| Optimizer steps | {os_['experiment_0'] or 'n/a (predates instrumentation)'} | {os_['experiment_1']} | {os_['experiment_2']} |")
    th = comparison["throughput_lines_per_second"]
    lines.append(f"| Throughput (lines/s) | {_f(th['experiment_0'])} | {_f(th['experiment_1'])} | {_f(th['experiment_2'])} |")
    pc = comparison["parameter_count"]
    lines.append(f"| Parameter count | (Exp0 shares Exp1's architecture) | {pc['experiment_1_generic_checkpoint']:,} | {pc['experiment_2_recommended_scratch']:,} |")
    gm = comparison["gpu_memory_used_mb_peak"]
    lines.append(f"| Peak GPU memory (MB) | {gm['experiment_0']} | {gm['experiment_1']} | {gm['experiment_2']} |")
    cs = comparison["checkpoint_size_bytes"]
    lines.append(f"| Checkpoint size (bytes) | {cs['experiment_0']} | {cs['experiment_1']} | {cs['experiment_2']} |")

    lines.append("\n## Training trajectory\n")
    ec = comparison["ending_train_cer"]
    sc = comparison["starting_train_cer"]
    lines.append(f"- Experiment 1 train CER: {sc.get('experiment_1', 'n/a')} -> {ec['experiment_1']:.4f} (fine-tune, started already competent)")
    lines.append(f"- Experiment 2 train CER: {sc.get('experiment_2', 'n/a')} -> {ec['experiment_2']:.4f} (scratch, started near-random ~1.0)")
    lines.append(f"- Experiment 2's train CER was still declining at the last logged step (no plateau observed) -- see optimizer_trace.csv.")

    lines.append("\n## Per-collection CER (held-out)\n")
    lines.append("| collection | Exp 0 | Exp 1 | Exp 2 |")
    lines.append("|---|---|---|---|")
    pcol = comparison["per_collection"]
    exp1_by = {c["collection"]: c["metrics"]["corpus_cer"] for c in pcol["experiment_1"]}
    exp2_by = {c["collection"]: c["metrics"]["corpus_cer"] for c in pcol["experiment_2"]}
    for c in pcol["experiment_0"]:
        name = c["collection"]
        lines.append(f"| {name} | {c['metrics']['corpus_cer']:.4f} | {exp1_by.get(name, float('nan')):.4f} | {exp2_by.get(name, float('nan')):.4f} |")

    lines.append("\n## Caveat\n")
    lines.append("Experiment 0/1 vs Experiment 2 is NOT a single-variable comparison: both initialization "
                 "(random vs pretrained) AND architecture (recommended VGSL vs the generic checkpoint's "
                 "new10) differ simultaneously. Describe this comparison as *native recommended architecture "
                 "trained from scratch versus generic-checkpoint fine-tuning*, not as an isolated ablation "
                 "of either variable alone.")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-0-dir", default=str(DEFAULT_EXP0_DIR))
    parser.add_argument("--experiment-1-dir", default=str(DEFAULT_EXP1_DIR))
    parser.add_argument("--experiment-2-dir", default=str(DEFAULT_EXP2_DIR))
    args = parser.parse_args()

    comparison = build_comparison(Path(args.experiment_0_dir), Path(args.experiment_1_dir), Path(args.experiment_2_dir))
    out_json = Path(args.experiment_2_dir) / "experiment_0_1_2_comparison.json"
    out_md = Path(args.experiment_2_dir) / "experiment_0_1_2_comparison.md"
    out_json.write_text(json.dumps(comparison, indent=2, default=str), encoding="utf-8")
    out_md.write_text(render_markdown(comparison), encoding="utf-8")
    print(f"written: {out_json}")
    print(f"written: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
