"""Run the end-of-lap evaluation suite against a completed corpus lap.

Performs one real container inference pass over the full 1,000-line validation set using the lap's
checkpoint, then writes `lap_evaluation.json` and `lap_evaluation.md` into the run directory.

Refuses to run unless a full lap has actually completed -- a shard is 1/57th of the corpus, and this
suite exists precisely to stop shard-level noise being read as epoch-level evidence.

    python scripts/evaluate_full_run_lap.py --run training/full-corpus-<stamp>

`--checkpoint-kind` picks which checkpoint of the lap to evaluate: `end_of_epoch` (default, the state
the model is actually in at the boundary) or `best_val` (the best-scoring shard checkpoint, which is
what a deployment would ship).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from archivetrust.htr.training.checkpoint_index import load_index, verify_checkpoint
from archivetrust.htr.training.full_run.lap_evaluation import (
    build_inference_argv,
    evaluate_lap,
    render_report,
    save_lap_evaluation,
)
from archivetrust.htr.training.full_run.run_state import load_run_state

PILOT_REFERENCE = {
    "pilot_best_val_cer": 0.1691,
    "pilot_best_val_cer_note": (
        "Best validation CER reached by the 9,999-line pilot training pass. NOT a corpus lap: the "
        "pilot repeatedly revisited one fixed 9,999-line subset, so it is a different quantity, "
        "shown for orientation rather than as a like-for-like target."
    ),
    "pilot_single_pass_cer": 0.2481,
    "pilot_single_pass_cer_note": (
        "CER after a single 9,999-line pilot training pass -- roughly comparable in optimizer steps "
        "to one full-corpus shard, not to a lap."
    ),
}


def _resolve_checkpoint(index_path: Path, kind: str) -> tuple[str, str, int]:
    entries = [e for e in load_index(index_path) if e.checkpoint_kind == kind]
    if not entries:
        raise SystemExit(f"no `{kind}` checkpoint recorded in {index_path}")
    latest = max(entries, key=lambda e: (e.epoch, e.created_at))
    return latest.checkpoint_dir, (latest.model_file_hash or ""), latest.epoch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--checkpoint-kind", default="end_of_epoch",
                        choices=("end_of_epoch", "best_val", "latest"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--inspection-per-collection", type=int, default=3)
    parser.add_argument("--image", default=None, help="defaults to the run's pinned container image")
    parser.add_argument("--allow-incomplete-lap", action="store_true",
                        help="evaluate even though no full corpus lap has completed; the report "
                             "records that its figures are not lap-scoped evidence")
    args = parser.parse_args()

    run_dir = Path(args.run).resolve()
    run_state_dir = run_dir / "run-state"
    state = load_run_state(run_state_dir)

    if state.epochs_completed < 1 and not args.allow_incomplete_lap:
        raise SystemExit(
            f"Refusing to evaluate: {state.shards_completed_in_current_epoch}/"
            f"{state.shards_per_epoch} shards completed, which is not a full corpus lap. "
            f"A shard's validation figure moves by less than the noise between adjacent shards. "
            f"Pass --allow-incomplete-lap only if you intend a non-lap-scoped diagnostic."
        )

    paths = json.loads((run_dir / "training_data_paths.json").read_text(encoding="utf-8"))
    validation_list = Path(paths["validation_list_path"])

    manifest = json.loads((run_dir / "launch_manifest.json").read_text(encoding="utf-8"))
    image_ref = args.image or (
        f"{manifest['container_image_name']}@{manifest['container_image_digest']}"
        if manifest.get("container_image_digest") else manifest["container_image_name"]
    )

    checkpoint_dir, checkpoint_hash, checkpoint_shard = _resolve_checkpoint(
        run_state_dir / "checkpoint_index.json", args.checkpoint_kind
    )

    # Recorded paths are repo-relative; docker -v needs absolute.
    repo_root = Path(__file__).resolve().parents[1]
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.is_absolute():
        checkpoint_path = (repo_root / checkpoint_path).resolve()

    ok, reason, _ = verify_checkpoint(checkpoint_path)
    if not ok:
        raise SystemExit(f"checkpoint failed verification before evaluation: {reason}")

    output_dir = run_dir / "lap-evaluation" / f"lap{state.epochs_completed}_{args.checkpoint_kind}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # The model directory has to be mounted read-write, because the tokenizer loader writes a
    # converted `tokenizer.json` back into it. Mounting the recorded checkpoint directly would let an
    # evaluation mutate the very artifact it is evidence about, so a copy is staged and mounted
    # instead -- the same reason `training_session.py` never hands the pristine base checkpoint to the
    # container either.
    staged = output_dir / "staged_checkpoint"
    if staged.exists():
        shutil.rmtree(staged)
    shutil.copytree(checkpoint_path, staged)

    argv = build_inference_argv(
        image_ref=image_ref, model_host=staged, output_host=output_dir,
        inference_list_host=validation_list, batch_size=args.batch_size,
    )
    print(f"checkpoint ({args.checkpoint_kind}, recorded at shard {checkpoint_shard}): {checkpoint_dir}")
    print(f"validation list: {validation_list}")
    print("inference command:\n  " + " ".join(argv), flush=True)

    started = time.monotonic()
    completed = subprocess.run(argv, capture_output=True, text=True, check=False)
    elapsed = time.monotonic() - started
    print(f"inference exited {completed.returncode} in {elapsed:.0f}s")

    results_path = output_dir / "results.txt"
    if not results_path.exists():
        (output_dir / "inference_stderr.txt").write_text(completed.stderr or "", encoding="utf-8")
        raise SystemExit(
            f"inference produced no {results_path}; stderr written beside it. "
            f"Tail:\n{(completed.stderr or '')[-2000:]}"
        )

    evaluation = evaluate_lap(
        run_id=state.run_id,
        checkpoint_dir=checkpoint_dir,
        checkpoint_hash=checkpoint_hash or "unrecorded",
        epochs_completed=state.epochs_completed,
        global_shards_completed=state.global_shards_completed,
        shards_per_epoch=state.shards_per_epoch,
        ground_truth_text=validation_list.read_text(encoding="utf-8"),
        results_text=results_path.read_text(encoding="utf-8"),
        comparison={
            **PILOT_REFERENCE,
            "checkpoint_kind_evaluated": args.checkpoint_kind,
            "checkpoint_recorded_at_shard": checkpoint_shard,
            "run_best_val_cer_during_lap": (state.best_metrics or {}).get("val_cer"),
            "final_shard_metrics": dict(state.latest_metrics or {}),
            "final_shard_metrics_note": (
                "Train and validation CER/WER/loss as the container itself reported them at the end "
                "of the lap's final shard. The train/val gap here is the overfitting signal; the "
                "corpus CER above is an independent recomputation from raw predictions."
            ),
            "in_training_val_cer_note": (
                "The in-training figures above come from the container's own validation at the end "
                "of each shard, on this same 1,000-line set. They are the right thing to compare "
                "this pass against; the pilot numbers are context, not a target."
            ),
            "inference_seconds": round(elapsed, 1),
        },
        inspection_per_collection=args.inspection_per_collection,
    )

    save_lap_evaluation(output_dir / "lap_evaluation.json", evaluation)
    (output_dir / "lap_evaluation.md").write_text(render_report(evaluation), encoding="utf-8")

    print(f"\ncorpus CER {evaluation.overall.corpus_cer:.4f} | "
          f"WER {evaluation.overall.corpus_wer:.4f} | "
          f"scored {evaluation.scored_line_count}/{evaluation.validation_line_count}")
    print(f"worst collection: {evaluation.worst_collection} | best: {evaluation.best_collection}")
    print(f"written: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
