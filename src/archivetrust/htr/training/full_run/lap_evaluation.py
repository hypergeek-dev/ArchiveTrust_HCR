"""End-of-lap evaluation: the evidence produced at a completed corpus lap, not at a shard.

**Why this is lap-scoped.** A shard is 1/57th of the corpus. Its validation figure moves by less than
the noise between adjacent shards -- lap 1's shards 49-53 all sat within 0.002 CER of each other --
so a single shard's result is not evidence of convergence, overfitting, or anything else worth acting
on. This module runs only when a full pass over the corpus has completed, which is the first point at
which validation movement reflects the whole corpus.

**What it produces**, all from one real inference pass over the full 1,000-line validation set with
the lap's checkpoint:

- corpus CER/WER overall (edit-weighted, comparable with the container's own `CERMetric`),
- the same per collection, so a corpus-wide average cannot hide a collapsing subgroup --
  `alvsborgs_losen` is tracked explicitly because its terse tax entries are the shortest lines in the
  corpus and its CER has run far above the mean,
- confidence calibration over all 1,000 lines: buckets with counts and mean/median CER, plus
  rejection coverage at candidate thresholds,
- a deterministic stratified inspection sample covering every collection, for human reading.

The inspection sample is *stratified and seeded*, never the head of a file. Reading the first N lines
of `val_list.txt` samples one collection, because the list is grouped by collection -- an earlier
manual review did exactly that and drew conclusions about the model from `alvsborgs_losen` alone.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from archivetrust.htr.training.full_run.evaluation_metrics import (
    ConfidenceBucket,
    CorpusMetrics,
    character_error_rate,
    confidence_buckets,
    corpus_metrics,
    rejection_coverage,
)

COLLECTION_SUFFIX = "_lines__"
"""Extracted line images are named `<collection>_lines__<source>.parquet__<row>.png`, so the
collection is the segment before the first `_lines__`. Derived from the real filenames rather than a
separate lookup table, which would be one more thing to keep in sync."""


def collection_of(image_path: str) -> str:
    name = Path(image_path).name
    head, sep, _ = name.partition(COLLECTION_SUFFIX)
    return head if sep else "unknown"


def parse_ground_truth_list(text: str) -> dict[str, str]:
    """`<image_path>\\t<ground_truth>` -- the exact UTF-8 format the container's data manager reads.

    Ground truth may itself contain tab characters, so the split is bounded to the first one."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        path, sep, truth = line.partition("\t")
        if sep:
            out[path] = truth
    return out


def parse_results_file(text: str) -> dict[str, tuple[float, str]]:
    """`<image_path>\\t<confidence>\\t<prediction>` -- the container's `--results_file` format.

    A prediction can legitimately be empty (the model output nothing for that line); that is a real
    result and is kept, not dropped, because discarding it would silently improve every metric."""
    out: dict[str, tuple[float, str]] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t", 2)
        if len(parts) < 2:
            continue
        path, raw_confidence = parts[0], parts[1]
        prediction = parts[2] if len(parts) > 2 else ""
        try:
            confidence = float(raw_confidence)
        except ValueError:
            continue
        out[path] = (confidence, prediction)
    return out


class LineResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    image_path: str
    collection: str
    ground_truth: str
    prediction: str
    confidence: float
    cer: float


def join_predictions(
    ground_truth: dict[str, str], results: dict[str, tuple[float, str]]
) -> tuple[list[LineResult], list[str]]:
    """Returns `(joined, missing)`. `missing` is every validation line the inference pass produced no
    result for -- reported rather than ignored, because a silently shrinking denominator would make
    the CER look better the more lines failed."""
    joined: list[LineResult] = []
    missing: list[str] = []
    for path, truth in ground_truth.items():
        if path not in results:
            missing.append(path)
            continue
        confidence, prediction = results[path]
        joined.append(LineResult(
            image_path=path, collection=collection_of(path), ground_truth=truth,
            prediction=prediction, confidence=confidence,
            cer=character_error_rate(truth, prediction),
        ))
    return joined, missing


class CollectionMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    collection: str
    metrics: CorpusMetrics
    mean_confidence: float


def per_collection_metrics(lines: list[LineResult]) -> list[CollectionMetrics]:
    """Sorted worst-CER-first: the question at a decision gate is which subgroup is failing, so the
    answer should be the first row rather than something the reader has to search for."""
    by_collection: dict[str, list[LineResult]] = {}
    for line in lines:
        by_collection.setdefault(line.collection, []).append(line)

    out = [
        CollectionMetrics(
            collection=name,
            metrics=corpus_metrics([(r.ground_truth, r.prediction) for r in rows]),
            mean_confidence=sum(r.confidence for r in rows) / len(rows),
        )
        for name, rows in by_collection.items()
    ]
    return sorted(out, key=lambda c: c.metrics.corpus_cer, reverse=True)


def stratified_inspection_sample(
    lines: list[LineResult], *, per_collection: int, seed_material: str
) -> list[LineResult]:
    """A fixed, reproducible sample with equal representation from every collection.

    Seeded from `seed_material` (the checkpoint identity) rather than a wall-clock or arbitrary seed,
    so the same lap always yields the same sample and two laps can be read side by side. Selection is
    by a stable hash of the image path, so it does not depend on dict iteration order, on how many
    lines a collection happens to have, or on Python's PRNG implementation.
    """
    by_collection: dict[str, list[LineResult]] = {}
    for line in lines:
        by_collection.setdefault(line.collection, []).append(line)

    out: list[LineResult] = []
    for name in sorted(by_collection):
        rows = sorted(
            by_collection[name],
            key=lambda r: hashlib.sha256(f"{seed_material}:{r.image_path}".encode()).hexdigest(),
        )
        out.extend(rows[:per_collection])
    return out


class LapEvaluation(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    checkpoint_dir: str
    checkpoint_hash: str
    epochs_completed: int
    global_shards_completed: int
    shards_per_epoch: int

    validation_line_count: int
    scored_line_count: int
    missing_result_count: int
    """Validation lines the inference pass returned nothing for. Non-zero means every figure below is
    computed over a subset, and the report says so rather than quietly renormalising."""

    overall: CorpusMetrics
    per_collection: list[CollectionMetrics]
    worst_collection: str
    best_collection: str

    confidence_buckets: list[ConfidenceBucket]
    rejection_coverage: list[dict]

    inspection_sample: list[LineResult]

    comparison: dict
    """Prior reference points this lap is measured against, each labelled with what it actually was --
    a pilot pass over 9,999 lines is not comparable to a corpus lap, and saying so is part of the
    evidence."""


def evaluate_lap(
    *,
    run_id: str,
    checkpoint_dir: str,
    checkpoint_hash: str,
    epochs_completed: int,
    global_shards_completed: int,
    shards_per_epoch: int,
    ground_truth_text: str,
    results_text: str,
    comparison: dict,
    inspection_per_collection: int = 3,
    rejection_thresholds: tuple[float, ...] = (0.0, 0.5, 0.7, 0.8, 0.9, 0.95),
) -> LapEvaluation:
    ground_truth = parse_ground_truth_list(ground_truth_text)
    results = parse_results_file(results_text)
    lines, missing = join_predictions(ground_truth, results)

    collections = per_collection_metrics(lines)
    calibration_records = [(r.confidence, r.cer) for r in lines]

    return LapEvaluation(
        run_id=run_id, checkpoint_dir=checkpoint_dir, checkpoint_hash=checkpoint_hash,
        epochs_completed=epochs_completed, global_shards_completed=global_shards_completed,
        shards_per_epoch=shards_per_epoch,
        validation_line_count=len(ground_truth),
        scored_line_count=len(lines),
        missing_result_count=len(missing),
        overall=corpus_metrics([(r.ground_truth, r.prediction) for r in lines]),
        per_collection=collections,
        worst_collection=collections[0].collection if collections else "",
        best_collection=collections[-1].collection if collections else "",
        confidence_buckets=confidence_buckets(calibration_records),
        rejection_coverage=rejection_coverage(calibration_records, thresholds=rejection_thresholds),
        inspection_sample=stratified_inspection_sample(
            lines, per_collection=inspection_per_collection, seed_material=checkpoint_hash,
        ),
        comparison=comparison,
    )


def save_lap_evaluation(path: Path, evaluation: LapEvaluation) -> None:
    """Atomic, same write-temp-then-`os.replace` discipline as every other state file here."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(evaluation.model_dump_json(indent=2))
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def build_inference_argv(
    *, image_ref: str, model_host: Path, output_host: Path, inference_list_host: Path,
    batch_size: int, gpu_flag: str = "all",
) -> list[str]:
    """Inference-only container invocation, mirroring `container_epoch_runner.py`'s mount discipline:
    the list file's *parent* is mounted at `/lists`, because the image paths inside the list are
    already written as `/lists/images/...`. The model directory is mounted read-write for the same
    real reason the training runner documents -- the tokenizer loader writes `tokenizer.json` back
    into it when loading a legacy checkpoint. Host paths are passed through unchanged, matching that
    runner: the Windows Docker CLI handles `D:\\...`-style paths in `-v` mounts itself.
    """
    return [
        "docker", "run", "--rm", "--gpus", gpu_flag,
        "-v", f"{model_host}:/model",
        "-v", f"{output_host}:/output",
        "-v", f"{inference_list_host.parent}:/lists:ro",
        "--entrypoint", "python3", image_ref, "main.py",
        "--model", "/model",
        "--inference_list", f"/lists/{inference_list_host.name}",
        "--results_file", "/output/results.txt",
        "--output", "/output",
        "--batch_size", str(batch_size),
        "--gpu", "0",
    ]


def render_report(evaluation: LapEvaluation) -> str:
    """Markdown for a human reading the decision gate."""
    e = evaluation
    lines = [
        f"# End-of-lap evaluation -- lap {e.epochs_completed}",
        "",
        f"- run: `{e.run_id}`",
        f"- checkpoint: `{e.checkpoint_dir}`",
        f"- checkpoint sha256: `{e.checkpoint_hash}`",
        f"- position: {e.global_shards_completed} shards = {e.epochs_completed} "
        f"full corpus lap(s) of {e.shards_per_epoch} shards each",
        f"- validation lines: {e.scored_line_count} scored of {e.validation_line_count}"
        + (f" -- **{e.missing_result_count} produced no result**" if e.missing_result_count else ""),
        "",
        "## Overall",
        "",
        f"| metric | value |",
        f"| --- | --- |",
        f"| corpus CER | **{e.overall.corpus_cer:.4f}** |",
        f"| corpus WER (true word error rate) | {e.overall.corpus_wer:.4f} |",
        f"| line error rate (what the container calls \"WER\") | {e.overall.line_error_rate:.4f} |",
        f"| mean per-line CER | {e.overall.mean_per_line_cer:.4f} |",
        f"| total edits | {e.overall.total_edits:,} |",
        f"| total reference chars | {e.overall.total_reference_chars:,} |",
        "",
        "Corpus CER is edit-weighted (`total edits / total reference chars`) and is the figure "
        "comparable with the container's own `CERMetric`. The per-line mean is shown beside it "
        "because they diverge when line lengths vary.",
        "",
        "## Per collection (worst first)",
        "",
        "| collection | CER | WER | lines | ref chars | mean confidence |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for c in e.per_collection:
        lines.append(
            f"| {c.collection} | {c.metrics.corpus_cer:.4f} | {c.metrics.corpus_wer:.4f} | "
            f"{c.metrics.sample_count} | {c.metrics.total_reference_chars:,} | {c.mean_confidence:.3f} |"
        )

    lines += ["", "## Confidence calibration", "",
              "| confidence bucket | lines | mean CER | median CER |", "| --- | --- | --- | --- |"]
    for b in e.confidence_buckets:
        mean = f"{b.mean_cer:.4f}" if b.mean_cer is not None else "-- (no lines)"
        median = f"{b.median_cer:.4f}" if b.median_cer is not None else "-- (no lines)"
        lines.append(f"| {b.lower:.1f} - {b.upper:.2f} | {b.sample_count} | {mean} | {median} |")

    lines += ["", "### Rejection coverage", "",
              "| threshold | coverage | accepted lines | CER of accepted |", "| --- | --- | --- | --- |"]
    for row in e.rejection_coverage:
        cer = f"{row['mean_cer_of_accepted']:.4f}" if row["mean_cer_of_accepted"] is not None else "--"
        lines.append(
            f"| {row['threshold']:.2f} | {row['coverage']:.1%} | {row['accepted_count']} | {cer} |"
        )

    lines += ["", "## Inspection sample", "",
              f"Stratified: the same number of lines from every collection, selected by a stable hash "
              f"seeded on the checkpoint identity, so this sample is reproducible and comparable "
              f"across laps.", ""]
    for r in e.inspection_sample:
        lines += [
            f"**{r.collection}** -- CER {r.cer:.3f}, confidence {r.confidence:.3f}",
            "",
            f"- truth: `{r.ground_truth}`",
            f"- pred:  `{r.prediction}`",
            "",
        ]

    lines += ["## Comparison with prior reference points", "", "```json",
              json.dumps(e.comparison, indent=2, ensure_ascii=False), "```", ""]
    return "\n".join(lines)
