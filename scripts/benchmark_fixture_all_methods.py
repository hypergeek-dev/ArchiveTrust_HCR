"""Warm, single-crop benchmark of Florence-2 and Swedish Lion on the shared fixture line, plus
CER/WER against its published ground truth.

Companion to `scripts/satrn_warm_benchmark.py` (SATRN must run under the isolated `.venv-satrn`
interpreter and is therefore a separate process/script -- see that file's docstring for why).

**"Warm" and "no architectural advantage" -- what that means here.** Both adapters' real facades
already cache the loaded (model, processor) pair in-process after the first call
(`_RealFlorence2Facade`/`_RealSwedishLionFacade`'s `_loaded` dict) -- unlike SATRN's adapter, which
deliberately restarts a subprocess and reloads the model on every call (a real, load-bearing
architectural choice, documented in `providers/satrn/facade.py`, not a defect). Calling
`run_inference` N+1 times in one process and discarding the first call isolates the same thing for
all three methods: the cost of one forward pass through an already-resident model, not the cost of
getting the model into memory. `scripts/satrn_warm_benchmark.py` gives SATRN the same treatment by
loading its model once and timing repeated calls in one subprocess, instead of the one-shot-per-call
path its production adapter uses.

Per-call GPU memory is read via `torch.cuda.reset_peak_memory_stats()` immediately before, then
`torch.cuda.max_memory_allocated()` immediately after, each individual call -- so it reports one
call's peak, not an accumulated whole-process peak.

CER/WER reuse `htr/evaluation/recognition.py::compute_recognition_metrics` (Levenshtein-based,
already tested elsewhere in this codebase) -- never recomputed here.

**Scope note.** This is a single-line, single-crop measurement -- real, but a sample size of one.
It is not a substitute for the corpus-scale reference-free reliability screening
(`docs/experiments/technical-reliability-screening/`), which remains the only source of the
distinct-output / repeat-rate / timing-distribution numbers reported there.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

FIXTURE = REPO_ROOT / "tests" / "fixtures" / "htr" / "trolldomskommissionen_sample_line.jpg"
GROUND_TRUTH_FILE = REPO_ROOT / "tests" / "fixtures" / "htr" / "trolldomskommissionen_sample_line.txt"

WARMUP_CALLS = 1
MEASURED_CALLS = 9


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    return ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0


def _bench_florence2() -> dict:
    from archivetrust.providers.florence2_htr.adapter import (
        DEFAULT_MODEL_ID,
        DEFAULT_MODEL_REVISION,
        DEFAULT_PROCESSOR_MODEL_ID,
        DEFAULT_PROCESSOR_REVISION,
        TASK_PROMPT,
    )
    from archivetrust.providers.florence2_htr.facade import real_florence2_facade

    facade = real_florence2_facade()
    calls = WARMUP_CALLS + MEASURED_CALLS
    results = []
    for i in range(calls):
        result = facade.run_inference(
            image_path=str(FIXTURE),
            device_request="cuda",
            model_id=DEFAULT_MODEL_ID,
            model_revision=DEFAULT_MODEL_REVISION,
            processor_model_id=DEFAULT_PROCESSOR_MODEL_ID,
            processor_revision=DEFAULT_PROCESSOR_REVISION,
            task_prompt=TASK_PROMPT,
        )
        results.append(result)
        print(f"  florence2 call {i + 1}/{calls}: {result.get('elapsed_seconds'):.4f}s", file=sys.stderr)

    warm = results[WARMUP_CALLS:]
    texts = {r["parsed_text"] for r in warm}
    timings = [r["elapsed_seconds"] for r in warm]
    gpu_mb = [r["peak_gpu_memory_mb"] for r in warm]
    return {
        "method": "florence2_htr",
        "model_revision": f"{DEFAULT_MODEL_ID}@{DEFAULT_MODEL_REVISION}",
        "text": warm[-1]["parsed_text"],
        "deterministic_across_warm_calls": len(texts) == 1,
        "distinct_texts_seen": sorted(texts),
        "warm_calls": len(warm),
        "runtime_seconds": {
            "median": _median(timings), "min": min(timings), "max": max(timings), "all": timings
        },
        "peak_gpu_memory_mb": {
            "median": _median(gpu_mb), "min": min(gpu_mb), "max": max(gpu_mb), "all": gpu_mb
        },
    }


def _bench_swedish_lion() -> dict:
    from archivetrust.providers.swedish_lion.adapter import (
        DEFAULT_MODEL_ID,
        DEFAULT_MODEL_REVISION,
        DEFAULT_PROCESSOR_MODEL_ID,
        DEFAULT_PROCESSOR_REVISION,
        DEFAULT_NUM_BEAMS,
    )
    from archivetrust.providers.swedish_lion.facade import real_swedish_lion_facade

    facade = real_swedish_lion_facade()
    calls = WARMUP_CALLS + MEASURED_CALLS
    results = []
    for i in range(calls):
        result = facade.run_inference(
            image_path=str(FIXTURE),
            device_request="cuda",
            model_id=DEFAULT_MODEL_ID,
            model_revision=DEFAULT_MODEL_REVISION,
            processor_model_id=DEFAULT_PROCESSOR_MODEL_ID,
            processor_revision=DEFAULT_PROCESSOR_REVISION,
            num_beams=DEFAULT_NUM_BEAMS,
        )
        results.append(result)
        print(f"  swedish_lion call {i + 1}/{calls}: {result.get('elapsed_seconds'):.4f}s", file=sys.stderr)

    warm = results[WARMUP_CALLS:]
    texts = {r["text"] for r in warm}
    timings = [r["elapsed_seconds"] for r in warm]
    gpu_mb = [r["peak_gpu_memory_mb"] for r in warm]
    return {
        "method": "swedish_lion",
        "model_revision": f"{DEFAULT_MODEL_ID}@{DEFAULT_MODEL_REVISION}",
        "text": warm[-1]["text"],
        "deterministic_across_warm_calls": len(texts) == 1,
        "distinct_texts_seen": sorted(texts),
        "warm_calls": len(warm),
        "runtime_seconds": {
            "median": _median(timings), "min": min(timings), "max": max(timings), "all": timings
        },
        "peak_gpu_memory_mb": {
            "median": _median(gpu_mb), "min": min(gpu_mb), "max": max(gpu_mb), "all": gpu_mb
        },
    }


def main() -> int:
    from archivetrust.htr.evaluation.recognition import compute_recognition_metrics

    ground_truth = GROUND_TRUTH_FILE.read_text(encoding="utf-8").strip()

    print("Benchmarking Florence-2 (warm, in-process)...", file=sys.stderr)
    florence2 = _bench_florence2()
    print("Benchmarking Swedish Lion (warm, in-process)...", file=sys.stderr)
    swedish_lion = _bench_swedish_lion()

    for entry in (florence2, swedish_lion):
        metrics = compute_recognition_metrics(ground_truth, entry["text"])
        entry["cer_normalized"] = metrics.character_error_rate_normalized
        entry["wer_normalized"] = metrics.word_error_rate_normalized
        entry["exact_match_normalized"] = metrics.exact_match_normalized

    output = {
        "ground_truth": ground_truth,
        "fixture": str(FIXTURE),
        "results": [florence2, swedish_lion],
    }
    out_path = REPO_ROOT / "docs" / "experiments" / "technical-reliability-screening" / "fixture-warm-benchmark" / "florence2_and_swedish_lion.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nWritten to {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
