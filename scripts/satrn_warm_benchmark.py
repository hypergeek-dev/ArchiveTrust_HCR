"""Warm, repeated-call SATRN benchmark -- run under the isolated `.venv-satrn` interpreter, NOT
the main project venv (same reason as `providers/satrn/_worker.py`, whose model-loading logic this
duplicates on purpose: `mmocr`/`mmcv`/`mmdet` are not installed in, and are incompatible with, the
main venv). Deliberately imports nothing from `archivetrust`.

**Why this script exists and why it does not touch `providers/satrn/_worker.py` or `facade.py`.**
SATRN's production adapter restarts a subprocess and reloads the model on *every* `recognize()`
call -- a real, deliberate, documented architectural choice (the isolation boundary those two files'
docstrings justify at length), not something to silently change. But it makes the production
adapter's own timing/GPU numbers incomparable to Florence-2's and Swedish Lion's, both of which
cache their model in-process. To get a like-for-like "cost of one forward pass through an
already-resident model" number for SATRN too, this script loads the model **once**, then runs
several inference calls against the same image in that one process, discarding the first as
warmup -- the same treatment `scripts/benchmark_fixture_all_methods.py` gives the other two. It is
a one-off diagnostic measurement, not a change to how SATRN is actually invoked by the real
pipeline.

Per-call GPU memory: `torch.cuda.reset_peak_memory_stats()` immediately before,
`torch.cuda.max_memory_allocated()` immediately after, each individual call.

Prints one JSON object to stdout; diagnostic progress to stderr (same stdout/stderr discipline as
`_worker.py`).
"""

from __future__ import annotations

import fnmatch
import json
import os
import sys
import time

MODEL_ID = "Riksarkivet/satrn_htr"
REVISION = None
WARMUP_CALLS = 1
MEASURED_CALLS = 9


def _hf_download_matching(repo_id: str, pattern: str, revision: str | None) -> str:
    from huggingface_hub import hf_hub_download, list_repo_files

    for filename in list_repo_files(repo_id):
        if fnmatch.fnmatch(filename, pattern):
            return hf_hub_download(repo_id, filename, revision=revision)
    raise FileNotFoundError(f"No file matching {pattern!r} in {repo_id!r}")


def _load_mmlabs(model_id: str, revision: str | None) -> tuple[str, str]:
    """Verbatim logic from `providers/satrn/_worker.py::_load_mmlabs` (duplicated, not imported,
    since that module is deliberately import-isolated and this script needs to run standalone
    too)."""
    from mmengine.config import Config

    weights_path = _hf_download_matching(model_id, "*.pth", revision)
    config_path = _hf_download_matching(model_id, "config.py", revision)
    try:
        dictionary_path = _hf_download_matching(model_id, "dictionary.txt", revision)
    except FileNotFoundError:
        dictionary_path = None
    if dictionary_path is not None:
        cfg = Config.fromfile(config_path)
        cfg.dictionary["dict_file"] = dictionary_path
        cfg.model["decoder"]["dictionary"]["dict_file"] = dictionary_path
        cfg.dump(config_path)
    return weights_path, config_path


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    return ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0


def main() -> int:
    import torch
    from mmocr.apis import TextRecInferencer

    if len(sys.argv) < 2:
        print("usage: satrn_warm_benchmark.py <image_path>", file=sys.stderr)
        return 2
    image_path = sys.argv[1]

    print("Loading SATRN model (once)...", file=sys.stderr)
    weights_path, config_path = _load_mmlabs(MODEL_ID, REVISION)
    inferencer = TextRecInferencer(model=config_path, weights=weights_path, device="cuda")

    calls = WARMUP_CALLS + MEASURED_CALLS
    results = []
    for i in range(calls):
        torch.cuda.reset_peak_memory_stats()
        started = time.time()
        raw = inferencer([image_path], batch_size=1, return_datasamples=False, progress_bar=False)
        elapsed = time.time() - started
        peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        prediction = raw["predictions"][0]
        results.append({"text": prediction["text"], "score": prediction.get("scores"),
                         "elapsed_seconds": elapsed, "peak_gpu_memory_mb": peak_mb})
        print(f"  satrn call {i + 1}/{calls}: {elapsed:.4f}s, {peak_mb:.1f} MB", file=sys.stderr)

    warm = results[WARMUP_CALLS:]
    texts = sorted({r["text"] for r in warm})
    timings = [r["elapsed_seconds"] for r in warm]
    gpu_mb = [r["peak_gpu_memory_mb"] for r in warm]

    output = {
        "method": "satrn",
        "model_revision": MODEL_ID,
        "text": warm[-1]["text"],
        "score": warm[-1]["score"],
        "deterministic_across_warm_calls": len(texts) == 1,
        "distinct_texts_seen": texts,
        "warm_calls": len(warm),
        "runtime_seconds": {
            "median": _median(timings), "min": min(timings), "max": max(timings), "all": timings
        },
        "peak_gpu_memory_mb": {
            "median": _median(gpu_mb), "min": min(gpu_mb), "max": max(gpu_mb), "all": gpu_mb
        },
        "gpu_name": torch.cuda.get_device_name(0),
    }
    print(json.dumps(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
