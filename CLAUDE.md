# ArchiveTrust -- agent notes

Read first: `docs/PROJECT_STATE.md`, then `docs/BENCHMARK_PROTOCOL.md` and `docs/DATASETS.md`.
`AGENTS.md` also applies (ChatGPT context-pack upkeep).

## Current phase: independent benchmark, not training

- **Do not train, fine-tune, resume or modify the Loghi model.** The epoch-7 checkpoint is frozen
  and pinned by SHA-256 in `src/archivetrust/htr/benchmark/model_registry.py`. The backup is the
  GitHub Release `model-loghi-swedish-scratch-exp2-epoch7`.
- Decoding and scoring decisions are locked in `docs/BENCHMARK_PROTOCOL.md` §1a. Do not change them
  after any benchmark result exists.
- Do not delete historical experiments, training runs or the training pipeline. Do not refactor
  broadly.
- Benchmark data lives in `benchmark-data/` (gitignored):
  - never write into `incoming/`;
  - never overwrite a frozen `benchmark/<id>/` or a predictions file;
  - never let model output inform GT decisions.
- Never put benchmark data into any training material.

## Working here

- Harness: `src/archivetrust/htr/benchmark/`. CLI: `python -m archivetrust.htr.benchmark --help`.
  Tests: `tests/htr/benchmark/`.
- Tests: `PYTHONPATH=src python -m pytest -q tests/htr/benchmark`.
- Readiness: `.\scripts\check_benchmark.ps1 [-Python <path>] [-Benchmark <id>]`.
- The repo `.venv` is broken (copied from another machine). Use a fresh venv, or pass `-Python`.
- `.env` holds `HF_TOKEN`. Never print it.
