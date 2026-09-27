# Project state

Snapshot: 2026-09-27.

## Where the project is

- **Training is finished.** The final model is the scratch-trained Loghi-HTR Experiment 2 epoch 7
  (`best_val`): val CER 0.0998, WER 0.3273 (beam 10, 1000 in-distribution lines).
  - It is frozen: no further epochs, architecture changes or substitute model.
  - Identity: `loghi-swedish-scratch-exp2-epoch7`. The SHA-256 pins are in
    `src/archivetrust/htr/benchmark/model_registry.py`.
  - Local copy: `training/experiment-2-epoch7-from-epoch6-checkpoint-20260813T124955Z/run-state/epoch_output/epoch_1/recommended/best_val/`
    (gitignored).
  - **Backup:** GitHub Release
    [`model-loghi-swedish-scratch-exp2-epoch7`](https://github.com/hypergeek-dev/ArchiveTrust_HCR/releases/tag/model-loghi-swedish-scratch-exp2-epoch7).
    The assets were verified by download and re-hash on 2026-09-27. The provenance record is
    `docs/models/loghi-swedish-scratch-exp2-epoch7.MODEL_MANIFEST.json`.
- **Next goal:** an independent benchmark of that model against Riksarkivet's Swedish Lion Libre, on
  an external GT dataset from a colleague that has not arrived yet.
  - The harness is `src/archivetrust/htr/benchmark/`.
  - The method is `docs/BENCHMARK_PROTOCOL.md`. Its locked decisions (§1a) are Lion
    `generation_config` and Loghi beam 10.
  - The data is described in `docs/DATASETS.md`, including the provenance checklist for the
    provider.

## Benchmark harness readiness

Run `.\scripts\check_benchmark.ps1`. The table below is updated whenever the machine state
changes.

| Check | State (2026-09-27) |
| --- | --- |
| Harness tests (`tests/htr/benchmark`) | pass (73) |
| Loghi checkpoint hashes | match; backed up externally |
| dataset-rgb mechanical dry run (build) | done: 60 pages → 2,678 crops, deterministic, no findings |
| Python environment | `.venv` rebuilt: Python 3.13.15, see `docs/benchmark-environment.txt` |
| Lion smoke inference | **pass**: 5 dry-run lines, `generation_config`, CUDA, torch 2.13.0+cu130, transformers 4.49.0 (no accuracy measured) |
| Docker / pinned Loghi image / Loghi smoke | **pending**. Docker Desktop is installed, but its engine answers HTTP 500 because WSL 2 is not installed. Firmware virtualisation (SVM) is already on. Fix: `wsl --install` in an admin shell, reboot, start Docker Desktop, `docker pull` the pinned digest. |
| NVIDIA GPU | RTX 3070 8 GB, driver 616.92, visible to torch |
| HF training corpus (overlap check) | not found on any mounted drive, so the contamination check is partial |
| External dataset | not delivered |

## Rebuilding the benchmark environment

```powershell
uv venv .venv --python 3.13 --seed          # or: py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev,benchmark,transformers]" "transformers==4.49.0"
# last, so that timm's torchvision cannot replace the CUDA build with a CPU-only torch from PyPI:
.venv\Scripts\python.exe -m pip install "torch==2.13.0" "torchvision==0.28.0" --index-url https://download.pytorch.org/whl/cu130
```

These versions match the source machine's reference freeze. The `gui`, `watch` and `dashboard`
extras are not needed for the benchmark.

In this environment the full repository suite has 33 failures and 6 errors. None of them are in
the benchmark:

- `pypdfium2` and `pywin32` are used by rendering, geometry validation, interop and the ACL tests,
  but are not declared in `pyproject.toml`.
- The SATRN tests and SATRN baselines need `.venv-satrn`, which is still a broken copy from the old
  machine.

## Historical material kept on purpose

- The training pipeline (`src/archivetrust/htr/training/`), experiment reports (`docs/experiments/`)
  and audits (`docs/audits/`) are history. They are not to be deleted or refactored.
- `src/archivetrust/benchmark/` (pre-HTR telemetry measurement) is unrelated to the HTR benchmark.
