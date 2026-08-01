# Running the Lion-vs-Loghi Comparison

Status: Current — describes the launcher as built; the full Swedish-Dutch comparison has not been run.

## Prerequisites

* `dataset-rgb/` restored (Swedish corpus — gitignored, not committed).
* `dataset-dutch-rgb/` extracted (Dutch corpus — see `dataset-provenance-dutch.md`; already present if
  you followed this integration's own setup).
* `swedish_lion`: main-venv `torch`/`transformers` install (same as Florence-2's).
* `loghi`: Docker Desktop (with WSL2 backend on Windows) or native Docker on Linux, **plus** real pins
  in `providers/loghi/pinned_versions.py` — see `docs/methods/loghi.md`'s "Install / Setup" section
  (in `providers/loghi/README.md`). Neither is done as part of this integration.

## The launcher

```
PYTHONPATH=src .venv/Scripts/python.exe scripts/run_lion_loghi_comparison.py
```

```
1. Check environments        -- real validate_environment() probes for both active methods
2. Prepare Swedish pages     -- real count from dataset-rgb/
3. Prepare Dutch pages       -- real count from the Dutch manifest, + provenance/licensing pointer
4. Run Loghi smoke test      -- 5 Swedish + 5 Dutch pages, real adapter call, honest result
5. Run Swedish Lion I locally -- real local TrOCR inference (no export/import step -- see README.md's
                                  "correction" section for why)
6. Build four-cell experiment family -- constructs and registers the four ExperimentVersions
7. Run or resume comparison  -- NOT AVAILABLE YET (see below)
8. Review outputs            -- lists registered research phases/comparison groups
9. Generate report           -- writes a real status JSON (environment/dataset presence)
0. Exit
```

Every action that touches real state prints what it is about to do and asks `[y/N]` first.

## Option 7 is intentionally not implemented

The brief's explicit instruction: **stop at the supervised feasibility checkpoint before launching the
full Swedish-Dutch experiment.** Option 4 (smoke test) is the checkpoint. A full run — the entire
Swedish and Dutch corpora, both methods, all four cells — is a separate, larger, explicitly-triggered
step that a human should review the smoke-test results before starting. This launcher deliberately does
not offer it.

## What "Run Loghi smoke test" actually does right now

On a machine with no Loghi Docker image pulled and placeholder pins (the state this integration leaves
the repository in), option 4 will:

1. Call `LoghiAdapter.validate_environment()` — reports `valid=False` with the specific reason (Docker/
   WSL2 present or absent, pins placeholder or real).
2. Attempt `recognize()` on each page anyway (never skipped just because validation failed — the brief:
   "do not exclude Loghi after one bad page") — each will honestly return `environment_unavailable`.
3. Record every attempt as real telemetry (`LoghiEnvironmentValidated`, `LoghiPipelineStarted`, per-page
   failure) and write a real JSON report to `docs/experiments/lion-loghi-comparison/runs/` (gitignored).

This is the honest, correct behavior at this checkpoint — not a bug, and not something to "fix" by
fabricating a pass. Once a real Loghi environment exists, the same option produces real per-stage
results with no code change required.

## After the smoke test is reviewed

1. Confirm or correct the Dutch corpus's license (`dataset-provenance-dutch.md`).
2. Pin a real `LoghiComponentVersions` and re-run option 1 until `loghi: valid=True`.
3. Re-run option 4 against real containers; review the resulting PAGE XML/text with a human.
4. Only then consider building a full-corpus runner — which does not exist yet, and would be a new,
   explicitly-scoped piece of work, not an extension of this launcher's option 7 placeholder.
