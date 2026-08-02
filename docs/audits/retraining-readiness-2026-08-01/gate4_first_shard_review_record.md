# Gate 4 — First Shard Review: Record of the First Real Attempt

**Date:** 2026-08-02
**Run:** `training/full-corpus-20260802T024341Z` (prepared at commit `ef5391b`)
**Outcome:** **FAILED** — and correctly so. This record preserves the evidence.

This is the first time any part of the full-corpus workflow was executed against real training data.
It was authorised explicitly by the operator ("run exactly one epoch"), bounded to a single shard by a
`--hours 0.1` wall-clock budget, and is exactly the review gate the readiness audit defined.

---

## Result

```
stop_reason              epoch_failed
shards_completed         0  (1 attempted)
runtime                  53s   (expected ~482s)
epoch_output/epoch_1/    empty -- container produced nothing
run_state.status         failed        (recorded honestly)
```

Against the checklist items:

| Check | Result |
|---|---|
| Expected vs actual runtime | 53 s vs ~482 s — failed fast |
| Throughput | n/a — no epoch completed |
| GPU health | container started, GPU engaged (12% util, 45 °C), no OOM |
| Loss/CER sanity | no metrics produced |
| Checkpoint validation | **no checkpoint created** |
| Telemetry continuity | telemetry written; `status.json` + `gpu_samples.jsonl` present |
| Disk growth | negligible |
| Stop criteria | **STOP** — did not proceed to shard 2 |

## Root cause

The container was handed `shard_00000.parquet` as `--train_list`, and
`val_manifest.parquet` as `--validation_list`.

Upstream `loghi-htr` `data/manager.py:257` reads a training list as UTF-8 text and splits each line on
tab into `<image_path>\t<ground_truth>`. It cannot read Parquet. Separately, the line images had never
been extracted for the full corpus at all — the run directory contained no `prepared-data/`, whereas
the pilot had `prepared-data/images/` (10,999 PNGs, 1.6 GB) plus `train_list.txt`.

The model itself loaded correctly first: the staged parent checkpoint was converted normally
(372,591,597 → 124,288,226 bytes with a `.old` backup), which is why the failure took 53 s rather than
failing instantly. It died on data, not on the model.

## Why three prior audit passes missed it

- The **preflight smoke test** used `DEFAULT_PILOT_RUN_DIR / "prepared-data" / "probe_train_list.txt"` —
  the *pilot's* real text list with real extracted images. It therefore exercised a data path the full
  run never uses, and passed 21/21 on four separate occasions while the real shard input was
  unreadable.
- The **dry-run** stops before invoking the trainer by design, so it never touched this.
- The audits verified shard **contents** exhaustively — line IDs, overlap, duplicates, hashes,
  determinism — but never that shards were in a **format the container could consume**. Content
  correctness and interface correctness are different properties, and only the former was checked.

The generalisable lesson: every prior verification either used known-good pilot data or stopped short
of the trainer. Nothing had ever fed real full-corpus artifacts to the real production interface.

## Safety behaviour — all correct

- Original Loghi checkpoint hash unchanged: `0da2c00ab2b12b23e9f64c01ec67ad29724f275eeadb4561843f8a29ff6fff95`
- Full-corpus checkpoints created: **0**
- Optimizer steps on the real corpus: **0**
- Containers left running: **0**
- Run status recorded honestly as `failed`, with `failure_detail`
- Stopped cleanly at a shard boundary; did not cascade into shard 2

The orchestrator's `epoch_failed` handling, the staging discipline, and the wall-clock bound all
behaved as designed. **Gate 4 did precisely the job it was created to do**: it caught a defect that
would otherwise have surfaced hours into an unattended multi-day run.

## Disposition

Fixed in commit `cbf9b21` (`shard_training_data.py` plus wiring). See
`opus_independent_review.md` and the revised risk register (R-023).

This run directory is **preserved unlaunched as evidence** and must not be reused — it predates the
fix and has no `training_data_paths.json`, so `_run_session` now refuses it explicitly rather than
rediscovering the fault ~50 s into a container start.
