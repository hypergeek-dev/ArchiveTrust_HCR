# Epoch-4 continuation: fourth-level comparison (Experiment 1 vs Experiment 2)

**Date:** 2026-08-11
**Methodology:** identical to every prior comparison in this project — `scripts/evaluate_full_run_lap.py`,
true Levenshtein CER/WER (never `difflib`), `best_val` checkpoint, all 1,000 held-out validation lines
scored (1000/1000 in every run below), same fixed validation set across all nine runs.

## Design recap

Both epoch-4 runs are **single, uninterrupted `--epochs 4` container invocations** — a fresh restart
from each experiment's original origin (pristine pinned checkpoint for Experiment 1, random
initialization for Experiment 2), not a resume of any preserved lower-epoch run. `model.fit(epochs=4)`
keeps one optimizer/LR-schedule object alive across all four internal Keras epochs in-process, so
optimizer continuity holds by construction — see `scripts/run_experiment_1_four_epochs.py` /
`run_experiment_2_four_epochs.py` for the full rationale (identical reasoning to the epoch-2/epoch-3
drivers: none of the preserved prior runs wrote an `optimizer_state/` checkpoint, so there is nothing
for a cross-process resume to restore from). Both runs completed cleanly this time (exit code 0,
`target_epochs_reached`), both checkpoints verified, launched back to back by the same detached
orchestrator (`training/_orchestration_logs/run_four_epoch_continuations.sh`).

**Operational note:** a transient `PermissionError` (WinError 5, access denied) killed the telemetry
sampler thread partway through Experiment 1's run — a file-lock collision on `status.json`, most likely
OneDrive or antivirus grabbing the file mid-write. This did not affect training: `log.csv` and
`optimizer_trace.csv` (written directly by the training process, not the telemetry sampler) show
continuous, healthy progress throughout, and the run completed and verified normally. Recorded for the
same reason the epoch-3 incident was — this checkout still lives under `OneDrive\Desktop\ArchiveTrust_HCR`
and OneDrive still restarts its sync service on its own between sessions.

## Results — all nine runs, independently evaluated

| Run | Corpus CER | Corpus WER | Notes |
|---|---|---|---|
| Experiment 0 (57-shard, optimizer reset per shard) | 0.1728 | 0.4945 | best_val; the flawed-method baseline |
| Experiment 1, epoch 1 (fine-tune, single continuous epoch) | 0.1747 | 0.4956 | best_val |
| Experiment 1, epoch 2 (fine-tune, two continuous epochs) | 0.1522 | 0.4458 | best_val |
| Experiment 1, epoch 3 (fine-tune, three continuous epochs) | 0.1419 | 0.4276 | best_val |
| **Experiment 1, epoch 4 (fine-tune, four continuous epochs)** | **0.1339** | **0.4112** | best_val; new |
| Experiment 2, epoch 1 (scratch, single continuous epoch) | 0.1918 | 0.5470 | best_val |
| Experiment 2, epoch 2 (scratch, two continuous epochs) | 0.1487 | 0.4546 | best_val |
| Experiment 2, epoch 3 (scratch, three continuous epochs) | 0.1257 | 0.3915 | best_val |
| **Experiment 2, epoch 4 (scratch, four continuous epochs)** | **0.1145** | **0.3674** | best_val; new |

## Epoch-over-epoch deltas

| | CER Δ | CER Δ% | WER Δ | WER Δ% |
|---|---|---|---|---|
| Experiment 1, epoch 1→2 | −0.0225 | −12.9% | −0.0498 | −10.0% |
| Experiment 1, epoch 2→3 | −0.0103 | −6.8% | −0.0182 | −4.1% |
| Experiment 1, epoch 3→4 | −0.0080 | −5.6% | −0.0164 | −3.8% |
| Experiment 2, epoch 1→2 | −0.0431 | −22.5% | −0.0924 | −16.9% |
| Experiment 2, epoch 2→3 | −0.0230 | −15.5% | −0.0631 | −13.9% |
| Experiment 2, epoch 3→4 | −0.0112 | −8.9% | −0.0241 | −6.2% |

## Findings

1. **Both experiments still improved at epoch 4, still no regression, still no overfitting signal** —
   in-training `val_loss` kept declining for both (Experiment 1: 21.9→19.5→18.5→17.1; Experiment 2:
   23.6→18.1→15.9→14.3).
2. **Experiment 1's improvement rate is stabilizing rather than continuing to collapse toward zero.**
   Its relative CER gain went −12.9% → −6.8% → −5.6% — the *rate of slowdown* itself slowed sharply
   between epoch 3 and epoch 4 (the epoch 2→3 gain was 53% of the epoch 1→2 gain; the epoch 3→4 gain
   was 82% of the epoch 2→3 gain). That pattern looks like a curve approaching a floor, not one still
   in free fall — Experiment 1 is the closer of the two to the point where a further epoch stops being
   worth its compute cost.
3. **Experiment 2 is still declining at a fairly consistent proportional rate** (CER: −22.5% → −15.5% →
   −8.9%, each roughly 60-70% of the prior gain) — slowing, but with no sign yet of the floor-approaching
   behavior Experiment 1 is showing. It has more room left.
4. **Experiment 2's lead over Experiment 1 keeps widening**, not narrowing: CER gap 0.0162 (epoch 3) →
   0.0194 (epoch 4); WER gap 0.0361 → 0.0438. The from-scratch "recommended" architecture continues to
   pull further ahead the longer both lineages train.
5. Same worst/best collections as every prior epoch for both experiments: worst `alvsborgs_losen` (both),
   best `bergskollegium_relationer_och_skrivelser` (both, and unchanged across every evaluation in this
   project to date).

## What this does and does not establish

- Experiment 1's flattening pattern is now visible across three consecutive deltas, not just one —
  reasonably strong evidence that it is approaching diminishing-returns territory, though not yet flat.
- Experiment 2 shows no comparable flattening signal yet; its curve looks like it would still benefit
  meaningfully from a fifth epoch.
- The compute cost of this design keeps growing linearly with epoch count under the current fresh-restart
  approach (epoch 4 took ~9h17m per experiment on this hardware, roughly matching the ~4/3 scaling
  predicted from the epoch-3 runtime) — worth weighing explicitly against the shrinking marginal gain,
  especially for Experiment 1.

## Artifacts

- `training/experiment-1-four-epochs-20260810T071121Z/` (run state, manifest, provenance)
- `training/experiment-1-four-epochs-20260810T071121Z/lap-evaluation/lap4_best_val/` (this evaluation)
- `training/experiment-2-four-epochs-20260810T162908Z/` (run state, manifest, provenance)
- `training/experiment-2-four-epochs-20260810T162908Z/lap-evaluation/lap4_best_val/` (this evaluation)
