# Epoch-5 continuation: fifth-level comparison (Experiment 1 vs Experiment 2)

**Date:** 2026-08-12
**Methodology:** identical to every prior comparison in this project — `scripts/evaluate_full_run_lap.py`,
true Levenshtein CER/WER (never `difflib`), `best_val` checkpoint, all 1,000 held-out validation lines
scored (1000/1000 in every run below), same fixed validation set across all eleven runs.

## Design recap

Both epoch-5 runs are **single, uninterrupted `--epochs 5` container invocations** — a fresh restart
from each experiment's original origin (pristine pinned checkpoint for Experiment 1, random
initialization for Experiment 2), not a resume of any preserved lower-epoch run. `model.fit(epochs=5)`
keeps one optimizer/LR-schedule object alive across all five internal Keras epochs in-process, so
optimizer continuity holds by construction — see `scripts/run_experiment_1_five_epochs.py` /
`run_experiment_2_five_epochs.py` for the full rationale (unchanged from the epoch-2/3/4 drivers: none
of the preserved prior runs wrote an `optimizer_state/` checkpoint, so there is nothing for a
cross-process resume to restore from). Both runs completed cleanly (exit code 0, `target_epochs_reached`),
both checkpoints verified, launched back to back by the same detached orchestrator
(`training/_orchestration_logs/run_five_epoch_continuations.sh`). No operational incidents this round —
OneDrive was stopped before launch and stayed off for the full ~23h duration of both runs combined.

## Results — all eleven runs, independently evaluated

| Run | Corpus CER | Corpus WER | Notes |
|---|---|---|---|
| Experiment 0 (57-shard, optimizer reset per shard) | 0.1728 | 0.4945 | best_val; the flawed-method baseline |
| Experiment 1, epoch 1 | 0.1747 | 0.4956 | best_val |
| Experiment 1, epoch 2 | 0.1522 | 0.4458 | best_val |
| Experiment 1, epoch 3 | 0.1419 | 0.4276 | best_val |
| Experiment 1, epoch 4 | 0.1339 | 0.4112 | best_val |
| **Experiment 1, epoch 5** | **0.1310** | **0.4027** | best_val; new |
| Experiment 2, epoch 1 | 0.1918 | 0.5470 | best_val |
| Experiment 2, epoch 2 | 0.1487 | 0.4546 | best_val |
| Experiment 2, epoch 3 | 0.1257 | 0.3915 | best_val |
| Experiment 2, epoch 4 | 0.1145 | 0.3674 | best_val |
| **Experiment 2, epoch 5** | **0.1075** | **0.3524** | best_val; new |

## Epoch-over-epoch deltas

| | CER Δ | CER Δ% | WER Δ | WER Δ% |
|---|---|---|---|---|
| Experiment 1, epoch 1→2 | −0.0225 | −12.9% | −0.0498 | −10.0% |
| Experiment 1, epoch 2→3 | −0.0103 | −6.8% | −0.0182 | −4.1% |
| Experiment 1, epoch 3→4 | −0.0080 | −5.6% | −0.0164 | −3.8% |
| Experiment 1, epoch 4→5 | −0.0029 | **−2.2%** | −0.0085 | **−2.1%** |
| Experiment 2, epoch 1→2 | −0.0431 | −22.5% | −0.0924 | −16.9% |
| Experiment 2, epoch 2→3 | −0.0230 | −15.5% | −0.0631 | −13.9% |
| Experiment 2, epoch 3→4 | −0.0112 | −8.9% | −0.0241 | −6.2% |
| Experiment 2, epoch 4→5 | −0.0070 | **−6.1%** | −0.0150 | **−4.1%** |

## Findings

1. **Both experiments still improved at epoch 5, no regression, no overfitting signal** — in-training
   `val_loss` kept declining for both (Experiment 1: 22.7→19.9→18.7→17.4→16.7; Experiment 2:
   23.5→17.7→15.4→13.5→13.5 [sic, essentially flat 4→5]).
2. **Experiment 1 has reached genuinely marginal-gain territory.** Its epoch 4→5 CER improvement
   (−2.2%) is its smallest yet by a wide margin, continuing a clear downward trend across all four
   measured deltas (−12.9%, −6.8%, −5.6%, −2.2%). This is no longer "diminishing but still substantial"
   — it is now small enough that a further epoch's expected gain (extrapolating the trend) would likely
   fall to roughly 1% relative CER, at the same ~11-13h compute cost as every prior epoch for this
   experiment.
3. **Experiment 2's decay is far more regular and has not reached the same territory.** Its relative
   CER deltas (−22.5%, −15.5%, −8.9%, −6.1%) shrink by a fairly consistent ~0.6-0.7× each epoch — a
   clean geometric pattern, unlike Experiment 1's sharper recent drop-off. Extrapolating that ratio,
   epoch 6 would likely still yield ~4% relative CER improvement — smaller than before, but not yet in
   Experiment 1's near-flat territory.
4. **Experiment 2's lead over Experiment 1 continues to widen**: CER gap 0.0194 (epoch 4) → 0.0235
   (epoch 5); WER gap 0.0438 → 0.0503.
5. Same worst/best collections as every prior epoch for both experiments: worst
   `jonkopings_radhusratt_och_magistrat` (both, at epoch 5), best
   `bergskollegium_relationer_och_skrivelser` (both, unchanged across every evaluation in this project).

## What this does and does not establish

- Experiment 1's curve now looks like it is approaching convergence for this architecture/data/optimizer
  configuration — a further epoch is a much weaker bet than it was at epoch 3 or 4.
- Experiment 2 shows no comparable convergence signal yet; a sixth epoch remains a reasonable bet on the
  numbers alone.
- The compute cost keeps growing with epoch count under the fresh-restart design (~11-13h/experiment for
  epoch 5) — for Experiment 1 specifically, that cost-to-expected-gain ratio has shifted sharply enough
  that continuing without a clear reason (beyond cross-experiment methodological symmetry) is a
  genuinely weaker call than it was at any prior epoch.

## Artifacts

- `training/experiment-1-five-epochs-20260811T034400Z/` (run state, manifest, provenance)
- `training/experiment-1-five-epochs-20260811T034400Z/lap-evaluation/lap5_best_val/` (this evaluation)
- `training/experiment-2-five-epochs-20260811T152858Z/` (run state, manifest, provenance)
- `training/experiment-2-five-epochs-20260811T152858Z/lap-evaluation/lap5_best_val/` (this evaluation)
