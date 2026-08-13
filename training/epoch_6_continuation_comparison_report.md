# Epoch-6 continuation: Experiment 2 only (Experiment 1 finalized at epoch 5)

**Date:** 2026-08-12
**Methodology:** identical to every prior comparison in this project — `scripts/evaluate_full_run_lap.py`,
true Levenshtein CER/WER (never `difflib`), `best_val` checkpoint, all 1,000 held-out validation lines
scored (1000/1000 in every run below), same fixed validation set across all twelve runs.

## Design recap

This is the first epoch-N continuation in this lineage without a matching Experiment 1 run.
`training/epoch_5_continuation_comparison_report.md` found Experiment 1's epoch 4→5 relative CER gain
(−2.2%) sharply smaller than its own prior deltas (−12.9%, −6.8%, −5.6%) — read as approaching
convergence. Experiment 2's decay had instead stayed a fairly consistent ~0.6-0.7× ratio each epoch, so
the user's call was to continue Experiment 2 alone (`scripts/run_experiment_2_six_epochs.py`) and treat
Experiment 1's 5-epoch result as final. Same single-continuous-invocation design as every prior epoch:
`model.fit(epochs=6)` keeps one optimizer/LR-schedule object alive across all six internal Keras epochs
in-process, fresh restart from random initialization (no `optimizer_state/` checkpoint exists to resume
from in any preserved prior run). Completed cleanly (exit code 0, `target_epochs_reached`, ~13h), launched
by a new single-experiment orchestrator (`training/_orchestration_logs/run_six_epoch_experiment_2_only.sh`)
rather than the two-experiment sequential script used for every prior epoch. No operational incidents.

## Results — all twelve runs, independently evaluated

| Run | Corpus CER | Corpus WER | Notes |
|---|---|---|---|
| Experiment 0 (57-shard, optimizer reset per shard) | 0.1728 | 0.4945 | best_val; the flawed-method baseline |
| Experiment 1, epoch 1 | 0.1747 | 0.4956 | best_val |
| Experiment 1, epoch 2 | 0.1522 | 0.4458 | best_val |
| Experiment 1, epoch 3 | 0.1419 | 0.4276 | best_val |
| Experiment 1, epoch 4 | 0.1339 | 0.4112 | best_val |
| Experiment 1, epoch 5 (final) | 0.1310 | 0.4027 | best_val; no epoch 6 |
| Experiment 2, epoch 1 | 0.1918 | 0.5470 | best_val |
| Experiment 2, epoch 2 | 0.1487 | 0.4546 | best_val |
| Experiment 2, epoch 3 | 0.1257 | 0.3915 | best_val |
| Experiment 2, epoch 4 | 0.1145 | 0.3674 | best_val |
| Experiment 2, epoch 5 | 0.1075 | 0.3524 | best_val |
| **Experiment 2, epoch 6** | **0.1011** | **0.3360** | best_val; new |

## Experiment 2 epoch-over-epoch deltas

| | CER Δ | CER Δ% | WER Δ | WER Δ% |
|---|---|---|---|---|
| epoch 1→2 | −0.0431 | −22.5% | −0.0924 | −16.9% |
| epoch 2→3 | −0.0230 | −15.5% | −0.0631 | −13.9% |
| epoch 3→4 | −0.0112 | −8.9% | −0.0241 | −6.2% |
| epoch 4→5 | −0.0070 | −6.1% | −0.0150 | −4.1% |
| epoch 5→6 | −0.0064 | **−5.95%** | −0.0164 | **−4.65%** |

## Findings

1. **Experiment 2's epoch 6 gain broke its own decay pattern.** Every prior delta had shrunk by
   roughly 30-40% relative to the one before it (a fairly clean ~0.6-0.7× geometric ratio). Epoch 5→6
   did not follow that: the CER relative gain barely slowed at all (6.1% → 5.95%, essentially flat),
   and the WER relative gain actually **increased** (4.1% → 4.65%) rather than continuing to shrink.
   Extrapolating the established decay ratio predicted roughly a 4% CER gain at epoch 6; the actual
   result (5.95%) came in noticeably stronger than that prediction.
2. **This is one data point, not proof of a trend reversal** — could be ordinary run-to-run noise
   rather than a genuine change in the improvement trajectory. But it does mean the simple
   extrapolation logic used to justify epoch 5 and epoch 6 is now less reliable, and there is no
   evidence yet that Experiment 2 is approaching Experiment 1's kind of floor.
3. **Still no overfitting signal** — in-training `val_loss` kept declining (23.5→17.7→15.4→13.5→12.8→
   12.8 [essentially flat 5→6, matching the WER/CER pattern of a genuine but no-longer-shrinking gain]).
4. **The gap to Experiment 1's final result keeps widening**: CER gap now 0.0299 (Experiment 1's 0.1310
   vs. Experiment 2's 0.1011); WER gap 0.0667 (0.4027 vs. 0.3360) — both the largest gaps recorded in
   this project.
5. Same worst/best collections as every prior evaluation: worst `jonkopings_radhusratt_och_magistrat`,
   best `bergskollegium_relationer_och_skrivelser`.

## What this does and does not establish

- The break from the decay pattern means a seventh epoch is a genuinely open question rather than a
  confident extrapolation either way — the prior geometric-decay logic that made epoch 5 and 6 clear
  calls no longer applies cleanly.
- If the epoch 5→6 gain (~6% relative CER) roughly repeats rather than continuing to shrink, a seventh
  epoch would still be a reasonable bet; if it was noise and the decay reasserts, a seventh epoch would
  land closer to Experiment 1's diminished-returns territory. One more epoch is the only way to tell
  which read is correct.
- Compute cost keeps growing regardless (~13h for epoch 6 alone) — worth weighing against that
  uncertainty explicitly rather than assuming continuation is free.

## Artifacts

- `training/experiment-2-six-epochs-20260812T040342Z/` (run state, manifest, provenance)
- `training/experiment-2-six-epochs-20260812T040342Z/lap-evaluation/lap6_best_val/` (this evaluation)
