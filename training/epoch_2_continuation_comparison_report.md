# Epoch-2 continuation: second-level comparison (Experiment 1 vs Experiment 2)

**Date:** 2026-08-06
**Methodology:** identical to every prior comparison in this project — `scripts/evaluate_full_run_lap.py`,
true Levenshtein CER/WER (never `difflib`), `best_val` checkpoint, all 1,000 held-out validation lines
scored (1000/1000 in every run below), same fixed validation set across all five runs.

## Design recap

Both epoch-2 runs are **single, uninterrupted `--epochs 2` container invocations** — a fresh restart
from each experiment's original origin (pristine pinned checkpoint for Experiment 1, random
initialization for Experiment 2), not a resume of the preserved single-epoch runs. `model.fit(epochs=2)`
keeps one optimizer/LR-schedule object alive across both internal Keras epochs in-process, so optimizer
continuity holds by construction — see `scripts/run_experiment_1_two_epochs.py` /
`run_experiment_2_two_epochs.py` for the full rationale. Both runs completed cleanly (exit code 0,
`target_epochs_reached`), both checkpoints verified.

## Results — all five runs, independently evaluated

| Run | Corpus CER | Corpus WER | Notes |
|---|---|---|---|
| Experiment 0 (57-shard, optimizer reset per shard) | 0.1728 | 0.4945 | best_val; the flawed-method baseline |
| Experiment 1, epoch 1 (fine-tune, single continuous epoch) | 0.1747 | 0.4956 | best_val |
| **Experiment 1, epoch 2 (fine-tune, two continuous epochs)** | **0.1522** | **0.4458** | best_val; new |
| Experiment 2, epoch 1 (scratch, single continuous epoch) | 0.1918 | 0.5470 | best_val |
| **Experiment 2, epoch 2 (scratch, two continuous epochs)** | **0.1487** | **0.4546** | best_val; new |

## Epoch 1 → epoch 2 deltas

| | CER Δ | CER Δ% | WER Δ | WER Δ% |
|---|---|---|---|---|
| Experiment 1 (fine-tune) | −0.0225 | −12.9% | −0.0498 | −10.0% |
| Experiment 2 (scratch) | −0.0431 | −22.5% | −0.0924 | −16.9% |

## Findings

1. **Both experiments improve substantially with a second epoch**, on both metrics, confirming the
   extra pass over the corpus is genuinely useful rather than noise — neither result regressed.
2. **The scratch-trained model (Experiment 2) improved more, in relative terms, than the fine-tune
   (Experiment 1)** — consistent with a from-scratch model having more "room to learn" in a second
   pass than a model that started from an already-competent pretrained checkpoint.
3. **By epoch 2, Experiment 2 (scratch) has the lower CER of the two** (0.1487 vs 0.1522) — a reversal
   from epoch 1, where fine-tuning was ahead on both metrics (0.1747/0.4956 vs 0.1918/0.5470).
   Experiment 1 still holds the lower WER (0.4458 vs 0.4546).
4. **Neither epoch-2 result comes close to Experiment 0's 57-shard baseline being competitive on its own
   terms** — both single-continuous-epoch-lineage runs (Experiment 1 and Experiment 2) now clearly beat
   Experiment 0 on both metrics after two epochs, having already roughly matched or trailed it after one.
5. Worst/best collections at epoch 2: Experiment 1 — worst `jonkopings_radhusratt_och_magistrat`, best
   `bergskollegium_relationer_och_skrivelser`. Experiment 2 — worst `alvsborgs_losen`, best
   `bergskollegium_relationer_och_skrivelser` (same best collection as Experiment 1).

## What this does and does not establish

- This is two epochs, not a convergence study — neither run was pushed to a third epoch or to
  early-stopping on a plateau, so "which architecture wins at convergence" remains open.
- Experiment 2 catching up to (and slightly passing, on CER) Experiment 1 after starting substantially
  behind is suggestive that the "recommended" from-scratch architecture is a viable competitor to
  fine-tuning generic-2023-02-15 on this corpus, not merely a slower path to a worse result — but two
  data points per lineage is not enough to call this decisively; a third epoch would be the natural next
  check if that question matters for the project's direction.

## Artifacts

- `training/experiment-1-two-epochs-20260805T181633Z/` (run state, manifest, provenance — heavy
  checkpoint/telemetry data gitignored as with every prior run)
- `training/experiment-1-two-epochs-20260805T181633Z/lap-evaluation/lap2_best_val/` (this evaluation)
- `training/experiment-2-two-epochs-20260806T015942Z/` (run state, manifest, provenance)
- `training/experiment-2-two-epochs-20260806T015942Z/lap-evaluation/lap2_best_val/` (this evaluation)
