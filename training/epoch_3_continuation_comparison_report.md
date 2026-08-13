# Epoch-3 continuation: third-level comparison (Experiment 1 vs Experiment 2)

**Date:** 2026-08-09
**Methodology:** identical to every prior comparison in this project — `scripts/evaluate_full_run_lap.py`,
true Levenshtein CER/WER (never `difflib`), `best_val` checkpoint, all 1,000 held-out validation lines
scored (1000/1000 in every run below), same fixed validation set across all seven runs.

## Design recap

Both epoch-3 runs are **single, uninterrupted `--epochs 3` container invocations** — a fresh restart
from each experiment's original origin (pristine pinned checkpoint for Experiment 1, random
initialization for Experiment 2), not a resume of the preserved one- or two-epoch runs. `model.fit(
epochs=3)` keeps one optimizer/LR-schedule object alive across all three internal Keras epochs
in-process, so optimizer continuity holds by construction — see `scripts/run_experiment_1_three_epochs.py`
/ `run_experiment_2_three_epochs.py` for the full rationale (identical reasoning to the epoch-2 drivers:
neither preserved prior run wrote an `optimizer_state/` checkpoint, so there is nothing for a
cross-process resume to restore from; a fresh restart with more internal epochs was the deliberate
alternative, consistent across both lineages for a clean comparison).

**Operational incident, noted for the record:** the first epoch-3 attempt for both experiments
(launched 2026-08-08 14:11 UTC) did not produce this data. Experiment 1's container crashed (exit 139)
92.5% of the way through its three epochs, and the resulting Experiment 2 run stalled at ~1,500 steps.
Root cause: this checkout lives under `OneDrive\Desktop\ArchiveTrust_HCR`, and OneDrive's sync client was
consuming ~4.9 GB of host RAM indexing the constantly-growing `training/` tree, leaving the host with as
little as 1.1 GB free; Docker Desktop's engine degraded as a consequence (API calls returning HTTP 500).
Fix: OneDrive sync stopped, Docker Desktop restarted, both experiments relaunched clean from their
original origins (new run directories, below) — this is the data that follows. Both runs referenced in
this report completed with `stop_reason=target_epochs_reached`, exit code 0, both checkpoints verified.

## Results — all seven runs, independently evaluated

| Run | Corpus CER | Corpus WER | Notes |
|---|---|---|---|
| Experiment 0 (57-shard, optimizer reset per shard) | 0.1728 | 0.4945 | best_val; the flawed-method baseline |
| Experiment 1, epoch 1 (fine-tune, single continuous epoch) | 0.1747 | 0.4956 | best_val |
| Experiment 1, epoch 2 (fine-tune, two continuous epochs) | 0.1522 | 0.4458 | best_val |
| **Experiment 1, epoch 3 (fine-tune, three continuous epochs)** | **0.1419** | **0.4276** | best_val; new |
| Experiment 2, epoch 1 (scratch, single continuous epoch) | 0.1918 | 0.5470 | best_val |
| Experiment 2, epoch 2 (scratch, two continuous epochs) | 0.1487 | 0.4546 | best_val |
| **Experiment 2, epoch 3 (scratch, three continuous epochs)** | **0.1257** | **0.3915** | best_val; new |

## Epoch-over-epoch deltas

| | CER Δ | CER Δ% | WER Δ | WER Δ% |
|---|---|---|---|---|
| Experiment 1, epoch 1→2 | −0.0225 | −12.9% | −0.0498 | −10.0% |
| Experiment 1, epoch 2→3 | −0.0103 | −6.8% | −0.0182 | −4.1% |
| Experiment 2, epoch 1→2 | −0.0431 | −22.5% | −0.0924 | −16.9% |
| Experiment 2, epoch 2→3 | −0.0230 | −15.5% | −0.0631 | −13.9% |

## Findings

1. **Both experiments keep improving at epoch 3** — neither result regressed, and in-training `val_loss`
   is still declining every epoch for both (Experiment 1: 21.9→20.1→18.3; Experiment 2: 23.1→17.3→15.5),
   so there is no sign of overfitting yet, just smaller gains.
2. **The rate of improvement is diminishing for both, but much more sharply for Experiment 1.** Its
   relative CER gain roughly halved epoch-over-epoch (−12.9% → −6.8%), and its relative WER gain fell by
   more than half (−10.0% → −4.1%). Experiment 2's gains are also slowing but far more gently (CER:
   −22.5% → −15.5%; WER: −16.9% → −13.9%).
3. **Experiment 2 (scratch) has pulled decisively ahead of Experiment 1 (fine-tune) on both metrics.**
   At epoch 2 the gap was narrow and CER-only (0.1487 vs 0.1522); at epoch 3 it is clear on both axes
   (0.1257 vs 0.1419 CER — an 11.4% relative gap; 0.3915 vs 0.4276 WER — an 8.4% relative gap). The
   from-scratch "recommended" architecture is no longer just catching up, it is winning outright on this
   corpus.
4. **Both epoch-3 results comfortably beat Experiment 0's 57-shard baseline**, extending the pattern
   already established at epoch 2 — the single-continuous-epoch-lineage design keeps compounding its
   advantage over the optimizer-reset-per-shard method as more epochs are added.
5. Worst/best collections at epoch 3: Experiment 1 — worst `alvsborgs_losen`, best
   `bergskollegium_relationer_och_skrivelser`. Experiment 2 — worst
   `jonkopings_radhusratt_och_magistrat`, best `bergskollegium_relationer_och_skrivelser` (same best
   collection as Experiment 1, and the same collection that has been best across every prior evaluation).

## What this does and does not establish

- Still not a convergence study, but the shape of it is becoming visible: Experiment 1's diminishing
  returns suggest it is approaching the point where further epochs stop being worth their compute cost,
  while Experiment 2's slower falloff suggests it has more room left. A fourth epoch is the natural next
  check to confirm or revise that read.
- The reversal first seen at epoch 2 (scratch overtaking fine-tune on CER) is now robust across both
  metrics and has widened, not narrowed — this is no longer a one-datapoint fluke.

## Artifacts

- `training/experiment-1-three-epochs-20260809T020441Z/` (run state, manifest, provenance — heavy
  checkpoint/telemetry data gitignored as with every prior run; supersedes the crashed
  `experiment-1-three-epochs-20260808T141115Z` attempt, preserved untouched as evidence)
- `training/experiment-1-three-epochs-20260809T020441Z/lap-evaluation/lap3_best_val/` (this evaluation)
- `training/experiment-2-three-epochs-20260809T100250Z/` (run state, manifest, provenance; supersedes
  the stalled `experiment-2-three-epochs-20260808T214722Z` attempt, preserved untouched as evidence)
- `training/experiment-2-three-epochs-20260809T100250Z/lap-evaluation/lap3_best_val/` (this evaluation)
