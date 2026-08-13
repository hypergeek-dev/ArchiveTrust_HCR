# Epoch-7 continuation: Experiment 2 only, via salvaged weights-only continuation

**Date:** 2026-08-13
**Methodology:** identical to every prior comparison in this project — `scripts/evaluate_full_run_lap.py`,
true Levenshtein CER/WER (never `difflib`), `best_val` checkpoint, all 1,000 held-out validation lines
scored (1000/1000 in every run below), same fixed validation set across all thirteen runs.

## Design recap and incident history

This epoch-7 result did **not** come from a single continuous `--epochs 7` invocation like every prior
epoch in this project. What actually happened, in order:

1. **First attempt** (`scripts/run_experiment_2_seven_epochs.py`, single continuous `--epochs 7`
   invocation, launched to resolve whether epoch 6 breaking the prior decay pattern was noise or real —
   see `epoch_6_continuation_comparison_report.md`): completed epochs 1-6 cleanly and checkpointed each
   one, then crashed (`epoch container exited 1`, no OOM/segfault signature, no captured stderr) almost
   immediately after entering epoch 7 — roughly 99% through the total planned work.
2. **User's call after the crash**: rather than redo all 7 epochs from scratch a second time (risking
   the same unexplained failure for zero new information), continue from the crashed run's own verified
   epoch-6 checkpoint (`training/experiment-2-seven-epochs-20260812T173551Z/.../best_val`, independently
   re-verified via `verify_checkpoint()` before use) for one salvage epoch
   (`scripts/run_experiment_2_epoch7_from_epoch6_checkpoint.py`).
3. **That salvage attempt's own first launch made zero progress**: a Windows Update scheduled OS-upgrade
   restart (`TrustedInstaller.exe`, reason code `0x80020003`) killed the host mid-launch, taking down
   Docker Desktop, the training container, and the orchestrating process with it. Relaunched clean after
   restarting Docker Desktop and confirming GPU passthrough; this second attempt completed normally.

**Why this result carries a methodological asterisk.** Every epoch-N script before this one insisted on
true single-invocation optimizer continuity — `model.fit(epochs=N)` keeping one optimizer/LR-schedule
object alive across all N epochs, by design, so the optimizer never crosses a process boundary. This
salvage epoch does not meet that bar: it fine-tunes from epoch 6's *weights only* (via
`ContainerEpochRunner`, the same mechanism Experiment 1 uses to fine-tune from Loghi's pretrained
checkpoint), with a fresh Adam optimizer and an LR schedule that restarts from the base rate rather than
continuing epoch 6's already-decayed schedule. There is no `optimizer_state/` checkpoint anywhere in this
lineage to do a true restore from (same limitation flagged at every epoch since epoch 3).

## Results — all thirteen runs, independently evaluated

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
| Experiment 2, epoch 6 | 0.1011 | 0.3360 | best_val |
| **Experiment 2, epoch 7 (salvaged, weights-only)** | **0.0998** | **0.3273** | best_val; new; see caveat above |

## Experiment 2 epoch-over-epoch deltas

| | CER Δ | CER Δ% | WER Δ | WER Δ% |
|---|---|---|---|---|
| epoch 4→5 | −0.0070 | −6.1% | −0.0150 | −4.1% |
| epoch 5→6 | −0.0064 | −5.95% | −0.0164 | −4.65% |
| epoch 6→7 (salvaged) | −0.0013 | **−1.29%** | −0.0087 | **−2.59%** |

## Findings

1. **The epoch 6→7 gain is much smaller than epoch 6's own gain**, and smaller than even the original
   pre-epoch-6 geometric-decay extrapolation would have predicted. Two readings remain live, and this
   data point can't cleanly separate them: (a) epoch 6 really was the anomaly and the underlying decay
   has reasserted itself, now landing in Experiment 1-like marginal-gain territory; or (b) the muted gain
   is a methodological artifact of the fresh-optimizer/fresh-LR-schedule restart, not a true reading of
   what a real continuous epoch 7 would have shown.
2. **CER dropped below 0.10 for the first time** in this project, for either lineage.
3. **The gap to Experiment 1's final (5-epoch) result is now the widest recorded**: CER 0.0312 (0.1310
   vs. 0.0998), WER 0.0754 (0.4027 vs. 0.3273).
4. Same worst/best collections as every prior evaluation: worst `jonkopings_radhusratt_och_magistrat`,
   best `bergskollegium_relationer_och_skrivelser`.
5. **Operational: two independent incidents in immediate succession** — an unexplained container crash
   (exit code 1, ~99% through a long run) and a Windows Update forced reboot that silently killed an
   in-progress training process and Docker Desktop entirely. Both are now part of this project's known
   risk list alongside the earlier OneDrive-memory-pressure incident. The Windows Update risk in
   particular applies to *any* future long run regardless of how it's launched (detached processes still
   die in a full OS reboot) and has not yet been mitigated.

## What this does and does not establish

- Given the methodological asterisk on this specific data point, it is weaker evidence than every prior
  epoch about where Experiment 2's true convergence point is. A genuine single-continuous-invocation
  epoch 8 (or a clean re-run of epoch 7 done the same way) would be needed to resolve the (a)/(b)
  ambiguity above with the same rigor as epochs 1-6.
- Independent of that ambiguity, the practical result stands: Experiment 2 now has a real, verified,
  independently-evaluated checkpoint at CER 0.0998 / WER 0.3273, the best of either lineage by a wide
  margin.

## Artifacts

- `training/experiment-2-seven-epochs-20260812T173551Z/` (the crashed single-invocation attempt —
  preserved as-is; epochs 1-6 checkpoints and evaluations remain valid and were the source for this
  salvage run's parent checkpoint)
- `training/experiment-2-epoch7-from-epoch6-checkpoint-20260813T124955Z/` (the successful salvage run —
  run state, manifest, provenance)
- `training/experiment-2-epoch7-from-epoch6-checkpoint-20260813T124955Z/lap-evaluation/lap1_best_val/`
  (this evaluation — named `lap1` by the evaluation tooling since this run's own accounting shows 1
  completed epoch, not 7; the human-readable "epoch 7" label in this report reflects total training
  history across both runs)
