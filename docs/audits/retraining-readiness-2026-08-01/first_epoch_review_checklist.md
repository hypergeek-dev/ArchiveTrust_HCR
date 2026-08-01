# First Full Epoch (First Lap) Review Checklist

"Epoch" here means one full pass over the corpus — 57 shards (`lap: 0` in `sharding_summary.json`),
**not** the shard-level "epoch" the orchestrator itself uses internally. Complete this at or before
shard 57. This is Gate 5 — the last checkpoint before trusting the run to continue unattended for
many more hours/days.

## Actual full-epoch (full-lap) duration

- [ ] Sum of `duration_seconds` across shards 1-57 (from `session_state.json: validation_history`):
  ______________ hours (extrapolated planning estimate from the pilot: ~7.6 hours for one lap, see
  `retraining_readiness_audit.md` §5)
- [ ] Material deviation from the extrapolation (>25%) should be understood and, if needed, used to
  recompute later-lap time estimates rather than trusting the original extrapolation blindly.

## Full validation result

- [ ] `val_cer` trend across shards 1-57 is generally decreasing (allow for real noise — compare
  against the pilot's own observed noise floor, `pilot_analysis.json: meaningful_improvement_threshold`
  and `validation_noise_stdev`).
- [ ] `best_val_cer` (from `run_state.json: best_metrics`) after 57 shards: ______________

## Baseline comparison

- [ ] Compare `best_val_cer` after one full lap against the pilot's own `best_pilot_val_cer`
  (`0.16913` at pilot epoch 22). A full-corpus lap should be expected to reach comparable or better
  CER given ~56x more distinct training data, though this is a real, first empirical data point, not
  an assumption to make in advance.

## CER delta and confidence

- [ ] Report the delta and, if feasible, an approximate confidence interval given the validation
  set's fixed size (1000 lines) — do not treat small deltas (within the pilot's own observed noise
  floor) as meaningful without checking against `meaningful_improvement_threshold`.

## Subgroup checks

- [ ] If per-collection/subgroup CER breakdown is available, specifically review the 7 collections
  flagged in R-009 (line-level split granularity) for anomalously good validation performance that
  might indicate the disclosed leakage risk is materializing, not just theoretical.

## Overfitting signals

- [ ] Compare `train_loss` vs `val_loss` trend across the lap (the same divergence check
  `pilot_analysis.py::_train_val_divergence` uses on the pilot — widening gap = investigate).
- [ ] `epochs_since_improvement` (shards since the last real best) — is it approaching
  `recommended_patience` (5)? If so, the run may stop soon; this is expected behavior, not a fault.

## Stopping-policy recalibration

- [ ] Given real, full-corpus-scale measurements now exist (throughput, noise floor, CER trajectory),
  decide explicitly: keep the pilot-derived `min_exposure_steps`/`recommended_patience` as-is, or
  override them (`analyze-pilot --patience`/`--corpus-line-count` and re-`prepare`, or an equivalent
  configuration update) before continuing further.
- [ ] Record the decision and its rationale.

## Disk and infrastructure re-check

- [ ] Free disk remaining after 57 shards ≈ (95.4GB minus ~41.5GB) if no retention policy is active —
  confirm actual matches expectation, and confirm the R-018 remediation (retention policy or
  provisioned disk) is holding as designed before continuing into a second lap.
- [ ] GPU health (temperature, VRAM headroom) remained stable across the full lap, not just the first
  shard.

## Continue / stop decision

- [ ] Explicit decision recorded: CONTINUE / STOP, by whom, and why.
- [ ] If CONTINUE: next scheduled review point recorded (e.g. shard 100, or end of lap 2).
