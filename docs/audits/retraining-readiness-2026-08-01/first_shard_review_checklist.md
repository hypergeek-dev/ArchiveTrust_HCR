# First-Shard Review Checklist

Run this after the **first** shard of the full-corpus run completes (i.e. as soon as
`status --run <dir>` first reports `current_epoch: 1`). Do not let the run continue unattended past
this point without completing this review — this is Gate 4.

## Expected vs. actual runtime

- [ ] Expected runtime for shard 1: ~482 seconds (~8.0 minutes), extrapolated 1:1 from the pilot's
  own measured mean epoch duration (`pilot_analysis.json: runtime_stability.epoch_duration_mean_seconds`),
  since `shard_line_count` was set equal to the pilot's own train line count.
- [ ] Actual runtime for shard 1: ______________ (read from `run_state.json: latest_metrics` timing
  or `session_state.json: validation_history[-1].duration_seconds`)
- [ ] If actual is >2x expected: stop and investigate before continuing (data-loader bottleneck,
  disk contention, or the extrapolation was materially wrong at real corpus I/O patterns).

## Throughput

- [ ] Lines/second for shard 1 = 9999 / actual_duration_seconds = ______________
- [ ] Compare to pilot's own measured throughput (`pilot_analysis.json: throughput_lines_per_second`).
  Material deviation (>20%) should be understood before trusting later duration estimates.

## GPU health

- [ ] `training/full-corpus-.../run-state/telemetry/gpu_samples.jsonl` contains real samples (not
  empty, not stale) for the shard-1 window.
- [ ] Peak VRAM usage stayed comfortably under the GPU's total (check `memory_used_mb` vs
  `memory_total_mb` in the samples) — no OOM signature in `stderr_tail`.
- [ ] GPU temperature stayed within a safe operating range across the sample window.

## Loss/CER sanity

- [ ] `val_cer` for shard 1 is a real, non-null number (`status` output's `latest_metrics.val_cer`).
- [ ] `val_cer` is not `NaN`/`Inf` (the orchestrator would have stopped with `nan_or_inf_detected` if
  so — confirm `run_state.json: status` is still `running`, not `failed`).
- [ ] `train_loss`/`val_loss` are present and plausible (not wildly larger than the pilot's own
  observed loss range, per `pilot_analysis.json`'s epoch observations).

## Skipped records

- [ ] Check the container's own `stdout_tail`/`stderr_tail` (captured in the epoch result, or
  directly in `run-state/epoch_output/epoch_1/log.csv` and stdout capture) for any "skipped sample"
  warnings from the pinned `loghi-htr` image. There is no ArchiveTrust-level skip-counting for this
  path — this must be read from the container's own log output.

## Checkpoint validation

- [ ] `checkpoint_index.json` under the run's `run-state/` now exists and has at least one `"latest"`
  entry for `epoch: 1`.
- [ ] That entry's `verification_status` is `"verified"` and `resumable` is `true`.
- [ ] Manually re-run `verify_checkpoint()` against the recorded `checkpoint_dir` (or use the CLI/
  Python one-liner) to independently confirm it still opens cleanly.

## Telemetry continuity

- [ ] `telemetry/status.json`'s `last_sample_at` is recent (not stale) as of this review.
- [ ] `run_state.json`'s `last_heartbeat_at` is recent and consistent with `telemetry/status.json`.

## Disk growth

- [ ] Real disk consumption after shard 1 ≈ 0.727GB (the pilot's own measured per-epoch cost). Record
  the actual value: ______________
- [ ] Free disk remaining: ______________. Confirm this is consistent with the R-018 remediation
  plan (retention policy active, or sufficient provisioned disk) — **do not let the run continue past
  this point if R-018 has not been resolved**, per the audit's blocking condition.

## Stop criteria for this review

- [ ] If any of the above is red/failing: stop the run gracefully (`stop --run <dir>`), do not let it
  continue to shard 2 unattended, and investigate before resuming.
- [ ] If all green: proceed, but schedule the first-full-epoch (first-lap) review no later than
  shard 57.
