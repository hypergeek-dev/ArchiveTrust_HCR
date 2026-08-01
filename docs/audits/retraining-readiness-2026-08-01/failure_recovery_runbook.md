# Failure Recovery Runbook — Loghi Full-Corpus Training

For each scenario: detection, immediate behavior, data-loss risk, checkpoint integrity, recovery
steps, and whether manual intervention is required. Verified against real code where marked
**(verified)**; marked **(unverified)** where this audit could not confirm behavior from source
reading alone and recommends testing before relying on it operationally.

---

### 1. Docker daemon stops mid-run **(verified)**
- **Detection:** the current shard's `subprocess.run(["docker", "run", ...])` returns non-zero or
  raises `subprocess.SubprocessError`.
- **Behavior:** `EpochResult(ok=False, ...)` → `run_training_session` sets `stop_reason="epoch_failed"`
  → `orchestrator.py` sets `stop_reason="epoch_failed"` → `run_state.py: mark_failed`.
- **Data loss:** none — the last verified checkpoint (previous shard) stands.
- **Checkpoint integrity:** unaffected.
- **Recovery:** restart Docker Desktop, confirm `docker ps` succeeds, then `resume --run <dir>
  --confirm-full-corpus-run`.
- **Manual intervention:** yes (restart Docker).

### 2. Container exits unexpectedly (non-zero exit code) **(verified)**
- Same path as #1 — `completed.returncode != 0` → `EpochResult(ok=False, exit_code=..., ...)`.
- **Recovery:** inspect `stderr_tail` (captured in the epoch result / logs) for the real cause before
  resuming blindly.

### 3. GPU OOM **(verified, detection only)**
- **Detection:** `orchestrator.py::_oom_signature_in` scans `stderr_tail` for
  `OutOfMemoryError`/`CUDA_ERROR_OUT_OF_MEMORY`/`ResourceExhaustedError`.
- **Behavior:** `stop_reason="likely_oom"` → `mark_failed`.
- **Recovery:** reduce `--batch-size`, then `resume`. No automatic batch-size backoff exists.
- **Manual intervention:** yes.

### 4. NaN/Inf loss **(verified)**
- **Detection:** `orchestrator.py::_has_nan_or_inf` on `train_loss`/`val_loss`/`val_cer` after every
  completed shard.
- **Behavior:** `stop_reason="nan_or_inf_detected"` → `mark_failed`.
- **Data loss:** none — checked only after a shard's own checkpoint was already written and verified.
- **Recovery:** investigate before resuming (a NaN loss on resume with the same configuration will
  likely recur); consider a lower learning rate.

### 5. Malformed batch / unreadable / missing / corrupt image **(unverified at the ArchiveTrust layer)**
- This is internal to the pinned `loghi-htr` container's own data loader. ArchiveTrust does not
  intercept or count skipped samples. If the container's data loader skips silently, this workflow
  will not surface it — check `log.csv` / captured stdout for the pinned image's own warnings.
- **Recommendation:** do not assume zero skipped samples; explicitly grep the first several shards'
  captured stdout for the pinned image's own skip/warning messages during Gate 4/5.

### 6. Corrupt manifest / dataset file changes after preparation **(verified, detected at launch only)**
- **Detection:** the launch guard's `current_dataset_hash`/`current_training_manifest_hash` checks,
  evaluated once at `start`/`resume` time.
- **Gap:** not re-checked mid-run — if the source inventory or a shard file changes *while training
  is running*, nothing detects it until the next `start`/`resume` invocation. See R-007.
- **Recovery:** if suspected, stop the run, re-verify hashes manually, re-`prepare` if needed.

### 7. Validation failure (missing `val_cer`) **(verified)**
- **Detection:** `orchestrator.py`: `if last_result.val_cer is None: stop_reason =
  "missing_validation_result"`.
- **Behavior:** clean stop, not marked `failed` (falls through to `mark_stopped`).
- **Recovery:** investigate the container's validation step before resuming.

### 8. Checkpoint write failure / partial checkpoint **(partially verified)**
- **Verified:** `checkpoint_index.py`'s own index file uses atomic write-temp-then-rename, so the
  *index* itself cannot be torn.
- **Unverified:** the underlying `.keras` file write is performed by the pinned container, not by
  ArchiveTrust's own atomic-write helpers. `verify_checkpoint()` would catch a resulting bad zip on
  the *next* read (structural check only, see R-008), but a checkpoint that fails to write cleanly
  mid-shard due to disk pressure has not been explicitly tested end-to-end.
- **Recovery:** if `verify_checkpoint` / `resumable` is `false` for the latest entry, use the last
  `resumable=true` entry (`latest_resumable_checkpoint()`) — this is exactly what `resume` already
  does via `TrainingSessionState.latest_checkpoint_dir`.

### 9. Disk nearly full / disk completely full **(unverified — the primary audit finding, R-018)**
- **Detection:** none, currently, mid-run. Preflight checks disk space once, at launch.
- **Behavior:** **unknown.** `tempfile.mkstemp()` (used by every atomic-write helper in this
  codebase) will raise `OSError` on `ENOSPC`; nothing in `checkpoint_index.py`, `run_state.py`, or
  `training_session.py` specifically catches this case, so it likely propagates as an uncaught
  exception out of `run_training_session`, crashing the CLI process rather than triggering a clean
  `stop_reason`.
- **Recovery (until R-018 is fixed):** monitor free disk manually throughout the run; if it
  approaches exhaustion, issue a graceful `stop` (sentinel-file based, takes effect at the next shard
  boundary) well before the disk actually fills, then free space (delete old, already-superseded
  `"latest"`/`"end_of_session"` checkpoint directories that are NOT the current best/latest, per
  `checkpoint_index.json`) before resuming.
- **Manual intervention:** yes, and this scenario is the audit's #1 recommended fix before launch.

### 10. Host reboot / power loss **(unverified end-to-end, plausible from design)**
- **Expected behavior (by design, not yet tested as a real reboot):** the container was running via
  `docker run --rm`, so it is gone after reboot; the last verified checkpoint before the reboot stands
  (atomic writes mean no torn state). No training PID survives a reboot (nothing in this workflow
  registers as a Windows service or auto-restarting process).
- **Recovery:** confirm Docker Desktop is running again, `status --run <dir>` to confirm state, then
  `resume --run <dir> --confirm-full-corpus-run`.
- **Manual intervention:** yes — nothing in this codebase auto-resumes after a host reboot.

### 11. User presses Ctrl+C **(unverified — see R-note in the audit's failure-mode review)**
- **Do not use Ctrl+C.** Use the documented graceful stop instead:
  `stop --run <dir>` (writes a `STOP_REQUESTED` sentinel; the orchestrator checks this between shards,
  never mid-shard).
- If Ctrl+C is used anyway: the in-flight `docker run` subprocess's fate is unverified — it may be
  orphaned (Windows `Ctrl+C` does not automatically propagate to a child `docker run` process spawned
  via `subprocess.run`). Check `docker ps` afterward and manually `docker stop`/`docker rm` any
  orphaned container before resuming.

### 12. Dashboard crashes **(verified safe)**
- Both dashboards are read-only by construction (verified by direct inspection — neither can start,
  stop, or modify a run). A dashboard crash has zero effect on a running training session.

### 13. Telemetry file becomes malformed **(verified, bounded)**
- `TelemetrySampler` writes atomically; a malformed `status.json` from a torn write is structurally
  prevented. If `_tick()` itself raises an unexpected exception (not just the guarded `ImportError`
  for `psutil`), the sampler thread silently dies (R-012) — training itself is unaffected, but
  telemetry goes stale, which the existing staleness detection (`_is_heartbeat_stale`) will surface
  as `INTERRUPTED` in monitoring state even though training is actually still running. Cross-check
  `docker ps` / actual process state before trusting an `INTERRUPTED` classification as ground truth
  if this is suspected.

### 14. Duplicate launch attempt **(verified, rejected)**
- The launch guard rejects `run_state.status in (RUNNING, STOPPING)`. Clean rejection, no side effects.

### 15. Resume of a completed/failed run **(verified, rejected — fixed in this audit)**
- The launch guard now rejects `is_resume=True` combined with `status in (COMPLETED, FAILED)`.

### 16. Resume uses changed code **(verified, rejected — fixed in this audit)**
- The launch guard now compares live `git rev-parse HEAD` against the frozen `code_commit_hash`,
  rejecting a mismatch unless `--allow-code-revision-drift` is explicitly passed.

### 17. Resume uses changed data **(verified, rejected)**
- `configuration_hash` mismatch raises a hard `ValueError` inside `run_training_session` itself (not
  just the launch guard) — a genuine, structural, double-enforced protection.

---

## General recovery procedure (any unexplained stop)

1. `status --run <dir>` — read `status`, `stop_reason`, `latest_checkpoint`, `best_checkpoint`.
2. `docker ps` — confirm no orphaned container is still running against this run's directories.
3. Inspect the last shard's captured stdout/stderr (`run-state/epoch_output/epoch_<N>/log.csv`, or
   the epoch result's `stdout_tail`/`stderr_tail` if still accessible) for the real root cause.
4. Confirm the latest checkpoint is genuinely `resumable=true` in `checkpoint_index.json` before
   trusting a `resume`.
5. If the root cause is understood and not one of the launch-guard-rejected conditions above:
   `resume --run <dir> --confirm-full-corpus-run`.
6. If the root cause is NOT understood: do not resume blindly. Escalate to the ML lead / repository
   maintainer per the go/no-go checklist's sign-off structure.
