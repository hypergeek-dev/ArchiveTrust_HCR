# Go / No-Go Checklist — Loghi Full-Corpus Retraining

Use this immediately before running the real launch command — re-verify every item against the
**current** state, not this document's snapshot from 2026-08-01. Every item is checkable with a real
command or a real file. Do not check an item from memory.

## Blocking (must all be ✅ or the answer is NO-GO)

- [ ] **Disk capacity.** `Get-PSDrive D | Select-Object Free` reports free space ≥ (measured per-shard
  checkpoint cost × configured `max_full_run_epochs`) + a real safety margin, **or** a checkpoint
  retention policy has been implemented and verified to actually prune old checkpoints during a real
  run. As of this audit: **not satisfied** (R-018). Command: `powershell -Command "Get-PSDrive D | Select-Object Free"`.
- [ ] **Preflight passes completely, freshly.** Re-run `preflight` right before launch; confirm the
  printed summary is `N/N passed` with `All critical checks passed: True`. Do not reuse a preflight
  result more than a few hours old.
- [ ] **Docker reachable and image present.** `docker ps` succeeds; `docker image inspect
  loghi/docker.htr:latest` returns the pinned digest `sha256:414fc89ac574a61fd745836ad9852842759bf3cf7046e4549ae96315ff9132e8`.
- [ ] **Base checkpoint hash matches the pin.** Recompute sha256 of
  `.loghi-upstream/pretrained-models/loghi-htr/generic-2023-02-15/model.keras` and compare to
  `0da2c00ab2b12b23e9f64c01ec67ad29724f275eeadb4561843f8a29ff6fff95`.
- [ ] **Pilot checkpoint is not the parent.** `launch_manifest.json`'s `pilot_checkpoint_used_as_parent`
  is `false`.
- [ ] **No active conflicting run.** `run_state.json` for the target run has `status: "prepared"`
  (not `running`/`stopping`); no other run under `training/` has `status: "running"`.
- [ ] **Code revision matches what was audited.** `git rev-parse HEAD` equals
  `launch_manifest.json`'s `code_commit_hash`, or the drift is explicitly, knowingly accepted via
  `--allow-code-revision-drift` with a written note of why.
- [ ] **Repository is clean, or the dirty state is explicitly reviewed.** `git status --porcelain`
  is empty, or every uncommitted change has been read and is understood before using
  `--allow-dirty-repository`.
- [ ] **Dataset and training-manifest hashes are unchanged since `prepare`.** These are re-verified
  automatically by the launch guard; if the guard rejects on either, do not override without
  understanding exactly what changed.
- [ ] **Explicit confirmation flag is present.** The command about to run includes
  `--confirm-full-corpus-run`, typed by a human, not scripted.

## High-priority conditions (should be true; each is an explicit, accepted risk if not)

- [ ] R-004/R-005 (`global_step` tracking / resume-proof labeling) has been read and accepted by the
  ML lead, or fixed.
- [ ] R-006 (`random_seed` silent-override-on-resume) has been read and accepted, or fixed. Confirm
  the same `--seed` value is used on every `start`/`resume` call for this run if not fixed.
- [ ] R-007 (shard-integrity self-reference at launch; validation-only overlap re-check) has been
  read and accepted, or fixed.
- [ ] R-009 (line-level split leakage risk for 7/11 collections) will be disclosed in any CER claims
  derived from this run.

## Evidence to have open/available at launch time

- [ ] `launch_manifest.json` for the target run
- [ ] `LAUNCH_COMMANDS.txt` for the target run (matches the command about to be typed, exactly)
- [ ] `retraining_readiness_audit.md` and `retraining_risk_register.json`
- [ ] Real, current output of `status --run <dir>`
- [ ] Real, current output of `docker ps`

## Sign-off

- [ ] Operator name/date: ______________________
- [ ] ML lead name/date (if R-004/R-005/R-006/R-007/R-009 conditions are being accepted rather than
  fixed): ______________________

## After confirming every box above

The exact command to run is in `manual_launch_preview.txt`. It has intentionally not been executed
by this audit, and should be typed by a human, not pasted by an automated agent.
