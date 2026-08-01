# Go / No-Go Checklist — Revised

Supersedes `retraining_go_no_go_checklist.md` for the fresh run. Re-verify every item against the
**current** state at launch time, not this snapshot (2026-08-01). Target run:
`training/full-corpus-20260801T213023Z`.

## Blocking (must all be ✅ or the answer is NO-GO)

- [x] **Disk capacity.** ~~Not satisfied at original audit time~~ — **RESOLVED**: 322.73GB free vs
  127.44GB required for the full 171-shard plan (61% margin), independently measured in this review.
  Re-verify at launch time: `powershell -Command "Get-Volume -DriveLetter D | Select-Object SizeRemaining"`.
- [ ] **Preflight passes completely, freshly.** Re-run `preflight` immediately before launch (do not
  reuse this review's 21/21 result if more than a few hours have passed).
- [ ] **Docker reachable and image present.** `docker ps` succeeds; `docker image inspect
  loghi/docker.htr:latest` returns digest `sha256:414fc89ac574a61fd745836ad9852842759bf3cf7046e4549ae96315ff9132e8`.
  (Note: this specific check has twice hit a transient 15s timeout in real testing — R-015 — retry
  once before concluding it has genuinely failed.)
- [ ] **Base checkpoint hash matches the pin.** Compare
  `.loghi-upstream/pretrained-models/loghi-htr/generic-2023-02-15/model.keras` sha256 against
  `0da2c00ab2b12b23e9f64c01ec67ad29724f275eeadb4561843f8a29ff6fff95`.
- [ ] **Pilot checkpoint is not the parent.** `launch_manifest.json`'s `pilot_checkpoint_used_as_parent`
  is `false`.
- [ ] **No active conflicting run.** Target run's `run_state.json` has `status: "prepared"`.
- [ ] **Code revision matches what was reviewed.** `git rev-parse HEAD` equals `bf9a53a8863d6ad4f4faf74b38e2aa9e9cc8fcf0`,
  or the drift is explicitly reviewed and accepted via `--allow-code-revision-drift`.
- [ ] **Repository is clean, or the dirty state is explicitly reviewed.**
- [ ] **Random seed is the same across every start/resume invocation for this run.** (Now enforced —
  a mismatch is rejected automatically — but confirm the intended seed, typically `42`, before the
  first `start`.)
- [ ] **Shard integrity re-check passes with no false positive.** A real dry-run against
  `training/full-corpus-20260801T213023Z` should report no "training manifest hash changed" or
  "reserved test manifest" complaints. If either appears, stop and investigate before overriding.
- [ ] **Explicit confirmation flag is present.** `--confirm-full-corpus-run`, typed by a human.

## No longer blocking (resolved in this review, verify once, not on every launch)

- [x] Random-seed drift on resume is now rejected automatically (no manual check needed beyond
  supplying a consistent `--seed`).
- [x] Shard-file tampering/corruption is now detected automatically by a real rehash (no manual
  re-verification needed beyond trusting a clean dry-run).
- [x] Functional checkpoint loadability is proven (no manual action needed).

## Governance / documentation conditions (should be true before final model acceptance, not before launch)

- [ ] `learning_rate_policy` label and monitoring docs corrected to reflect the real, confirmed
  behavior (no continuous cross-shard decay — see R-019).
- [ ] R-009 (line-level split leakage for 7/11 collections) will be disclosed in any CER claims.
- [ ] Success criteria, minimum-improvement threshold, and approving authority defined (see
  `opus_independent_review.md` Phase 4 for a proposed structure).
- [ ] Weights license explicitly confirmed with the upstream Loghi project, if public release/
  publication/commercial use is intended (not needed for private/internal use).

## Evidence to have open/available at launch time

- [ ] `training/full-corpus-20260801T213023Z/launch_manifest.json`
- [ ] `training/full-corpus-20260801T213023Z/LAUNCH_COMMANDS.txt`
- [ ] `retraining_readiness_audit_revised.md` and `opus_independent_review.md`
- [ ] Real, current output of `status --run training/full-corpus-20260801T213023Z`
- [ ] Real, current output of `docker ps`

## Sign-off

- [ ] Operator name/date: ______________________
- [ ] ML lead name/date (confirming acceptance of the governance/documentation conditions above): ______________________

## After confirming every blocking box above

The exact command is in `manual_launch_preview.txt`. It has intentionally not been executed by this
review, and should be typed by a human.
