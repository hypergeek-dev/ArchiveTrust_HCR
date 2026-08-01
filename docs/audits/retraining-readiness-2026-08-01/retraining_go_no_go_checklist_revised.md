# Go / No-Go Checklist — Revised (independent review, 2026-08-02)

Supersedes `retraining_go_no_go_checklist.md`.
**Authoritative run: `training/full-corpus-20260801T222412Z`** (commit `11693cd`).

Re-verify against the **current** state at launch time, not this snapshot.

---

## Blocking — all must be ✅

- [x] **Disk capacity.** RESOLVED: 322.31 GB free vs ~128.3 GB worst case (60% reserve), independently
      measured. Re-check: `powershell -Command "(Get-Volume -DriveLetter D).SizeRemaining"` — want ≥ 150 GB.
- [ ] **Preflight passes, freshly.** Re-run immediately before launch; expect `21/21`. The guard rejects
      a preflight older than 6 h. *(If `container_image_available` fails on a 15 s timeout — R-015 — retry
      once; it has done this three times across two reviews and always passes on retry.)*
- [ ] **Docker + image.** `docker ps` succeeds; `docker image inspect loghi/docker.htr:latest` returns
      `sha256:414fc89ac574a61fd745836ad9852842759bf3cf7046e4549ae96315ff9132e8`.
- [ ] **Base checkpoint hash.** `.loghi-upstream/pretrained-models/loghi-htr/generic-2023-02-15/model.keras`
      = `0da2c00ab2b12b23e9f64c01ec67ad29724f275eeadb4561843f8a29ff6fff95`.
- [ ] **Parent is not the pilot.** `launch_manifest.json` → `pilot_checkpoint_used_as_parent: false`.
- [ ] **Run is `prepared`.** `run_state.json` → `status: "prepared"`, `pid: null`, no `checkpoint_index.json`.
- [ ] **Commit matches.** `git rev-parse HEAD` == `launch_manifest.json.code_commit_hash`.
      **Prepare LAST** — any commit after preparing (even docs) invalidates the run (R-016).
- [ ] **Working tree clean**, or `--allow-dirty-repository` used with the diff actually read.
- [ ] **Dry run is clean.** Expect *only* "missing confirmation" (+ dirty-repo if applicable).
      Any hash / overlap / shard complaint → **stop and investigate**, do not override.
- [ ] **Seed decided.** Typically `42`; must be identical on every later `resume` (now enforced).
- [ ] **`--confirm-full-corpus-run` typed by a human.**

## Verified once — no per-launch action

- [x] Shard-byte integrity (real rehash, tamper-tested) · test-manifest overlap re-checked
- [x] Exact leakage: train↔val 0, train↔test 0, no duplicates within any lap — measured on all 171 files
- [x] Parent checkpoint functionally loadable; pristine copy provably untouched by staging
- [x] Partial/empty checkpoints excluded from resume and best-selection
- [x] Crash recovery works (R-020 fixed) · terminal-status and code-drift guards active

## Understand before launching (not blockers)

- [ ] **No cross-shard optimizer/LR continuity (R-019).** Every shard starts a fresh Adam at base LR
      1e-4. Expect **171 chained short fine-tunes**, not one smoothly decaying run. This is the regime
      the pilot validated (val_CER 0.55 → 0.169). To decay LR later, lower `--learning_rate` manually
      between `resume` calls — nothing does it automatically.
- [ ] **No checkpoint retention (R-018).** ~0.745 GB per shard, forever. Watch free space at Gate 4/5.
- [ ] **Document/writer leakage unknowable for 7/11 collections (R-009)** — disclose in any CER claim.

## Governance — before final model acceptance, not before launch

- [ ] Baseline evaluated on the held-out test set (no code path does this yet)
- [ ] Test-set evaluation for the trained model
- [ ] Per-collection subgroup CER with the R-009 caveat
- [ ] Success criteria, minimum improvement, approving authority
- [ ] Weights licence confirmed if publishing/redistributing (code is MIT — verified)
- [ ] Model card

## Mandatory human gates

- [ ] **Gate 4 — after shard 1.** Use `first_shard_review_checklist.md`. Do not leave unattended before this.
- [ ] **Gate 5 — after the first 57-shard lap.** Use `first_epoch_review_checklist.md`. Decide whether the
      constant-LR regime warrants a manual LR reduction for later laps.

**Recommended first invocation:** `--hours 1.0` (not 5.0) so Gate 4 arrives quickly.

## Sign-off

- [ ] Operator: ______________________  Date: __________
- [ ] ML lead (accepting R-009, R-018, R-019): ______________________  Date: __________

Command: see `manual_launch_preview.txt`. Type it — do not paste from an agent.
