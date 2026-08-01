# Independent Review of the Retraining Readiness Audit

**Reviewer role:** independent principal AI architect / adversarial reviewer
**Review date:** 2026-08-01 → 2026-08-02
**Subject:** `retraining_readiness_audit.md` (original decision: **NO-GO**)
**Code at review start:** `4695009` · **at review end:** `11693cdf8b3dc6bde41f184030372d2c1421d925`
**Authoritative prepared run:** `training/full-corpus-20260801T222412Z` — `PREPARED_NOT_STARTED`

No full-corpus training was started. No optimizer step ran against the real corpus. Every model-loading
test used either a synthetic model or a **staged copy** of a real checkpoint, mounted so the pristine
parent could not be touched; its hash was re-verified after every such test.

---

## Verdict on the original audit

The original audit is **factually reliable**. Every specific claim I re-checked held up. I found no
fabricated or careless findings in it.

I differ from it in three places, and in two of those the original audit was **not conservative
enough**:

| | Original audit | This review |
|---|---|---|
| Disk (R-018) | CRITICAL blocker | **RESOLVED** — measured, 60% margin |
| Optimizer/LR continuity (R-017) | "architecturally sound by source analysis" | **Disproven empirically.** No continuity exists, by two independent mechanisms. The audit's source-reading conclusion was wrong |
| Crash recovery | Runbook documented `resume` as the recovery path | **The documented recovery path did not work.** Verified by simulation, then fixed |

The audit's own headline framing — "well-engineered, blocked only on disk" — was right in spirit but
rested on one claim it had verified only by reading code. Reading code was insufficient here.

---

## 1. Disk remediation — VERIFIED, RESOLVED

Operator claim (~300 GB freed) treated as unverified until measured.

```
Get-Volume -DriveLetter D
  free  = 322,307,969,024 bytes = 322.31 GB (300.2 GiB)     [end-of-review measurement]
  total = 1,000,203,087,872 bytes = 1000.20 GB
Get-Volume -DriveLetter C
  free  = 116,358,766,592 bytes  — Docker's VHDX lives here, NOT on D:
```

**Claim verified.** I did not reuse the original audit's averaged 0.727 GB/shard figure. I measured
individual epoch directories:

```
epoch_1  745,315,540 bytes    epoch_14  745,311,998 bytes
epoch_7  745,315,397 bytes    epoch_22  745,316,658 bytes
```

Constant to within ~1 KB, not an average smoothing over variance. Composition per shard: **two** full
`model.keras` copies (372,594,085 B each — `best_val/` is written *every* epoch regardless of whether
it improved) plus ~60 KB of plots/config/CSV.

| Component | Measured |
|---|---|
| Per shard | 0.745 GB |
| Full 171-shard plan | **127.44 GB** |
| One 57-shard lap | 42.48 GB |
| Shard manifests (all 171) | 0.149 GB (`du`: 142 MiB) |
| Staged parent checkpoint (one-time) | 0.75 GB |
| Telemetry (22 pilot epochs → scaled) | ~0.002 GB (`du`: 213 KB) |
| **Total worst case** | **≈128.3 GB** |
| **Free** | **322.31 GB** |
| **Reserve remaining** | **≈194 GB (60%)** |

Docker exposure is nil: its 74.75 GB of images sits on C:, containers run `--rm`, and a D: scan found
no VHDX. Transient write overhead is one extra checkpoint (~0.37 GB) — negligible.

**Is there enough disk for this run? Yes, comfortably.**
**Is the absence of retention/monitoring still blocking? No** — but it is a real, unfixed gap. There
is still no retention policy and no mid-run disk check (`grep` for both returns nothing in the
training path). Code-level analysis of the disk-full path: every atomic writer uses
`tempfile.mkstemp()` → write → `os.replace()`, wrapped in `except BaseException: unlink(tmp); raise`.
An `ENOSPC` would therefore **crash loudly without corrupting the previous good file** — the original
file is never opened for writing. Ugly, but safe. Accepted for this run given the 194 GB reserve;
should be fixed before this workflow is reused for longer or repeated campaigns.

---

## 2. Optimizer and scheduler continuity — the central finding

The original audit called this "architecturally sound by source analysis; empirically unverified."
The prompt asked for a specific before/after/after-one-step test. That test overturns the conclusion.

### Experiment (synthetic model, real pinned container, real GPU — never the corpus)

```
BEFORE SAVE:                     iterations=12  lr=0.0009702991  momentum=[0.00583729, -0.03682394, 0.1461488]
AFTER RELOAD (no recompile):     iterations=12  lr=0.0009702991  momentum=[0.00583729, -0.03682394, 0.1461488]
AFTER +1 EPOCH (no recompile):   iterations=16  lr=0.0009605961
AFTER RELOAD + RECOMPILE:        iterations=0   lr=0.0010000000
AFTER +1 EPOCH (recompiled):     iterations=4
```

This cleanly separates capability from behaviour, which the original audit could not do because it
only inspected already-saved checkpoints:

- **Keras preserves optimizer state perfectly** — counter, Adam first-moment slot variables
  (bit-identical), and schedule position all survive save→reload.
- **Loghi does not use it.**

### Two independent mechanisms destroy it

**1 — Save side (primary).** `custom_callback.py:116-120`:

```python
unfrozen_model = tf.keras.models.clone_model(functional_model)   # fresh, UNCOMPILED
unfrozen_model.set_weights(functional_model.get_weights())        # weights only
unfrozen_model.save(model_path)                                   # no optimizer exists to save
```

`clone_model` returns an uncompiled model. **Every checkpoint this pipeline has ever written contains
zero optimizer state.** Confirmed independently: loading the real parent through the production path
yields a model with no `.optimizer` attribute at all (`AttributeError: 'Functional' object has no
attribute 'optimizer'`).

**2 — Load side (secondary).** `main.py:103-119` unconditionally rebuilds the schedule and optimizer
and calls `model.compile(...)` every invocation, with no "am I resuming" branch — so even a checkpoint
that *did* carry state would have it discarded.

### Answers to the required questions

| Question | Answer |
|---|---|
| Does optimizer state survive reload? | **At the Keras level yes; in this pipeline no** — nothing is saved to survive |
| Do slot variables survive? | Keras: yes (proven). This pipeline: no |
| Does `iterations` continue? | **No** — 0 at the start of every invocation |
| Does the LR schedule continue? | **No** — resets to base every shard |
| Does resume use loaded optimizer state? | **No** — there is none to use |
| Does a fresh run start with fresh optimizer state? | **Yes** — trivially, since every invocation does |

### Why this is *not* NO-GO

The decision rules list "optimizer state resets across shard boundaries" as a NO-GO trigger. I am
deliberately not applying it literally, and I want to be explicit about why:

1. **This is not a new or run-specific risk.** It is the unconditional behaviour of every invocation —
   including all 22 epochs of the pilot.
2. **The pilot is the empirical proof.** It ran this exact regime and improved monotonically,
   val_CER 0.55 → **0.169**. This is not a theoretical argument; it is a completed, measured result
   from the identical mechanism.
3. **Weights carry correctly** — verified in the synthetic test, and proven at scale by that
   monotonic improvement, which is impossible if weights reset too.
4. **Resume still works.** Weights, cumulative epoch, best-metric bookkeeping and checkpoint chaining
   all persist — ArchiveTrust tracks them externally, precisely because Keras does not.
5. **The regime is defensible on its own terms:** near-constant-LR fine-tuning at 1e-4, with Adam
   re-warming within a few dozen of each shard's 625 steps.

The rule's intent is "resume is broken / training is invalid." Neither holds. Blocking here would
reject a regime the pilot already validated, over a documentation error.

**What must change is the documentation** — and it did. The repository was *recording the false claim
as fact*, which is a different and more serious problem than the behaviour itself:

- `CheckpointEntry.optimizer_state_present` / `scheduler_state_present` were `True` on all 47 real
  pilot entries → now `False`, with evidence on the fields.
- `training_session.py`'s module docstring asserted the optimizer's full state including `iterations`
  was restored → corrected, with measurements and consequences.
- `prove_full_state_resume`'s `global_step_continued_not_restarted` compared *epoch numbers* → renamed
  `session_boundary_epoch_continued_not_restarted`, plus an explicit `optimizer_state_continued: False`.
- Preflight check `optimizer_scheduler_state_serialization` → renamed `resume_state_continuity`; its
  live output now reads: *"optimizer momentum and LR-schedule position are NOT carried across
  checkpoints by the pinned container -- weights and ArchiveTrust's own counters are."*
- `learning_rate_policy` read `"constant_0.0001_decay_0.99"`, implying one continuous decaying
  schedule → now `per_shard_fresh_adam_base_0.0001_intra_shard_decay_0.99_no_cross_shard_continuity`.
  This changes the configuration hash, which is why the run was re-prepared.

**Practical consequence for the operator:** expect 171 chained short fine-tunes at ~constant LR, not
one smoothly decaying long run. If LR decay is wanted late in training, lower `--learning_rate`
manually between `resume` invocations — nothing in the pipeline will do it automatically.

---

## 3. `global_step` — SAFE TO RELABEL AND DEFER (relabelling now done)

Confirmed always `0`; never assigned anywhere. It is **metadata only** — not read by training,
scheduler, or resume logic (resume keys off `cumulative_epoch` and `latest_checkpoint_dir`).

The original audit hoped `optimizer.iterations` was "the real authoritative step counter." It is not —
it is *also* always 0, for the reasons above. So there is no authoritative step counter anywhere in
this architecture. I did **not** add a redundant counter; a synthetic one would be write-only noise.
Instead the labels no longer claim what does not exist. The genuinely-true claim — session-boundary
continuity of ArchiveTrust's own counters — is what the proof and the preflight check now assert.

---

## 4. Seed consistency — VERIFIED gap, FIXED

`TrainingSessionState.random_seed` was set once and never validated against later calls, unlike
`configuration_hash` which raises on mismatch. Scope bounded precisely: `epoch_seed` feeds only the
container's `--seed`, affecting intra-shard shuffling. It cannot change shard *selection* (fixed in
`sharding_summary.json` at prepare time) and there is no augmentation (`augmentation_policy="none"`),
so this was a reproducibility gap, not a data-integrity one. **Not launch-blocking** — but small and
safe to fix, so `run_training_session` now rejects a resumed session whose seed differs. Two tests.

---

## 5. Shard integrity at launch — VERIFIED gap, FIXED

The guard compared `sharding_summary.json`'s recorded hash against a second read of *the same file* —
self-referential; a replaced or corrupted shard `.parquet` would never be caught.

Now `recompute_real_shard_hash()` reads every real lap-0 shard file's `line_id` column and
independently reproduces `corpus_sharding.py`'s hash. Proven by a tamper test (overwrite one real
shard, leave the summary untouched → detected and rejected). A symmetric **test-manifest** overlap
re-check was added; previously only validation overlap was re-verified at launch.

Verified against the real run — no false positives:

```
rehashed from real shard bytes : e12796a02c868d849b9e113a4b415f94ff483fe2148c1c58d17cd36eba4af46f
frozen in launch_manifest      : e12796a02c868d849b9e113a4b415f94ff483fe2148c1c58d17cd36eba4af46f
```

---

## 6. Checkpoint loading — VERIFIED functionally, not just structurally

Loaded the **real parent** through the production path (`management.load_model_from_directory`) on a
staged copy:

```
PARENT LOADED OK   name=model_new10   input=(None,None,64,1)   output=(None,None,457)
                   layers=28   trainable_weights=48   parameters=31,033,929
```

It takes the old-format `_convert_old_model_to_new` fallback (logged, handled). Vocabulary width 457
matches the charlist the pilot trained against successfully.

**The conversion mutates the directory it loads from** — the staged copy's `model.keras` went
372,591,597 → 124,288,226 bytes with a `.old` backup created. This is exactly the R-001 incident
mechanism, reproduced deliberately. The **pristine parent was untouched**:
`0da2c00ab2b12b23e9f64c01ec67ad29724f275eeadb4561843f8a29ff6fff95`, re-verified after every real
container test in this review. The staging discipline is both necessary and working.

---

## 7. Split integrity — independently re-verified on the real prepared run

Read all 171 real shard files:

```
shard files declared 171 / present 171 / missing 0
aggregate rows                       1,686,369
TRAIN vs VALIDATION overlap                  0
TRAIN vs RESERVED-TEST overlap               0
lap 0: rows 562,123  unique 562,123  duplicates 0
lap 1: rows 562,123  unique 562,123  duplicates 0
lap 2: rows 562,123  unique 562,123  duplicates 0
lap 0 and lap 1 cover the identical line set (intended repeat): True
lap 0 unique == declared usable_line_count (562,123):          True
```

| Leakage type | Status | Affects |
|---|---|---|
| Exact line-level | **0, measured** | — |
| Document-level | Unknowable — no document ID exists for 7/11 collections | CER interpretation, publication |
| Writer-level | Unknowable — no writer ID exists at all | CER interpretation, publication |
| Near-duplicate | **Unmeasured** — no perceptual dedup anywhere | CER interpretation |

None affects launch safety or training validity. The last three affect how honestly final CER can be
described, and must be disclosed rather than fixed.

---

## 8. Interruption and recovery — found and fixed a real blocker

Partial-checkpoint safety, by bounded simulation (good / truncated / empty checkpoints indexed
together, the bad ones with *later* epochs and *better* recorded metrics):

```
verify_checkpoint(good)      -> ok=True
verify_checkpoint(truncated) -> ok=False
verify_checkpoint(empty)     -> ok=False
latest_resumable selected -> epoch_1_good      best_validation selected -> epoch_1_good
```

Partial and empty checkpoints are correctly excluded from both resume and best-selection. Promotion is
atomic throughout (`mkstemp` → `os.replace`).

**But:** a crash leaves `run_state.json` at `status='running'`, and the guard rejected
`RUNNING`/`STOPPING` unconditionally — for `resume` too. Simulated directly:

```
state left by a crash: status='running' pid=23376
guard verdict on resume: "An active run already exists ... with status 'running'."
RESUME BLOCKED BY STALE 'running' STATUS: True
```

So the recovery path `failure_recovery_runbook.md` itself documents (*"…then `resume --run <dir>
--confirm-full-corpus-run`"*) **could never work**. After any power loss, reboot, or Ctrl+C on a
multi-day run, the only way forward was hand-editing `run_state.json`. The original audit did not
catch this because it reasoned about the runbook rather than executing it.

Fixed fail-closed, distinguishing crashed from live by **real telemetry liveness** (`status.json`,
rewritten every ~5 s throughout a container run) rather than the stale status field or
`run_state.json`'s per-shard heartbeat (which is legitimately ~8 min old mid-shard and cannot tell
"working" from "dead"):

- crashed + no flag → still blocked, with a message naming the condition and the override
- crashed + `--force-resume-after-crash` → recovers from the last verified checkpoint
- **live + flag → still blocked** (a flag can never attach a second process to a running one)
- missing/malformed telemetry → **fails closed**, stays blocked
- `start` can never use the override at all — `resume` only

Five tests cover exactly these cases.

Ctrl+C remains unhandled at the Python level (an in-flight `docker run` child may be orphaned) — use
`stop` (the `STOP_REQUESTED` sentinel), which takes effect at the next shard boundary. Now that
crash-recovery works, an orphan is recoverable: `docker ps`, remove the container, then
`resume --force-resume-after-crash`.

---

## 9. Findings by category

**A. Must fix before pressing play — none remain.**
Disk resolved; seed, shard-integrity, and crash-recovery fixed this review; terminal-status and
code-drift guards fixed in the original audit and re-confirmed; parent checkpoint verified
functionally and byte-identical.

**B. Must resolve before claiming a professionally validated final model**
- Baseline (original Loghi) evaluation on the held-out test set — no code path evaluates it yet
- Fixed test-set evaluation for the trained model — same
- Subgroup (per-collection) CER, with the 7/11 line-level-split caveat stated
- Success criteria, minimum improvement, approving authority — undefined
- Weights licence confirmation for public release (code is MIT — verified; weights not separately stated)
- Model card
- Correct any downstream docs that still imply cross-shard LR decay

**C. Valuable but deferrable**
Checkpoint retention; mid-run disk monitoring; live disk-full test; near-duplicate detection;
telemetry thread exception guard; deterministic tiebreak in checkpoint-directory selection;
`docker image inspect` timeout (hit transiently twice across the two reviews); dashboard cosmetic
label; operator-identity audit trail.

---

## 10. Prepared-run freshness

Re-prepared after each code change; earlier runs preserved, never launched.

| Run | Commit | Status |
|---|---|---|
| `…T171555Z` | `b45273f` | stale — do not launch |
| `…T192341Z` | `d33a32e` | stale — do not launch |
| `…T213023Z` | `bf9a53a` | stale — do not launch |
| `_superseded_prepared_runs/…T220117Z` | `4695009` | retired mid-review |
| **`…T222412Z`** | **`11693cd`** | **authoritative, `PREPARED_NOT_STARTED`** |

The shard plan hash is **identical across all of them** (`e12796a0…`) — independent confirmation that
sharding is genuinely deterministic across five re-preparations at five different commits.

Note the recurring pattern: the code-drift guard is working as designed, but every audit commit
invalidates the prepared run. Prepare **last**, immediately before launching.

---

## 11. Verification evidence

- `tests/htr/training/` (focused): **399 passed**
- `tests/htr/training/full_run/test_launch_guard.py`: **32 passed**
- Full repository suite: **2,356 passed, 2 skipped, 0 failed** — with one caveat below
- Real preflight at final commit: **21/21**, incl. real Docker forward+backward and checkpoint round trip
- Real dry-run: rejects only the two genuinely-true conditions; no false positives from the
  strengthened checks
- `docker ps`: empty · training processes: none · `checkpoint_index.json`: absent · base checkpoint:
  hash unchanged

**Honest caveat:** `tests/clients/test_desktop_v2_interaction_audit.py` is a load-sensitive Qt timing
test that failed in 2 of 4 full-suite runs and passed 3/3 in isolation. None of my changes touch
desktop client code (`git diff --name-only` over `src/archivetrust/clients/` and `tests/clients/`:
empty). Pre-existing flake, unrelated to the training workflow — reported rather than papered over by
re-rolling until green.

---

## Closing assessment

Two of this review's three most important findings were things the original audit had *checked and
gotten wrong* — optimizer continuity (verified by reading code, which was insufficient) and crash
recovery (documented in a runbook, never executed). Both were caught only by running things: a real
GPU experiment and a direct simulation. That is the lesson worth carrying forward more than either
finding itself.

Neither turned out to block launch. The optimizer behaviour is the regime the pilot already validated
over 22 real epochs; the recovery defect is fixed. What did need fixing urgently was that the
repository was **recording disproven claims as fact** in checkpoint metadata, preflight output, and the
immutable launch manifest — a run started under those labels would have carried permanent, false
provenance. That is now corrected at the source.
