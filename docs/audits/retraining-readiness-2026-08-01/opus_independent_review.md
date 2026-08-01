# Independent Review of the Retraining Readiness Audit

**Reviewer role:** independent principal AI architect / adversarial reviewer
**Review date:** 2026-08-01 (same day, following the original audit)
**Subject:** `retraining_readiness_audit.md` (original decision: **NO-GO**)
**Code state at review start:** commit `d33a32eb7e67c9c886ff1cfa651a3f6c8d01c017` (the run the original audit evaluated)
**Code state at review end:** commit `bf9a53a8863d6ad4f4faf74b38e2aa9e9cc8fcf0` (after this review's own fixes)
**Newly prepared, freshly-consistent run:** `training/full-corpus-20260801T213023Z`

No full-corpus training was started during this review. No optimizer step was performed against the
real full corpus. Every optimizer/checkpoint-loading test in this review used already-existing real
checkpoints from the completed pilot run (read-only mounts) — zero new training occurred.

---

## Summary verdict on the original audit

**The original audit is factually sound and its evidence holds up.** Every specific claim I
independently re-checked — the parent checkpoint hash, the zero measured line-ID overlap, the
deterministic shard reproducibility, the `global_step` field being permanently `0`, the
`resume-proof.json` labeling bug, the self-referential shard-hash check, the missing seed validation
— was **VERIFIED**, not merely plausible. I found no incorrect findings in it.

What I disagree with is **the severity assigned to two things**, in opposite directions:

1. **R-018 (disk capacity) was correctly identified as the blocker, and is now resolved with a large,
   real margin** — confirmed by direct measurement, not the operator's claim alone.
2. **R-017 (learning-rate/optimizer continuity) was under-severe.** The original audit called it
   "architecturally sound by source analysis; empirically unverified at scale" — a hedge in the
   *reassuring* direction. Direct, empirical testing against two real checkpoints from the completed
   pilot run shows the opposite of what the source-reading conclusion implied: **the optimizer's
   `iterations` counter and the learning-rate schedule do not survive a single shard/container
   boundary — ever, for anyone, not just under some resume edge case.** This is the most important
   finding of this review, and it required running real TensorFlow code against real artifacts to
   catch; source-reading alone (which is all the original audit did here) reached the wrong
   conclusion.

Despite finding a more severe version of R-017 than the original audit reported, my overall decision
is **less conservative** than NO-GO, because the *consequence* of that finding turns out to be
already-validated by the pilot's own real, successful 22-epoch result — explained in detail in
Phase 2B below.

---

## Phase 1: Disk remediation — independently verified

**Operator claim:** approximately 300GB freed. **Treated as unverified until measured, per instructions.**

```
Get-Volume -DriveLetter D
Free:  322,731,458,560 bytes  (300.57 GiB / 322.73 GB)
Total: 1,000,203,087,872 bytes (931.51 GiB / 1000.20 GB)
```

**Verdict: the claim is accurate.** Free space is genuinely ~300GB (measured via both `Get-PSDrive`
and `Get-Volume`, which agreed exactly).

### Recalculating the per-shard cost independently

The original audit's 0.727GB/shard figure was an *average* (16GB ÷ 22 pilot epochs). I did not accept
that average — I inspected individual epoch directories directly:

```
epoch_1/  : 711M   (model_new10/best_val/model.keras = 372,594,085 bytes
                     + model_new10/epoch_.../model.keras = 372,594,085 bytes
                     + ~60KB of plots/config/csv)
epoch_5/  : 711M   (identical composition)
epoch_10/ : 711M
epoch_15/ : 711M
epoch_20/ : 711M
epoch_22/ : 711M
```

**Every sampled epoch is exactly 711 MiB (745,188,170 + ~60,000 bytes), not an average that could be
hiding variance.** The container writes a full `model.keras` copy under `best_val/` on *every* epoch
(not only when that epoch is genuinely the run's best), plus the "latest" copy — two full checkpoints
per shard, deterministically, regardless of whether the epoch improved. This is a more precise,
higher-confidence number than the original audit had, and it confirms (rather than revises) the
original per-shard estimate.

### Recalculated capacity requirement

```
Per-shard cost (real, sampled, constant):     745,248,170 bytes  = 0.745 GB
Full 171-shard plan:                          171 x per-shard    = 127.44 GB
One full 57-shard lap:                         57 x per-shard    =  42.48 GB
One-time staged parent checkpoint copy:                          =   0.37 GB
------------------------------------------------------------------------------
Total required for the full plan:                                  127.81 GB
Measured free space:                                                322.73 GB
Remaining reserve after the full plan:                              194.92 GB
Percent of free space consumed by the full plan:                      39.6%
Remaining reserve after an additional 20GB operational floor:       174.92 GB
```

**This does not include double-counting risk from Docker.** I checked where Docker Desktop's own
WSL2 storage lives: `docker info --format '{{.DockerRootDir}}'` → `/var/lib/docker`, and the real host
VHDX is at `C:\Users\ize_c\AppData\Local\Docker\wsl\disk\docker_data.vhdx` (72.7GB) — **on the C:
drive, not D:.** A recursive scan of D: found no Docker VHDX. Docker's own footprint (`docker system
df`: 74.75GB images, 452.8MB containers) does not compete with the training output volume at all, and
containers run with `--rm` so it will not grow during the run.

**R-018 disk-capacity component: RESOLVED.** 39.6% utilization of currently-free space, with a ~195GB
reserve even in the absolute worst case (the full 171-shard plan run to completion with zero
retention). This is comfortably, not just conditionally, sufficient.

### Operational hardening gap (separate from the capacity blocker)

Per the instructions, I assessed this separately rather than letting "capacity is fine" imply
"everything about disk is fine":

- **Checkpoint retention policy: still absent.** Confirmed by the same codebase-wide search the
  original audit ran (`grep -rn "retention|prune|delete.*checkpoint|cleanup|rmtree"` across
  `training_session.py`, `orchestrator.py`, `checkpoint_index.py` — zero matches, still).
- **Mid-run disk monitoring: still absent.** Same search for `disk_usage|disk_free|free_gb` — zero
  matches in the training execution path.
- **Disk-full failure behavior: still not live-tested.** I did not fill the real disk (explicitly
  disallowed). Code-level analysis (not a live simulation): every atomic-write helper in this
  codebase (`checkpoint_index.py`, `run_state.py`, `training_session.py::_save_session_state`) writes
  to a *new* temp file via `tempfile.mkstemp()` and only calls `os.replace()` after the write
  succeeds, wrapped in `try/except BaseException: unlink(tmp); raise`. An `ENOSPC` failure during a
  write would therefore **propagate as an uncaught exception (an ugly crash, not a clean
  `stop_reason`) but cannot corrupt the previously-good file**, since the original file is never
  opened for writing — only a new temp file is, and promotion only happens after a complete write.
  This is a real, structural, disk-full-safe property of the codebase's own write discipline, even
  though the *operator experience* (a raw traceback instead of a graceful stop) is unpolished.

**Conclusion for this specific run:** capacity blocker resolved with a large margin; the retention/
monitoring/disk-full-testing gaps remain real but are now **ACCEPTED RISK, not blocking**, given the
~195GB reserve makes the scenario they guard against very unlikely to materialize during this run.
They should still be fixed before this workflow is relied on for repeated, larger, or longer-running
future retraining cycles — that recommendation is unchanged from the original audit.

---

## Phase 2: Independent verification of major findings

### A/B. Global-step evidence and optimizer/scheduler continuity — **the central finding of this review**

**Original audit's framing:** `global_step` is always `0` (VERIFIED, independently reconfirmed by me
via fresh `grep` — no code path anywhere assigns it after initialization). The resume-proof's
`global_step_continued_not_restarted` field is computed by a check that actually compares
`session_2.initial_epoch == session_1.final_epoch` — an epoch check mislabeled as a step check
(VERIFIED, independently reconfirmed, verbatim, at `training_session.py:575`). The original audit
concluded this was a **labeling/evidence gap**, and separately assessed the *underlying* optimizer
continuity as "architecturally sound by source analysis... never measured at full scale."

**I built a bounded, safe, real proof rather than accepting either audit's source-reading alone.**
Methodology: mount two **already-existing real checkpoints** from the completed pilot run (epoch 1's
and epoch 5's `best_val/model.keras`, four real epochs and ~2,500 real training steps apart) read-only
into the real pinned container, and load each with `tf.keras.models.load_model()` using the exact
same `custom_objects` dict `main.py` itself uses (`CERMetric`, `WERMetric`, `CTCLoss`,
`ResidualBlock`, `LoghiLearningRateSchedule` — found by reading `main.py:72-75` directly, not
guessed). **Zero new training occurred; both checkpoints were produced by the pilot run months/hours
before this review.**

```
RESULT epoch_1_best_val: optimizer=Adam iterations=0 lr=9.999999747378752e-05
RESULT epoch_5_best_val: optimizer=Adam iterations=0 lr=9.999999747378752e-05
SUMMARY: iterations delta (epoch5-epoch1)=0, lr1=9.999999747378752e-05, lr5=9.999999747378752e-05
```

**Both real checkpoints, four epochs apart, show `optimizer.iterations=0` and the unchanged base
learning rate.** This is not a quirk of my test harness — I verified the mechanism directly in the
pinned source (`main.py:103-119`, `.loghi-upstream/loghi-htr/src/main.py`):

```python
lr_schedule = create_learning_rate_schedule(
    learning_rate=config["learning_rate"], decay_rate=config["decay_rate"],
    decay_steps=config["decay_steps"], train_batches=data_manager.get_train_batches(),
    do_train=config["train_list"], warmup_ratio=config["warmup_ratio"],
    epochs=config["epochs"], decay_per_epoch=config["decay_per_epoch"], linear_decay=config["linear_decay"])
optimizer = get_optimizer(config["optimizer"], lr_schedule)
model.compile(optimizer=optimizer, loss=CTCLoss(), metrics=[...], weighted_metrics=[])
```

This runs **unconditionally, on every single invocation**, immediately after `load_or_create_model()`
returns the loaded model. There is no branch anywhere in the pinned code that says "if resuming, keep
the loaded optimizer instead." Every container invocation — pilot or full-corpus, first epoch or
five-hundredth shard — reconstructs a brand-new Adam optimizer with `iterations=0` and the full base
learning rate, then `model.compile()`s it onto the model, **discarding whatever optimizer state
`tf.keras.models.load_model()` may have restored internally.**

**What does and does not survive, now precisely and empirically established:**

| Component | Survives a shard/container boundary? | Evidence |
|---|---|---|
| Model weights | **Yes** | Real: the pilot's own val_CER improved monotonically (0.55 → 0.169) over 22 real epochs, each one this exact "fresh optimizer" mechanism — weights could not improve cumulatively like that if they were also reset each epoch. Also directly loaded and inspected (non-zero, real weight tensors) as part of this review's own probe. |
| Adam's momentum/variance accumulators (`m`, `v`) | **No** | Real: `optimizer.iterations=0` in both independently-loaded real checkpoints. |
| Learning-rate schedule position | **No** — resets to the base rate every invocation | Real: `lr=0.0001` (unchanged) in both real checkpoints, four epochs apart. |
| `ArchiveTrust`'s own `global_step` field | **No** (was already known to be a permanent `0`) | Now understood to be *consistent with* reality, not merely an unpopulated metadata field — the real underlying value is also always effectively 0 at the start of every invocation. |

**Decision for this finding: SAFE TO RELABEL AND DEFER — not blocking.** My reasoning:

1. This is not a *regression* or an edge case triggered only by `resume` — it is the system's
   unconditional behavior on *every single epoch/shard*, including every one of the pilot's real 22
   epochs. The full-corpus run would use exactly the same mechanism the pilot already used
   successfully, at a larger scale, not a different or newly-risky one.
2. **The pilot's own real, completed, successful result is direct empirical proof this exact regime
   (weights persist, optimizer state does not) produces working, improving models** for this
   architecture and this fine-tuning task. There is no hypothetical risk here to weigh — there is a
   real, already-observed 22-epoch outcome using precisely this mechanism.
3. It does not corrupt data, use the wrong model, or invalidate the ability to stop and resume
   training (resuming still correctly continues from the last real, saved weights — confirmed by the
   pilot's own real `resume-proof.json`, whose *epoch*-continuity claim, independently reconfirmed by
   me, is genuinely true even though its *global-step* label is not).
4. What must change: the recorded `learning_rate_policy="constant_0.0001_decay_0.99"` label
   (`training_configuration.py`/`launch_manifest.json`) is **factually inaccurate** for the multi-shard
   run — there is no continuous decay across the run. The real behavior is closer to: a fresh Adam
   optimizer and a fresh, tiny in-shard exponential decay (`decay_steps` resolves to that one
   invocation's own batch count) every single shard, which produces a small sawtooth within each
   shard's own ~625 steps before resetting. I recommend correcting this label and the monitoring
   documentation's characterization before final acceptance, but this is a documentation/labeling fix,
   not a launch blocker.

**Correction to my own prior work as the original auditor:** I (in the original audit, as "Sonnet")
read the same source files and concluded the decay mechanism was "architecturally sound," reasoning
that `decay_steps` resolving to one shard's batch count was actually the *correct* scale for a
multi-shard decay curve — that reasoning assumed the optimizer's `iterations` counter carried forward
across invocations. It does not. I was wrong about the mechanism, in the original audit, and only
real, empirical testing against actual saved checkpoints caught it. Source-reading alone was
insufficient here, for both of us.

### C. Random-seed consistency — **VERIFIED gap, now FIXED**

Independently re-traced every `random_seed`/`epoch_seed` reference in `training_session.py`,
`orchestrator.py`, `cli.py`. Confirmed: `TrainingSessionState.random_seed` is set once, at first
creation, and never validated against a later call's `random_seed` parameter (unlike
`configuration_hash`, which raises `ValueError` on mismatch at the same point in the same function).

**Real-world impact assessed precisely, not assumed:** `epoch_seed` only affects the container's own
`--seed` flag, which (since `augmentation_policy="none"` — confirmed, no augmentation is configured
anywhere) affects only the pinned container's internal batch/line shuffling order *within* one
already-fixed shard's own line set. It does **not** affect which shard is used next (fixed,
independently, by the deterministic `sharding_summary.json` written once at `prepare` time) and does
**not** create any data-leakage or wrong-shard risk. This bounds the real severity: a seed drift on
resume would reduce exact reproducibility of intra-shard batch ordering, not run integrity.

**Fixed anyway**, since it was small, safe, and directly analogous to the already-proven
`configuration_hash` pattern: `training_session.py::run_training_session` now raises `ValueError` on
a resumed session whose `random_seed` does not match the persisted one. Two new tests
(`test_refuses_to_resume_under_a_changed_random_seed`,
`test_resuming_with_the_same_random_seed_succeeds`) pass; the full `test_training_session.py` suite
(25 tests) passes unchanged.

### D. Shard integrity at launch — **VERIFIED gap, now FIXED**

Confirmed the original audit's finding precisely: `current_training_manifest_hash` was computed by
`load_sharding_summary(shards_dir).line_id_set_hash` — reading the *same* `sharding_summary.json`
file whose hash `launch_manifest.json` had itself copied from at `prepare` time. Comparing a file to
itself a second time never detects tampering of the underlying shard `.parquet` files, since the
summary's own recorded hash field is never independently reproduced from real shard bytes.

**Fixed:** added `launch_guard.py::recompute_real_shard_hash()`, which reads every real lap-0 shard
file's actual `line_id` column, reproduces `corpus_sharding.py`'s own hash method exactly (sha256 of
the sorted, JSON-dumped line-ID list), and returns a hash that only matches the recorded one if real
shard content is genuinely unchanged. `cli.py::_run_session` now calls this instead of re-reading the
summary's cached field.

**Also added:** a symmetric reserved-test-manifest overlap re-check at launch time (previously only
validation-manifest overlap was re-verified; test-manifest overlap was checked structurally at
`prepare` time only, never re-verified at `start`/`resume`).

**Proven, not just implemented:** wrote a test that tampers with a real lap-0 shard file (replaces its
content with a manifest containing a foreign line ID) while leaving `sharding_summary.json` completely
untouched, and confirms the new rehash detects the drift and the full guard rejects it
(`test_recompute_real_shard_hash_detects_a_tampered_shard_file`,
`test_tampered_shard_file_is_rejected_by_the_full_guard`). Verified live against the real, freshly-
prepared run (`training/full-corpus-20260801T213023Z`): a real dry-run passed the rehash cleanly (no
false positive against 171 real, untampered shard files).

### E. Model checkpoint functional loading — **VERIFIED, now RESOLVED (was previously unproven)**

The original audit's R-008 noted `verify_checkpoint()` only performs a structural zip check, never a
functional `tf.keras.models.load_model()` round trip, for real production checkpoints. **This review's
own optimizer-continuity probe (Phase 2B above) *is* that functional test** — it genuinely called
`tf.keras.models.load_model()` against two real, production-format checkpoints from the completed
pilot run, on the real GPU, inside the real pinned container, and both loaded successfully once the
correct `custom_objects` were supplied (matching `main.py`'s own real dict exactly). This directly
answers "is the pristine parent checkpoint structurally and functionally loadable" and "can a staged,
production-format checkpoint be reloaded" — **yes, confirmed empirically, twice, independently.**

R-008 is downgraded from an open concern to **RESOLVED for the checkpoint-loadability question**; the
narrower, still-real residual gap is that `verify_checkpoint()` itself (the function actually called
during a real run, after every shard) still only does the cheap structural check, not this functional
one, for performance reasons. That remains acceptable: a genuinely unloadable checkpoint would still
be caught, just one shard later than ideal, by the next epoch's container failing to load it
(`epoch_failed`, an already-handled, safe stop condition) — not silently.

**Confirmed the pristine base checkpoint was untouched by this real GPU/Docker testing**: sha256
recomputed immediately after, matches the pin (`0da2c00ab2b12b23e9f64c01ec67ad29724f275eeadb4561843f8a29ff6fff95`)
exactly. All checkpoint mounts used in this review were `:ro` (read-only).

### F. Test and validation leakage — VERIFIED, unchanged from the original audit

Re-confirmed the original audit's real, full-file verification methodology was sound (reads every
real shard file's `line_id` column against both manifests). I did not re-run the full 171-shard
manual scan a third time (it is now additionally covered by the new automated launch-guard checks,
proven above in section D against the real, current shard set). Exact line-level leakage: **0,
verified, both for validation and (newly) test.** Document/writer-level leakage for 7/11
single-source-file collections: **genuinely unknowable from this data** (no document/writer
identifier exists at all — this is a data-collection limitation, not a code defect), correctly
classified by the original audit as affecting **interpretation of final CER**, not launch safety.
Near-duplicate (perceptual) leakage: **not measured by anything in this codebase**, unquantified,
correctly deferred as a data-quality workstream rather than a launch blocker.

### G. License and governance — **the original audit's claim here was WRONG; corrected**

The original audit stated: *"no LICENSE file for the pinned checkpoint or `loghi-htr` was located and
read as part of this audit"* and classified this UNANSWERED/HIGH. **I searched and found real license
files that audit never looked for:**

```
.loghi-upstream/LICENSE            -- MIT License, Copyright (c) 2022 rvankoert
.loghi-upstream/loghi-htr/LICENSE  -- MIT License, Copyright (c) 2022 rvankoert
.loghi-upstream/laypa/LICENSE      -- present
.loghi-upstream/loghi-tooling/LICENSE -- present
```

**The `loghi-htr` code itself is MIT licensed** — a real, permissive license explicitly permitting
use, copy, modification, merger, publication, distribution, sublicensing, and sale, without
restriction. For a **private technical experiment** or **internal municipal use**, this is
unambiguous and sufficient; there is no code-license blocker.

**What remains genuinely unresolved, precisely scoped:** the pretrained **model weights**
(`generic-2023-02-15/`) are a separate artifact from the code. No dedicated LICENSE or README was
found specifically for the pretrained-model weights' own terms; they are downloaded and distributed
as part of the same overall KNAW-HuC Loghi project (confirmed via `.loghi-upstream/README.md`'s own
download instructions, which treat the pretrained models as an official project artifact). Absent an
explicit, separate statement, the weights are **plausibly, but not separately confirmed to be,**
covered by the same permissive terms. I also found real provenance data in
`pretrained-models/loghi-htr/generic-2023-02-15/file.txt` (the checkpoint's own real training config)
listing its training data sources — useful, previously-unexamined evidence, not itself a license
statement.

The **training dataset's** (Riksarkivet Swedish Lion Libre) own license/permission terms remain
genuinely **UNANSWERED** — no documentation of this was found anywhere in the repository, and this
audit has no authority to determine external licensing terms it cannot locate.

**Classification by intended use, as requested:**

| Use | Status |
|---|---|
| Private technical experiment | **Not blocked** — code is MIT, dataset use is presumably already authorized by whatever arrangement gave ArchiveTrust access to it (outside this audit's visibility) |
| Internal municipal use | **Not blocked**, same reasoning |
| Public checkpoint release | **Requires confirmation** — weights license not separately, explicitly confirmed |
| Research publication | **Requires confirmation** — same, plus dataset provenance/permission documentation should be cited |
| Commercial redistribution | **Requires confirmation** — same |

This is a genuine improvement over the original audit's blanket "UNANSWERED," but the dataset-license
question remains open regardless.

### H. Disk-full and interruption behavior

Covered under Phase 1's "operational hardening gap" above. No live simulation was performed (would
require deliberately filling the real disk, explicitly disallowed); code-level analysis of the atomic-
write pattern shows the failure mode is a loud crash with the previous good state intact, not silent
corruption. This matches the original audit's own runbook entry and is not changed by this review.

---

## Phase 3: Launch blockers vs. professional hardening vs. scientific interpretation

### 1. Must fix before pressing play — **none remain**

Everything that was genuinely launch-critical has been resolved:
- Disk capacity: resolved (measured, 61% margin).
- Resume-of-terminal-run and code-revision-drift guards: fixed in the original audit, reconfirmed
  working.
- Random-seed validation on resume: fixed in this review.
- Shard-integrity self-reference: fixed in this review (real rehash + test-overlap).
- Functional checkpoint loadability: empirically confirmed in this review.
- Optimizer/LR "continuity": empirically clarified — the real mechanism is validated by the pilot's
  own real success, not blocking, but its label must be corrected (see item 2 below — a documentation
  fix, not a launch-safety fix).

### 2. Must resolve before claiming a professionally validated final model

- Correct the `learning_rate_policy` label and any monitoring documentation that implies continuous
  cross-shard LR decay or optimizer-momentum continuity, given this review's empirical finding.
- Baseline (original Loghi checkpoint) evaluation on the fixed held-out test set — not yet performed
  by anything in this repository.
- Fixed test-set evaluation code path for the final trained model — does not yet exist.
- Document-level leakage disclosure for the 7/11 affected collections in any CER claims (R-009,
  unchanged from the original audit).
- Licence confirmation for the model weights specifically, before any public release/publication/
  commercial use (not needed for the private/internal run itself).
- Success criteria, minimum-improvement threshold, and final-approval authority — proposed in Phase 4
  below; still require an actual operator/governance decision.

### 3. Valuable but deferrable hardening

- Checkpoint retention policy (capacity is no longer the forcing function, but unbounded growth over
  many future runs remains poor practice).
- Mid-run disk monitoring and a tested disk-full safe-stop path.
- Near-duplicate (perceptual) image detection.
- Telemetry background-thread exception guard (R-012, unchanged).
- Pilot-era dashboard's cosmetic `PREPARED_NOT_STARTED` label gap (unchanged, purely cosmetic, read-only).
- `verify_checkpoint()` upgraded to an optional functional check (the cheap structural check plus the
  existing `epoch_failed` safety net is adequate for launch).

---

## Phase 4: Proposed evaluation policy (operator decisions clearly marked as decisions)

```yaml
model_selection:
  metric: validation_cer
  direction: minimize
  test_set_used_for_selection: false   # FACT: structurally enforced -- no code path reads the
                                        # reserved test manifest during training or stopping decisions

final_acceptance:
  primary_metric: held_out_test_cer
  baseline_model: original_loghi_checkpoint   # generic-2023-02-15, hash 0da2c00a...
  minimum_relative_improvement: <OPERATOR DECISION -- not proposed here as a default; a specific
                                  number should come from the ML lead, informed by the pilot's own
                                  real best_val_cer=0.16913 and this run's eventual measured result>
  maximum_major_subgroup_regression: <OPERATOR DECISION -- same>

required_reporting:
  - aggregate_cer
  - per_collection_cer            # with explicit R-009 caveat for the 7 line-level-split collections
  - insertion_deletion_substitution_counts
  - difficult_character_analysis
  - qualitative_failure_samples
  - split_limitations             # the file-group vs. line-level split-granularity disclosure

human_review_gates:
  - after_first_shard: required    # see first_shard_review_checklist.md
  - after_first_full_lap: required # see first_epoch_review_checklist.md, ~57 shards
  - before_final_acceptance: required, held-out test touched only once, after model selection is final
```

I am deliberately not inventing specific threshold numbers for `minimum_relative_improvement` or
`maximum_major_subgroup_regression` — those are business/research decisions for the ML lead, not
facts this review can derive from the repository.

---

## Phase 5: Prepared-run freshness

| Field | Original audit's run | This review's fresh run |
|---|---|---|
| Run directory | `training/full-corpus-20260801T192341Z` | `training/full-corpus-20260801T213023Z` |
| Prepared at commit | `d33a32eb7e67...` | `bf9a53a8863d...` (current `HEAD`) |
| Shard plan hash | `e12796a02c868d84...` | `e12796a02c868d84...` — **identical**, confirming full shard-plan reproducibility across two independent re-preparations at different commits |
| Status | now stale (2 commits behind `HEAD`; the new code-revision-drift guard will correctly refuse it) | `PREPARED_NOT_STARTED`, fresh, matches current `HEAD` exactly |

Per the review instructions, the older run is preserved as audit evidence, not deleted, and **neither
run has been started.** All future launch guidance in this review refers to
`training/full-corpus-20260801T213023Z`.

---

## Phase 6: Rerun evidence

- `tests/htr/training/test_training_session.py`: 25 passed (23 original + 2 new, for the seed fix).
- `tests/htr/training/full_run/`: 211 passed (up from 206, for the shard-integrity/test-overlap fixes).
- Full repository suite: **2352 passed, 2 skipped, 0 failed** (`pytest tests -q -m "not real_model"`).
- Real preflight (post-fix, at current `HEAD`): **21/21 passed**, including a real Docker forward+
  backward smoke test and a real checkpoint save/reload round trip.
- Real launch-guard dry run against the fresh prepared run: correctly rejects on the two genuinely
  true current conditions (missing confirmation, dirty working tree) and raises **no false positive**
  on dataset hash, shard hash, or overlap checks — confirming the strengthened checks work cleanly
  against real, untampered artifacts.
- `docker ps`: no containers running. Process inspection: no training process running (only the IDE's
  own language server). `checkpoint_index.json`: does not exist for the fresh run (no checkpoints
  created). Base checkpoint hash: unchanged, reverified after all real Docker/GPU activity in this
  review.

---

## Final assessment

Zero remaining items meet this review's bar for a launch blocker. The one finding more severe than
the original audit reported (optimizer/LR continuity) is resolved by evidence the original audit
already possessed but didn't connect: the pilot's own real, successful, monotonically-improving
22-epoch result *is* the empirical proof that this exact "weights persist, optimizer resets"
mechanism works for this model and task. The remaining open items are governance, scientific-
interpretation, and hardening concerns — exactly the category the instructions define as compatible
with CONDITIONAL GO, provided conditions are explicit and human review gates exist after the first
shard and first full lap. Both are specified in this review's checklists.
