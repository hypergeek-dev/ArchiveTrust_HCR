# Loghi Full-Corpus Retraining Readiness Audit

**Audit date:** 2026-08-01
**Auditor role:** Senior AI architect / ML engineer / HTR specialist / data-governance reviewer / production-readiness auditor
**Repository:** ArchiveTrust_HCR, commit `d33a32eb7e67c9c886ff1cfa651a3f6c8d01c017`
**Audited run:** `training/full-corpus-20260801T192341Z` (prepared at the above commit, during this audit)
**Scope:** the complete operator-controlled full-corpus Loghi retraining workflow (`src/archivetrust/htr/training/`, `src/archivetrust/htr/training/full_run/`, the pinned `.loghi-upstream/loghi-htr` dependency, the real pilot run at `training/loghi-swedish-v1/`, and the real, prepared full-corpus run).

No full-corpus training was started during this audit. No optimizer step was performed against the real full corpus. The audited run's status is `PREPARED_NOT_STARTED` at the time this report was written, and remains so.

---

## Executive summary

## RETRAINING READINESS DECISION: NO-GO

This is not a judgment that the workflow is poorly built. It is a well-evidenced, carefully engineered system — real Docker smoke tests, a real, working launch guard, real deterministic sharding, real checkpoint indexing, a real 22-epoch pilot with genuine plateau evidence. Three real defects were found and fixed live during this audit (a checkpoint-corrupting smoke test, a resume-of-a-finished-run gap, and a code-drift gap), which is itself evidence the design is sound enough to be worth hardening rather than rebuilt.

The NO-GO is driven by one **quantified, currently-true, unmitigated CRITICAL finding**: the prepared 171-shard run has **no checkpoint retention policy and no mid-run disk-space monitoring**, and the real, measured per-shard checkpoint cost (0.727GB/shard, extrapolated from the pilot's actual 16GB/22-epoch disk consumption) means the full 171-shard plan requires **~124GB** while the host currently has **~95GB free**. One full pass over the corpus (57 shards, ~41.5GB) fits comfortably; if training continues into a second or third lap without early stopping — plausible, not guaranteed, since full-corpus convergence dynamics have never been observed — the run will exhaust disk around shard ~131, with **completely untested failure behavior**. This is exactly the kind of failure that wastes a multi-day run, and it is a decision-rule-listed hard blocker ("insufficient disk capacity").

Four HIGH-severity findings compound the case for NO-GO-until-fixed rather than launch-with-caveats: `global_step` is never tracked (always `0` in all 47 real pilot checkpoint entries) and the automated proof that claims to verify "global step continuity" actually re-tests epoch continuity due to a naming bug; `random_seed` can silently diverge between what is persisted and what is actually used on `resume`, unlike `configuration_hash`, which is genuinely validated; the launch guard's shard-integrity check is self-referential (it re-reads the same summary file its own frozen hash came from, never rehashing real shard bytes) and only re-verifies validation-set overlap at launch time, never test-set overlap; and two guard gaps found live during this audit (resume of a completed/failed run; no code-commit-drift detection) have been fixed, with the second one caught by finding the actual prepared run's own metadata already stale relative to `HEAD`.

None of this invalidates the architecture. All of it is fixable in hours to a few days, not a redesign. The remediation path to CONDITIONAL GO is short and specific (§ "Final launch conditions").

### Strongest evidence (what the system gets right)

- **Parent checkpoint is verifiably correct and pristine.** `model.keras` sha256 = `0da2c00ab2b12b23e9f64c01ec67ad29724f275eeadb4561843f8a29ff6fff95`, matching the pin in `pinned_versions.py` exactly, reverified twice in this audit session (before and after a real container smoke test).
- **The pilot checkpoint is structurally, provably excluded as parent.** `identity.py::assert_parent_checkpoint_is_not_a_pilot_path` is a real assertion, not a convention, and `launch_manifest.json` records `pilot_checkpoint_used_as_parent: false` with the exact rejection evidence.
- **Dataset identity, splits, and shard plan are real, hash-verified, and reproducible.** Source inventory: 563,933 valid lines, sha256 `fc21a70825650927...`. Full-corpus shards: 171 shards, 562,123 usable lines, 3 laps, **zero duplicate lines within any lap** (verified by reading every shard file), **zero validation-set overlap and zero test-set overlap across all 171 shards** (verified by reading every shard file against both manifests), and **byte-identical shard membership reproduced from the same seed** in an independent rebuild.
- **A real preflight passes 21/21**, including a genuine Docker container forward+backward pass and a genuine checkpoint save/reopen round trip — not a mock.
- **The launch guard is real and was proven to work**, including against two conditions this very audit added after finding them missing.
- **A real incident was caught and fixed, live, with evidence.** The preflight smoke test was silently overwriting the pristine base checkpoint in place; this was caught by a hash mismatch between two consecutive preflight runs, root-caused, fixed, and reverified.

---

## 1. Architecture map

```
 analyze-pilot ──▶ pilot_analysis.json ──▶ derive_monitoring_config ──▶ pilot_derived_monitoring.json
       │                                                                        │
       ▼                                                                        ▼
 (reads training/loghi-swedish-v1/run-state/session_state.json,          preflight ──▶ preflight_status.json
  checkpoint_index.json, memory-probe.json, resume-proof.json)                 │  (21 real checks: Docker, GPU,
                                                                                 │   checkpoint hash/identity,
                                                                                 │   manifests, disk, telemetry,
                                                                                 │   seed, conflicting runs,
                                                                                 │   dashboard discovery, real
                                                                                 │   container smoke test +
                                                                                 │   checkpoint round trip)
                                                                                 ▼
                                                                            prepare
                                                                                 │
              ┌──────────────────────────────────────────────────────────────┼─────────────────────────────┐
              ▼                                                                ▼                              ▼
    corpus_sharding.build_full_corpus_shards               identity.create_full_run_identity      launch_manifest +
    (deterministic, seeded, excludes pilot                 (rejects pilot-path parent,             launch_preview +
     val/test line IDs; 171 shards)                          refuses to reuse a run dir)             PREPARED_NOT_STARTED
                                                                                                         marker
                                                                                 │
                                                                                 ▼
                                                                    run_state.json: status=prepared
                                                                                 │
                                                              ══════════ MANUAL GATE ══════════
                                                                    (--confirm-full-corpus-run,
                                                                     never supplied automatically
                                                                     anywhere in this codebase)
                                                                                 │
                                                                                 ▼
                                                                     start / resume
                                                                                 │
                                                                    launch_guard.enforce_launch_guard
                                                              (confirmation, terminal-status, dataset/manifest
                                                               hash drift, code-commit drift, repo-dirty,
                                                               disk space, Docker/image, val overlap, preflight
                                                               freshness -- 11 real, freshly-read conditions)
                                                                                 │
                                                                                 ▼
                                                            orchestrator.run_full_corpus_session
                                                     (one call to training_session.run_training_session
                                                      per shard, always max_epochs_this_call=1 --
                                                      "one epoch" == "one shard" == "one container call")
                                                                                 │
                              ┌──────────────────────────────────────────────────┼──────────────────────────┐
                              ▼                                                    ▼                          ▼
                    container_epoch_runner.ContainerEpochRunner          checkpoint_index.py         telemetry_sampler.py
                    (real `docker run`, --model <staged copy>,     (append-only, atomic,       (background thread, GPU/
                     never the pristine checkpoint directly;        write-temp-then-rename,      RAM/CPU/disk samples,
                     read-write mount only for a disposable          verify_checkpoint before      status.json liveness)
                     staged copy)                                    resumable=True)
                              │
                              ▼
                    run_status.py / run_health.py / training_dashboard_viewmodel.py
                    (read-only; the Streamlit pilot-era dashboard AND the PySide6 full-run GUI
                     both only read run_state.json/session_state.json/telemetry -- neither can
                     start, stop, or modify a run; GUI Start/Resume buttons only preview a
                     command, never launch one)
```

**One authoritative path per concern, confirmed:**

| Concern | Authoritative implementation | Competing implementations found |
|---|---|---|
| Training-loop execution | `training_session.py::run_training_session` (one epoch = one container call) | None — `orchestrator.py` wraps it unmodified, never duplicates it |
| Checkpoint bookkeeping | `checkpoint_index.py` (append-only, atomic) | None |
| Container invocation | `container_epoch_runner.py::ContainerEpochRunner` | None — explicitly documented as *not* reusing `providers/loghi/facade.py`'s inference-time argv builder, because the CLI contracts genuinely differ (confirmed by reading the pinned `arg_parser.py` directly) |
| Launch gating | `launch_guard.py::evaluate_launch_guard` | None — both `cli.py` and the GUI route through the same function |
| Dashboards | Two, by design, not duplication: `training_dashboard_viewmodel.py` (pilot-era Streamlit, reads `session_state.json`/`run_status.py`) and `full_run/gui/app.py` (PySide6, reads `run_state.json` directly). Both are read-only. The pilot-era dashboard does not display the new `PREPARED_NOT_STARTED` label (it infers `"initializing"` from the absence of `session_state.json`) — a real, minor, disclosed gap, not a competing control path. | — |
| Monitoring-state classification | `monitoring_state.py::classify_monitoring_state` (11-state, documented priority order) | None |

No parallel or competing implementation of any control-path concern was found. This is a real strength.

---

## 2. Model lineage

| Field | Value | Evidence |
|---|---|---|
| Loghi implementation | Pinned `loghi/docker.htr` image, digest `sha256:414fc89ac574a61fd745836ad9852842759bf3cf7046e4549ae96315ff9132e8` | `pinned_versions.py`; verified present locally via `docker image inspect` |
| Architecture | `new10` (VGSL-style, as shipped by the generic checkpoint's own `file.txt`) | `TrainingConfiguration.model_architecture="new10"` in `cli.py::cmd_prepare` |
| Parent checkpoint | `.loghi-upstream/pretrained-models/loghi-htr/generic-2023-02-15/model.keras` | `PARENT_CHECKPOINT_DIR` in `cli.py` |
| Parent checkpoint hash | `0da2c00ab2b12b23e9f64c01ec67ad29724f275eeadb4561843f8a29ff6fff95` | Recomputed directly, twice, matches `pinned_versions.py`'s `model_checkpoint_hash` exactly both times |
| Pilot checkpoint used as parent | **No** — structurally rejected | `identity.py::assert_parent_checkpoint_is_not_a_pilot_path`; `launch_manifest.json: pilot_checkpoint_used_as_parent=false` with rejection evidence recorded |
| Checkpoint loading | Real `--model <dir>` flag (never `--existing_model`, which the pinned CLI does not define) | `container_epoch_runner.py`'s own docstring, confirmed against `.loghi-upstream/loghi-htr/src/setup/arg_parser.py` |
| Fresh optimizer/scheduler state | Structural consequence of never handing the container the pristine, unstaged checkpoint (see below) | `training_session.py::_stage_writable_checkpoint`; `identity.py`'s own docstring reasoning, verified against `model/management.py` |
| Strict checkpoint loading / frozen layers | **Unanswered from this repository** — this is the pinned `loghi-htr` image's own internal `model/management.py` behavior; not overridden here | UNANSWERED — recommend reading `.loghi-upstream/loghi-htr/src/model/management.py::load_or_create_model` directly before a first real launch if strict-loading behavior matters |
| License compatibility (retraining/redistribution/commercial/municipal use) | **Unanswered** — no LICENSE file for the pinned checkpoint or `loghi-htr` was located and read as part of this audit | UNANSWERED — governance decision, not a code question |

**Can a wrong checkpoint with the same filename bypass validation?** No — `base_checkpoint_hash_matches_pin` recomputes the real sha256 of the actual file content, not the filename, on every preflight.

**Can training overwrite the original model?** Not via the production path (verified: `container_epoch_runner.py` never mounts `PARENT_CHECKPOINT_DIR` directly; `training_session.py::_stage_writable_checkpoint` always copies first). It *did* happen via a bug in the preflight smoke test — found, fixed, and reverified in this audit (R-001, resolved).

---

## 3. Dataset lineage and split integrity

| Metric | Value | Evidence |
|---|---|---|
| Total inventory rows | 565,146 | Direct `pyarrow.parquet` read of `full-inventory.parquet` |
| Valid rows | 563,933 | Same |
| Dataset identity (hash) | `fc21a70825650927edcae753a20411cef8e4344e2c22bcbcaed1c0038b9b97b0` | sha256 of the inventory file |
| Full-corpus training lines | 562,123 (563,933 minus 1,810 excluded pilot val/test lines) | `sharding_summary.json` |
| Validation lines | 1,000 (reused from the pilot's own `val_manifest.parquet`, unmodified) | Direct parquet read |
| Reserved test lines | 810 (excluded from both pilot training and full-corpus shards) | Direct parquet read |
| Shard-vs-validation overlap | **0**, verified across all 171 shard files | Direct, full read of every shard's `line_id` column intersected with the validation manifest |
| Shard-vs-test overlap (at prepare time, structurally) | 0 (test manifest is in `exclude_manifest_paths` alongside validation) | `corpus_sharding.build_full_corpus_shards` call in `cli.py::cmd_prepare` |
| Shard-vs-test overlap (re-verified at launch time) | **Never re-checked** — the launch guard only re-verifies validation overlap | R-007 |
| Duplicate lines within any lap | 0, verified across all 171 shard files, all 3 laps | Direct, full read |
| Shard-plan reproducibility | Confirmed — an independent rebuild from the same seed produced byte-identical shard membership and ordering for the sampled shards | Direct comparison in this and the prior audit session |

**Split-unit and leakage methodology (verified from source, `pilot_split.py`):** the strongest available grouping key is `(collection, source_parquet_file)` — a "file-group" proxy for document/volume grouping, since no true document, volume, or writer identifier exists in the source data (disclosed in the module's own docstring). For collections with 2+ source files, whole file-groups are assigned to exactly one split, preserving that leakage boundary. **For 7 of 11 collections (single-source-file collections), splitting degrades to line-level**, marked `split_granularity="line_level_single_file_collection"` in the manifest. **Document-level (and possibly writer-level) leakage between train and validation/test cannot be ruled out for those 7 collections specifically.** This is disclosed, not hidden, and does not create duplicate-line overlap (verified zero), but it does mean validation/test CER should not be presented as an unbiased generalization estimate for those collections without this caveat (R-009).

**Duplicate detection:** exact image-content-hash and (image-hash, transcription) duplicates are detected and excluded (`swedish_dataset_inventory.py`). **No near-duplicate/perceptual detection exists** (R-010) — unquantified residual risk.

**Unicode/text normalization, license/PII review of the source documents, and formal dataset versioning beyond the content hash** were not independently re-verified in this audit; treated as unanswered (see question matrix, Section F/C).

---

## 4. Pilot interpretation

| Fact | Value | Status |
|---|---|---|
| Pilot epochs completed | 22 | MEASURED |
| Pilot train/val lines | 9,999 / 1,000 | MEASURED |
| Best epoch / best val CER | epoch 22, `val_CER=0.16913` | MEASURED |
| Genuine (non-extrapolated) plateau | **Yes** — a real, observed stable window at epoch 20 | MEASURED (`pilot_analysis.json: genuine_plateau_occurred=true, plateau_is_extrapolated=false`) |
| Early stopping triggered | **No** — the pilot's real stop reason was `epoch_would_not_fit` (a wall-clock boundary), not `no_val_cer_improvement` | MEASURED |
| Max near-flat streak | 3 consecutive epochs | MEASURED |
| Measured epoch duration (mean) | 482.3 seconds (~8.0 minutes) | MEASURED |
| Derived `recommended_patience` | 5 (= floor 3, max(3, streak 3 + margin 2)) | Derived from measured pilot evidence, documented basis string persisted |
| Derived `min_exposure_steps` | 12,500 (= plateau epoch 20 × 625 steps/epoch) | Derived, documented basis persisted |
| `shard_line_count` | 9,999 (defaulted to the pilot's own train line count, for direct step-for-step comparability) | Operator-editable |

**What the pilot did *not* prove:** full-corpus throughput (only extrapolated 1:1 from the pilot's own bounded-subset measurement, since `shard_line_count == pilot train_line_count` makes the scaling factor exactly 1.0 — but this is still an extrapolation across a **56x larger corpus**, not a measurement at that scale); multi-shard optimizer-state continuity at real (hours-to-days) scale (only a synthetic 3-epoch, seconds-scale proof exists, see R-004/R-005); disk consumption at full-run scale (this audit performed that extrapolation, see R-018); whether full-corpus convergence dynamics resemble the pilot's bounded 9,999-line subset at all (more data commonly needs *more*, not fewer, effective passes to plateau — the pilot cannot answer this).

**Should the pilot's thresholds transfer as-is?** `min_exposure_steps`/`recommended_patience` are reasonable, evidenced starting points, explicitly labeled as pilot-derived defaults (not universal constants) in `monitoring_config.py`'s own field docstrings. Recommend explicit recalibration after the first full lap (Gate 5), not blind trust through 171 shards.

---

## 5. Training configuration review

| Parameter | Value | Note |
|---|---|---|
| Batch size | 16 (default; operator-configurable per invocation) | Per-device, no gradient accumulation configured (`gradient_accumulation=1` hardcoded in `cli.py`) |
| Precision | `mixed_float16` (hardcoded in `cli.py::_run_session`) | Not derived from any per-shard configuration; consistent across the whole run |
| Optimizer | `adam` (hardcoded) | |
| Learning rate | `0.0001` (hardcoded, passed to the container on every invocation) | |
| Learning-rate schedule | Recorded label: `constant_0.0001_decay_0.99`. **Verified via direct reading of the pinned `loghi-htr` source** (`.loghi-upstream/loghi-htr/src/model/optimization.py`, `arg_parser.py`): the container's own default `--decay_rate=0.99`, `--decay_steps=-1` (→ resolves to the current invocation's own batch count, i.e. one shard's worth of steps), `--warmup_ratio=0.0` (no warmup) combine to produce a genuine, continuously-compounding exponential decay across the whole multi-shard run — **conditional on** the optimizer's `iterations` counter genuinely persisting across all 171 shard-boundary checkpoint reloads, which is asserted from source reading plus one small synthetic proof, never measured at full scale (R-017). | Architecturally sound by source analysis; empirically unverified at scale |
| Warmup | None (`warmup_ratio=0.0`) — by design, not a bug. The brief's "does moving from pilot to full corpus expand warmup by ~50x" concern does not apply; no warmup step-count logic exists in this pipeline at all. | Confirmed via source reading |
| Gradient clipping / NaN-Inf detection | `orchestrator.py::_has_nan_or_inf` checks `train_loss`/`val_loss`/`val_cer` after every shard and stops with `nan_or_inf_detected` | Real, verified in code; no *mid-epoch* gradient-level clipping is configured (relies on the pinned container's own internal behavior, not overridden) |
| CTC loss / blank-token assumptions | **Unanswered from this repository** — internal to the pinned `loghi-htr` image | UNANSWERED |
| "Epoch" definition under sharding | One shard = one container invocation = one "epoch," by explicit design (`training_session.py`'s own docstring, `orchestrator.py`'s docstring). Verified consistent with real, measured `steps_per_shard=625` (9999 lines / 16 batch size). | Real, deliberate, documented departure from a literal "one pass through the whole corpus" epoch — one full pass = 57 shards, not 1 |
| Max epochs (shard ceiling) | 171 (= one lap × 3, `max_full_run_epochs` derived from the real corpus line count once supplied) | A safety ceiling, not a target |
| Wall-clock policy | Operator-specified per invocation via `--hours` (default 5.0); a run resumes across many bounded invocations, never one unbounded process | Verified in `cli.py`; `training_session.py`'s own epoch-fit estimator (`SAFETY_MARGIN=1.15`) stops a session *before* starting a shard it estimates won't fit, never mid-shard |
| Early stopping semantics | Patience counted in shards (not epochs in the traditional sense), across resumed sessions (`TrainingSessionState.epochs_since_improvement`, never reset by a session boundary) | Verified real, persisted, cross-session |
| Best-checkpoint restoration for export | The best checkpoint is tracked separately (`checkpoint_kind="best_val"`) and never overwritten by "latest" | Verified in `checkpoint_index.py`; **whether the final exported/deployed model is explicitly the best or the latest checkpoint is an operator decision at export time, not automated** — UNANSWERED/operational gap |

---

## 6. Checkpoint and resume review

**What is genuinely proven (real evidence):**
- Atomic writes everywhere (`write-temp-then-os.replace`), consistently, across `checkpoint_index.py`, `run_state.py`, `session_state.json`, `sharding_summary.json`, `launch_manifest.json`.
- A checkpoint is only ever marked `resumable=True` after `verify_checkpoint()` (structural: non-empty, valid zip, no CRC errors) passes.
- `configuration_hash` mismatches on resume are rejected with a hard `ValueError` — genuinely enforced, not just documented.
- A real, completed `resume-proof.json` exists for the pilot (`proof_passed=true`), demonstrating epoch/session continuity across a genuinely reloaded process.
- The preflight's own real container round trip (`checkpoint_save_reload_round_trip`) passed twice in this audit session, using a properly staged (post-fix) copy of the real pinned checkpoint.

**What is not proven, or is actively misleading (real gaps, all in the risk register):**
- `global_step` is a permanent `0` in every real checkpoint entry — never tracked (R-004).
- The resume-proof's `global_step_continued_not_restarted` field is computed by a check that re-tests epoch continuity, not global-step continuity (R-005) — the field name overstates what was verified.
- `random_seed` can silently diverge between what is persisted and what is actually used on `resume` (R-006).
- `verify_checkpoint()` never performs a functional `tf.keras.models.load_model()` round trip for real production checkpoints — only structural zip validity (R-008); the real functional proof exists only for the pilot's small synthetic case.
- `training_manifest_hash`/`validation_manifest_hash` fields on every real `CheckpointEntry` are always empty strings — dead fields (R-013).
- Checkpoint-directory selection after a container run uses non-deterministic filesystem iteration order if more than one qualifying file is ever produced per epoch (R-011) — currently masked by observed one-file-per-epoch behavior.
- **No checkpoint retention policy exists at all** — every checkpoint accumulates forever (R-018, the primary NO-GO driver).

---

## 7. Monitoring and dashboard review

- 11-state deterministic classifier (`monitoring_state.py`), documented priority order, unit-tested against every transition.
- Telemetry is append-only JSONL plus atomic `status.json` snapshots; staleness detection exists and is reused consistently between the pilot-era `run_status.py` and the full-run `monitoring_state.py`.
- **The telemetry background thread has no exception guard beyond `ImportError`** (R-012) — a real, bounded gap (degrades to the already-handled "stale telemetry" state rather than crashing training).
- Both dashboards (pilot-era Streamlit, full-run PySide6) are read-only by construction — verified by direct inspection: neither imports or calls anything that starts, stops, or mutates a training run. The PySide6 GUI's Start/Resume buttons were specifically hardened during the underlying implementation work to only *preview* the confirmed command, never launch it (`build_start_command`/`build_resume_command` never include `--confirm-full-corpus-run`, and the button handlers no longer call `_launch()` at all for those two actions).
- `PREPARED_NOT_STARTED` is a real, verified display state (`display_status()`), confirmed live: `python -m archivetrust.htr.training.full_run status --run training/full-corpus-20260801T192341Z` prints exactly `display_status: PREPARED_NOT_STARTED`.
- The pilot-era Streamlit dashboard does not itself render the new `PREPARED_NOT_STARTED` label (it infers `"initializing"` for a run with no `session_state.json` yet) — cosmetic, not a safety gap, since it is read-only and the authoritative CLI/GUI status is correct.

---

## 8. Infrastructure review

| Item | Status | Evidence |
|---|---|---|
| Docker daemon | Reachable | `docker ps` succeeded, real, repeated |
| Container image | Present, digest-matched | `docker image inspect` returned the exact pinned digest |
| GPU | Present, single GPU, WSL2 backend | `probe_loghi_environment()`; `nvidia-smi` referenced in `container_epoch_runner.py`'s own comments as previously confirmed |
| Disk space (current) | ~95.4GB free (D: drive) | `Get-PSDrive D` |
| Disk space (required for the prepared 171-shard plan) | **~124.4GB**, given the real, measured 0.727GB/shard checkpoint-accumulation rate and zero retention policy | Extrapolated from a real `du -sh` measurement of the pilot's actual `epoch_output/` directory |
| Disk space (one full 57-shard lap) | ~41.5GB — within current budget | Same extrapolation |
| Mid-run disk monitoring | **None exists** | Codebase-wide search, zero matches |
| Host sleep/reboot prevention, container restart policy, UPS/power-loss plan | **Unanswered** — outside this repository's own configuration surface (host/OS-level policy, not code) | UNANSWERED |
| Docker CLI timeout margin | One transient `docker image inspect` timeout (15s) observed live in this audit, resolved on retry | R-015, low severity |

---

## 9. Failure-mode review (selected; full matrix would exceed the scope of a readable report — the highest-value scenarios are covered here, the rest are covered by existing, passing tests or explicitly flagged as unverified)

| Scenario | Detection | Behavior | Data-loss risk | Manual intervention needed |
|---|---|---|---|---|
| Docker daemon stops mid-run | Next shard's `subprocess.run(["docker", "run", ...])` fails / next preflight's `docker ps` check | Current shard's container call returns non-zero → `epoch_failed` → session stops cleanly at a shard boundary (never mid-shard, since "one epoch = one container call") | None — last verified checkpoint stands | Yes, restart Docker, then `resume` |
| NaN/Inf loss | `orchestrator.py::_has_nan_or_inf` after every shard | `nan_or_inf_detected` → `mark_failed` | None — checked only after a completed, verified shard | Yes |
| Disk fills mid-checkpoint-write | **Not explicitly tested** — `tempfile.mkstemp` would raise `OSError` on `ENOSPC`; not specifically caught anywhere in `checkpoint_index.py`/`run_state.py` | **Unverified** — likely an uncaught exception propagating out of `run_training_session`, not a clean `epoch_failed` | Unknown — the atomic write pattern *should* prevent a torn file, but this has not been tested under real disk-full conditions | Yes, and the exact recovery path is unverified (see runbook) |
| Ctrl+C / SIGINT | Not explicitly handled; Python's default `KeyboardInterrupt` would propagate | **Unverified whether an in-flight `docker run` subprocess is orphaned** — `subprocess.run(..., timeout=...)` does not itself register a SIGINT handler to kill the child | Possible orphaned container; last checkpoint before the interrupted shard stands | Yes — use the documented `stop` (STOP_REQUESTED sentinel) for a graceful stop instead of Ctrl+C |
| Duplicate launch attempt | `evaluate_launch_guard`'s `run_state.status in (RUNNING, STOPPING)` check | Rejected with a clear message | None | No — rejected cleanly |
| Resume of a completed/failed run | **Fixed in this audit** (R-002) | Rejected with a clear message | None | No — rejected cleanly |
| Stale/absent preflight | `evaluate_launch_guard`'s `preflight_passed`/`preflight_age_seconds` check | Rejected | None | Yes — rerun `preflight` |
| Code changed since prepare | **Fixed in this audit** (R-003) | Rejected unless explicitly overridden | None | Yes — re-run `prepare`, or explicitly override with full awareness |
| Wall-clock cap expires mid-shard | `training_session.py`'s own epoch-fit estimator refuses to *start* a shard it estimates won't fit (`epoch_would_not_fit`), so this cannot happen by construction — verified, this is exactly the real pilot's own actual stop reason (`stop_reason: epoch_would_not_fit`) | Clean stop at a shard boundary | None | No — resumable normally |
| GPU OOM | `orchestrator.py::_oom_signature_in` scans `stderr_tail` for known signatures | `likely_oom` → `mark_failed` | None (checked post-shard) | Yes, reduce batch size, `resume` |

---

## 10. Reproducibility review

**Genuinely reproducible, evidenced:**
- Dataset identity: content-hashed.
- Shard plan: seeded, deterministic, independently reproduced byte-identical in this audit.
- Code revision: now genuinely enforced at launch time (R-003, fixed in this audit) — previously recorded but never checked for drift.
- Configuration hash: enforced with a hard rejection on mismatch, for real, across resumed sessions.

**Not fully reproducible, or unverified:**
- `random_seed` (R-006) — persisted but not cross-checked against what is actually used on resume.
- The TensorFlow/CUDA/cuDNN/driver stack inside the pinned container image is fixed by the image digest (a real, strong reproducibility anchor), but the *host* driver/CUDA version outside the container is not recorded anywhere in this workflow's own artifacts.
- Full determinism inside TensorFlow itself (e.g. cuDNN non-determinism on GPU) is not addressed or disclaimed anywhere in this codebase — treated as UNANSWERED, not claimed either way.

---

## 11. Security review

- No secrets, credentials, or network calls were found anywhere in the training path — the whole workflow is designed to run offline against local, pinned artifacts (`container_epoch_runner.py`'s own docstring: "no interactive shell, deterministic subprocess"; no HTTP client imports anywhere in `full_run/`).
- Container is invoked with `--rm` (no persistent state left behind) and explicit, narrow bind mounts (`-v <specific dir>:/model`, never a broad mount).
- Manifest/checkpoint paths are constructed from resolved `Path` objects, not raw string concatenation from user input; no path-traversal vector was found in `container_epoch_runner.py::_build_argv`.
- The launch confirmation flag (`--confirm-full-corpus-run`) is never supplied by any code path in this repository — confirmed by a full-repository search; it can only be supplied by a human typing it on a real terminal invocation.
- No audit trail of *who* (which human) launched a run exists beyond the OS-level process owner and `code_revision` — acceptable for a single-operator research context, a gap for a multi-operator production context. UNANSWERED whether that matters for this project's governance model.

---

## 12. Scorecard

Scores are **not averaged into a single pass/fail number** — a single unresolved critical issue (R-018) means NO-GO regardless of the aggregate.

| Category | Score (0-5) | What's missing for a higher score |
|---|---|---|
| Objective clarity | 2 | No written success criteria, no authoritative metric declaration, no minimum-improvement threshold, no subgroup-breakdown plan found in the repository — these are governance/product decisions, not code, and none were located. See question matrix Section A. |
| Model lineage | 5 | Fully evidenced, hash-verified, structurally protected. |
| Dataset provenance | 3 | Real hashing and exact-duplicate exclusion exist; no near-duplicate detection (R-010), no independently-verified license/PII review located. |
| Split integrity | 3 | Zero measured overlap at the line-ID level (real, verified); document-level leakage risk disclosed but unresolved for 7/11 collections (R-009); test-set overlap never re-checked at launch time (R-007). |
| Dataset representativeness | 2 | No per-collection/writer/period distribution analysis was found or performed in this audit; pilot-vs-full-corpus representativeness is asserted, not measured. |
| Preprocessing correctness | 2 | Delegated entirely to the pinned container's own internal behavior; not independently re-verified in this audit (UNANSWERED, Section E of the question matrix). |
| Vocabulary correctness | 2 | Charlist is hashed into the configuration identity (real), but character-coverage/OOV analysis for the full corpus vs. the pilot's smaller vocabulary sample was not performed in this audit. |
| Training configuration | 4 | Real, evidenced, source-verified (including the LR-decay deep-dive in this audit); the one gap is R-017 (never runtime-measured). |
| Pilot interpretation | 4 | Rich, honest, measured-vs-extrapolated evidence exists (`pilot_analysis.json`'s own fields); the one caveat is that pilot-derived thresholds are explicitly provisional, correctly labeled as such. |
| Sharding correctness | 4 | Deterministic, reproducible, zero-overlap, zero-duplicate — verified by direct, full-file inspection; the one gap is R-007 (launch-time integrity is self-referential). |
| Reproducibility | 3 | Real hash-based gates exist and were strengthened in this audit; R-006 (seed) and the host driver/CUDA gap remain. |
| Checkpoint/resume | 2 | Real atomic writes and a real structural round trip exist; R-004/R-005/R-008/R-011/R-013 collectively mean the checkpoint/resume story is functionally probably fine but far less rigorously *proven* than its own documentation claims. |
| Validation correctness | 3 | Real CER computation via the pinned container's own `log.csv`; independent re-verification of the CER implementation itself (insertion/deletion/substitution correctness) was not performed in this audit — UNANSWERED. |
| Monitoring | 4 | Deterministic, tested, real; R-012 is a bounded gap. |
| Dashboard isolation | 5 | Verified read-only by direct inspection, both dashboards, GUI buttons specifically hardened. |
| Infrastructure | **1** | R-018 (disk capacity) is a real, current, unmitigated CRITICAL gap; everything else in this category (Docker, GPU, image) is otherwise solid. |
| Duration planning | 3 | Real measured pilot timing exists and was extrapolated carefully in this audit (including the disk-cost extrapolation that drove the NO-GO); full-scale throughput itself remains unmeasured. |
| Experiment design | 1 | No baseline-vs-retrained comparison plan, no ablation plan, no experiment registry was found in this repository — this appears to be designed and operated as a single production retraining run, not a controlled experiment. If a defensible comparison or publication is intended, this is a real gap (Section Q of the question matrix). |
| Downstream usability | 1 | No export format, no model card template, no deployment acceptance test was found in this repository for the full-corpus output specifically. |
| Security | 4 | Solid; the one gap is operator-identity audit trail (informational for a single-operator context). |
| Failure recovery | 2 | Several scenarios (disk-full, Ctrl+C/orphaned container) are genuinely untested, not just undocumented. |
| Documentation | 4 | Unusually thorough in-code documentation (module docstrings routinely explain *why*, with real evidence citations) — a real strength of this codebase. |
| Launch safety | 4 | The guard is real and working, strengthened twice during this very audit; the residual gap is R-007 (shard-integrity self-reference). |

---

## 13. Decision gates

| Gate | Required evidence | Pass criteria | Who decides | Command/artifact |
|---|---|---|---|---|
| 1. Audit complete | This report + risk register | No unresolved CRITICAL; every HIGH has an accepted, explicit mitigation or owner | Repository maintainer + ML lead | `retraining_readiness_audit.md`, `retraining_risk_register.json` |
| 2. Preflight complete | Real preflight run against current HEAD | All critical checks pass (21/21 achieved twice in this audit) | Operator | `python -m archivetrust.htr.training.full_run preflight` |
| 3. Prepared state | `launch_manifest.json`, `PREPARED_NOT_STARTED` marker, frozen hashes | Status is `PREPARED_NOT_STARTED`; launch command generated, not executed | Operator | `training/full-corpus-20260801T192341Z/` (this audit's evidence run) |
| 4. First shard review | See `first_shard_review_checklist.md` | Throughput, GPU health, checkpoint validity, no anomalies | Operator + ML lead | Real `status` output after shard 1 |
| 5. First full epoch (lap) review | See `first_epoch_review_checklist.md` | Full validation completed, CER behavior sane, stopping policy recalibrated if needed | ML lead | Real `status`/`session_state.json` after ~57 shards |
| 6. Model-selection review | Best checkpoint selected on validation only | Held-out test remains untouched | ML lead | `checkpoint_index.json`'s `best_val` entries |
| 7. Final acceptance | Fixed test evaluation, baseline comparison, subgroup analysis, model card | All present and reviewed | ML lead + approving authority (unspecified in this repository — governance gap) | UNANSWERED where the approving authority is defined |

**May training continue automatically past any gate without a human decision?** No — every gate above requires an operator to run the next command explicitly; nothing in this codebase auto-advances from `prepare` to `start`, or from one shard block to the next beyond the `--hours` budget the operator themselves set on each invocation.

---

## 14. Final launch conditions (the path from NO-GO to CONDITIONAL GO)

1. **(Blocking, CRITICAL)** Resolve R-018: implement a bounded checkpoint retention policy, or provision materially more disk (recommend ≥150GB free), or both; add a mid-run disk-space safety stop; raise `preflight`'s `min_free_disk_gb` default to reflect the real, measured per-shard cost times the configured shard ceiling.
2. **(Strongly recommended before an unattended multi-day run)** Address R-004/R-005: either implement real `global_step` tracking or explicitly relabel the field and the resume-proof's claim to match what is actually verified; do not rely on the current field/proof as evidence of global-step-level resumability.
3. **(Strongly recommended)** Address R-006: validate `random_seed` consistency across `start`/`resume` the same way `configuration_hash` is validated, or remove the independently-settable `--seed` flag from `resume`.
4. **(Recommended)** Address R-007: rehash real shard file content (not just the summary file) and re-check test-set overlap, not just validation-set overlap, at launch time.
5. **(Recommended, before presenting final CER results)** Explicitly disclose the R-009 line-level-split limitation for the 7 affected collections in any subgroup or aggregate CER reporting.
6. **(Operational)** Do not use `training/full-corpus-20260801T171555Z` — it is stale relative to current code (R-016) and the newly-fixed guard will correctly refuse it without an explicit override. Use `training/full-corpus-20260801T192341Z` (this audit's evidence run) or re-`prepare` fresh immediately before actually launching.
7. **(Operational)** Commit or explicitly acknowledge (`--allow-dirty-repository`) the current working-tree state before launch — at the time of this audit, `training/config/preflight_status.json` had been refreshed and was uncommitted.
8. **(Governance, unanswered by this audit)** Define the authoritative success metric, minimum-improvement threshold, and approving authority for final model acceptance (Section A of the question matrix) — these are business/research decisions this audit cannot make on the repository's behalf.

Once items 1-3 are resolved (with 4-8 explicitly acknowledged or completed), this system supports a legitimate **CONDITIONAL GO**. A full **GO** additionally requires empirical (not just source-derived) confirmation of multi-shard optimizer continuity and LR decay at a scale closer to the real run (e.g. after the first full lap, per Gate 5), and the governance items in condition 8.
