# Full-corpus Loghi retraining — lap 1 report

Run `full-corpus-20260802T044250Z`. Covers the terminology correction, the resume that completed
lap 1, and the evidence produced at the lap-1 decision gate.

**Status: stopped at the lap-1 decision gate. Lap 2 has not begun and will not begin without an
explicit decision.**

---

## 1. Terminology: what was wrong and what it cost

One shard is one container invocation over ~9,999 lines. **57 shards is one epoch** — one full pass
over the 563,933-line corpus. The full plan is 171 shards = 3 epochs.

The code counted shards in a field named `current_epoch` and reported that number as an epoch count.
At shard 53 the run therefore claimed "epoch 53" when it had completed **0** epochs and was 93% of
the way through the first. That is a 57x overstatement of progress, and it is not merely cosmetic:

- Early stopping was evaluated against `min_exposure_steps`, derived from the pilot's plateau at a
  9,999-line epoch. One shard is the same size as one pilot epoch, so that threshold is satisfied
  after roughly 20 shards — about a third of a real epoch. Patience then became authoritative.
- The run duly stopped at **shard 53 of 57** with `no_val_cer_improvement`, and marked itself
  `completed`. It had never completed a pass over the corpus.

Corrected fields, now reported everywhere (CLI `status`, GUI panel, `run_state.json`, checkpoint
metadata):

```
shards_completed_in_current_epoch: 53 / 57
epochs_completed: 0
epoch_progress: 0.9298
global_shards_completed: 53 / 171
```

`current_epoch` is retained so existing persisted `run_state.json` files stay readable, but its
docstring now states plainly that it counts shards, and `test_terminology.py` fails if that warning
is removed. The pilot is described throughout as a **pilot training pass** over a fixed 9,999-line
subset, never as a full-corpus epoch.

## 2. Why shard 53's stop was not evidence of a plateau

Trajectory over the final ten shards of lap 1's first attempt:

| shard | val CER | val WER | train CER |
| --- | --- | --- | --- |
| 44 | 0.1784 | 0.8960 | 0.1587 |
| 45 | 0.1804 | 0.8960 | 0.1545 |
| 46 | 0.1789 | 0.8910 | 0.1548 |
| 47 | 0.1779 | 0.9000 | 0.1529 |
| 48 | **0.1742** | 0.8960 | 0.1521 |
| 49 | 0.1766 | 0.9000 | 0.1535 |
| 50 | 0.1759 | 0.9020 | 0.1498 |
| 51 | 0.1759 | 0.8890 | 0.1504 |
| 52 | 0.1750 | 0.8910 | 0.1528 |
| 53 | 0.1746 | 0.8970 | 0.1491 |

Shards 49–53 all sit within 0.002 CER of the shard-48 best. That is the scale of shard-to-shard
noise, not a plateau: each shard trains on a *different* 9,999 lines, so consecutive validation
figures differ partly because the training data differed. Five such shards satisfied a patience of 5.

Treating that as convergence would have ended training on evidence from a fraction of the corpus.

## 3. The lap gate

Two changes, both in `orchestrator.py`:

- **Patience now requires a completed lap as well as minimum exposure.** The counter keeps
  accumulating and is still displayed; only its authority to stop the run is withheld until
  `epochs_completed >= 1`.
- **`--stop-at-epoch-boundary`** stops the run the moment a lap completes, rather than rolling into
  the next one. It records `stopped`, never `completed`: the plan has further laps, and marking a
  boundary as completion would make them unreachable, because resuming a terminal run is refused.

The run's own premature `completed` status was corrected to `stopped` through the normal
`mark_stopped` transition — not a raw JSON edit — with metrics, checkpoints and shard counts
untouched, and the correction script refuses to act if a lap genuinely had completed.

## 4. Files changed

| File | Change |
| --- | --- |
| `full_run/orchestrator.py` | lap gate on patience; `--stop-at-epoch-boundary`; `end_of_epoch` checkpoint at the real boundary |
| `full_run/epoch_accounting.py` | new — shard↔epoch arithmetic, `is_epoch_boundary` |
| `full_run/lap_evaluation.py` | new — end-of-lap evaluation suite |
| `full_run/evaluation_metrics.py` | new — true Levenshtein CER/WER, confidence bucketing |
| `full_run/integrity_cache.py` | new — fingerprint-cached resume integrity validation |
| `full_run/launch_guard.py` | delegates the image check to preflight's retrying implementation |
| `full_run/preflight.py` | bounded-retry `docker image inspect` with failure classification |
| `full_run/run_state.py` | honest epoch/shard fields; `current_epoch` documented as a shard counter |
| `full_run/cli.py` | corrected terminology in output; `--force-integrity-check`; `--stop-at-epoch-boundary` |
| `full_run/gui/app.py` | removed the self-contradictory "Current epoch (shard)" label |
| `scripts/evaluate_full_run_lap.py` | new — runs the lap evaluation against a real checkpoint |

## 5. Tests

| File | Tests |
| --- | --- |
| `test_lap_evaluation.py` | 23 (new) |
| `test_evaluation_metrics.py` | 18 (new) |
| `test_integrity_cache.py` | 14 (new) |
| `test_epoch_accounting.py` | 10 (new) |
| `test_docker_image_inspect_retry.py` | 10 (new) |
| `test_terminology.py` | 6 (new) |
| `test_orchestrator.py` | 23 (5 new: lap gate and boundary stop) |

`tests/htr/training/full_run/`: **321 passed**. Full suite: **2435 passed, 2 skipped, 0 failed**.

Notable regression guards:

- `test_difflib_is_not_a_substitute_for_levenshtein` pins the real error that reported CER 0.82 where
  the true figure was 0.248. `difflib` finds matching blocks, never a minimum edit script, so it
  cannot account for substitutions and systematically overestimates distance.
- `test_inspection_sample_covers_every_collection` pins the second half of that same error: reading
  the head of `val_list.txt` samples one collection, because the list is grouped by collection.
- `test_corpus_cer_is_edit_weighted_not_a_mean_of_per_line_rates` — the corpus figure must be
  comparable with the container's own `CERMetric`, which is edit-weighted.
- `test_legacy_current_epoch_field_is_documented_as_a_shard_counter` — fails if the warning is dropped.

## 6. Resume-time integrity checking

Full validation rehashes every shard manifest and computes exact train/val/test line-ID overlap.
Measured at **6.2s** on this corpus — not the "minutes" an earlier note claimed.

It is now fingerprint-cached on shard count, sizes, mtimes, dataset hash, validation hash and config
hash. On a cache hit the run still **re-reads real bytes** from a deterministic seeded sample of
shards, so validation is never skipped outright — only narrowed when nothing has changed. Measured
live on this run:

```
Shard integrity: cached (fingerprint matches the cached full validation);
re-read 8 sampled shards; overlap val=0 test=0 (0.5s / 2.1s across two resumes)
```

`--force-integrity-check` restores full revalidation on demand. Any fingerprint mismatch forces it
automatically.

## 7. Docker `image inspect` timeout

`docker image inspect` timed out at the original 15s limit on six real preflight runs and succeeded
on retry every time — a contended Docker Desktop daemon, not a missing image. It now uses a 30s
timeout, up to 3 attempts, and 2s incremental backoff, and classifies the failure rather than
collapsing every case into one message: `image_absent`, `daemon_unreachable`, `inspect_error`,
`malformed_response`. A genuinely absent image fails immediately with `docker pull` as the fix,
because it will not appear on retry. When a retry succeeds, the transient failure is still reported
in the message — never silently swallowed.

**This caused a real false negative during this work.** `launch_guard.py` kept its own private copy of
the check with the old 15s timeout that mapped *every* failure to "image not available", and it
refused a legitimate resume while the pinned image was present locally with the exact expected
digest `sha256:414fc89a…`. The fix for this failure class had been applied to `preflight.py` and this
second call site was missed. Both now share one implementation.

## 8. Checkpoint verification

All checkpoints written during lap 1, from `checkpoint_index.json`:

| | |
| --- | --- |
| total entries | 141 |
| `verification_status` | `verified` × 141 |
| `resumable` | `True` × 141 |
| entries with no `model_file_hash` | 0 |
| `optimizer_state_present` | `False` × 141 |
| `scheduler_state_present` | `False` × 141 |

The two `False` columns are the honest record of a container behaviour established by experiment
during the earlier audit, not an omission: `custom_callback.py` saves the output of
`tf.keras.models.clone_model()`, which is uncompiled and carries no optimizer state, and `main.py`
unconditionally rebuilds the LR schedule and optimizer on every invocation. Weights and
ArchiveTrust's own counters carry across shards; optimizer momentum and LR-schedule position do not.

An `end_of_epoch` entry is recorded at the real lap boundary, verified before being marked resumable,
as a separate index entry pointing at the same directory as `latest` — `checkpoint_index.py` is
append-only and already models one physical checkpoint carrying several kinds.

## 9. Lap 1 evaluation

One real inference pass over the full 1,000-line validation set with the lap-1 `end_of_epoch`
checkpoint (`epoch_57/model_new10/epoch_0_CER_0.1463_val_0.1756`), 111s, **1000/1000 lines scored**.
All figures below are recomputed independently from raw predictions using true Levenshtein distance —
not read back from the container.

### 9.1 Overall

| metric | value |
| --- | --- |
| corpus CER | **0.1735** |
| corpus WER (true word error rate) | **0.4878** |
| line error rate (what the container calls "WER") | 0.8930 |
| mean per-line CER | 0.1731 |
| exact-match lines | 107 / 1000 |

Container-reported figures at the same shard: `val_cer 0.1756`, `train_cer 0.1463`, `val_loss 21.74`,
`train_loss 17.69`. The independent recomputation (0.1735) agrees with the container's own CER to
within 0.002, which corroborates both.

Per-line CER distribution: p25 0.065, **median 0.129**, p75 0.240, p90 0.375, p99 0.714. Only 36
lines (3.6%) exceed CER 0.5. There is no catastrophic failure mode hiding inside the average.

### 9.2 The container's "WER" is not a word error rate

`WERMetric` in `model/metrics.py` computes `tf.edit_distance` over character labels and then counts
lines where the distance is **non-zero**, dividing by line count. Its local variable is named
`correct_words_amount` but it counts *incorrect lines*.

Verified numerically: the container reported `val_wer 0.8930` at shard 57, and exactly **893 of 1000
lines** contain at least one error. That is a line error rate.

The true word error rate, computed as word-level Levenshtein over the same predictions, is **0.4878**.

Every earlier figure in this project that cited `val_WER ≈ 0.89` as a word error rate overstated
word-level error by roughly 1.8x. The corrected reading is that ~89% of lines contain at least one
error while ~49% of *words* are wrong.

### 9.3 Per collection (worst first)

| collection | CER | WER | lines | ref chars | mean confidence |
| --- | --- | --- | --- | --- | --- |
| alvsborgs_losen | **0.2788** | 0.5074 | 91 | 1,797 | 0.720 |
| jonkopings_radhusratt_och_magistrat | 0.2679 | 0.7191 | 91 | 3,386 | 0.729 |
| gota_hovratt | 0.2350 | 0.6039 | 91 | 3,243 | 0.786 |
| bergmastaren_i_nora_htr | 0.2235 | 0.6433 | 91 | 2,756 | 0.767 |
| trolldomskommissionen | 0.2217 | 0.5737 | 90 | 3,302 | 0.791 |
| krigshovrattens_dombocker | 0.1656 | 0.4790 | 91 | 2,917 | 0.827 |
| goteborgs_poliskammare_fore_1900 | 0.1301 | 0.3931 | 91 | 2,445 | 0.885 |
| svea_hovratt | 0.1207 | 0.4044 | 91 | 2,593 | 0.892 |
| carl_fredrik_pahlmans_resejournaler | 0.1145 | 0.3801 | 91 | 3,973 | 0.892 |
| frihetstidens_utskottshandlingar | 0.0949 | 0.3537 | 91 | 2,412 | 0.885 |
| bergskollegium_relationer_och_skrivelser | **0.0582** | 0.2277 | 91 | 2,730 | 0.949 |

**A 4.8x spread between best and worst.** The corpus figure of 0.1735 describes no single collection
well. `alvsborgs_losen` was tracked explicitly as a subgroup and is confirmed as the weakest: its
terse tax entries are the shortest lines in the corpus (1,797 reference characters across 91 lines,
~20 chars/line against ~44 for `carl_fredrik_pahlmans_resejournaler`), leaving almost no context for
a sequence model to exploit. Its mean confidence (0.720) is also the lowest, so the model is not
overconfident about the collection it handles worst — which is the desirable direction.

### 9.4 Confidence calibration

Over all 1,000 lines — not a handful of examples.

| confidence bucket | lines | mean CER | median CER |
| --- | --- | --- | --- |
| 0.0 – 0.20 | 13 | 0.6313 | 0.6494 |
| 0.2 – 0.40 | 20 | 0.5219 | 0.4653 |
| 0.4 – 0.60 | 44 | 0.4133 | 0.3984 |
| 0.6 – 0.80 | 231 | 0.2749 | 0.2647 |
| 0.8 – 1.01 | 692 | 0.1051 | 0.0889 |

**Monotonic across every bucket**, by both mean and median: higher confidence really does track lower
error. Confidence is informative and usable for triage.

Two honest limits on that claim: the distribution is heavily top-weighted (692 of 1,000 lines land in
the top bucket), and the two lowest buckets rest on 13 and 20 lines respectively, so their means are
noisy. The monotonic trend is well supported; the precise error rate of the lowest bucket is not.

Rejection coverage:

| threshold | coverage | accepted lines | CER of accepted |
| --- | --- | --- | --- |
| 0.00 | 100.0% | 1000 | 0.1731 |
| 0.50 | 95.5% | 955 | 0.1555 |
| 0.70 | 86.1% | 861 | 0.1342 |
| 0.80 | 69.2% | 692 | 0.1051 |
| 0.90 | 41.6% | 416 | 0.0689 |
| 0.95 | 19.1% | 191 | 0.0506 |

Accepting only lines above 0.90 confidence covers 41.6% of the corpus at CER 0.069 — a fourfold
error reduction for a bit under half the volume.

### 9.5 Manual inspection

Fixed stratified sample, 3 lines from each of the 11 collections, selected by a stable hash seeded on
the checkpoint identity so the same sample is reproducible and comparable across laps. Full text in
`lap-evaluation/lap1_end_of_epoch/lap_evaluation.md`.

Recurring error classes visible by eye:

- **Case**: `Kan`→`kan`, `Malmslag`→`malmslag`, `Anders Andersson`→`anders andernen`.
- **Diacritics**: `Sölff`→`Solff` (ö→o), `öhrfihlar`→`åhrfihlar` (ö→å).
- **Abbreviation and notation marks**: `C.`→`C:`, `stn`→`st:`, spurious em-dash insertion
  (`pennr 73 mkr`→`penndr — 73 mk`), and the line-end soft hyphen `¬` frequently dropped.

**These are not the bulk of the error, and it would be wrong to report them as such.** Measured
rather than assumed:

| normalisation | corpus CER | reduction |
| --- | --- | --- |
| as-is | 0.1735 | — |
| case-insensitive | 0.1696 | 2.3% |
| + diacritics folded | 0.1609 | 7.3% |
| + punctuation removed | 0.1621 | 6.6% |

Folding case, diacritics and punctuation together removes under 8% of the error. The remainder is
genuine letter-level misrecognition — the most under-produced characters are ordinary letters
(`e`, `r`, `i`, `t`, `n`, `a`, `s`) and the most over-produced are likewise (`a`, `e`, `t`, `o`, `n`).
The model is misreading letters, not merely normalising them differently.

### 9.6 Best checkpoint vs end-of-lap checkpoint

Both evaluated by the same independent pass over the same 1,000 lines.

| | shard 48 (`best_val`) | shard 57 (`end_of_epoch`) | delta |
| --- | --- | --- | --- |
| corpus CER | 0.1728 | 0.1735 | +0.0007 |
| corpus WER | 0.4945 | 0.4878 | −0.0067 |

Per collection the deltas span −0.0046 to +0.0068 with mixed sign — seven collections marginally
worse, four marginally better. **The two checkpoints are equivalent within noise.** The nine shards of
training after the best-scoring shard neither helped nor harmed measurably.

Practical consequence: there is no accuracy reason to prefer one over the other. `best_val` is
selected on a metric whose shard-to-shard movement is smaller than its noise, so its "best" status is
close to arbitrary. The `end_of_epoch` checkpoint has the better provenance — it is the state of the
model after exactly one complete, auditable pass over the corpus.

### 9.7 Comparison with the pilot — lap 1 reached parity, it did not win

The full-corpus run validates against **the pilot's own `val_manifest.parquet`**
(`training/loghi-swedish-v1/manifests/val_manifest.parquet`, hash `92f74011…`) — the same 1,000
lines. The comparison is therefore like-for-like on validation data.

| model | training exposure | best val CER |
| --- | --- | --- |
| pilot, epoch 22 | 22 passes over a fixed 9,999-line subset | **0.1691** |
| full corpus, shard 48 | 1 partial pass over 563,933 lines | 0.1742 (container) / 0.1728 (recomputed) |
| full corpus, shard 57 | 1 complete pass over 563,933 lines | 0.1756 (container) / 0.1735 (recomputed) |

**After a full lap over 57x more data, the model has not beaten the 9,999-line pilot.** It is roughly
0.004 CER behind, which is itself within noise — so the honest statement is parity, not regression.
But parity is not what more data was expected to buy.

Line-exposure counts make this sharper: the pilot performed ~220,000 line-exposures (22 × 9,999); lap
1 performed ~570,000 (57 × 9,999). Lap 1 had **2.6x more exposure** and did not pull ahead.

**Most likely cause, stated as a hypothesis rather than a finding.** Optimizer momentum and
LR-schedule position are not carried across shards — established by experiment during the earlier
audit and recorded on all 141 checkpoint entries as `optimizer_state_present: False`. The container
rebuilds the schedule and optimizer on every invocation. The pilot therefore ran 22 optimizer
cold-starts; lap 1 ran **57**. Every shard restarts the learning-rate schedule at its initial value,
so the model never receives a sustained low-learning-rate consolidation phase — precisely the phase
that produces the final increments of accuracy.

If that is right, the ceiling here is set by the training harness, not by data volume, and further
laps will keep hitting it. Confirming or refuting it is the single highest-value next experiment, and
it is cheap relative to another lap: a small number of shards run with a manually continued
learning-rate schedule would settle it. **It has not been tested, and nothing in this report should be
read as having established it.**

## 10. Disk

Measured rather than estimated, because an earlier reading of the same numbers was wrong:

| | |
| --- | --- |
| free on D: | 170.9 GB |
| `epoch_output` after 55 shard dirs | 37.5 GB |
| per shard | **0.68 GB** |
| projected for a further lap (57 shards) | ~39 GB |

Free space fell from 322 GB to 171 GB during lap 1, but that was dominated by the one-time ~85 GB
extracted line-image pool created at preparation, not by per-shard growth. Disk is **not** a
constraint on continuing: laps 2 and 3 together need roughly 78 GB against 171 GB free.

## 11. Unresolved risks

| # | Risk | Status |
| --- | --- | --- |
| 1 | **Optimizer/LR-schedule discontinuity across shards.** 57 cold starts per lap; no sustained low-LR phase. Plausibly the accuracy ceiling (§9.7). | Open, untested, highest value to resolve |
| 2 | **More data has not yet produced more accuracy.** 2.6x the pilot's exposure yields parity, not improvement. | Open — this is the decision the gate exists for |
| 3 | **`alvsborgs_losen` at 0.2788 CER**, 4.8x the best collection. Shortest lines in the corpus. | Open; may be a data property rather than a fixable defect |
| 4 | **Container "WER" is a line error rate** (§9.2). Any prior document citing `val_WER` as word error rate is wrong by ~1.8x. | Fixed here; older documents not yet swept |
| 5 | **Calibration evidence is top-heavy** — 692 of 1,000 lines in the top confidence bucket; lowest two buckets rest on 13 and 20 lines. | Trend well supported; low-bucket rates are not |
| 6 | **A second false negative of the `docker image inspect` class** may exist elsewhere. Two call sites had it; one was fixed and the other missed for weeks. | Both now share one implementation; no third site found |
| 7 | **Reserved test set (810 lines) remains unused.** All figures here are validation figures, and validation has now influenced checkpoint selection. | By design — the sealed benchmark is for the final candidate only |

## 12. Decision gate — lap 2 has not begun

The run is `stopped` at `epoch_boundary_reached`, 57/171 shards, `epochs_completed: 1`. It is
resumable and will not proceed without an explicit instruction.

What the evidence supports:

- Lap 1 completed cleanly. All 141 checkpoints verified and hashed; no failed shard, no NaN, no OOM.
- The model is at CER 0.1735 corpus-wide, ~49% true WER, with usable confidence calibration.
- It has reached **parity with the pilot, not improvement on it**, despite 2.6x the exposure.

The three options, with what each is worth:

1. **Run lap 2.** Tests directly whether a second exposure helps. Costs ~7 hours and ~39 GB. If the
   optimizer-discontinuity hypothesis is right, it will produce another parity result.
2. **Test the optimizer-discontinuity hypothesis first** (§9.7). A few shards with a manually
   continued LR schedule. Far cheaper than a lap, and it determines whether laps 2 and 3 are worth
   running at all. **This is the recommendation.**
3. **Stop and evaluate the current checkpoint against the sealed 810-line test set.** Appropriate only
   if this model is to be treated as a candidate now; it spends the sealed set.

Nothing above is actioned. Awaiting an explicit decision.
