# Loghi Swedish HTR: Experiments 0, 1, and 2 — combined research report

Generated 2026-08-04. Covers three completed, immutable training runs. Every substantive claim below
is tagged **MEASURED** (directly observed in this work), **VERIFIED FROM SOURCE** (confirmed by reading
the pinned `loghi-htr` source or this repo's code, not assumed), **INFERRED** (a reasonable conclusion
from measured evidence, not itself directly measured), **HYPOTHESIS** (plausible but untested), or
**UNRESOLVED** (a real open question this report does not resolve).

## 1. Research objective

Determine whether a Loghi HTR model can be productively adapted to Swedish historical handwriting, and
whether the training *method* used to do so materially affects the result. Three questions, addressed
in sequence across the three experiments:
- Was the original shard-based fine-tuning methodology (Experiment 0) sound? [answered: no — it
  silently reset the optimizer and LR schedule 57 times, see §7]
- Does fixing that flaw (Experiment 1) change the outcome? [answered: not materially, see §6]
- Can a natively-architected model trained from scratch on this corpus compete with, or productively
  diverge from, fine-tuning a foreign-pretrained checkpoint (Experiment 2)? [partially answered after
  one epoch, see §8–§18]

## 2. Swedish Lion benchmark motivation

**VERIFIED FROM SOURCE.** This repository's active research phase (`docs/experiments/
lion-loghi-comparison/README.md`) compares two locally-executed methods, `swedish_lion` and `loghi`,
in-domain and cross-domain on Swedish and Dutch historical handwriting. `swedish_lion` in this phase
means **Swedish Lion Libre** (`Riksarkivet/trocr-base-handwritten-hist-swe-2`), a real, local TrOCR
model run in-process — explicitly *not* `transkribus_swedish_lion_1`, the external, manual-upload
Transkribus workflow that the phrase "Swedish Lion I" refers to elsewhere in this codebase. This
distinction is load-bearing and was made deliberately (`docs/CAPABILITY_MATRIX_HTR.md` §6).

The motivation for training a Loghi model at all is to give `loghi` a fair, Swedish-adapted competitor
to `swedish_lion` in that comparison, rather than evaluating the pinned Dutch-pretrained Loghi
checkpoint zero-shot on Swedish text.

## 3. Limits caused by Lion's partly unavailable training corpus

**VERIFIED FROM SOURCE, with an UNRESOLVED gap disclosed honestly.** The corpus these three
experiments train on is Riksarkivet's own "Training data for Swedish Lion Libre" HuggingFace
collection (`training_identity.py::TRAINING_DATASET_ID = "riksarkivet_swedish_lion_libre_training_data"`,
`swedish_dataset_inventory.py`'s own docstring). This is the same named dataset associated with the
Swedish Lion Libre model itself.

**UNRESOLVED**: this report cannot confirm the exact overlap between (a) the 562,123 lines these three
experiments trained on, (b) the 1,000/810-line validation/reserved-test splits used for evaluation, and
(c) whatever exact subset of this collection Riksarkivet actually used to train the released Swedish
Lion Libre model. If Swedish Lion Libre was trained on lines that overlap this project's validation or
reserved-test sets, any future benchmark against it on those sets would be optimistic for Swedish Lion,
not for Loghi. This is exactly why §22 of the brief this report responds to insists that no Swedish
Lion conclusion is possible without a **held-out set both models are known not to have trained on** —
restated explicitly in §23 below.

## 4. Dataset and split identity

**MEASURED / VERIFIED FROM SOURCE.**
- Full corpus: 562,123 usable training lines, 1,000-line fixed validation set, 810-line sealed reserved
  test set (never opened by any of the three experiments or by this report).
- `dataset_hash`: `fc21a70825650927edcae753a20411cef8e4344e2c22bcbcaed1c0038b9b97b0` — identical across
  all three experiments (confirmed by direct comparison in `experiment_0_1_2_comparison.json`).
- `train_manifest_hash`: `e12796a02c868d849b9e113a4b415f94ff483fe2148c1c58d17cd36eba4af46f` — identical
  across all three.
- `val_manifest_hash`: `92f74011c568b04798c9c53725311633f2df2ea4fb8c408e1d73b4f197cea85c` — identical
  across all three, independently re-hashed at each launch, not merely copied forward.
- Train/validation image-path overlap: 0 (measured directly by set intersection over both list files).
- Train/reserved-test overlap: not independently re-measured by opening the sealed file (by design);
  inherited structurally from `corpus_sharding.py`'s own line-ID exclusion logic, exercised when the
  shared pool was first built for Experiment 0/1 and reused unmodified by Experiment 2.

## 5. Experiment 0 methodology and result

**Loghi generic fine-tune — shard-reset baseline.** Generic pretrained Loghi checkpoint
(`generic-2023-02-15`, sha256 `0da2c00a...`), fine-tuned over the full corpus through 57 separate
shard-container invocations. Weights carried forward between shards; **optimizer state and
learning-rate schedule position did not** — both were rebuilt fresh at every shard boundary
(`training_session.py`'s own docstring, confirmed empirically 2026-08-01: real checkpoints inspected
report `optimizer.iterations == 0`). Runtime: 25,093s cumulative training time across 57 sessions.
Held-out result (independent evaluation, best_val checkpoint at shard 48): **corpus CER 0.1728, true
WER 0.4945**.

## 6. Experiment 1 methodology and result

**Loghi generic fine-tune — one continuous epoch.** Same generic pretrained checkpoint, same corpus,
same validation set, same optimizer/LR hyperparameters as Experiment 0 — the only changed variable is
trainer lifecycle: one uninterrupted `main.py --epochs 1` container invocation instead of 57. Optimizer
iterations ran 0 → 35,099 continuously, proven via a purpose-built instrumentation callback, not
assumed. Runtime: 16,543s. Held-out result: **corpus CER 0.1747, true WER 0.4956** — a delta of
+0.0019 CER / +0.0011 WER relative to Experiment 0, within observed noise. **Correcting optimizer
continuity was methodologically necessary but did not materially change the outcome for this
architecture/corpus/one-epoch budget** (see the preserved Experiment 0 vs 1 comparison,
`training/preserved-experiments/loghi-finetuning-baselines-20260804/`).

## 7. Why optimizer continuity was tested

**VERIFIED FROM SOURCE.** Reading `custom_callback.py::_save_model` showed every checkpoint this
pipeline ever produced was saved via `tf.keras.models.clone_model()` — a fresh, uncompiled model
carrying weights only, never optimizer state — and `main.py:104-119` unconditionally rebuilds a fresh
optimizer and LR schedule on every invocation with no "am I resuming" branch. Experiment 0's 57-shard
design therefore never ran one continuous training process; it ran 57 independent short fine-tunes
chained by weights alone. Experiment 1 exists specifically to isolate whether that flaw mattered.

## 8. Experiment 2 scratch-training hypothesis

**Loghi recommended VGSL — trained from scratch.** Hypothesis: a model trained natively on Swedish
data from random initialization, using Loghi's own recommended architecture, might learn a
representation better suited to this corpus than adapting a checkpoint pretrained on a different
language's handwriting — or might simply need more than one epoch to compete. This experiment tests
the first epoch of that trajectory only; see §21 for why one epoch cannot answer the full question.

## 9. Recommended VGSL architecture

**VERIFIED FROM SOURCE, not guessed.** `--model recommended` resolves via `model/management.py::
get_model_library()` to:

```
None,None,64,1 Cr3,3,24 Mp2,2,2,2 Bn Cr3,3,48 Bn Cr3,3,96 Mp2,2,2,2 Bn Cr3,3,96 Mp2,2,2,2 Bn Rc3
Bl512 D50 Bl512 D50 Bl512 D50 Bl512 D50 Bl512 D50 Fs92
```

4 conv blocks (24→48→96→96 channels, 3× max-pool: height 64→32→16→8) → reshape → 5× BiLSTM(512,
dropout 50% between each) → dense output. The literal `Fs92` is irrelevant: `customize_model` always
resizes the final layer to `len(tokenizer)` for any from-scratch build, confirmed by direct
measurement (126, not 92, in the real built model).

## 10. Character vocabulary

**MEASURED**, directly from the real 562,123-line train list and 1,000-line validation list (not the
generic checkpoint's Dutch-oriented charlist). 124 unique characters in training, 85 in validation,
**0 validation-only characters** — the gate built for this purpose (`analyze_experiment_2_character_
vocabulary.py`) passed cleanly, meaning no validation lines were silently dropped from Keras's
in-training evaluation partition (a real risk identified and checked, not assumed away — see
`data/manager.py::_is_valid_ground_truth`). Tokenizer vocabulary size: 126 (124 characters + `[PAD]` +
`[UNK]`). SHA-256: `c0fb042e0dff9ad337dc60465c6af8cc738151505abc5b1e151155582fb09d92`.

## 11. Parameter comparison

**MEASURED, in-container.** Experiment 2 (recommended, post-vocab-resize): 30,694,654 total
parameters (30,694,126 trainable, 528 non-trainable — BatchNorm running statistics). Experiment 1's
real loaded checkpoint: 31,033,929 parameters. **Similar but not identical**: same conv+5×BiLSTM(512)
backbone topology and input shape; Experiment 1's checkpoint has Dropout layers directly after each
conv block that the recommended spec lacks; the ~340K parameter gap is almost entirely the
vocabulary-driven output layer (468,425 params for 457 Dutch-inclusive classes vs 129,150 for 126
Swedish-only classes — a 339,275-parameter difference against a 339,275 total gap).

## 12. Training trajectory

**MEASURED**, from `experiment_2_training_metrics.parquet` (72 logged rows, every 500 optimizer
steps). Train CER: ~1.03 (near-random) at step 500 → 0.2749 at step 35,111 (train_end). Loss: 95.3 →
32.0. The decline was still undiminished at the last logged interval (steps 34,500→35,111: CER
0.27741 → 0.27491, no flattening) — the standard signal that this model has not reached a
capacity-limited plateau within one epoch.

## 13. Overall validation metrics

**MEASURED**, independent evaluation (`scripts/evaluate_full_run_lap.py`, true Levenshtein CER/WER,
never `difflib`, best_val checkpoint, all 1,000 validation lines scored, 0 missing results):

| metric | Experiment 2 |
|---|---|
| Corpus CER | 0.1918 |
| True WER | 0.5470 |
| Line error rate | 0.935 |
| Mean per-line CER | 0.1998 |
| Total reference characters | 31,554 |
| Total edits | 6,051 |

## 14. Per-collection results

**MEASURED.** Same ranking pattern as Experiments 0 and 1: worst collection `alvsborgs_losen` (CER
0.3489), best `bergskollegium_relationer_och_skrivelser` (CER 0.0736). Full table in
`experiment_0_1_2_comparison.md`. No collection shows a qualitatively different pattern from the
fine-tuned baselines — the same subsets of the corpus are hard or easy regardless of training method.

## 15. Confidence calibration

**MEASURED.** Monotonic: mean CER falls as confidence bucket rises (bucket 0.0–0.2: mean CER 0.773;
bucket 0.8–1.01: mean CER 0.103). The model's own confidence is a meaningful, well-ordered signal
after one epoch from scratch, not noise.

## 16. Error analysis

**MEASURED (aggregate) / UNRESOLVED (substitution/insertion/deletion breakdown).** Corpus-level edit
count (6,051 edits over 31,554 reference characters) is recorded; a per-operation (substitution vs.
insertion vs. deletion) breakdown and common-confusion-pair analysis was not computed in this pass —
the evaluation harness (`evaluation_metrics.py::levenshtein`) computes total edit distance, not an
operation-typed alignment. Flagged here as a genuine gap rather than presenting a fabricated
breakdown; a follow-up pass could extend `evaluation_metrics.py` to return typed edits if this
granularity is wanted.

## 17. Comparison of pretrained versus scratch initialization

**MEASURED**, with an explicit scope caveat. Held-out CER: Experiment 0 0.1728, Experiment 1 0.1747,
Experiment 2 0.1918. Experiment 2 trails the pretrained baselines by 0.017–0.019 CER after one epoch.
**This is not a single-variable comparison** — Experiment 2 differs from Experiments 0/1 in both
initialization (random vs. pretrained) *and* architecture (recommended VGSL vs. the generic
checkpoint's architecture) simultaneously (§11). The correct description, used throughout this report,
is *native recommended architecture trained from scratch versus generic-checkpoint fine-tuning* — not
an isolated ablation of either variable.

## 18. Interpretation

**INFERRED**, from measured evidence. A model starting from complete random initialization closed to
within ~0.02 CER of two checkpoints that started already pretrained on a much larger prior corpus,
within a single epoch, while still improving at a steady rate when that epoch ended. This is evidence
that the recommended architecture learns Swedish handwriting efficiently from scratch, not evidence
that it has plateaued below the fine-tuned baselines. Per the brief's own explicit instruction, this
report does not interpret one epoch from random initialization as the architecture's ceiling, and does
not claim scratch training is inferior merely because it trailed after one epoch of slower initial
convergence than a warm-started model.

## 19. Threats to validity

- **Confounded comparison** (§17): architecture and initialization both changed at once in Experiment
  2; no isolated "same architecture, random vs. pretrained init" or "same init, recommended vs. new10
  architecture" experiment exists yet.
- **Single seed, single epoch**: none of the three experiments were repeated with different seeds;
  variance across seeds is unmeasured. The 0.0019 CER delta between Experiments 0 and 1 is itself
  evidence that noise on this order of magnitude exists.
- **No error-type breakdown** (§16).
- **Swedish Lion training-data overlap unresolved** (§3).
- **Dropout-driven train/val CER inversion** (Experiment 2's val CER 0.194 is lower than its train CER
  0.275): plausible and consistent with active dropout during training-mode forward passes, but not
  independently confirmed by disabling dropout and re-measuring train CER.

## 20. Sealed-test status

**VERIFIED.** The 810-line reserved test manifest (`training/loghi-swedish-v1/manifests/
test_reserved_manifest.parquet`) was not opened, read, or hashed by Experiment 2's training, its
evaluation, or this report. Existence and file metadata (size, mtime) were checked before and after
this work and are unchanged (`ls`-level check only, no content access).

## 21. Recommendation for epoch 2

**RUN_SCRATCH_EPOCH_2.** See `experiment_2_decision.json` for the full criterion-by-criterion evidence
trail. Summary: all five conditions favoring another epoch are met (still-improving trajectory at
epoch end, stable training, no overfitting signal, performance approaching the fine-tuned baselines,
and an undiminished improvement rate); none of the four stop conditions are triggered. Per the brief's
explicit instruction, no second epoch was started automatically — this is a recommendation, not an
action.

## 22. Recommendation for a later custom VGSL architecture

**HYPOTHESIS.** Given Experiment 2's backbone is nearly identical in size and topology to the generic
checkpoint's own architecture (§11), a genuinely different architecture family (different depth, width,
or recurrent unit choice) has not yet been tried on this corpus at all — every experiment to date uses
essentially the same conv+BiLSTM+CTC shape. If a second scratch epoch (§21) still leaves a persistent
gap to the fine-tuned baselines after the trajectory has clearly plateaued, a genuinely different
architecture would be a more informative next step than a third epoch of the same one. This is a
hypothesis about future work, not a claim about the current architecture's adequacy.

## 23. Explicit statement: no Swedish Lion conclusion is yet possible

No conclusion about Loghi versus Swedish Lion Libre can be drawn from this report. These three
experiments establish an internal Loghi baseline only. A Swedish Lion comparison requires, at minimum:
(a) a held-out set both models are independently confirmed not to have trained on (§3's unresolved
overlap question must be closed first), (b) identical preprocessing/normalization applied to both
models' inputs, and (c) the explicit framing — stated in the original request that began this body of
work — that any such comparison is an **external benchmark**, not a controlled architecture
comparison, because the two systems differ in architecture, pretraining, and training data beyond what
either party's HTR pipeline controls.
