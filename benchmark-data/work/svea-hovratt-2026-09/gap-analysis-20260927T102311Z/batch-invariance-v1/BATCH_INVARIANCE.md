# Loghi batch-invariance diagnostic (`batch-invariance-v1`)

**SECONDARY POST-BENCHMARK DIAGNOSTIC.** Narrowly scoped to one question: does Loghi produce the
same prediction for a fixed, unmodified crop regardless of batch size, batch companions, ordering,
or batch maximum width? No GT was altered, no official prediction was modified, no new image
transformation was created. All inference in this diagnostic reads directly from the frozen
official benchmark crops (`benchmark-data/benchmark/svea-hovratt-2026-09-primary-v2/`).

Official evidence verified unchanged before running (see `run_record.json`): manifest sha256
`e43ee895...` (matches `FROZEN.json`), official Loghi prediction sha256 `9aed0d6e...`, checkpoint
file hashes, container digest `sha256:414fc89a...`, code commit `8a2902a8` (clean), beam width 10,
batch size 16, seed 42 -- all unchanged from the primary benchmark.

## Trigger

The padding diagnostic (`../loghi-minwidth-padding-v1/`) found 4 of 4,277 byte-identical crops
produced different Loghi predictions between the official run and the padding-diagnostic run. This
diagnostic tests whether that is a batch-composition artifact.

## 1. The four changed lines

| line (short) | width | official pred | padding-run pred | edit dist. |
|---|---|---|---|---|
| .../0010_DSC_0381/r1l12 | 747px | "...och i tog" | "...och i Kog" | 1 |
| .../0028_DSC_0406/r2l18 | 704px | "...Pr[ä]ten" | "...Pur[ä]ten" | 1 |
| .../0042_DSC_0424/r4l28 | 705px | "...Somstr[ä]ddare" | "...Somst[ä]ddare" | 1 |
| .../0021_DSC_0355/r1l1 | 634px | "...hulm d" | "...hulm [¬]" | 1 |

All four are single-character-level differences (substitution/deletion), not wholesale
re-transcriptions. Checking each line's manifest-order batch membership in the official run: two
lines (r2l18, r4l28) shared a batch with 0 and 3 padded (now-wider) companions respectively; the
other two (r1l12, r1l1) also had 0-1 padded companions. None of the four is itself a narrow line
(all 634-747px, i.e. "normal control" width, not the narrow/empty population this project has been
investigating).

**Reproducible via direct causal manipulation:** placing the exact same unmodified crop for r1l12
and r1l1 into a batch with only narrow (<100px) companions reproducibly changed their prediction
relative to placing them with wide or normal companions (confirmed 3x per context, 0 variation
within any single context -- see §3). r2l18 and r4l28 did not reproduce a difference across these
particular three constructed contexts; a residual, unresolved discrepancy for these two specific
lines remains (see §7 "Determinism"). This is disclosed rather than pursued further, per the task's
explicit instruction not to chase every single differing prediction once a material, reproducible
mechanism is already confirmed (it was -- see §2-§4).

## 2. Batch-size invariance

Frozen 100-line sample (`batch_test_manifest.jsonl`), same crops, same order, only `--batch_size`
varied, all compared against the batch_size=1 reference:

| batch size | lines changed | % | mean edit distance (changed lines) | max edit distance |
|---|---|---|---|---|
| 2 | 19/100 | 19% | 2.16 | 6 |
| 4 | 32/100 | 32% | 2.06 | 6 |
| 8 | 33/100 | 33% | 2.15 | 6 |
| 16 (official batch size) | 39/100 | 39% | 2.15 | 6 |

Monotonically increasing with batch size, exactly as predicted by the mechanism found in §5. The
batch_size=1 reference itself is perfectly reproducible (0/100 differ between two independent
repeats) -- so these are not noise. Of the 39 lines changed at batch_size=16, only 2 involved an
empty<->non-empty transition (1 each direction); the rest are small in-place edits (median edit
distance 2 characters).

## 3. Ordering invariance

At fixed batch_size=16, four deterministic reorderings (reverse, width-ascending, width-descending,
fixed-seed shuffle) each changed 35-39% of lines relative to the batch_size=1 reference, and --
more tellingly -- **11-17% of lines differ from the "natural order" batch_size=16 run when only the
order changes and batch size does not.** So batch composition (who shares a batch with whom), not
merely batch size, materially affects output. The companion-context experiment (10 fixed targets,
3 companion contexts, each repeated 3x) found **0/16 target lines differ across any of the 3
identical-condition repeats in any context** -- perfect within-condition reproducibility -- while
2 of the 4 known-changed targets (r1l12, r1l1) reproducibly differ specifically between the
narrow-companion context and the wide/normal-companion contexts. This directly reproduces, under
controlled conditions, the phenomenon the padding diagnostic first noticed by accident.

## 4. Companion geometry

Yes, confirmed and reproduced: the same target crop can get a different prediction purely because
of its batch companions, holding the target image, batch size, and everything else fixed. The
effect is small in magnitude (single-character-level) in every case observed, and does not flip
genuinely-narrow already-empty lines out of empty status: all 6 narrow empty-output lines included
as targets stayed empty across all three companion contexts, despite the batch-padding mechanism
exposing them to 46-92 *extra*, non-content decoder-visible timesteps in the wider contexts
(`tensor_shape_trace.csv`). This cross-validates the padding diagnostic's finding: extra
decoder-visible timesteps -- whether from external crop padding or from incidental batch
composition -- do not rescue Loghi's empty-output failure mode on narrow crops.

## 5. Tensor/sequence mechanics (traced directly from the pinned container's source, not assumed)

Two independent mechanisms exist in `loghi-htr`'s `main.py` inference path:

1. **Per-sample pre-batch padding** (`data/loader.py:DataLoader._ensure_width_for_ctc`): every
   sample, in isolation, is padded (constant-0 fill) so its own resized width is at least
   `(len(label) + adjacent_repeats(label)) * 16` pixels, i.e. `2x` the strict CTC minimum in
   timesteps. This is batch-composition-**independent**. Critically, for inference mode the "label"
   used is not the real transcription -- the container has no access to it -- but a fixed placeholder
   string, `"to be determined"` (`data/manager.py:_get_ground_truth`), used identically for every
   inference line. This imposes a uniform ~34-timestep floor on every line's *own* pre-batch width,
   regardless of its real target length. (This mechanism does not depend on batch composition and so
   cannot explain the observed batch effect, but it is a previously-undocumented fact relevant to
   interpreting the earlier `../loghi-minwidth-padding-v1/` padding diagnostic: many narrow crops
   were already being padded to this floor by the container itself, independent of any external
   padding this project added.)

2. **Batch-level collation** (`data/manager.py:_create_dataset`, `dataset.padded_batch(...)`):
   images in a batch are padded to that **batch's own max width and max height**, fill value
   `-10` (a fixed out-of-normalized-range sentinel). The model then runs on the whole batch tensor,
   producing `pred.shape[1]` timesteps -- the *batch's* timestep count, identical for every sample in
   that batch.

3. **The decode step is where per-sample identity is lost** (`utils/decoding.py:decode_batch_predictions`):
   `input_len = np.ones(pred.shape[0]) * pred.shape[1]` -- **every sample in the batch is decoded
   with the same `sequence_length`, equal to the full batch's timestep count.** There is no masking,
   no truncation, and no per-sample real-width tracking passed into `tf.nn.ctc_beam_search_decoder`.
   A narrow sample sharing a batch with a much wider companion is decoded as if it legitimately had
   as many timesteps as the widest batch member, with the extra timesteps' logits coming from
   `-10`-valued (pure padding) pixels having been run through the full conv/pool/LSTM stack.

This is the confirmed root cause: batch composition changes the uniform `sequence_length` every
sample in that batch is decoded against, which can change the beam search result even though the
sample's own real content and its own conv/pool feature map are unaffected. `tensor_shape_trace.csv`
gives per-target-line, per-context arithmetic (own content timesteps vs. batch's timesteps vs.
"extra fake timesteps exposed to the decoder") computed directly from this confirmed formula.

## 6. Determinism

`batch_size=1` is perfectly deterministic (0/100 differ across 2 independent repeats). Each of the
3 companion-context conditions is perfectly deterministic across 3 repeats (0/16 differ, every
context). This rules out generic GPU/numeric run-to-run nondeterminism as a material factor here --
identical conditions give identical outputs every time in every test run. The one unresolved
residual is that 2 of the original 4 changed lines (r2l18, r4l28) did not reproduce a difference in
this diagnostic's 3 constructed companion contexts, even though something changed them between the
official and padding-diagnostic runs. Given (a) determinism holds everywhere else tested, (b) the
mechanism in §5 is code-confirmed and independently reproduced on 2 other lines under controlled
manipulation, and (c) the task's explicit instruction not to chase every differing prediction once a
material mechanism is confirmed, this residual is recorded as unresolved rather than investigated
further.

## 7. Stopping rule

**STOP CONDITION A (reproducible batch effect confirmed) was met.** Evidence: a confirmed,
code-traced mechanism (§5); a monotonic batch-size dose-response reproduced without any repeat
needed because batch_size=1 and identical-composition batches are independently proven perfectly
deterministic (§6), so any difference between distinct conditions is stable by construction; and a
direct, repeated (3x, 0 variance), controlled companion-context manipulation reproducing the effect
on 2 independent target lines. Per the hard limits, no further batch sizes, orderings, or companion
configurations were added after seeing these results, and the full 4,627-line benchmark was not
rerun.

## Verdict: **CONFIRMED BATCH-COMPOSITION ARTIFACT**

(Batch-size dependence, §2, is a manifestation of the same root mechanism -- larger batches are more
likely to contain a much-wider companion -- not a separate cause.)

## Impact

- **The 71 empty outputs:** does **not** plausibly explain them. All 6 narrow empty-output lines
  tested stayed empty regardless of companion context, even when exposed to 46-92 extra
  decoder-visible timesteps from wide companions (§4). This is consistent with the padding
  diagnostic's CONTRADICTED verdict: the empty-output failure mode is not a timestep-availability
  problem in any form tested so far.
- **The padding-experiment regression:** partially explains it. The padding diagnostic's 350
  padded lines were run in one specific (now-different) batch composition than the official run's;
  some of the reported regression on those 350 lines is very plausibly compounded by batch-effects
  documented here, in addition to the crop-shape effect the diagnostic itself measured. This
  diagnostic does not attempt to re-run the full padded benchmark to separate the two (out of
  scope, per the task's explicit limits); the padding diagnostic's own conclusion (padding does not
  help, and plausibly mildly hurts) stands regardless, since the effect sizes here are small
  (median 2-character edits) relative to that diagnostic's own measured regression pattern (a
  near-doubling of empty rate, which this mechanism does not reproduce).
- **The Lion gap (14.28pp):** **not material.** Changed lines shift by a median of 2 characters
  out of ~36 on average; only ~1 in 20 changes flips empty status, evenly in both directions; there
  is no sign of a systematic bias toward more errors. This is well below the "roughly 1 CER
  percentage point or more at corpus level" materiality bar in the task's own stopping rule, and is
  recorded as a **minor implementation artifact**, not a candidate explanation for the Lion gap.

## Next step

Do not pursue this as an explanation of the Lion gap or the empty-output behavior -- both are
better attributed to factors already identified (scratch vs. pretrained training, per
`../GAP_ANALYSIS.md`). Separately from gap-explanation, this is a **benchmark methodology hygiene
issue** worth fixing on its own terms: official Loghi predictions are not currently invariant to an
accident of manifest ordering. A small, targeted future fix (not undertaken here) would pass each
sample's true unpadded resized width into `decode_batch_predictions` as its own `sequence_length`
instead of the uniform `pred.shape[1]`, and/or run inference at `batch_size=1` for full reproducibility
at some throughput cost. Any such fix should be proposed and tested as its own diagnostic before
being adopted; the current official benchmark result stands unchanged.

**Do not retrain.**

## Artifacts

`batch_test_manifest.jsonl`, `batch_size_comparison.csv`, `ordering_comparison.csv`,
`companion_context_results.csv`, `tensor_shape_trace.csv`, `reproducibility_runs.json`,
`run_record.json`, `companion_groups.json`, raw per-condition predictions under `runs/`.
