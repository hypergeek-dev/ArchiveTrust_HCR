# Loghi true-sequence-length decoder fix (`true-seqlen-decode-v1`)

**SECONDARY POST-BENCHMARK CAUSAL FOLLOW-UP.** Attempts the smallest possible fix to
`batch-invariance-v1`'s confirmed root cause: every sample in a Loghi inference batch is
CTC-decoded with the batch's uniform max-width timestep count instead of its own true length. No
GT was altered, no official prediction was modified, no image preprocessing/beam width/tokenizer
changed, and the pinned checkpoint/container image were never rebuilt -- the fix is applied only as
three files bind-mounted read-only over the container's own copies at container start.

## 0. Dry-run safety phase: **PASS**

Before any patched inference ran: all protected artifacts (official manifest, `FROZEN.json`,
official `scores.json`, official Loghi/Lion predictions, `GAP_ANALYSIS.md`, the padding and
batch-invariance diagnostics' reports, all 3 checkpoint files) were re-hashed directly from disk
(not taken from any cached value) and matched the values on record. The new `true-seqlen-decode-v1/`
directory was created with no path collision against any protected directory (checked via resolved
real paths; no symlinks/junctions present). A no-inference dry run resolved paths, verified the
checkpoint, and printed a plan (`dry_run_plan.json`) without calling docker or writing predictions;
protected hashes were identical before and after. A 4-line smoke test (one previously batch-sensitive
line each from `r1l12`/`r1l1`, one narrow 64px line, one normal 694px line) then ran the actual
patched container successfully (no crash, all 4 lines returned plausible results, the narrow line
stayed empty as expected); protected hashes were re-verified unchanged, and only the intended
diagnostic files existed under `true-seqlen-decode-v1/` afterward. See `run_record.json` for the
exact hash values (identical at every checkpoint, including the final one after the full matrix
re-run below).

## 1. Mechanism fix

Traced directly from source (`data/manager.py:_create_dataset`): `padded_batch` fills batch-added
padding with the exact sentinel constant `-10`, which normalized real image content (range roughly
`[-0.5, 0.5]`) never equals. This makes it possible to recover each sample's true pre-network content
width directly from the actual input tensor -- not by re-deriving the resize/padding arithmetic --
by finding the batch-padding-free prefix of each sample's width axis. Three files were patched via
read-only bind mount (hashes in `true_sequence_length_config.json`):

- **`modes/utils.py`**: computes `true_width` per sample from the `-10` columns, maps it to the
  network's own timestep axis via the confirmed 8x downsampling factor (`ceil(width/8)`, clamped to
  `[1, batch_timesteps]`), and threads it through the decode-worker queue as a 5th tuple element.
- **`utils/threading.py`**: `DecodingWorker.run()`/`.stop()` unpack the added element and pass it to
  `decode_batch_predictions` as `sequence_lengths`.
- **`utils/decoding.py`**: `decode_batch_predictions` gains an optional `sequence_lengths` parameter;
  when given, it replaces the buggy uniform `np.ones(pred.shape[0]) * pred.shape[1]`. Passing `None`
  reproduces the exact original behavior (verified: this is how batch composition still produced
  differences, since the argument is optional).

## 2. Unit tests

Run inside the pinned container (`patch/test_true_seqlen.py`), synthetic data, no benchmark crops:

| test | result |
|---|---|
| A: per-sample true width differs correctly in a mixed-width batch, independent of companions | **PASS** |
| B: batch-max padding timesteps are excluded from decode when the true length is supplied (buggy=`'ac'` vs fixed=`'a'`) | **PASS** |
| C: the same crop's logits decode identically regardless of the batch's own width, once fixed | **PASS** |

All three confirm the patch does exactly what it is supposed to do, in isolation, on synthetic
inputs where the only source of variation is the thing being tested.

## 3. Batch invariance, before vs. after (real crops, the exact frozen matrix)

The exact `batch-invariance-v1` matrix was reproduced byte-for-byte in its inputs (same
`batch_test_manifest.jsonl`, same `companion_groups.json`, same beam width 10 / seed 42, same 12
conditions) with only the decoder patched in.

| check | before fix (`batch-invariance-v1`) | after fix (`true-seqlen-decode-v1`) |
|---|---|---|
| batch_size=1 reproducibility (2 repeats) | 0/100 differ | 0/100 differ |
| bs16 (official) vs bs1 reference | 39/100 changed (39%) | 40/100 changed (40%) |
| bs2 vs bs1 | 19/100 (19%) | 20/100 (20%) |
| bs4 vs bs1 | 32/100 (32%) | 34/100 (34%) |
| bs8 vs bs1 | 33/100 (33%) | 35/100 (35%) |
| ordering sweep vs bs16-natural (same batch size, order only) | 11-17% differ | 9-17% differ |
| companion-context within-condition reproducibility (3 repeats) | 0/16 differ, every context | 0/16 differ, every context |
| companion-context across-context (10 targets) | 2/10 targets differ by context | 2/10 targets differ by context (same two: `r1l12`, `r1l1`) |

**The fix did not reduce batch-composition dependence at all** -- the percentages are statistically
indistinguishable from before (within the noise of a 100-line sample) and the same two target lines
that were causally shown to flip with companions before the fix still flip after it. Determinism
elsewhere (bs1 repeats, companion-context repeats) is preserved, ruling out a new source of
nondeterminism introduced by the patch itself -- whatever is happening is still a deterministic
function of batch composition, just not the one this fix addressed. Full detail:
`batch_invariance_after_fix.csv`, `batch_invariance_after_fix_summary.json`, raw per-condition
outputs in `batch_matrix_runs/`.

## 4. Why the fix likely didn't work (a plausible, unconfirmed explanation)

The patch corrects the *decode-time* symptom (uniform `sequence_length`) but not a more fundamental
issue one level earlier: the `-10` padding sentinel is an extreme, deliberately out-of-range value,
and it still passes through the model's convolutional layers *before* the sequence-length fix ever
sees the logits. Small-kernel convolutions with `'same'` padding have a local receptive field that
extends a few pixels into neighboring columns; a real-content column near the batch-padded edge can
therefore have its *feature values themselves* (not just its declared length) perturbed by an
adjacent run of `-10` pixels, and that perturbation depends on how much `-10` padding is adjacent --
i.e., on batch composition. If this is what's happening, no purely decode-side fix can remove the
effect, because the logits at the real timesteps are already slightly different before decoding
starts. Confirming this would require inspecting intermediate feature maps or changing how padding
is applied to the input tensor (e.g. a different fill value, or masking within the conv stack) --
which is out of scope for a decoder-only fix and is not attempted here, consistent with the task's
explicit constraint not to change image preprocessing. This is offered as the most likely explanation
given what has been traced and tested, not as a proven mechanism.

## 5. Corpus result, prediction changes, empty outputs, under-production, geometry, statistics, Lion-gap recovery

**Not applicable.** Per the preregistered stopping rule, these all depend on running the corrected
decoder over the full 4,627-line benchmark, which only happens if the fix passes the batch-invariance
gate above. It did not, so the full run was **not performed** -- no corrected Loghi predictions exist
to score, diff against the official run, or use for Lion-gap recovery. The official benchmark result
(CER 32.70% / WER 67.53%, 71 empty outputs) and the Lion gap (14.28pp) are exactly as recorded in the
primary run and `GAP_ANALYSIS.md`, unchanged and unaffected by this diagnostic.

## Verdict: **MECHANISM INCOMPLETE**

The decode-time sequence-length bug identified in `batch-invariance-v1` is real and the fix for it is
correctly implemented (proven by unit tests A-C in isolation), but it is not the whole story: on real
crops, batch-composition dependence persists at essentially the same magnitude with the fix applied
as without it. Something else -- plausibly convolutional receptive-field leakage from the `-10`
padding sentinel into real-content feature values, per §4 -- is producing most or all of the observed
effect. This diagnostic does not attempt to identify or fix that second mechanism.

## Stopping rule

**STOP CONDITION per the task's own gate was triggered**: batch-composition dependence did not
disappear after the fix, so the full 4,627-line corrected-decoder benchmark run was correctly not
performed, and no additional fix attempt was made within this task.

## Next step

**INFERENCE/BATCHING BRANCH CLOSED** for this specific fix attempt. A residual mechanism (plausibly
conv-level padding leakage, per §4) remains unexplained and, if pursued, needs its own separate
future diagnostic -- not undertaken here. Independently of that open question, `batch-invariance-v1`'s
own materiality finding still stands unchanged: the effect (whatever its exact cause) is small
(median edit distance ~2 characters, does not explain the 71 empty outputs, well under the ~1pp
materiality bar for the 14.28pp Lion gap) and is not a candidate explanation for the Lion gap. The
official benchmark result and `GAP_ANALYSIS.md` remain exactly as recorded; nothing here changes them.

**Do not retrain.**

## Artifacts

`true_sequence_length_config.json`, `run_record.json`, `dry_run_plan.json`, `smoke_test_run/`,
`patch/` (the 3 patched source files + unit tests, hashed), `batch_matrix_runs/` (raw per-condition
output for all 12 conditions), `batch_matrix_results.json`, `batch_invariance_after_fix.csv`,
`batch_invariance_after_fix_summary.json`.
