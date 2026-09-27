# Gap analysis: ArchiveTrust Loghi vs Riksarkivet Lion

Root-cause investigation of the performance gap on `svea-hovratt-2026-09-primary-v2`. This is **not**
an attempt to beat Lion. The question is how far a privately trained, scratch, hobby-level HTR model
got relative to an institutional model, and what explains the remaining gap. All work here is
diagnostic/secondary; nothing in this document changes the official primary run
(`svea-hovratt-v2-primary-20260927T102311Z`), the frozen benchmark, or any GT/eligibility/crop.

Every claim below is labelled **[FACT]** (directly counted from official evidence), **[ASSOCIATION]**
(a statistical pattern in the official evidence), **[HYPOTHESIS]** (a plausible, code- or
evidence-grounded explanation, not directly proven), or **[CONFIRMED CAUSE]** (an association that a
targeted diagnostic experiment actually tested and supports/rules out).

## 0. Official baseline [FACT]

| Metric | Loghi (beam 10) | Lion (`generation_config`, 4 beams) |
|---|---:|---:|
| CER | 32.70% | 18.42% |
| WER | 67.53% | 43.63% |
| Substitutions / Insertions / Deletions (chars) | 21,080 / 762 / 32,506 | 17,926 / 3,220 / 9,468 |
| Empty outputs | 71 | 0 |
| Failed lines | 0 | 0 |

Paired bootstrap (105-page clusters, 2000 resamples): CER gap 95% CI [−15.25, −13.30] pp, WER gap
95% CI [−25.28, −22.60] pp (Lion − Loghi); Lion ahead in all 2000 resamples and on all 4 documents and
all 105 pages (`page_document_summary.csv`: 105/105 pages have Lion CER below Loghi CER).
Full detail: `benchmark-data/reports/svea-hovratt-v2-primary-20260927T102311Z/`.

Evidence integrity for this analysis was re-checked before starting: benchmark manifest, `FROZEN.json`,
both official prediction files and `scores.json` all re-hashed to the values already recorded in the
official report; code commit unchanged (`8a2902a`).

## 1. Empty-output analysis (Phase 2)

**[FACT]** 71 of 4,627 Loghi predictions (1.5%) are empty (the container emitted no usable text after
canonicalization, not a formatting artifact — `normalization.py`'s `canonicalize_prediction` only does
NFC + outer-whitespace strip). Lion has 0 empty outputs on the same 4,627 lines.

**[ASSOCIATION -- very strong]** The 71 empty-output lines are geometrically extreme outliers, not a
random 1.5% of the corpus (`empty_output_analysis.json`):

| | Empty lines (n=71) | Rest of corpus (n=4,556) |
|---|---:|---:|
| Median crop width | 86 px | 695 px |
| Median aspect ratio (w/h) | 1.26 | 7.02 |
| Median reference characters | 3 | 35 |
| Median reference words | 1 | 6 |
| Median pixel area | 6,344 | 66,828 |

Empty-output lines are ~8x narrower, ~28x shorter in text, and almost always a single short word or
token. They are not randomly scattered: `by_document_rate` shows a 1.3–2.0% empty rate in every
document, i.e. present everywhere a short/narrow line occurs, not concentrated in one document. Lion's
own CER on these *same* 71 lines is also elevated (mean 48%, median 33%, vs corpus mean 20%/median
13%) — these are genuinely hard lines for both models, but only Loghi fails completely on them.

**[HYPOTHESIS]** Loghi resizes every crop to a fixed 64px height and lets width scale with the
original aspect ratio (`model_registry.py`'s VGSL input `None,None,64,1`); a very narrow/short crop
therefore produces very few CTC timesteps after the network's width downsampling. CTC decoding needs
at least as many timesteps as emitted symbols (before duplicate-collapse); if the timestep budget for
a tiny crop is below what a short word needs, the correct output can become structurally unreachable —
independent of beam width. Lion's fixed 384x384 ViT grid gives every crop the same patch budget
regardless of original size, so it has no equivalent failure mode. See `preprocessing_comparison.md`.

**[CONFIRMED CAUSE, partial]** This is *not* primarily a decoding-search-width problem: the beam=1
diagnostic (Section 6) produced *more* empty outputs (91), not fewer, ruling out "beam 10 is too
narrow" as the explanation and being consistent with the "too few timesteps, independent of beam
width" hypothesis. It was not confirmed at the timestep level itself (the CLI/results.txt does not
expose per-line timestep counts), so the CTC-timestep mechanism remains a strong, evidence-consistent
hypothesis, not a fully confirmed cause.

## 2/3/4. Deletion-heavy behavior and length bias (Phases 3-4)

**[FACT]** Deletions are 59.8% of Loghi's character edits (32,506 of 54,348) vs 30.9% for Lion (9,468
of 30,614) -- `length_bias.json`. This is corpus-wide, not just the 71 empty lines.

**[FACT]** Prediction/reference length ratio (`length_bias.json`):

| | Loghi | Lion |
|---|---:|---:|
| Mean | 0.80 | 0.98 |
| Median | 0.86 | 0.98 |
| P10 | 0.50 | 0.86 |
| Severe under-production (<50% of reference length) | 9.2% of lines | 0.65% of lines |
| Under-production (<75% of reference length) | 26.2% of lines | 3.5% of lines |

**[ASSOCIATION]** This is a genuine, corpus-wide systematic under-production bias for Loghi, well
beyond the 71 empty-output edge cases -- roughly a quarter of all Loghi predictions come in materially
shorter than their reference.

## 5. Geometry sensitivity (Phase 5)

**[FACT / ASSOCIATION -- the single strongest pattern in this analysis]**
(`geometry_sensitivity.json`, quantile-bucketed):

| Bucket (shortest/narrowest -> longest/widest) | Loghi mean CER | Lion mean CER | Loghi empty rate |
|---|---:|---:|---:|
| Aspect ratio 0.32-5.69 (narrowest) | 64.3% | 43.6% | 7.7% |
| Aspect ratio 5.69-6.67 | 36.0% | 19.5% | 0.0% |
| Aspect ratio 6.67-7.38 | 25.7% | 13.2% | 0.0% |
| Aspect ratio 7.38-8.49 | 26.0% | 14.7% | 0.0% |
| Aspect ratio 8.49-17.5 (widest) | 24.3% | 11.8% | 0.0% |

The same monotonic shape appears bucketed by width and by reference-character count: **short/narrow
lines are the hardest for both models, and almost the entire Loghi empty-output population sits in the
worst bucket.** This runs opposite to a naive "long lines are harder" intuition -- long lines are
comparatively *easier* for both models here, and the gap (in absolute CER terms) is largest in the
narrowest bucket (64.3% vs 43.6% = 20.7 pp) and smallest in the widest bucket (24.3% vs 11.8% = 12.5
pp), i.e. **Loghi's relative disadvantage is worst exactly where crops are smallest**, consistent with
the timestep hypothesis in Section 1.

## 6. Character-level diagnostics (Phase 6)

**[FACT]** Overall character match rate: Loghi 67.8%, Lion 83.5% (`char_stats_*.csv`).

**[ASSOCIATION]** Swedish diacritics are disproportionately weak for Loghi relative to its own
average, and much less so for Lion relative to its own average (`watch_char_stats.json`):

| Char | Loghi match rate | Lion match rate | Loghi del rate | Lion del rate |
|---|---:|---:|---:|---:|
| ä | 51.2% | 80.4% | 19.9% | 4.1% |
| ö | 58.0% | 83.3% | 21.0% | 3.8% |
| å | 66.1% | 84.5% | 16.4% | 4.4% |

Common Latin letters (e, a, n, t, r, s, l, d, i, o, h, space) sit close to or above each model's own
corpus average. å/ä/ö sit well *below* Loghi's average (67.8%) but only slightly below Lion's (83.5%).
The single out-of-vocabulary reference character (`¼`, 1 occurrence) is a negligible, already-documented
caveat, not a driver of the gap.

**[HYPOTHESIS]** Consistent with "scratch training vs pretrained representation" (category 5): a
pretrained ViT backbone (Lion) starts with generic visual-feature discrimination learned from large-
scale pretraining, which likely transfers to subtle diacritic strokes; a from-scratch CNN/CTC stack
must learn those less-frequent glyph shapes purely from the (smaller, less diverse) training corpus.
This is plausible, not confirmed -- confirming it would need a controlled ablation (e.g. per-character
training frequency vs match rate) not attempted here.

**[FACT]** Both models share several of the same top confusions (`top_confusions.json`): `a→e`,
`o→a`, `m→n`/`n→m`, `H→h`, `e→i`/`i→e` -- classic cursive visual-similarity errors common to both, a
shared intrinsic-difficulty effect, not something distinguishing the two models. Both also share
`s→ß` and `-→¬` as top confusions: the training corpus (for both models, per `model_registry.py`'s
`training_data_note` for Lion, and the project's own earlier training-convention audit for Loghi) uses
`ß` for certain double-s forms and `¬` for line-final hyphenation, so both models reproduce those
training-time conventions even though the benchmark's GT normalizes to `s`/`-`. **[DATASET/REFERENCE
EFFECT]**, not a model-quality difference -- this is exactly what the pre-registered D5 sensitivity
score exists to isolate, and D5's own near-zero effect on the official scores (Section 8 of the primary
report) confirms it is a small effect either way.

## 7. Positional deletion analysis (Phase 7)

**[FACT]** Deletions concentrate at the start and end of the reference line and are least frequent in
the middle, for **both** models, proportionally (`positional_deletions.json`, 9 usable bins of
normalized position 0-1): Loghi bin0 (line start) 7,551 deletions vs bin4 (middle) 2,358 vs bin8 (near
end) 4,977; Lion shows the identical U-shape at roughly a quarter of the magnitude (2,120 / 692 / 1,940).

**[HYPOTHESIS]** Because *both* models show the same qualitative shape, this looks like a shared,
crop-derived effect (tight bounding boxes are more likely to clip or under-represent the first/last
glyph of a line) rather than something specific to either architecture. Loghi is simply far more
sensitive to it in absolute terms, consistent with lower overall capacity/robustness rather than a
distinct failure mode.

## 8. Page/document clustering (Phase 8)

**[FACT]** Lion has lower CER than Loghi on **105 of 105 pages** -- there is no page where Loghi ties
or beats Lion (`page_document_summary.csv`). The smallest Lion advantage is 6.5 pp (4502445 p17: Loghi
34.1% vs Lion 27.6%); the largest is 28.5 pp (4502443 p3: Loghi 57.8% vs Lion 29.3%). Only 4 pages have
Lion CER above 30% (i.e. pages hard for Lion too); on those, Loghi is still worse by a similar margin
(mean 43.2% vs 31.6%). **There is no page-level pattern where the two models fail in qualitatively
different ways** -- the gap is remarkably uniform in direction, only its size varies, and that size
tracks the geometry effect in Section 5 (documents/pages with more short lines show a larger gap).

## 9. Preprocessing comparison (Phase 9)

See `preprocessing_comparison.md` (code-derived, no new computation). Both models are fed the exact
same, hash-verified crop file. Loghi: grayscale, fixed height 64px, variable width. Lion: RGB, fixed
384x384 ViT grid regardless of original aspect ratio. This is the structural basis for the Section 1/5
hypothesis: Loghi's input representation loses "resolution" (timesteps) on small crops in a way Lion's
fixed-grid representation cannot.

## 10. Crop integrity / review sample (Phase 10)

A deterministic sample (`crop_review_sample.csv`, seed 20260927) of 120 lines was built for optional
human visual follow-up: all 71 empty-output lines, the 20 highest Loghi-deletion-rate lines, the 9
additional (non-overlapping) largest Lion-vs-Loghi-gap lines, and 20 seeded near-perfect-for-both
control lines. **This diagnostic pass did not repeat a full manual visual review of every crop** --
the earlier D6 image review already established that visual crop review in this project is done by an
AI assistant, not a trained palaeographer, and the findings here are grounded in quantitative geometry
and text signals (width, aspect ratio, length ratios, character-level rates) rather than a new round of
subjective visual judgment. A human is invited to spot-check the sample, in particular the narrowest
empty-output crops, to confirm they are genuinely single short words/tokens and not, e.g., systematically
mis-cropped multi-word lines.

## 11. Decoding diagnostic (Phase 11) -- SECONDARY, NOT PRIMARY

**Preregistered hypothesis** (stated before running): if the empty-output and under-production pattern
were a beam-search-width artifact, a narrower beam (greedy, beam=1) would make it worse or unchanged;
the currently supported profiles (`model_registry.py`) only allow testing beam=1 vs the official
beam=10, not a wider beam.

**[CONFIRMED CAUSE -- ruled out]** Result (`decoding_diagnostic.json`, run
`svea-hovratt-v2-diag-loghi-greedy-20260927T102311Z`, non-official):

| | Loghi beam=1 (greedy) | Loghi beam=10 (official) |
|---|---:|---:|
| CER | 33.00% | 32.70% |
| WER | 67.61% | 67.53% |
| Empty outputs | 91 | 71 |
| Deletions | 33,065 | 32,506 |

Beam 10 is marginally better on every measure (confirming it as the right official choice), but the
effect size is tiny: 0.30 CER percentage points, out of a 14.3-point gap to Lion (2% of the gap).
Empty outputs drop from 91 to 71, not to zero. **Beam width does not materially explain the gap or the
empty-output phenomenon.** No further decoding variants were run (per Phase 14, only supported/already-
defined profiles were used; no new profile was invented to chase a result).

## 12. Preprocessing sensitivity experiment

**Not run.** The passive evidence (Sections 1, 5, 9) supports a preprocessing/architecture-interaction
hypothesis, but testing it properly (e.g. a deterministic minimum-width pad before Loghi's resize)
would require a new, hashed crop variant and a new inference run -- a real, non-trivial experiment, not
a quick check. Given the decoding diagnostic already shows the gap is not primarily decoding, and the
project's stated goal here is root-cause characterization rather than fixing it in this task, this is
named as the **single most promising next experiment** (Section 15) rather than attempted here.

## 13. Cause classification

**CONFIRMED IMPLEMENTATION / INFERENCE ISSUE**
- None. Decoding (beam width) was tested directly and ruled out as a material cause (Section 11).

**STRONG EVIDENCE OF MODEL LIMITATION (architecture/capacity/training, not yet retraining-confirmed)**
- Geometry-dependent catastrophic failure on short/narrow crops (Sections 1, 5): strong, code-grounded
  hypothesis (fixed-height/variable-width CTC input vs fixed-grid ViT input); fixable in principle
  without retraining (crop-side padding), not yet tested. Confidence: high that geometry sensitivity is
  real and much stronger for Loghi (directly measured); moderate-high that the CTC-timestep mechanism
  is the actual reason (plausible from code, not instrumented directly).
- Systematic corpus-wide under-production/deletion bias (Sections 2-4): real and large (26% of lines
  under-produce by >25%), only very weakly moved by beam width (Section 11), so most of it sits
  upstream of decoding -- consistent with model capacity/training rather than search behavior.
  Confidence: high that it's real; moderate on the exact mechanism.
- Diacritic (å/ä/ö) weakness disproportionate to Loghi's own baseline (Section 6): consistent with
  scratch training lacking the transfer benefit of a pretrained visual backbone. Confidence: moderate
  (plausible, not ablated).

**DATASET / REFERENCE EFFECT**
- `ß`/`s` and `¬`/`-` shared top confusions (Section 6): both models learned a training-time convention
  the benchmark's GT normalizes away; already isolated by the pre-registered D5 sensitivity score,
  which changes almost nothing (documented in the primary report). Confidence: high.
- Shared cursive-handwriting confusions (a/e, o/a, m/n, H/h) and shared start/end positional deletion
  shape (Section 7): intrinsic difficulty of the source material and/or crop tightness, common to both
  models, not a differentiator of the gap. Confidence: high that it's shared; the crop-tightness
  mechanism itself is a hypothesis.

**PLAUSIBLE BUT UNCONFIRMED**
- The exact CTC-timestep mechanism for empty outputs (Section 1): plausible and consistent with every
  test run so far, but not directly instrumented (would need internal timestep counts or a padding
  experiment).
- Pretrained-representation transfer as the explanation for diacritic weakness (Section 6): plausible,
  not ablated.

**UNKNOWN**
- Whether the remaining ~12+ percentage points of the CER gap not attributable to the above (geometry
  effects concentrate mostly in the narrow-crop tail, which is a minority of lines) is better described
  as "more training data/diversity would close most of it" vs "a structural capacity ceiling" -- this
  benchmark and this diagnostic pass cannot distinguish those without an actual training experiment.

Correlation-to-causation discipline: every "STRONG EVIDENCE" item above is an association that has a
plausible, code-grounded mechanism, but only the decoding item was actually tested by an intervention
(Section 11) and it was a negative result (ruled out as a material cause), not a positive confirmation
of another cause. Nothing here should be read as "confirmed" beyond that one ruled-out hypothesis.

## 14. Gap decomposition (from actual experiments only)

Official gap: CER 32.70% (Loghi) vs 18.42% (Lion) = 14.28 pp.

The only actual intervention run was the beam-width diagnostic: beam=1 -> beam=10 recovered **0.30 pp**
of CER, i.e. **~2% of the 14.28-pp gap**. No other diagnostic changed Loghi's output (all other
sections are analysis of the existing official predictions, not new experiments). **The other ~98% of
the gap is not decomposed by an actual experiment in this task** -- Sections 1-10 give strong,
evidence-grounded hypotheses for where a large share of it plausibly sits (geometry/architecture
interaction on short crops, general under-production, diacritic weakness), but per the task's own rule,
speculation about how much retraining could recover is explicitly not offered as a number.

## 15. Retraining decision

**Recommendation: A (fix inference/preprocessing first), with B (targeted retraining) as the next step
if A does not close a meaningful share of the gap.**

Evidence: the decoding diagnostic (Section 11) shows the currently-deployed decoding is already
close to the best available with existing profiles, so there's no cheap decoding win left. But the
geometry-sensitivity finding (Section 5) is the most actionable, evidence-backed lead in this entire
analysis, and it is testable **without retraining**: pad narrow/short crops to a minimum width before
Loghi's resize (an inference-time, crop-derived-variant experiment, per Phase 12, not yet run) would
directly test the CTC-timestep hypothesis. If that experiment recovers a material share of the gap
(especially the empty-output and worst-bucket-CER numbers), it argues the current scratch checkpoint's
raw recognition ability is better than 32.70% CER suggests, and any future retraining should include
this fix in the input pipeline rather than trying to have the model learn around a preprocessing
artifact. If it recovers little, that argues the remaining gap is a genuine training/capacity limitation
(category 5/6: scratch initialization, training-data diversity), and targeted retraining (more diverse
line lengths/geometries, possibly a pretrained visual initialization) becomes the better-justified next
investment. Either way, retraining is **not** recommended as the very next action -- the padding
experiment is cheaper, faster, and would sharpen which retraining investment (if any) is justified.

## 16. Best next experiment

**Deterministic minimum-crop-width padding diagnostic on Loghi**: define a new, separately-hashed crop
variant (e.g. pad narrow/short-aspect-ratio crops to a minimum width before Loghi's own resize;
Lion/GT/eligibility untouched), re-run Loghi (beam 10, unofficial/diagnostic run ID) on that variant
only, and compare CER/empty-output count against the official run on exactly the lines the padding
touches. This directly tests the strongest, most concrete hypothesis in this report (Section 1/5/9) with
a small, cheap, well-scoped experiment, and its result should drive the retraining decision in Section 15
rather than guessing.
