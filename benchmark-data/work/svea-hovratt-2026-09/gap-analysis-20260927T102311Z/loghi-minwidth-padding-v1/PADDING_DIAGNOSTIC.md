# Loghi minimum-width padding diagnostic (`loghi-minwidth-padding-v1`)

**SECONDARY POST-BENCHMARK DIAGNOSTIC / IMPROVED INFERENCE VARIANT.**
Does not replace, and is not comparable in status to, the official primary result
(`benchmark-data/reports/svea-hovratt-v2-primary-20260927T102311Z/BENCHMARK_REPORT.md`):
Loghi (beam 10) CER **32.70%** / WER 67.53%; Lion (`generation_config`) CER **18.42%** / WER 43.63%.
That result is unchanged and remains the project's primary evidence. This diagnostic tests one
narrow, preregistered hypothesis about Loghi's failure mode on narrow crops, using only Loghi
inference on a derived, separately-hashed crop set; Lion was not rerun.

Chronology: official benchmark → root-cause gap analysis (`../GAP_ANALYSIS.md`) → this
preregistered hypothesis and intervention → measured outcome (below).

## Hypothesis (preregistered before any padded-crop result existed)

Ensuring a minimum horizontal input width (hence CTC timestep budget) reduces Loghi's empty-output
and under-production failure modes on short/narrow line crops — with the explicit, honest caveat
recorded in `padding_config.json` *before* running anything: the strict CTC-impossibility mechanism
only explains 2 of the 71 official empty lines (`T < T_min`); 69/71 already had a theoretically
sufficient timestep budget. The real proposed mechanism was softer: narrow crops leave little
*margin* above the bare CTC minimum (mean slack 6.83 timesteps for empty lines vs. 18.74 for
non-empty lines), and a weakly-trained scratch model may be disproportionately likely to collapse
to empty/degenerate output under a tight margin even when the minimum is technically satisfiable.

## 1. Architecture derivation

Confirmed by building the real "recommended" VGSL model inside the pinned container
(`loghi/docker.htr:latest@sha256:414fc89a...`) and running 25 real forward passes at synthetic
widths 8–900 (`vgsl_timestep_probe.py`): three `Mp2,2,2,2` stride-2 maxpool stages give an exact
8x horizontal downsampling, confirmed empirically as **T = ceil(resized_width / 8)**, where
`resized_width = original_crop_width * 64 / original_crop_height` (the model resizes every crop to
fixed height 64). Input axis order confirmed empirically as `(batch, width[None], height[64],
channels[1])`. CTC minimum alignment: `T_min = L + R` (L = reference length, R = adjacent-repeat
count). Full derivation and the corpus-wide sufficiency check are in `padding_config.json` →
`architecture_derivation` and `../ctc_timestep_summary.json`.

**MIN_WIDTH = 230px** (original, pre-resize crop width), derived *only* from geometry
(image_width/height) and GT reference-length facts — never from `loghi_empty`/`loghi_cer` outcomes:
`needed_T = narrow_T_min_p95(7) + healthy_slack_p25(12) = 19` timesteps → `resized_width_needed =
152px` → `MIN_WIDTH = ceil(152 * 95/64) = 226px`, rounded up to 230px, using the p99 narrow-line
height (95px) as the conservative worst case. Full formula and the three explicitly-flagged
judgment calls (p95/p25/p99 rather than max/median/median) are in `padding_config.json`.

## 2. Crop transformation

Symmetric background padding only — no rescaling, no rotation, no contrast/denoising, no
GT-dependent processing. 350/4,627 lines (7.6%) had `image_width < 230px` and were padded; the
remaining 4,277 were copied byte-identical into the derived set (confirmed: 0 hash mismatches).
Fill colour is per-line, image-derived only (mean colour of that line's own 2px edge strip on each
side), never GT-dependent. Verified before inference: all 4,627 `line_id`s match the official
manifest 1:1; all `gt_canonical_sha256` unchanged; all original crop hashes match the official
manifest; no dimension ever shrank; the generator was run twice and produced byte-identical
manifests (`diff` clean). Artifacts: `padding_config.json`, `derived_crop_manifest.jsonl`
(sha256 `7c74a70c...`).

## 3. Empty outputs: before 71, after 138, recovered 18

Official Loghi had 71 empty outputs. After padding, Loghi produced **138** empty outputs overall —
**more**, not fewer. Of the original 71: 53 are still empty, 8 became partially correct, 10 became
non-empty-but-incorrect, 0 became exact (18/71 "recovered" from empty status, mean CER of those 18
= **85.5%** — barely better than an empty guess). Separately, **85 new empty outputs** appeared on
lines that were *not* empty officially — and all 85 of these are among the 350 *padded* lines
themselves (0 among the 4,277 unchanged lines). Within the narrowest geometry bucket (image width
24–651px, n=945, the same quintile bucket used in `../GAP_ANALYSIS.md`), the empty rate nearly
**doubled**: 7.41% → 14.50%. All four wider buckets show *zero* change (`padding_geometry_comparison.csv`),
confirming the effect is isolated to the padded population, not a general regression.

## 4. Primary diagnostic result

Full-corpus Loghi CER: official 32.70% → padded **32.72%** (Lion unchanged reference: 18.42%).
WER: 67.53% → 67.59%. Both moved in the wrong direction, though by a very small margin.

## 5. Narrow-line result

Within the padded population itself (the 350 lines the intervention targeted), things got worse on
every measure, not just empty rate: mean prediction/reference length ratio dropped from 0.526 to
0.430 (median 0.5 → 0.333); the share of severely under-produced lines (<25% of reference length)
rose from 23.7% to 43.1%. This is the opposite of the hypothesized direction.

## 6. Under-production

Corpus-wide, under-production did not improve; it worsened specifically where the intervention
applied (see §5). No evidence that added timestep margin reduced deletion-heavy behaviour.

## 7. Statistical evidence

Paired cluster bootstrap (same page/document clustering and 2,000-resample convention as the
official scorer): official-minus-padded CER 95% CI = **[-0.043pp, -0.001pp]** — i.e. official CER is
lower than padded CER in 98.15% of resamples; the interval excludes zero in the direction of padding
being (slightly) worse. Padded-vs-Lion CER 95% CI = [13.32pp, 15.27pp] (Lion still ahead in 100% of
resamples) — essentially unchanged from the official [13.30pp, 15.25pp] gap. Full results in
`padding_scores.json`.

**Reproducibility note:** 4 of the 4,277 byte-identical (unchanged) crops produced a different
prediction text between the official and padded runs (0.09%). All four are plausibly explained by
batch-composition sensitivity — the padded run's item list interleaves wider crops among the same
`batch_size=16` groups used officially, and GPU-batched inference can be composition-sensitive even
under a fixed seed. This is stated as a plausible, unconfirmed explanation, not a proven cause; it
is too small (4/4277) to affect any conclusion above, but is disclosed rather than hidden.

## 8. Gap recovery

CER improvement: **-0.02 percentage points** (i.e. a 0.02pp *regression*) against the original
14.28-point gap (32.70% − 18.42%) — **0% of the gap recovered; if anything, the padded variant is
statistically slightly further from Lion, not closer.** WER moved similarly (+0.06pp, worse). This
is not a "tiny positive effect" to round up — it is a small net-negative effect, concentrated
entirely in the targeted narrow-line population, and must be reported as such.

## 9. Hypothesis verdict: **CONTRADICTED**

Not merely unsupported: the intervention produced a measurable effect *in the opposite direction*
from the one predicted, isolated cleanly to the exact population it targeted (narrow lines: empty
rate nearly doubled, length-ratio and severe-under-production both worsened) while leaving every
wider line untouched. The softer "margin" mechanism proposed in `../GAP_ANALYSIS.md` — which was
already only weakly supported by the CTC sufficiency check (2/71 strictly forced) — does not survive
this direct causal test. A plausible (unconfirmed, not tested here) explanation: Loghi's scratch
training crops were presumably tight to the text bounding box, so a crop with wide uniform-colour
margins is an out-of-distribution input shape the model never saw in training; the added timestep
budget does not help and the unfamiliar shape appears to actively confuse the network. This is
speculation flagged as such, not a demonstrated cause.

## 10. Next step

**Do not retrain.** Do not adopt minimum-width padding as an inference-time fix — the evidence shows
it does not help and mildly hurts the exact population it targets. Deprioritize the CTC-timestep
margin hypothesis as an explanation for Loghi's empty-output/under-production behaviour on narrow
crops; the gap to Lion on this benchmark remains attributable to factors upstream of this specific
inference-time intervention (most plausibly the scratch-vs-pretrained training distinction already
noted in `../GAP_ANALYSIS.md`, e.g. the å/ä/ö weakness and Lion's pretrained ViT backbone), not to a
fixable CTC-timestep budget problem. If a narrow-crop failure mode is worth pursuing further, the
next diagnostic should test whether the *shape* mismatch itself (out-of-distribution aspect ratio),
rather than the raw timestep count, is the operative variable — e.g. by checking whether Loghi's
scratch training data ever included letterboxed/padded crops — before proposing any training change.

## Artifacts

- `padding_config.json` — preregistered architecture derivation, MIN_WIDTH derivation, padding rule.
- `derived_crop_manifest.jsonl` (sha256 `7c74a70c4095ea38a596ddfe5397248737607ffae0791f5a4771ea0285ad98ee`) — per-line original/derived hashes, dimensions, padding, byte-identity flag.
- `run_record.json` — model identity, decoding profile, backend environment, provenance, prediction file hash.
- `predictions/loghi_padded.jsonl` (sha256 `e98a76439c5371a8fbeee8df3b6eb98c21dbf4260dfcdf59bbaf048a6ee1a508`, read-only) — diagnostic Loghi predictions on the derived crops.
- `padding_scores.json` — full-corpus and changed-lines-only aggregates, bootstrap CIs, length-ratio distributions, geometry-bucket comparison, gap recovery.
- `padding_line_results.csv`, `padding_geometry_comparison.csv`, `padding_empty_outputs.csv` — per-line and per-bucket detail.
