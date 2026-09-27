# Svea Hovrätt HTR benchmark — primary run

Run ID: `svea-hovratt-v2-primary-20260927T102311Z`
Status: **OFFICIAL.** Benchmark frozen before any model output was seen; predictions and scores were
computed after that freeze and are immutable evidence.

## Benchmark identity

- Benchmark: `svea-hovratt-2026-09-primary-v2` — 4 documents, 105 pages, 4,627 lines, 166,201
  reference characters, 28,872 reference words.
- `manifest.jsonl` SHA-256 `e43ee895851036f47e92801ce9aa5ca102c22a1f67d8c15225d41261c5d5b8e1`
- `decisions.jsonl` SHA-256 `c32103e3ed7c450cdad0b7e81a0223552b31f5c1bbb0feeb83dbd22d8c5d4206`
- `FROZEN.json` SHA-256 `3b6eadc6e408eb434c8d2f007cb1ef1e9f1efb41b990799873b1bb6f9c835917`
- Frozen 2026-09-27T09:35:38Z, before any model run.
- The prior freeze `svea-hovratt-2026-09-primary` (6,486 lines) is **superseded — do not use for
  primary accuracy**: the completeness audit found evidence of uncorrected Transkribus recognition
  output in a subset of its reference; D6 removed that subset in v2. Not used in this run.
- Provenance: source is a colleague's Transkribus export of a student-transcribed 18th-century Swedish
  court record collection; every page carries Transkribus status `IN_PROGRESS` (none `GT`/`FINAL`).
  The benchmark measures agreement with the supplied, D6-cleaned reference transcription, not a
  certified ground truth.
- Overlap check (`scripts` `overlap` command) against the known Svea Hovrätt HF training subset:
  **0 of 4,627 lines** show a strong contamination signal. Caveat: no match is not proof of no
  overlap — this only covers the known training subset, not every source either model may have seen.

## Models

### ArchiveTrust Loghi
- `loghi-swedish-scratch-exp2-epoch7` — scratch-trained (no pretraining), Experiment 2, epoch 7
  (`best_val`), 124-character set.
- Decoding profile `validated`: beam width 10, greedy false, batch size 16, seed 42.
- Checkpoint file hashes verified against the pinned identity (`model.keras`, `config.json`,
  `tokenizer.json`).
- Container: `loghi/docker.htr:latest@sha256:414fc89ac574a61fd745836ad9852842759bf3cf7046e4549ae96315ff9132e8`.

### Riksarkivet Lion
- `riksarkivet-swedish-lion-libre` (TrOCR), HF revision `aa79fcb1850bf3155ebc442570d6c6bfc0ac8100`;
  processor `microsoft/trocr-base-handwritten` revision `eaacaf452b06415df8f10bb6fad3a4c11e609406`.
- Decoding profile `generation_config` (the pinned primary profile): num_beams=4,
  no_repeat_ngram_size=3, length_penalty=2.0, early_stopping=true, max_length=256, do_sample=false.
  `htrflow` and `greedy` were **not** run — those are sensitivity configurations for a separate task.
- Device: CUDA, NVIDIA GeForce RTX 3070.

## Primary results (raw scoring; corpus-level, every line counted, failures/empties scored as empty)

| Metric | ArchiveTrust Loghi | Lion |
|---|---:|---:|
| CER | 32.70% | 18.42% |
| WER | 67.53% | 43.63% |
| Substitutions (chars) | 21,080 | 17,926 |
| Insertions (chars) | 762 | 3,220 |
| Deletions (chars) | 32,506 | 9,468 |
| Reference characters | 166,201 | 166,201 |
| Reference words | 28,872 | 28,872 |
| Failed lines | 0 | 0 |
| Empty outputs | 71 | 0 |
| Exact-line accuracy | 0.89% | 7.33% |

Lion has substantially lower raw CER and WER than Loghi on this benchmark.

## Pairwise uncertainty (preregistered paired cluster bootstrap)

- Resampling unit: page (105 clusters), 2,000 resamples, seed 20260927, 95% percentile interval.
  Pairing preserved per line/page in every resample (both models scored on the identical 4,627 lines).
- Loghi CER 95% CI: [31.16%, 34.25%]. Lion CER 95% CI: [17.29%, 19.65%].
- Loghi WER 95% CI: [65.95%, 69.07%]. Lion WER 95% CI: [42.25%, 45.07%].
- **Paired difference (Lion − Loghi)**: CER 95% CI [−15.25 pp, −13.30 pp]; WER 95% CI
  [−25.28 pp, −22.60 pp]. Neither interval crosses zero — in all 2,000 resamples Lion's CER and WER
  were lower than Loghi's (`share_of_resamples_a_lower = 1.0`). The benchmark clearly separates the
  two models at this size, in Lion's favor.

## D5 sensitivity (secondary)

**SECONDARY SENSITIVITY ANALYSIS — NOT PRIMARY RESULT.** Line-final `¬`/`-` treated as equivalent;
nothing else changed.

| Metric | Raw | D5 sensitivity |
|---|---:|---:|
| Loghi CER | 32.70% | 32.56% |
| Lion CER | 18.42% | 18.23% |
| Loghi WER | 67.53% | 67.29% |
| Lion WER | 43.63% | 42.98% |

D5 changes almost nothing here: after D6 removed the pages that used `¬` as an uncorrected-model-output
marker, very few reference lines still end in `¬`/`-` at all, so the sensitivity rule has little left
to affect. This is expected and reported honestly rather than adjusted for.

## Per-document results

| Document (export job) | Lines | Chars | Words | Loghi CER | Lion CER | Loghi WER | Lion WER | Pairwise ΔCER (Lion−Loghi) | Pairwise ΔWER |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4502443 | 397 | 18,529 | 3,086 | 39.76% | 18.82% | 73.56% | 45.50% | −20.95 pp | −28.06 pp |
| 4502444 | 776 | 37,812 | 6,227 | 30.83% | 14.12% | 69.33% | 40.32% | −16.72 pp | −29.00 pp |
| 4502445 | 2,137 | 68,549 | 12,279 | 35.83% | 23.58% | 68.24% | 48.44% | −12.25 pp | −19.80 pp |
| 4502446 | 1,317 | 41,311 | 7,280 | 26.05% | 13.62% | 62.24% | 37.55% | −12.43 pp | −24.68 pp |

Lion is ahead of Loghi on every retained document, by comparable margins. Descriptive only — the
benchmark's unit of resampling and its verdict stay at the page/corpus level defined above; these
per-document numbers do not redefine the benchmark.

## Error analysis (after primary results, hashes and bootstrap were locked)

Line-level categories (4,627 lines):

| Category | Lines |
|---|---:|
| Both correct | 25 |
| Both wrong, Lion better | 3,576 |
| Both wrong, Loghi better | 371 |
| Both wrong, equal edits | 287 |
| Both wrong, identical output | 38 |
| Only Lion correct | 314 |
| Only Loghi correct | 16 |

Catastrophic lines (failed/missing, or line CER ≥ 50%): Lion 481, Loghi 982, both 442.

Character-edit types: Loghi has roughly double Lion's edits in every category (letter, spacing,
abbreviation-word, digit, punctuation, diacritic, special-character); both models' single most common
edit is deleting a character that should have been kept (` →∅`, `e→∅`, `a→∅`, `s→∅`…), i.e. under-
production relative to the reference — more pronounced for Loghi. Full confusion tables and the 30
largest per-category disagreements (both directions) are in `report.md` and `scores.json` alongside
this report; they are not reproduced in full here.

No likely reference-transcription (GT) errors beyond the ones already known from the completeness
audit and D6 cleanup were specifically hunted for in this pass; none of this analysis changed or will
change the frozen benchmark.

### The two known sanity-check lines (from the earlier completeness audit / image review)

- `4502442 p30 r2l3` (image read as "Carl Grips"; supplied GT said "Carl Grims") — **confirmed absent
  from v2**: document 4502442 is excluded whole by D6. Not scored in this run.
- `4502443 p10 r2l29` (image read as "Häradzhöfdingen"; supplied GT said "Hindran gaff ingen") —
  **confirmed absent from v2**: 4502443 pages 9–12 are excluded by D6. Not scored in this run.

Verified directly against the v2 manifest: 0 lines reference document 4502442, and 0 lines reference
4502443 pages 9–12. Both lines remain evidence supporting the earlier D6 cleanup decision only; they
were never part of this benchmark run and were not used to alter it.

## Dataset caveats

- The reference transcription is student-created in Transkribus, not a professional/certified
  transcription; full student transcription guidelines are unavailable.
- Every source page still carries Transkribus status `IN_PROGRESS`; none is `GT`/`FINAL`.
- The completeness audit found a subset of pages (`4502442` whole, `4502443` pp. 9–12; 1,859 lines)
  that were likely uncorrected Transkribus recognition output; D6 removed them **before this model
  run**, without changing any surviving GT text.
- This benchmark measures model agreement with the supplied, D6-cleaned reference transcription — not
  agreement with a certified ground truth.
- The overlap check covers only the known Svea Hovrätt HF training subset; it does not, and cannot,
  rule out every possible source either model was ever trained on.

## Integrity

- Code commit: `8a2902a8c8614d29b0252ab1858a8c1c7338c5bd` (clean tree; checked paths `src`, `scripts`,
  `pyproject.toml`).
- Benchmark hashes: manifest `e43ee895…`, decisions `c32103e3…`, `FROZEN.json` `3b6eadc6…`.
- Loghi checkpoint hashes verified (`model.keras`, `config.json`, `tokenizer.json`) against
  `model_registry.py`'s pinned identity.
- Prediction hashes: `predictions/loghi.jsonl` `9aed0d6e2d3bcd73c116d813af15e82e191da04f39f54b094eb8ce1098feadb4`;
  `predictions/lion.jsonl` `aef14754070f8ff037818a3689ef75f9966df7e2be60e1c8193975826e34450c`.
  Both files are read-only on disk.
- Docker image digest: `loghi/docker.htr:latest@sha256:414fc89ac574a61fd745836ad9852842759bf3cf7046e4549ae96315ff9132e8`.
- Environment: Python 3.13.15, torch 2.13.0+cu130, transformers 4.49.0, Windows-11-10.0.26200.
- GPU: NVIDIA GeForce RTX 3070 (both runs).
- Timestamps: Loghi 2026-09-27T10:23:37Z–10:25:05Z; Lion 2026-09-27T10:25:32Z–10:54:32Z; scored
  2026-09-27T10:55:33Z.
- Machine-readable summary: `results.json` in this same directory. Full detail: `scores.json`,
  `report.md`, `line_scores_loghi.jsonl`, `line_scores_lion.jsonl`.

## Final conclusion

On the frozen `svea-hovratt-2026-09-primary-v2` benchmark, under each model's preregistered primary
decoding configuration (Loghi beam width 10; Lion `generation_config`, 4 beams):

**Lion had lower raw CER and WER than ArchiveTrust Loghi.** The paired cluster bootstrap (105-page
clusters, 2,000 resamples) puts the CER difference at −15.25 to −13.30 percentage points and the WER
difference at −25.28 to −22.60 percentage points (Lion minus Loghi), with neither interval crossing
zero and Lion ahead in every resample. The result held in the same direction, by comparable margins,
on all four retained documents.

This claim is scoped to performance on the frozen `svea-hovratt-2026-09-primary-v2` benchmark under
these preregistered primary configurations. It does not generalize beyond this dataset, and it is not
a claim that one model is universally better at Swedish historical HTR — the reference transcription
is a cleaned but still student-produced, non-certified transcription, and the models differ in more
than one respect (architecture, decoding, training data) that this benchmark does not isolate.
