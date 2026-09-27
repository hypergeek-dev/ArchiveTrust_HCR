# Provenance: svea-hovratt-2026-09

## Delivery

- Delivered 2026-09-27 by a colleague, as a zip created on macOS (`__MACOSX` metadata included).
- Extracted to `benchmark-data/incoming/svea-hovratt-2026-09/`.
- The outer folder was renamed from `Svea Hovrätt etc` to the delivery ID. No file inside it was
  changed.
- Tree digest from `candidate/build.json`: 1,618 files, SHA-256
  `ae80127b0504d8bc5544dd3676df97c40cd9b09bd96b47b0d4fc88408baa0894`.
- Contents: Transkribus export jobs 4502437–4502450, with PAGE-XML, ALTO and JPG files.

## What the delivery itself shows (not the provider's statement)

- The material passed through Transkribus: the folders are export jobs with Transkribus collection
  and document IDs.
- The GT is line-level, with line polygons and baselines in PAGE-XML. It is unknown whether the
  polygons were corrected by hand.
- The Transkribus sample documents are included: German, Dutch and English Handwriting, and Wiener
  Diarium.
- Two collections are named `TRAINING_VALIDATION_SET_romain_test_` and
  `TRAINING_VALIDATION_SET_Jämtlands_1702`.
- Editorial markup occurs in 41 lines of the kept collections (`[???]` 38 times, plus `[skoo?]`,
  `[…]` and `[dee???]`).

## Questions sent to the provider (Q1 and Q4 answered; Q2, Q3 and Q5 closed as unanswerable, see Status)

1. Were any of the Svea Hovrätt pages in this delivery used to train, fine-tune, validate or test
   Riksarkivet Lion, Transkribus models, or any model derived from the public Riksarkivet HTR
   datasets?
2. What exactly are the two `TRAINING_VALIDATION_SET_*` collections, and which model(s) were they
   used with?
3. What does `[???]` mean in the transcription convention? Is it literal text visible on the page,
   or an editorial marker for unreadable or uncertain text?
4. Were the German, Dutch, English and Wiener Diarium sample documents intentionally included, or
   are they standard Transkribus sample material that can be ignored?
5. Is the transcription diplomatic/exact, or has spelling, punctuation, abbreviation or
   capitalization been normalized? How are line-end hyphenation (`-` vs `¬`) and the `ß` ligature
   transcribed?

## Overlap conclusion (2026-09-27)

These points must be carried into every report.

- No identical line images.
- No meaningful exact or normalized transcription overlap. The only matches are short fragments
  like "2." and "och", under the 20-character minimum.
- No near-identical image matches (dHash threshold 4; the nearest was ≥ 8).
- Only negligible phrase overlap (0.1% of word 4-grams), consistent with common legal language.
- Same court and era, different series.
- The check covers only the Svea Hovrätt training subset (40,983 lines), not the entire historical
  training corpus.

## Answers

(record verbatim, with the date received)

- **Q1 (2026-09-27, relayed by the project owner):** "no". None of the Svea Hovrätt pages in
  this delivery were used to train, fine-tune, validate or test Lion, Transkribus models, or models
  derived from the public Riksarkivet HTR datasets.
  - The independent `overlap` check against the full training corpus has not been run, because
    the corpus is not on this machine. It remains a stated limit.
  - **Partial independent check, 2026-09-27.** The check used the Svea Hovrätt part of the
    training corpus: `benchmark-data/training-reference/svea_hovratt_lines_{1,2}.parquet`, 40,983
    lines. The index is `benchmark-data/overlap/svea-hovratt-training-index.sqlite`.
    - Result on the 6,504 candidate lines: **0 matches**. No identical image, no identical or
      normalized-identical text (lines of 20+ characters), and no near-identical image. The
      nearest image had dHash distance ≥ 8; the threshold is 4.
    - Output: `overlap-precheck/overlap.json`.
    - Shared word 4-grams: 0.1% of the benchmark's 4-grams, all standard legal phrases.
    - The training text never mentions Jämtland; Ragunda appears 12 times in the benchmark and 0
      times in the training text. Its dated text is mostly from the 1690s to the 1750s.
    - Conclusion: the benchmark is from the same court and era as the training data, but from a
      different series. No evidence of shared pages.
  - **Transcription convention differs from the training data.**
    - Line-end hyphenation: `¬` on 27% of training lines, but `-` on 11% of benchmark lines.
    - `ß` (1,077 in training, 0 in the benchmark) and `;` (1,100 vs 0) never occur in the
      benchmark.
- **Q4 (2026-09-27, answered by the project owner, not the provider):** "They are not supposed to
  be included." The German, Dutch and English Handwriting and Wiener Diarium sample documents are
  not part of the delivery's intended content. D1 already excludes them, so no decision changes.
  - The Wiener Diarium page is dated 1722; it is German-language and probably a printed newspaper.
  - English: 1807. Dutch: 1891–1893. German: no transcription or date.

- **Q2, Q3, Q5 and the audit follow-ups (2026-09-27, relayed by the project owner):**
  - The supplier is the teacher of the students who made the transcriptions in Transkribus.
  - There is no organised provenance trail between the teacher and the students. The teacher has
    no complete documented transcription guideline and no detailed record of each student's work.
  - The teacher supplied the best information available.
  - There is no organised way to get authoritative answers from the original transcribers.
  - Classification: **PROVENANCE UNAVAILABLE / CANNOT BE RESOLVED FROM SOURCE.** This is a
    documented limitation, not an open blocker.
  - No answer is inferred. The unanswerable questions are:
    - Q2: the `TRAINING_VALIDATION_SET_*` collections (excluded regardless, D2).
    - Q3: the meaning of `[???]` and unbracketed `??`/`???` (excluded, D4b).
    - Q5: diplomatic vs normalized transcription; line-end hyphenation (`-` vs `¬`); `ß`.
    - Whether the collections used different guidelines. The audit found that line-end `¬` vs
      `-`, `:` as an abbreviation mark, and `dh` spellings differ by collection.
    - Whether the Transkribus status `IN_PROGRESS` means anything.

## What the benchmark measures

Agreement with the **supplied reference transcription**: student-made Transkribus transcriptions
used as delivered, subject only to the pre-registered exclusions and transformations (D1–D5). It
does **not** measure accuracy against an independently adjudicated or formally certified diplomatic
ground truth.

## Status

Frozen as `benchmark-data/benchmark/svea-hovratt-2026-09-primary/` on 2026-09-27, before any model
run. Every model-independent fact and caveat above is also embedded in its `FROZEN.json`
(`dataset_card`, generated by `make_dataset_card.py`).

- Manifest `7c47b9eebdc26b1f…`: 6,486 lines, 140 pages, 5 documents.
- Decisions `24c815be33917b35…`, including the D5 scoring note and the D4b extension note. A copy is
  in the frozen benchmark.
- The D4b extension excludes unbracketed `??` / `???` placeholders (18 lines, all in 4502444).
- The superseded files are kept as `decisions.v1.jsonl` (`a534138b…`, candidate `76c4e38e…`) and
  `editorial_markup_review.v1.jsonl`.
- The convention audit is in `convention-audit/AUDIT.md`.
- **Caveat:** all 140 kept pages have Transkribus status `IN_PROGRESS`, not `GT` or `FINAL`.
- Answered: Q1 (no) and Q4 (samples not intended). Q2, Q3 and Q5 are unanswerable (above).
- See `docs/BENCHMARK_PROTOCOL.md` §3, "Delivery svea-hovratt-2026-09".

## Superseded freeze and v2 (2026-09-27, before any model run)

- `svea-hovratt-2026-09-primary` is **SUPERSEDED — DO NOT USE FOR PRIMARY ACCURACY BENCHMARK**. The completeness audit (`completeness-audit/`) found likely uncorrected Transkribus recognition output in the reference. The benchmark is kept unchanged; its tree hash was verified identical after the v2 freeze.
- D6 excludes 35 whole pages (4502442 pp1–31, 4502443 pp9–12; 1,859 lines). The confirmation is in `image-review/REVIEW.md`, the page signals in `d6_pages.jsonl`, and the excluded text, unchanged, in `d6_excluded_lines.jsonl`.
- `svea-hovratt-2026-09-primary-v2` was frozen at code commit e1e4428: 4,627 lines, manifest e43ee895…, `FROZEN.json` 3b6eadc6…. The diff from v1 is in `v1-v2-diff/`; the v2 audits are in `completeness-audit-primary-v2/` and `convention-audit-primary-v2/`.
