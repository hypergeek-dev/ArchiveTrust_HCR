# Independent HTR benchmark protocol: frozen Loghi vs Swedish Lion

Status: written 2026-09-27. Implemented in `src/archivetrust/htr/benchmark/`. The CLI is
`python -m archivetrust.htr.benchmark` (also `archivetrust-benchmark`).

## 1. Question

How do two frozen line recognizers transcribe the same unseen, ground-truthed historical Swedish
line images?

| Short name | Identity | Pinned by |
| --- | --- | --- |
| `loghi` | `loghi-swedish-scratch-exp2-epoch7`: scratch-trained Loghi-HTR, Experiment 2 epoch 7 `best_val` | SHA-256 of `model.keras`, `config.json` and `tokenizer.json`; container `loghi/docker.htr@sha256:414fc89a…` |
| `lion` | `riksarkivet-swedish-lion-libre`: `Riksarkivet/trocr-base-handwritten-hist-swe-2` | HF revision `aa79fcb1…`; processor `microsoft/trocr-base-handwritten@eaacaf45…` |

The pins live in `model_registry.py`. A run refuses a Loghi checkpoint whose files do not hash to the
pinned values.

This protocol does not cover training, fine-tuning, or architecture changes.

## 1a. Locked decisions (pre-registered 2026-09-27)

These decisions were approved by the project owner on 2026-09-27. At that time no benchmark dataset
had been delivered, inspected or run. They may be changed only by a dated amendment made **before**
any primary result exists. Nothing here may change because of what a result shows.

| # | Decision |
| --- | --- |
| D1 | **Primary Lion configuration: `generation_config`.** This is the model's own default generation: num_beams 4, no_repeat_ngram_size 3, length_penalty 2.0, early_stopping, max_length 256, do_sample false. Every parameter is passed explicitly. The reason is to benchmark Lion as it is meant to be used, not tuned for ArchiveTrust. `htrflow` and `greedy` exist only as labelled sensitivity runs. |
| D2 | **Primary Loghi configuration: `validated`, beam width 10** (greedy false, batch 16, seed 42). This is the configuration that actually produced val CER 0.0998 / WER 0.3273. The beam 1 in the checkpoint's `config.json` was never used for that evaluation (see §5). `greedy` exists only as a sensitivity run. |
| D3 | Failed, missing and empty predictions count as failures: they are scored as empty output and also counted separately. |
| D4 | Ground-truth characters that Loghi cannot emit stay in the reference and count as errors. |
| D5 | The primary score is on canonical GT, which differs from the delivered text only by the rules in §3. Relaxed scores (e.g. whitespace-normalized) are secondary. |
| D6 | GT decisions are recorded and frozen before any model is run on the benchmark. |
| D7 | Benchmark data is frozen by hashes (`FROZEN.json`) before any model is run. |
| D8 | Both models read the identical line-crop files for the primary comparison. |
| D9 | Other crop policies and decoding profiles are separate, labelled sensitivity runs. They never replace the primary result. |
| D10 | Bad lines are never dropped silently. Every exclusion is a recorded decision or a recorded `--exclude-unresolved` at freeze. |
| D11 | No methodology change (normalization, scoring, decoding, crops, exclusions) after primary results have been seen. |

The frozen Loghi model is backed up outside Git history, as a GitHub Release asset (see §1b).

## 1b. Frozen model backup

- **Release:** <https://github.com/hypergeek-dev/ArchiveTrust_HCR/releases/tag/model-loghi-swedish-scratch-exp2-epoch7>
  - Tag: `model-loghi-swedish-scratch-exp2-epoch7`, on commit `f0de635`, the commit that finalized
    Experiment 2 at epoch 7.
  - The repository is private.
- **Assets:** `model.keras` (368,525,198 bytes), `config.json`, `tokenizer.json` and
  `MODEL_MANIFEST.json`. The assets are byte-identical copies. On 2026-09-27 they were downloaded
  back from GitHub and re-hashed against the pins in `model_registry.py`.
- **Provenance record:** a copy of the manifest is kept at
  `docs/models/loghi-swedish-scratch-exp2-epoch7.MODEL_MANIFEST.json`. It holds the hashes, sizes,
  training run, dataset and manifest hashes, git commits, and the reference decoding.
- **Restore:**
  1. Download the three model files into one directory.
  2. Set `ARCHIVETRUST_LOGHI_CHECKPOINT` to that directory, or restore it to the path pinned in
     `model_registry.py`.
  3. Run `check`. The harness refuses any file whose hash differs.
- The model binary is never committed to normal Git history, and no training data is in the release.

## 2. Pipeline

```text
benchmark-data/incoming/<source>        delivered data. Never written by the harness.
   inspect  -> work/<source>/inspection/ inspection.md, inspection.json, findings.jsonl
   build    -> work/<source>/candidate/  lines/, manifest.jsonl, review_queue.jsonl, excluded.jsonl, build.json
   (human)  -> work/<source>/decisions.jsonl
   freeze   -> benchmark/<id>/           lines/, manifest.jsonl, decisions.jsonl, FROZEN.json (read-only, never overwritten)
   run      -> reports/<run>/predictions/{loghi,lion}.jsonl + .run.json (immutable)
   score    -> reports/<run>/scores.json, line_scores_*.jsonl, report.md
   overlap  -> overlap/<id>/overlap.json (training-contamination signals)
```

### When the external dataset arrives

1. Copy the delivery unchanged into `benchmark-data/incoming/<delivery-id>/`. Start the provenance
   record right away (`docs/DATASETS.md`, "Provenance checklist").
2. Run `inspect <delivery-id>` and read `work/<delivery-id>/inspection/inspection.md`. It covers:
   - format;
   - GT completeness;
   - image validity;
   - PAGE/ALTO structure;
   - duplicate IDs and images;
   - characters outside the Loghi charset;
   - page/line alignment;
   - whether segmentation is needed.
3. Clean up only through recorded decisions in `work/<delivery-id>/decisions.jsonl` (§3). Never
   edit `incoming/`.
4. Run `build`, and repeat it until the review queue is empty or every remaining item has a
   recorded decision.
5. Run `freeze ... --official`: the manifest, hashes and decisions become immutable.
6. Run `verify <benchmark_id>`.
7. Run `overlap <benchmark_id>` if the training-corpus index exists (§8). Otherwise record that the
   contamination check is partial.
8. Run the primary models: `run ... --model loghi` (beam 10, D2) and `run ... --model lion`
   (`generation_config`, D1).
9. Run `score`: both models, the same GT and the same crops.
10. Sensitivity runs, if wanted, go into their own `predictions/<model>@<profile>.jsonl` files.
    They never replace the primary result.
11. Write up from `report.md`. After step 8, nothing in §1a may change.

Typical session:

```powershell
.\scripts\check_benchmark.ps1                              # environment + model identity
python -m archivetrust.htr.benchmark inspect friend-2026
python -m archivetrust.htr.benchmark build friend-2026 --dataset-id friend-2026
#   ...edit benchmark-data/work/friend-2026/decisions.jsonl, rebuild with --overwrite until the review queue is empty
python -m archivetrust.htr.benchmark freeze friend-2026 friend-2026-v1 --official
python -m archivetrust.htr.benchmark overlap friend-2026-v1          # needs the HF corpus index
python -m archivetrust.htr.benchmark run friend-2026-v1 r1 --model loghi --official
python -m archivetrust.htr.benchmark run friend-2026-v1 r1 --model lion --official
python -m archivetrust.htr.benchmark score friend-2026-v1 r1 --official
```

## 3. Ground truth

- **Source GT** is kept verbatim in `gt_source`. **Canonical GT** (`gt_canonical`) is the text that
  gets scored. Every difference between them is named in `normalization_applied`.
- Protocol `gt-normalization/1` allows only these rules:
  - `unicode_nfc`: code-point composition.
  - `strip_file_line_terminator`: one trailing newline that belongs to a per-line `.txt` file.
  - `human_correction`: a recorded reviewer decision made before freeze.
- **Nothing else is applied.** That means no whitespace collapsing, no case folding, no
  punctuation, diacritic or historical-letter changes, no abbreviation expansion, and no removal of
  characters a model cannot emit.
- Anything doubtful goes to the review queue. The harness never guesses. Examples: outer
  whitespace, embedded newlines, invisible characters, mojibake, bracketed editorial markup,
  multiple `TextEquiv`s, and page-level text without line coordinates.
- GT decisions are made **before** any model is run. They must never be informed by model output.
  The freeze records the SHA-256 of the decisions file.

### Decisions (`work/<source>/decisions.jsonl`)

```json
{"target": "docA/lines/l3", "action": "exclude", "reason": "torn, illegible", "reviewer": "DJ"}
{"target": "docA/lines/l7", "action": "accept", "codes": ["gt.possible_editorial_markup"], "reason": "brackets are on the page"}
{"target": "docA/lines/l9", "action": "set_gt", "gt": "Anno 1723", "reason": "GT typo 1732"}
{"target": "file:d/scan_004.jpg", "action": "exclude", "reason": "duplicate scan"}
```

- `accept` cannot resolve these findings, because they could not be scored fairly:
  - `gt.empty`, `gt.whitespace_only`, `gt.missing`
  - `gt.outer_whitespace` (predictions are compared with outer whitespace stripped)
- To freeze while lines are still in the review queue, pass `--exclude-unresolved`. The
  exclusions are recorded, and they can bias the benchmark.

### Delivery `svea-hovratt-2026-09`: decisions approved 2026-09-27 (before any model run)

This delivery is a Transkribus export with 14 export jobs. The decisions are generated
deterministically by `benchmark-data/work/svea-hovratt-2026-09/make_decisions.py`. That script reads
only the delivery and the inspection findings.

- **D1 (collections).** The primary set keeps only the five transcribed Svea Hovrätt collections:
  export jobs 4502442–4502446.
  - Excluded by whole page:
    - the Transkribus sample documents (German, Dutch, English, Wiener Diarium);
    - the collections with no GT or too little GT (4502437, 4502438, 4502441).
  - One line in a kept collection has no transcription and is excluded.
- **D2 (held back).** Both `TRAINING_VALIDATION_SET_*` collections (4502439, 4502440) are excluded
  completely until their provenance is confirmed.
- **D3 (format).** PAGE-XML is the authoritative source format: `--adapter page_xml`, never
  auto-detection. The auto-detection scores for PAGE and ALTO were tied. The ALTO files are unused.
- **D4a (outer whitespace).** Leading and trailing whitespace is trimmed from canonical GT, and
  nothing else changes.
  - This is implemented as `set_gt` decisions with `gt == gt_source.strip()` and reason `D4a`. The
    manifest therefore labels these lines `human_correction`, even though the change is a
    deterministic trim.
  - `gt_source` keeps the delivered text.
  - Inner whitespace, spelling and punctuation are never changed.
- **D4b (editorial markup).** Lines with editorial markup such as `[???]` are excluded from the
  primary set, pending the provider's convention. They are listed in
  `work/svea-hovratt-2026-09/editorial_markup_review.jsonl`.
  - **Extended 2026-09-27**, after the deterministic convention audit and before any model run.
  - The rule: any line whose delivered GT matches the regex `\?{2,}` is an unreadable-text
    placeholder. That is two or more consecutive `?` anywhere in the line, including embedded
    forms such as `oppbur???` and `21???`.
  - A single `?` is not affected. The GT text is not modified.
  - Effect: 18 lines in 4502444 moved from the primary set to the review queue.
  - The rule is recorded as reason `D4b (extended …)` on each decision and as a comment line in
    `decisions.jsonl`. The superseded decisions are kept as `decisions.v1.jsonl`.
  - The audit script is `scripts/benchmark_convention_audit.py`. Its outputs are in
    `work/svea-hovratt-2026-09/convention-audit/`.
- **Kept as supplied:**
  - characters outside Loghi's charset (they are reported as charset coverage);
  - identical transcriptions on different line images;
  - crops clamped to the page raster (flagged `crop_clamped` in the line's `source_metadata`).
- **D5 (scoring).** The line-end hyphen sensitivity score (§6) is pre-registered. Raw scoring stays
  primary. It is recorded as a comment line in `decisions.jsonl`, and `FROZEN.json` pins it.
- **D6 (probable uncorrected recognition output), approved 2026-09-27, after the v1 freeze and
  before any model run.** 35 whole pages are excluded from primary: export job 4502442 pages 1–31
  and 4502443 pages 9–12 (1,859 lines).
  - Why: every page was prefilled by a Transkribus HTR model, so a line with text is not
    necessarily a corrected line. The completeness audit (`scripts/benchmark_completeness_audit.py`)
    found a block of pages whose signals point to recognition output that nobody corrected.
  - Rule (page level, from the audit's pre-declared signals; no single signal decides):
    - the page's source document has at least one page whose last save followed the previous save
      in that document within ≤ 1.0 s per line **and** that carries no student convention (0
      line-final `-`, 0 lines with `[` or `??`);
    - in such a document, a page is excluded if it carries no student convention, or if it mixes
      line-final `-` and `¬`.
  - Neither `IN_PROGRESS`, save time, `¬` nor text oddness is used alone.
  - Confirmation: a seeded, stratified image review (`scripts/benchmark_image_review_sample.py`;
    15 pages, 46 lines, compared with images only, never with model output):
    - suspect lines: 11 likely uncorrected, 18 ambiguous, 2 corrected (the latter on 4502443 p9,
      above its first `¬`);
    - control lines from all four other collections: 15 corrected, none otherwise.
    - The review is in `work/svea-hovratt-2026-09/image-review/`.
  - `make_decisions.py` derives the page set from the audit output, pinned by SHA-256, and refuses
    to run if the result differs from the reviewed set.
  - 4502443 p9 is corrected at the top and uncorrected from its first `¬`. No metadata marks the
    boundary, so the **whole page** is excluded. There are no partial pages.
  - Lines already excluded by D1–D4b keep their earlier decision. No GT text is changed. The
    excluded lines are preserved unchanged in `d6_excluded_lines.jsonl`, and the per-page signals
    in `d6_pages.jsonl`.
  - The decisions frozen with v1 are kept as `decisions.primary-v1.jsonl`. The v2 decisions are
    those same 1,210 decisions plus the 35 D6 page decisions and a D6 comment line.
- **Caveat: Transkribus status.** Every kept page carries Transkribus page status `IN_PROGRESS`;
  none is `GT` or `FINAL`. The status has not been changed. After D6, the completeness audit finds
  no material deterministic evidence of incomplete transcription on the retained pages. That is an
  inference, not a certification. Every report on this benchmark states this caveat.
- **Provenance limitation (2026-09-27).** The transcriptions were made by students in Transkribus.
  - The supplying teacher has no complete transcription guideline and no detailed provenance trail
    for the students' work. There is no organised way to get authoritative answers from the
    transcribers.
  - The questions about the conventions are therefore closed, unanswered, as **PROVENANCE
    UNAVAILABLE / CANNOT BE RESOLVED FROM SOURCE**. This covers `[???]` and unbracketed `??`/`???`,
    diplomatic vs normalized transcription, line-end hyphenation, `ß`, per-collection guidelines,
    and the meaning of `IN_PROGRESS`.
  - No answers are inferred.
- **What the benchmark measures.** Agreement with the supplied reference transcription, used as
  delivered, subject only to D1–D6. It is not accuracy against an independently adjudicated or
  certified diplomatic ground truth.
- **Caveat: transcription convention.** The student guidelines are unavailable. The convention
  audit shows differences between collections.
- **Caveat: charset.** Characters outside Loghi's 124-character output set stay in the benchmark
  (v2: `¼` ×1; v1 also had `æ` ×2). They count as ordinary errors when Loghi cannot emit them.
- **Spot-check lines.** Some lines stay in the primary set unchanged. These are the audit's
  convention examples: single `?` and unusual line-end hyphenation. After D6, no reference line
  ends in `¬`. They are not reviewed against model predictions before the primary run.
- **v1 `svea-hovratt-2026-09-primary`: SUPERSEDED — DO NOT USE FOR PRIMARY ACCURACY BENCHMARK.**
  - It was frozen 2026-09-27 before any model run (`--official`; code commit `779f4c3`, clean
    tree): manifest `7c47b9ee…`, decisions `24c815be…`, `FROZEN.json` `ffd2a22f…`. It holds 5
    documents, 140 pages, 6,486 lines, 213,824 characters and 37,545 words.
  - Why it is superseded: the completeness audit identified likely uncorrected recognition output
    in the reference set before any model inference (see D6).
  - It is kept unchanged as a historical snapshot. It is not deleted and not rewritten, and no model
    is to be run on it.

## 4. Line images and segmentation

- **Supplied line images** are copied byte-identical. Other formats are converted losslessly to
  PNG and the conversion is recorded. Images with non-opaque alpha, 16-bit, CMYK or multiple
  frames go to review, because the two model stacks would decode them differently.
- **Pages with line polygons** (PAGE, ALTO) use crop policy `bbox_v1`:
  - the polygon's bounding box is cut from the full-resolution page decoded as RGB;
  - the box is clamped to the raster. A line whose polygon runs past the raster (the same test as
    `layout.polygon_out_of_bounds`) gets `crop_clamped: true` in `source_metadata`;
  - PNG output uses `optimize=False, compress_level=6`;
  - the geometry (rounding and clamping) is the same as `florence2_line_detector.crop_lines`. On
    2026-09-27 all 2,678 dataset-rgb dry-run crops were pixel-identical to that run's crops; only
    the PNG byte encoding differs.
- Line image files live at `lines/<document>/<page>/<line>.png`. Any path component longer than
  48 characters is shortened, with a hash suffix, to stay under the Windows path limit. IDs in the
  manifest are never shortened.
- `polygon_mask_v1` (outside-polygon pixels set to white) is only for a separate, labelled
  sensitivity benchmark. One manifest never mixes crop policies.
- **EXIF orientation is never applied silently.** A non-identity orientation tag, or a declared page
  size that is swapped relative to the raster, goes to review or blocks the build.
- **Pages without line coordinates** need segmentation, and page-level GT cannot be aligned to lines
  automatically. The harness reports this rather than attempting it. Segmentation quality is a
  separate question from recognition, and it is not scored here.
- **Both models read exactly the same files** (`benchmark/<id>/lines/`). The files are re-verified
  against the manifest hashes before and after every run.

### Mechanical dry run without ground truth

`dryrun-build <pages_dir> <segmentation.jsonl> <dryrun_id>` and
`dryrun-run <dryrun_id> --model M --limit N` exercise every stage of the pipeline except GT
alignment and scoring:

- page discovery and probing;
- imported line segmentation;
- `bbox_v1` crops;
- a hashed manifest;
- smoke inference through the real backends, with the primary decoding profiles.

Everything the dry run writes is labelled `NO_GROUND_TRUTH / NOT_AN_ACCURACY_BENCHMARK`. It has no
GT fields and cannot be scored. Its output shows that the machinery works, not how accurate either
model is.

## 5. Decoding (decide and record before seeing any results)

| Model | Profile | Parameters | Why |
| --- | --- | --- | --- |
| loghi | `validated` (**primary, locked D2**) | beam_width 10, greedy false, batch 16, seed 42 | Produced the reference val CER 0.0998. Note: the checkpoint's `config.json` says beam 1, but Loghi reads that file only with `--config_file`, which the evaluation did not pass. |
| loghi | `greedy` | beam_width 1, `--greedy` | Diagnostic |
| lion | `generation_config` (**primary, locked D1**) | num_beams 4, no_repeat_ngram_size 3, length_penalty 2.0, early_stopping, max_length 256 | What the model card's `model.generate(pixel_values)` does, spelled out |
| lion | `htrflow` | num_beams 1, no_repeat_ngram_size 3, max_length 256 | What the existing ArchiveTrust Lion adapter ran (`num_beams=1` with inherited generation_config) |
| lion | `greedy` | num_beams 1, no_repeat_ngram_size 0, max_length 256 | Diagnostic without the inherited repetition ban |

- The primary comparison is each model at its default profile.
- Other profiles are sensitivity runs. They are written as `predictions/<model>@<profile>.jsonl`
  and are never substituted after the fact.
- Every Lion parameter is passed explicitly, so nothing is inherited silently from
  `generation_config.json`.

## 6. Scoring (`scoring.py`, `SCORING_VERSION = 2`)

- **Primary score.** Reference = `gt_canonical`. Hypothesis = prediction after NFC and removal of
  outer whitespace, applied identically to both models. The number of predictions that had outer
  whitespace stripped is recorded per run.
- **Corpus CER** = Σ character edits / Σ reference characters over **all** benchmark lines. **WER**
  is computed the same way over whitespace-split words. Totals and substitutions / insertions /
  deletions are reported.
- **Failed, missing and empty predictions** are scored as empty output, which makes every reference
  character a deletion. They are also counted separately. Loghi silently skips undecodable images;
  those lines show up as `missing`.
- A prediction file is scorable only if it was made on the frozen manifest's SHA-256 and covers
  every line exactly once with matching image hashes. Partial (`--limit`) runs cannot be scored.
- **Secondary scores**: whitespace-normalized CER/WER (`evaluation.metrics.normalize_text`), exact
  line accuracy, and a macro mean of per-line CER.
- **Sensitivity score** `line_end_hyphen_harmonized` (added in version 2, pre-registered 2026-09-27
  before any model run on a GT benchmark):
  - It uses the raw texts, except that a **line-final** `¬` counts as `-` in both reference and
    prediction.
  - Nothing else changes: internal hyphens, `ß`, `;` and all other characters stay as they are.
  - Reason: the training data marks line-end hyphenation with `¬` (27% of lines), while the
    `svea-hovratt-2026-09` GT uses `-` (11% of lines).
  - The score is reported next to raw CER/WER in every report, and never replaces the primary
    score.
  - `FROZEN.json` pins the scoring version and this rule (`scoring`).
- **Confidence intervals** use a 95% percentile cluster bootstrap: 2000 resamples, seed 20260927.
  - The cluster is the document if there are at least 10 documents, else the page, else the line.
    Line-level clusters are flagged as too narrow.
  - The model difference uses a **paired** bootstrap (the same clusters for both models).
  - An interval that contains 0 means the benchmark does not separate the models.
- **Breakdowns** are given per collection and per document.
- **Pairwise analysis**:
  - line categories: both correct, only one correct, both wrong identically, one better;
  - catastrophic lines: failed, missing, or line CER ≥ 50%;
  - error tags: spacing, case, diacritic, punctuation, digit, special/historical character,
    abbreviation word, letter;
  - top confusions;
  - a ranked disagreement sample.

## 7. Integrity and provenance

Each stage records its own provenance:

| Record | Contents |
| --- | --- |
| `FROZEN.json` | Manifest SHA-256, build record hash, decisions hash (the decisions file is copied beside it and re-hashed by `verify`), delivery tree hash, normalization protocol, scoring rules, crop policy, line/character/word counts, exclusion counts, code commit and dirty state, and an optional `dataset_card` (`freeze --dataset-card`) with model-independent provenance status and caveats |
| `*.run.json` | Model identity and decoding, backend environment (checkpoint hashes, container reference and argv; or torch/transformers versions, device and GPU), predictions SHA-256 |
| `scores.json` | Scoring and metrics versions |

Every stage also records the git commit, the dirty state of `src/ scripts/ pyproject.toml`, the
Python and package versions, and UTC timestamps.

`--official` refuses a dirty tree. Anything else is labelled **UNOFFICIAL** in the report. A report
is OFFICIAL only if the freeze, both runs and scoring were all official.

## 8. Contamination

- Both models were trained on Riksarkivet's public HF line collections: 11 collections, 565,146
  rows. The Loghi model was trained on a subset whose split manifests were deleted; only their
  hashes survive. The whole corpus is therefore the conservative "possibly seen" set.
- To check, run `overlap-index <parquet_dir>` once, then `overlap <benchmark_id>`. The signals are:
  - identical image bytes;
  - near-identical image (dHash ≤ 4 bits) with the same normalized text;
  - identical exact or normalized transcription, for lines of 20 or more characters only;
  - near-identical image alone. This is a weak signal: it is reported, but nothing is concluded
    from it.
- **No match is not proof of no overlap.** Re-cropped, re-scanned or re-transcribed material
  evades all of these checks.
- Freeze and score before any adaptation to the new data. Results on this benchmark are
  *pre-adaptation*. Benchmark data must never enter training.

## 9. Known limits to state in any write-up

- The Loghi val CER (0.0998) is in-distribution and is not comparable to an external benchmark.
- Loghi's charset has 124 characters. Reference characters outside it are unavoidable errors for
  Loghi. The rate is reported, and nothing is removed.
- Loghi durations are batch-amortized; Lion's are per line. Only Loghi gives a per-line
  confidence.
- The training corpus crops (the HF line images) define what "line image" meant for both models.
  If the new data's crops differ systematically (tighter or looser, masked or not), both models are
  affected, possibly unequally. Run `polygon_mask_v1` as a sensitivity benchmark when the source
  has polygons.
