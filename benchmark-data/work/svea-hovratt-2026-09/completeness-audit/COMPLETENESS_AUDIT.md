# Completeness audit: svea-hovratt-2026-09 (primary candidate)

Date: 2026-09-27. Read-only audit. No model was run, and no GT, decisions or eligibility rules
were changed.

Question: all 140 retained pages carry Transkribus status `IN_PROGRESS`. Is that real
incompleteness, or stale workflow metadata?

## Status of the candidate

The audit request says the candidate is not yet frozen. In fact it is already frozen as
`benchmark/svea-hovratt-2026-09-primary`:

- `FROZEN.json` `ffd2a22f…`;
- manifest `7c47b9ee…`;
- decisions `24c815be…`;
- code commit 779f4c3.

The audit reads that frozen benchmark. It did not modify it.

## Reproduction

```
.venv/Scripts/python.exe scripts/benchmark_completeness_audit.py svea-hovratt-2026-09 \
    svea-hovratt-2026-09-primary --training-parquet-dir benchmark-data/training-reference --overwrite
```

- The script was run twice. All 13 deterministic outputs were byte-identical. `run_record.json`
  holds timestamps and is excluded from that comparison.
- `run_record.json` records:
  - every threshold constant;
  - the input hashes (FROZEN.json, manifest, decisions, excluded set, the script itself);
  - the training reference files;
  - the git commit (b78b36b) and dirty paths. The script is still uncommitted.
- Personal metadata is redacted:
  - Transkribus `userId` is written only as "present";
  - Creator strings are reduced to tool names;
  - the run aborts if any output contains an email address.

Labels: **DETERMINISTIC** means an exact rule over the data. **HEURISTIC** means an explicit
threshold, with its interpretation open. **MANUAL SPOT-CHECK ONLY** means a human reading, not a
measurement.

## Headline

Structurally, the dataset is essentially complete:

- every source line has text except one;
- no page ends in empty lines;
- no region is empty;
- PAGE and ALTO agree.

But the Creator metadata shows that every page was **prefilled by a Transkribus HTR model**. So a
line with text is not necessarily a corrected line. Coverage checks cannot see the difference
between a corrected line and an uncorrected one.

Three independent signals point to a block of pages that were **not corrected**:

- save timing (HEURISTIC);
- the absence of any student conventions (HEURISTIC);
- a text-only reading (MANUAL).

That block is all of 4502442 and part of 4502443. The incompleteness there is not missing text.
It is recognition output standing in as reference text.

## Findings

Percentages are relative to the 6,546 source TextLines on the 140 retained pages, or to the 6,486
retained lines, as stated.

### 1–3. Coverage, unfinished tails, internal gaps

| ID | Label | Rule | Result | File |
| --- | --- | --- | --- | --- |
| S0 | DETERMINISTIC | every source TextLine of a retained page is retained, excluded by a recorded decision, or unaccounted | 0 unaccounted | `page_completeness.csv` |
| C1 | DETERMINISTIC | source TextLines carrying non-empty GT | 6,545 / 6,546 (99.985%) | `page_completeness.csv` |
| C2 | DETERMINISTIC | retained pages where every source line carries GT | 139 / 140 (99.29%) | `page_completeness.csv` |
| C3 | DETERMINISTIC | pages whose reading order ends in ≥ 2 lines without GT (`MIN_TAIL_LINES=2`) | 0 | `unfinished_tails.csv` |
| C4 | DETERMINISTIC | runs of lines without GT between lines with GT | 1 run of length 1: 4502442 `0029_DSC_0577` `r3l25` (region r3, position 50) | `internal_gaps.csv` |
| C5 | HEURISTIC | GT lines with ≤ 3 non-space characters (`SHORT_MAX_CHARS=3`) | 416 (6.4%); 239 of them in 4502442 (14% of its lines) | `page_completeness.csv` |

- Protocol exclusions (D1/D2/D4b) are counted separately from missing GT. They are not evidence of
  incompleteness.
- Limit: Transkribus HTR filled every line, so C1–C4 cannot detect uncorrected text. They only
  show that no line was left empty.

### 4. Regions

| ID | Label | Rule | Result | File |
| --- | --- | --- | --- | --- |
| R1 | DETERMINISTIC | regions with no lines, or no line carrying GT | 0 of 309 | `region_coverage.csv` |
| R2 | DETERMINISTIC | region type labels (`@type` / structure tag) | none present on any of the 309 regions | `region_coverage.csv` |
| R3 | HEURISTIC | regions narrower than 0.25 of the page width | 43 regions, 102/102 lines with GT (mostly 4502442) | `region_coverage.csv` |
| R4 | HEURISTIC | unsegmented space below a region's last line ≥ 3.0 × the median line height | 0 | `region_coverage.csv` |
| T5 | DETERMINISTIC | region TextEquiv differs from its lines joined by newlines (after `\r\n`→`\n`) | 0 of 309 | `region_coverage.csv` |

- Narrow regions (possible marginalia or headings) are transcribed; there is no sign of a
  "main body only" convention.
- Limit: text with no line geometry at all is only probed by R4, which is heuristic.

### 5. Crop vs text-length outliers

| ID | Label | Rule | Result | File |
| --- | --- | --- | --- | --- |
| O1 | HEURISTIC | robust z (MAD) of log(non-space chars / crop aspect) ≥ 4.0 within the collection, or aspect ≥ 6.0 with ≤ 2 chars | 176 retained lines (2.7%); 114 in 4502442 | `crop_text_outliers.csv` |

- Almost all outliers are "text short for crop". Examples: 4502444 `0009_DSC_0442`
  `line_1638890141661_2347` has GT `13.` on a 655 × 79 crop; 4502445 `0025_DSC_0403` has `un-`.
- Many look like page numbers, section marks or word fragments on wide segmentation lines, which
  is plausible for this material.
- 4502442 stands out: mean 24.3 characters per line against 31–49 for the other collections. That
  is consistent with segmentation-driven HTR output (see T2), but it is not proof.
- No line was excluded because of O1.

### 6–7. Placeholders and truncation

| ID | Label | Rule | Result | File |
| --- | --- | --- | --- | --- |
| P01–P13, P16 | DETERMINISTIC | repeated `?`, `..`, `…`, `_`, `--`, TODO, `xxx`, `<…>`, `[`, empty brackets, XML escape debris, U+FFFD, U+FFFC, dangling open bracket at line end | 0 in retained lines (100 hits, all in lines already excluded by D4b) | `placeholder_inventory.csv` |
| P14 | HEURISTIC | single `?` | 6 retained lines | `placeholder_inventory.csv` |
| P15 | HEURISTIC | double space | 23 retained lines | `placeholder_inventory.csv` |
| T6 | DETERMINISTIC | TextLines with more than one TextEquiv | 0 | `completeness_findings.json` |

- A single `?` is kept under D4b as approved.
- Double spaces collapse under the scorer's whitespace normalisation.
- No line looks cut off: no alternative text versions exist, and there are no dangling brackets.

### 8. PAGE vs ALTO

| ID | Label | Rule | Result | File |
| --- | --- | --- | --- | --- |
| X1 | DETERMINISTIC | page IDs, line IDs, region and line counts, or non-space characters differ by > 2% (`ALTO_CHAR_DIFF_FRAC`), or ALTO missing | 0 of 140 | `page_alto_comparison.csv` |
| X2 | DETERMINISTIC | whitespace-collapsed line text differs | 0 lines | `page_alto_comparison.csv` |

- Limit: ALTO was exported from the same Transkribus transcript version at the same time. This
  shows the export is consistent. It cannot reveal work that was never done.
- PAGE remains the authoritative source.

### 9. Versions and duplicates

| ID | Label | Rule | Result | File |
| --- | --- | --- | --- | --- |
| V1 | DETERMINISTIC | identical page-image SHA-256, Transkribus pageId or imageId across all 14 export jobs | 10 conflicts = 5 page images × (image hash + imageId) | `version_conflicts.csv` |

- The 5 retained pages duplicated in the held-back export jobs are:
  - 4502443 `0007_DSC_0455`;
  - 4502444 `0007_DSC_0440`;
  - 4502445 `0022_DSC_0399`;
  - 4502446 `0028_DSC_0364`;
  - 4502445 `0040_DSC_0422`.
- The copies sit in 4502439 `TRAINING_VALIDATION_SET_romain_test_` and 4502440
  `TRAINING_VALIDATION_SET_Jämtlands_1702`.
- In each pair the text differs (similarity 0.82–0.93). The held-back copy:
  - was saved later (2021-12-09);
  - has the same creation time and line IDs;
  - lists a newer recognition model in its Creator: `romain test` (htr_id 38615), or PyLaia for
    4502440.
- Interpretation (HEURISTIC): the held-back copies are later model runs over validation pages. The
  benchmark uses the earlier, human-edited version, **not an older or incomplete one**.
- These five pages are also recorded as a validation set for a Transkribus-internal model. That
  model is neither Lion nor Loghi, so this is a provenance note, not a contamination finding.

### 10. Recognition provenance and status correlation

`IN_PROGRESS` is constant across all 140 pages, so the status carries no information on its own.
It was tested against the measures below.

| ID | Label | Rule | Result | File |
| --- | --- | --- | --- | --- |
| T1 | DETERMINISTIC | PAGE Creator lists an HTR model and layout-analysis tools | 140 / 140: `Jaemtlands_domsagas_M1` (HTR), `LA73_249_0mod360`, `B2PSeamMultiOriented` | `recognition_provenance.csv` |
| T2 | HEURISTIC | page's last save followed the previous save in the same document within ≤ 1.0 s per line (`BATCH_MAX_SEC_PER_LINE`) **and** the page has no student convention (0 line-final `-`, 0 lines with `[` or `??`) | **29 pages, 1,539 retained lines (23.7%)** | `recognition_provenance.csv` |
| T3 | MANUAL SPOT-CHECK ONLY | exactly one of the two T2 signals | 16 pages | `recognition_provenance.csv` |
| T4 | HEURISTIC | page mixes line-final `-` and `¬` | 1 page: 4502443 `0009_DSC_0457`, sequence `---¬¬¬` | `recognition_provenance.csv` |
| T7 | HEURISTIC | out-of-vocabulary rate against the 20,114-word training-reference vocabulary | see below | `recognition_provenance.csv` |

T2 in detail:

- **4502442 pages 5–31** (27 pages, 1,458 lines):
  - all were saved in a single run on 2021-12-08 between 16:32 and 16:40, roughly 17–24 s apart
    (0.33–0.40 s per line);
  - the run finished within 41 minutes of the document's creation;
  - no student marks; line ends use `¬` only.

  That pattern fits an HTR batch job. It does not fit a student correcting pages.
- **4502443 pages 11–12** (81 lines).

T3 in detail:

- 4502442 pp. 1–4:
  - p. 4 is the first save in the batch run;
  - pp. 1–3 were saved the next morning at 9–18 minute intervals, but have no student marks and
    `¬` only.
- 4502443 p. 10: 1.18 s per line; no student convention.
- Fast saves that do show student edits:
  - 4502443 pp. 4, 5, 6, 8;
  - 4502444 pp. 4, 5, 8, 10, 13;
  - 4502446 pp. 2, 29.

T4: the page has corrected-style `-` at the top and `¬` from about line 44 on. That is consistent
with correction having stopped partway down the page.

T7 (HEURISTIC), median OOV rate by collection and group:

| Collection | Group | Median OOV |
| --- | --- | --- |
| 4502442 | flagged pages | 0.340 |
| 4502442 | ambiguous pages | 0.400 |
| 4502443 | flagged pages | 0.365 |
| 4502443 | edited pages | 0.312 |
| 4502444 | all | ~0.31 |
| 4502445 | all | 0.230 |
| 4502446 | all | ~0.229 |

- OOV is confounded by the year and the spelling of each volume. It is weak support, not a test.
- An ad-hoc comparison of the upper and lower halves of pages did not separate the pages reliably.
  It is not part of the outputs and is not relied on.

**Status conclusion: B, SOME EVIDENCE OF INCOMPLETENESS (pooled).** Per collection:

| Collection | Verdict | Basis |
| --- | --- | --- |
| 4502442 | **A, strong** | 27 flagged + 4 ambiguous pages = the whole document; manual reading agrees |
| 4502443 | **B, some** | pp. 11–12 flagged, p. 10 ambiguous, p. 9 stops partway |
| 4502444 | **C, no material evidence** | no flagged pages; ambiguous pages show student edits |
| 4502445 | **C, no material evidence** | 0 flagged, 0 ambiguous |
| 4502446 | **C, no material evidence** | no flagged pages; ambiguous pages show student edits |

The evidence for A and B rests on HEURISTIC timing and convention signals plus a MANUAL reading.
None of it comes from missing text.

The earlier "`¬` versus `-` convention difference" between collections is best explained by this
finding: `¬` is the HTR model's own output, and `-` is the students' correction convention.

### 11. Collection profile

From `collection_completeness.csv`:

| Collection | Pages | Retained lines | % lines with GT | Chars/line | Words/line | Short (≤ 3) | O1 | Gaps | Tails | Line-end `-` / `¬` | Likely-uncorrected pages (lines) | Ambiguous pages |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4502442 | 31 | 1,661 | 99.94 | 24.33 | 4.44 | 239 | 114 | 1 | 0 | 0 / 45 | 27 (1,458) | 4 |
| 4502443 | 12 | 595 | 100.0 | 43.25 | 7.35 | 27 | 11 | 0 | 0 | 52 / 9 | 2 (81) | 5 |
| 4502444 | 15 | 776 | 100.0 | 48.73 | 8.02 | 29 | 7 | 0 | 0 | 77 / 0 | 0 | 5 |
| 4502445 | 49 | 2,137 | 100.0 | 32.08 | 5.75 | 75 | 19 | 0 | 0 | 447 / 0 | 0 | 0 |
| 4502446 | 33 | 1,317 | 100.0 | 31.37 | 5.53 | 46 | 25 | 0 | 0 | 149 / 0 | 0 | 2 |

- Placeholder candidates in retained lines: 0 in every collection.
- PAGE/ALTO disagreements: 0 in every collection.

### 12. Manual spot-check queue

- Files: `manual_spotcheck_queue.csv` and `manual_spotcheck.html`. The HTML links to the frozen
  line crops and to the source page JPGs by relative path; open it from this folder.
- 98 items, selected by rule with seed 20260927 and `NORMAL_PAGES_PER_COLLECTION=2`:

| Reason | Items |
| --- | --- |
| O1 outliers (cap 40) | 40 |
| T2 likely-uncorrected pages | 29 |
| T3 fast save with edits | 11 |
| Random normal pages | 8 |
| T3 no convention | 5 |
| C1 missing GT | 1 |
| C4 internal gap | 1 |
| Minimum coverage | 1 |
| Maximum coverage | 1 |
| T4 mixed line ends | 1 |

#### Text-only reading (MANUAL SPOT-CHECK ONLY)

This was a reading of the GT text only. No images were compared, and no GT was changed.

- **Flagged pages** read like fluent but incoherent recognition output:
  - 4502442 p. 10: "Hwad dhett lösn wijdhomer i så", "junij 166 m:r hans på tinget d";
  - 4502443 p. 11: "emene skall hafwa pucistelnet sin den".
- **Ambiguous pages** 4502442 pp. 1–3 and 4502443 p. 10 read the same way ("görehades
  contributions längden", "Pedersöltes").
- **4502443 p. 9** is coherent in its early lines and garbled from about line 44 ("sna nu, den
  arfwar af slagzmåhl i sin som"). This matches T4.
- **Edited pages** 4502443 pp. 6–7 read as coherent legal prose.

A reading of the text cannot establish what the image says. A **visual comparison of a few T2/T3
pages against their images** (via the HTML) is the step that would confirm or refute this. It has
not been done.

### 13. Descriptive metrics

These are not collapsed into a single score.

| Metric | Value |
| --- | --- |
| % source line regions with GT | 99.985 |
| % retained pages with full eligible GT coverage | 99.29 (139/140) |
| Pages with unfinished tails | 0 |
| Pages with internal gaps | 1 |
| Pages with whole untranscribed regions | 0 |
| Placeholder-bearing retained lines (deterministic rules) | 0 |
| Crop/text extreme outliers | 176 |
| PAGE/ALTO major disagreements | 0 |
| Duplicate/version conflicts | 10 (5 page images; the benchmark uses the earlier, edited version) |
| Pages likely uncorrected recognition output (HEURISTIC) | 29 (1,539 retained lines, 23.7%) |
| Pages with ambiguous recognition signals | 16 |

## Answers

1. **Deterministic evidence of unfinished transcription?**
   - No, in the sense of missing text: 99.985% of lines have GT, and there are no tails and no
     empty regions.
   - Yes, in the sense of unfinished correction. The deterministic metadata (T1: every page
     HTR-prefilled; save timestamps), read through explicit HEURISTIC thresholds, marks 29 pages
     as likely uncorrected.
2. **Transcription stops partway down a page?**
   - Not as empty lines: C3 = 0.
   - As correction stopping partway: one page, 4502443 p. 9 (T4, HEURISTIC; the manual reading
     agrees).
3. **Systematic untranscribed regions?** No. R1 = 0, R4 = 0, and narrow regions are fully
   transcribed.
4. **Material internal gaps?** No. There is one single-line gap, in 4502442 `0029_DSC_0577` `r3l25`.
5. **Do PAGE and ALTO suggest missing content?** No: 0 disagreements. Both come from the same
   export, so this check is weak.
6. **Collections materially less complete than the others?**
   - Yes: **4502442** (the whole document, 1,661 lines, 25.6% of the benchmark);
   - and **4502443** pp. 9 (lower part), 10, 11 and 12 (up to 198 lines).
   - 4502444, 4502445 and 4502446 show no material evidence.
7. **Is `IN_PROGRESS` supported by the data?**
   - For 4502442 and part of 4502443, yes. The data support it as an inference from heuristic and
     manual evidence.
   - For 4502444–4502446 it looks like stale metadata, also as an inference: "The Transkribus
     workflow status remains `IN_PROGRESS`, but these collections show no material deterministic
     evidence of incomplete transcription."
   - The status has not been upgraded, and the dataset is not certified ground truth.
8. **Is the 6,486-line candidate suitable to freeze as is?** No. About a quarter of its lines are
   probably uncorrected HTR output. Scoring against them would measure agreement with a third
   model, `Jaemtlands_domsagas_M1`, rather than with the students' reference transcription.

## Recommendation: NEEDS DATASET CLEANUP BEFORE FREEZE

The benchmark is already frozen as `svea-hovratt-2026-09-primary`. No model has been run on it
yet, so the protocol still allows a change. Cleanup would need:

1. **A human visual check** of a few T2/T3 pages, using `manual_spotcheck.html`, to confirm the
   finding.
2. **A separate, user-approved decision** (for example D6) that excludes the uncorrected pages by
   an explicit rule recorded in `decisions.jsonl` and the protocol. The rule would be at page
   level: T2 pages plus the confirmed T3 pages. The treatment of 4502443 p. 9 (whole page or its
   lower part) has to be decided explicitly.
3. **A new benchmark ID** (for example `svea-hovratt-2026-09-primary-v2`), with the current
   frozen benchmark kept untouched and marked superseded before any model run.

Nothing in this list has been done.
