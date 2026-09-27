# Convention audit: svea-hovratt-2026-09 candidate (2026-09-27)

> **Re-run after the D4b extension (2026-09-27).** The candidate is now `7c47b9ee…` (6,486 lines)
> and the decisions `24c815be…`.
>
> - E9 in the candidate is now **0**. The 18 lines moved to the review queue, so
>   `E9@excluded_markup_review` is 58 matches.
> - E5 in the candidate is now 6 lines. All are single `?` in questions.
> - A1 is 986.
> - All other counts below are unchanged.
> - Manual queue: 96 rows (E5 6, H4 36, H6 54).
>
> The pre-extension numbers below are kept for the record. The first run was on candidate
> `76c4e38e…`, decisions `a534138b…`.

## Scope and inputs

- Script: `scripts/benchmark_convention_audit.py`, run with:
  ```
  .venv\Scripts\python.exe scripts/benchmark_convention_audit.py svea-hovratt-2026-09 --training-parquet-dir benchmark-data/training-reference
  ```
- Inputs:
  - Candidate `76c4e38e…`: 6,504 lines, decisions `a534138b…`.
  - The 41 lines in the markup review queue.
  - The PAGE-XML and `metadata.xml` files in the delivery.
  - The comparison corpus `training_ref`: the Svea Hovrätt training subset, 40,983 lines.
- The audit is read-only. It changed no GT, decisions or candidate files.
- A second run gave byte-identical outputs.
- Input hashes and all constants are in `run_record.json`. Every finding, with its rule, matches, collections and examples, is in `audit_findings.json`.
- Labels:
  - **DET**: exact count under a stated rule.
  - **HEUR**: a stated rule that only approximates the concept.
  - **MANUAL**: a case the rules cannot decide. These are listed in `manual_review_queue.csv` (116 rows).

Collections: 4502442 (1708), 4502443 (1713, pp. 372–383), 4502444 (1713, pp. 103–117), 4502445 (1702, pp. 366–414), 4502446 (1702, pp. 124–156).

## Findings

| # | Label | Finding | Matches | Collections | Output |
|---|---|---|---|---|---|
| E9 | DET | **Unbracketed `???` / `??` in included candidate lines.** Rule: `\?{2,}`. These were missed by the D4b bracket check. Examples: `oppbur???`, `sedan ???`, `21??? nödig`, `Li???` | 19 matches in 18 lines on 6 pages | 4502444 only | `editorial_markup.csv`, queue `E9_*` |
| E1 | DET | `[...]` markup: all of it is in the 41 excluded review lines; none is in the candidate. | 41 | 4502443–46 | `editorial_markup.csv` |
| E5 | HEUR → MANUAL | `?` next to a letter in the candidate. 2 of these are E9 lines; the other 6 look like ordinary questions in testimony, e.g. `såldt tröyan? S.` | 8 | 4502444–46 | queue `E5_*` |
| H1/H2/H5 | DET | **Line-final hyphen mark differs by collection.** 4502442 uses only `¬` (45). 4502443 mixes them (`-` 52, `¬` 9). 4502444–46 use only `-` (77, 447, 149). Training uses `¬` on 27.3% of lines and `-` on 0.06%. | `-` 725, `¬` 54 | all | `hyphenation_analysis.csv` |
| H3 | HEUR | The line ends in `-` or `¬`, and the next line in reading order starts in lowercase. This means word-break hyphenation. | 731 of 779 | all | `hyphenation_analysis.csv` |
| H4 | MANUAL | A hyphen-like mark ends the line, but the next line starts with a capital (27, e.g. `Läns- ‖ Man`, a compound with a capitalised part) or a digit (9, probably marginal or page numbers in reading order). | 36 | 4502442, 45, 46 | queue `H4_*` |
| H6 | MANUAL | Every candidate line ending in `¬`, for a spot-check against the image. | 54 | 4502442, 43 | queue `H6_*` |
| U3 | DET | Characters frequent in training and absent from the candidate: `;` 1,100 → 0; `ß` 1,077 → 0; `ü` 102 → 0; `é` 92 → 0. `æ` is 393 → 2. | 4 chars | – | `unicode_inventory.csv` |
| U1 | DET | Candidate characters outside the Loghi charset: `æ` ×2 and `¼` ×1. | 3 | 4502442, 45 | `unicode_inventory.csv` |
| U2 | DET | No combining marks, control characters, format characters, private-use characters or non-ASCII spaces. The GT is clean NFC. | 0 | – | `unicode_inventory.csv` |
| M1 | DET | All 140 pages have Transkribus page status `IN_PROGRESS`; none is `GT` or `FINAL`. | 140 | all | `metadata_summary.json` |
| M2/A2 | DET | No Transkribus custom tags (abbrev, unclear, gap, sic, textStyle) on any candidate line; only `readingOrder`. The conventions exist only as plain text. | 0 | all | `metadata_summary.json` |
| M3 | DET | No `TextEquiv@conf` on any line, so there is no recognition-confidence trace. This does not prove the text was typed by hand. | 0 | – | `metadata_summary.json` |
| A1 | HEUR | Abbreviations are kept unexpanded, as 1–4 letters followed by `.` or `:`, e.g. `d.` 152, `H.` 145, `ell.` 80, `Smt.` 70, `hoo.` 41. As a density, `:` per 10k chars is 28 in 4502442, 11.7 in 4502443, about 0 in 4502444–46, and 35.9 in training. | 988 (≥3 occurrences) | all | `abbreviation_candidates.csv` |
| C1 | HEUR | Capitalised tokens in mid-sentence: 15.8% in the candidate vs 31.7% in training; 4502442 has 9.4%, the others 16–19%. | – | all | `capitalization_profile.csv` |
| S1 | HEUR | Old spellings `hw-`, `qw-` and `fw` occur at rates close to training (11.9/10.7, 0.50/0.50, 21.3/23.4 per 1000 words). Both corpora look unmodernised. `dh` varies: 22–39 in 4502442–44, about 2 in 4502445–46, and 0.26 in training. `ff` is 18.5 vs 9.7. | – | all | `collection_comparison.csv` |

## Conclusions from the outputs

1. **E9 is a gap in the current candidate: DET.**
   - 18 included primary lines contain the same unreadable-text placeholder that D4b excludes, only without brackets. All are in 4502444.
   - Under the rule D4b states, these lines should be in the review queue, not in the primary set.
   - Fixing this needs an approved decision change (extend D4b to `\?{2,}`) and a rebuild. That is allowed before the freeze. The audit did not make the change.
2. **Line-end hyphenation is a per-collection transcription choice: DET for the counts, HEUR for the meaning.** 4502442 matches the training convention (`¬`), and 4502444–46 do not. The pre-registered sensitivity score exists for exactly this. Report CER per collection alongside it.
3. **`ß`, `;`, `ü` and `é` are absent from the candidate: DET.** A model trained on the training convention can emit them where the benchmark GT has other characters. Whether that is a convention difference (e.g. `ß` written as `ss`) or a genuine difference in the source text needs Q5.
4. **No page is marked GT/FINAL in Transkribus, and there are no structural tags: DET.** The transcription status must come from the provider (Q5).
5. **Capitalisation, `dh` and `:` differ between collections: HEUR.** They follow the 1708/1713 vs 1702 split and not the date order, so they may reflect different transcribers. This is not established; add it to Q5.

## For the provider (adds to Q3/Q5; not sent)

- Is `???` without brackets the same marker as `[???]`? It occurs in 18 lines of the 1713 `103–117` document.
- Were the collections transcribed by different people or under different guidelines? `¬` vs `-` at line end, `:` as an abbreviation mark, and `dh` spellings differ by collection.
- Is the Transkribus status `IN_PROGRESS` meaningful, or were the pages simply never marked final?
