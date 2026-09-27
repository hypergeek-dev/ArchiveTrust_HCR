# Image review of the completeness finding (2026-09-27)

This review checks one question: does the supplied GT visibly correspond to the handwriting, or
does it look like raw, uncorrected recognition output?

- It compared page images, line crops and the delivered GT only.
- No model was run. No Lion or Loghi prediction exists or was consulted.
- No GT was changed.

## Sample

- Script: `scripts/benchmark_image_review_sample.py`.
- Input: the v1 completeness audit's `recognition_provenance.csv`.
- Seed: 20260927.
- Size: 15 pages, 46 lines. The strata are fixed in the script (see its docstring).
- Record: `sample_record.json`.
- Two runs produced byte-identical `sample.csv`.

Pages by stratum:

- **Suspect pages (10):**
  - 4502442: p1 (T3 without convention), p5, p18 and p31 (first, middle and last of the save
    cluster), p20 (most `¬` line ends) and p30 (highest OOV rate);
  - 4502443: p9 (T4; 2 lines above and 2 lines below the first `¬`), p10, p11 and p12.
- **Control pages (5):** 4502443 p2, 4502444 p7 and 4502446 p10 (edited), 4502445 p12 (edited),
  and 4502444 p13 (fast save with student edits).

## Reviewer and limits

The reviewer was Claude, an AI assistant, comparing line crops with GT by eye. It is not a trained
palaeographer: many 18th-century hands were only partly legible to it. So:

- **LIKELY UNCORRECTED** is used only where a word legible in the image appears in the GT as a
  recognition-style misreading or a non-word.
- Lines where the GT matched the image, or where the image was not legible enough, are
  **AMBIGUOUS**.
- A human check of a few lines is advisable, for example:
  - 4502442 p30 r2l3 (`Carl Grips` vs GT `Carl Grims`);
  - 4502443 p10 r2l29 (`Häradzhöfdingen` vs GT `Hindran gaff ingen`).

Line-by-line results are in `assessments.csv`.

## Results

| Group | Lines | Corrected / human-like | Likely uncorrected | Ambiguous |
| --- | --- | --- | --- | --- |
| Suspect: 4502442 (6 pages) | 18 | 0 | 6 | 12 |
| Suspect: 4502443 p9, above first `¬` | 2 | 2 | 0 | 0 |
| Suspect: 4502443 p9, from first `¬` | 2 | 0 | 2 | 0 |
| Suspect: 4502443 p10–12 | 9 | 0 | 3 | 6 |
| Controls (5 pages, 4 collections) | 15 | 15 | 0 | 0 |

- On the suspect pages, likely-uncorrected lines were found on 4502442 p1, p5, p18 and p30, and on
  4502443 p9 (lower part), p10, p11 and p12.
- The sampled lines of 4502442 p20 and p31 were all ambiguous. Their text is consistent with the
  image, so they give no contrary evidence.
- No suspect line below the first `¬` showed a sign of correction, such as an expanded
  abbreviation or a student-style `-`.
- Every control line matched its image. The few slips found are the kind a human leaves, such as
  a dropped letter ("åns" for "Måns"). One control line expands the abbreviation `D:r` to "Daler",
  which a recognition model would not do.

## Conclusions

**4502442:**

- Recognition-style misreadings occur at the start (p5), middle (p18) and near the end (p30) of
  the 8-minute save cluster, and on p1, which lies outside the cluster but has no student
  convention.
- No sampled line shows a sign of correction.
- The evidence supports treating the document as uncorrected recognition output, pp1–31. This is
  an inference from 6 of 31 pages: p20 and p31 were consistent with either reading, and pp2–4 were
  not sampled.

**4502443:**

- pp9–12 are suspect. The sample supports this: each has at least one clear misreading, and
  none shows a correction sign.
- p9 is mixed. Its lines above the first `¬` read as corrected and its lines from the first `¬`
  on read as uncorrected. No metadata marks the boundary, so the page is excluded whole (D6).
- The other pages: the sampled control p2 reads as corrected. pp1–8 carry student `-` line ends.
  pp4, 5, 6 and 8 were saved fast but show student edits; one page with the same signal
  (4502444 p13) was sampled as a control and reads as corrected.

The cluster is confirmed, so D6 applies. See `../decisions.jsonl` and docs/BENCHMARK_PROTOCOL.md
§3.
