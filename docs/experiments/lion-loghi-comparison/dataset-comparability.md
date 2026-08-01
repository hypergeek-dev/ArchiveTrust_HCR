# Swedish vs. Dutch Corpus Comparability

Status: Current
Governs: Whether the Swedish (`dataset-rgb/`) and Dutch (`dataset-dutch-rgb/`) corpora are matched
enough to support claims about cross-domain transfer, and which dimensions are not.
Source of measured figures: `docs/experiments/technical-reliability-screening/dataset-inventory.csv`
(Swedish, 766 pages) and `docs/experiments/lion-loghi-comparison/dataset-dutch-inventory.csv` (Dutch,
515 pages), both computed directly from the actual files, not estimated.

Per the brief: **do not claim the corpora are matched unless the evidence supports it.** Every row below
is classified `Well matched` / `Approximately matched` / `Known mismatch` / `Unknown`, and a mismatch is
preserved as a limitation, not smoothed over.

| Dimension | Swedish (`dataset-rgb/`) | Dutch (`dataset-dutch-rgb/`) | Classification |
|---|---|---|---|
| Century | 17th–18th c. (witch-trial records, 1584–1764 per collection names) | 17th–18th c. inference (Staten-Generaal fonds `1.01.02`; exact date range of the 515 sampled pages not yet extracted from the XML/filenames) | **Approximately matched** — same broad period, exact Dutch date range not yet confirmed page-by-page |
| Document type | Witch-trial court/commission records (legal proceedings) | States-General government resolutions (administrative/legislative) | **Known mismatch** — different genres of historical document; legal-proceeding prose and legislative-resolution prose are not the same register |
| Number of pages | 766 (full corpus); 60-page stratified sample used by the sealed reliability run | 515 (465 train + 50 val, as packaged) | **Known mismatch** in raw count; not itself a problem if per-condition sample sizes for the new experiment are drawn and reported explicitly (§ four-cell design) |
| Page dimensions | width 2507–8248 px (avg ≈5411), height 2736–5624 px (avg ≈4311) | width 4571–6338 px (avg ≈5363), height 3761–4808 px (avg ≈4186) | **Approximately matched** on average dimensions; Swedish corpus has a much wider spread (min width 2507 vs. 4571) |
| Scan quality | Not classified here — requires human review | Not classified here — requires human review | **Unknown** |
| Layout complexity | Swedish inventory flags most pages "Structurally difficult" (717/766, driven mostly by double-page-spread aspect ratios) — see that inventory's own classification | Not computed (no equivalent classification run for the Dutch corpus — see `scripts/inventory_dutch_dataset.py`'s note on why the Swedish thresholds are not assumed to transfer) | **Unknown** for a like-for-like comparison; the Swedish figure exists but was produced by a corpus-specific heuristic not run against Dutch pages |
| Handwriting density | Not classified here — requires human review | Not classified here — requires human review | **Unknown** |
| Writer diversity | Not recorded | Not recorded (Transkribus GT metadata does not name a writer per page in what was inspected) | **Unknown** |
| Damage | Not classified here — requires human review | Not classified here — requires human review | **Unknown** |
| Bleed-through | Not classified here — requires human review | Not classified here — requires human review | **Unknown** |
| Marginalia | Not classified here — requires human review | Not classified here — requires human review | **Unknown** |
| Double-page spreads | Explicitly flagged in the Swedish inventory (`flag_double_page_spread`, driven by an aspect-ratio ≥1.15 heuristic) — a large majority of pages | Not assessed; the Dutch pages' aspect ratios (width/height ≈1.28 on average, computed from the measured dimensions above) are in a broadly similar range to the Swedish corpus's threshold, but no per-page flag has been computed | **Unknown**, with a **Known mismatch risk** flagged: the Swedish corpus's own documentation treats most of its pages as double-page spreads, and an equivalent assessment has not been run on the Dutch corpus |
| Image resolution | JPEG/PNG-derived, page counts and dimensions above are the resolution proxy available; DPI/PPI metadata was not extracted | Same — DPI/PPI metadata was not extracted from the JPEG EXIF in this pass | **Unknown** |

## What this means for the planned experiment

The two corpora are **not** claimed to be matched overall. The clearest, evidence-backed statement this
document can make is: broadly comparable historical period and broadly comparable average page
dimensions, against a **known mismatch in document genre** (legal-proceeding vs. legislative-resolution
prose) and a **known gap** in every layout/handwriting/condition dimension that requires human
judgment rather than a file-property measurement. Any cross-domain finding drawn from the four-cell
comparison must state this as a limitation — a transfer-performance difference between Swedish and Dutch
pages cannot be attributed to language domain alone when document genre also differs, per the brief's
own "language-domain status does not explain all transfer effects."

## Follow-up

The `Unknown` rows above are not fillable without either (a) a human structured-review pass over sample
pages from both corpora, or (b) building a Dutch-corpus-specific classification pass analogous to
(but not copied from) `scripts/inventory_technical_reliability_screening.py`'s Swedish-specific
heuristics. Neither is performed in this integration pass — consistent with the brief's "do not
fabricate... before the Swedish and Dutch datasets have been prepared."
