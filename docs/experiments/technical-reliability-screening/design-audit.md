# Design audit — Swedish Historical HTR Technical Reliability Screening

**Stage 0 deliverable. Checkpoint 1 material. Nothing in this document has been implemented.**

This audit answers one question: *what does this repository already have, and what does it not have,
for an experiment type it has never run before* — a **technical reliability screening** over real,
full-page Swedish historical court records.

It is deliberately blunt about the gaps. The headline finding (§4) is that the single most important
prerequisite for Stage 2 does not exist in any form, and that no amount of configuration will
produce it.

Scope boundary observed while producing this: **no model inference was run, no GPU was used, no
network call was made.** Stage 1's inventory is pure metadata inspection over
`dataset-rgb/`, which was opened read-only throughout.

---

## 1. What this screening is, and why it is not the baseline comparison

The existing experiment
([`docs/experiments/baseline-comparison/README.md`](../baseline-comparison/README.md)) is an
**accuracy comparison**: two recognizers, one byte-identical line crop, CER/WER against a known
ground-truth string.

A technical reliability screening asks a different question, and the difference drives every gap
below:

| | Baseline comparison (exists) | Technical reliability screening (does not) |
|---|---|---|
| Question | Which method transcribes more accurately? | Does the pipeline **survive** real archival pages at all? |
| Input | 1 pre-cropped line + 1 hand-authored PAGE XML | 766 real full-page scans, ~20 MP each |
| Ground truth | Required | **None available and none planned** |
| Primary metrics | CER, WER, accuracy | Non-accuracy: completion, degradation, plausibility |
| Human role | Blind dual transcription → CER agreement | Plausibility judgement without a reference |
| Output | A better/worse ranking | A screening **decision** per method/category |

The second column is what this repository was built for. The third is genuinely new.

---

## 2. Reusable components — what genuinely carries over

These are real assets and the screening should build on them rather than around them.

### 2.1 `HtrMethodAdapter` — reusable as-is
`src/archivetrust/providers/htr_adapter.py`. Every method is a Protocol implementer with
`get_metadata` / `get_capabilities` / `validate_environment` / `recognize` / `health_check`.

Two properties matter here specifically:

* `MethodCapabilities` declares every flag as a **required** boolean with no default, so
  "cannot do X" is never indistinguishable from "not yet declared". For a screening whose entire
  subject is what methods *cannot* do, that discipline is exactly right.
* `validate_environment()` and `health_check()` already separate "prerequisites unsatisfied" from
  "runtime unhealthy". A screening needs precisely this distinction to avoid recording an
  environment failure as a method failure.

`RecognitionResult.raw_response` preserves provider-specific structure unflattened — necessary,
because degradation modes (truncation, repetition, empty output) are often only visible in raw
output.

### 2.2 `DurableHtrResearchStore` / `HtrJournal` — reusable as-is
`htr/persistence/durable_store.py`, `application/htr_journal.py`. Append-only, telemetry-backed,
hash-chained, replayable. The event is written **before** the in-memory projection updates, so a
crash can never leave a projection entry with no durable event behind it.

This is the single most valuable asset for this screening. A screening run over hundreds of pages
*will* crash partway through, and the design already guarantees that everything completed before the
crash is durable and replayable. `tests/htr/persistence/test_real_baseline_reconstruction.py`
demonstrates full reconstruction from the log alone.

### 2.3 `htr/preprocessing/rgb_normalization.py` — reusable, and better suited than expected
Built 2026-07-30. Contract: RGB, 3 channels, 8 bits, no alpha, no ICC, no EXIF, lossless PNG,
geometry unchanged except a physically-applied EXIF orientation. **No enhancement by construction**,
enforced by an AST scan (`ALLOWED_PILLOW_OPERATIONS` +
`tests/htr/preprocessing/test_no_enhancement_contract.py`) rather than by prose.

Two of its details turn out to matter for this dataset:

* Its `_observe()` function records exactly the source properties Stage 1's inventory needs
  (mode, bit depth, channel count, ICC presence and hash, alpha, EXIF orientation). Stage 1's
  inventory script mirrors `_channel_count`/`_bit_depth` verbatim so the inventory and the
  normalization stage describe the same image identically.
* `PageImageArtifact.compute_hash` (`page_image_<sha256>`) is the correct content address for these
  files — they *are* original, un-normalized page images — and Stage 1 reuses it verbatim rather
  than inventing a hashing scheme.

**Status for this screening: ready, and currently a no-op for content.** Stage 1 measured every one
of the 766 files as already 8-bit RGB PNG with no ICC profile, no alpha, and no EXIF orientation
(§6). The stage will still re-encode them to its canonical form — that is its documented behaviour,
and an already-conforming image may come out byte-identical — but it will not change a single pixel
value in this corpus. It remains necessary for Transkribus page prep because it makes the uploaded
bytes a *recorded fact*, not because this corpus needs repair.

### 2.4 `htr/knowledge/*` — reusable, and the right shape for screening conclusions
`ResearchObservation` / `ResearchFinding` with a structured `ResearchScope` (`extra="forbid"`,
wildcard placeholders rejected), typed `EvidenceReference`s, an eight-state finding lifecycle, and
explicit `ObservationConfidence`. `MEASUREMENT_LIMITATION` already exists as an observation type —
the natural home for "this could not be measured because X".

The lifecycle ceiling documented in `docs/knowledge-lifecycle.md` applies: with one run, findings
cap at `Provisionally supported`. A screening producing a second, structurally different run is what
would let any existing finding reach `Supported` or be `Superseded`.

### 2.5 `review/blind_review/*` — reusable **mechanism**, wrong **workflow** (see §3.2)
`BlindReviewStore` mechanically enforces blind isolation (a reviewer cannot read the other's
submission before finalizing their own) and submission locking. `ExclusionRecord` requires a
non-empty reason. `resolve_benchmark_outcome` never overwrites an original recorded result.

The isolation/locking/exclusion machinery is directly reusable. The *agreement computation* is not
(§3.2).

### 2.6 The reporting pattern — `research/reports/export.py`
`report_to_json` / `report_from_json` / `report_to_csv` / `write_report`, with
`UnsupportedExportFormatError` rather than a silent fallback. The baseline directory demonstrates
the intended arrangement: an append-only log is the source of truth, and the report is *derived from*
it — delete the report and it regenerates; delete the log and it does not. The screening should
reproduce that inversion exactly.

---

## 3. Missing components — what a screening needs that does not exist

### 3.1 A non-accuracy metric set — **24 of 26 existing metrics are unusable here**
`htr/evaluation/definitions.py` registers 26 metric definitions. Counted honestly against a corpus
with no ground truth:

* **Ground-truth-dependent, therefore unusable (24):** CER raw/normalized, WER raw/normalized, exact
  line accuracy, exact word accuracy, the six character/word insertion-deletion-substitution counts,
  historical-feature precision/recall, and every segmentation metric (region/line precision and
  recall, mean matched IoU, missed/duplicate/merged/split line rates, reading-order and region-order
  accuracy, crop coverage, crop contamination — all of which take a *reference* geometry sequence).
* **Usable as-is (2):** `EXECUTION_TIME_MS`, `GPU_MEMORY_MB`.

A screening therefore needs a new metric family that is computable with **no reference at all**.
Nothing in this repository computes such a metric today. Candidates to be specified in Stage 2:

- completion rate (pages/lines attempted vs. producing any output)
- hard-failure rate by cause (OOM, timeout, decode failure, adapter error)
- empty-output and whitespace-only-output rate
- degenerate-output rate (character/n-gram repetition loops — a known VLM failure mode)
- output-length plausibility against detected line geometry
- confidence distribution shape, **per method, never cross-method** (the baseline README already
  establishes that SATRN's decode probability and Florence-2's `exp(sequences_scores)` proxy are not
  comparable to each other)
- wall-clock and peak-memory distribution per page category

`MetricDefinition`/`MetricResult` (`htr/experiment/models.py`) can carry these without modification —
the gap is the definitions and the functions computing them, not the storage model.

### 3.2 A human-plausibility review workflow — distinct from blind dual transcription
`review/blind_review/agreement.py` computes agreement by **CER between two independently transcribed
strings**, classified against `AgreementPolicy` thresholds. That presupposes two humans each produce
a full transcription of the same target.

That is the wrong instrument here, for a concrete reason: full manual transcription of 17th-century
Swedish secretary hand is expensive expert work, and the screening question does not require it. The
question is *"is this output plausibly a transcription of this image?"* — answerable by a
palaeographically competent reviewer in seconds, against no reference text.

What is missing is a rating workflow with:
- a **reference-free** ordinal plausibility scale (to be defined at Checkpoint 1)
- single-rater primary path with a **double-rated overlap subset** for inter-rater reliability
- reviewer sees image + output, and is **blind to which method produced it**

The blind-isolation store, assignment pairing, and `ExclusionRecord` from `blind_review` are reusable
underneath it. `compute_agreement`'s CER computation is not — an ordinal plausibility scale needs a
weighted-kappa-style statistic, which does not exist in this repository.

Note also that `review/htr_models.py::ReviewSubmission` is shaped around a transcription string; a
plausibility rating is an ordinal code plus a rationale. This is a new model, not a new field.

### 3.3 A screening-policy / decision entity — no analogue exists
The existing model can record that a method run happened and what it scored. It cannot record **a
decision**: "method M is / is not technically viable for page category C, under policy P, because of
evidence E."

`FindingStatus` is about the epistemic lifecycle of a research claim, not an operational verdict, and
conflating them would corrupt the knowledge model. A screening needs, as new entities:

- `ScreeningPolicy` — the versioned, pre-registered thresholds a decision is made against
- `ScreeningDecision` — (method, category, policy version) → verdict + typed evidence references
- an explicit `INSUFFICIENT_EVIDENCE` verdict, so "not enough data" is never silently rendered as
  "failed"

**The policy must be recorded before results are seen.** Choosing thresholds after looking at the
outcomes converts a screening into a post-hoc rationalization. `ExperimentVersion`'s immutability
machinery (`assert_experiment_mutable`) is the right enforcement mechanism to reuse.

### 3.4 An `ExternalImport` record for Transkribus
The store and event kind exist; the baseline run never created one (its own README lists this as an
honest gap). Any Transkribus participation in this screening must produce one.

---

## 4. **THE MAJOR FINDING — there is no line segmentation, and there never has been**

This is the finding that determines whether Stage 2 is a small task or a significant new capability.
It is the latter.

### 4.1 The asymmetry
| Method | line_level | page_level | Consequence for a full page |
|---|---|---|---|
| SATRN | `True` | **`False`** | Cannot accept a page. Needs line crops. |
| Florence-2 | `True` | **`False`** | Cannot accept a page. Needs line crops. |
| Transkribus | `True` | `True` | Accepts a page; segments **externally, inside the vendor service**. |

So for two of the three methods, a full page is not valid input at all.

### 4.2 The gap, stated plainly
**`src/archivetrust/htr/segmentation/` does not exist.** Not "exists but is a stub" — the package was
never created. Verified by direct search across the repository.

What *does* exist is a consistent set of references to it, which is why the gap is easy to miss:

* `docs/htr-domain-design.md` §102–108 specifies `SegmentationAdapter` with `detect_regions` /
  `detect_lines` / `order_lines`.
* `htr/corpus/models.py::Region` and `::TextLine` docstrings both name
  `SegmentationAdapter.detect_regions` / `.detect_lines` as their producer.
* `domain/telemetry/events.py` defines `SegmentationRunCompleted` (and `RegionDetected` /
  `TextLineDetected`).
* `providers/florence2_htr/adapter.py` and its README document a confirmed extension point.
* `providers/base.py` and `providers/transkribus/parsing_models.py` both refer to it.

Every one of those is a reference to a package that does not exist. The domain model, the telemetry
vocabulary, and the evaluation metrics for segmentation are all in place — **only the thing that
actually looks at an image and finds lines is absent.**

### 4.3 Why this was never noticed
Because no real page image has ever entered this pipeline. From the baseline run's own README:

> **No segmentation run was recorded**, deliberately: the controlled fixture is already a pre-cropped
> line, and Transkribus's segmentation ran externally.

And from `htr/experiment/baseline_template.py:297`, describing the crop's origin:

> "the fixture's own bytes rather than produced by a `htr/segmentation/SegmentationAdapter`"

The entire recorded history of this system consists of:
- **one** pre-cropped line JPEG (`tests/fixtures/htr/trolldomskommissionen_sample_line.jpg`,
  2568×231), and
- **one hand-authored PAGE XML fixture with no image behind it at all** —
  `tests/fixtures/transkribus/` contains only `.xml`/`.txt`. The `RegionDetected` ×3 and
  `TextLineDetected` ×4 events in the committed log were parsed from that XML's `Coords`, not
  produced by any detector.

The geometry in this system has always arrived pre-made. Stage 1's corpus is the first real page
imagery this repository has ever held.

### 4.4 What Stage 2 must therefore solve
This is a real engineering task, not paperwork:

1. **Build `htr/segmentation/` and a real `SegmentationAdapter` implementation.** No line detector of
   any kind exists — there is no OpenCV, scikit-image, kraken, or docTR dependency anywhere in the
   codebase (verified: the only non-Pillow image-stack imports are `torch`, inside the two
   recognizers).
2. **Handle double-page spreads.** Stage 1 found that **717 of 766 pages (93.6%) are double-page
   spreads** — an open bound volume photographed flat, with a central gutter, two facing text blocks,
   two independent reading orders, and a dark non-page surround. Before line detection can run,
   something must locate the page(s) within the photograph and decide whether to split the spread.
   Neither capability exists. This was visually confirmed on real files, not inferred from aspect
   ratio alone.
3. **Decide the detector, and accept that it is a new dependency.** The most defensible option
   already documented in this repository is
   `nazounoryuu/florence_base__mixed__page__line_od` — the companion line-detection checkpoint to the
   Florence-2 OCR model already in use, confirmed to exist and named in
   `providers/florence2_htr/adapter.py`. It is trained on the same Riksarkivet-adjacent Swedish
   material. **But note the confound this creates:** using a Florence-2 detector to produce the crops
   that SATRN is then screened on gives one method's family control of the other's input. That must
   be stated as a limitation, or a method-neutral classical detector used instead.
4. **Segmentation quality has no ground truth here.** `htr/evaluation/segmentation.py`'s metrics all
   require reference geometry, and this corpus has none. Segmentation quality in this screening is
   therefore itself only assessable by human plausibility review — which folds back into §3.2 and
   materially increases the review burden.

**Estimate: this is the dominant cost of Stage 2**, larger than the screening harness itself.

---

## 5. RGB-normalization status and external-import requirements

**RGB normalization: built, tested, ready.** See §2.3. `image_color_normalization_required=True` is
already declared on the Transkribus adapter and `False` on SATRN and Florence-2 — the latter
deliberately, because routing the committed baseline's controlled crops through the stage would
change the bytes they read and invalidate the recorded baseline results. That asymmetry is correct
and must be preserved.

For this screening the stage's role is **Transkribus page prep**: what is uploaded *is* the
experimental input, so its colour representation must be a recorded fact.

**External import (Transkribus):** requires an `ExternalImport` record (§3.4); requires a decision on
whether real archival material may be uploaded to a third-party service at all
(`docs/DATA_HANDLING_POLICY.md` / `docs/SECURITY_AND_DATA_HANDLING.md` govern this and the decision
is **not** mine to make); and requires that a page-level vendor result be recorded without inventing
line-level structure it did not return.

---

## 6. What Stage 1 measured (summary; full detail in the manifest)

Real inventory over all 766 files. See `dataset-manifest.json` and `dataset-inventory.csv`.

| Measure | Value |
|---|---|
| Total files | 766 (all `.png`) |
| Usable / included | **766** |
| Unreadable or undecodable | **0** |
| Duplicates by content hash | **0** (766 distinct hashes) |
| Collections / documents | 31 / 31 (one archival volume per collection directory) |
| Automatic: `Ordinary` | **48** |
| Automatic: `Technically difficult` | **1** |
| Automatic: `Structurally difficult` | **717** |
| Total corpus size | ~13.9 GB (mean 18.6 MB/page) |

The corpus is **technically homogeneous and clean**: every file is a readable 8-bit RGB PNG, with
zero ICC profiles, zero alpha channels, and zero EXIF orientations across all 766. Median dimensions
5131×4230 (~20 MP); smallest dimension observed 2507 px, so resolution is never a limiting factor.
The directory name `dataset-rgb` is accurate — this material has evidently already been conformed to
RGB upstream.

**The single `Technically difficult` page is a borderline call** and should be reviewed as such: it
sits at 0.3075 encoded bytes/pixel against a 0.31 threshold. It is flagged, not condemned.

**Category precedence masks 35 technical flags, by design.** 36 pages carry at least one technical
flag (8 low-entropy, 10 high-entropy, 18 large-raster), but 35 of them are double-page spreads and so
take `Structurally difficult` as their primary category. This is exactly why every signal is *also*
written to its own `flag_*` column: a reviewer who wants a technical-first stratification can derive
it from the manifest without re-running anything, and no signal is lost to the one-primary-category
requirement.

The consequence for classification is important and is stated in the manifest: **there are almost no
technical defects to detect.** The classic technical signals (palette mode, 16-bit samples, alpha,
ICC, low resolution, unsupported format) fire on nothing. The corpus's real difficulty is
*structural* — overwhelmingly double-page spreads — which is why the category distribution is so
lopsided.

Two honesty notes carried into the manifest:

* **Bytes-per-pixel is not a compression-artifact proxy here.** Every file is lossless PNG, so it
  measures raster *entropy*, not codec loss. It is used only as a faint-scan / heavy-grain outlier
  flag, with thresholds taken from this corpus's own observed p01/p99.
* **Structural difficulty beyond spread geometry is not automatically detectable.** Marginalia
  density, tables, multi-column layout, insertions, and physical damage all require a layout model,
  which is out of scope for an automatic pass. Pages carrying such difficulty are conservatively
  reported as `Ordinary`, and every such record says so in its `classification_reason`. Supervised
  human correction is expected and is the documented purpose of the empty
  `supervised_final_category` column.

---

## 7. Supervised decisions required at Checkpoint 1 — **not decided here**

1. **Sample size.** Stage 1 proposes a stratified sample; adopt, resize, or reject it.
2. **Whether to approve the sample at all before correcting categories.** The proposal is explicitly
   marked non-binding: it stratifies on categories known to be incomplete, so approving it now would
   freeze a stratification that is expected to change.
3. **Category correction**, especially `Ordinary` pages that are in fact structurally difficult.
4. **Double-page spreads: split or process whole?** This affects 93.6% of the corpus and is the
   single highest-leverage decision on this list.
5. **The plausibility rating scale** and what counts as a passing rating.
6. **Review sampling rate** and the size of the double-rated overlap subset.
7. **Screening policy thresholds**, which must be fixed *before* results are seen (§3.3).
8. **Whether Transkribus participates at all**, given that it requires uploading real archival
   material externally.
9. **Which line detector**, and whether the Florence-2-family confound (§4.4.3) is acceptable.

---

## 8. Planned implementation for later stages — described, not built

- **Stage 2 — segmentation capability.** `htr/segmentation/` + a real `SegmentationAdapter`; page
  localization within the photograph; spread handling; emit the already-defined `RegionDetected` /
  `TextLineDetected` / `SegmentationRunCompleted` events; produce content-addressed `InputCrop`s via
  the existing `InputCrop.create`. **The largest piece of work in the whole plan.**
- **Stage 3 — screening metric set.** New `MetricDefinition`s per §3.1 and the reference-free
  functions computing them.
- **Stage 4 — screening harness.** Per-page, per-method execution with hard timeouts and memory
  caps, writing through `DurableHtrResearchStore` so a crash loses nothing.
- **Stage 5 — plausibility review.** New rating models over the existing blind-isolation store; a
  weighted-agreement statistic for the overlap subset.
- **Stage 6 — decisions.** `ScreeningPolicy` / `ScreeningDecision`, applied to recorded evidence.
- **Stage 7 — report + knowledge.** Derived from the log per §2.6; observations and findings via
  `htr/knowledge/*`.

---

## 9. Risks

1. **Segmentation quality silently dominates every result.** If line detection is poor, all three
   methods score poorly and the screening measures the detector, not the recognizers. **Highest risk
   on this list.** Mitigation: human plausibility review must rate segmentation separately from
   recognition, and the screening must be able to conclude "not assessable".
2. **The detector-family confound** (§4.4.3).
3. **Scale.** 766 pages × ~20 MP × 3 methods, with SATRN paying subprocess start-up on every call and
   holding no warm worker. Full-corpus runs are not casually repeatable; the sample exists partly for
   this reason.
4. **Category imbalance is severe and caps what the small strata can support.** The automatic pass
   yields 717 / 48 / 1. The `Technically difficult` stratum has **exactly one member**, so it
   supports no statistical claim whatsoever — the proposed sample allocates it 1 slot purely so the
   category is represented, not because n=1 is informative. This is a finding to report, not a
   defect to engineer around, and it is a strong argument for correcting the categories before
   fixing any sample.
5. **Reviewer capacity** is the binding constraint on review volume, not compute.
6. **Post-hoc threshold selection** (§3.3) — mitigated only by pre-registering the policy.
7. **`.venv-satrn` fragility.** Hand-assembled, uncommitted, with a manually patched `mmdet` version
   guard. It can break between sessions, and a broken environment must be recorded as an environment
   failure, never as a method failure.
8. **Confidence values are not cross-method comparable** and must never be pooled.

---

## 10. Benchmark limitations — to be stated in any output

- **No ground truth. No accuracy claim of any kind is possible from this screening.** It measures
  whether the pipeline survives real pages, not whether it reads them correctly.
- **One corpus, one archival domain, one period** (Swedish witchcraft trials, 1584–1764,
  Riksarkivet-style). No generalization to other hands, languages, or centuries.
- **One checkpoint per method.** Conclusions bind to the pinned revisions, not to "SATRN" or
  "Florence-2" as such.
- **Structural categories are partly human-assigned**, so the strata are not fully reproducible from
  the automatic pass alone.
- **Plausibility ratings are subjective**, bounded by the inter-rater reliability actually measured
  on the overlap subset.
- **Transkribus is not comparable to the local recognizers** — different execution model, external
  segmentation, vendor confidence. The baseline run already refused to compute CER for it; the same
  discipline applies here.
- **A screening is a filter, not a ranking.** Its output is "viable / not viable / insufficient
  evidence", never "method A is better than method B".

---

## 11. Provenance of this audit

Produced by reading the code and documentation directly, not from summaries. The load-bearing
verifications: `htr/segmentation/` absent (repository-wide search); no OpenCV/scikit-image/kraken/
docTR dependency (import search across `src/`); 24 of 26 metric definitions ground-truth-dependent
(`htr/evaluation/definitions.py`); `page_level_supported=False` for SATRN and Florence-2 (both
adapters); pre-cropped-fixture provenance (`baseline_template.py:297` and the baseline README's own
"honest gaps"); corpus properties measured directly by Stage 1's script over all 766 files.

**No model inference was performed. No GPU or network operation was performed. `dataset-rgb/` was
read only and is unmodified.**
