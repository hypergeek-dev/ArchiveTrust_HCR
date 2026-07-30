# ArchiveTrust HTR Evaluation Protocol

Status: Current
Scope: How this codebase measures HTR recognition and segmentation, and what a measurement is allowed to claim
Governs: Comparison boundaries, metric definitions, normalization, reliability classification, exclusions
Applies to: `src/archivetrust/htr/evaluation/` at `EVALUATION_ENGINE_VERSION = 1`
Supersedes: this document's own pre-HTR content (the OCR-era protocol summary, and the supersession notice that stood in place of real content)
Last verified against code: 2026-07-30

> **What changed on 2026-07-30.** This file previously held a supersession notice plus four paragraphs
> of the OCR-era evaluation protocol, and `docs/htr-repository-cleanup.md` carried it as a residual
> gap: *"annotated with supersession notices rather than fully rewritten for
> SATRN/Florence-2/Transkribus"*. It is now rewritten against the code that actually exists. Every
> metric, threshold, classification and exclusion below names the real module and function that
> implements it, and §7 describes the comparison boundary the **real 2026-07-30 baseline run actually
> drew**, not a generic one.
>
> The retained blind dual-annotation/adjudication machinery the old text described is **not** discarded
> — it is §11, and it is still `evaluation/ground_truth.py` + `review/blind_review/`.

## 1. Two comparisons, not one, and they are not interchangeable

Everything in this protocol depends on which of two questions is being asked. The baseline experiment
runs both, as **two separate `ExperimentRun`s with two different `correlation_id`s**, and no event
belongs to both.

| | Controlled line-level comparison | End-to-end page comparison |
|---|---|---|
| Question | Given the *same* line image, which recognizer reads it better? | Given a page, what does a full pipeline produce? |
| Unit of analysis | one `InputCrop` (`ScopeUnit.LINE_CROP`) | one `Page` |
| Segmentation | **none runs** — input is already a cropped line | external, already completed, not re-run here |
| `ExperimentRun.is_end_to_end` | `false` | `true` |
| Baseline run id | `experiment_run_30ba2bcc18a14c06af0f9ca291442cf8` | `experiment_run_30ef72f1ca494a8e8eebf0b4fde33d73` |
| Methods that can participate | `line_level_supported = yes` | `page_level_supported = yes` |
| CER/WER computed | yes | **no, in this run** — see §7 |

**Why the split is structural rather than a convention.** A controlled comparison isolates recognition
by holding segmentation constant; an end-to-end comparison measures segmentation and recognition
together and cannot attribute an error to either. Reporting them as one number would silently blend a
recognizer's error with a segmenter's. `ExperimentRun.is_end_to_end` is the field that keeps them
apart, and `htr/experiment/baseline_template.py`'s `segmentation_strategy` states in the stored
configuration which of the two each run is.

**The controlled comparison's precondition is a byte-identical input, verified rather than assumed.**
`InputCrop.hash` is recomputed from the fixture's bytes and asserted equal across every controlled
method run (`baseline_template.py::exclusion_criteria`: *"excluded … if its input crop's content hash
does not match every other controlled-run's input crop hash for the same TextLine"*). Pointing two
adapters at the same path in source code is not evidence they read the same bytes;
`tests/htr/persistence/test_real_baseline_reconstruction.py` re-asserts the hash **from the committed
log alone**, recomputing it from the fixture on every run.

## 2. Segmentation versus recognition

`htr/evaluation/` separates these into two modules with no shared state, because they answer different
questions and because a method can be scorable on one and not the other at all.

* **Recognition** — `htr/evaluation/recognition.py`. Text in, text out. Needs a reference
  transcription. Does not know what a bounding box is.
* **Segmentation** — `htr/evaluation/segmentation.py`. Geometry in, geometry out. Needs reference
  geometry. Does not read text.

**Consequence for the three real methods**: `satrn` and `florence2_htr` both report
`geometry_supported = no` (`docs/CAPABILITY_MATRIX_HTR.md`), so *no* segmentation metric can be computed
for them — not "they scored zero", but "there is nothing to score". Only
`transkribus_swedish_lion_1` reports geometry. A segmentation table with two empty columns is the honest
rendering; a zero in those columns would be a fabricated failure.

**Reading-order and line-break errors are segmentation findings, not recognition findings.**
`ReliabilityFlag.READING_ORDER_ERROR`, `INVENTED_LINE_BREAKS` and `LOST_LINE_BREAKS` exist in
`failures.py` for exactly this reason: a method that read every character correctly in the wrong order
has an ordering problem, and charging it to CER would mislabel the cause.

## 3. Recognition metrics: raw and normalized, both, always

`htr/evaluation/recognition.py::compute_recognition_metrics(reference, hypothesis)` returns a
`RecognitionMetrics` carrying **both** a raw and a normalized CER and WER simultaneously. Neither
overwrites the other. This is a deliberate departure from the more common practice of reporting one
number and mentioning normalization in a caption.

| Metric | Definition constant | Computed by |
|---|---|---|
| CER, raw | `definitions.CHARACTER_ERROR_RATE_RAW` | character Levenshtein over unnormalized strings |
| CER, normalized | `definitions.CHARACTER_ERROR_RATE_NORMALIZED` | `evaluation/metrics.py::compare_text`, which normalizes both sides first |
| WER, raw | `definitions.WORD_ERROR_RATE_RAW` | `recognition.py::_raw_wer` — word-level Levenshtein over unnormalized whitespace splits |
| WER, normalized | `definitions.WORD_ERROR_RATE_NORMALIZED` | `compare_text` |
| Exact line accuracy | `definitions.EXACT_LINE_ACCURACY` | `recognition.py::exact_line_accuracy` |
| Exact word accuracy | `definitions.EXACT_WORD_ACCURACY` | proportion of reference words matched exactly |
| Char ins/del/subst | `CHARACTER_INSERTIONS`, `CHARACTER_DELETIONS`, `CHARACTER_SUBSTITUTIONS` | `recognition.py::classify_char_edits` |
| Word ins/del/subst | `WORD_INSERTIONS`, `WORD_DELETIONS`, `WORD_SUBSTITUTIONS` | `recognition.py::classify_word_edits` |

`recognition.py::recognition_metric_results` is what turns a `RecognitionMetrics` into the
`MetricResult` entities that reach the durable log.

**Edit-operation classification is a real backtrace, cross-checked against the retained primitive.**
`evaluation/metrics.py::levenshtein` returns only a distance and cannot say whether an edit was an
insertion, deletion or substitution. `recognition.py::_classify_edits` is a DP-table backtrace that can,
and the invariant `EditOpCounts.total_edits == levenshtein(reference, hypothesis)` is asserted in tests
— so the classifier is provably consistent with the primitive rather than a divergent
reimplementation. Two further invariants hold and are tested:
`matches + substitutions + deletions == len(reference)` and
`matches + substitutions + insertions == len(hypothesis)`.

**Why insertions and deletions are reported separately and never summed.** They are different failure
modes with different causes. In the baseline run SATRN produced `0 / 38 / 8` characters
(ins/del/subst) and Florence-2 `4 / 2 / 19`: SATRN dropped most of the line, Florence-2 attempted all of
it and got most characters wrong. Two CERs alone (0.7931 vs 0.4310) do not distinguish those shapes, and
`ReliabilityFlag.OMITTED_TEXT` fires on the first and not the second.

## 4. Text normalization: exactly one profile, named in every scope

> **Not to be confused with image normalization.** Since 2026-07-30 this system also has an
> `image_color_normalization` stage (`htr/preprocessing/`, §4.1 below), which normalizes an input
> *image's colour representation*. The two are unrelated, operate on different things, and are
> versioned independently: this section is about normalizing *text* before comparing it to a reference.
> Where a document or a field name could be ambiguous, prefer "text normalization" and "image
> normalization" explicitly.

There is one text normalization function, `evaluation/metrics.py::normalize_text`, and it does exactly
two things:

1. Unicode **NFC** composition;
2. collapse every whitespace run to a single space, then strip.

**No case folding.** Its own docstring gives the reason: *"casing is real information a transcription
can get wrong."* No punctuation stripping, no diacritic folding, no historical-orthography
normalization, no lowercasing.

That profile is named in full in every `ResearchScope.normalization_profile` —
`evaluation.metrics.normalize_text v1 (Unicode NFC composition + whitespace-run collapse, no case
folding)` — because **a CER is not comparable across normalization profiles**, and a scope that omits
the profile is under-specified rather than merely terse. `ResearchScope`'s own field docstring says so.

**What NFC does and does not do to this corpus.** The baseline ground truth
(`bekiendt. Säger och deth hon Minnes hoon Tuå gånger waritt`) contains Swedish diacritics and the
upstream dataset's own convention for uncertain characters. NFC composes them consistently on both
sides; it does not *interpret* them. In the baseline run raw and normalized CER/WER coincide exactly
(0.7931/0.7931 and 0.4310/0.4310) because neither string contained a whitespace run or a decomposed
sequence — a real result worth stating, and not evidence that normalization is a no-op in general.

**Unsupported normalization is a reliability finding, not a silent adjustment.**
`ReliabilityFlag.UNSUPPORTED_NORMALIZATION` fires when a method's output carries normalization this
profile does not model, rather than the engine quietly reconciling it.

## 4.1 Image colour normalization is an input variable, not a metric adjustment

Added 2026-07-30 with the `image_color_normalization` stage (`htr/preprocessing/`, full documentation in
`docs/methods/transkribus-swedish-lion-1.md`).

**It computes no metric and adjusts no score.** It normalizes the *input image's colour
representation* — to 8-bit RGB, alpha composited over a recorded background, ICC recorded-and-stripped,
EXIF orientation applied — before that image is handed to a method. It performs no enhancement of any
kind (machine-checked, not merely asserted), so it cannot improve or degrade a recognition result by
adjusting appearance; it can only make the representation a stated fact instead of an accident.

**Why that matters for evaluation.** For a page-level, externally-processed method such as Transkribus
Swedish Lion I, the image handed over *is* the experimental input. Its colour representation is
therefore an experimental variable, and an unrecorded variable is exactly what makes two runs
non-comparable. The stage's `configuration_hash` and implementation version are recorded on
`ExperimentVersion.pipeline_configuration_ref`, so:

> **A result is not comparable across image-normalization configurations**, for the same structural
> reason a CER is not comparable across text-normalization profiles (§4). Changing the normalization
> configuration forces a new `ExperimentVersion` — enforced through the existing
> `to_ref()`/`assert_experiment_mutable` mechanism, proven in
> `tests/htr/preprocessing/test_experiment_versioning.py` — so two runs with different image
> normalization cannot be presented as the same pipeline configuration.

**Scope limits, stated plainly:**

- The committed baseline (`docs/experiments/baseline-comparison/`) ran **without** this stage. Its
  Transkribus run had no page image at all, and no normalization event was added to it retroactively
  (§7 of the method doc). Any comparison against that baseline must not claim normalized input.
- A page whose normalization *failed* is excluded from its export package and never submitted in
  un-normalized form. The exclusion is durable evidence (`ImageNormalizationFailed`), not a silent
  omission — the same discipline §7 applies to comparison boundaries.
- The stage is **not** applied to controlled line-level `InputCrop`s. Doing so would change the bytes
  the local recognizers read relative to the committed baseline and would invalidate its recorded
  results. Line-level controlled comparison remains governed by §1 and §2.

## 5. The four transcript stages, and which of them metrics attach to

A method run records up to four distinct texts, each as its own telemetry event:

| Stage | Event kind | What it is |
|---|---|---|
| raw | `RawMethodResultRecorded` | the model's literal output, special tokens and all |
| parsed | `ParsedMethodResultRecorded` | after the adapter's own parse step |
| normalized | `NormalizedMethodResultRecorded` | after `normalize_text` |
| reference | `GroundTruthTextRecorded` | the human transcription |

Normalized CER/WER compare *normalized against normalized*; raw CER/WER compare *raw against raw*.
Mixing stages across sides would measure the parser rather than the model.

**Missing stages are real and are not filled in.** SATRN emits no `ParsedMethodResultRecorded` because
it has no separate parse step; Transkribus emits no `RawMethodResultRecorded` because a PAGE XML export
has no pre-parse model text. `docs/experiments/baseline-comparison/README.md` lists both asymmetries
explicitly: *"No event was invented to make the two methods' chains look alike."*

**A special token surviving into a later stage is itself a finding.**
`failures.py::_MALFORMED_MARKERS` (`<s>`, `</s>`, `<pad>`, `�`) are expected on Florence-2's *raw*
output and are a `MALFORMED_OUTPUT` flag if they reach parsed or normalized — a stage-aware check that
only exists because the stages are stored separately.

## 6. Reliability classification: eleven flags, and what a flag is *not*

`htr/evaluation/failures.py::classify_reliability` inspects one method run's raw/parsed/normalized
triple plus ground truth and returns zero or more `FailureRecord`s. `ReliabilityFlag` has eleven
members: `HALLUCINATED_TEXT`, `OMITTED_TEXT`, `REPEATED_TEXT`, `UNSUPPORTED_NORMALIZATION`,
`INVENTED_LINE_BREAKS`, `LOST_LINE_BREAKS`, `READING_ORDER_ERROR`, `OUTPUT_TRUNCATION`, `EMPTY_OUTPUT`,
`MALFORMED_OUTPUT`, `CONFIDENCE_CALIBRATION_DISAGREEMENT`.

Three properties of this function matter more than the list itself:

**1. A reliability flag is not a failure.** A run can be flagged and still have
`outcome = "succeeded"`. The baseline's SATRN run is flagged both
`confidence_calibration_disagreement` and `omitted_text` and *succeeded* — it produced text, and the
text was wrong. "Did it run?" and "should you trust the output?" are separate questions and stay
separate.

**2. Adapter-reported failures pass through unchanged.** When the adapter itself already reported a
failure (SATRN's `cuda_oom`/`malformed_input`, Transkribus's `malformed_xml`), `classify_reliability`
passes that `FailureRecord` through **without reclassifying it** — Constitution Article 6: the adapter's
own category is the truth about *why the run failed*, and this module only adds findings for runs that
technically succeeded but are suspect.

**3. Thresholds are this function's own parameters, and a finding resting on one must say so.**
`CONFIDENCE_CALIBRATION_DISAGREEMENT` fires when the gap between reported confidence and measured
accuracy exceeds `classify_reliability`'s `confidence_disagreement_threshold` (default `0.35`);
`REPEATED_TEXT` fires when a word repeats consecutively at least `repeated_run_threshold` times
(`_detect_repeated_run`'s `minimum_run`).
`docs/research-findings.md`'s `satrn_confidence_not_aligned` carries this as an explicit limitation:
*"the 0.35 disagreement threshold is `classify_reliability`'s own parameter — a different threshold
would not have flagged the run."* A metric derived from a threshold is a claim about that threshold too.

**One honest modelling stretch, already on the record.** `classify_reliability` returns
`FailureRecord`s, so a reliability flag on a *succeeded* run is persisted both as
`ReliabilityIssueClassified` (the correct evaluation-layer kind) and as `MethodRunFailed` carrying the
record — and `MethodRunFailed` is documented as "a MethodRun that could not complete". The stretch is
`failures.py`'s own modelling choice and predates the telemetry work; it is listed in the baseline
README's "Honest gaps", not hidden.

## 7. Comparison boundaries and exclusions — the real decision this run made

**Transkribus is excluded from all CER and WER in the 2026-07-30 baseline. This is the protocol's
concrete worked example, and it was declared in advance.**

The reason, from `baseline_template.py::exclusion_criteria` verbatim:

> Transkribus's page-level result is unconditionally excluded from the controlled set and from any
> line-level CER computed against the shared fixture's ground truth, because its fixture's transcribed
> content is a different, unrelated Swedish court-record phrase with no established line-to-line
> correspondence to that ground truth — computing a CER between them would compare two unrelated
> texts, not measure recognition accuracy.

Four facts make this a boundary rather than an excuse:

1. **It is in the stored configuration**, inside `ExperimentVersion.pipeline_configuration_ref`,
   written before the run — so it cannot be a post-hoc rationalisation of an inconvenient number.
2. **The absence is total and asserted.**
   `tests/htr/persistence/test_real_baseline_reconstruction.py::test_no_invalid_transkribus_cer_or_wer_exists_anywhere_in_the_log`
   checks every recorded `MetricResult` *and* every `MetricCalculated` event, so the absence cannot be a
   projection artifact concealing a computed value.
3. **It is a null that was never computed, not a null written over a value.** There is no `MetricResult`
   carrying `None`; there is no `MetricResult` at all.
4. **It is capability-driven, not vendor-driven.** Transkribus is `page_level_supported`; the controlled
   comparison is line-level. Any method with the same capability profile and the same fixture
   non-correspondence would be excluded identically.

**What is still reported for Transkribus**: its transcription, its vendor-reported confidence (0.86333),
its geometry, and its own `ExperimentRun`. Exclusion from *a metric* is never exclusion from *the
record*.

**Exclusions must not silently shrink an aggregate.**
`failures.py::aggregate_method_run_metrics` reports, side by side and always: a succeeded-only mean, an
all-runs-including-failures mean (misses scored as CER 1.0), and a `failure_categories` breakdown that
is never empty when failures exist. This mirrors `evaluation/evaluate.py::_summaries`' existing
`mean_cer_all` convention rather than inventing a second aggregation philosophy. The brief's requirement
— *"must NOT silently exclude failures from aggregate metrics"* — is met by reporting both numbers,
never by choosing one.

## 8. Segmentation metrics, in full

`htr/evaluation/segmentation.py`, computable only for a method reporting `geometry_supported = yes`.

| Group | Functions | Definition constants |
|---|---|---|
| Geometry matching | `iou`, `intersection_area`, `area`, `match_geometry` → `GeometryMatchResult` | `MEAN_MATCHED_IOU` |
| Precision / recall | `precision_recall` → `PrecisionRecallResult` | `REGION_PRECISION`, `REGION_RECALL`, `LINE_PRECISION`, `LINE_RECALL` |
| Line-count errors | `missed_line_rate`, `duplicate_line_rate`, `merged_line_rate`, `split_line_rate` | `MISSED_LINE_RATE`, `DUPLICATE_LINE_RATE`, `MERGED_LINE_RATE`, `SPLIT_LINE_RATE` |
| Ordering | `reading_order_accuracy`, `region_order_accuracy` (both via `_order_concordance`) | `READING_ORDER_ACCURACY`, `REGION_ORDER_ACCURACY` |
| Crop quality | `crop_coverage`, `crop_contamination` | `CROP_COVERAGE`, `CROP_CONTAMINATION` |

`segmentation_metric_results` assembles these into `MetricResult` entities.

**Merged and split lines are counted separately from missed and duplicated ones**, because they are
different segmenter behaviours: a merge joins two reference lines into one hypothesis line, a split does
the reverse, and both would appear as an uninterpretable mixture of "missed" and "duplicated" if
collapsed into those two.

**`crop_coverage` and `crop_contamination` return `float | None`.** `None` means "not computable for
this input", not zero — the same discipline as a missing confidence, and for the same reason.

**No segmentation run was recorded in the baseline**, deliberately: the controlled fixture is already a
pre-cropped line, and Transkribus's segmentation ran externally. Emitting `SegmentationRunCompleted` for
either would claim an execution that did not happen.

## 9. Historical-feature evaluation

`htr/evaluation/historical_features.py` — the mechanism that replaced the deleted NER-style
`ObservationType.NAMED_ENTITY`/`RELATIONSHIP` ontology types. Historical-feature accuracy is an
*evaluator*, not an ontology observation type (`docs/htr-repository-cleanup.md`'s domain-model table).

`HISTORICAL_FEATURE_REGISTRY` holds four `HistoricalFeatureEvaluator` implementations:
`PersonalNameEvaluator`, `HistoricalDateEvaluator`, `AbbreviationEvaluator` and
`UnusualCharacterEvaluator` (the last over `ARCHAIC_CHARACTERS` against
`_STANDARD_SWEDISH_CHARACTERS`). `evaluate_feature` produces a `HistoricalFeatureEvaluationResult`;
`historical_feature_metric_results` turns it into `HISTORICAL_FEATURE_PRECISION` /
`HISTORICAL_FEATURE_RECALL` `MetricResult`s.

## 10. Operational metrics are measurements, not quality

`htr/evaluation/operational.py::operational_metric_results` emits `EXECUTION_TIME_MS` and
`GPU_MEMORY_MB` from `Evidence`; `aggregate_operational_metrics` produces an `OperationalAggregate`.
These are **not** accuracy metrics and are not comparable across machines, sessions, or — for GPU memory
— allocator states.

The baseline demonstrates why that caveat is load-bearing rather than boilerplate: Florence-2's peak GPU
memory measured 1210.64 MiB here against ~3983 MB documented from an earlier session — a factor of ~3.3
— while SATRN reproduced to within 0.3 MB across the same pair of sessions. That disagreement is the
entire basis of the `Disputed` finding `florence2_environment_reproducible`
(`docs/research-findings.md`). Operational numbers are recorded, and they are not treated as stable.

## 11. Reference reliability classification, and the retained blind-review workflow

A metric is only as good as its reference, so references are classified too. Three tiers, and the
baseline sits in the weakest one.

| Tier | What it means | Machinery |
|---|---|---|
| **Adjudicated** | two independent blind reviewers, disagreement resolved by a third with a required rationale | `review/blind_review/` + `evaluation/ground_truth.py` |
| **Agreed** | two independent blind reviewers agreeing within threshold | `blind_review/agreement.py` → `AgreementResult` |
| **Single external annotation** | one transcription taken verbatim from an upstream published dataset | what the baseline actually has |

The retained workflow — this document's original subject, unchanged and still the mechanism:

* `blind_review/assignment.py::create_blind_review_pair` creates the two `ReviewAssignment`s and
  **refuses to assign one person both roles** ("a blind dual review by one person is not a blind dual
  review at all").
* `blind_review/store.py::BlindReviewStore.submission_for_other_reviewer` **mechanically refuses** a
  cross-read until the requesting reviewer's own submission is finalized — enforced, not merely
  documented.
* `blind_review/agreement.py::AgreementPolicy` classifies by **normalized** CER, at
  `AGREEMENT_POLICY_VERSION`, with two visible versioned constants rather than magic numbers:
  `minor_disagreement_cer_max = 0.02` (≈ one character on a 60-character court-record line — what two
  careful readers routinely differ by) and `material_disagreement_cer_max = 0.15`. `AGREED` at or below
  2%, `MINOR_DISAGREEMENT` up to 15%, `REQUIRES_ADJUDICATION` above. Both are documented in that class
  as **deliberate placeholders pending real inter-annotator-agreement calibration**, so a better-
  calibrated version supersedes rather than silently overwrites them.
* An **illegibility mismatch** (one reviewer marks illegible, the other transcribes) is always
  `REQUIRES_ADJUDICATION` regardless of any numeric threshold, because there is no CER to compute
  between text and "no text".
* `blind_review/adjudication.py::adjudicate` refuses to adjudicate an item not classified
  `REQUIRES_ADJUDICATION`, and requires a non-empty `rationale`.
* All three positions stay independently readable afterwards: `blind_review/outcome.py::BenchmarkOutcome`
  derives a view without mutating anything, and adjudicating a contested item does **not** retroactively
  reclassify it as `AGREED` — `status` keeps the original computed classification.

**The baseline's reference is tier 3, and every finding derived from it says so.** From
`baseline_template.py::ground_truth_requirements`: *"No blind dual-annotation + adjudication was
performed for this ground truth … it is a single external annotation, not this project's own adjudicated
ground truth."* This is limitation 3 of `baseline_knowledge.SHARED_LIMITATIONS`, present on all five
findings.

**No `TranscriptionConvention` record exists for it either.**
`ResearchFinding.transcription_convention` is `None` on all five findings, and the field is nullable
precisely so a fabricated convention id was not forced. A convention governs how uncertain characters,
abbreviations and line breaks are transcribed; without one, two references are not strictly comparable
even under the same normalization profile.

**Retention of review and adjudication evidence** — including the reviewer-identity question — is
[`docs/telemetry-retention.md`](telemetry-retention.md) §§8–9, not this document.

## 12. Scope limitations: what a measurement from this engine may claim

The engine computes correct numbers over whatever it is given. What those numbers *mean* is bounded by
`ResearchScope`, which is structural rather than advisory — `sample_size` is a **derived property**
(`len(covered_unit_ids)`), so it cannot be inflated or omitted, and placeholders (`"*"`, `"all"`, `""`)
are refused outright.

For the 2026-07-30 baseline, the honest bound is:

* **N=1.** One line crop, one experiment run, one checkpoint per method, for the controlled comparison.
  One hand-authored page for the end-to-end result.
* **No ranking of methods is inferable.** Two error rates on one crop are two measurements, not a
  ranking. This is limitation 2 of `SHARED_LIMITATIONS`, and `florence2_lower_error_rates` — the
  finding that Florence-2 scored better — carries it as one of its own stated limitations.
* **Confidence is not comparable across methods.** SATRN's is the model's own scalar, Florence-2's a
  proxy `exp(sequences_scores)`, Transkribus's the vendor's. See
  `docs/CAPABILITY_MATRIX_HTR.md` §1; `florence2_lower_error_rates` therefore rests on CER/WER only.
* **Calibration is unmeasurable at this N.** Calibration is a distributional property and one sample
  cannot estimate it. The confidence-anomaly observation tags itself
  `this_run_and_this_sample_only` and `not_a_calibration_claim`, and the open research question
  `satrn_confidence_calibration_question` exists precisely because of that gap.
* **`Supported` is unreachable here.** `htr/knowledge/lifecycle.py::transition_finding_status` requires
  reproduction evidence from an experiment run *outside* the finding's own scope, and this repository has
  one controlled run. The honest ceiling reached is `Provisionally supported`. That refusal is the
  mechanism working, not a limitation worked around.
* **The Transkribus fixture is hand-authored**, not genuine vendor output
  (`tests/fixtures/transkribus/sample_page.xml`), and its `TextEquiv` content is itself a stand-in for a
  recognition result rather than a reference transcription.

## 13. Reproducibility requirements

A metric is quotable only if the run that produced it is reconstructable without re-running a model.

| Requirement | Mechanism |
|---|---|
| Metric definitions are versioned | `definitions.EVALUATION_ENGINE_VERSION = 1`; every `MetricResult` names its `MetricDefinition` |
| Metric functions are pure | `htr/evaluation/*` has no I/O, no store reference, no telemetry — persistence is the caller's job |
| Every result is replayable from the log | `HtrJournal.replay` rebuilds the whole graph; `test_real_baseline_reconstruction.py` asserts the committed `research_report.json` regenerates from it field for field |
| Environment is captured | `ReproducibilityManifest` + `Evidence.software_environment` / `hardware_environment` / `execution_device` |
| Model revisions are pinned | each adapter's `get_metadata().model_revision`; `ResearchScope` pairs `method_ids` with `model_version_ids` positionally and validates equal length |
| The input is hash-verified | `InputCrop.hash`, recomputed from the fixture on every test run |

**The purity of `htr/evaluation/*` is a deliberate constraint, not an accident.**
`docs/htr-telemetry-knowledge-gap-analysis.md` §7 classifies these modules as *"Already sufficient as
pure functions — the gap is entirely in what the caller does with their output"*, and §10 forbids
folding them into a persistence layer. A metric function that wrote telemetry could not be called twice
to check itself.

## 14. Related documents

| Document | What it covers that this one does not |
|---|---|
| [`docs/CAPABILITY_MATRIX_HTR.md`](CAPABILITY_MATRIX_HTR.md) | Which method can produce what, and why three capability flags mislead |
| [`docs/research-observations.md`](research-observations.md) | The five real observations extracted from the baseline |
| [`docs/research-findings.md`](research-findings.md) | The five real findings and their statuses |
| [`docs/knowledge-lifecycle.md`](knowledge-lifecycle.md) | How a finding advances past `Candidate`, and who may do it |
| [`docs/experiments/baseline-comparison/README.md`](experiments/baseline-comparison/README.md) | The executed run: every measured number, and its honest gaps |
| [`docs/telemetry-retention.md`](telemetry-retention.md) | How long evaluation and review evidence is kept, and reviewer privacy |
| [`HUMAN_REVIEW_SPECIFICATION.md`](../HUMAN_REVIEW_SPECIFICATION.md) | The operational review workflow, which is distinct from evaluation |
