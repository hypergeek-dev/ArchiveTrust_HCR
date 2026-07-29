"""Versioned `MetricDefinition` registry for the HTR evaluation engine
(docs/htr-migration-plan.md Stage 9; docs/htr-domain-design.md §3: "`MetricDefinition`: versioned
like `METRICS_VERSION` already is in `evaluation/metrics.py` -- extended, not replaced").

Every metric this package computes has exactly one `MetricDefinition` here, created once at import
time and referenced by every module below -- so a `MetricResult.metric_definition_id` always
resolves to a stable name/version pair within a process, mirroring how `evaluation/metrics.py`
stamps every output with `METRICS_VERSION`. `EVALUATION_ENGINE_VERSION` is bumped whenever any
metric's *meaning* changes (not when a new metric is merely added).

`MetricDefinition.metric_definition_id` is a random id (`new_id`, see `domain/shared/ids.py`)
generated once per process import, exactly like every other `MetricDefinition.create()` call site
in this codebase (see `tests/htr/experiment/test_models.py`) -- callers key on the module-level
*name* (e.g. `CHARACTER_ERROR_RATE_NORMALIZED`), never on a hand-typed id string.
"""

from __future__ import annotations

from archivetrust.htr.experiment.models import MetricDefinition

EVALUATION_ENGINE_VERSION = 1
"""Recorded as every `MetricDefinition.version` below. Bump when any metric's *definition*
changes meaning (e.g. redefining what counts as a "line" for exact-line accuracy) -- not when a
new metric is added alongside existing, unchanged ones."""


def _definition(name: str, *, description: str, higher_is_better: bool) -> MetricDefinition:
    return MetricDefinition.create(
        name=name,
        version=EVALUATION_ENGINE_VERSION,
        description=description,
        higher_is_better=higher_is_better,
    )


# --- Recognition metrics (recognition.py) --------------------------------------------------

CHARACTER_ERROR_RATE_RAW = _definition(
    "character_error_rate_raw",
    description=(
        "Levenshtein character edit distance / reference length, computed on the exact raw "
        "strings with no normalization applied to either side. Distinct from "
        "character_error_rate_normalized -- see that definition's description for what "
        "'normalized' means here."
    ),
    higher_is_better=False,
)
CHARACTER_ERROR_RATE_NORMALIZED = _definition(
    "character_error_rate_normalized",
    description=(
        "Levenshtein character edit distance / reference length, after "
        "evaluation.metrics.normalize_text (Unicode NFC composition + whitespace-run collapse, "
        "no case folding) is applied to both reference and hypothesis. Reuses "
        "evaluation.metrics.compare_text -- not a reimplementation."
    ),
    higher_is_better=False,
)
WORD_ERROR_RATE_RAW = _definition(
    "word_error_rate_raw",
    description=(
        "Word-level Levenshtein edit distance / reference word count, tokenized by str.split() "
        "on the raw, non-NFC-normalized strings."
    ),
    higher_is_better=False,
)
WORD_ERROR_RATE_NORMALIZED = _definition(
    "word_error_rate_normalized",
    description=(
        "Word-level Levenshtein edit distance / reference word count, after "
        "evaluation.metrics.normalize_text is applied to both sides before tokenizing. Reuses "
        "evaluation.metrics.compare_text."
    ),
    higher_is_better=False,
)
EXACT_LINE_ACCURACY = _definition(
    "exact_line_accuracy",
    description=(
        "Fraction of index-aligned reference/hypothesis line pairs (after normalize_text on "
        "each line) that match exactly. Assumes the caller supplies reference and hypothesis "
        "lines in the same reading-order alignment -- it does not itself solve line alignment "
        "(see segmentation.py for that when ground-truth geometry exists)."
    ),
    higher_is_better=True,
)
EXACT_WORD_ACCURACY = _definition(
    "exact_word_accuracy",
    description=(
        "1 - (word substitutions + word deletions) / reference word count, after "
        "normalize_text. Distinct from 1 - word_error_rate_normalized: WER's numerator also "
        "counts insertions, so word accuracy and (1 - WER) diverge whenever the hypothesis "
        "contains inserted words."
    ),
    higher_is_better=True,
)
CHARACTER_INSERTIONS = _definition(
    "character_insertions",
    description="Count of character-level insertion edits (normalized text), from the "
    "edit-operation-classifying Levenshtein backtrace in recognition.classify_char_edits.",
    higher_is_better=False,
)
CHARACTER_DELETIONS = _definition(
    "character_deletions",
    description="Count of character-level deletion edits (normalized text).",
    higher_is_better=False,
)
CHARACTER_SUBSTITUTIONS = _definition(
    "character_substitutions",
    description="Count of character-level substitution edits (normalized text).",
    higher_is_better=False,
)
WORD_INSERTIONS = _definition(
    "word_insertions",
    description="Count of word-token insertion edits (normalized text), from "
    "recognition.classify_word_edits.",
    higher_is_better=False,
)
WORD_DELETIONS = _definition(
    "word_deletions",
    description="Count of word-token deletion edits (normalized text).",
    higher_is_better=False,
)
WORD_SUBSTITUTIONS = _definition(
    "word_substitutions",
    description="Count of word-token substitution edits (normalized text).",
    higher_is_better=False,
)

# --- Historical-document metrics (historical_features.py) -----------------------------------

HISTORICAL_FEATURE_PRECISION = _definition(
    "historical_feature_precision",
    description=(
        "Per HistoricalFeatureEvaluator: |matched hypothesis spans| / |all hypothesis spans|, "
        "set-based over normalized span text. One MetricResult per (feature_name, method_run)."
    ),
    higher_is_better=True,
)
HISTORICAL_FEATURE_RECALL = _definition(
    "historical_feature_recall",
    description="Per HistoricalFeatureEvaluator: |matched reference spans| / |all reference spans|.",
    higher_is_better=True,
)
HISTORICAL_FEATURE_F1 = _definition(
    "historical_feature_f1",
    description="Harmonic mean of historical_feature_precision and historical_feature_recall.",
    higher_is_better=True,
)

# --- Segmentation metrics (segmentation.py) --------------------------------------------------

REGION_PRECISION = _definition(
    "region_precision",
    description="Matched hypothesis regions (IoU >= threshold, greedy one-to-one) / all "
    "hypothesis regions.",
    higher_is_better=True,
)
REGION_RECALL = _definition(
    "region_recall",
    description="Matched reference regions / all reference regions.",
    higher_is_better=True,
)
LINE_PRECISION = _definition(
    "line_precision",
    description="Matched hypothesis text lines (IoU >= threshold, greedy one-to-one) / all "
    "hypothesis text lines.",
    higher_is_better=True,
)
LINE_RECALL = _definition(
    "line_recall",
    description="Matched reference text lines / all reference text lines.",
    higher_is_better=True,
)
MEAN_MATCHED_IOU = _definition(
    "mean_matched_iou",
    description="Mean axis-aligned bounding-box IoU over every matched (reference, hypothesis) "
    "pair -- not over unmatched geometry.",
    higher_is_better=True,
)
MISSED_LINE_RATE = _definition(
    "missed_line_rate",
    description="Reference lines with no matching hypothesis line (IoU >= threshold) / all "
    "reference lines.",
    higher_is_better=False,
)
DUPLICATE_LINE_RATE = _definition(
    "duplicate_line_rate",
    description="Hypothesis lines whose best-IoU reference line already has a distinct "
    "best-matching hypothesis line (over-prediction of the same true line) / all hypothesis "
    "lines.",
    higher_is_better=False,
)
MERGED_LINE_RATE = _definition(
    "merged_line_rate",
    description="Hypothesis lines whose bounding box has IoU >= threshold with more than one "
    "reference line (under-segmentation: two true lines merged into one predicted line) / all "
    "hypothesis lines.",
    higher_is_better=False,
)
SPLIT_LINE_RATE = _definition(
    "split_line_rate",
    description="Reference lines whose bounding box has IoU >= threshold with more than one "
    "hypothesis line (over-segmentation: one true line split into multiple predicted lines) / "
    "all reference lines.",
    higher_is_better=False,
)
READING_ORDER_ACCURACY = _definition(
    "reading_order_accuracy",
    description="Fraction of matched-pair combinations (over TextLine.reading_order_index) "
    "whose relative order agrees between reference and hypothesis (pairwise order concordance).",
    higher_is_better=True,
)
REGION_ORDER_ACCURACY = _definition(
    "region_order_accuracy",
    description="Same as reading_order_accuracy, computed over Region.order_index instead of "
    "TextLine.reading_order_index.",
    higher_is_better=True,
)
CROP_COVERAGE = _definition(
    "crop_coverage",
    description="Sum over reference lines of that line's best single-crop intersection area, "
    "divided by total reference line area. Axis-aligned-bbox approximation: does not compute a "
    "true multi-crop union, so it is a lower bound on true coverage when crops overlap the same "
    "line.",
    higher_is_better=True,
)
CROP_CONTAMINATION = _definition(
    "crop_contamination",
    description="Sum over hypothesis crops of (crop area - that crop's best single-reference-"
    "line overlap), divided by total crop area -- how much of the produced crops falls outside "
    "any true line.",
    higher_is_better=False,
)

# --- Operational metrics (operational.py) ----------------------------------------------------

EXECUTION_TIME_MS = _definition(
    "execution_time_ms",
    description="Evidence.execution_time_ms, aggregated verbatim -- no new timing capture, "
    "only what the adapter already recorded.",
    higher_is_better=False,
)
GPU_MEMORY_MB = _definition(
    "gpu_memory_mb",
    description="Evidence.gpu_memory_mb, aggregated verbatim.",
    higher_is_better=False,
)
