"""Versioned, configurable, reference-free reliability heuristics.

Every heuristic in this module satisfies three requirements the screening design imposes:

1. **Reference-free.** Computable from the output string, the crop geometry and the run telemetry
   alone. No ground truth is read, because none exists for this corpus.
2. **Versioned.** `RELIABILITY_HEURISTICS_VERSION` changes whenever a threshold's *meaning* or a
   rule's behaviour changes, so a recorded result names the rule set that produced it. Thresholds
   additionally travel in `ReliabilityThresholds`, which is frozen and hashable, so a report can
   state the exact configuration rather than the module default.
3. **Documented with its threshold.** `HEURISTIC_CATALOG` pairs every heuristic id with what it
   detects, which threshold governs it, and -- importantly -- what it does *not* mean. That last
   field exists because these signals are easy to over-read: "flagged" here means "worth a human
   look", never "wrong".

**No accuracy signal is computed here and none may be added.** CER, WER, exact-match and every
other reference-comparing quantity are out of scope by construction, not by omission.

## On the two aggregate heuristics

Most of these are per-output predicates. Two are not, and they are the ones that carry the most
information at scale:

* **`repeated_output_across_inputs`** -- a method can execute successfully on every crop, return
  non-empty non-repetitive text every time, and still be broken, if it returns nearly the *same*
  text regardless of what it is shown. No per-string check can see that. It is only visible across
  a population of distinct inputs, which is exactly what a 60-page run provides and a 15-crop smoke
  test could not.
* **`output_length_outlier`** -- "too long" and "too short" have no absolute meaning for a line of
  17th-century secretary hand, so the bound is derived from the method's own observed distribution
  by a median/MAD rule rather than fixed in advance.

Both are therefore functions of a whole run, and both are reported *per method, never pooled*: the
baseline comparison already established that these two recognizers' confidence scales are not
comparable, and the same discipline applies to their output-shape distributions.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

RELIABILITY_HEURISTICS_VERSION = "1.0.0"
"""Version of the *rule set*, not of this file. Bumped when a rule's behaviour or a threshold's
meaning changes -- so a recorded reliability result can always be traced to the rules that produced
it, even after the rules move on."""


class ReliabilityThresholds(BaseModel):
    """Every tunable number the heuristics use, in one frozen, hashable object.

    Frozen so a run cannot mutate its own thresholds partway through; hashable
    (`configuration_hash`) so a report can prove which configuration it was computed under, in the
    same way `RgbNormalizationConfig.configuration_hash` does for the normalization stage.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    degenerate_min_length: int = Field(
        default=12,
        description=(
            "Below this many non-whitespace-stripped characters, repetition is not evidence of "
            "anything -- 'aa' or 'no no' are ordinary short outputs, not decoder loops."
        ),
    )
    degenerate_max_distinct_characters: int = Field(
        default=2,
        description=(
            "An output of at least degenerate_min_length built from at most this many distinct "
            "characters is a degenerate loop (e.g. 'aaaaaaaaaaaaaa')."
        ),
    )
    degenerate_ngram_sizes: tuple[int, ...] = Field(
        default=(2, 3, 4),
        description="Repeating-unit sizes tested for a whole-string tiling loop.",
    )
    degenerate_min_ngram_repeats: int = Field(
        default=6,
        description=(
            "A unit must tile the string at least this many times before the tiling is called a "
            "loop, so 'ababab' (3) is not flagged but 'abababababab' (6) is."
        ),
    )

    very_short_output_max_length: int = Field(
        default=2,
        description=(
            "Outputs of at most this many characters are flagged `very_short`. Not an error: a "
            "correctly-detected folio number genuinely is one or two characters. It is flagged "
            "because it is the regime in which the recognizers were observed to behave oddly, and "
            "because such a crop is usually not a text line at all."
        ),
    )

    length_outlier_modified_z: float = Field(
        default=3.5,
        description=(
            "Iglewicz-Hoaglin modified z-score cutoff, |0.6745*(x-median)/MAD| > this, computed "
            "over one method's own successful output lengths. 3.5 is that method's standard "
            "recommendation and is used unchanged rather than tuned to this corpus, because tuning "
            "an outlier rule on the data it will judge is how a screening becomes circular."
        ),
    )
    length_outlier_min_sample: int = Field(
        default=20,
        description=(
            "Below this many successful outputs, no length-outlier bound is computed at all and "
            "the signal reports `not_assessable` rather than a bound derived from too few points."
        ),
    )

    truncation_char_length: Mapping[str, int] = Field(
        default={"satrn": 95, "florence2": 900},
        description=(
            "Per-method character length at or above which an output is a truncation suspect. "
            "SATRN: its checkpoint's NRTRDecoder declares max_seq_len=100 (read from the "
            "checkpoint's own config.py during the repetition diagnostic), so an output at 95+ "
            "characters is at the decoder's ceiling and may have been cut. Florence-2: generation "
            "is bounded by max_new_tokens, and 900 characters is the conservative character-side "
            "proxy for approaching it. These are suspicion thresholds, not proof -- a genuinely "
            "long line can reach them honestly."
        ),
    )

    distinct_output_ratio_floor: float = Field(
        default=0.9,
        description=(
            "A method whose distinct-output ratio over distinct inputs falls below this is flagged "
            "`repeated_output_across_inputs`. Set high deliberately: over a population of lines "
            "from different documents, near-total distinctness is the expected result, and any "
            "material shortfall is the signal. It is not a pass mark and implies no verdict."
        ),
    )
    repeat_group_min_size: int = Field(
        default=2,
        description=(
            "Number of *distinct* inputs that must yield the identical output before those outputs "
            "form a repeat group. Two is the minimum that can mean anything."
        ),
    )

    crop_min_aspect_ratio: float = Field(
        default=2.0,
        description=(
            "Width/height below which a crop is flagged `crop_geometry_implausible` -- a text line "
            "of running hand is a wide strip, so a square or portrait crop is very unlikely to be "
            "a line. This is a *segmentation* signal attached to the crop, not a recognizer "
            "signal, and the SATRN repetition diagnostic recorded exactly this shape of crop "
            "(aspect 0.70-1.07) among its inputs."
        ),
    )

    @property
    def configuration_hash(self) -> str:
        """Deterministic sha256 over the resolved thresholds, so a report states its configuration
        by content rather than by reference to a default that may since have changed."""
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return "reliability_thresholds_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class HeuristicDefinition:
    """One documented heuristic: what it is, what governs it, and what it must not be read as."""

    heuristic_id: str
    scope: str
    """`output` (per recognized crop), `crop` (per input crop), or `run` (per method, whole run)."""
    detects: str
    threshold_fields: tuple[str, ...]
    does_not_mean: str


HEURISTIC_CATALOG: tuple[HeuristicDefinition, ...] = (
    HeuristicDefinition(
        heuristic_id="execution_failed",
        scope="output",
        detects="The adapter returned no text at all: a hard failure with a typed cause.",
        threshold_fields=(),
        does_not_mean=(
            "Not necessarily a method defect -- an environment failure (missing venv, OOM, "
            "timeout) is recorded here too and is separated by the failure category, never "
            "collapsed into 'the method failed'."
        ),
    ),
    HeuristicDefinition(
        heuristic_id="empty_or_whitespace_only",
        scope="output",
        detects="Execution succeeded but the returned text is empty or only whitespace.",
        threshold_fields=(),
        does_not_mean=(
            "Not proof the crop is blank. A blank crop *should* produce empty output; this signal "
            "cannot distinguish the two without looking at the image, which is a human-review task."
        ),
    ),
    HeuristicDefinition(
        heuristic_id="degenerate_repetition",
        scope="output",
        detects=(
            "A character or n-gram repetition loop within a single output -- a known VLM/seq2seq "
            "decoder degradation mode."
        ),
        threshold_fields=(
            "degenerate_min_length",
            "degenerate_max_distinct_characters",
            "degenerate_ngram_sizes",
            "degenerate_min_ngram_repeats",
        ),
        does_not_mean=(
            "Distinct from `repeated_output_across_inputs`: this is repetition *inside* one "
            "output. The two are independent and an output can trip either, both, or neither."
        ),
    ),
    HeuristicDefinition(
        heuristic_id="very_short_output",
        scope="output",
        detects="An output of at most `very_short_output_max_length` characters.",
        threshold_fields=("very_short_output_max_length",),
        does_not_mean=(
            "Not a defect. Folio numbers and marginal marks are legitimately this short. It is "
            "recorded because it marks the input regime where degradation was already observed."
        ),
    ),
    HeuristicDefinition(
        heuristic_id="output_length_outlier",
        scope="run",
        detects=(
            "An output whose length is a median/MAD outlier against that same method's own "
            "observed length distribution on this run."
        ),
        threshold_fields=("length_outlier_modified_z", "length_outlier_min_sample"),
        does_not_mean=(
            "Never compared across methods. The two recognizers have different decoders and "
            "different length behaviour; pooling their lengths would manufacture outliers that are "
            "only differences between methods."
        ),
    ),
    HeuristicDefinition(
        heuristic_id="truncation_suspected",
        scope="output",
        detects=(
            "An output at or above the method's configured character ceiling, or a run the adapter "
            "itself reported as truncated."
        ),
        threshold_fields=("truncation_char_length",),
        does_not_mean=(
            "Not confirmed truncation. Without ground truth there is no way to know whether text "
            "was lost; this flags the outputs where it is possible."
        ),
    ),
    HeuristicDefinition(
        heuristic_id="repeated_output_across_inputs",
        scope="run",
        detects=(
            "The same output string returned for two or more *demonstrably different* input crops "
            "(different content hashes), aggregated into repeat groups and a distinct-output ratio."
        ),
        threshold_fields=("distinct_output_ratio_floor", "repeat_group_min_size"),
        does_not_mean=(
            "Not an accuracy statement, and not automatically a bug: the SATRN repetition "
            "diagnostic established this behaviour as genuine model behaviour on short and "
            "distorted crops, not an integration defect. It is a reliability signal about how much "
            "the output depends on the input."
        ),
    ),
    HeuristicDefinition(
        heuristic_id="crop_geometry_implausible",
        scope="crop",
        detects="A crop whose width/height is below `crop_min_aspect_ratio` -- not line-shaped.",
        threshold_fields=("crop_min_aspect_ratio",),
        does_not_mean=(
            "A segmentation signal, not a recognizer signal. It says the detector produced "
            "something that is probably not a text line; it says nothing about what any recognizer "
            "then did with it."
        ),
    ),
    HeuristicDefinition(
        heuristic_id="timing_distribution",
        scope="run",
        detects="Median and p90 wall-clock per method, plus min/max. Descriptive, never a flag.",
        threshold_fields=(),
        does_not_mean=(
            "Not comparable between methods as a quality signal: SATRN restarts a subprocess and "
            "reloads its model on every call while Florence-2 caches in-process, so their timings "
            "measure different execution models, not different efficiencies."
        ),
    ),
    HeuristicDefinition(
        heuristic_id="gpu_memory",
        scope="run",
        detects="Peak GPU memory observed per method, where the runtime reported it.",
        threshold_fields=(),
        does_not_mean="Descriptive only. No threshold, no flag, and absent where unreported.",
    ),
)


class OutputSignals(BaseModel):
    """The per-output reference-free signals for one recognized crop."""

    model_config = ConfigDict(frozen=True)

    produced_output: bool
    failure_category: str | None = None
    failure_message: str | None = None
    output_length: int | None = None
    empty_or_whitespace_only: bool = False
    degenerate_repetition: bool = False
    very_short_output: bool = False
    truncation_suspected: bool = False


def _is_degenerate(text: str, thresholds: ReliabilityThresholds) -> bool:
    """Character/n-gram repetition loop inside a single output.

    Kept behaviourally identical to the rule the Checkpoint 3 smoke test ran, so smoke-test and
    full-run degeneracy counts remain comparable; the only change is that its constants are now
    configuration rather than literals.
    """
    stripped = text.strip()
    if len(stripped) < thresholds.degenerate_min_length:
        return False
    if len(set(stripped)) <= thresholds.degenerate_max_distinct_characters:
        return True
    for size in thresholds.degenerate_ngram_sizes:
        if len(stripped) >= size * thresholds.degenerate_min_ngram_repeats:
            unit = stripped[:size]
            repeats = len(stripped) // size
            if unit * repeats == stripped[: size * repeats]:
                return True
    return False


def evaluate_output(
    text: str | None,
    *,
    method: str,
    raw_response: Mapping[str, object] | None = None,
    thresholds: ReliabilityThresholds | None = None,
) -> OutputSignals:
    """All per-output signals for one recognition result.

    `text is None` means the adapter failed and is recorded as such with its typed category; it is
    never conflated with an empty string, which means the adapter succeeded and returned nothing.
    """
    thresholds = thresholds or ReliabilityThresholds()
    raw = dict(raw_response or {})
    if text is None:
        return OutputSignals(
            produced_output=False,
            failure_category=(
                str(raw.get("category")) if raw.get("category") is not None else None
            ),
            failure_message=(str(raw.get("message")) if raw.get("message") is not None else None),
        )

    ceiling = thresholds.truncation_char_length.get(method)
    adapter_reported_truncation = bool(raw.get("output_truncated", False))
    return OutputSignals(
        produced_output=True,
        output_length=len(text),
        empty_or_whitespace_only=not text.strip(),
        degenerate_repetition=_is_degenerate(text, thresholds),
        very_short_output=len(text.strip()) <= thresholds.very_short_output_max_length,
        truncation_suspected=(
            adapter_reported_truncation or (ceiling is not None and len(text) >= ceiling)
        ),
    )


def percentile(values: Sequence[float], fraction: float) -> float | None:
    """Linear-interpolated percentile. Returns `None` for an empty sequence rather than raising or
    inventing a zero -- "no observations" and "a value of zero" are different facts."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def _median(values: Sequence[float]) -> float | None:
    return percentile(values, 0.5)


def length_outlier_bounds(
    lengths: Sequence[int], *, thresholds: ReliabilityThresholds | None = None
) -> dict[str, object]:
    """Median/MAD outlier bounds over one method's own output lengths.

    Returns `assessable=False` below `length_outlier_min_sample`, and also when MAD is zero (every
    output the same length), because a zero MAD makes the modified z-score undefined rather than
    infinite -- reporting every point as an outlier there would be an artifact of the formula, not
    a finding.
    """
    thresholds = thresholds or ReliabilityThresholds()
    if len(lengths) < thresholds.length_outlier_min_sample:
        return {
            "assessable": False,
            "reason": (
                f"only {len(lengths)} successful outputs; the rule requires at least "
                f"{thresholds.length_outlier_min_sample}"
            ),
            "sample_size": len(lengths),
        }
    values = [float(x) for x in lengths]
    median = _median(values)
    assert median is not None
    mad = _median([abs(v - median) for v in values])
    assert mad is not None
    if mad == 0.0:
        return {
            "assessable": False,
            "reason": (
                "median absolute deviation is zero (every output has the same length), which makes "
                "the modified z-score undefined"
            ),
            "sample_size": len(lengths),
            "median": median,
        }
    half_width = thresholds.length_outlier_modified_z * mad / 0.6745
    return {
        "assessable": True,
        "sample_size": len(lengths),
        "median": median,
        "median_absolute_deviation": mad,
        "modified_z_cutoff": thresholds.length_outlier_modified_z,
        "lower_bound": median - half_width,
        "upper_bound": median + half_width,
    }


class RepeatedOutputSummary(BaseModel):
    """Aggregate repeated-output-across-different-inputs result for one method on one run."""

    model_config = ConfigDict(frozen=True)

    outputs: int
    distinct_inputs: int
    distinct_outputs: int
    distinct_ratio: float | None
    outputs_in_repeat_groups: int
    repeat_rate: float | None
    repeat_group_count: int
    largest_repeat_group_size: int
    most_repeated_output: str | None
    flagged: bool
    repeat_groups: tuple[dict, ...] = ()


_WHITESPACE = re.compile(r"\s+")


def repeated_output_summary(
    observations: Sequence[tuple[str, str]],
    *,
    thresholds: ReliabilityThresholds | None = None,
    max_groups_reported: int = 25,
) -> RepeatedOutputSummary:
    """Repeated-output detection over `(input_hash, output_text)` pairs.

    Takes the **input hash** rather than an index so the "different inputs" half of the claim is
    verified rather than assumed: two identical outputs for the same crop hash are not a repeat,
    and are excluded from every group. Comparison is on whitespace-collapsed text, so a repeat is
    not missed over a trailing space; the reported group text is the collapsed form.
    """
    thresholds = thresholds or ReliabilityThresholds()
    by_text: dict[str, set[str]] = {}
    for input_hash, text in observations:
        collapsed = _WHITESPACE.sub(" ", text).strip()
        by_text.setdefault(collapsed, set()).add(input_hash)

    total_outputs = len(observations)
    distinct_inputs = len({h for h, _ in observations})
    distinct_outputs = len(by_text)
    groups = [
        {"output": text, "distinct_inputs": len(hashes)}
        for text, hashes in by_text.items()
        if len(hashes) >= thresholds.repeat_group_min_size
    ]
    groups.sort(key=lambda g: (-int(g["distinct_inputs"]), str(g["output"])))
    in_groups = sum(int(g["distinct_inputs"]) for g in groups)
    distinct_ratio = (distinct_outputs / distinct_inputs) if distinct_inputs else None
    return RepeatedOutputSummary(
        outputs=total_outputs,
        distinct_inputs=distinct_inputs,
        distinct_outputs=distinct_outputs,
        distinct_ratio=round(distinct_ratio, 4) if distinct_ratio is not None else None,
        outputs_in_repeat_groups=in_groups,
        repeat_rate=round(in_groups / distinct_inputs, 4) if distinct_inputs else None,
        repeat_group_count=len(groups),
        largest_repeat_group_size=max((int(g["distinct_inputs"]) for g in groups), default=0),
        most_repeated_output=str(groups[0]["output"]) if groups else None,
        flagged=(
            distinct_ratio is not None and distinct_ratio < thresholds.distinct_output_ratio_floor
        ),
        repeat_groups=tuple(groups[:max_groups_reported]),
    )


def timing_summary(seconds: Sequence[float]) -> dict[str, float | int | None]:
    """Descriptive wall-clock distribution. No threshold and no flag -- see the catalog entry."""
    values = [float(s) for s in seconds]
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "median": _median(values),
        "p90": percentile(values, 0.90),
        "max": max(values) if values else None,
        "total": round(sum(values), 3) if values else None,
    }
