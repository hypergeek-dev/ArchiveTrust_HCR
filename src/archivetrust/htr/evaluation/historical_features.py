"""Historical-document metric evaluators (docs/htr-migration-plan.md Stage 9's "historical-
document metrics"): personal names, place names, dates, occupations, monetary values, legal
expressions, abbreviations, archaic spellings, unusual characters, damaged passages, illegible
passages, marginal notes, crossed-out passages.

**What is implemented for real here** (three evaluators, each with a concrete `extract()`
regex/lookup implementation and real tests in `tests/htr/evaluation/test_historical_features.py`):

- `HistoricalDateEvaluator` -- Swedish 17th/18th-century court-record date formats (Latin-
  influenced month names -- "Januarii", "Februarii", etc. -- and "Anno <year>" constructions),
  matched against `tests/fixtures/transkribus/sample_page.xml`'s real "Anno 1712 den 3 Januarii"
  line and against SATRN's real documented output "till den 23 Januarii" (see
  `providers/satrn/README.md`).
- `AbbreviationEvaluator` -- a small known-abbreviation list for Swedish archival/legal material
  (e.g. "NB", matched against the same real PAGE XML fixture's "NB dombook" marginalia line).
- `UnusualCharacterEvaluator` -- flags archaic/non-standard characters (e.g. the long s, `ſ`) and
  any character outside a defined standard-modern-Swedish-plus-punctuation set.

**The plugin interface** (`HistoricalFeatureEvaluator`) is intentionally left open for the rest of
the brief's list (personal names, place names, occupations, monetary values, legal expressions,
archaic spellings beyond single characters, damaged/illegible passages, marginal notes, crossed-
out passages) -- those require either NER-quality name/place gazetteers or layout-aware damage/
strikethrough detection this phase does not build. Rather than leaving silent, forgotten stub
classes for them, the extension contract is: implement `extract(text) -> tuple[str, ...]`
(optionally override `_normalize_span`) and add one line to `HISTORICAL_FEATURE_REGISTRY`. One
concrete worked example, matching this exact contract:

```python
class PersonalNameEvaluator(HistoricalFeatureEvaluator):
    '''Swedish patronymic personal names ("Andersson", "Andersdotter") -- a real, well-defined
    historical-Swedish name pattern, distinguishable from a full NER model: a capitalized word
    ending in "son" or "dotter", not at the start of a sentence.'''

    feature_name = "personal_names"
    _PATTERN = re.compile(r"(?<!^)(?<!\\. )\\b[A-ZÅÄÖ][a-zåäö]+(?:son|dotter)\\b")

    def extract(self, text: str) -> tuple[str, ...]:
        return tuple(match.group(0) for match in self._PATTERN.finditer(text))


HISTORICAL_FEATURE_REGISTRY["personal_names"] = PersonalNameEvaluator()
```

`evaluate_feature`/`historical_feature_metric_results` (below) work unmodified for any evaluator
registered this way -- they only ever call `.feature_name` and `.extract()`, the two members of
the `HistoricalFeatureEvaluator` contract.
"""

from __future__ import annotations

import re
import unicodedata
from abc import ABC, abstractmethod
from typing import ClassVar

from pydantic import BaseModel, ConfigDict

from archivetrust.htr.evaluation import definitions
from archivetrust.htr.experiment.models import MetricResult


class HistoricalFeatureEvaluator(ABC):
    """The plugin contract every historical-feature evaluator implements. Deliberately minimal --
    one required method -- so a new evaluator (see module docstring's `PersonalNameEvaluator`
    worked example) never has to touch `evaluate_feature`/`historical_feature_metric_results`.
    """

    feature_name: ClassVar[str]

    @abstractmethod
    def extract(self, text: str) -> tuple[str, ...]:
        """Returns every span of `text` this evaluator considers an instance of its feature, in
        the order found. Duplicates are legal (e.g. the same abbreviation appearing twice) --
        `evaluate_feature` decides how to compare, this method only extracts."""
        raise NotImplementedError

    def _normalize_span(self, span: str) -> str:
        """Span-comparison key. Default: case-insensitive, NFC-normalized, stripped -- overridable
        by an evaluator for which case or exact form matters (e.g. an evaluator distinguishing
        "NB" from "nb" would override this to be a no-op)."""
        return unicodedata.normalize("NFC", span).strip().casefold()


class HistoricalFeatureEvaluationResult(BaseModel):
    """Set-based precision/recall for one evaluator over one (reference, hypothesis) pair --
    matching is by normalized span text, not position (historical-text OCR rarely preserves exact
    character offsets even when it gets the content right)."""

    model_config = ConfigDict(frozen=True)

    feature_name: str
    reference_spans: tuple[str, ...]
    hypothesis_spans: tuple[str, ...]
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float | None
    recall: float | None
    f1: float | None


def evaluate_feature(
    evaluator: HistoricalFeatureEvaluator, *, reference: str, hypothesis: str
) -> HistoricalFeatureEvaluationResult:
    reference_spans = evaluator.extract(reference)
    hypothesis_spans = evaluator.extract(hypothesis)
    ref_keys = {evaluator._normalize_span(span) for span in reference_spans}
    hyp_keys = {evaluator._normalize_span(span) for span in hypothesis_spans}

    true_positives = len(ref_keys & hyp_keys)
    false_positives = len(hyp_keys - ref_keys)
    false_negatives = len(ref_keys - hyp_keys)

    precision = true_positives / len(hyp_keys) if hyp_keys else (None if not ref_keys else 0.0)
    recall = true_positives / len(ref_keys) if ref_keys else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall) > 0
        else None
    )

    return HistoricalFeatureEvaluationResult(
        feature_name=evaluator.feature_name,
        reference_spans=reference_spans,
        hypothesis_spans=hypothesis_spans,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def historical_feature_metric_results(
    evaluator: HistoricalFeatureEvaluator, *, reference: str, hypothesis: str, method_run_id: str
) -> tuple[MetricResult, ...]:
    """`MetricResult`s for one evaluator -- `None`-valued precision/recall/f1 (no reference or no
    hypothesis spans at all) produce no `MetricResult` for that component rather than a fabricated
    0.0/1.0 (mirrors the codebase-wide "never fabricate a value for genuine absence" discipline,
    e.g. `ParsedLine.confidence`)."""
    result = evaluate_feature(evaluator, reference=reference, hypothesis=hypothesis)
    results: list[MetricResult] = []
    if result.precision is not None:
        results.append(
            MetricResult.create(
                metric_definition_id=definitions.HISTORICAL_FEATURE_PRECISION.metric_definition_id,
                method_run_id=method_run_id,
                value=result.precision,
            )
        )
    if result.recall is not None:
        results.append(
            MetricResult.create(
                metric_definition_id=definitions.HISTORICAL_FEATURE_RECALL.metric_definition_id,
                method_run_id=method_run_id,
                value=result.recall,
            )
        )
    if result.f1 is not None:
        results.append(
            MetricResult.create(
                metric_definition_id=definitions.HISTORICAL_FEATURE_F1.metric_definition_id,
                method_run_id=method_run_id,
                value=result.f1,
            )
        )
    return tuple(results)


# --- Concrete evaluator 1: historical dates --------------------------------------------------


class HistoricalDateEvaluator(HistoricalFeatureEvaluator):
    """Swedish 17th/18th-century court-record date formats: Latin-influenced month names
    ("Januarii", "Februarii", ... -- the genitive Latin forms Swedish court scribes of this period
    actually used, distinct from modern Swedish "januari") and "Anno <year>" constructions. Real
    positive: `tests/fixtures/transkribus/sample_page.xml`'s "Anno 1712 den 3 Januarii holltes
    ting" line. Real negative-but-plausible: SATRN's real documented output "till den 23 Januarii"
    (`providers/satrn/README.md`) -- a date phrase the model hallucinated/misrecognized relative
    to a ground truth that contains no date at all, so this evaluator correctly reports it as a
    false positive against that ground truth, not a false "match"."""

    feature_name = "historical_dates"

    _MONTHS = (
        "Januarii",
        "Februarii",
        "Martii",
        "Aprilis",
        "Maji",
        "Junii",
        "Julii",
        "Augusti",
        "Septembris",
        "Octobris",
        "Novembris",
        "Decembris",
    )
    _DATE_PATTERN = re.compile(
        r"\b(?:den\s+\d{1,2}\s+(?:" + "|".join(_MONTHS) + r")|"
        r"(?:" + "|".join(_MONTHS) + r")|"
        r"Anno\s+1[5-9]\d{2})\b",
        re.IGNORECASE,
    )

    def extract(self, text: str) -> tuple[str, ...]:
        return tuple(match.group(0) for match in self._DATE_PATTERN.finditer(text))


# --- Concrete evaluator 2: known historical abbreviations --------------------------------------


class AbbreviationEvaluator(HistoricalFeatureEvaluator):
    """A small known-abbreviation list for Swedish archival/legal material. Real positive:
    `tests/fixtures/transkribus/sample_page.xml`'s marginalia line "NB dombook" (`NB`, Latin
    "nota bene"). Deliberately a fixed lookup list, not a generative pattern -- per the brief's
    "at least 2-3 concrete evaluators... not just interface", an abbreviation detector's honest
    minimal form *is* a known-list membership check; growing the list is the expected maintenance
    path, not a design gap."""

    feature_name = "abbreviations"

    KNOWN_ABBREVIATIONS: ClassVar[tuple[str, ...]] = (
        "NB",
        "S:t",
        "K.M:t",
        "Kongl. Maj:t",
        "M:r",
        "d:o",
        "etc.",
        "p:r",
        "N:o",
    )
    _PATTERN = re.compile(
        r"\b(?:" + "|".join(re.escape(a) for a in KNOWN_ABBREVIATIONS) + r")",
    )

    def extract(self, text: str) -> tuple[str, ...]:
        return tuple(match.group(0) for match in self._PATTERN.finditer(text))

    def _normalize_span(self, span: str) -> str:
        # Abbreviation punctuation (":", ".") is part of the identity -- only casefold, don't
        # strip/NFC-fold away the marks that distinguish e.g. "M:r" from "Mr".
        return span.casefold()


# --- Concrete evaluator 3: unusual / archaic characters ----------------------------------------


ARCHAIC_CHARACTERS: dict[str, str] = {
    "ſ": "long s (historical Latin/Swedish script variant of 's')",
    "æ": "ae ligature",
    "Æ": "AE ligature",
    "œ": "oe ligature",
    "Œ": "OE ligature",
    "þ": "thorn",
    "ð": "eth",
    "⁊": "Tironian et (historical 'and' abbreviation mark)",
}
"""Characters this evaluator always flags by name, even though most also fall outside
`_STANDARD_SWEDISH_CHARACTERS` anyway -- kept as an explicit table so `extract()` can report *why*
a character was flagged (a real archaic-spelling marker) rather than only *that* it was."""

_STANDARD_SWEDISH_CHARACTERS = set(
    "abcdefghijklmnopqrstuvwxyzåäöABCDEFGHIJKLMNOPQRSTUVWXYZÅÄÖ0123456789 .,;:!?'\"-()/\n\t"
)


class UnusualCharacterEvaluator(HistoricalFeatureEvaluator):
    """Flags archaic/non-standard Unicode characters: every character in `ARCHAIC_CHARACTERS` by
    name, plus (generically) any non-whitespace character outside the standard modern-Swedish-
    plus-basic-punctuation set. Real archaic-spelling test case: the long s, `ſ` (U+017F) -- a
    genuine 17th-century Swedish/German-influenced handwriting convention, not a synthetic
    placeholder character."""

    feature_name = "unusual_characters"

    def extract(self, text: str) -> tuple[str, ...]:
        found = []
        for char in text:
            if char in ARCHAIC_CHARACTERS or (
                not char.isspace() and char not in _STANDARD_SWEDISH_CHARACTERS
            ):
                found.append(char)
        return tuple(found)

    def _normalize_span(self, span: str) -> str:
        # Case matters for single characters here (e.g. Æ vs æ are two distinct archaic glyphs);
        # only NFC-normalize, never casefold.
        return unicodedata.normalize("NFC", span)


HISTORICAL_FEATURE_REGISTRY: dict[str, HistoricalFeatureEvaluator] = {
    "historical_dates": HistoricalDateEvaluator(),
    "abbreviations": AbbreviationEvaluator(),
    "unusual_characters": UnusualCharacterEvaluator(),
}
"""Every evaluator this phase implements, keyed by `feature_name`. A future evaluator (see module
docstring's `PersonalNameEvaluator` worked example) is added with one more `dict` entry -- no
other code in this module changes."""
