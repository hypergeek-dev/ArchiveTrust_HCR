"""Agreement calculation between two finalized blind `ReviewSubmission`s (task brief's "Agreement
calculation" -- classify into the brief's exact five benchmark categories, using real CER/WER, and
preserve disagreement LOCATIONS, not just a scalar).

**Primitive reuse, not reimplementation.** The actual character/word error rates and edit-operation
counts come from `htr/evaluation/recognition.py::compute_recognition_metrics` (which itself reuses
`evaluation/metrics.py::compare_text`/`normalize_text`/`levenshtein`) -- this module never
recomputes a Levenshtein distance itself. What `recognition.py`'s public API deliberately does NOT
expose is *where* two texts diverge: `EditOpCounts` (`classify_char_edits`/`classify_word_edits`)
is an aggregate (`matches`/`substitutions`/`insertions`/`deletions` counts), not a sequence of
positioned spans, and its backtrace (`_classify_edits`) is private to that module. For LOCATIONS,
this module uses Python's stdlib `difflib.SequenceMatcher` over the same `normalize_text`-normalized
word tokens `compute_recognition_metrics` already normalizes -- a different, well-tested stdlib
algorithm applied purely to answer "where do the two readings diverge", never to recompute "how much
do they diverge" (that number always comes from `recognition.py`, and only from there).
"""

from __future__ import annotations

import difflib
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from archivetrust.evaluation.metrics import normalize_text
from archivetrust.htr.evaluation.recognition import RecognitionMetrics, compute_recognition_metrics
from archivetrust.review.htr_models import AgreementResult, ReviewSubmission

AGREEMENT_POLICY_VERSION = 1


class BenchmarkStatus(str, Enum):
    """The task brief's exact five benchmark categories for one `GroundTruthItem`'s blind dual
    review. `EXCLUDED_FROM_BENCHMARK` is never produced by `compute_agreement` itself -- it is only
    ever set by an explicit, reasoned `exclude_from_benchmark` call (`exclusion.py`), and always
    overrides whatever this classification would otherwise say (see `outcome.py`)."""

    AGREED = "agreed"
    MINOR_DISAGREEMENT = "minor_disagreement"
    MATERIAL_DISAGREEMENT = "material_disagreement"
    REQUIRES_ADJUDICATION = "requires_adjudication"
    EXCLUDED_FROM_BENCHMARK = "excluded_from_benchmark"


class AgreementPolicy(BaseModel):
    """Versioned, documented CER-based thresholds for `BenchmarkStatus` (mirrors
    `review/triage.py::TriagePolicy`'s pattern: explicit, visible, versioned constants rather than
    magic numbers buried in `compute_agreement`). Computed on
    `RecognitionMetrics.character_error_rate_normalized` -- normalized, so whitespace/Unicode-form
    noise between two honest transcriptions never inflates the disagreement class.

    Threshold rationale (deliberate placeholders pending real inter-annotator-agreement corpus
    calibration, recorded here so a later, better-calibrated version supersedes rather than
    silently overwrites them -- the same discipline `TriagePolicy`'s own docstring states):

    - `minor_disagreement_cer_max = 0.02` (2%): on a typical Swedish 17th/18th-century
      court-record line (`tests/fixtures/htr/trolldomskommissionen_sample_line.txt` is 60
      characters), 2% is roughly one character -- a single dropped/added letter (e.g. a doubled
      consonant transcribed once, "waritt" vs "warit") or one punctuation mark. Differences at
      this scale are the kind two careful, honest readers of the same handwriting routinely
      produce and do not indicate either reader is wrong.
    - `material_disagreement_cer_max = 0.15` (15%): above 2% and up to 15% covers multi-word or
      multi-character divergence (a misread word, a dropped diacritic across a word, a missing
      short word) -- a real, worth-surfacing disagreement, but not so severe that the two readings
      are irreconcilable without a third opinion.
    - Above 15%: the two readings differ enough (different words entirely, a mis-segmented line,
      one reviewer reading a substantially different passage) that only an independent
      adjudicator's judgment resolves which -- if either -- is correct. Classified
      `REQUIRES_ADJUDICATION`.

    An illegibility mismatch (one reviewer marks the target illegible, the other transcribes it)
    is always `REQUIRES_ADJUDICATION` regardless of these numeric thresholds -- there is no CER to
    compute between text and "no text", and the disagreement itself (legible vs. not) is exactly
    the kind of judgment call adjudication exists for.
    """

    model_config = ConfigDict(frozen=True)

    version: int = AGREEMENT_POLICY_VERSION
    minor_disagreement_cer_max: float = 0.02
    material_disagreement_cer_max: float = 0.15

    @model_validator(mode="after")
    def _validate(self) -> "AgreementPolicy":
        if not (0.0 <= self.minor_disagreement_cer_max < self.material_disagreement_cer_max):
            raise ValueError(
                "AgreementPolicy requires 0.0 <= minor_disagreement_cer_max < "
                "material_disagreement_cer_max"
            )
        return self


DEFAULT_AGREEMENT_POLICY = AgreementPolicy()


class DisagreementSpan(BaseModel):
    """One located span where the two blind submissions diverge, at word-token granularity
    (`difflib.SequenceMatcher` opcodes on `normalize_text`-normalized, whitespace-split tokens).
    `op="equal"` spans are never included -- only actual disagreement locations, per the task
    brief's "disagreement locations preserved" requirement (a scalar CER alone cannot answer
    "which words differed").
    """

    model_config = ConfigDict(frozen=True)

    op: Literal["replace", "insert", "delete"]
    """difflib opcode tag, `"equal"` excluded. "replace": both sides have a differing span at this
    location. "insert": reviewer B has words reviewer A does not. "delete": reviewer A has words
    reviewer B does not."""
    reviewer_a_word_index: int
    """Start index into reviewer A's normalized word tokens."""
    reviewer_b_word_index: int
    """Start index into reviewer B's normalized word tokens."""
    reviewer_a_text: str
    """The differing span from reviewer A's tokens, joined by a single space; `""` for `insert`."""
    reviewer_b_text: str
    """The differing span from reviewer B's tokens, joined by a single space; `""` for `delete`."""


class AgreementAssessment(BaseModel):
    """Everything computed from two finalized blind `ReviewSubmission`s for one target: the frozen
    `AgreementResult` (`htr_models.py`, unmodified), a benchmark classification `htr_models.py`
    does not carry, the underlying `RecognitionMetrics` (`None` only for the both-illegible /
    illegibility-mismatch cases, where there is no text to compare), and the disagreement locations.
    Wraps `AgreementResult` rather than extending it, so `htr_models.py` stays exactly as the
    earlier phase defined it.
    """

    model_config = ConfigDict(frozen=True)

    agreement_result: AgreementResult
    status: BenchmarkStatus
    recognition_metrics: RecognitionMetrics | None
    disagreement_locations: tuple[DisagreementSpan, ...]
    policy_version: int


def _word_disagreement_locations(reference: str, hypothesis: str) -> tuple[DisagreementSpan, ...]:
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()
    matcher = difflib.SequenceMatcher(a=ref_words, b=hyp_words, autojunk=False)
    spans: list[DisagreementSpan] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        spans.append(
            DisagreementSpan(
                op=tag,  # type: ignore[arg-type]  # difflib's tag literals match our Literal set
                reviewer_a_word_index=i1,
                reviewer_b_word_index=j1,
                reviewer_a_text=" ".join(ref_words[i1:i2]),
                reviewer_b_text=" ".join(hyp_words[j1:j2]),
            )
        )
    return tuple(spans)


def compute_agreement(
    *,
    target_ref: str,
    submission_a: ReviewSubmission,
    submission_b: ReviewSubmission,
    computed_at: str,
    policy: AgreementPolicy | None = None,
) -> AgreementAssessment:
    """Computes the `AgreementResult` plus benchmark classification and disagreement locations for
    one target's two finalized blind submissions. Pure and idempotent: calling this twice on the
    same two submissions always yields the same classification (no hidden state, no telemetry) --
    the same discipline `review/triage.py`'s classification functions already follow.
    """
    policy = policy or DEFAULT_AGREEMENT_POLICY
    if submission_a.assignment_id == submission_b.assignment_id:
        raise ValueError("Agreement requires two distinct assignments' submissions")

    if submission_a.illegible or submission_b.illegible:
        both_illegible = submission_a.illegible and submission_b.illegible
        status = BenchmarkStatus.AGREED if both_illegible else BenchmarkStatus.REQUIRES_ADJUDICATION
        agrees = both_illegible
        agreement_result = AgreementResult.create(
            target_ref=target_ref,
            submission_a_id=submission_a.submission_id,
            submission_b_id=submission_b.submission_id,
            agrees=agrees,
            computed_at=computed_at,
            similarity_score=1.0 if both_illegible else None,
        )
        return AgreementAssessment(
            agreement_result=agreement_result,
            status=status,
            recognition_metrics=None,
            disagreement_locations=(),
            policy_version=policy.version,
        )

    reference = submission_a.submitted_value or ""
    hypothesis = submission_b.submitted_value or ""
    metrics = compute_recognition_metrics(reference, hypothesis)
    cer = metrics.character_error_rate_normalized

    if cer == 0.0:
        status = BenchmarkStatus.AGREED
    elif cer <= policy.minor_disagreement_cer_max:
        status = BenchmarkStatus.MINOR_DISAGREEMENT
    elif cer <= policy.material_disagreement_cer_max:
        status = BenchmarkStatus.MATERIAL_DISAGREEMENT
    else:
        status = BenchmarkStatus.REQUIRES_ADJUDICATION

    agrees = status in (BenchmarkStatus.AGREED, BenchmarkStatus.MINOR_DISAGREEMENT)
    similarity_score = max(0.0, min(1.0, 1.0 - cer))
    disagreement_locations = _word_disagreement_locations(reference, hypothesis)

    agreement_result = AgreementResult.create(
        target_ref=target_ref,
        submission_a_id=submission_a.submission_id,
        submission_b_id=submission_b.submission_id,
        agrees=agrees,
        computed_at=computed_at,
        similarity_score=similarity_score,
    )
    return AgreementAssessment(
        agreement_result=agreement_result,
        status=status,
        recognition_metrics=metrics,
        disagreement_locations=disagreement_locations,
        policy_version=policy.version,
    )
