"""The final, queryable resolution for one target -- exclusion always wins over the computed
classification; otherwise the computed `AgreementAssessment` plus any recorded `Adjudication`
(task brief item 5: "update `AgreementResult`'s final benchmark status accordingly").

`AgreementResult` (`htr_models.py`) carries no status field to update in place, and this package
does not redefine it (see `blind_review/__init__.py`'s module docstring). `BenchmarkOutcome` is the
derived view instead: it never mutates `AgreementResult`, `Adjudication`, or either
`ReviewSubmission` -- Constitution Article 15 (supersession, never erasure) applied to this
workflow means reviewer A's, reviewer B's, and the adjudicator's results all stay independently
readable through `assessment`/`adjudication` even after resolution.

Adjudicating a `REQUIRES_ADJUDICATION` item does not retroactively reclassify it as `AGREED` --
`status` stays the original computed classification (the fact that the two blind readings diverged
that much is a fact about what happened, not erased by later resolving it). What changes is
`ready_for_benchmark`/`benchmark_value`: once adjudicated, the target has a usable value again.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from archivetrust.review.blind_review.agreement import (
    AgreementAssessment,
    AgreementPolicy,
    BenchmarkStatus,
)
from archivetrust.review.blind_review.exclusion import ExclusionRecord
from archivetrust.review.blind_review.store import BlindReviewStore
from archivetrust.review.htr_models import Adjudication


class BenchmarkOutcome(BaseModel):
    """The current, fully-resolved-as-far-as-possible state for one target."""

    model_config = ConfigDict(frozen=True)

    target_ref: str
    status: BenchmarkStatus
    ready_for_benchmark: bool
    """Whether this target currently has a usable value for benchmark inclusion: `True` for
    `AGREED`/`MINOR_DISAGREEMENT` (reviewer A's reading is usable without further action), `True`
    for anything else only once an `Adjudication` has been recorded, always `False` for
    `EXCLUDED_FROM_BENCHMARK`. `MATERIAL_DISAGREEMENT`/`REQUIRES_ADJUDICATION` pending resolution
    are `False` here but remain fully visible/queryable via `status` and `assessment` -- never
    silently dropped (task brief item 7)."""
    benchmark_value: str | None
    """The resolved transcription text this target would contribute to a benchmark dataset:
    reviewer A's `submitted_value` for `AGREED`/`MINOR_DISAGREEMENT`, the adjudicator's
    `resolved_value` once adjudicated, else `None`."""
    assessment: AgreementAssessment | None
    """`None` only when the target was excluded before both blind submissions were finalized."""
    adjudication: Adjudication | None = None
    exclusion: ExclusionRecord | None = None


def resolve_benchmark_outcome(
    store: BlindReviewStore,
    target_ref: str,
    *,
    computed_at: str,
    policy: AgreementPolicy | None = None,
) -> BenchmarkOutcome:
    """Resolves the current `BenchmarkOutcome` for `target_ref`. Raises `ValueError` if the target
    has no recorded exclusion and does not yet have both blind submissions finalized -- there is
    nothing to resolve yet (callers with a batch of targets in mixed states should check
    `store.both_submissions_if_complete`/`store.exclusion_for` first, which is exactly what
    `queries.batch_completion_status` does).
    """
    exclusion = store.exclusion_for(target_ref)
    pair = store.both_submissions_if_complete(target_ref)

    if pair is None:
        if exclusion is not None:
            # A target may be excluded (e.g. a corrupt source image discovered mid-review) before
            # either blind reviewer finishes -- exclusion never requires completed review.
            return BenchmarkOutcome(
                target_ref=target_ref,
                status=BenchmarkStatus.EXCLUDED_FROM_BENCHMARK,
                ready_for_benchmark=False,
                benchmark_value=None,
                assessment=None,
                adjudication=None,
                exclusion=exclusion,
            )
        raise ValueError(
            f"target {target_ref!r} does not yet have both blind submissions finalized and has no "
            "recorded exclusion -- nothing to resolve yet"
        )

    submission_a, _submission_b = pair
    assessment = store.get_or_compute_agreement(
        target_ref, computed_at=computed_at, policy=policy
    )
    adjudication = store.adjudication_for(assessment.agreement_result.agreement_result_id)

    if exclusion is not None:
        return BenchmarkOutcome(
            target_ref=target_ref,
            status=BenchmarkStatus.EXCLUDED_FROM_BENCHMARK,
            ready_for_benchmark=False,
            benchmark_value=None,
            assessment=assessment,
            adjudication=adjudication,
            exclusion=exclusion,
        )

    if assessment.status in (BenchmarkStatus.AGREED, BenchmarkStatus.MINOR_DISAGREEMENT):
        return BenchmarkOutcome(
            target_ref=target_ref,
            status=assessment.status,
            ready_for_benchmark=True,
            benchmark_value=submission_a.submitted_value,
            assessment=assessment,
            adjudication=adjudication,
        )

    if adjudication is not None:
        return BenchmarkOutcome(
            target_ref=target_ref,
            status=assessment.status,
            ready_for_benchmark=True,
            benchmark_value=adjudication.resolved_value,
            assessment=assessment,
            adjudication=adjudication,
        )

    return BenchmarkOutcome(
        target_ref=target_ref,
        status=assessment.status,
        ready_for_benchmark=False,
        benchmark_value=None,
        assessment=assessment,
        adjudication=None,
    )
