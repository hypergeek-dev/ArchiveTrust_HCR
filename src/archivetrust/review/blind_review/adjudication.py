"""Adjudication of a `Requires adjudication` item (task brief's "Adjudication" -- a third
reviewer/adjudicator resolves the disagreement; reviewer A's, reviewer B's, and the adjudicated
result are all preserved separately, never overwritten).
"""

from __future__ import annotations

from archivetrust.review.blind_review.agreement import AgreementAssessment, BenchmarkStatus
from archivetrust.review.blind_review.store import BlindReviewStore
from archivetrust.review.htr_models import Adjudication


def adjudicate(
    *,
    store: BlindReviewStore,
    assessment: AgreementAssessment,
    adjudicator_ref: str,
    resolved_value: str | None,
    rationale: str,
    adjudicated_at: str,
    illegible: bool = False,
) -> Adjudication:
    """Records an `Adjudication` resolving `assessment`, which must be classified
    `REQUIRES_ADJUDICATION` -- adjudicating an item that does not require it (e.g. one already
    `AGREED`) is rejected, matching the brief's "when classified Requires adjudication, allow a
    third reviewer/adjudicator to resolve it".

    `rationale` is required and must be non-empty (the brief: "reasoning (a required text field --
    don't allow empty reasoning)"). `Adjudication.rationale` on the underlying model is already a
    required `str` field (not `str | None`), but Pydantic accepts an empty string there; this
    service-level guard is what actually enforces "non-empty", without redefining the model.

    Neither `submission_a`'s nor `submission_b`'s `ReviewSubmission`, nor `assessment`'s
    `AgreementResult`, is mutated or superseded by this call -- `Adjudication` is a new, separate
    record pointing at the `agreement_result_id` it resolves (`htr_models.py`'s own shape), and
    `store.record_adjudication` refuses a second adjudication for the same `agreement_result_id`
    (locking, same discipline as submission locking).
    """
    if assessment.status is not BenchmarkStatus.REQUIRES_ADJUDICATION:
        raise ValueError(
            f"target {assessment.agreement_result.target_ref!r} is classified "
            f"{assessment.status.value!r}, not {BenchmarkStatus.REQUIRES_ADJUDICATION.value!r} "
            "-- only items requiring adjudication may be adjudicated"
        )
    if not rationale or not rationale.strip():
        raise ValueError(
            "Adjudication.rationale is required and must be non-empty -- an adjudicator must "
            "record why they resolved the disagreement the way they did"
        )
    adjudication = Adjudication.create(
        agreement_result_id=assessment.agreement_result.agreement_result_id,
        adjudicator_ref=adjudicator_ref,
        resolved_value=resolved_value,
        rationale=rationale,
        adjudicated_at=adjudicated_at,
        illegible=illegible,
    )
    store.record_adjudication(adjudication)
    return adjudication
