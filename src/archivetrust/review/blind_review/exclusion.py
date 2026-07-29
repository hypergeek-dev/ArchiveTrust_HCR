"""Excluding a target from the benchmark (task brief's "Exclusion" -- allowed ONLY with a recorded,
required reason).

`ExclusionRecord` has no analogue in `review/htr_models.py` or `evaluation/ground_truth.py` (the
latter's `GroundTruthAnnotation.exclusion_reason` is a per-annotation optional field on a different,
more general model; this is a purpose-built record for the blind-dual-review benchmark-inclusion
decision, with its own required-reason validation). It is new because the workflow it belongs to is
new, not a redefinition of an existing model.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_validator

from archivetrust.domain.shared.ids import new_id
from archivetrust.review.blind_review.store import BlindReviewStore


class ExclusionRecord(BaseModel):
    """One target excluded from benchmark use. `reason` is required and validated non-empty --
    construction itself fails for a missing or blank reason, so there is no way to hold an
    `ExclusionRecord` instance that does not carry a real reason."""

    model_config = ConfigDict(frozen=True)

    exclusion_id: str
    target_ref: str
    reason: str
    excluded_by: str
    excluded_at: str

    @model_validator(mode="after")
    def _validate(self) -> "ExclusionRecord":
        if not self.reason or not self.reason.strip():
            raise ValueError("ExclusionRecord.reason is required and must be non-empty")
        return self

    @classmethod
    def create(
        cls, *, target_ref: str, reason: str, excluded_by: str, excluded_at: str
    ) -> "ExclusionRecord":
        return cls(
            exclusion_id=new_id("benchmark_exclusion"),
            target_ref=target_ref,
            reason=reason,
            excluded_by=excluded_by,
            excluded_at=excluded_at,
        )


def exclude_from_benchmark(
    *, store: BlindReviewStore, target_ref: str, reason: str, excluded_by: str, excluded_at: str
) -> ExclusionRecord:
    """Excludes `target_ref` from the benchmark. Rejects an empty/missing `reason` (via
    `ExclusionRecord`'s own validator, raised before anything is recorded) and rejects a second
    exclusion for a target that already has one (via `store.record_exclusion`'s
    `DuplicateExclusionError` -- exclusion, like submission, is not silently overwritten).
    """
    record = ExclusionRecord.create(
        target_ref=target_ref, reason=reason, excluded_by=excluded_by, excluded_at=excluded_at
    )
    store.record_exclusion(record)
    return record
