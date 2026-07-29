"""`CanonicalResult` (docs/htr-domain-design.md §1, §2, §4).

Extends the retained `CanonicalObservation`/`CanonicalDocument` substrate with an explicit
`source_method_run_id` per selected span and a `CanonicalizationStrategy` + version, satisfying
§2's "every selected segment must preserve a pointer to its source result." A new, additive file
-- `domain/canonical/observation.py` is not modified.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, model_validator

from archivetrust.domain.shared.ids import new_id


class CanonicalizationStrategy(str, Enum):
    """How a `CanonicalResult`'s spans were selected across competing `MethodRun`s (§1)."""

    BEST_SINGLE_METHOD = "best_single_method"
    METRIC_SELECTED_METHOD = "metric_selected_method"
    SEGMENT_LEVEL_SELECTION = "segment_level_selection"
    HUMAN_APPROVED_SELECTION = "human_approved_selection"
    CONSENSUS_BASED_SELECTION = "consensus_based_selection"


class CanonicalResultSpan(BaseModel):
    """One selected span within a `CanonicalResult`, pointing at the exact `MethodRun` (and,
    transitively, the `Evidence`/`InputCrop`/`TextLine` chain, §4) that produced it -- never an
    embedded copy of the recognized text alone with no traceable source."""

    model_config = ConfigDict(frozen=True)

    text_line_id: str
    source_method_run_id: str
    text: str


class CanonicalResult(BaseModel):
    """ArchiveTrust's best-supported line/region-level HTR result, after `CanonicalizationStrategy`
    selection across competing `MethodRun`s. Each strategy implementation carries its own
    `strategy_version` (§3): "a later change to strategy logic doesn't retroactively reinterpret
    old canonical results." Supersession, never overwrite (Constitution Article 15): a new
    `CanonicalResult` for the same `page_id` is a new record with `supersedes` set, never an edit.
    """

    model_config = ConfigDict(frozen=True)

    canonical_result_id: str
    page_id: str
    strategy: CanonicalizationStrategy
    strategy_version: int
    spans: tuple[CanonicalResultSpan, ...]
    created_at: str
    supersedes: str | None = None

    @model_validator(mode="after")
    def _validate(self) -> "CanonicalResult":
        if not self.spans:
            raise ValueError(
                "CanonicalResult.spans must be non-empty -- there is no canonical result without "
                "at least one traceable source span (Constitution Article 4)"
            )
        if self.strategy_version < 1:
            raise ValueError("CanonicalResult.strategy_version must be >= 1")
        return self

    @classmethod
    def create(
        cls,
        *,
        page_id: str,
        strategy: CanonicalizationStrategy,
        strategy_version: int,
        spans: tuple[CanonicalResultSpan, ...],
        created_at: str,
        supersedes: str | None = None,
    ) -> "CanonicalResult":
        return cls(
            canonical_result_id=new_id("canonical_result"),
            page_id=page_id,
            strategy=strategy,
            strategy_version=strategy_version,
            spans=spans,
            created_at=created_at,
            supersedes=supersedes,
        )
