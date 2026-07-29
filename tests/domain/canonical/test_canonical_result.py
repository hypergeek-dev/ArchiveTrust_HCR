from __future__ import annotations

import pytest
from pydantic import ValidationError

from archivetrust.domain.canonical.result import (
    CanonicalizationStrategy,
    CanonicalResult,
    CanonicalResultSpan,
)


def _span() -> CanonicalResultSpan:
    return CanonicalResultSpan(text_line_id="text_line_1", source_method_run_id="method_run_1", text="hej")


def test_canonical_result_requires_at_least_one_span():
    with pytest.raises(ValidationError):
        CanonicalResult(
            canonical_result_id="canonical_result_x",
            page_id="page_1",
            strategy=CanonicalizationStrategy.BEST_SINGLE_METHOD,
            strategy_version=1,
            spans=(),
            created_at="2026-01-01T00:00:00Z",
        )


def test_canonical_result_round_trips_and_preserves_source_method_run_id():
    result = CanonicalResult.create(
        page_id="page_1",
        strategy=CanonicalizationStrategy.CONSENSUS_BASED_SELECTION,
        strategy_version=1,
        spans=(_span(),),
        created_at="2026-01-01T00:00:00Z",
    )
    assert result.spans[0].source_method_run_id == "method_run_1"
    restored = CanonicalResult.model_validate(result.model_dump())
    assert restored == result


def test_canonical_result_supersession_chain():
    first = CanonicalResult.create(
        page_id="page_1",
        strategy=CanonicalizationStrategy.BEST_SINGLE_METHOD,
        strategy_version=1,
        spans=(_span(),),
        created_at="2026-01-01T00:00:00Z",
    )
    second = CanonicalResult.create(
        page_id="page_1",
        strategy=CanonicalizationStrategy.HUMAN_APPROVED_SELECTION,
        strategy_version=1,
        spans=(_span(),),
        created_at="2026-01-02T00:00:00Z",
        supersedes=first.canonical_result_id,
    )
    assert second.supersedes == first.canonical_result_id
