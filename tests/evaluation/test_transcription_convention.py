from __future__ import annotations

import pytest
from pydantic import ValidationError

from archivetrust.evaluation.ground_truth import GroundTruthGranularity, TranscriptionConvention


def test_transcription_convention_round_trips():
    convention = TranscriptionConvention(
        convention_id="convention_1",
        version=1,
        name="Swedish court-record abbreviation expansion",
        rules="Expand common abbreviations; preserve superscript markers as [^x].",
        created_at="2026-01-01T00:00:00Z",
    )
    restored = TranscriptionConvention.model_validate(convention.model_dump())
    assert restored == convention


def test_transcription_convention_rejects_version_below_one():
    with pytest.raises(ValidationError):
        TranscriptionConvention(
            convention_id="convention_1",
            version=0,
            name="x",
            rules="x",
            created_at="2026-01-01T00:00:00Z",
        )


def test_ground_truth_granularity_has_line_level_value():
    assert GroundTruthGranularity.LINE.value == "line"
