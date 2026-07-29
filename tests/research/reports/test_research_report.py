from __future__ import annotations

import pytest
from pydantic import ValidationError

from archivetrust.research.reports import ResearchReport


def test_research_report_requires_at_least_one_experiment_run():
    with pytest.raises(ValidationError):
        ResearchReport(
            report_id="research_report_x",
            title="x",
            generated_at="2026-01-01T00:00:00Z",
            experiment_run_ids=(),
        )


def test_research_report_round_trips():
    report = ResearchReport.create(
        title="Swedish Historical HTR Baseline Comparison",
        generated_at="2026-01-01T00:00:00Z",
        experiment_run_ids=("experiment_run_1",),
        legacy_experiments_included=False,
    )
    restored = ResearchReport.model_validate(report.model_dump())
    assert restored == report
