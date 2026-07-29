"""`ResearchReport` (docs/htr-domain-design.md §1, §5).

Fields only at this stage -- the generator that reads telemetry/read-models to populate one is a
later phase (per §5: "owned by a new `src/archivetrust/research/reports/` package, reading only
from telemetry/read-models"). This module defines no reader; `existing research/` package's
read-only convention is honored by construction (no writes anywhere in this file).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from archivetrust.domain.shared.ids import new_id


class ResearchReport(BaseModel):
    """A generated research report over one or more `ExperimentRun`s -- e.g. "Swedish Historical
    HTR Baseline Comparison" (docs/htr-migration-plan.md Stage 12)."""

    model_config = ConfigDict(frozen=True)

    report_id: str
    title: str
    generated_at: str
    experiment_run_ids: tuple[str, ...]
    methodology: str | None = None
    dataset_description: str | None = None
    results_summary: str | None = None
    limitations: str | None = None
    reproducibility_manifest_refs: tuple[str, ...] = ()
    legacy_experiments_included: bool = False
    """Whether any included ExperimentRun's MethodRuns are labeled `legacy: true`
    (docs/htr-domain-design.md §8) -- must be surfaced explicitly, never silently blended into
    baseline aggregate statistics."""
    metadata: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate(self) -> "ResearchReport":
        if not self.experiment_run_ids:
            raise ValueError("ResearchReport.experiment_run_ids must be non-empty")
        return self

    @classmethod
    def create(
        cls,
        *,
        title: str,
        generated_at: str,
        experiment_run_ids: tuple[str, ...],
        methodology: str | None = None,
        dataset_description: str | None = None,
        results_summary: str | None = None,
        limitations: str | None = None,
        reproducibility_manifest_refs: tuple[str, ...] = (),
        legacy_experiments_included: bool = False,
        metadata: dict | None = None,
    ) -> "ResearchReport":
        return cls(
            report_id=new_id("research_report"),
            title=title,
            generated_at=generated_at,
            experiment_run_ids=experiment_run_ids,
            methodology=methodology,
            dataset_description=dataset_description,
            results_summary=results_summary,
            limitations=limitations,
            reproducibility_manifest_refs=reproducibility_manifest_refs,
            legacy_experiments_included=legacy_experiments_included,
            metadata=metadata or {},
        )
