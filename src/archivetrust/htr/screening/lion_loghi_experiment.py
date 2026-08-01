"""`build_lion_loghi_comparison` -- constructs the four structurally-separate Lion-vs-Loghi
experiment cells (docs/experiments/lion-loghi-comparison/README.md).

Four conditions, four independent `Experiment`/`ExperimentVersion` pairs, never one `ExperimentRun`
covering all four (brief: "Do not put all four conditions into one ExperimentRun"):

```
swedish_lion x Swedish corpus   -- in_domain
swedish_lion x Dutch corpus     -- cross_domain
loghi x Swedish corpus          -- cross_domain
loghi x Dutch corpus            -- in_domain
```

`loghi`'s in-domain/cross-domain assignment assumes a Dutch-trained model, per the brief's own worked
example -- recorded explicitly here as a constant, never inferred from dataset file location (the
brief's own "do not infer domain solely from file location").

**Nothing is persisted by this module.** Like `presentation/htr_experiment_builder_viewmodel.py`'s
`build()`, this returns the constructed entities; registering them (via `DurableHtrResearchStore.
register_experiment` / `register_experiment_version` / `register_comparison_group` /
`record_domain_relationship`) is the caller's decision.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from archivetrust.htr.experiment.models import (
    DomainRelationship,
    Experiment,
    ExperimentComparisonGroup,
    ExperimentVersion,
)

SWEDISH_LION_METHOD_ID = "swedish_lion"
LOGHI_METHOD_ID = "loghi"

SWEDISH_LANGUAGE = "sv"
DUTCH_LANGUAGE = "nl"

SWEDISH_LION_PRIMARY_DOMAIN = "sv"
"""`swedish_lion` (Riksarkivet/trocr-base-handwritten-hist-swe-2) is fine-tuned on Swedish historical
handwriting -- see `providers/swedish_lion/adapter.py`'s module docstring."""
LOGHI_PRIMARY_DOMAIN = "nl"
"""Assumes Loghi's Dutch-trained model/checkpoint, per the brief's own worked example ("Loghi Dutch
model on Dutch corpus = in_domain"). Recorded here as an explicit constant rather than left implicit --
a future Loghi checkpoint trained on a different primary language would need this constant updated
alongside `pinned_versions.CURRENT_PINNED_VERSIONS.model_checkpoint_id`, never silently reinterpreted."""


class ExperimentCell(BaseModel):
    """One of the four structurally-separate conditions."""

    model_config = ConfigDict(frozen=True)

    label: str
    """Human-readable, e.g. `"Swedish Lion I on Swedish pages"` -- never used as an identifier."""
    experiment: Experiment
    experiment_version: ExperimentVersion
    method_id: str
    corpus_language: str
    domain_relationship: DomainRelationship


class LionLoghiComparison(BaseModel):
    """The full four-cell construction: the four `ExperimentCell`s plus the
    `ExperimentComparisonGroup` tying them together by id -- never collapsing them into one
    `ExperimentRun` or one result set."""

    model_config = ConfigDict(frozen=True)

    comparison_group: ExperimentComparisonGroup
    cells: tuple[ExperimentCell, ExperimentCell, ExperimentCell, ExperimentCell]

    def cell_for(self, *, method_id: str, corpus_language: str) -> ExperimentCell:
        """The one cell matching `(method_id, corpus_language)` -- raises if zero or more than one
        match, since the four-cell design guarantees exactly one per pairing and a caller asking for
        a pairing that does not exist has a bug worth surfacing immediately."""
        matches = [
            cell
            for cell in self.cells
            if cell.method_id == method_id and cell.corpus_language == corpus_language
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one cell for method_id={method_id!r} "
                f"corpus_language={corpus_language!r}, found {len(matches)}"
            )
        return matches[0]


def build_lion_loghi_comparison(
    *,
    research_project_id: str,
    swedish_dataset_version_id: str,
    dutch_dataset_version_id: str,
    created_at: str,
) -> LionLoghiComparison:
    """Constructs (never persists) the four `Experiment`/`ExperimentVersion` pairs and their parent
    `ExperimentComparisonGroup`. Each `ExperimentVersion.method_ids` is a one-element tuple (each cell
    tests exactly one method against one corpus) -- a controlled comparison, not an end-to-end
    multi-method run.
    """
    cell_specs = (
        (
            "Swedish Lion I on Swedish pages",
            SWEDISH_LION_METHOD_ID,
            swedish_dataset_version_id,
            SWEDISH_LANGUAGE,
            DomainRelationship.IN_DOMAIN,
        ),
        (
            "Swedish Lion I on Dutch pages",
            SWEDISH_LION_METHOD_ID,
            dutch_dataset_version_id,
            DUTCH_LANGUAGE,
            DomainRelationship.CROSS_DOMAIN,
        ),
        (
            "Loghi on Swedish pages",
            LOGHI_METHOD_ID,
            swedish_dataset_version_id,
            SWEDISH_LANGUAGE,
            DomainRelationship.CROSS_DOMAIN,
        ),
        (
            "Loghi on Dutch pages",
            LOGHI_METHOD_ID,
            dutch_dataset_version_id,
            DUTCH_LANGUAGE,
            DomainRelationship.IN_DOMAIN,
        ),
    )

    cells: list[ExperimentCell] = []
    for label, method_id, dataset_version_id, corpus_language, domain_relationship in cell_specs:
        primary_domain = (
            SWEDISH_LION_PRIMARY_DOMAIN if method_id == SWEDISH_LION_METHOD_ID else LOGHI_PRIMARY_DOMAIN
        )
        experiment = Experiment.create(
            name=label,
            research_project_id=research_project_id,
            description=(
                f"One condition of the Swedish Lion I vs. Loghi domain-transfer comparison: {label}."
            ),
            created_at=created_at,
        )
        experiment_version = ExperimentVersion.create(
            experiment_id=experiment.experiment_id,
            version=1,
            dataset_version_id=dataset_version_id,
            method_ids=(method_id,),
            created_at=created_at,
            corpus_language=corpus_language,
            method_primary_language_domain=primary_domain,
            domain_relationship=domain_relationship,
        )
        cells.append(
            ExperimentCell(
                label=label,
                experiment=experiment,
                experiment_version=experiment_version,
                method_id=method_id,
                corpus_language=corpus_language,
                domain_relationship=domain_relationship,
            )
        )

    comparison_group = ExperimentComparisonGroup.create(
        name="Swedish Lion I vs. Loghi: in-domain and reciprocal cross-domain comparison",
        experiment_ids=tuple(cell.experiment.experiment_id for cell in cells),
        created_at=created_at,
    )

    return LionLoghiComparison(
        comparison_group=comparison_group,
        cells=(cells[0], cells[1], cells[2], cells[3]),
    )
