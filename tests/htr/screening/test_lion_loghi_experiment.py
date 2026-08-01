"""`htr/screening/lion_loghi_experiment.py::build_lion_loghi_comparison` -- the four-cell design."""

from __future__ import annotations

import pytest

from archivetrust.htr.experiment.models import DomainRelationship
from archivetrust.htr.screening.lion_loghi_experiment import build_lion_loghi_comparison


@pytest.fixture()
def comparison():
    return build_lion_loghi_comparison(
        research_project_id="research_project_1",
        swedish_dataset_version_id="dataset_version_sv",
        dutch_dataset_version_id="dataset_version_nl",
        created_at="2026-08-01T00:00:00Z",
    )


def test_produces_exactly_four_cells(comparison) -> None:
    assert len(comparison.cells) == 4


def test_each_cell_is_a_structurally_separate_experiment(comparison) -> None:
    """Never one `ExperimentRun` covering all four -- each cell has its own `Experiment` and
    `ExperimentVersion` id."""
    experiment_ids = {cell.experiment.experiment_id for cell in comparison.cells}
    experiment_version_ids = {cell.experiment_version.experiment_version_id for cell in comparison.cells}
    assert len(experiment_ids) == 4
    assert len(experiment_version_ids) == 4


def test_each_experiment_version_has_exactly_one_method(comparison) -> None:
    for cell in comparison.cells:
        assert cell.experiment_version.method_ids == (cell.method_id,)


def test_domain_relationships_match_the_briefs_worked_example(comparison) -> None:
    lion_swedish = comparison.cell_for(method_id="swedish_lion", corpus_language="sv")
    lion_dutch = comparison.cell_for(method_id="swedish_lion", corpus_language="nl")
    loghi_swedish = comparison.cell_for(method_id="loghi", corpus_language="sv")
    loghi_dutch = comparison.cell_for(method_id="loghi", corpus_language="nl")

    assert lion_swedish.domain_relationship is DomainRelationship.IN_DOMAIN
    assert lion_dutch.domain_relationship is DomainRelationship.CROSS_DOMAIN
    assert loghi_swedish.domain_relationship is DomainRelationship.CROSS_DOMAIN
    assert loghi_dutch.domain_relationship is DomainRelationship.IN_DOMAIN


def test_correct_dataset_version_per_corpus(comparison) -> None:
    for cell in comparison.cells:
        expected = "dataset_version_sv" if cell.corpus_language == "sv" else "dataset_version_nl"
        assert cell.experiment_version.dataset_version_id == expected


def test_comparison_group_names_all_four_experiments_without_merging_results(comparison) -> None:
    group = comparison.comparison_group
    experiment_ids = {cell.experiment.experiment_id for cell in comparison.cells}
    assert set(group.experiment_ids) == experiment_ids
    assert len(group.experiment_ids) == 4  # not collapsed into one


def test_cell_for_raises_for_a_pairing_that_does_not_exist(comparison) -> None:
    with pytest.raises(ValueError):
        comparison.cell_for(method_id="satrn", corpus_language="sv")


def test_nothing_is_persisted_by_the_builder(comparison) -> None:
    """Pure construction -- no store, no telemetry sink involved at all."""
    assert comparison.comparison_group.comparison_id.startswith("experiment_comparison_group_")
