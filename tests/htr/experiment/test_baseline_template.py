"""Tests for the "Swedish Historical HTR Baseline Comparison" template
(`htr/experiment/baseline_template.py`, docs/htr-migration-plan.md Stage 12).

No adapter is imported for inference here -- `baseline_template.py` only reads each adapter
module's `METHOD_ID` constant and the Transkribus adapter's honest "unpinned" model-revision
string, none of which require a GPU or a network call. This test file therefore needs no
`real_model` marker and runs in the default fast suite.
"""

from __future__ import annotations

import json

import pytest

from archivetrust.htr.experiment.baseline_template import (
    BASELINE_TEMPLATE_TITLE,
    BASELINE_TEMPLATE_VERSION,
    BaselineExperimentDefinition,
    build_baseline_experiment,
    default_baseline_definition,
)
from archivetrust.htr.experiment.models import ExperimentImmutableError, assert_experiment_mutable
from archivetrust.providers.florence2_htr.adapter import METHOD_ID as FLORENCE2_METHOD_ID
from archivetrust.providers.satrn.adapter import METHOD_ID as SATRN_METHOD_ID
from archivetrust.providers.transkribus.adapter import METHOD_ID as TRANSKRIBUS_METHOD_ID


def _definition() -> BaselineExperimentDefinition:
    return default_baseline_definition(satrn_model_revision="rev_satrn", florence2_model_revision="rev_florence2")


def test_default_definition_names_all_three_methods_in_a_stable_order():
    definition = _definition()
    assert definition.method_ids == (SATRN_METHOD_ID, FLORENCE2_METHOD_ID, TRANSKRIBUS_METHOD_ID)
    assert definition.title == BASELINE_TEMPLATE_TITLE


def test_default_definition_method_versions_use_the_passed_in_real_revisions_not_hardcoded_ones():
    definition = _definition()
    assert definition.method_versions[SATRN_METHOD_ID] == "rev_satrn"
    assert definition.method_versions[FLORENCE2_METHOD_ID] == "rev_florence2"
    # Transkribus's own adapter honestly reports "unpinned" -- never a fabricated version string.
    assert "unpinned" in definition.method_versions[TRANSKRIBUS_METHOD_ID]


def test_definition_carries_every_experiment_definition_field_the_brief_names():
    """Every field name from the brief's "Experiment definition" list must be a real, non-empty
    field on this model -- this test exists so a future edit that silently drops one of them is
    caught immediately, not discovered by a reader of the template months later."""
    definition = _definition()
    required_fields = (
        "title",
        "research_question",
        "hypothesis",
        "dataset_description",
        "dataset_version_note",
        "inclusion_criteria",
        "exclusion_criteria",
        "sampling_strategy",
        "transcription_convention",
        "ground_truth_requirements",
        "method_ids",
        "method_versions",
        "segmentation_strategy",
        "controlled_variables",
        "independent_variables",
        "dependent_variables",
        "metric_names",
        "failure_policy",
        "acceptance_criteria",
        "hardware_environment_requirements",
        "software_environment_requirements",
        "random_seed",
        "reporting_strategy",
    )
    for field_name in required_fields:
        value = getattr(definition, field_name)
        assert value not in (None, "", (), {}), f"{field_name} must be populated, got {value!r}"


def test_definition_honestly_discloses_its_scope_caveats():
    """The brief's "do not fabricate a larger dataset than exists" instruction -- this asserts the
    honest caveats are actually present, not merely that the field exists."""
    definition = _definition()
    assert len(definition.scope_caveats) >= 1
    joined = " ".join(definition.scope_caveats).lower()
    assert "one line" in joined or "not a corpus" in joined or "one document" in joined


def test_to_ref_is_deterministic_for_identical_field_values():
    definition_a = _definition()
    definition_b = _definition()
    assert definition_a.to_ref() == definition_b.to_ref()
    # And it really is valid JSON, not an opaque string a caller could not parse back if needed.
    json.loads(definition_a.to_ref())


def test_to_ref_changes_when_a_field_changes():
    definition_a = _definition()
    definition_b = default_baseline_definition(satrn_model_revision="different_rev", florence2_model_revision="rev_florence2")
    assert definition_a.to_ref() != definition_b.to_ref()


def test_build_baseline_experiment_produces_real_experiment_and_version():
    definition = _definition()
    built = build_baseline_experiment(
        definition,
        research_project_id="research_project_1",
        dataset_version_id="dataset_version_1",
        created_at="2026-01-01T00:00:00+00:00",
    )
    assert built.experiment.name == BASELINE_TEMPLATE_TITLE
    assert built.experiment.research_project_id == "research_project_1"
    assert built.experiment_version.experiment_id == built.experiment.experiment_id
    assert built.experiment_version.dataset_version_id == "dataset_version_1"
    assert built.experiment_version.method_ids == definition.method_ids
    assert built.experiment_version.version == BASELINE_TEMPLATE_VERSION
    # pipeline_configuration_ref carries the full definition (see module docstring's
    # SELECTION_NOT_MODELED-precedent explanation) -- verify it round-trips as real JSON, not a
    # truncated or lossy summary.
    ref_payload = json.loads(built.experiment_version.pipeline_configuration_ref)
    assert ref_payload["research_question"] == definition.research_question
    assert ref_payload["method_ids"] == list(definition.method_ids)


def test_built_experiment_version_is_well_formed_pydantic():
    definition = _definition()
    built = build_baseline_experiment(
        definition,
        research_project_id="research_project_1",
        dataset_version_id="dataset_version_1",
        created_at="2026-01-01T00:00:00+00:00",
    )
    restored = type(built.experiment_version).model_validate(built.experiment_version.model_dump())
    assert restored == built.experiment_version


def test_experiment_is_mutable_before_any_run_exists():
    definition = _definition()
    built = build_baseline_experiment(
        definition,
        research_project_id="research_project_1",
        dataset_version_id="dataset_version_1",
        created_at="2026-01-01T00:00:00+00:00",
    )
    # No ExperimentRun has been created yet -- this must not raise.
    assert_experiment_mutable(built.experiment.experiment_id, existing_runs=(), existing_versions=(built.experiment_version,))


def test_experiment_becomes_immutable_once_a_run_references_its_version():
    from archivetrust.htr.experiment.models import ExperimentRun

    definition = _definition()
    built = build_baseline_experiment(
        definition,
        research_project_id="research_project_1",
        dataset_version_id="dataset_version_1",
        created_at="2026-01-01T00:00:00+00:00",
    )
    run = ExperimentRun.create(
        experiment_version_id=built.experiment_version.experiment_version_id,
        is_end_to_end=False,
        started_at="2026-01-01T00:00:01+00:00",
    )
    with pytest.raises(ExperimentImmutableError):
        assert_experiment_mutable(
            built.experiment.experiment_id, existing_runs=(run,), existing_versions=(built.experiment_version,)
        )


def test_exclusion_criteria_explicitly_addresses_transkribus_non_correspondence():
    """The brief's explicit hard requirement: Transkribus must never be silently folded into a
    controlled line-level comparison it has no geometric correspondence to. Assert this is stated
    in the template's own definition, not left implicit."""
    definition = _definition()
    assert "transkribus" in definition.exclusion_criteria.lower()
    assert "controlled" in definition.exclusion_criteria.lower()
