"""A normalization-configuration change forces a new `ExperimentVersion`.

Uses the *existing* versioning mechanism -- `BaselineExperimentDefinition.to_ref()` ->
`ExperimentVersion.pipeline_configuration_ref` -> `assert_experiment_mutable` -- rather than a
parallel one. These tests prove the chain actually closes: that two runs with different normalization
configurations cannot be presented as the same pipeline configuration.
"""

from __future__ import annotations

import json

import pytest

from archivetrust.htr.experiment.baseline_template import (
    ImageColorNormalizationRequirement,
    build_baseline_experiment,
    default_baseline_definition,
    swedish_lion_1_page_level_definition,
)
from archivetrust.htr.experiment.models import (
    ExperimentImmutableError,
    ExperimentRun,
    assert_experiment_mutable,
)
from archivetrust.htr.preprocessing.models import (
    RGB_NORMALIZATION_VERSION,
    AlphaCompositingPolicy,
    RgbNormalizationConfig,
)

AT = "2026-07-30T09:00:00+00:00"


def _definition(**kwargs):
    return swedish_lion_1_page_level_definition(
        dataset_description="One page image for the normalization demonstration.",
        dataset_version_note="DatasetVersion.version=1.",
        **kwargs,
    )


def test_the_page_level_definition_requires_image_color_normalization():
    definition = _definition()

    requirement = definition.image_color_normalization
    assert requirement is not None
    assert requirement.required is True
    assert requirement.stage == "image_color_normalization"
    assert requirement.version == RGB_NORMALIZATION_VERSION
    assert requirement.configuration_hash == RgbNormalizationConfig().configuration_hash
    assert requirement.enhancement == "none"
    assert requirement.target_mode == "RGB"
    assert requirement.alpha_policy == "white_background"
    assert requirement.icc_profile_policy == "record_and_strip_without_applying"
    assert "ArchiveTrust performs no upload" in requirement.external_processing_boundary


def test_the_requirement_lands_inside_pipeline_configuration_ref():
    """The requirement must reach `ExperimentVersion`, not just the template object -- otherwise the
    experiment record would not carry the configuration it was run under."""
    built = build_baseline_experiment(
        _definition(),
        research_project_id="research_project_1",
        dataset_version_id="dataset_version_1",
        created_at=AT,
    )

    payload = json.loads(built.experiment_version.pipeline_configuration_ref)

    assert payload["image_color_normalization"]["required"] is True
    assert payload["image_color_normalization"]["configuration_hash"] == (
        RgbNormalizationConfig().configuration_hash
    )
    assert payload["image_color_normalization"]["version"] == RGB_NORMALIZATION_VERSION


def test_a_different_normalization_config_forces_a_different_pipeline_configuration():
    """The central versioning guarantee: two runs with different normalization configs cannot share a
    `pipeline_configuration_ref`, so they cannot be presented as the same pipeline configuration."""
    default = _definition()
    black = _definition(
        normalization_config=RgbNormalizationConfig(
            alpha_policy=AlphaCompositingPolicy.BLACK_BACKGROUND
        )
    )

    assert default.to_ref() != black.to_ref()
    assert (
        default.image_color_normalization.configuration_hash
        != black.image_color_normalization.configuration_hash
    )


def test_the_same_normalization_config_is_deterministically_the_same_ref():
    """The complement: identical configuration must not spuriously produce a new version."""
    assert _definition().to_ref() == _definition().to_ref()


def test_a_config_change_forces_a_new_experiment_version_through_the_existing_mechanism():
    """End to end, using the real immutability mechanism.

    An `ExperimentVersion` with a run cannot be edited (`assert_experiment_mutable` raises), so the
    only way to record the changed normalization configuration is a new `ExperimentVersion` -- which
    then carries a genuinely different `pipeline_configuration_ref`.
    """
    built = build_baseline_experiment(
        _definition(),
        research_project_id="research_project_1",
        dataset_version_id="dataset_version_1",
        created_at=AT,
    )
    run = ExperimentRun.create(
        experiment_version_id=built.experiment_version.experiment_version_id,
        is_end_to_end=True,
        started_at=AT,
    )

    # Once a run exists, the experiment is frozen -- the pre-existing mechanism, not a new one.
    with pytest.raises(ExperimentImmutableError):
        assert_experiment_mutable(
            built.experiment.experiment_id, [run], [built.experiment_version]
        )

    # So the changed configuration is recorded as version 2, superseding version 1.
    changed = build_baseline_experiment(
        _definition(
            normalization_config=RgbNormalizationConfig(
                alpha_policy=AlphaCompositingPolicy.BLACK_BACKGROUND
            )
        ),
        research_project_id="research_project_1",
        dataset_version_id="dataset_version_1",
        created_at=AT,
        version=2,
    )

    assert changed.experiment_version.version == 2
    assert (
        changed.experiment_version.pipeline_configuration_ref
        != built.experiment_version.pipeline_configuration_ref
    )
    assert run.experiment_version_id == built.experiment_version.experiment_version_id, (
        "the existing run still points at the exact version it was executed under"
    )


def test_bumping_the_normalization_implementation_version_also_changes_the_ref():
    """Not only the configuration but the *implementation* version is part of the pipeline
    configuration, so a code change that could alter output bytes forces a new version too."""
    config = RgbNormalizationConfig()
    current = ImageColorNormalizationRequirement.from_config(
        config, external_processing_boundary="boundary"
    )
    future = current.model_copy(update={"version": "2.0.0"})

    assert current.version != future.version
    assert current.model_dump() != future.model_dump()


def test_a_normalization_config_can_be_recorded_as_considered_and_not_required():
    """`required=False` is expressible, so a controlled line-level configuration can record that it
    thought about normalization rather than being silent about it."""
    requirement = ImageColorNormalizationRequirement.from_config(
        RgbNormalizationConfig(), required=False, external_processing_boundary="n/a"
    )

    assert requirement.required is False


def test_the_committed_baseline_definition_still_records_no_normalization():
    """The pre-existing controlled-comparison template must remain untouched: its Transkribus run had
    no page image and no normalization, and `None` is the truthful value. Populating it would be
    fabricating history."""
    definition = default_baseline_definition(
        satrn_model_revision="rev-satrn", florence2_model_revision="rev-florence2"
    )

    assert definition.image_color_normalization is None
    payload = json.loads(definition.to_ref())
    assert payload["image_color_normalization"] is None
