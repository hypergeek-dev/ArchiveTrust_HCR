from __future__ import annotations

import pytest
from pydantic import ValidationError

from archivetrust.htr.experiment import (
    Experiment,
    ExperimentImmutableError,
    ExperimentRun,
    ExperimentVersion,
    FailureRecord,
    MetricDefinition,
    MetricResult,
    MethodRun,
    ReproducibilityManifest,
    assert_experiment_mutable,
)


def test_experiment_round_trips():
    experiment = Experiment.create(
        name="SATRN vs Florence-2", research_project_id="research_project_1", created_at="2026-01-01T00:00:00Z"
    )
    restored = Experiment.model_validate(experiment.model_dump())
    assert restored == experiment


def test_experiment_version_requires_at_least_one_method():
    with pytest.raises(ValidationError):
        ExperimentVersion(
            experiment_version_id="experiment_version_x",
            experiment_id="experiment_x",
            version=1,
            dataset_version_id="dataset_version_x",
            method_ids=(),
            created_at="2026-01-01T00:00:00Z",
        )


def test_experiment_version_valid_construction():
    version = ExperimentVersion.create(
        experiment_id="experiment_x",
        version=1,
        dataset_version_id="dataset_version_x",
        method_ids=("method_satrn",),
        created_at="2026-01-01T00:00:00Z",
    )
    assert version.method_ids == ("method_satrn",)


def test_assert_experiment_mutable_raises_once_a_run_exists():
    version = ExperimentVersion.create(
        experiment_id="experiment_x",
        version=1,
        dataset_version_id="dataset_version_x",
        method_ids=("method_satrn",),
        created_at="2026-01-01T00:00:00Z",
    )
    run = ExperimentRun.create(
        experiment_version_id=version.experiment_version_id,
        is_end_to_end=False,
        started_at="2026-01-01T00:00:00Z",
    )
    with pytest.raises(ExperimentImmutableError):
        assert_experiment_mutable("experiment_x", existing_runs=(run,), existing_versions=(version,))


def test_assert_experiment_mutable_allows_when_no_run_exists():
    version = ExperimentVersion.create(
        experiment_id="experiment_x",
        version=1,
        dataset_version_id="dataset_version_x",
        method_ids=("method_satrn",),
        created_at="2026-01-01T00:00:00Z",
    )
    assert_experiment_mutable("experiment_x", existing_runs=(), existing_versions=(version,))


def test_method_run_round_trips():
    run = MethodRun.create(
        experiment_run_id="experiment_run_1",
        method_id="method_satrn",
        evidence_id="evidence_1",
        outcome="succeeded",
        started_at="2026-01-01T00:00:00Z",
    )
    restored = MethodRun.model_validate(run.model_dump())
    assert restored == run


def test_failure_record_round_trips():
    record = FailureRecord.create(method_run_id="method_run_1", reason="out of memory")
    restored = FailureRecord.model_validate(record.model_dump())
    assert restored == record


def test_metric_definition_rejects_version_below_one():
    with pytest.raises(ValidationError):
        MetricDefinition(metric_definition_id="metric_definition_x", name="CER", version=0)


def test_metric_result_round_trips():
    definition = MetricDefinition.create(name="CER", version=1, higher_is_better=False)
    result = MetricResult.create(
        metric_definition_id=definition.metric_definition_id, method_run_id="method_run_1", value=0.12
    )
    assert result.value == pytest.approx(0.12)


def test_reproducibility_manifest_round_trips():
    manifest = ReproducibilityManifest.create(
        experiment_run_id="experiment_run_1",
        created_at="2026-01-01T00:00:00Z",
        software_environment={"python": "3.11"},
    )
    restored = ReproducibilityManifest.model_validate(manifest.model_dump())
    assert restored == manifest
