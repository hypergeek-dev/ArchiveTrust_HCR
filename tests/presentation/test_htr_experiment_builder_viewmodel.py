"""Experiment builder ViewModel: continuous validation and real entity construction."""

from __future__ import annotations

import json

import pytest

from archivetrust.htr.experiment.models import Experiment, ExperimentVersion
from archivetrust.presentation.htr_experiment_builder_viewmodel import (
    EvaluationMode,
    ExperimentBuilderViewModel,
    FailurePolicy,
)
from tests.presentation._htr_fixtures import build_fixture_corpus

CREATED_AT = "2026-07-29T12:00:00Z"


def _valid_builder(corpus) -> ExperimentBuilderViewModel:
    vm = ExperimentBuilderViewModel(
        available_method_ids=("satrn", "florence2_htr", "transkribus_swedish_lion_1"),
        available_metric_definition_ids=("cer_normalized", "wer_normalized"),
    )
    vm.set_field("name", "SATRN vs Florence-2")
    vm.set_field("research_project_id", corpus.project_id)
    vm.set_field("dataset_id", corpus.dataset_id)
    vm.set_field("dataset_version_id", corpus.dataset_version_id)
    vm.set_field("segmentation_configuration_ref", "segmentation_baseline_v1")
    vm.toggle_method("satrn")
    vm.toggle_method("florence2_htr")
    vm.toggle_metric("cer_normalized")
    return vm


def test_a_fresh_builder_cannot_build_and_says_exactly_why() -> None:
    vm = ExperimentBuilderViewModel(available_method_ids=("satrn",))

    assert vm.can_build.value is False
    fields = {issue.field for issue in vm.validation_issues.value}
    assert {"name", "research_project_id", "dataset_version_id", "method_ids"} <= fields


def test_observables_notify_as_the_form_becomes_valid() -> None:
    corpus = build_fixture_corpus()
    seen: list[bool] = []
    vm = ExperimentBuilderViewModel(available_method_ids=("satrn",))
    vm.can_build.subscribe(seen.append)

    vm.set_field("name", "Baseline")
    vm.set_field("research_project_id", corpus.project_id)
    vm.set_field("dataset_version_id", corpus.dataset_version_id)
    vm.set_field("segmentation_configuration_ref", "seg_v1")
    assert vm.can_build.value is False
    vm.toggle_method("satrn")

    assert vm.can_build.value is True
    assert seen == [True]  # observable fires only on an actual transition


def test_a_valid_builder_constructs_the_real_domain_entities() -> None:
    corpus = build_fixture_corpus()
    built = _valid_builder(corpus).build(created_at=CREATED_AT)

    assert isinstance(built.experiment, Experiment)
    assert isinstance(built.experiment_version, ExperimentVersion)
    assert built.experiment.name == "SATRN vs Florence-2"
    assert built.experiment.research_project_id == corpus.project_id
    assert built.experiment_version.experiment_id == built.experiment.experiment_id
    assert built.experiment_version.dataset_version_id == corpus.dataset_version_id
    assert built.experiment_version.method_ids == ("satrn", "florence2_htr")
    assert built.experiment_version.version == 1


def test_the_unmodeled_selections_are_serialized_into_pipeline_configuration_ref() -> None:
    """Evaluation mode, metrics, convention, hardware profile, failure policy and report template
    have no `ExperimentVersion` field; they must land in `pipeline_configuration_ref` rather than
    being dropped or bolted onto the model."""
    corpus = build_fixture_corpus()
    vm = _valid_builder(corpus)
    vm.set_field("evaluation_mode", EvaluationMode.END_TO_END_METHOD_SEGMENTATION)
    vm.set_field("transcription_convention_id", "riksarkivet_diplomatic")
    vm.set_field("transcription_convention_version", 2)
    vm.set_field("hardware_profile", "rtx_3070_cuda")
    vm.set_field("failure_policy", FailurePolicy.PRESERVE_AND_RETRY_ONCE)
    vm.set_field("report_template", "swedish_historical_htr_baseline")

    built = vm.build(created_at=CREATED_AT)
    payload = json.loads(built.experiment_version.pipeline_configuration_ref)

    assert payload["evaluation_mode"] == "end_to_end_method_segmentation"
    assert payload["metric_definition_ids"] == ["cer_normalized"]
    assert payload["transcription_convention_id"] == "riksarkivet_diplomatic"
    assert payload["transcription_convention_version"] == 2
    assert payload["hardware_profile"] == "rtx_3070_cuda"
    assert payload["failure_policy"] == "preserve_and_retry_once"
    assert payload["report_template"] == "swedish_historical_htr_baseline"


def test_identical_selections_produce_an_identical_configuration_ref() -> None:
    """Reproducibility comparison across runs depends on this being deterministic, not on the
    insertion order two researchers happened to click metrics in."""
    corpus = build_fixture_corpus()
    first = _valid_builder(corpus)
    first.toggle_metric("wer_normalized")

    second = _valid_builder(corpus)
    # Same two metrics, selected in the opposite order.
    second.toggle_metric("wer_normalized")
    second.toggle_metric("cer_normalized")
    second.toggle_metric("cer_normalized")

    assert (
        first.pipeline_configuration().to_ref() == second.pipeline_configuration().to_ref()
    )


def test_controlled_mode_requires_a_segmentation_configuration() -> None:
    """§7: in a controlled comparison every method reads the same crops, so one segmentation
    configuration must be chosen. End-to-end mode has no such requirement."""
    corpus = build_fixture_corpus()
    vm = _valid_builder(corpus)
    vm.set_field("segmentation_configuration_ref", None)

    assert vm.can_build.value is False
    assert any(
        issue.field == "segmentation_configuration_ref" for issue in vm.validation_issues.value
    )

    vm.set_field("evaluation_mode", EvaluationMode.END_TO_END_METHOD_SEGMENTATION)
    assert vm.can_build.value is True


def test_a_method_with_no_registered_adapter_is_rejected_by_name() -> None:
    corpus = build_fixture_corpus()
    vm = _valid_builder(corpus)
    vm.toggle_method("deepseek_ocr")

    assert vm.can_build.value is False
    message = next(
        issue.message for issue in vm.validation_issues.value if issue.field == "method_ids"
    )
    assert "deepseek_ocr" in message


def test_a_convention_needs_both_id_and_version_or_neither() -> None:
    corpus = build_fixture_corpus()
    vm = _valid_builder(corpus)
    vm.set_field("transcription_convention_id", "riksarkivet_diplomatic")

    assert vm.can_build.value is False
    vm.set_field("transcription_convention_version", 1)
    assert vm.can_build.value is True


def test_a_sampled_experiment_without_a_seed_is_not_reproducible_and_is_rejected() -> None:
    corpus = build_fixture_corpus()
    vm = _valid_builder(corpus)
    vm.set_field("sample_size", 50)

    assert vm.can_build.value is False
    assert any(issue.field == "sample_seed" for issue in vm.validation_issues.value)

    vm.set_field("sample_seed", 1)
    assert vm.can_build.value is True


def test_a_zero_sample_size_is_rejected_distinctly_from_the_whole_dataset() -> None:
    corpus = build_fixture_corpus()
    vm = _valid_builder(corpus)
    vm.set_field("sample_seed", 1)
    vm.set_field("sample_size", 0)

    assert vm.can_build.value is False
    assert any(issue.field == "sample_size" for issue in vm.validation_issues.value)


def test_build_raises_listing_every_issue_when_validation_fails() -> None:
    vm = ExperimentBuilderViewModel(available_method_ids=("satrn",))
    with pytest.raises(ValueError) as excinfo:
        vm.build(created_at=CREATED_AT)

    message = str(excinfo.value)
    assert "name" in message
    assert "dataset_version_id" in message


def test_setting_an_unknown_field_is_refused_rather_than_silently_ignored() -> None:
    vm = ExperimentBuilderViewModel()
    with pytest.raises(AttributeError):
        vm.set_field("evaluatoin_mode", EvaluationMode.CONTROLLED_SHARED_SEGMENTATION)


def test_toggling_a_method_twice_removes_it() -> None:
    vm = ExperimentBuilderViewModel(available_method_ids=("satrn", "florence2_htr"))
    vm.toggle_method("satrn")
    vm.toggle_method("florence2_htr")
    assert vm.method_ids == ("satrn", "florence2_htr")
    vm.toggle_method("satrn")
    assert vm.method_ids == ("florence2_htr",)


def test_method_choices_use_htr_display_labels_sorted_by_label() -> None:
    vm = ExperimentBuilderViewModel(
        available_method_ids=("transkribus_swedish_lion_1", "satrn", "florence2_htr")
    )
    labels = [label for _id, label in vm.method_choices()]

    assert labels == [
        "Florence-2 (vlm-htr line OCR)",
        "SATRN (Riksarkivet)",
        "Transkribus Swedish Lion I",
    ]


def test_evaluation_mode_maps_onto_the_experiment_run_end_to_end_flag() -> None:
    assert EvaluationMode.CONTROLLED_SHARED_SEGMENTATION.is_end_to_end is False
    assert EvaluationMode.END_TO_END_METHOD_SEGMENTATION.is_end_to_end is True
    assert "not a pure recognizer comparison" in (
        EvaluationMode.END_TO_END_METHOD_SEGMENTATION.label
    )


def test_pinned_model_versions_can_be_set_and_cleared() -> None:
    corpus = build_fixture_corpus()
    vm = _valid_builder(corpus)
    vm.pin_model_version("satrn", "a40c7093")
    assert vm.model_version_ids == {"satrn": "a40c7093"}
    vm.pin_model_version("satrn", None)
    assert vm.model_version_ids == {}
