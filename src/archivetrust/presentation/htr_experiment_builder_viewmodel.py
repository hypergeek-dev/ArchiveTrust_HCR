"""Experiment builder ViewModel (Stage 11, brief's "User interface" -> experiment builder).

Form-backing state for composing an experiment, with validation that runs continuously (so a View
can disable "Create" and show *why*) and a `build()` that constructs a real
`Experiment` + `ExperimentVersion` from `htr/experiment/models.py`. Those models are reused
unchanged -- this module defines no competing experiment type.

**What is a real model field vs. a configuration reference.** `ExperimentVersion` has exactly one
free-form pair of configuration hooks: `segmentation_configuration_ref` and
`pipeline_configuration_ref`. The brief's remaining knobs -- evaluation mode, metric selection,
transcription convention, hardware profile, failure policy, report template -- have no dedicated
field there, and inventing six new fields on a model an earlier stage froze would be a domain change
smuggled in through a UI stage. They are instead collected here and serialized into a stable,
sorted `pipeline_configuration_ref` string (`PipelineConfiguration.to_ref()`), which is exactly what
that field is for. `SELECTION_NOT_MODELED` records this so the choice is visible rather than
looking like an oversight.

**Nothing is persisted by this ViewModel.** `build()` returns the constructed entities; registering
them in a store is the caller's decision, keeping this class pure and testable.
"""

from __future__ import annotations

import json
from enum import Enum

from pydantic import BaseModel, ConfigDict

from archivetrust.htr.experiment.models import Experiment, ExperimentVersion
from archivetrust.presentation.display_names import method_label
from archivetrust.presentation.observable import Observable

SELECTION_NOT_MODELED = (
    "Evaluation mode, metric selection, transcription convention, hardware profile, failure "
    "policy, and report template have no dedicated ExperimentVersion field; they are serialized "
    "into pipeline_configuration_ref rather than added as new model fields by a UI stage."
)


class EvaluationMode(str, Enum):
    """Which comparison the run is (`docs/htr-domain-design.md` §7). This is the single most
    consequential choice on the form: it decides whether the results may be read as a recognizer
    comparison at all, and it maps directly onto `ExperimentRun.is_end_to_end`."""

    CONTROLLED_SHARED_SEGMENTATION = "controlled_shared_segmentation"
    END_TO_END_METHOD_SEGMENTATION = "end_to_end_method_segmentation"

    @property
    def is_end_to_end(self) -> bool:
        return self is EvaluationMode.END_TO_END_METHOD_SEGMENTATION

    @property
    def label(self) -> str:
        return {
            EvaluationMode.CONTROLLED_SHARED_SEGMENTATION: (
                "Controlled - every method reads the same input crops"
            ),
            EvaluationMode.END_TO_END_METHOD_SEGMENTATION: (
                "End-to-end - each method uses its own segmentation (not a pure recognizer "
                "comparison)"
            ),
        }[self]


class FailurePolicy(str, Enum):
    """What a failed `MethodRun` does to the run. `PRESERVE_AND_CONTINUE` is the only value
    consistent with `docs/htr-domain-design.md` §1's "FailureRecord ... preserved, never excluded";
    the other two change scheduling, never whether the failure is recorded."""

    PRESERVE_AND_CONTINUE = "preserve_and_continue"
    PRESERVE_AND_HALT = "preserve_and_halt"
    PRESERVE_AND_RETRY_ONCE = "preserve_and_retry_once"

    @property
    def label(self) -> str:
        return {
            FailurePolicy.PRESERVE_AND_CONTINUE: "Record the failure and continue",
            FailurePolicy.PRESERVE_AND_HALT: "Record the failure and halt the run",
            FailurePolicy.PRESERVE_AND_RETRY_ONCE: "Record the failure and retry once",
        }[self]


class ValidationIssue(BaseModel):
    """One reason the form cannot be submitted yet, tied to the field that caused it so a View can
    put the message next to the right control."""

    model_config = ConfigDict(frozen=True)

    field: str
    message: str


class PipelineConfiguration(BaseModel):
    """The brief's non-`ExperimentVersion`-modeled selections, as one serializable record. Its
    `to_ref()` is what lands in `ExperimentVersion.pipeline_configuration_ref`."""

    model_config = ConfigDict(frozen=True)

    evaluation_mode: EvaluationMode
    metric_definition_ids: tuple[str, ...]
    transcription_convention_id: str | None
    transcription_convention_version: int | None
    hardware_profile: str | None
    failure_policy: FailurePolicy
    report_template: str | None
    sample_size: int | None
    """`None` means the whole dataset version; an integer means a sample of that many items."""
    sample_seed: int | None

    def to_ref(self) -> str:
        """A deterministic, sorted JSON string. Deterministic because two experiment versions built
        from identical selections must produce the identical ref -- otherwise reproducibility
        comparison across runs becomes a string-diff exercise."""
        return json.dumps(
            {
                "evaluation_mode": self.evaluation_mode.value,
                "metric_definition_ids": sorted(self.metric_definition_ids),
                "transcription_convention_id": self.transcription_convention_id,
                "transcription_convention_version": self.transcription_convention_version,
                "hardware_profile": self.hardware_profile,
                "failure_policy": self.failure_policy.value,
                "report_template": self.report_template,
                "sample_size": self.sample_size,
                "sample_seed": self.sample_seed,
            },
            sort_keys=True,
            separators=(",", ":"),
        )


class BuiltExperiment(BaseModel):
    """What `build()` returns -- both entities, plus the configuration that produced the version's
    `pipeline_configuration_ref`, so a caller can record all three together."""

    model_config = ConfigDict(frozen=True)

    experiment: Experiment
    experiment_version: ExperimentVersion
    pipeline_configuration: PipelineConfiguration


class ExperimentBuilderViewModel:
    """Mutable form state with continuous validation. Every setter re-validates and pushes onto the
    `validation_issues`/`can_build` observables, so a View binds once and never polls."""

    def __init__(
        self,
        *,
        available_method_ids: tuple[str, ...] = (),
        available_metric_definition_ids: tuple[str, ...] = (),
    ) -> None:
        self._available_method_ids = available_method_ids
        self._available_metric_definition_ids = available_metric_definition_ids

        self.name: str = ""
        self.description: str | None = None
        self.research_project_id: str | None = None
        self.dataset_id: str | None = None
        self.dataset_version_id: str | None = None
        self.method_ids: tuple[str, ...] = ()
        self.model_version_ids: dict[str, str] = {}
        """`method_id -> pinned model version`. Optional per method; an unpinned method runs at the
        adapter's own default revision, which `MethodMetadata.model_revision` already reports."""
        self.segmentation_configuration_ref: str | None = None
        self.evaluation_mode: EvaluationMode = EvaluationMode.CONTROLLED_SHARED_SEGMENTATION
        self.metric_definition_ids: tuple[str, ...] = ()
        self.transcription_convention_id: str | None = None
        self.transcription_convention_version: int | None = None
        self.hardware_profile: str | None = None
        self.failure_policy: FailurePolicy = FailurePolicy.PRESERVE_AND_CONTINUE
        self.report_template: str | None = None
        self.sample_size: int | None = None
        self.sample_seed: int | None = None

        self.validation_issues: Observable[tuple[ValidationIssue, ...]] = Observable(())
        self.can_build: Observable[bool] = Observable(False)
        self.revalidate()

    # -- Choices a View renders ----------------------------------------------------------------

    def method_choices(self) -> tuple[tuple[str, str], ...]:
        """`(method_id, display label)` pairs, alphabetical by label."""
        return tuple(
            sorted(
                ((method_id, method_label(method_id)) for method_id in self._available_method_ids),
                key=lambda pair: pair[1],
            )
        )

    def evaluation_mode_choices(self) -> tuple[tuple[EvaluationMode, str], ...]:
        return tuple((mode, mode.label) for mode in EvaluationMode)

    def failure_policy_choices(self) -> tuple[tuple[FailurePolicy, str], ...]:
        return tuple((policy, policy.label) for policy in FailurePolicy)

    def metric_choices(self) -> tuple[str, ...]:
        return self._available_metric_definition_ids

    # -- Mutation ------------------------------------------------------------------------------

    def set_field(self, field: str, value: object) -> None:
        """One setter for every scalar field, so a View wires N controls to one slot and this class
        does not grow N near-identical methods. Rejects an unknown field rather than silently
        creating a new attribute that would never be read by `build()`."""
        if field not in self._settable_fields():
            raise AttributeError(f"{field!r} is not a settable experiment-builder field")
        setattr(self, field, value)
        self.revalidate()

    def toggle_method(self, method_id: str) -> None:
        if method_id in self.method_ids:
            self.method_ids = tuple(m for m in self.method_ids if m != method_id)
        else:
            self.method_ids = (*self.method_ids, method_id)
        self.revalidate()

    def toggle_metric(self, metric_definition_id: str) -> None:
        if metric_definition_id in self.metric_definition_ids:
            self.metric_definition_ids = tuple(
                m for m in self.metric_definition_ids if m != metric_definition_id
            )
        else:
            self.metric_definition_ids = (*self.metric_definition_ids, metric_definition_id)
        self.revalidate()

    def pin_model_version(self, method_id: str, model_version_id: str | None) -> None:
        if model_version_id is None:
            self.model_version_ids.pop(method_id, None)
        else:
            self.model_version_ids[method_id] = model_version_id
        self.revalidate()

    @staticmethod
    def _settable_fields() -> frozenset[str]:
        return frozenset(
            {
                "name",
                "description",
                "research_project_id",
                "dataset_id",
                "dataset_version_id",
                "segmentation_configuration_ref",
                "evaluation_mode",
                "transcription_convention_id",
                "transcription_convention_version",
                "hardware_profile",
                "failure_policy",
                "report_template",
                "sample_size",
                "sample_seed",
            }
        )

    # -- Validation ----------------------------------------------------------------------------

    def revalidate(self) -> tuple[ValidationIssue, ...]:
        issues: list[ValidationIssue] = []

        if not self.name.strip():
            issues.append(ValidationIssue(field="name", message="An experiment needs a name."))
        if not self.research_project_id:
            issues.append(
                ValidationIssue(
                    field="research_project_id", message="Select the research project."
                )
            )
        if not self.dataset_version_id:
            issues.append(
                ValidationIssue(
                    field="dataset_version_id",
                    message=(
                        "Select a dataset version. An experiment references an immutable "
                        "DatasetVersion, never a mutable Dataset."
                    ),
                )
            )
        if not self.method_ids:
            issues.append(
                ValidationIssue(field="method_ids", message="Select at least one method to run.")
            )

        unknown = [m for m in self.method_ids if m not in self._available_method_ids]
        if unknown:
            issues.append(
                ValidationIssue(
                    field="method_ids",
                    message=f"No adapter is registered for: {', '.join(sorted(unknown))}.",
                )
            )

        if (
            self.evaluation_mode is EvaluationMode.CONTROLLED_SHARED_SEGMENTATION
            and not self.segmentation_configuration_ref
        ):
            issues.append(
                ValidationIssue(
                    field="segmentation_configuration_ref",
                    message=(
                        "A controlled comparison needs one segmentation configuration, because "
                        "every method must read the same input crops."
                    ),
                )
            )

        if (self.transcription_convention_id is None) != (
            self.transcription_convention_version is None
        ):
            issues.append(
                ValidationIssue(
                    field="transcription_convention_version",
                    message=(
                        "A transcription convention is identified by id AND version; supply both "
                        "or neither."
                    ),
                )
            )

        if self.sample_size is not None and self.sample_size < 1:
            issues.append(
                ValidationIssue(
                    field="sample_size",
                    message="A sample size must be at least 1; leave it empty for the whole dataset version.",
                )
            )
        if self.sample_size is not None and self.sample_seed is None:
            issues.append(
                ValidationIssue(
                    field="sample_seed",
                    message="A sampled experiment needs a seed, or it is not reproducible.",
                )
            )

        frozen = tuple(issues)
        self.validation_issues.value = frozen
        self.can_build.value = not frozen
        return frozen

    # -- Construction --------------------------------------------------------------------------

    def pipeline_configuration(self) -> PipelineConfiguration:
        return PipelineConfiguration(
            evaluation_mode=self.evaluation_mode,
            metric_definition_ids=self.metric_definition_ids,
            transcription_convention_id=self.transcription_convention_id,
            transcription_convention_version=self.transcription_convention_version,
            hardware_profile=self.hardware_profile,
            failure_policy=self.failure_policy,
            report_template=self.report_template,
            sample_size=self.sample_size,
            sample_seed=self.sample_seed,
        )

    def build(self, *, created_at: str, version: int = 1) -> BuiltExperiment:
        """Constructs the real `Experiment` + `ExperimentVersion`. Raises `ValueError` listing every
        outstanding issue if validation does not pass -- a View that binds `can_build` never reaches
        this, but a programmatic caller cannot bypass it."""
        issues = self.revalidate()
        if issues:
            raise ValueError(
                "Experiment is not valid yet: "
                + "; ".join(f"{issue.field}: {issue.message}" for issue in issues)
            )

        assert self.research_project_id is not None
        assert self.dataset_version_id is not None

        configuration = self.pipeline_configuration()
        experiment = Experiment.create(
            name=self.name.strip(),
            research_project_id=self.research_project_id,
            description=self.description,
            created_at=created_at,
        )
        experiment_version = ExperimentVersion.create(
            experiment_id=experiment.experiment_id,
            version=version,
            dataset_version_id=self.dataset_version_id,
            method_ids=self.method_ids,
            segmentation_configuration_ref=self.segmentation_configuration_ref,
            pipeline_configuration_ref=configuration.to_ref(),
            created_at=created_at,
        )
        return BuiltExperiment(
            experiment=experiment,
            experiment_version=experiment_version,
            pipeline_configuration=configuration,
        )
