"""Experiment entities (docs/htr-domain-design.md §1, §3, §5).

`Experiment → ExperimentVersion → ExperimentRun → MethodRun → {RawResult, ParsedResult,
NormalizedResult, FailureRecord, MetricResult}`. Owned by this package, orchestrated by
`application/pipeline.py` (a later stage).

**Immutability rule (§3):** "An `Experiment` is mutable only until its first `ExperimentRun` is
created (mirrors the existing 'supersede, don't overwrite' pattern...). Any edit after that
creates a new `ExperimentVersion`; every `ExperimentRun` points at the exact `ExperimentVersion`
used." `Experiment`/`ExperimentVersion` cannot enforce this from their own fields alone -- an
`Experiment` has no way to know whether any `ExperimentRun` referencing it exists. Enforcement
therefore lives at the service/orchestration boundary (the future `application/pipeline.py`
experiment-editing entry point), using `assert_experiment_mutable` below, which every mutation
path must call before constructing a new `ExperimentVersion` in place of editing an old one.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from archivetrust.domain.shared.ids import new_id


class ExperimentImmutableError(ValueError):
    """Raised when a caller attempts to mutate an `Experiment` that already has at least one
    `ExperimentRun` -- see module docstring."""


class DomainRelationship(str, Enum):
    """Whether a method's primary training/language domain matches the corpus an `ExperimentVersion`
    runs it over (docs/experiments/lion-loghi-comparison/README.md "Language-domain model").

    Deliberately a controlled, explicitly-recorded value on `ExperimentVersion` -- never inferred from
    a dataset's file location or method name. "Loghi on Dutch pages" and "Loghi on Swedish pages" are
    both real `ExperimentVersion`s with the *same* method and *different* `domain_relationship`
    values, and that difference must be a recorded fact, not a naming convention someone has to
    remember to interpret correctly.
    """

    IN_DOMAIN = "in_domain"
    CROSS_DOMAIN = "cross_domain"
    MIXED_DOMAIN = "mixed_domain"
    UNKNOWN = "unknown"


class Experiment(BaseModel):
    """A named research question -- e.g. "SATRN vs. Florence-2 on 19th-century Swedish court
    records". Carries no configuration itself; configuration lives on `ExperimentVersion` so it
    can evolve pre-first-run without being a new `Experiment`."""

    model_config = ConfigDict(frozen=True)

    experiment_id: str
    name: str
    description: str | None = None
    research_project_id: str
    created_at: str

    @classmethod
    def create(
        cls,
        *,
        name: str,
        research_project_id: str,
        description: str | None = None,
        created_at: str,
    ) -> "Experiment":
        return cls(
            experiment_id=new_id("experiment"),
            name=name,
            description=description,
            research_project_id=research_project_id,
            created_at=created_at,
        )


class ExperimentVersion(BaseModel):
    """One versioned configuration of an `Experiment` (§3): which `DatasetVersion`, which methods,
    which segmentation/pipeline configuration. Frozen once any `ExperimentRun` references it --
    enforced the same way (`assert_experiment_mutable`), never by a field on this type.
    """

    model_config = ConfigDict(frozen=True)

    experiment_version_id: str
    experiment_id: str
    version: int
    dataset_version_id: str
    method_ids: tuple[str, ...]
    segmentation_configuration_ref: str | None = None
    pipeline_configuration_ref: str | None = None
    created_at: str
    supersedes: str | None = None
    corpus_language: str | None = None
    """The corpus's language, e.g. `"sv"`/`"nl"` -- additive field (docs/experiments/
    lion-loghi-comparison/README.md), `None` for every pre-existing `ExperimentVersion` so the sealed
    reliability-run records replay unchanged with no fabricated retroactive classification."""
    method_primary_language_domain: str | None = None
    """The method's own primary training/language domain, recorded explicitly rather than assumed
    from its name (a method named "Swedish Lion" is not proof its primary domain is Swedish)."""
    domain_relationship: DomainRelationship | None = None
    """`corpus_language` vs. `method_primary_language_domain`, as an explicit classification -- see
    `DomainRelationship`'s docstring. `None` for every `ExperimentVersion` that predates this field,
    which is the honest state: no domain classification was ever recorded for those runs, not
    `UNKNOWN` (a real classification value someone chose) standing in for "never asked"."""

    @model_validator(mode="after")
    def _validate(self) -> "ExperimentVersion":
        if self.version < 1:
            raise ValueError("ExperimentVersion.version must be >= 1")
        if not self.method_ids:
            raise ValueError("ExperimentVersion.method_ids must be non-empty")
        return self

    @classmethod
    def create(
        cls,
        *,
        experiment_id: str,
        version: int,
        dataset_version_id: str,
        method_ids: tuple[str, ...],
        segmentation_configuration_ref: str | None = None,
        pipeline_configuration_ref: str | None = None,
        created_at: str,
        supersedes: str | None = None,
        corpus_language: str | None = None,
        method_primary_language_domain: str | None = None,
        domain_relationship: DomainRelationship | None = None,
    ) -> "ExperimentVersion":
        return cls(
            experiment_version_id=new_id("experiment_version"),
            experiment_id=experiment_id,
            version=version,
            dataset_version_id=dataset_version_id,
            method_ids=method_ids,
            segmentation_configuration_ref=segmentation_configuration_ref,
            pipeline_configuration_ref=pipeline_configuration_ref,
            created_at=created_at,
            supersedes=supersedes,
            corpus_language=corpus_language,
            method_primary_language_domain=method_primary_language_domain,
            domain_relationship=domain_relationship,
        )


class ExperimentRun(BaseModel):
    """One execution of an `ExperimentVersion` -- produces one `MethodRun` per (method, input)
    pair. `is_end_to_end` labels which segmentation mode was used (§7: "the ExperimentRun record
    labels which mode was used, per the brief's 'do not present end-to-end comparisons as pure
    recognizer comparisons'")."""

    model_config = ConfigDict(frozen=True)

    experiment_run_id: str
    experiment_version_id: str
    is_end_to_end: bool
    """True: each method used its own preferred segmentation. False: every method consumed the
    same, once-produced `InputCrop`s (a controlled comparison, §7)."""
    started_at: str
    completed_at: str | None = None

    @classmethod
    def create(
        cls,
        *,
        experiment_version_id: str,
        is_end_to_end: bool,
        started_at: str,
        completed_at: str | None = None,
    ) -> "ExperimentRun":
        return cls(
            experiment_run_id=new_id("experiment_run"),
            experiment_version_id=experiment_version_id,
            is_end_to_end=is_end_to_end,
            started_at=started_at,
            completed_at=completed_at,
        )


def assert_experiment_mutable(
    experiment_id: str, existing_runs: Iterable[ExperimentRun], existing_versions: Iterable[ExperimentVersion]
) -> None:
    """The enforcement point for the module docstring's immutability rule. A caller about to
    mutate an `Experiment` (or edit an `ExperimentVersion` in place instead of creating a new one)
    must call this first. Raises `ExperimentImmutableError` if any `ExperimentRun` already
    references any `ExperimentVersion` of this `Experiment`.
    """
    version_ids = {v.experiment_version_id for v in existing_versions if v.experiment_id == experiment_id}
    for run in existing_runs:
        if run.experiment_version_id in version_ids:
            raise ExperimentImmutableError(
                f"Experiment {experiment_id!r} already has ExperimentRun "
                f"{run.experiment_run_id!r} -- create a new ExperimentVersion instead of "
                "mutating this one (docs/htr-domain-design.md §3)"
            )


class MethodRun(BaseModel):
    """One method x input execution within an `ExperimentRun` (§1, §4). Chains to its
    `RawResult`/`ParsedResult`/`NormalizedResult` via the retained Evidence/Observation substrate
    (§2) -- this type itself carries only the ids of whichever `Evidence`/`Observation` records
    those stages produced, never an embedded copy (Constitution Article 7)."""

    model_config = ConfigDict(frozen=True)

    method_run_id: str
    experiment_run_id: str
    method_id: str
    model_version_id: str | None = None
    input_crop_id: str | None = None
    """Set for line-level, controlled-comparison runs; `None` for a `MethodRun` that consumed its
    own end-to-end segmentation instead (§7)."""
    evidence_id: str
    """The `Evidence` record this run's raw output was captured into (§4's traceability chain)."""
    outcome: str
    """`"succeeded" | "failed" | "no_output"` -- mirrors `ProviderInvocationOutcome`'s triad
    (Constitution Article 18) at the HTR method-run granularity."""
    started_at: str
    completed_at: str | None = None

    @classmethod
    def create(
        cls,
        *,
        experiment_run_id: str,
        method_id: str,
        evidence_id: str,
        outcome: str,
        started_at: str,
        model_version_id: str | None = None,
        input_crop_id: str | None = None,
        completed_at: str | None = None,
    ) -> "MethodRun":
        return cls(
            method_run_id=new_id("method_run"),
            experiment_run_id=experiment_run_id,
            method_id=method_id,
            model_version_id=model_version_id,
            input_crop_id=input_crop_id,
            evidence_id=evidence_id,
            outcome=outcome,
            started_at=started_at,
            completed_at=completed_at,
        )


class FailureRecord(BaseModel):
    """A `MethodRun` that could not complete -- preserved, never excluded (§1: "FailureRecord (if
    applicable — preserved, never excluded)"), mirroring `ProviderObservationAttempted`'s
    `FAILED` outcome discipline (Constitution Article 18) at method-run granularity."""

    model_config = ConfigDict(frozen=True)

    failure_record_id: str
    method_run_id: str
    reason: str
    category: str | None = None

    @classmethod
    def create(cls, *, method_run_id: str, reason: str, category: str | None = None) -> "FailureRecord":
        return cls(
            failure_record_id=new_id("failure_record"),
            method_run_id=method_run_id,
            reason=reason,
            category=category,
        )


class MetricDefinition(BaseModel):
    """A versioned metric (mirrors `METRICS_VERSION` in `evaluation/metrics.py`, §3: "versioned
    like METRICS_VERSION ... extended, not replaced")."""

    model_config = ConfigDict(frozen=True)

    metric_definition_id: str
    name: str
    version: int
    description: str | None = None
    higher_is_better: bool = True

    @model_validator(mode="after")
    def _validate(self) -> "MetricDefinition":
        if self.version < 1:
            raise ValueError("MetricDefinition.version must be >= 1")
        return self

    @classmethod
    def create(
        cls,
        *,
        name: str,
        version: int,
        description: str | None = None,
        higher_is_better: bool = True,
    ) -> "MetricDefinition":
        return cls(
            metric_definition_id=new_id("metric_definition"),
            name=name,
            version=version,
            description=description,
            higher_is_better=higher_is_better,
        )


class MetricResult(BaseModel):
    """One `MetricDefinition` computed for one `MethodRun`."""

    model_config = ConfigDict(frozen=True)

    metric_result_id: str
    metric_definition_id: str
    method_run_id: str
    value: float

    @classmethod
    def create(cls, *, metric_definition_id: str, method_run_id: str, value: float) -> "MetricResult":
        return cls(
            metric_result_id=new_id("metric_result"),
            metric_definition_id=metric_definition_id,
            method_run_id=method_run_id,
            value=value,
        )


class ReproducibilityManifest(BaseModel):
    """One per `ExperimentRun` (§1, §4) -- everything needed to reproduce it: software/hardware
    environment, pinned method/model versions, configuration hashes. Mirrors
    `ProvenanceContextEstablished`'s "fixed environment" role (domain/telemetry/events.py) but at
    experiment-run rather than per-document granularity."""

    model_config = ConfigDict(frozen=True)

    manifest_id: str
    experiment_run_id: str
    git_commit: str | None = None
    software_environment: dict = Field(default_factory=dict)
    hardware_environment: dict = Field(default_factory=dict)
    pipeline_configuration_hash: str | None = None
    created_at: str

    @classmethod
    def create(
        cls,
        *,
        experiment_run_id: str,
        created_at: str,
        git_commit: str | None = None,
        software_environment: dict | None = None,
        hardware_environment: dict | None = None,
        pipeline_configuration_hash: str | None = None,
    ) -> "ReproducibilityManifest":
        return cls(
            manifest_id=new_id("reproducibility_manifest"),
            experiment_run_id=experiment_run_id,
            git_commit=git_commit,
            software_environment=software_environment or {},
            hardware_environment=hardware_environment or {},
            pipeline_configuration_hash=pipeline_configuration_hash,
            created_at=created_at,
        )


class ExperimentComparisonGroup(BaseModel):
    """A named parent grouping over several `Experiment`s -- e.g. the four Lion-vs-Loghi cells
    (Lion×Swedish, Lion×Dutch, Loghi×Swedish, Loghi×Dutch). Modeled on `htr/corpus/models.py::
    Collection`'s "named group of related things, referenced by id" shape: this groups experiments for
    reporting/navigation without collapsing them into one `ExperimentRun` or one result set (the
    brief's explicit "do not put all four conditions into one ExperimentRun... use a parent comparison
    identifier to group them").
    """

    model_config = ConfigDict(frozen=True)

    comparison_id: str
    name: str
    experiment_ids: tuple[str, ...]
    created_at: str

    @model_validator(mode="after")
    def _validate(self) -> "ExperimentComparisonGroup":
        if not self.experiment_ids:
            raise ValueError("ExperimentComparisonGroup.experiment_ids must be non-empty")
        if len(self.experiment_ids) != len(set(self.experiment_ids)):
            raise ValueError("ExperimentComparisonGroup.experiment_ids must not repeat an id")
        return self

    @classmethod
    def create(
        cls, *, name: str, experiment_ids: tuple[str, ...], created_at: str
    ) -> "ExperimentComparisonGroup":
        return cls(
            comparison_id=new_id("experiment_comparison_group"),
            name=name,
            experiment_ids=experiment_ids,
            created_at=created_at,
        )
