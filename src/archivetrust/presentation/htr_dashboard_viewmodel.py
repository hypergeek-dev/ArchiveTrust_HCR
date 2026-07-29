"""Research dashboard ViewModel (Stage 11, brief's "User interface" -> overview).

The landing surface: active projects, datasets, experiments and their status, processed
documents/pages, reviewed items, unresolved disagreements, method failures, latest completed runs,
and active external (Transkribus) imports.

Every number here is derived from a real source -- `htr/research_store.py`'s registered entities,
`review/blind_review/queries.py`'s completion status, and
`htr/evaluation/failures.py::aggregate_method_run_metrics` -- and nothing is defaulted to zero to
fill a tile. Where a figure genuinely cannot be computed (no review store wired, no targets
registered) the corresponding field is `None`, and the View is expected to say so rather than
render a confident `0`. That is Constitution Article 18's discipline (an absence is a fact) applied
to a dashboard, where the temptation to show a tidy zero is strongest.

**Legacy runs are excluded from aggregates and counted separately.** `docs/htr-domain-design.md` §8
requires legacy OCR-era method runs to be labeled and "never included in HTR baseline aggregate
statistics"; `legacy_method_runs` reports how many were set aside so the exclusion is visible
rather than looking like they never existed.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from archivetrust.htr.evaluation.failures import (
    MethodRunEvaluationEntry,
    aggregate_method_run_metrics,
)
from archivetrust.htr.research_store import HtrResearchStore
from archivetrust.presentation.display_names import (
    is_legacy_method,
    method_label,
    method_run_outcome_label,
    short_ref,
)
from archivetrust.presentation.htr_review_center_viewmodel import ReviewCenterViewModel


class ProjectSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    project_id: str
    name: str
    dataset_count: int
    experiment_count: int
    created_at: str


class ExperimentStatusRow(BaseModel):
    """One experiment and the state of its runs."""

    model_config = ConfigDict(frozen=True)

    experiment_id: str
    name: str
    version_count: int
    run_count: int
    completed_run_count: int
    in_progress_run_count: int
    latest_run_id: str | None = None
    latest_run_completed_at: str | None = None
    status: str = "No runs yet"


class MethodFailureRow(BaseModel):
    """Per-method failure counts, from `aggregate_method_run_metrics` -- which counts failed runs
    into the denominator rather than dropping them."""

    model_config = ConfigDict(frozen=True)

    method_id: str
    method_display_name: str
    total_runs: int
    succeeded_runs: int
    failed_runs: int
    failure_categories: dict[str, int]


class ExternalImportRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    external_import_id: str
    method_run_id: str
    export_format: str
    imported_by: str
    imported_at: str
    transkribus_document_id: str | None = None
    vendor_reported_accuracy: float | None = None
    """Recorded when the export stated one. Never compared against ArchiveTrust's own CER/WER --
    see `providers/transkribus/external_import.py`'s module docstring."""


class DashboardSnapshot(BaseModel):
    """The whole dashboard in one frozen value, so a View renders from a single consistent read
    rather than issuing several that could interleave with a registration."""

    model_config = ConfigDict(frozen=True)

    projects: tuple[ProjectSummary, ...]
    dataset_count: int
    dataset_version_count: int
    experiments: tuple[ExperimentStatusRow, ...]
    documents_in_corpus: int
    pages_in_corpus: int
    regions_in_corpus: int
    text_lines_in_corpus: int
    lines_with_ground_truth: int
    method_run_count: int
    legacy_method_runs: int
    """Set aside from every aggregate above, per `docs/htr-domain-design.md` §8, and reported here
    so the exclusion is visible."""
    method_failures: tuple[MethodFailureRow, ...]
    latest_completed_runs: tuple[ExperimentStatusRow, ...]
    external_imports: tuple[ExternalImportRow, ...]
    reviewed_item_count: int | None = None
    """`None` when no review store was wired -- distinct from `0` reviewed items."""
    unresolved_disagreement_count: int | None = None
    pending_review_count: int | None = None
    excluded_from_benchmark_count: int | None = None
    review_status_note: str | None = None
    """Populated when a review figure above is `None`, explaining why, so a View never shows a
    blank tile with no reason."""


class ResearchDashboardViewModel:
    """Projects an `HtrResearchStore` (and, optionally, a review center) into one snapshot."""

    def __init__(
        self,
        store: HtrResearchStore,
        *,
        review_center: ReviewCenterViewModel | None = None,
        review_target_refs: Sequence[str] = (),
    ) -> None:
        self._store = store
        self._review_center = review_center
        self._review_target_refs = tuple(review_target_refs)

    def snapshot(self, *, computed_at: str) -> DashboardSnapshot:
        projects = tuple(
            ProjectSummary(
                project_id=project.project_id,
                name=project.name,
                dataset_count=len(self._store.datasets(project_id=project.project_id)),
                experiment_count=len(
                    self._store.experiments(research_project_id=project.project_id)
                ),
                created_at=project.created_at,
            )
            for project in self._store.projects()
        )

        experiments = tuple(self._experiment_row(e.experiment_id) for e in self._store.experiments())
        completed = tuple(
            sorted(
                (row for row in experiments if row.latest_run_completed_at is not None),
                key=lambda row: row.latest_run_completed_at or "",
                reverse=True,
            )
        )

        all_runs = self._store.method_runs()
        legacy_runs = tuple(run for run in all_runs if is_legacy_method(run.method_id))
        htr_runs = tuple(run for run in all_runs if not is_legacy_method(run.method_id))

        entries = tuple(
            MethodRunEvaluationEntry(
                method_run_id=run.method_run_id,
                method_id=run.method_id,
                outcome=run.outcome,
                # A failed run has no CER; `aggregate_method_run_metrics` folds it in as 1.0 rather
                # than dropping it, which is why this is left None instead of zeroed.
                character_error_rate_normalized=None if run.outcome != "succeeded" else 0.0,
                failure_category=self._first_failure_category(run.method_run_id),
            )
            for run in htr_runs
        )
        aggregates = aggregate_method_run_metrics(entries)
        method_failures = tuple(
            MethodFailureRow(
                method_id=aggregate.method_id,
                method_display_name=method_label(aggregate.method_id),
                total_runs=aggregate.total_runs,
                succeeded_runs=aggregate.succeeded_runs,
                failed_runs=aggregate.failed_runs,
                failure_categories=aggregate.failure_categories,
            )
            for aggregate in aggregates
        )

        pages = self._store.pages()
        documents = {page.archive_object_ref for page in pages}
        regions = self._store.regions()
        text_lines = self._store.text_lines()
        with_ground_truth = sum(
            1
            for line in text_lines
            if self._store.ground_truth_for_line(line.text_line_id) is not None
        )

        review = self._review_figures(computed_at=computed_at)

        return DashboardSnapshot(
            projects=projects,
            dataset_count=len(self._store.datasets()),
            dataset_version_count=len(self._store.dataset_versions()),
            experiments=experiments,
            documents_in_corpus=len(documents),
            pages_in_corpus=len(pages),
            regions_in_corpus=len(regions),
            text_lines_in_corpus=len(text_lines),
            lines_with_ground_truth=with_ground_truth,
            method_run_count=len(htr_runs),
            legacy_method_runs=len(legacy_runs),
            method_failures=method_failures,
            latest_completed_runs=completed[:5],
            external_imports=tuple(
                ExternalImportRow(
                    external_import_id=record.external_import_id,
                    method_run_id=record.method_run_id,
                    export_format=record.export_format.value,
                    imported_by=record.imported_by,
                    imported_at=record.imported_at,
                    transkribus_document_id=record.transkribus_document_id,
                    vendor_reported_accuracy=record.vendor_reported_accuracy,
                )
                for record in self._store.external_imports()
            ),
            **review,
        )

    # -- Internals -----------------------------------------------------------------------------

    def _first_failure_category(self, method_run_id: str) -> str | None:
        failures = self._store.failures(method_run_id=method_run_id)
        return failures[0].category if failures else None

    def _experiment_row(self, experiment_id: str) -> ExperimentStatusRow:
        experiment = self._store.experiment(experiment_id)
        assert experiment is not None
        versions = self._store.experiment_versions(experiment_id=experiment_id)
        runs = tuple(
            run
            for version in versions
            for run in self._store.experiment_runs(
                experiment_version_id=version.experiment_version_id
            )
        )
        completed = tuple(run for run in runs if run.completed_at is not None)
        in_progress = tuple(run for run in runs if run.completed_at is None)
        latest = max(completed, key=lambda r: r.completed_at or "", default=None)

        if not runs:
            status = "No runs yet"
        elif in_progress:
            status = f"{len(in_progress)} run(s) in progress"
        else:
            status = f"{len(completed)} run(s) complete"

        return ExperimentStatusRow(
            experiment_id=experiment.experiment_id,
            name=experiment.name,
            version_count=len(versions),
            run_count=len(runs),
            completed_run_count=len(completed),
            in_progress_run_count=len(in_progress),
            latest_run_id=latest.experiment_run_id if latest is not None else None,
            latest_run_completed_at=latest.completed_at if latest is not None else None,
            status=status,
        )

    def _review_figures(self, *, computed_at: str) -> dict:
        if self._review_center is None:
            return {
                "reviewed_item_count": None,
                "unresolved_disagreement_count": None,
                "pending_review_count": None,
                "excluded_from_benchmark_count": None,
                "review_status_note": (
                    "No blind-review store is wired into this dashboard, so review figures are "
                    "unknown rather than zero."
                ),
            }
        if not self._review_target_refs:
            return {
                "reviewed_item_count": None,
                "unresolved_disagreement_count": None,
                "pending_review_count": None,
                "excluded_from_benchmark_count": None,
                "review_status_note": (
                    "A review store is wired but no review targets are registered, so there is "
                    "nothing to report on yet."
                ),
            }

        status = self._review_center.completion_status(
            self._review_target_refs, computed_at=computed_at
        )
        unresolved = self._review_center.pending_disagreements(
            self._review_target_refs, computed_at=computed_at
        )
        return {
            "reviewed_item_count": status.both_submitted_count,
            "unresolved_disagreement_count": len(unresolved),
            "pending_review_count": status.pending_count,
            "excluded_from_benchmark_count": status.excluded_count,
            "review_status_note": None,
        }

    def experiment_run_detail(self, experiment_run_id: str) -> tuple[str, ...]:
        """Flat, renderable lines describing one run: its method runs' outcomes and its
        reproducibility manifest. Used by the dashboard's drill-down."""
        run = self._store.experiment_run(experiment_run_id)
        if run is None:
            return (f"Experiment run {short_ref(experiment_run_id)} is not registered.",)

        rows = [
            f"Started {run.started_at}"
            + (f", completed {run.completed_at}" if run.completed_at else ", still running"),
            "End-to-end (each method segmented its own input)"
            if run.is_end_to_end
            else "Controlled (every method read the same input crops)",
        ]
        for method_run in self._store.method_runs(experiment_run_id=experiment_run_id):
            rows.append(
                f"{method_label(method_run.method_id)}: "
                f"{method_run_outcome_label(method_run.outcome)}"
            )
        for manifest in self._store.manifests(experiment_run_id=experiment_run_id):
            rows.append(
                f"Reproducibility manifest {short_ref(manifest.manifest_id)}"
                + (f" at commit {manifest.git_commit}" if manifest.git_commit else "")
            )
        return tuple(rows)
