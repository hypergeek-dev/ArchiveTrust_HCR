"""In-memory research store: the read source the Stage 11 research-interface ViewModels project.

**Why this file exists.** Stages 1-10 landed the HTR corpus/experiment entities
(`htr/corpus/models.py`, `htr/experiment/models.py`), `CanonicalResult`
(`domain/canonical/result.py`), and Transkribus `ExternalImport`
(`providers/transkribus/external_import.py`) as frozen Pydantic models -- but nothing that *holds*
them. A grep before writing this module found their only consumers were each other and
`research/reports/models.py`: there is no repository, no telemetry event, and no read-model
projection carrying a `Dataset` or an `ExperimentRun`. The Stage 11 ViewModels are required to read
real domain data rather than fabricate it, so the missing piece is this: one place those entities
live and are queried from.

**In-memory by design**, mirroring `review/blind_review/store.py::BlindReviewStore`'s own stated
rationale (which in turn mirrors `review/sampling/log.py`'s `InMemorySamplingLogSink`/
`FileSamplingLogSink` split): a durable, file-backed or telemetry-replayed variant can be added the
same way when a deployment needs one. Persisting HTR experiment entities into the append-only
telemetry stream is real work with real schema consequences (`docs/htr-domain-design.md` §2 lists
the HTR telemetry events it would need) and is deliberately *not* invented here as a side effect of
building a UI layer.

**Append-only, never mutate.** Every `register_*` method appends or refuses; nothing stored here is
edited in place. Superseding entities (`DatasetVersion.supersedes`, `ExperimentVersion.supersedes`,
`CanonicalResult.supersedes`) are stored alongside what they supersede, never replacing it -- the
same discipline Constitution Article 15 requires of the Canonical layer, so the evidence chain
`docs/htr-domain-design.md` §4 describes stays walkable backwards.
"""

from __future__ import annotations

import threading

from pydantic import BaseModel, ConfigDict

from archivetrust.domain.canonical.result import CanonicalResult
from archivetrust.htr.corpus.models import (
    Collection,
    Dataset,
    DatasetVersion,
    InputCrop,
    Page,
    Region,
    ResearchProject,
    TextLine,
)
from archivetrust.htr.experiment.models import (
    Experiment,
    ExperimentRun,
    ExperimentVersion,
    FailureRecord,
    MethodRun,
    MetricDefinition,
    MetricResult,
    ReproducibilityManifest,
)
from archivetrust.providers.transkribus.external_import import ExternalImport


class DuplicateRegistrationError(ValueError):
    """Raised when an id is registered twice. Registration is append-only: a second registration of
    the same id is a caller bug (two different entities minted the same id, or the same entity was
    registered twice), never a silent overwrite."""


class MethodRunTranscript(BaseModel):
    """The four *pre-canonical* text stages for one `MethodRun`, kept as four distinct fields that
    never collapse into one another.

    This exists because `docs/htr-domain-design.md` §1 names `RawResult`/`ParsedResult`/
    `NormalizedResult` as separate entities in the `MethodRun` subtree, and the comparison surface
    is explicitly required to distinguish them rather than show "the transcription". The
    Observation-layer counterparts are `RawTranscriptionPayload`/`ParsedTranscriptionPayload`/
    `NormalizedTranscriptionPayload` (`domain/ontology/payloads/transcription.py`); this record is
    the method-run-scoped index over them, so a UI does not have to replay telemetry per cell.

    `reviewed_text` is the fourth stage and belongs to a different lineage than the first three: it
    comes from a human `ReviewSubmission`/`Adjudication`, not from the method. It is `None` --
    never back-filled from `normalized_text` -- when no human has reviewed this run's line, so
    "nobody has reviewed this" and "a reviewer agreed with the machine" stay distinguishable.

    The fifth stage, canonical, is deliberately NOT here: it belongs to `CanonicalResult`, which is
    page-scoped and selects across competing method runs, so storing it per-method-run would
    misrepresent what canonicalization is.
    """

    model_config = ConfigDict(frozen=True)

    method_run_id: str
    raw_text: str | None = None
    parsed_text: str | None = None
    normalized_text: str | None = None
    reviewed_text: str | None = None
    reviewer_ref: str | None = None
    """Who produced `reviewed_text`, when one exists. `None` whenever `reviewed_text` is `None`."""


class HtrResearchStore:
    """Holds the registered HTR corpus, experiment, canonical, and external-import entities for one
    research deployment, and answers the id-keyed lookups the Stage 11 ViewModels need.

    Every accessor returns frozen Pydantic models or tuples of them -- no caller can mutate stored
    state through a returned value.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._projects: dict[str, ResearchProject] = {}
        self._datasets: dict[str, Dataset] = {}
        self._dataset_versions: dict[str, DatasetVersion] = {}
        self._collections: dict[str, Collection] = {}
        self._pages: dict[str, Page] = {}
        self._regions: dict[str, Region] = {}
        self._text_lines: dict[str, TextLine] = {}
        self._crops: dict[str, InputCrop] = {}
        self._experiments: dict[str, Experiment] = {}
        self._experiment_versions: dict[str, ExperimentVersion] = {}
        self._experiment_runs: dict[str, ExperimentRun] = {}
        self._method_runs: dict[str, MethodRun] = {}
        self._transcripts: dict[str, MethodRunTranscript] = {}
        self._failures: dict[str, FailureRecord] = {}
        self._metric_definitions: dict[str, MetricDefinition] = {}
        self._metric_results: dict[str, MetricResult] = {}
        self._manifests: dict[str, ReproducibilityManifest] = {}
        self._canonical_results: dict[str, CanonicalResult] = {}
        self._external_imports: dict[str, ExternalImport] = {}
        self._ground_truth: dict[str, str] = {}
        """`text_line_id -> reference transcription`. Reference text for a line, from whatever
        authority produced it (a closed blind dual review, an imported gold standard). Kept as a
        plain mapping rather than a new entity: `GroundTruthItem` already exists in the design and
        `evaluation/ground_truth.py` owns its workflow -- this store only needs the resolved string
        to drive metric display, and inventing a competing entity here would duplicate that."""

    # -- Registration (append-only) -------------------------------------------------------------

    def _put(self, bucket: dict, key: str, value: object, kind: str) -> None:
        with self._lock:
            if key in bucket:
                raise DuplicateRegistrationError(
                    f"{kind} {key!r} is already registered -- this store is append-only and never "
                    "overwrites a registered entity"
                )
            bucket[key] = value

    def register_project(self, project: ResearchProject) -> None:
        self._put(self._projects, project.project_id, project, "ResearchProject")

    def register_dataset(self, dataset: Dataset) -> None:
        self._put(self._datasets, dataset.dataset_id, dataset, "Dataset")

    def register_dataset_version(self, version: DatasetVersion) -> None:
        self._put(self._dataset_versions, version.dataset_version_id, version, "DatasetVersion")

    def register_collection(self, collection: Collection) -> None:
        self._put(self._collections, collection.collection_id, collection, "Collection")

    def register_page(self, page: Page) -> None:
        self._put(self._pages, page.page_id, page, "Page")

    def register_region(self, region: Region) -> None:
        self._put(self._regions, region.region_id, region, "Region")

    def register_text_line(self, line: TextLine) -> None:
        self._put(self._text_lines, line.text_line_id, line, "TextLine")

    def register_input_crop(self, crop: InputCrop) -> None:
        self._put(self._crops, crop.crop_id, crop, "InputCrop")

    def register_experiment(self, experiment: Experiment) -> None:
        self._put(self._experiments, experiment.experiment_id, experiment, "Experiment")

    def register_experiment_version(self, version: ExperimentVersion) -> None:
        self._put(
            self._experiment_versions,
            version.experiment_version_id,
            version,
            "ExperimentVersion",
        )

    def register_experiment_run(self, run: ExperimentRun) -> None:
        self._put(self._experiment_runs, run.experiment_run_id, run, "ExperimentRun")

    def register_method_run(self, run: MethodRun) -> None:
        self._put(self._method_runs, run.method_run_id, run, "MethodRun")

    def register_transcript(self, transcript: MethodRunTranscript) -> None:
        self._put(self._transcripts, transcript.method_run_id, transcript, "MethodRunTranscript")

    def register_failure(self, failure: FailureRecord) -> None:
        self._put(self._failures, failure.failure_record_id, failure, "FailureRecord")

    def register_metric_definition(self, definition: MetricDefinition) -> None:
        self._put(
            self._metric_definitions,
            definition.metric_definition_id,
            definition,
            "MetricDefinition",
        )

    def register_metric_result(self, result: MetricResult) -> None:
        self._put(self._metric_results, result.metric_result_id, result, "MetricResult")

    def register_manifest(self, manifest: ReproducibilityManifest) -> None:
        self._put(self._manifests, manifest.manifest_id, manifest, "ReproducibilityManifest")

    def register_canonical_result(self, result: CanonicalResult) -> None:
        self._put(
            self._canonical_results, result.canonical_result_id, result, "CanonicalResult"
        )

    def register_external_import(self, record: ExternalImport) -> None:
        self._put(
            self._external_imports, record.external_import_id, record, "ExternalImport"
        )

    def register_ground_truth(self, *, text_line_id: str, text: str) -> None:
        self._put(self._ground_truth, text_line_id, text, "ground truth")

    # -- Listing --------------------------------------------------------------------------------

    def projects(self) -> tuple[ResearchProject, ...]:
        with self._lock:
            return tuple(sorted(self._projects.values(), key=lambda p: (p.created_at, p.name)))

    def datasets(self, *, project_id: str | None = None) -> tuple[Dataset, ...]:
        with self._lock:
            rows = [
                d
                for d in self._datasets.values()
                if project_id is None or d.project_id == project_id
            ]
        return tuple(sorted(rows, key=lambda d: (d.created_at, d.name)))

    def dataset_versions(self, *, dataset_id: str | None = None) -> tuple[DatasetVersion, ...]:
        with self._lock:
            rows = [
                v
                for v in self._dataset_versions.values()
                if dataset_id is None or v.dataset_id == dataset_id
            ]
        return tuple(sorted(rows, key=lambda v: v.version))

    def collections(self, *, dataset_id: str | None = None) -> tuple[Collection, ...]:
        with self._lock:
            rows = [
                c
                for c in self._collections.values()
                if dataset_id is None or c.dataset_id == dataset_id
            ]
        return tuple(sorted(rows, key=lambda c: (c.created_at, c.name)))

    def pages(self, *, archive_object_ref: str | None = None) -> tuple[Page, ...]:
        with self._lock:
            rows = [
                p
                for p in self._pages.values()
                if archive_object_ref is None or p.archive_object_ref == archive_object_ref
            ]
        return tuple(sorted(rows, key=lambda p: (p.archive_object_ref, p.page_number)))

    def regions(self, *, page_id: str | None = None) -> tuple[Region, ...]:
        with self._lock:
            rows = [r for r in self._regions.values() if page_id is None or r.page_id == page_id]
        return tuple(sorted(rows, key=lambda r: (r.order_index if r.order_index is not None else 0)))

    def text_lines(self, *, region_id: str | None = None) -> tuple[TextLine, ...]:
        with self._lock:
            rows = [
                line
                for line in self._text_lines.values()
                if region_id is None or line.region_id == region_id
            ]
        return tuple(sorted(rows, key=lambda line: line.reading_order_index))

    def crops_for_line(self, text_line_id: str) -> tuple[InputCrop, ...]:
        with self._lock:
            return tuple(c for c in self._crops.values() if c.text_line_id == text_line_id)

    def experiments(self, *, research_project_id: str | None = None) -> tuple[Experiment, ...]:
        with self._lock:
            rows = [
                e
                for e in self._experiments.values()
                if research_project_id is None or e.research_project_id == research_project_id
            ]
        return tuple(sorted(rows, key=lambda e: (e.created_at, e.name)))

    def experiment_versions(self, *, experiment_id: str | None = None) -> tuple[ExperimentVersion, ...]:
        with self._lock:
            rows = [
                v
                for v in self._experiment_versions.values()
                if experiment_id is None or v.experiment_id == experiment_id
            ]
        return tuple(sorted(rows, key=lambda v: v.version))

    def experiment_runs(self, *, experiment_version_id: str | None = None) -> tuple[ExperimentRun, ...]:
        with self._lock:
            rows = [
                r
                for r in self._experiment_runs.values()
                if experiment_version_id is None
                or r.experiment_version_id == experiment_version_id
            ]
        return tuple(sorted(rows, key=lambda r: r.started_at))

    def method_runs(self, *, experiment_run_id: str | None = None) -> tuple[MethodRun, ...]:
        with self._lock:
            rows = [
                r
                for r in self._method_runs.values()
                if experiment_run_id is None or r.experiment_run_id == experiment_run_id
            ]
        return tuple(sorted(rows, key=lambda r: (r.started_at, r.method_id)))

    def failures(self, *, method_run_id: str | None = None) -> tuple[FailureRecord, ...]:
        with self._lock:
            rows = [
                f
                for f in self._failures.values()
                if method_run_id is None or f.method_run_id == method_run_id
            ]
        return tuple(sorted(rows, key=lambda f: f.failure_record_id))

    def metric_results(self, *, method_run_id: str | None = None) -> tuple[MetricResult, ...]:
        with self._lock:
            rows = [
                m
                for m in self._metric_results.values()
                if method_run_id is None or m.method_run_id == method_run_id
            ]
        return tuple(sorted(rows, key=lambda m: m.metric_result_id))

    def manifests(self, *, experiment_run_id: str | None = None) -> tuple[ReproducibilityManifest, ...]:
        with self._lock:
            rows = [
                m
                for m in self._manifests.values()
                if experiment_run_id is None or m.experiment_run_id == experiment_run_id
            ]
        return tuple(sorted(rows, key=lambda m: m.created_at))

    def canonical_results(self, *, page_id: str | None = None) -> tuple[CanonicalResult, ...]:
        with self._lock:
            rows = [
                c
                for c in self._canonical_results.values()
                if page_id is None or c.page_id == page_id
            ]
        return tuple(sorted(rows, key=lambda c: c.created_at))

    def external_imports(self) -> tuple[ExternalImport, ...]:
        with self._lock:
            return tuple(sorted(self._external_imports.values(), key=lambda e: e.imported_at))

    # -- Id-keyed lookup ------------------------------------------------------------------------

    def project(self, project_id: str) -> ResearchProject | None:
        with self._lock:
            return self._projects.get(project_id)

    def dataset(self, dataset_id: str) -> Dataset | None:
        with self._lock:
            return self._datasets.get(dataset_id)

    def dataset_version(self, dataset_version_id: str) -> DatasetVersion | None:
        with self._lock:
            return self._dataset_versions.get(dataset_version_id)

    def collection(self, collection_id: str) -> Collection | None:
        with self._lock:
            return self._collections.get(collection_id)

    def page(self, page_id: str) -> Page | None:
        with self._lock:
            return self._pages.get(page_id)

    def region(self, region_id: str) -> Region | None:
        with self._lock:
            return self._regions.get(region_id)

    def text_line(self, text_line_id: str) -> TextLine | None:
        with self._lock:
            return self._text_lines.get(text_line_id)

    def input_crop(self, crop_id: str) -> InputCrop | None:
        with self._lock:
            return self._crops.get(crop_id)

    def experiment(self, experiment_id: str) -> Experiment | None:
        with self._lock:
            return self._experiments.get(experiment_id)

    def experiment_version(self, experiment_version_id: str) -> ExperimentVersion | None:
        with self._lock:
            return self._experiment_versions.get(experiment_version_id)

    def experiment_run(self, experiment_run_id: str) -> ExperimentRun | None:
        with self._lock:
            return self._experiment_runs.get(experiment_run_id)

    def method_run(self, method_run_id: str) -> MethodRun | None:
        with self._lock:
            return self._method_runs.get(method_run_id)

    def transcript(self, method_run_id: str) -> MethodRunTranscript | None:
        with self._lock:
            return self._transcripts.get(method_run_id)

    def metric_definition(self, metric_definition_id: str) -> MetricDefinition | None:
        with self._lock:
            return self._metric_definitions.get(metric_definition_id)

    def ground_truth_for_line(self, text_line_id: str) -> str | None:
        with self._lock:
            return self._ground_truth.get(text_line_id)

    # -- Derived traversal ----------------------------------------------------------------------

    def method_runs_for_crop(self, crop_id: str) -> tuple[MethodRun, ...]:
        """Every method run that consumed one shared `InputCrop` -- the controlled-comparison row
        (`docs/htr-domain-design.md` §7: byte-identical crops shared across all local recognizers)."""
        with self._lock:
            rows = [r for r in self._method_runs.values() if r.input_crop_id == crop_id]
        return tuple(sorted(rows, key=lambda r: r.method_id))

    def latest_canonical_result(self, page_id: str) -> CanonicalResult | None:
        """The most recent non-superseded `CanonicalResult` for a page. Superseded results stay
        stored and readable -- this only picks which one is current."""
        results = self.canonical_results(page_id=page_id)
        if not results:
            return None
        superseded = {r.supersedes for r in results if r.supersedes is not None}
        current = [r for r in results if r.canonical_result_id not in superseded]
        return current[-1] if current else results[-1]

    def external_import_for_method_run(self, method_run_id: str) -> ExternalImport | None:
        with self._lock:
            for record in self._external_imports.values():
                if record.method_run_id == method_run_id:
                    return record
        return None
