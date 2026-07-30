"""The HTR research **projection**: the in-memory read model the research ViewModels query.

**Role changed 2026-07-30** (`docs/architecture/htr-telemetry.md`). This class was previously the
sole source of truth for every HTR research entity -- a bare `dict`-of-Pydantic-models with no disk
backing and no telemetry emission, disclosed as in-memory-by-design at the time it was written and
identified by `docs/htr-telemetry-knowledge-gap-analysis.md` §2 as the core persistence gap. It is
no longer a source of truth. The durable, append-only `FileTelemetrySink` event stream is, and this
class is the derived projection over it.

Concretely, it now has exactly four jobs:

1. **The query shape.** Its id-keyed lookups, listings, and derived traversals are unchanged --
   the gap analysis §10 is explicit that "this store's dict-of-Pydantic-models query interface is
   [not] wrong ... the gap is purely that nothing durable backs it". Every `presentation/htr_*`
   ViewModel keeps consuming exactly this interface, unmodified.
2. **The projection `HtrJournal.replay` builds.** `application/htr_journal.py::HtrJournal.replay`
   returns one of these, reconstructed from the telemetry log alone. There is deliberately no
   separate, parallel `HtrResearchState` class with a duplicate query surface: this *is* that state.
3. **The base class `htr/persistence/durable_store.py::DurableHtrResearchStore` extends.** That
   subclass overrides every `register_*` to append a telemetry event *before* updating this
   projection, and is what `composition.py::AppContext.htr_research_store` now hands out. Because
   it is an `HtrResearchStore`, no ViewModel constructor or type annotation needed to change.
4. **A test fixture, when constructed bare.** A bare `HtrResearchStore()` with no sink behind it is
   a projection with no durable backing -- which is precisely, and honestly, what a unit-test
   fixture wants. Existing tests that construct one directly stay valid and still mean what they
   say. What is no longer supported is *production* code treating a bare one as storage;
   `AppContext` no longer does.

**Append-only registration, projection-only advancement.** Every `register_*` method appends or
refuses (`DuplicateRegistrationError`); nothing registered is edited in place. Superseding entities
(`DatasetVersion.supersedes`, `ExperimentVersion.supersedes`, `CanonicalResult.supersedes`) are
stored alongside what they supersede, never replacing it -- the discipline Constitution Article 15
requires of the Canonical layer, so the evidence chain `docs/htr-domain-design.md` §4 describes stays
walkable backwards.

The two `_advance_*`/`_apply_*` methods at the end of the registration section are the one
exception, and only in this projection: a later event may carry a *further state of the same
entity* (an `ExperimentRun` that has since completed, a `MethodRunTranscript` gaining its parsed
stage). Advancing a projection entry is not rewriting history -- the durable log still holds every
event in order, and a fresh replay produces the same result -- and it is exactly what
`application/journal.py::Journal._apply` already does when it overwrites
`_alignment_state_by_observation[event.observation_id]`.
"""

from __future__ import annotations

import threading

from pydantic import BaseModel, ConfigDict

from archivetrust.domain.canonical.result import CanonicalResult
from archivetrust.evaluation.ground_truth import TranscriptionConvention
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


class UnknownEntityError(KeyError):
    """Raised when a projection advancement names an entity that was never registered -- e.g. an
    `ExperimentRunCompleted` event whose `ExperimentRunStarted` is missing from the stream. A
    distinct error from `DuplicateRegistrationError` so "the log is incomplete" is never silently
    turned into a first registration (the same Article-18 silence-vs-failure discipline the rest of
    this codebase applies)."""


_TRANSCRIPT_STAGE_FIELDS: dict[str, str] = {
    "raw": "raw_text",
    "parsed": "parsed_text",
    "normalized": "normalized_text",
    "reviewed": "reviewed_text",
}
"""`MethodRunTranscript`'s four stages, mapped to the field each fills. Named explicitly rather than
derived from the model so adding a field to `MethodRunTranscript` can never silently become a new
replayable stage without a matching telemetry event kind."""


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
        self._conventions: dict[tuple[str, int], TranscriptionConvention] = {}
        """`(convention_id, version) -> TranscriptionConvention`. Keyed by the *pair*, not the id
        alone, because `docs/htr-domain-design.md` §3 freezes a convention per version: a change
        creates a new version and existing ground truth keeps pointing at the old one, so both must
        remain resident and separately addressable."""
        self._ground_truth: dict[str, str] = {}
        """`text_line_id -> reference transcription`. Reference text for a line, from whatever
        authority produced it (a closed blind dual review, an imported gold standard). Kept as a
        plain mapping rather than a new entity: `GroundTruthItem` already exists in the design and
        `evaluation/ground_truth.py` owns its workflow -- this store only needs the resolved string
        to drive metric display, and inventing a competing entity here would duplicate that."""

    # -- Registration (append-only) -------------------------------------------------------------

    def _put(self, bucket: dict, key: object, value: object, kind: str) -> None:
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

    def register_convention(self, convention: TranscriptionConvention) -> None:
        self._put(
            self._conventions,
            (convention.convention_id, convention.version),
            convention,
            "TranscriptionConvention",
        )

    def register_ground_truth(self, *, text_line_id: str, text: str) -> None:
        self._put(self._ground_truth, text_line_id, text, "ground truth")

    # -- Projection advancement (see module docstring) -------------------------------------------

    def _advance(self, bucket: dict, key: str, value: object, kind: str) -> None:
        """Replaces a registered entity with a *later state of that same entity*, for the terminal
        markers that follow a start event (`ExperimentRunCompleted` after `ExperimentRunStarted`).

        Not an overwrite of history: the durable event log holds both events in append order, and a
        fresh `HtrJournal.replay` over it reproduces this same final state deterministically. Refuses
        a key that was never registered -- advancing an unknown entity means the log is missing the
        event that should have established it, which must surface, not be silently papered over into
        a registration.
        """
        with self._lock:
            if key not in bucket:
                raise UnknownEntityError(
                    f"cannot advance {kind} {key!r}: it was never registered -- the event that "
                    "should have established it is missing from the stream"
                )
            bucket[key] = value

    def advance_experiment_run(self, run: ExperimentRun) -> None:
        """Records an `ExperimentRun`'s terminal state (its `completed_at`) over the started one.

        There is deliberately no `advance_method_run` counterpart: a `MethodRun` is constructed once,
        already terminal (`htr/experiment/models.py`), so `MethodRunStarted` carries its final state
        and `MethodRunCompleted` is a marker with nothing further to advance.
        """
        self._advance(self._experiment_runs, run.experiment_run_id, run, "ExperimentRun")

    _PROJECTION_BUCKETS = (
        "_projects",
        "_datasets",
        "_dataset_versions",
        "_collections",
        "_pages",
        "_regions",
        "_text_lines",
        "_crops",
        "_experiments",
        "_experiment_versions",
        "_experiment_runs",
        "_method_runs",
        "_transcripts",
        "_failures",
        "_metric_definitions",
        "_metric_results",
        "_manifests",
        "_canonical_results",
        "_external_imports",
        "_conventions",
        "_ground_truth",
    )
    """Every bucket `adopt_projection` copies. Named explicitly rather than discovered by
    reflection, so adding a bucket without deciding how it hydrates is a visible omission here
    rather than a silently-empty bucket after a restart."""

    def adopt_projection(self, other: "HtrResearchStore") -> None:
        """Copies another projection's entire contents into this one, without going through
        `register_*`.

        This is how `htr/persistence/durable_store.py::DurableHtrResearchStore.open` hydrates itself
        from a replay. Bypassing `register_*` is the whole point and is not a shortcut: on a durable
        store those methods *emit telemetry*, so replaying through them would append a second copy of
        every historical event on every process start -- the log would double in size each launch.
        Hydration must therefore write to the projection only.
        """
        with self._lock:
            for name in self._PROJECTION_BUCKETS:
                getattr(self, name).update(getattr(other, name))

    def apply_transcript_stage(
        self,
        *,
        method_run_id: str,
        stage: str,
        text: str | None,
        reviewer_ref: str | None = None,
    ) -> None:
        """Merges one of `MethodRunTranscript`'s four stages into the projection, creating the
        record on the first stage seen and filling in the named field on each later one.

        This is replay's path into a transcript: the four stages arrive as four separate events
        (`RawMethodResultRecorded` -> `ParsedMethodResultRecorded` ->
        `NormalizedMethodResultRecorded` -> `ReviewedResultRecorded`), each causally chained to the
        previous, because `MethodRunTranscript`'s own docstring requires the four stages "never
        collapse into one another" -- and in particular that `reviewed_text` stay distinguishable
        from the machine stages, since it has a human rather than a method lineage. Direct callers
        that already hold a whole transcript keep using `register_transcript` instead.
        """
        if stage not in _TRANSCRIPT_STAGE_FIELDS:
            raise ValueError(
                f"unknown transcript stage {stage!r}; expected one of "
                f"{sorted(_TRANSCRIPT_STAGE_FIELDS)}"
            )
        field = _TRANSCRIPT_STAGE_FIELDS[stage]
        with self._lock:
            current = self._transcripts.get(method_run_id)
            base = current or MethodRunTranscript(method_run_id=method_run_id)
            update: dict[str, object] = {field: text}
            if reviewer_ref is not None:
                update["reviewer_ref"] = reviewer_ref
            self._transcripts[method_run_id] = base.model_copy(update=update)

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

    def conventions(self) -> tuple[TranscriptionConvention, ...]:
        with self._lock:
            return tuple(
                sorted(self._conventions.values(), key=lambda c: (c.convention_id, c.version))
            )

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

    def convention(self, convention_id: str, version: int) -> TranscriptionConvention | None:
        """The frozen convention for one `(convention_id, version)` pair -- never "the latest
        version of this convention", since ground truth authored under an older version must keep
        resolving to the rules that were actually in force (docs/htr-domain-design.md §3)."""
        with self._lock:
            return self._conventions.get((convention_id, version))

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
