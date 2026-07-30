"""`DurableHtrResearchStore`: telemetry-backed, append-only persistence for HTR research entities.

The core deliverable of the telemetry/provenance follow-up
(`docs/architecture/htr-telemetry.md`). Every mutation:

1. validates the registration (append-only: a duplicate id is refused, never silently overwritten),
2. constructs the matching `HtrTelemetryEvent`,
3. appends it to a `TelemetrySink` -- a `FileTelemetrySink` in any real deployment, reused entirely
   unmodified, hash-chain sidecar and all,
4. *then* updates the in-memory projection.

That order is `evaluation/ground_truth.py::FileGroundTruthStore.append`'s order, deliberately: the
durable record is written before the in-memory view of it exists, so a crash can leave an event the
projection has not yet seen (harmless -- the next replay picks it up) but never a projection entry
with no durable event behind it (which would vanish on restart, exactly the bug this whole pass
exists to remove).

**Why a subclass of `HtrResearchStore`.** The gap analysis §10 is explicit that the store's query
interface is not the problem -- "the gap is purely that nothing durable backs it". Subclassing keeps
that interface literally identical, so a `DurableHtrResearchStore` *is* an `HtrResearchStore` to
every `presentation/htr_*` ViewModel: not one constructor signature, type annotation, or ViewModel
test needed to change. The parent supplies the projection; this class supplies durability and the
causal metadata. See `htr/research_store.py`'s module docstring for the parent's four jobs.

**Two persistence patterns, one mechanism** (gap analysis §11's split, resolved here):

* *Event-sourced with replay* -- everything on the experiment-execution path. These are the records
  research conclusions rest on, and their history is itself evidence.
* *Whole-object JSON snapshot* -- additionally, and only as a **derived cache**, for the coarse
  registration entities (`ResearchProject`, `Dataset`, `DatasetVersion`, `Collection`,
  `TranscriptionConvention`), following `workspace/store.py::WorkspaceStore`'s idiom. Every one of
  those *also* emits its telemetry event and is fully replayable; the snapshot is a fast-path
  lookup, never a second source of truth, and `HtrCoarseEntitySnapshot.rebuild` regenerates it from
  the event log at any time. See `docs/architecture/htr-telemetry.md` §6 for why the split landed
  this way rather than making these entities JSON-only.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from archivetrust.domain.canonical.result import CanonicalResult
from archivetrust.domain.evidence.models import Evidence
from archivetrust.domain.shared.ids import new_id
from archivetrust.domain.telemetry.events import (
    HTR_RESEARCH_SCOPE,
    AdjudicationRecorded,
    AgreementCalculatedHtr,
    CandidateFindingCreated,
    CanonicalResultCreated,
    CollectionCreated,
    DatasetCreated,
    DatasetVersionCreated,
    DocumentRegistered,
    EvidenceCreated,
    ExperimentCreated,
    ExperimentRunCompleted,
    ExperimentRunFailed,
    ExperimentRunStarted,
    ExperimentVersionCreated,
    ExternalResultImported,
    FindingReviewed,
    FindingStatusChanged,
    GroundTruthTextRecorded,
    HtrActorType,
    HtrTelemetryEvent,
    InputCropCreated,
    MethodRunCompleted,
    MethodRunFailed,
    MethodRunStarted,
    MetricCalculated,
    MetricDefinitionRegistered,
    NormalizedMethodResultRecorded,
    PageRegistered,
    ParsedMethodResultRecorded,
    RawMethodResultRecorded,
    RegionDetected,
    ReliabilityIssueClassified,
    ReproducibilityManifestRecorded,
    ResearchObservationCreated,
    ResearchProjectCreated,
    ReviewAssigned,
    ReviewedResultRecorded,
    ReviewSubmissionRecorded,
    SegmentationRunCompleted,
    TelemetryEvent,
    TextLineDetected,
    TranscriptionConventionRegistered,
    stamp_recorded_at,
)
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
from archivetrust.htr.knowledge.lifecycle import transition_finding_status
from archivetrust.htr.knowledge.models import (
    INITIAL_FINDING_STATUSES,
    ContradictoryEvidence,
    EvidenceReference,
    FindingStatus,
    ResearchFinding,
    ResearchObservation,
)
from archivetrust.htr.research_store import (
    DuplicateRegistrationError,
    HtrResearchStore,
    MethodRunTranscript,
)

_SOURCE_COMPONENT = "htr.persistence.durable_store"


class DurableHtrResearchStore(HtrResearchStore):
    """An `HtrResearchStore` whose every registration is durably recorded as telemetry first.

    `sink` is any object with `append(event)` and `all_events()` -- `FileTelemetrySink` for real
    durability, `InMemoryTelemetrySink` where a caller genuinely wants event sourcing without a file
    (a unit test, or a deployment that persists at a higher layer). The distinction is the caller's
    to make and is not hidden: `is_durable` reports it.
    """

    def __init__(
        self,
        sink: Any,
        *,
        actor_id: str | None = None,
        application_commit: str | None = None,
        snapshot: "HtrCoarseEntitySnapshot | None" = None,
    ) -> None:
        super().__init__()
        self._sink = sink
        self._actor_id = actor_id
        self._application_commit = application_commit
        self._snapshot = snapshot
        self._emit_lock = threading.Lock()
        """Serializes validate -> append -> project, so two threads registering different entities
        can never interleave into an append order that disagrees with projection order. The parent's
        own `_lock` guards each individual bucket write; this one guards the whole triple."""
        self._correlation_id: str | None = None

    # -- Introspection ---------------------------------------------------------------------------

    @property
    def sink(self) -> Any:
        """The telemetry sink every registration is recorded to. Exposed so a caller can replay it
        (`HtrJournal().replay(store.sink.all_events())`) or verify its hash chain."""
        return self._sink

    @property
    def is_durable(self) -> bool:
        """Whether this store's events actually survive process exit. `False` for an in-memory sink
        -- reported rather than assumed, so no caller mistakes an in-memory event log for a durable
        one (the confusion `docs/htr-telemetry-knowledge-gap-analysis.md` §0 traces real data loss
        to)."""
        return hasattr(self._sink, "_path")

    # -- Correlation and causation (docs/architecture/htr-event-model.md §4) ---------------------

    @contextmanager
    def correlated_to(self, correlation_id: str) -> Iterator[str]:
        """Scopes every event emitted inside the block to one logical unit of work.

        The `correlation_id` for an experiment's execution is that `ExperimentRun`'s own
        `experiment_run_id` (this pass's choice, per event-model doc §4's "one correlation id per
        `ExperimentRun`"): it is already unique, already stable, already meaningful to a researcher
        reading the log, and needs no second identifier minted alongside it.

        Nests and restores, so an inner scope cannot leak into the outer one.
        """
        previous = self._correlation_id
        self._correlation_id = correlation_id
        try:
            yield correlation_id
        finally:
            self._correlation_id = previous

    @property
    def correlation_id(self) -> str | None:
        """The correlation id currently in scope, or `None` outside any unit of work -- an honest
        `None` rather than a synthesized per-call value, since a bare `ResearchProject` registration
        genuinely belongs to no run."""
        return self._correlation_id

    # -- The emit-then-project core ---------------------------------------------------------------

    def _emit(
        self,
        event: HtrTelemetryEvent,
        *,
        bucket: dict,
        key: object,
        kind: str,
        project,
    ) -> str:
        """Validate, append, project -- in that order. Returns the emitted event's `event_id`, which
        a caller passes as the next registration's `caused_by` to build the causal chain."""
        with self._emit_lock:
            with self._lock:
                if key in bucket:
                    raise DuplicateRegistrationError(
                        f"{kind} {key!r} is already registered -- this store is append-only and "
                        "never overwrites a registered entity"
                    )
            self._sink.append(stamp_recorded_at(event))
            project()
        return event.event_id

    def _emit_only(self, event: TelemetryEvent) -> str:
        """Appends an event that establishes no new projection entry of its own (a terminal marker, a
        summary, a review record whose projection is owned elsewhere).

        Typed against the shared `TelemetryEvent` base rather than `HtrTelemetryEvent` because
        `record_evidence` legitimately emits the retained substrate's own `EvidenceCreated` onto this
        same stream -- see that method's docstring for why that is not a second event vocabulary.
        """
        with self._emit_lock:
            self._sink.append(stamp_recorded_at(event))
        return event.event_id

    def _scope(
        self,
        *,
        causation_id: str | None,
        correlation_id: str | None,
        actor_type: HtrActorType = HtrActorType.SYSTEM,
        actor_id: str | None = None,
        document_ref: str = HTR_RESEARCH_SCOPE,
        **scope: Any,
    ) -> dict[str, Any]:
        """The field block every emitted event shares."""
        return {
            "event_id": new_id("event"),
            "document_ref": document_ref,
            "correlation_id": correlation_id if correlation_id is not None else self._correlation_id,
            "causation_id": causation_id,
            "actor_type": actor_type,
            "actor_id": actor_id if actor_id is not None else self._actor_id,
            "source_component": _SOURCE_COMPONENT,
            "application_commit": self._application_commit,
            **scope,
        }

    # -- Coarse registration entities -------------------------------------------------------------

    def register_project(  # type: ignore[override]
        self,
        project: ResearchProject,
        *,
        caused_by: str | None = None,
        correlation_id: str | None = None,
    ) -> str:
        event_id = self._emit(
            ResearchProjectCreated(
                **self._scope(
                    causation_id=caused_by,
                    correlation_id=correlation_id,
                    project_id=project.project_id,
                    subject_id=project.project_id,
                ),
                project=project,
            ),
            bucket=self._projects,
            key=project.project_id,
            kind="ResearchProject",
            project=lambda: HtrResearchStore.register_project(self, project),
        )
        self._snapshot_put("projects", project.project_id, project)
        return event_id

    def register_dataset(  # type: ignore[override]
        self,
        dataset: Dataset,
        *,
        caused_by: str | None = None,
        correlation_id: str | None = None,
    ) -> str:
        event_id = self._emit(
            DatasetCreated(
                **self._scope(
                    causation_id=caused_by,
                    correlation_id=correlation_id,
                    project_id=dataset.project_id,
                    dataset_id=dataset.dataset_id,
                    subject_id=dataset.dataset_id,
                ),
                dataset=dataset,
            ),
            bucket=self._datasets,
            key=dataset.dataset_id,
            kind="Dataset",
            project=lambda: HtrResearchStore.register_dataset(self, dataset),
        )
        self._snapshot_put("datasets", dataset.dataset_id, dataset)
        return event_id

    def register_dataset_version(  # type: ignore[override]
        self,
        version: DatasetVersion,
        *,
        caused_by: str | None = None,
        correlation_id: str | None = None,
    ) -> str:
        event_id = self._emit(
            DatasetVersionCreated(
                **self._scope(
                    causation_id=caused_by,
                    correlation_id=correlation_id,
                    dataset_id=version.dataset_id,
                    dataset_version_id=version.dataset_version_id,
                    subject_id=version.dataset_version_id,
                ),
                dataset_version=version,
            ),
            bucket=self._dataset_versions,
            key=version.dataset_version_id,
            kind="DatasetVersion",
            project=lambda: HtrResearchStore.register_dataset_version(self, version),
        )
        self._snapshot_put("dataset_versions", version.dataset_version_id, version)
        return event_id

    def register_collection(  # type: ignore[override]
        self,
        collection: Collection,
        *,
        caused_by: str | None = None,
        correlation_id: str | None = None,
    ) -> str:
        event_id = self._emit(
            CollectionCreated(
                **self._scope(
                    causation_id=caused_by,
                    correlation_id=correlation_id,
                    dataset_id=collection.dataset_id,
                    subject_id=collection.collection_id,
                ),
                collection=collection,
            ),
            bucket=self._collections,
            key=collection.collection_id,
            kind="Collection",
            project=lambda: HtrResearchStore.register_collection(self, collection),
        )
        self._snapshot_put("collections", collection.collection_id, collection)
        # One DocumentRegistered per member, caused by the collection that names it. Records the
        # membership fact only: no Document *model* exists (htr/corpus/models.py's docstring --
        # "ArchiveObject *is* the document concept at this layer").
        for archive_object_ref in collection.archive_object_refs:
            self._emit_only(
                DocumentRegistered(
                    **self._scope(
                        causation_id=event_id,
                        correlation_id=correlation_id,
                        document_ref=archive_object_ref,
                        dataset_id=collection.dataset_id,
                        subject_id=archive_object_ref,
                    ),
                    archive_object_ref=archive_object_ref,
                    collection_id=collection.collection_id,
                )
            )
        return event_id

    def register_convention(  # type: ignore[override]
        self,
        convention: TranscriptionConvention,
        *,
        caused_by: str | None = None,
        correlation_id: str | None = None,
    ) -> str:
        key = (convention.convention_id, convention.version)
        event_id = self._emit(
            TranscriptionConventionRegistered(
                **self._scope(
                    causation_id=caused_by,
                    correlation_id=correlation_id,
                    subject_id=convention.convention_id,
                ),
                convention_id=convention.convention_id,
                convention_version=convention.version,
                record=convention.model_dump(mode="json", by_alias=True),
            ),
            bucket=self._conventions,
            key=key,
            kind="TranscriptionConvention",
            project=lambda: HtrResearchStore.register_convention(self, convention),
        )
        self._snapshot_put(
            "conventions", f"{convention.convention_id}@{convention.version}", convention
        )
        return event_id

    # -- Corpus / segmentation --------------------------------------------------------------------

    def register_page(  # type: ignore[override]
        self,
        page: Page,
        *,
        caused_by: str | None = None,
        correlation_id: str | None = None,
    ) -> str:
        return self._emit(
            PageRegistered(
                **self._scope(
                    causation_id=caused_by,
                    correlation_id=correlation_id,
                    # The one HTR kind with a genuine Archive Object scope.
                    document_ref=page.archive_object_ref,
                    subject_id=page.page_id,
                ),
                page=page,
            ),
            bucket=self._pages,
            key=page.page_id,
            kind="Page",
            project=lambda: HtrResearchStore.register_page(self, page),
        )

    def register_region(  # type: ignore[override]
        self,
        region: Region,
        *,
        caused_by: str | None = None,
        correlation_id: str | None = None,
    ) -> str:
        return self._emit(
            RegionDetected(
                **self._scope(
                    causation_id=caused_by,
                    correlation_id=correlation_id,
                    subject_id=region.region_id,
                ),
                region=region,
            ),
            bucket=self._regions,
            key=region.region_id,
            kind="Region",
            project=lambda: HtrResearchStore.register_region(self, region),
        )

    def register_text_line(  # type: ignore[override]
        self,
        line: TextLine,
        *,
        caused_by: str | None = None,
        correlation_id: str | None = None,
    ) -> str:
        return self._emit(
            TextLineDetected(
                **self._scope(
                    causation_id=caused_by,
                    correlation_id=correlation_id,
                    subject_id=line.text_line_id,
                ),
                text_line=line,
            ),
            bucket=self._text_lines,
            key=line.text_line_id,
            kind="TextLine",
            project=lambda: HtrResearchStore.register_text_line(self, line),
        )

    def register_input_crop(  # type: ignore[override]
        self,
        crop: InputCrop,
        *,
        caused_by: str | None = None,
        correlation_id: str | None = None,
    ) -> str:
        """Records a content-addressed `InputCrop`.

        The hash is `InputCrop`'s own (`InputCrop.compute_hash`, validated by its `_validate_hash`)
        and is neither recomputed nor reinterpreted here -- the brief's "preserved exactly as
        `htr/corpus/models.py` already implements it", and event-model doc §2's "no second hashing
        scheme introduced".
        """
        return self._emit(
            InputCropCreated(
                **self._scope(
                    causation_id=caused_by,
                    correlation_id=correlation_id,
                    subject_id=crop.crop_id,
                ),
                input_crop=crop,
            ),
            bucket=self._crops,
            key=crop.crop_id,
            kind="InputCrop",
            project=lambda: HtrResearchStore.register_input_crop(self, crop),
        )

    def record_segmentation_run(
        self,
        *,
        page_id: str,
        segmentation_adapter_name: str,
        region_ids: tuple[str, ...],
        text_line_ids: tuple[str, ...],
        input_crop_ids: tuple[str, ...],
        caused_by: str | None = None,
        correlation_id: str | None = None,
        segmentation_run_id: str | None = None,
    ) -> str:
        """Records that one segmentation adapter run over a Page completed.

        Gives `SEGMENTATION_RUN_COMPLETED` -- defined by the prior transformation with zero producers
        anywhere in `src/` (gap analysis §1) -- its first real producer. The regions/lines/crops it
        names are the ones already registered individually above; this is the summary that ties them
        to one adapter invocation.
        """
        return self._emit_only(
            SegmentationRunCompleted(
                **self._scope(
                    causation_id=caused_by,
                    correlation_id=correlation_id,
                    actor_type=HtrActorType.METHOD,
                    actor_id=segmentation_adapter_name,
                    subject_id=page_id,
                ),
                segmentation_run_id=segmentation_run_id or new_id("segmentation_run"),
                page_id=page_id,
                segmentation_adapter_name=segmentation_adapter_name,
                region_ids=region_ids,
                text_line_ids=text_line_ids,
                input_crop_ids=input_crop_ids,
            )
        )

    # -- Experiment execution ---------------------------------------------------------------------

    def register_experiment(  # type: ignore[override]
        self,
        experiment: Experiment,
        *,
        caused_by: str | None = None,
        correlation_id: str | None = None,
    ) -> str:
        return self._emit(
            ExperimentCreated(
                **self._scope(
                    causation_id=caused_by,
                    correlation_id=correlation_id,
                    project_id=experiment.research_project_id,
                    experiment_id=experiment.experiment_id,
                    subject_id=experiment.experiment_id,
                ),
                experiment=experiment,
            ),
            bucket=self._experiments,
            key=experiment.experiment_id,
            kind="Experiment",
            project=lambda: HtrResearchStore.register_experiment(self, experiment),
        )

    def register_experiment_version(  # type: ignore[override]
        self,
        version: ExperimentVersion,
        *,
        caused_by: str | None = None,
        correlation_id: str | None = None,
    ) -> str:
        return self._emit(
            ExperimentVersionCreated(
                **self._scope(
                    causation_id=caused_by,
                    correlation_id=correlation_id,
                    experiment_id=version.experiment_id,
                    experiment_version_id=version.experiment_version_id,
                    dataset_version_id=version.dataset_version_id,
                    subject_id=version.experiment_version_id,
                ),
                experiment_version=version,
            ),
            bucket=self._experiment_versions,
            key=version.experiment_version_id,
            kind="ExperimentVersion",
            project=lambda: HtrResearchStore.register_experiment_version(self, version),
        )

    def register_experiment_run(  # type: ignore[override]
        self,
        run: ExperimentRun,
        *,
        caused_by: str | None = None,
        correlation_id: str | None = None,
    ) -> str:
        """Records the start of an `ExperimentRun`.

        Defaults `correlation_id` to the run's own `experiment_run_id` when no scope is active: this
        event is the causal root of the run, so it belongs to that run's correlation by definition,
        not by a caller remembering to say so.
        """
        return self._emit(
            ExperimentRunStarted(
                **self._scope(
                    causation_id=caused_by,
                    correlation_id=correlation_id
                    or self._correlation_id
                    or run.experiment_run_id,
                    experiment_version_id=run.experiment_version_id,
                    experiment_run_id=run.experiment_run_id,
                    subject_id=run.experiment_run_id,
                ),
                experiment_run=run,
            ),
            bucket=self._experiment_runs,
            key=run.experiment_run_id,
            kind="ExperimentRun",
            project=lambda: HtrResearchStore.register_experiment_run(self, run),
        )

    def complete_experiment_run(
        self,
        run: ExperimentRun,
        *,
        caused_by: str | None = None,
        correlation_id: str | None = None,
    ) -> str:
        """Records an `ExperimentRun`'s terminal success, carrying the run with `completed_at` set."""
        event_id = self._emit_only(
            ExperimentRunCompleted(
                **self._scope(
                    causation_id=caused_by,
                    correlation_id=correlation_id
                    or self._correlation_id
                    or run.experiment_run_id,
                    experiment_version_id=run.experiment_version_id,
                    experiment_run_id=run.experiment_run_id,
                    subject_id=run.experiment_run_id,
                ),
                experiment_run=run,
            )
        )
        self.advance_experiment_run(run)
        return event_id

    def fail_experiment_run(
        self,
        *,
        experiment_run_id: str,
        reason: str,
        caused_by: str | None = None,
        correlation_id: str | None = None,
    ) -> str:
        """Records that an `ExperimentRun` could not complete -- an explicit fact, never inferred
        from a missing completion (Constitution Article 18)."""
        return self._emit_only(
            ExperimentRunFailed(
                **self._scope(
                    causation_id=caused_by,
                    correlation_id=correlation_id or self._correlation_id or experiment_run_id,
                    experiment_run_id=experiment_run_id,
                    subject_id=experiment_run_id,
                ),
                reason=reason,
            )
        )

    def register_method_run(  # type: ignore[override]
        self,
        run: MethodRun,
        *,
        caused_by: str | None = None,
        correlation_id: str | None = None,
    ) -> str:
        return self._emit(
            MethodRunStarted(
                **self._scope(
                    causation_id=caused_by,
                    correlation_id=correlation_id,
                    actor_type=HtrActorType.METHOD,
                    actor_id=run.method_id,
                    experiment_run_id=run.experiment_run_id,
                    method_run_id=run.method_run_id,
                    subject_id=run.input_crop_id or run.method_run_id,
                ),
                method_run=run,
            ),
            bucket=self._method_runs,
            key=run.method_run_id,
            kind="MethodRun",
            project=lambda: HtrResearchStore.register_method_run(self, run),
        )

    def complete_method_run(
        self,
        run: MethodRun,
        *,
        caused_by: str | None = None,
        correlation_id: str | None = None,
        failure_reason: str | None = None,
    ) -> str:
        """Records a `MethodRun`'s terminal outcome.

        Gives `METHOD_RUN_COMPLETED` its first real producer, and is the second half of the pair
        event-model doc §4 uses as its worked causation example: pass the `register_method_run`
        event id as `caused_by` and the log carries an explicit `MethodRunStarted ->
        MethodRunCompleted` edge.
        """
        return self._emit_only(
            MethodRunCompleted(
                **self._scope(
                    causation_id=caused_by,
                    correlation_id=correlation_id,
                    actor_type=HtrActorType.METHOD,
                    actor_id=run.method_id,
                    experiment_run_id=run.experiment_run_id,
                    method_run_id=run.method_run_id,
                    subject_id=run.method_run_id,
                ),
                # `method_run_id`/`experiment_run_id` are the base scope fields this kind narrows to
                # required, so they arrive via `_scope` above rather than being repeated here.
                method_id=run.method_id,
                evidence_id=run.evidence_id,
                outcome=run.outcome,
                failure_reason=failure_reason,
            )
        )

    def register_failure(  # type: ignore[override]
        self,
        failure: FailureRecord,
        *,
        caused_by: str | None = None,
        correlation_id: str | None = None,
    ) -> str:
        return self._emit(
            MethodRunFailed(
                **self._scope(
                    causation_id=caused_by,
                    correlation_id=correlation_id,
                    method_run_id=failure.method_run_id,
                    subject_id=failure.method_run_id,
                ),
                failure_record=failure,
            ),
            bucket=self._failures,
            key=failure.failure_record_id,
            kind="FailureRecord",
            project=lambda: HtrResearchStore.register_failure(self, failure),
        )

    # -- Provenance evidence (the retained substrate's own record, on this stream) -----------------

    def record_evidence(
        self,
        evidence: Evidence,
        *,
        document_ref: str,
        invocation_id: str,
        caused_by: str | None = None,
        correlation_id: str | None = None,
    ) -> str:
        """Durably records a real captured `Evidence` -- the retained substrate's own
        `EvidenceCreated`, appended to this store's stream.

        **Why this exists.** `docs/htr-telemetry-knowledge-gap-analysis.md` §4 point 3 names a
        specific, separate gap from the one the rest of this class closes: the baseline experiment's
        `Evidence` records "are also never appended to a `TelemetrySink`, so in practice they only
        ever lived in one Python process's memory". `HtrResearchStore` deliberately has no `Evidence`
        bucket (`docs/htr-domain-design.md` §2: Evidence belongs to the Evidence/Observation
        substrate, not this research-facing store), so this method emits *only* -- it establishes no
        projection entry here, and `HtrJournal._apply` skips the event as a pre-existing kind that is
        `application/journal.py::Journal`'s business, exactly as its closing comment says.

        **Why the pre-existing kind rather than a new HTR one.** Gap analysis §10 forbids inventing a
        second, incompatible mechanism, and `EVIDENCE_CREATED` already exists, already carries the
        full `Evidence` object, and already has a `Journal._apply` branch that reconstructs it. Adding
        an HTR-specific near-duplicate would have produced two kinds meaning one thing. Consequently
        this stream is a legitimately mixed one, which `HtrJournal.replay` documents as supported
        ("a caller may legitimately hand this the whole of a mixed stream") and `Journal._apply`
        likewise tolerates in the other direction via its `HtrTelemetryEvent` no-op branch.

        `document_ref` is the genuine `archive_object_ref` the evidence concerns, not
        `HTR_RESEARCH_SCOPE`: an `Evidence` record really does belong to exactly one Archive Object,
        so per-document replay (`FileTelemetrySink.events_for_document`) keeps working for it.
        """
        return self._emit_only(
            EvidenceCreated(
                event_id=new_id("event"),
                document_ref=document_ref,
                correlation_id=(
                    correlation_id if correlation_id is not None else self._correlation_id
                ),
                causation_id=caused_by,
                invocation_id=invocation_id,
                evidence=evidence,
            )
        )

    # -- Method-result stages ----------------------------------------------------------------------

    def register_transcript(  # type: ignore[override]
        self,
        transcript: MethodRunTranscript,
        *,
        caused_by: str | None = None,
        correlation_id: str | None = None,
        evidence_id: str | None = None,
    ) -> str:
        """Records a `MethodRunTranscript` as its constituent stage events, causally chained.

        Emits one event per *populated* stage -- `RawMethodResultRecorded` ->
        `ParsedMethodResultRecorded` -> `NormalizedMethodResultRecorded` -> `ReviewedResultRecorded`
        -- each caused by the previous, because `MethodRunTranscript`'s own docstring requires the
        four stages "never collapse into one another". An unpopulated stage emits nothing: no event
        is the honest record of "this stage never happened", and in particular a missing
        `ReviewedResultRecorded` keeps "nobody has reviewed this" distinguishable from "a reviewer
        agreed with the machine", which that docstring explicitly requires.

        Returns the *last* emitted event id, so a caller can keep chaining (a `MetricCalculated`
        naturally causes from the stage whose text it scored).
        """
        run = self.method_run(transcript.method_run_id)
        scope: dict[str, Any] = {
            "method_run_id": transcript.method_run_id,
            "subject_id": transcript.method_run_id,
        }
        if run is not None:
            scope["experiment_run_id"] = run.experiment_run_id

        with self._emit_lock:
            with self._lock:
                if transcript.method_run_id in self._transcripts:
                    raise DuplicateRegistrationError(
                        f"MethodRunTranscript {transcript.method_run_id!r} is already registered -- "
                        "this store is append-only and never overwrites a registered entity"
                    )
            previous = caused_by
            emitted: list[str] = []
            for event in self._transcript_stage_events(
                transcript,
                scope=scope,
                correlation_id=correlation_id,
                evidence_id=evidence_id,
                first_causation=caused_by,
            ):
                # Re-stamp causation so each stage chains to the one actually emitted before it.
                chained = event.model_copy(update={"causation_id": previous})
                self._sink.append(stamp_recorded_at(chained))
                previous = chained.event_id
                emitted.append(chained.event_id)
            HtrResearchStore.register_transcript(self, transcript)
        return emitted[-1] if emitted else (caused_by or "")

    def _transcript_stage_events(
        self,
        transcript: MethodRunTranscript,
        *,
        scope: dict[str, Any],
        correlation_id: str | None,
        evidence_id: str | None,
        first_causation: str | None,
    ) -> list[HtrTelemetryEvent]:
        # `method_run_id` arrives via `scope` on every stage: each of these kinds narrows the base
        # scope field to required rather than declaring a second one of its own.
        events: list[HtrTelemetryEvent] = []
        if transcript.raw_text is not None:
            events.append(
                RawMethodResultRecorded(
                    **self._scope(
                        causation_id=first_causation,
                        correlation_id=correlation_id,
                        actor_type=HtrActorType.METHOD,
                        **scope,
                    ),
                    text=transcript.raw_text,
                    evidence_id=evidence_id,
                )
            )
        if transcript.parsed_text is not None:
            events.append(
                ParsedMethodResultRecorded(
                    **self._scope(
                        causation_id=None,
                        correlation_id=correlation_id,
                        actor_type=HtrActorType.METHOD,
                        **scope,
                    ),
                    text=transcript.parsed_text,
                )
            )
        if transcript.normalized_text is not None:
            events.append(
                NormalizedMethodResultRecorded(
                    **self._scope(
                        causation_id=None,
                        correlation_id=correlation_id,
                        actor_type=HtrActorType.METHOD,
                        **scope,
                    ),
                    text=transcript.normalized_text,
                )
            )
        if transcript.reviewed_text is not None:
            events.append(
                ReviewedResultRecorded(
                    **self._scope(
                        causation_id=None,
                        correlation_id=correlation_id,
                        actor_type=HtrActorType.HUMAN,
                        actor_id=transcript.reviewer_ref,
                        **scope,
                    ),
                    text=transcript.reviewed_text,
                    reviewer_ref=transcript.reviewer_ref,
                )
            )
        return events

    # -- Evaluation --------------------------------------------------------------------------------

    def register_metric_definition(  # type: ignore[override]
        self,
        definition: MetricDefinition,
        *,
        caused_by: str | None = None,
        correlation_id: str | None = None,
    ) -> str:
        return self._emit(
            MetricDefinitionRegistered(
                **self._scope(
                    causation_id=caused_by,
                    correlation_id=correlation_id,
                    subject_id=definition.metric_definition_id,
                ),
                metric_definition=definition,
            ),
            bucket=self._metric_definitions,
            key=definition.metric_definition_id,
            kind="MetricDefinition",
            project=lambda: HtrResearchStore.register_metric_definition(self, definition),
        )

    def register_metric_result(  # type: ignore[override]
        self,
        result: MetricResult,
        *,
        caused_by: str | None = None,
        correlation_id: str | None = None,
    ) -> str:
        """Records one computed metric. The computation itself stays a pure function in
        `htr/evaluation/*` (gap analysis §7); this is the caller persisting its return value, and
        `caused_by` should be the transcript-stage event whose text was scored."""
        run = self.method_run(result.method_run_id)
        extra: dict[str, Any] = {}
        if run is not None:
            extra["experiment_run_id"] = run.experiment_run_id
        return self._emit(
            MetricCalculated(
                **self._scope(
                    causation_id=caused_by,
                    correlation_id=correlation_id,
                    method_run_id=result.method_run_id,
                    subject_id=result.metric_result_id,
                    **extra,
                ),
                metric_result=result,
            ),
            bucket=self._metric_results,
            key=result.metric_result_id,
            kind="MetricResult",
            project=lambda: HtrResearchStore.register_metric_result(self, result),
        )

    def record_reliability_issue(
        self,
        *,
        method_run_id: str,
        classification: str,
        detail: str | None = None,
        caused_by: str | None = None,
        correlation_id: str | None = None,
    ) -> str:
        """Persists `htr/evaluation/failures.py::classify_reliability`'s verdict for one run."""
        return self._emit_only(
            ReliabilityIssueClassified(
                **self._scope(
                    causation_id=caused_by,
                    correlation_id=correlation_id,
                    method_run_id=method_run_id,
                    subject_id=method_run_id,
                ),
                classification=classification,
                detail=detail,
            )
        )

    def register_ground_truth(  # type: ignore[override]
        self,
        *,
        text_line_id: str,
        text: str,
        caused_by: str | None = None,
        correlation_id: str | None = None,
    ) -> str:
        return self._emit(
            GroundTruthTextRecorded(
                **self._scope(
                    causation_id=caused_by,
                    correlation_id=correlation_id,
                    subject_id=text_line_id,
                ),
                text_line_id=text_line_id,
                text=text,
            ),
            bucket=self._ground_truth,
            key=text_line_id,
            kind="ground truth",
            project=lambda: HtrResearchStore.register_ground_truth(
                self, text_line_id=text_line_id, text=text
            ),
        )

    # -- Reproducibility, canonicalization, external import ----------------------------------------

    def register_manifest(  # type: ignore[override]
        self,
        manifest: ReproducibilityManifest,
        *,
        caused_by: str | None = None,
        correlation_id: str | None = None,
    ) -> str:
        return self._emit(
            ReproducibilityManifestRecorded(
                **self._scope(
                    causation_id=caused_by,
                    correlation_id=correlation_id,
                    experiment_run_id=manifest.experiment_run_id,
                    subject_id=manifest.manifest_id,
                ),
                manifest=manifest,
            ),
            bucket=self._manifests,
            key=manifest.manifest_id,
            kind="ReproducibilityManifest",
            project=lambda: HtrResearchStore.register_manifest(self, manifest),
        )

    def register_canonical_result(  # type: ignore[override]
        self,
        result: CanonicalResult,
        *,
        caused_by: str | None = None,
        correlation_id: str | None = None,
    ) -> str:
        return self._emit(
            CanonicalResultCreated(
                **self._scope(
                    causation_id=caused_by,
                    correlation_id=correlation_id,
                    subject_id=result.page_id,
                ),
                canonical_result_id=result.canonical_result_id,
                page_id=result.page_id,
                strategy=result.strategy.value,
                strategy_version=result.strategy_version,
                canonical_result=result,
            ),
            bucket=self._canonical_results,
            key=result.canonical_result_id,
            kind="CanonicalResult",
            project=lambda: HtrResearchStore.register_canonical_result(self, result),
        )

    def register_external_import(  # type: ignore[override]
        self,
        record: Any,
        *,
        caused_by: str | None = None,
        correlation_id: str | None = None,
    ) -> str:
        return self._emit(
            ExternalResultImported(
                **self._scope(
                    causation_id=caused_by,
                    correlation_id=correlation_id,
                    method_run_id=record.method_run_id,
                    subject_id=record.external_import_id,
                ),
                external_import_id=record.external_import_id,
                record=record.model_dump(mode="json"),
            ),
            bucket=self._external_imports,
            key=record.external_import_id,
            kind="ExternalImport",
            project=lambda: HtrResearchStore.register_external_import(self, record),
        )

    # -- Blind-review evidence (events durable now, projection deferred) ---------------------------
    #
    # These four give `REVIEW_SUBMISSION_RECORDED` and `ADJUDICATION_RECORDED` -- two of the five
    # producerless kinds the prior transformation left behind -- their first real producers, and add
    # the two new review kinds alongside them. `review/blind_review/store.py::BlindReviewStore`
    # keeps owning the blind-isolation *workflow* and its query surface; only the durable *record*
    # moves here this pass. See `docs/architecture/htr-telemetry.md` §7 for that scope decision.

    def record_review_assignment(
        self,
        assignment: Any,
        *,
        caused_by: str | None = None,
        correlation_id: str | None = None,
    ) -> str:
        return self._emit_only(
            ReviewAssigned(
                **self._scope(
                    causation_id=caused_by,
                    correlation_id=correlation_id,
                    actor_type=HtrActorType.HUMAN,
                    actor_id=assignment.reviewer_ref,
                    subject_id=assignment.target_ref,
                ),
                assignment_id=assignment.assignment_id,
                target_ref=assignment.target_ref,
                record=assignment.model_dump(mode="json"),
            )
        )

    def record_review_submission(
        self,
        submission: Any,
        *,
        caused_by: str | None = None,
        correlation_id: str | None = None,
    ) -> str:
        return self._emit_only(
            ReviewSubmissionRecorded(
                **self._scope(
                    causation_id=caused_by,
                    correlation_id=correlation_id,
                    actor_type=HtrActorType.HUMAN,
                    actor_id=submission.reviewer_ref,
                    subject_id=submission.assignment_id,
                ),
                submission_id=submission.submission_id,
                assignment_id=submission.assignment_id,
                reviewer_ref=submission.reviewer_ref,
                record=submission.model_dump(mode="json"),
            )
        )

    def record_agreement_result(
        self,
        agreement: Any,
        *,
        caused_by: str | None = None,
        correlation_id: str | None = None,
    ) -> str:
        return self._emit_only(
            AgreementCalculatedHtr(
                **self._scope(
                    causation_id=caused_by,
                    correlation_id=correlation_id,
                    subject_id=agreement.target_ref,
                ),
                agreement_result_id=agreement.agreement_result_id,
                target_ref=agreement.target_ref,
                agrees=agreement.agrees,
                similarity_score=agreement.similarity_score,
                record=agreement.model_dump(mode="json"),
            )
        )

    def record_adjudication(
        self,
        adjudication: Any,
        *,
        caused_by: str | None = None,
        correlation_id: str | None = None,
    ) -> str:
        return self._emit_only(
            AdjudicationRecorded(
                **self._scope(
                    causation_id=caused_by,
                    correlation_id=correlation_id,
                    actor_type=HtrActorType.HUMAN,
                    actor_id=adjudication.adjudicator_ref,
                    subject_id=adjudication.agreement_result_id,
                ),
                adjudication_id=adjudication.adjudication_id,
                agreement_result_id=adjudication.agreement_result_id,
                adjudicator_ref=adjudication.adjudicator_ref,
                record=adjudication.model_dump(mode="json"),
            )
        )

    # -- Research knowledge (event-model doc §1 layers 10-12) ---------------------------------------
    #
    # These four methods are the first producers of `RESEARCH_OBSERVATION_CREATED`,
    # `CANDIDATE_FINDING_CREATED`, `FINDING_REVIEWED` and `FINDING_STATUS_CHANGED` -- four of the five
    # kinds `docs/architecture/htr-telemetry.md` §6 disclosed as "schema-only with no producer
    # anywhere in `src/`, ... pending the knowledge lifecycle". This is that lifecycle.
    #
    # `ResearchReportGenerated` deliberately still has no producer, for the reason recorded in its own
    # docstring: announcing a report as a *published research artifact* is a different act from
    # generating one, and Article 33 forbids a projection emitting telemetry about itself.

    def register_research_observation(
        self,
        observation: ResearchObservation,
        *,
        caused_by: str | None = None,
        correlation_id: str | None = None,
    ) -> str:
        """Durably records one `ResearchObservation`.

        **Correlation** defaults to the observation's own `source_experiment_run_id` when no scope is
        active -- the same rule `register_experiment_run` applies, and for the same reason: an
        observation extracted from a run's records belongs to that run's unit of work by definition,
        not by a caller remembering to say so. An observation spanning several runs (the GPU-memory
        one does) still correlates to the run it was *sourced* from, and names the others in its
        `scope.experiment_run_ids`; correlation is one value by construction, so the multi-run
        relationship lives in the scope where it can be plural.

        **Causation** should be the telemetry event that supplied the evidence -- typically the
        `MetricCalculated` or `ReliabilityIssueClassified` event the observation was extracted from.
        Passing it is the caller's honesty, not something this method can infer: the whole point of
        event-model doc §1's hard rule is that no automatic step turns an event into an observation,
        so there is deliberately no "figure out which event caused this" logic here.
        """
        return self._emit(
            ResearchObservationCreated(
                **self._scope(
                    causation_id=caused_by,
                    correlation_id=correlation_id
                    or self._correlation_id
                    or observation.source_experiment_run_id,
                    experiment_id=observation.source_experiment_id,
                    experiment_version_id=observation.scope.experiment_version_id,
                    experiment_run_id=observation.source_experiment_run_id,
                    dataset_id=observation.affected_dataset_id,
                    dataset_version_id=observation.affected_dataset_version_id,
                    subject_id=observation.observation_id,
                    actor_id=observation.author_or_source_component,
                ),
                observation_ref=observation.observation_id,
                summary=observation.title,
                evidence_refs=tuple(
                    ref.reference_id for ref in observation.supporting_evidence
                ),
                observation=observation,
            ),
            bucket=self._observations,
            key=observation.observation_id,
            kind="ResearchObservation",
            project=lambda: HtrResearchStore.register_research_observation(self, observation),
        )

    def register_candidate_finding(
        self,
        finding: ResearchFinding,
        *,
        caused_by: str | None = None,
        correlation_id: str | None = None,
    ) -> str:
        """Durably records one `ResearchFinding` at `Draft`/`Candidate` status.

        Refuses anything further along, so the log can never contain a `CandidateFindingCreated`
        announcing an already-`Supported` claim. `ResearchFinding.create` refuses it too; this is the
        persistence-layer half of the same rule, because a caller holding a legitimately-transitioned
        finding could otherwise register it here as if it were new and lose the review trail.

        `caused_by` should be the `ResearchObservationCreated` event of the observation the finding
        rests on -- that is the layer-10-to-layer-11 causal edge.
        """
        if finding.review_status not in INITIAL_FINDING_STATUSES:
            raise ValueError(
                f"register_candidate_finding refuses a finding at status "
                f"{finding.review_status.value!r}: this method records a finding's *creation*, and a "
                f"finding is only ever created at "
                f"{sorted(s.value for s in INITIAL_FINDING_STATUSES)}. Record a later status with "
                "record_finding_transition, which emits FindingReviewed/FindingStatusChanged and "
                "keeps the review trail."
            )
        return self._emit(
            CandidateFindingCreated(
                **self._scope(
                    causation_id=caused_by,
                    correlation_id=correlation_id,
                    experiment_id=finding.scope.experiment_id,
                    experiment_version_id=finding.scope.experiment_version_id,
                    experiment_run_id=finding.scope.experiment_run_ids[0],
                    dataset_version_id=finding.scope.dataset_version_id,
                    subject_id=finding.finding_id,
                    actor_id=finding.author,
                ),
                finding_ref=finding.finding_id,
                statement=finding.statement,
                observation_refs=finding.supporting_observations,
                finding=finding,
            ),
            bucket=self._findings,
            key=finding.finding_id,
            kind="ResearchFinding",
            project=lambda: HtrResearchStore.register_finding(self, finding),
        )

    def record_finding_transition(
        self,
        finding: ResearchFinding,
        new_status: FindingStatus,
        *,
        reviewer: str,
        reasoning: str,
        at: str,
        reproduction_evidence: tuple[EvidenceReference, ...] = (),
        superseded_by: str | None = None,
        contradiction: ContradictoryEvidence | None = None,
        caused_by: str | None = None,
        correlation_id: str | None = None,
    ) -> tuple[ResearchFinding, str]:
        """Transitions a finding and durably records both halves of the act.

        Returns `(transitioned_finding, status_change_event_id)`.

        Two events, in causal order, because they are two different facts and conflating them would
        lose one of them:

        1. `FindingReviewed` -- a named human looked at this finding and reached a verdict.
        2. `FindingStatusChanged`, caused by (1) -- the finding's status therefore changed, carrying
           the whole post-transition entity including its complete `revision_history`.

        A review that reached a verdict but changed nothing would emit only (1); a status change with
        no (2) before it cannot happen, since this is the only path that emits it.

        The transition rules themselves are **not** duplicated here: this delegates to
        `htr/knowledge/lifecycle.py::transition_finding_status` and lets its
        `InvalidFindingTransitionError` propagate *before* anything is appended, so a refused
        transition leaves no trace in the log. That ordering is deliberate -- `_emit`'s
        validate-then-append discipline applied to a domain rule rather than to a duplicate id.
        """
        transitioned = transition_finding_status(
            finding,
            new_status,
            reviewer=reviewer,
            reasoning=reasoning,
            at=at,
            reproduction_evidence=reproduction_evidence,
            superseded_by=superseded_by,
            contradiction=contradiction,
        )
        scope_fields: dict[str, Any] = {
            "experiment_id": transitioned.scope.experiment_id,
            "experiment_version_id": transitioned.scope.experiment_version_id,
            "experiment_run_id": transitioned.scope.experiment_run_ids[0],
            "subject_id": transitioned.finding_id,
        }
        review_event_id = self._emit_only(
            FindingReviewed(
                **self._scope(
                    causation_id=caused_by,
                    correlation_id=correlation_id,
                    actor_type=HtrActorType.HUMAN,
                    actor_id=reviewer,
                    **scope_fields,
                ),
                finding_ref=transitioned.finding_id,
                reviewer_ref=reviewer,
                verdict=new_status.value,
            )
        )
        status_event_id = self._emit_only(
            FindingStatusChanged(
                **self._scope(
                    causation_id=review_event_id,
                    correlation_id=correlation_id,
                    actor_type=HtrActorType.HUMAN,
                    actor_id=reviewer,
                    **scope_fields,
                ),
                finding_ref=transitioned.finding_id,
                new_status=new_status.value,
                previous_status=finding.review_status.value,
                reason=reasoning,
                finding=transitioned,
            )
        )
        self.advance_finding(transitioned)
        return transitioned, status_event_id

    # -- Coarse-entity JSON snapshot (derived cache) ------------------------------------------------

    def _snapshot_put(self, kind: str, key: str, entity: Any) -> None:
        if self._snapshot is not None:
            self._snapshot.put(kind, key, entity)

    # -- Reconstruction ------------------------------------------------------------------------------

    @classmethod
    def open(
        cls,
        sink: Any,
        *,
        actor_id: str | None = None,
        application_commit: str | None = None,
        snapshot: "HtrCoarseEntitySnapshot | None" = None,
    ) -> "DurableHtrResearchStore":
        """Constructs a store whose projection is already rebuilt from `sink`'s existing events.

        This is what makes a restart transparent: the process that opens an existing telemetry file
        sees every entity a previous process registered, reconstructed from the log alone -- never
        from a serialized snapshot of a previous store object, which the follow-up brief is explicit
        does not count as event sourcing.

        Replay deliberately runs into a **plain** `HtrResearchStore` first, whose contents are then
        adopted. Replaying directly through `self` would go through this class's emitting
        `register_*` overrides and append a second copy of every historical event -- doubling the log
        on every process start. Hydration writes to the projection only; the only events this store
        ever appends are ones a caller actually registers.
        """
        # Imported here, not at module scope, to keep `htr` -> `application` off the package-level
        # import graph: `application/htr_journal.py` legitimately depends on this package's sibling
        # `htr/research_store.py`, and a module-level edge back would make that mutual.
        from archivetrust.application.htr_journal import HtrJournal

        store = cls(
            sink,
            actor_id=actor_id,
            application_commit=application_commit,
            snapshot=snapshot,
        )
        store.adopt_projection(HtrJournal().replay(sink.all_events()))
        return store


class HtrCoarseEntitySnapshot:
    """Whole-object JSON persistence for the coarse registration entities, in
    `workspace/store.py::WorkspaceStore`'s idiom (one JSON file, Pydantic round-trip, overwrite in
    place, no event log).

    **A derived cache, never a source of truth.** Every entity written here also has a telemetry
    event behind it and is fully reconstructable by `HtrJournal.replay`; `rebuild` proves it by
    regenerating this file from the event stream. Its purpose is a cheap "what projects exist?"
    answer for a first-paint UI without parsing the whole event log, which is exactly what the gap
    analysis §11 offers `WorkspaceStore`'s pattern for -- entities that "don't change after
    creation".

    If this file is deleted, nothing is lost. That is the test of whether it is a cache, and it is
    why the split landed here rather than making these entities JSON-only (see
    `docs/architecture/htr-telemetry.md` §6).
    """

    SCHEMA = "archivetrust.htr_coarse_entities.v1"

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._data: dict[str, dict[str, Any]] = {}
        if self._path.exists():
            payload = json.loads(self._path.read_text(encoding="utf-8") or "{}")
            self._data = payload.get("entities", {})

    @property
    def path(self) -> Path:
        return self._path

    def put(self, kind: str, key: str, entity: Any) -> None:
        with self._lock:
            self._data.setdefault(kind, {})[key] = entity.model_dump(mode="json", by_alias=True)
            self._flush()

    def records(self, kind: str) -> tuple[dict[str, Any], ...]:
        with self._lock:
            return tuple(self._data.get(kind, {}).values())

    def _flush(self) -> None:
        self._path.write_text(
            json.dumps({"schema": self.SCHEMA, "entities": self._data}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def rebuild(self, store: HtrResearchStore) -> None:
        """Regenerates the snapshot from a (replayed) projection -- the proof that this file is
        derived. Called after a replay to bring a stale or deleted snapshot back in line."""
        with self._lock:
            self._data = {}
            for kind, entities, key in (
                ("projects", store.projects(), lambda e: e.project_id),
                ("datasets", store.datasets(), lambda e: e.dataset_id),
                ("dataset_versions", store.dataset_versions(), lambda e: e.dataset_version_id),
                ("collections", store.collections(), lambda e: e.collection_id),
                (
                    "conventions",
                    store.conventions(),
                    lambda e: f"{e.convention_id}@{e.version}",
                ),
            ):
                bucket = self._data.setdefault(kind, {})
                for entity in entities:
                    bucket[key(entity)] = entity.model_dump(mode="json", by_alias=True)
            self._flush()
