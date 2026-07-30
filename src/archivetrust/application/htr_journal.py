"""HTR read-model reconstruction: rebuild every HTR research entity from telemetry alone.

The HTR counterpart of `application/journal.py::Journal.replay`, and deliberately a *separate*
journal rather than new branches on the existing one.

**Why separate** (the choice `docs/htr-telemetry-knowledge-gap-analysis.md` §11 anticipated and
`docs/architecture/htr-event-model.md` §2 pre-committed to): `Journal.replay` returns a
`JournalState` whose entire query surface is Evidence / Observation graphs / CanonicalObservation
chains / confidence evolution -- and `HtrJournal.replay` returns an `HtrResearchStore` whose entire
query surface is projects / datasets / experiments / method runs / metrics. The two reconstruct
**disjoint entity sets from disjoint event kinds**. Merging them would have produced one class with
two unrelated halves, where every caller uses one half and ignores the other, and where a bug in HTR
replay could corrupt Trust Engine state. The two streams are also stored in separate files
(`composition.py`), so no single replay ever needs to serve both.

What was *not* duplicated: the mechanism. `FileTelemetrySink` is reused unmodified, the event
vocabulary is the same closed `TelemetryEventKind` enum, and this module follows
`Journal._apply`'s exhaustive-`isinstance`-chain shape exactly, including its discipline of ending
in an explicit comment naming every kind that is a deliberate no-op rather than letting unhandled
kinds fall through unremarked.

This module is `application`, not `domain`, for the same reason `journal.py` is: it orchestrates
domain types without being one. It is also the layer allowed to import `review/`, `evaluation/` and
`providers/` -- which is what lets it reconstruct the typed models that
`domain/telemetry/events.py` may only carry as `record` dicts (see that module's import-block
comment, and `docs/architecture/htr-telemetry.md` §5).
"""

from __future__ import annotations

from collections.abc import Iterable

from archivetrust.domain.telemetry.events import (
    AdjudicationRecorded,
    AgreementCalculatedHtr,
    CandidateFindingCreated,
    CanonicalResultCreated,
    CollectionCreated,
    DatasetCreated,
    DatasetVersionCreated,
    DocumentRegistered,
    ExperimentCreated,
    ExperimentRunCompleted,
    ExperimentRunFailed,
    ExperimentRunStarted,
    ExperimentVersionCreated,
    ExternalResultImported,
    FindingReviewed,
    FindingStatusChanged,
    GroundTruthTextRecorded,
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
    ResearchReportGenerated,
    ReviewAssigned,
    ReviewedResultRecorded,
    ReviewSubmissionRecorded,
    SegmentationRunCompleted,
    TelemetryEvent,
    TextLineDetected,
    TranscriptionConventionRegistered,
)
from archivetrust.evaluation.ground_truth import TranscriptionConvention
from archivetrust.htr.research_store import HtrResearchStore
from archivetrust.providers.transkribus.external_import import ExternalImport

HtrJournalState = HtrResearchStore
"""The name `docs/architecture/htr-event-model.md` §2 uses for what replay produces, bound to the
class that actually is it.

There is deliberately no separate `HtrJournalState` *class*: `HtrResearchStore` already has exactly
the query surface a reconstructed HTR read model needs (the gap analysis §10 is explicit that its
query interface is not the problem), and every `presentation/htr_*` ViewModel already consumes it.
A parallel class would have duplicated that surface field-for-field so that two objects could answer
the same questions -- and would have guaranteed the two drifted. See
`htr/research_store.py`'s module docstring for the store's four current jobs.
"""


class HtrJournal:
    """Replays a stream of `TelemetryEvent`s into an `HtrResearchStore` projection.

    Stateless across calls -- each `replay()` starts a fresh store, so a reconstruction is
    reproducible purely from the stored events, which is the whole point (Constitution Article 17:
    rebuild state from the log, never by re-running a provider or a recognition method).
    """

    def replay(self, events: Iterable[TelemetryEvent]) -> HtrResearchStore:
        """Reconstructs the HTR read model from an event stream.

        Non-HTR events are skipped rather than rejected: a caller may legitimately hand this the
        whole of a mixed stream, and the pre-existing OCR-era kinds are `Journal`'s business, not
        this one's.
        """
        store = HtrResearchStore()
        for event in events:
            self._apply(store, event)
        return store

    def _apply(self, store: HtrResearchStore, event: TelemetryEvent) -> None:
        # -- Corpus registration ---------------------------------------------------------------
        if isinstance(event, ResearchProjectCreated):
            store.register_project(event.project)
        elif isinstance(event, DatasetCreated):
            store.register_dataset(event.dataset)
        elif isinstance(event, DatasetVersionCreated):
            store.register_dataset_version(event.dataset_version)
        elif isinstance(event, CollectionCreated):
            store.register_collection(event.collection)
        elif isinstance(event, PageRegistered):
            store.register_page(event.page)
        elif isinstance(event, RegionDetected):
            store.register_region(event.region)
        elif isinstance(event, TextLineDetected):
            store.register_text_line(event.text_line)
        elif isinstance(event, InputCropCreated):
            # The crop's content address was computed and validated by `InputCrop` itself before it
            # was ever put in an event; nothing is re-hashed here (event-model doc §2: no second
            # hashing scheme).
            store.register_input_crop(event.input_crop)
        elif isinstance(event, TranscriptionConventionRegistered):
            if event.record is not None:
                store.register_convention(TranscriptionConvention.model_validate(event.record))

        # -- Experiment execution ---------------------------------------------------------------
        elif isinstance(event, ExperimentCreated):
            store.register_experiment(event.experiment)
        elif isinstance(event, ExperimentVersionCreated):
            store.register_experiment_version(event.experiment_version)
        elif isinstance(event, ExperimentRunStarted):
            store.register_experiment_run(event.experiment_run)
        elif isinstance(event, ExperimentRunCompleted):
            # The same run, now carrying `completed_at`. Advancing the projection entry is not
            # rewriting history: both events remain in the durable log in append order, and a fresh
            # replay reaches this same state deterministically.
            store.advance_experiment_run(event.experiment_run)
        elif isinstance(event, MethodRunStarted):
            store.register_method_run(event.method_run)
        elif isinstance(event, MethodRunFailed):
            # Preserved, never excluded (docs/htr-domain-design.md §1).
            store.register_failure(event.failure_record)

        # -- Method-result stages (MethodRunTranscript's four, kept distinct) -------------------
        elif isinstance(event, RawMethodResultRecorded):
            store.apply_transcript_stage(
                method_run_id=event.method_run_id, stage="raw", text=event.text
            )
        elif isinstance(event, ParsedMethodResultRecorded):
            store.apply_transcript_stage(
                method_run_id=event.method_run_id, stage="parsed", text=event.text
            )
        elif isinstance(event, NormalizedMethodResultRecorded):
            store.apply_transcript_stage(
                method_run_id=event.method_run_id, stage="normalized", text=event.text
            )
        elif isinstance(event, ReviewedResultRecorded):
            store.apply_transcript_stage(
                method_run_id=event.method_run_id,
                stage="reviewed",
                text=event.text,
                reviewer_ref=event.reviewer_ref,
            )

        # -- Evaluation --------------------------------------------------------------------------
        elif isinstance(event, MetricDefinitionRegistered):
            store.register_metric_definition(event.metric_definition)
        elif isinstance(event, MetricCalculated):
            store.register_metric_result(event.metric_result)
        elif isinstance(event, GroundTruthTextRecorded):
            store.register_ground_truth(text_line_id=event.text_line_id, text=event.text)

        # -- Reproducibility, canonicalization, external import ---------------------------------
        elif isinstance(event, ReproducibilityManifestRecorded):
            store.register_manifest(event.manifest)
        elif isinstance(event, CanonicalResultCreated):
            if event.canonical_result is not None:
                store.register_canonical_result(event.canonical_result)
        elif isinstance(event, ExternalResultImported):
            if event.record is not None:
                store.register_external_import(ExternalImport.model_validate(event.record))

        # -- Research knowledge (event-model doc §1 layers 10-12) -------------------------------
        # Added 2026-07-30 with the knowledge lifecycle. Each of these three kinds carries its full
        # typed entity, so a replay rebuilds an observation's evidence links and a finding's entire
        # revision history from the log alone.
        elif isinstance(event, ResearchObservationCreated):
            if event.observation is not None:
                store.register_research_observation(event.observation)
        elif isinstance(event, CandidateFindingCreated):
            if event.finding is not None:
                store.register_finding(event.finding)
        elif isinstance(event, FindingStatusChanged):
            # A later state of the same finding, carrying every revision that produced it -- the same
            # projection-advancement case as `ExperimentRunCompleted`, and `advance_finding` refuses a
            # finding whose `CandidateFindingCreated` is missing rather than inventing one at a status
            # it could never have been created in.
            if event.finding is not None:
                store.advance_finding(event.finding)

        # Every remaining HTR kind is a deliberate no-op for *this* projection, for one of four
        # reasons -- named individually rather than left to fall through silently, following
        # `Journal._apply`'s own closing-comment discipline:
        #
        # 1. Terminal markers whose entity was already fully established by the event they chain
        #    from, so there is nothing further to reconstruct:
        #      `MethodRunCompleted` (a MethodRun is constructed once, already terminal -- see
        #      `advance_experiment_run`'s docstring for why it has no MethodRun counterpart).
        # 2. Summary events that reference by id entities which are each announced individually
        #    with their full object, so replaying the summary would duplicate them:
        #      `SegmentationRunCompleted` (its regions/lines/crops arrive as `RegionDetected`/
        #      `TextLineDetected`/`InputCropCreated`), `DocumentRegistered` (the membership it
        #      records is already a field on the `Collection` that `CollectionCreated` carried).
        # 3. Facts with no bucket in this projection's query surface, which remain fully readable
        #    from the sink they were appended to -- nothing is dropped, only not indexed here:
        #      `ExperimentRunFailed`, `ReliabilityIssueClassified`.
        # 4. Review-lineage and research-knowledge kinds owned by a different projection:
        #      `ReviewAssigned`, `ReviewSubmissionRecorded`, `AgreementCalculatedHtr`,
        #      `AdjudicationRecorded` -- `review/blind_review/store.py::BlindReviewStore` owns the
        #      review query surface, and replacing *its* persistence is explicitly deferred to a
        #      later pass (docs/architecture/htr-telemetry.md §7). Their events are durable and
        #      correlated as of this pass; only their projection is deferred.
        #      `FindingReviewed` -- the *act* of review, whose outcome is carried by the
        #      `FindingStatusChanged` it causes; projecting both would apply one state twice.
        #      Nothing is lost: `FindingReviewed` remains durable and is what the reviewer
        #      attribution is read from, and the reviewer is also on the finding itself.
        #      `ResearchReportGenerated` -- still schema-only, no producer anywhere in `src/`
        #      (announcing a published report is a separate, later concern -- see its docstring).
        elif isinstance(
            event,
            (
                MethodRunCompleted,
                SegmentationRunCompleted,
                DocumentRegistered,
                ExperimentRunFailed,
                ReliabilityIssueClassified,
                ReviewAssigned,
                ReviewSubmissionRecorded,
                AgreementCalculatedHtr,
                AdjudicationRecorded,
                FindingReviewed,
                ResearchReportGenerated,
            ),
        ):
            return
        # Anything else is a pre-existing OCR-era event kind belonging to `Journal`, not here.


def causation_chain(
    events: Iterable[TelemetryEvent], *, from_event_id: str
) -> tuple[TelemetryEvent, ...]:
    """Walks the `causation_id` DAG forward from one event, following *only* causation pointers.

    Returns the chain in causal order, starting with the named event. Deliberately consults no
    timestamp, no append position, and no shared domain id -- the gap analysis §1 found that
    "correlation today is implicit, via shared domain ids embedded per-event", and the whole point of
    `causation_id` (event-model doc §4) is to replace that inference with an explicit, queryable
    edge. A chain that only reproduces append order would prove nothing.

    Where one event caused several (a `MethodRunStarted` causing both a completion marker and a raw
    result), the branch that itself continues furthest is followed, so the returned chain is the
    longest causal path rather than an arbitrary sibling.
    """
    by_id: dict[str, TelemetryEvent] = {}
    children: dict[str, list[str]] = {}
    for event in events:
        by_id[event.event_id] = event
        if event.causation_id is not None:
            children.setdefault(event.causation_id, []).append(event.event_id)

    if from_event_id not in by_id:
        raise KeyError(f"no event with event_id {from_event_id!r} in the supplied stream")

    def longest(event_id: str) -> list[TelemetryEvent]:
        best: list[TelemetryEvent] = []
        for child_id in children.get(event_id, ()):
            candidate = longest(child_id)
            if len(candidate) > len(best):
                best = candidate
        return [by_id[event_id], *best]

    return tuple(longest(from_event_id))
