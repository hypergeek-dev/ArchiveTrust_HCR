"""Execution engine for the reliability benchmark: resumable, interruptible, telemetry-first.

Its execution logic is `scripts/run_segmentation_smoke_test.py`'s, unchanged in substance --
real Florence-2 line detection, byte-identical crops handed to both recognizers, the crop-hash
equality assertion, the reference-free reliability signals. What is added is the only thing the
smoke test did not need: **every unit of work is skippable, and the decision to skip is made from
the durable telemetry log.**

## Order of operations, and why it is the order

For every crop, for every method:

1. compute the task identity and ask the log whether its completion marker exists; skip if it does;
2. re-read the crop from disk and re-verify its content hash against the recorded `InputCrop.hash`
   (the controlled-comparison guard the domain design requires the runner to make);
3. call the adapter -- the only slow step, and the only step that can be interrupted;
4. *then* append `MethodRunStarted`, the `Evidence`, the transcript stages, the reliability
   classifications, and finally `MethodRunCompleted`.

Step 4 cannot begin before step 3 returns, because `MethodRun` requires the call's `outcome` and
`evidence_id` as constructor arguments. An interruption in step 3 therefore leaves no marker at all,
and a task that was interrupted mid-execution can never be recorded as completed. That is a
structural property of the model, not a convention this module could drift away from.

`FileTelemetrySink.append` opens, writes, and closes the file per event and extends the hash-chain
sidecar in the same call, so "flush telemetry" is what every single append already did. The stop
path re-verifies the chain rather than flushing a buffer, because there is no buffer.

## Safe stop and Ctrl+C are one path

`stop_signal.requested` is polled between bounded units of work. `KeyboardInterrupt` is caught at
the same level and converted into the identical stop, so Ctrl+C and `S` finish through one code
path with one set of guarantees -- there is deliberately no second, weaker shutdown.

A genuinely hung SATRN subprocess is already bounded: `providers/satrn/facade.py` runs the worker
under a 300-second timeout and maps the expiry to a real `FailureRecord` with
`category == "timeout"`. This module reuses that rather than adding a second timeout, so a hung call
returns a recorded *failure*, never a fabricated success and never an unbounded wait.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from archivetrust.htr.corpus.models import InputCrop, ResearchProject
from archivetrust.htr.experiment.baseline_execution import InputCropHashMismatchError
from archivetrust.htr.experiment.models import (
    Experiment,
    ExperimentRun,
    ExperimentVersion,
    MethodRun,
    ReproducibilityManifest,
)
from archivetrust.htr.research_store import DuplicateRegistrationError, MethodRunTranscript
from archivetrust.htr.screening.reliability_signals import (
    ReliabilityThresholds,
    evaluate_output,
)
from archivetrust.htr.screening.run_configuration import (
    FLORENCE2_METHOD_ID,
    RECOGNITION_METHOD_IDS,
    SATRN_METHOD_ID,
    ResolvedConfiguration,
)
from archivetrust.htr.screening.run_state import (
    CONFIGURATION_MANIFEST_KEY,
    IntegrityError,
    RunPaths,
    RunProgress,
    RunState,
    write_progress_cache,
)
from archivetrust.htr.segmentation import (
    LineDetectionFailedError,
    PageImage,
    SegmentationService,
)
from archivetrust.infrastructure.storage.integrity import verify_hash_chain_sidecar

THRESHOLD_METHOD_KEYS = {SATRN_METHOD_ID: "satrn", FLORENCE2_METHOD_ID: "florence2"}
"""`ReliabilityThresholds.truncation_char_length` is keyed by the short method names the smoke test
and the SATRN diagnostic used. Mapping here rather than renaming the catalog keys keeps previously
recorded threshold hashes meaningful."""

ACTOR_ID = "reliability-test-harness"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest}"


class SafeStopRequested(Exception):
    """Raised inside the execution loop at a safe boundary. Never escapes `ReliabilityRunner.run`."""


@dataclass
class StopSignal:
    """A cooperative stop flag. The interactive script's keyboard watcher sets it; a test sets it
    directly. Polled only at boundaries where no method call is in flight."""

    requested: bool = False
    reason: str | None = None

    def request(self, reason: str) -> None:
        if not self.requested:
            self.requested = True
            self.reason = reason


@dataclass
class RunEvent:
    """One display update. The runner never prints; it reports, and the caller decides what a
    terminal, a test, or a log does with it."""

    kind: str
    page_index: int = 0
    page_total: int = 0
    crop_index: int = 0
    crop_total: int = 0
    message: str = ""
    progress: RunProgress | None = None


@dataclass
class RunOutcome:
    """What one `run()` call actually did -- reported honestly, including a stop."""

    run_id: str
    stopped: bool
    stop_reason: str | None
    completed: bool
    tasks_executed: int = 0
    tasks_skipped: int = 0
    pages_segmented: int = 0
    elapsed_seconds: float = 0.0
    progress: RunProgress | None = None
    failures: list[str] = field(default_factory=list)
    chain_verified: bool = False


class _SegmentationStoreProxy:
    """Makes `SegmentationService` resumable without modifying it.

    Two overrides, both narrow:

    * `record_segmentation_run` substitutes the deterministic, identity-derived run id, so the
      terminal marker for a page's segmentation is findable by task key on the next process start.
      `SegmentationService` mints a `new_id()` one, which is correct for a one-shot script and
      useless as a resume key.
    * `register_page` tolerates a page that a previous, interrupted attempt already registered.
      `DurableHtrResearchStore` is append-only and refuses a duplicate id -- correctly -- so without
      this a page interrupted *during* segmentation could never be retried. The prior
      `PageRegistered` event id is returned so the causal chain still points at the real event.

    Everything else delegates untouched: the same store, the same append-then-project ordering, the
    same lock.
    """

    def __init__(self, store, *, segmentation_run_id: str, page_event_ids: dict[str, str]) -> None:
        self._store = store
        self._segmentation_run_id = segmentation_run_id
        self._page_event_ids = page_event_ids

    def register_page(self, page, *, caused_by=None, correlation_id=None):
        try:
            event_id = self._store.register_page(
                page, caused_by=caused_by, correlation_id=correlation_id
            )
        except DuplicateRegistrationError:
            existing = self._page_event_ids.get(page.page_id)
            if existing is None:
                raise
            return existing
        self._page_event_ids[page.page_id] = event_id
        return event_id

    def record_segmentation_run(self, **kwargs):
        kwargs["segmentation_run_id"] = self._segmentation_run_id
        return self._store.record_segmentation_run(**kwargs)

    def __getattr__(self, name):
        return getattr(self._store, name)


class ReliabilityRunner:
    """Executes pending tasks for one run, skipping everything the telemetry log says is done."""

    def __init__(
        self,
        *,
        repo_root: Path,
        paths: RunPaths,
        configuration: ResolvedConfiguration,
        store,
        segmentation_service,
        adapters: dict[str, object],
        thresholds: ReliabilityThresholds | None = None,
        observer: Callable[[RunEvent], None] | None = None,
    ) -> None:
        self._repo_root = repo_root
        self._paths = paths
        self._configuration = configuration
        self._store = store
        self._segmentation = segmentation_service
        self._adapters = adapters
        self._thresholds = thresholds or ReliabilityThresholds()
        self._observer = observer or (lambda event: None)

    # -- run scaffolding ---------------------------------------------------------------------------

    def ensure_run_registered(self, state: RunState) -> None:
        """Registers the run's scaffolding entities, exactly once, idempotently.

        Idempotent by construction rather than by a flag: every id is a deterministic function of
        the configuration, and `DurableHtrResearchStore` refuses a duplicate registration, so this
        checks the replayed projection first and registers only what is genuinely absent. A resumed
        run therefore appends no second copy of its own scaffolding -- the append-idempotency
        property the harness is required to hold.
        """
        configuration = self._configuration
        project_id = _stable_id("research_project", "technical-reliability-screening")
        experiment_id = _stable_id(
            "experiment", "swedish-historical-htr-technical-reliability-screening"
        )
        created_at = _now_iso()

        if self._store.project(project_id) is None:
            self._store.register_project(
                ResearchProject(
                    project_id=project_id,
                    name="Swedish Historical HTR Technical Reliability Screening",
                    description=(
                        "Reference-free technical reliability screening of SATRN and Florence-2 on "
                        "real Swedish witchcraft-trial court records. No ground truth exists for "
                        "this corpus; no accuracy metric is computed."
                    ),
                    created_at=created_at,
                )
            )
        if self._store.experiment(experiment_id) is None:
            self._store.register_experiment(
                Experiment(
                    experiment_id=experiment_id,
                    name="Technical reliability screening (60-page stratified sample)",
                    description=(
                        "Execution success/failure, timing, GPU memory and output-shape signals "
                        "only. Segmentation is Florence-2-family; that confound is recorded, not "
                        "neutralized."
                    ),
                    research_project_id=project_id,
                    created_at=created_at,
                )
            )
        version_id = configuration.experiment_version_id
        if self._store.experiment_version(version_id) is None:
            self._store.register_experiment_version(
                ExperimentVersion(
                    experiment_version_id=version_id,
                    experiment_id=experiment_id,
                    version=1,
                    dataset_version_id=configuration.dataset_version_id,
                    method_ids=configuration.method_ids,
                    segmentation_configuration_ref=(
                        configuration.segmentation_configuration_hash
                    ),
                    pipeline_configuration_ref=configuration.configuration_hash,
                    created_at=created_at,
                )
            )
        if not state.exists:
            run = ExperimentRun(
                experiment_run_id=self._paths.run_id,
                experiment_version_id=version_id,
                is_end_to_end=False,
                started_at=created_at,
            )
            self._store.register_experiment_run(run)
            self._store.register_manifest(
                ReproducibilityManifest(
                    manifest_id=_stable_id("reproducibility_manifest", self._paths.run_id),
                    experiment_run_id=self._paths.run_id,
                    software_environment={
                        CONFIGURATION_MANIFEST_KEY: configuration.model_dump(mode="json"),
                        "configuration_hash": configuration.configuration_hash,
                        "dataset_fingerprint": configuration.dataset_fingerprint,
                    },
                    pipeline_configuration_hash=configuration.configuration_hash,
                    created_at=created_at,
                )
            )
            state.experiment_run_id = self._paths.run_id
            state.started_at = created_at
            state.recorded_configuration = configuration

    def seal(self, state: RunState) -> None:
        """Appends the run's terminal marker once every task is genuinely complete."""
        run = self._store.experiment_run(self._paths.run_id)
        completed = ExperimentRun(
            experiment_run_id=self._paths.run_id,
            experiment_version_id=self._configuration.experiment_version_id,
            is_end_to_end=False,
            started_at=(run.started_at if run is not None else state.started_at or _now_iso()),
            completed_at=_now_iso(),
        )
        self._store.complete_experiment_run(completed)
        state.sealed = True
        state.seal_kind = "completed"

    # -- execution ---------------------------------------------------------------------------------

    def run(self, state: RunState, stop_signal: StopSignal) -> RunOutcome:
        """Executes every pending task, or stops safely. Never raises for a stop."""
        started = time.monotonic()
        outcome = RunOutcome(
            run_id=self._paths.run_id, stopped=False, stop_reason=None, completed=False
        )
        with self._store.correlated_to(self._paths.run_id):
            try:
                self._execute(state, stop_signal, outcome)
            except SafeStopRequested:
                outcome.stopped = True
                outcome.stop_reason = stop_signal.reason or "stop requested"
            except KeyboardInterrupt:
                # Routed through the identical finalization below -- same guarantees, no separate
                # or weaker path. Nothing was appended for the task that was in flight, so nothing
                # is falsely complete.
                stop_signal.request("keyboard interrupt (Ctrl+C)")
                outcome.stopped = True
                outcome.stop_reason = stop_signal.reason
            finally:
                # Finalization must never mask the exception that brought us here, and must never
                # itself become the reason a stop loses its record. Every step is independently
                # guarded; the durable events are already on disk regardless of what happens next.
                outcome.elapsed_seconds = time.monotonic() - started
                try:
                    outcome.progress = state.progress(self._configuration)
                    write_progress_cache(self._paths, outcome.progress)
                except Exception as error:  # noqa: BLE001
                    outcome.failures.append(f"progress could not be recomputed: {error}")
                try:
                    outcome.chain_verified = verify_hash_chain_sidecar(self._paths.events).ok
                except Exception as error:  # noqa: BLE001
                    outcome.failures.append(f"hash chain could not be verified: {error}")
        if not outcome.stopped and outcome.progress is not None and outcome.progress.finished:
            if not state.sealed:
                self.seal(state)
            outcome.completed = True
            outcome.progress = state.progress(self._configuration)
            write_progress_cache(self._paths, outcome.progress)
        return outcome

    def _check_stop(self, stop_signal: StopSignal) -> None:
        if stop_signal.requested:
            raise SafeStopRequested(stop_signal.reason or "stop requested")

    def _execute(self, state: RunState, stop_signal: StopSignal, outcome: RunOutcome) -> None:
        configuration = self._configuration
        pages = configuration.pages
        for page_index, page in enumerate(pages, start=1):
            self._check_stop(stop_signal)
            selected = state.selected_crops_for_page(configuration, page)
            if selected is None:
                self._observer(
                    RunEvent(
                        kind="segmenting",
                        page_index=page_index,
                        page_total=len(pages),
                        message=f"segmenting {page.page_id[:24]}",
                    )
                )
                selected = self._segment_page(state, page, page_index)
                if selected is None:
                    # Recorded as a real failure by `_segment_page`, and deliberately not counted
                    # as a segmented page: a page the detector could not read is not a page this
                    # run prepared.
                    outcome.failures.append(f"segmentation failed for page {page.page_id}")
                    continue
                outcome.pages_segmented += 1
            for position, (crop_index, crop) in enumerate(selected, start=1):
                for method_id in RECOGNITION_METHOD_IDS:
                    identity = state.recognition_identity(
                        configuration,
                        page_id=page.page_id,
                        crop_index=crop_index,
                        crop_hash=crop.hash,
                        method_id=method_id,
                    )
                    if identity.method_run_id in state.completed_method_run_ids:
                        outcome.tasks_skipped += 1
                        continue
                    # The stop is checked immediately before a method call, never during one.
                    self._check_stop(stop_signal)
                    self._observer(
                        RunEvent(
                            kind="recognizing",
                            page_index=page_index,
                            page_total=len(pages),
                            crop_index=position,
                            crop_total=len(selected),
                            message=method_id,
                            progress=state.progress(configuration),
                        )
                    )
                    self._recognize(
                        state=state,
                        page=page,
                        crop=crop,
                        identity=identity,
                        method_id=method_id,
                        outcome=outcome,
                    )
                    outcome.tasks_executed += 1
                    # Persisted after EVERY crop-method: the durable events are already on disk
                    # (append writes through), and the derived cache is brought back in line here
                    # so a hard kill one instruction later still leaves a consistent display.
                    write_progress_cache(self._paths, state.progress(configuration))

    def _segment_page(self, state: RunState, page, page_index: int):
        configuration = self._configuration
        identity = state.segmentation_identity(configuration, page)
        page_path = self._repo_root / page.relative_path
        image_bytes = page_path.read_bytes()
        page_image = PageImage(
            page_id=page.page_id, image_bytes=image_bytes, source_path=str(page_path)
        )
        proxy = _SegmentationStoreProxy(
            self._store,
            segmentation_run_id=identity.segmentation_run_id,
            page_event_ids=state.registered_page_event_ids,
        )
        # A per-page service over the same adapter, differing only in that its store is the resume
        # proxy. `SegmentationService` takes its store at construction, and the deterministic
        # segmentation-run id is per page, so this is the seam -- the adapter (and its loaded model)
        # is reused, not rebuilt.
        service = SegmentationService(
            adapter=self._segmentation.adapter, store=proxy, crop_directory=self._paths.crops
        )
        try:
            segmented = service.segment_and_record(
                page_image,
                archive_object_ref=page.document_id,
                page_number=page.page_number,
                correlation_id=self._paths.run_id,
                # Short, unique, stable per page: a `page_id` is 69 characters and Windows MAX_PATH
                # is the reason `crop_directory_name` exists at all.
                crop_directory_name=f"p{page_index:03d}",
            )
        except LineDetectionFailedError as error:
            self._store.record_reliability_issue(
                method_run_id=identity.segmentation_run_id,
                classification=error.category or "segmentation_failed",
                detail=str(error),
            )
            return None
        state.completed_segmentation_run_ids.add(identity.segmentation_run_id)
        state.segmentation_pages[identity.segmentation_run_id] = page.page_id
        state.segmentation_crop_ids[identity.segmentation_run_id] = tuple(
            crop.crop_id for crop in segmented.input_crops
        )
        for crop in segmented.input_crops:
            state.crops_by_id[crop.crop_id] = crop
        return state.selected_crops_for_page(configuration, page)

    def _recognize(
        self,
        *,
        state: RunState,
        page,
        crop: InputCrop,
        identity,
        method_id: str,
        outcome: RunOutcome,
    ) -> None:
        from archivetrust.providers.htr_adapter import RecognitionInput  # noqa: PLC0415

        crop_path = Path(crop.storage_path)
        if not crop_path.is_file():
            raise IntegrityError(
                f"crop {crop.crop_id} is recorded in telemetry but its file is missing at "
                f"{crop_path}. Completed evidence cannot be reconstructed; refusing to continue "
                "rather than silently re-segmenting the page."
            )
        on_disk = crop_path.read_bytes()
        recomputed = InputCrop.compute_hash(on_disk)
        if recomputed != crop.hash:
            raise InputCropHashMismatchError(
                f"crop {crop.crop_id} on disk hashes to {recomputed!r} but telemetry recorded "
                f"{crop.hash!r}. The controlled comparison's byte-identity guarantee is broken."
            )

        adapter = self._adapters[method_id]
        started_at = _now_iso()
        wall_started = time.monotonic()
        # --- the only interruptible step. Nothing has been appended for this task yet. ---
        result = adapter.recognize(RecognitionInput(input_crop_id=str(crop_path)))
        wall_seconds = time.monotonic() - wall_started
        completed_at = _now_iso()

        if method_id == SATRN_METHOD_ID:
            from archivetrust.providers.satrn.adapter import (  # noqa: PLC0415
                build_evidence,
                build_failure_record,
                normalize_transcription,
            )

            raw_text, parsed_text = result.text, None
        else:
            from archivetrust.providers.florence2_htr.adapter import (  # noqa: PLC0415
                build_evidence,
                build_failure_record,
                normalize_transcription,
            )

            raw_text = result.raw_response.get("raw_decoded") if result.text is not None else None
            parsed_text = result.text

        evidence = build_evidence(result)
        method_run = MethodRun(
            method_run_id=identity.method_run_id,
            experiment_run_id=self._paths.run_id,
            method_id=method_id,
            model_version_id=result.model_revision,
            input_crop_id=crop.crop_id,
            evidence_id=(
                evidence.evidence_id
                if evidence is not None
                else _stable_id("evidence_missing", identity.task_key)
            ),
            outcome="succeeded" if result.text is not None else "failed",
            started_at=started_at,
            completed_at=completed_at,
        )
        failure = build_failure_record(result, method_run_id=method_run.method_run_id)

        reattempt = False
        try:
            started_event = self._store.register_method_run(method_run)
        except DuplicateRegistrationError:
            # A previous attempt appended the started marker and died before the completion marker
            # -- a window of pure appends, but a real one. Recorded as a fact rather than hidden,
            # and the fresh result below is what gets completed.
            reattempt = True
            started_event = None
            self._store.record_reliability_issue(
                method_run_id=method_run.method_run_id,
                classification="interrupted_attempt_reexecuted",
                detail=(
                    "a prior process appended MethodRunStarted for this task but no "
                    "MethodRunCompleted; the method call was re-executed"
                ),
            )
        if evidence is not None:
            self._store.record_evidence(
                evidence,
                document_ref=page.document_id,
                invocation_id=method_run.method_run_id,
                caused_by=started_event,
            )
        if not reattempt:
            normalized = (
                normalize_transcription(result.text) if result.text is not None else None
            )
            self._store.register_transcript(
                MethodRunTranscript(
                    method_run_id=method_run.method_run_id,
                    raw_text=raw_text,
                    parsed_text=parsed_text,
                    normalized_text=normalized,
                ),
                caused_by=started_event,
                evidence_id=method_run.evidence_id,
            )

        signals = evaluate_output(
            result.text,
            method=THRESHOLD_METHOD_KEYS.get(method_id, method_id),
            raw_response=result.raw_response,
            thresholds=self._thresholds,
        )
        for classification, flagged, detail in (
            ("execution_failed", not signals.produced_output, signals.failure_message),
            ("empty_or_whitespace_only", signals.empty_or_whitespace_only, None),
            ("degenerate_repetition", signals.degenerate_repetition, None),
            ("very_short_output", signals.very_short_output, None),
            ("truncation_suspected", signals.truncation_suspected, None),
        ):
            if flagged:
                self._store.record_reliability_issue(
                    method_run_id=method_run.method_run_id,
                    classification=classification,
                    detail=detail,
                    caused_by=started_event,
                )
        crop_aspect = (
            (crop.width / crop.height) if (crop.width and crop.height) else None
        )
        if crop_aspect is not None and crop_aspect < self._thresholds.crop_min_aspect_ratio:
            self._store.record_reliability_issue(
                method_run_id=method_run.method_run_id,
                classification="crop_geometry_implausible",
                detail=f"width/height = {crop_aspect:.3f}",
                caused_by=started_event,
            )

        # THE completion marker -- appended last, only now that the call has genuinely returned.
        self._store.complete_method_run(
            method_run,
            caused_by=started_event,
            failure_reason=failure.reason if failure is not None else None,
        )
        state.completed_method_run_ids.add(method_run.method_run_id)
        state.method_run_outcomes[method_run.method_run_id] = method_run.outcome
        state.method_run_methods[method_run.method_run_id] = method_id
        state.method_run_crop_ids[method_run.method_run_id] = crop.crop_id
        if result.text is None:
            outcome.failures.append(
                f"{method_id} failed on crop {crop.crop_id[:20]}: "
                f"{result.raw_response.get('message')}"
            )
        self._observer(
            RunEvent(
                kind="recognized",
                # Outcome and cost only. The transcription itself is never printed: this codebase
                # has no verbose-logging flag to gate it behind, and one is deliberately not
                # invented here. The text is fully recorded where it belongs -- in the durable
                # transcript-stage events, readable via the report.
                message=(
                    f"{method_id} {'ok' if result.text is not None else 'FAIL'} "
                    f"{wall_seconds:.2f}s"
                ),
            )
        )


def summarize_pending(
    state: RunState, configuration: ResolvedConfiguration
) -> tuple[int, dict[str, int]]:
    """`(pending pages, pending tasks per method)` -- computed from telemetry, for the resume
    summary and for deciding whether an unfinished run has any work left."""
    pending_pages = 0
    pending: dict[str, int] = {method: 0 for method in RECOGNITION_METHOD_IDS}
    for page in configuration.pages:
        selected = state.selected_crops_for_page(configuration, page)
        if selected is None:
            pending_pages += 1
            for method in RECOGNITION_METHOD_IDS:
                pending[method] += configuration.crops_per_page
            continue
        page_pending = False
        for crop_index, crop in selected:
            for method in RECOGNITION_METHOD_IDS:
                identity = state.recognition_identity(
                    configuration,
                    page_id=page.page_id,
                    crop_index=crop_index,
                    crop_hash=crop.hash,
                    method_id=method,
                )
                if identity.method_run_id not in state.completed_method_run_ids:
                    pending[method] += 1
                    page_pending = True
        if page_pending:
            pending_pages += 1
    return pending_pages, pending


def last_completed_point(
    state: RunState, configuration: ResolvedConfiguration
) -> str | None:
    """The last task the log records as complete, in the run's own execution order -- what the
    resume summary calls the 'last safe completion point'. Derived from telemetry ordering, never
    remembered from a previous process."""
    last: str | None = None
    for page_index, page in enumerate(configuration.pages, start=1):
        try:
            selected = state.selected_crops_for_page(configuration, page)
        except IntegrityError:
            return last
        if selected is None:
            continue
        for position, (crop_index, crop) in enumerate(selected, start=1):
            for method in RECOGNITION_METHOD_IDS:
                identity = state.recognition_identity(
                    configuration,
                    page_id=page.page_id,
                    crop_index=crop_index,
                    crop_hash=crop.hash,
                    method_id=method,
                )
                if identity.method_run_id in state.completed_method_run_ids:
                    last = (
                        f"page {page_index}/{configuration.page_count}, "
                        f"crop {position}/{len(selected)}, {method}"
                    )
    return last


def build_adapters(method_ids: Sequence[str] | None = None) -> dict[str, object]:
    """The real adapters, constructed lazily so importing this module needs no GPU."""
    from archivetrust.providers.florence2_htr.adapter import Florence2Adapter  # noqa: PLC0415
    from archivetrust.providers.satrn.adapter import SatrnAdapter  # noqa: PLC0415

    available = {SATRN_METHOD_ID: SatrnAdapter(), FLORENCE2_METHOD_ID: Florence2Adapter()}
    if method_ids is None:
        return available
    return {method_id: available[method_id] for method_id in method_ids}
