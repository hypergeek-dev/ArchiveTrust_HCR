"""Run discovery, telemetry-replayed state, the derived progress cache, and the resume checks.

**The durable telemetry log is the only source of truth here.** Every "is this done?" answer in this
module is computed by reading the run's `events.jsonl` -- the same append-only, hash-chained stream
every other HTR entity in this codebase goes through (`htr/persistence/durable_store.py`,
`application/htr_journal.py`). `ProgressCache` exists purely so the menu can print counts without
re-reading a large log, and `RunState.rebuild_progress_cache` regenerates it from the events alone;
deleting the cache changes nothing a user can observe except the speed of the first paint.

## What a run is on disk

    docs/experiments/technical-reliability-screening/full-run/<run_id>/
        events.jsonl                 the durable, append-only record  (authoritative)
        events.jsonl.chain.jsonl     the tamper-evidence sidecar      (authoritative)
        progress-cache.json          fast display only                (derived, disposable)
        crops/                       the real line crops              (large, gitignored)
        blobs/                       normalization artifacts          (large, gitignored)

A run id is minted once and never reused; `allocate_run_id` refuses to hand back an id whose
directory already exists, so a prior run's telemetry can never be overwritten.

## "Unfinished"

A run is **unfinished** when its log contains an `ExperimentRunStarted` and contains neither
`ExperimentRunCompleted` nor `ExperimentRunFailed` for that same `experiment_run_id`. That is a
statement about recorded facts only -- it deliberately does not consult the progress cache, the
filesystem, or a process table, because none of those survive a machine reboot and all of them can
disagree with the log.

An unfinished run with zero pending tasks is a *finished but unsealed* run: all its work is genuinely
recorded, only the terminal marker is missing (the process died between the last crop and the seal).
Resume seals it and reports completion rather than re-executing anything.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from archivetrust.domain.telemetry.events import (
    ExperimentRunCompleted,
    ExperimentRunFailed,
    ExperimentRunStarted,
    InputCropCreated,
    MethodRunCompleted,
    MethodRunStarted,
    PageRegistered,
    ReproducibilityManifestRecorded,
    SegmentationRunCompleted,
)
from archivetrust.htr.corpus.models import InputCrop
from archivetrust.htr.screening.run_configuration import (
    RECOGNITION_METHOD_IDS,
    ResolvedConfiguration,
    SEGMENTATION_METHOD_ID,
    evenly_spaced_indices,
    screening_directory,
)
from archivetrust.htr.screening.task_identity import TaskIdentity, crop_slot_id
from archivetrust.infrastructure.storage.integrity import verify_hash_chain_sidecar

RUNS_DIR_NAME = "full-run"
PROGRESS_CACHE_NAME = "progress-cache.json"
EVENTS_NAME = "events.jsonl"
PROGRESS_CACHE_SCHEMA = "archivetrust.reliability_progress_cache.v1"

CONFIGURATION_MANIFEST_KEY = "reliability_run_configuration"
"""Key under `ReproducibilityManifest.software_environment` where the whole resolved configuration
is stored. `ReproducibilityManifest` is the existing, replayable entity for "everything needed to
reproduce this run" (`htr/experiment/models.py`); recording the configuration anywhere else would
have been a second persistence mechanism for the one fact this harness most depends on."""


class RunStateError(RuntimeError):
    """A run's recorded state could not be read or reconstructed cleanly."""


class ConfigurationMismatchError(RunStateError):
    """The current configuration differs from the one the unfinished run recorded.

    Carries the specific differing fields, because "configuration changed" on its own is not an
    actionable message and the whole point of the check is that a user must be told exactly what
    would have been silently mixed.
    """

    def __init__(self, differences: Mapping[str, tuple[str, str]]) -> None:
        self.differences = dict(differences)
        lines = [
            f"  {name}:\n      recorded now : {current}\n      recorded then: {recorded}"
            for name, (recorded, current) in self.differences.items()
        ]
        super().__init__(
            "This run cannot be resumed: the current configuration differs from the one it was "
            "started under.\n" + "\n".join(lines) + "\n"
            "Resuming would mix two configurations in one body of evidence. Start a new run, or "
            "restore the configuration this run was started under."
        )


class IntegrityError(RunStateError):
    """The run's telemetry hash chain did not verify, or its recorded evidence is unreadable."""


@dataclass(frozen=True)
class RunPaths:
    """Where one run's files live. Constructed, never guessed at call sites."""

    run_id: str
    directory: Path

    @property
    def events(self) -> Path:
        return self.directory / EVENTS_NAME

    @property
    def progress_cache(self) -> Path:
        return self.directory / PROGRESS_CACHE_NAME

    @property
    def crops(self) -> Path:
        return self.directory / "crops"

    @property
    def blobs(self) -> Path:
        return self.directory / "blobs"

    @property
    def report(self) -> Path:
        return self.directory / "reliability-run-report.json"

    @classmethod
    def for_run(cls, runs_root: Path, run_id: str) -> "RunPaths":
        return cls(run_id=run_id, directory=runs_root / run_id)


def runs_root(repo_root: Path) -> Path:
    return screening_directory(repo_root) / RUNS_DIR_NAME


def allocate_run_id(runs_root_dir: Path, *, today: date | None = None) -> str:
    """Mints a fresh, human-readable run id that no existing run directory uses.

    `reliability-YYYY-MM-DD`, with `-2`, `-3`, ... appended if that day already has a run. Chosen
    over a uuid because the user reads and says this id out loud; chosen over a bare date because a
    second run on the same day must never collide with the first. This function *never* returns an
    id whose directory exists, which is the mechanism by which a prior run's telemetry cannot be
    overwritten.
    """
    stamp = (today or date.today()).isoformat()
    candidate = f"reliability-{stamp}"
    suffix = 1
    while (runs_root_dir / candidate).exists():
        suffix += 1
        candidate = f"reliability-{stamp}-{suffix}"
    return candidate


def discover_runs(runs_root_dir: Path) -> tuple[RunPaths, ...]:
    """Every run directory that has a telemetry file, oldest id first."""
    if not runs_root_dir.is_dir():
        return ()
    found = [
        RunPaths(run_id=child.name, directory=child)
        for child in sorted(runs_root_dir.iterdir())
        if child.is_dir() and (child / EVENTS_NAME).is_file()
    ]
    return tuple(found)


def find_unfinished_run(runs_root_dir: Path) -> "tuple[RunPaths, RunState] | None":
    """The one unfinished run, or `None`.

    This is the predicate the menu's "Run" refuses on and the predicate "Resume" acts on, so it
    lives here where it is tested rather than inside the interactive script. Most recent run id
    first, because a run started later is the one a user means. An `IntegrityError` from any
    candidate propagates: a run whose hash chain does not verify must surface, never be skipped
    over into a reassuring "no unfinished run".
    """
    for paths in reversed(discover_runs(runs_root_dir)):
        state = RunState.load(paths)
        if state.unfinished:
            return paths, state
    return None


def latest_run(runs_root_dir: Path) -> "tuple[RunPaths, RunState] | None":
    runs = discover_runs(runs_root_dir)
    if not runs:
        return None
    return runs[-1], RunState.load(runs[-1])


@dataclass
class RunState:
    """One run's state, reconstructed from its durable telemetry log and nothing else."""

    paths: RunPaths
    experiment_run_id: str | None = None
    started_at: str | None = None
    sealed: bool = False
    seal_kind: str | None = None
    recorded_configuration: ResolvedConfiguration | None = None
    completed_segmentation_run_ids: set[str] = field(default_factory=set)
    segmentation_pages: dict[str, str] = field(default_factory=dict)
    """segmentation_run_id -> page_id."""
    segmentation_crop_ids: dict[str, tuple[str, ...]] = field(default_factory=dict)
    """segmentation_run_id -> the page's crop ids, in detected reading order."""
    crops_by_id: dict[str, InputCrop] = field(default_factory=dict)
    completed_method_run_ids: set[str] = field(default_factory=set)
    method_run_outcomes: dict[str, str] = field(default_factory=dict)
    method_run_methods: dict[str, str] = field(default_factory=dict)
    method_run_crop_ids: dict[str, str] = field(default_factory=dict)
    registered_page_event_ids: dict[str, str] = field(default_factory=dict)
    event_count: int = 0

    @property
    def exists(self) -> bool:
        return self.experiment_run_id is not None

    @property
    def unfinished(self) -> bool:
        return self.exists and not self.sealed

    # -- reconstruction ---------------------------------------------------------------------------

    @classmethod
    def replay(cls, paths: RunPaths, events: Iterable) -> "RunState":
        """Builds the state from an event stream.

        Deliberately reads the raw stream rather than `HtrJournal.replay`'s projection: the four
        facts this harness needs -- which segmentation runs completed, which method runs completed,
        which crops exist with which content hash, and what configuration was recorded -- are
        carried by `SegmentationRunCompleted`, `MethodRunCompleted`, `InputCropCreated` and
        `ReproducibilityManifestRecorded` respectively, and two of those four are among the kinds
        `HtrJournal` documents as deliberate projection no-ops. The projection is not wrong; it
        simply does not index terminal markers, which is exactly what a completion check is.
        """
        state = cls(paths=paths)
        for event in events:
            state.event_count += 1
            if isinstance(event, ExperimentRunStarted):
                state.experiment_run_id = event.experiment_run.experiment_run_id
                state.started_at = event.experiment_run.started_at
            elif isinstance(event, ExperimentRunCompleted):
                state.sealed = True
                state.seal_kind = "completed"
            elif isinstance(event, ExperimentRunFailed):
                state.sealed = True
                state.seal_kind = "failed"
            elif isinstance(event, ReproducibilityManifestRecorded):
                payload = event.manifest.software_environment.get(CONFIGURATION_MANIFEST_KEY)
                if payload is not None:
                    state.recorded_configuration = ResolvedConfiguration.model_validate(payload)
            elif isinstance(event, PageRegistered):
                state.registered_page_event_ids[event.page.page_id] = event.event_id
            elif isinstance(event, InputCropCreated):
                state.crops_by_id[event.input_crop.crop_id] = event.input_crop
            elif isinstance(event, SegmentationRunCompleted):
                state.completed_segmentation_run_ids.add(event.segmentation_run_id)
                state.segmentation_pages[event.segmentation_run_id] = event.page_id
                state.segmentation_crop_ids[event.segmentation_run_id] = tuple(
                    event.input_crop_ids
                )
            elif isinstance(event, MethodRunStarted):
                run = event.method_run
                state.method_run_methods[run.method_run_id] = run.method_id
                if run.input_crop_id is not None:
                    state.method_run_crop_ids[run.method_run_id] = run.input_crop_id
            elif isinstance(event, MethodRunCompleted):
                state.completed_method_run_ids.add(event.method_run_id)
                state.method_run_outcomes[event.method_run_id] = event.outcome
                state.method_run_methods.setdefault(event.method_run_id, event.method_id)
        return state

    @classmethod
    def load(cls, paths: RunPaths, *, verify_chain: bool = True) -> "RunState":
        """Opens a run's durable log and replays it. Verifies the hash chain first by default --
        a run whose tamper-evidence does not verify is refused, never partially adopted."""
        from archivetrust.infrastructure.storage.telemetry_sink import (  # noqa: PLC0415
            FileTelemetrySink,
        )

        if not paths.events.is_file() or paths.events.stat().st_size == 0:
            # An empty file is what `FileTelemetrySink` creates at construction, before anything is
            # appended. It has no chain sidecar yet (the appender is built lazily on the first
            # append, precisely so a read-only consumer never writes one), and there is nothing to
            # replay -- so this is "no run here", not an integrity failure.
            return cls(paths=paths)
        if verify_chain:
            verification = verify_hash_chain_sidecar(paths.events)
            if not verification.ok:
                raise IntegrityError(
                    f"the telemetry hash chain for run {paths.run_id!r} did not verify "
                    f"({verification.reason}). The durable record of this run cannot be trusted, so "
                    "it will not be resumed. Its events.jsonl is untouched and available for "
                    "inspection at "
                    f"{paths.events}."
                )
        sink = FileTelemetrySink(paths.events, blob_dir=paths.blobs)
        if sink.corrupt_records:
            first = sink.corrupt_records[0]
            raise IntegrityError(
                f"run {paths.run_id!r} has {len(sink.corrupt_records)} unreadable telemetry "
                f"record(s) (first at line {first.line_number}: {first.reason}). Completed evidence "
                "cannot be reconstructed cleanly, so this run will not be resumed."
            )
        return cls.replay(paths, sink.all_events())

    # -- pending work -----------------------------------------------------------------------------

    def segmentation_identity(
        self, configuration: ResolvedConfiguration, page
    ) -> TaskIdentity:
        return TaskIdentity(
            experiment_version_id=configuration.experiment_version_id,
            method_id=SEGMENTATION_METHOD_ID,
            model_revision=configuration.segmentation_configuration_hash,
            page_or_crop_id=page.page_id,
            input_hash=page.content_hash,
            preprocessing_version=configuration.preprocessing_version,
            sampling_version=configuration.sampling_version,
            configuration_hash=configuration.configuration_hash,
        )

    def recognition_identity(
        self,
        configuration: ResolvedConfiguration,
        *,
        page_id: str,
        crop_index: int,
        crop_hash: str,
        method_id: str,
    ) -> TaskIdentity:
        return TaskIdentity(
            experiment_version_id=configuration.experiment_version_id,
            method_id=method_id,
            model_revision=configuration.model_revisions.get(method_id, "unknown"),
            page_or_crop_id=crop_slot_id(page_id, crop_index),
            input_hash=crop_hash,
            preprocessing_version=configuration.preprocessing_version,
            sampling_version=configuration.sampling_version,
            configuration_hash=configuration.configuration_hash,
        )

    def selected_crops_for_page(
        self, configuration: ResolvedConfiguration, page
    ) -> tuple[tuple[int, InputCrop], ...] | None:
        """The `(index, crop)` pairs this page's recognition tasks run over, or `None` if the page
        has not been segmented yet.

        Raises `IntegrityError` when segmentation *was* recorded but the crops it named are not
        reconstructable -- the "completed evidence can't be reconstructed cleanly" case the resume
        checks must refuse on, rather than silently re-segmenting and doubling the page's lines.
        """
        identity = self.segmentation_identity(configuration, page)
        run_id = identity.segmentation_run_id
        if run_id not in self.completed_segmentation_run_ids:
            return None
        crop_ids = self.segmentation_crop_ids.get(run_id, ())
        missing = [crop_id for crop_id in crop_ids if crop_id not in self.crops_by_id]
        if missing:
            raise IntegrityError(
                f"segmentation for page {page.page_id} is recorded as complete but "
                f"{len(missing)} of its {len(crop_ids)} InputCropCreated event(s) are absent from "
                "the log. Completed evidence cannot be reconstructed cleanly; this run will not be "
                "resumed."
            )
        crops = [self.crops_by_id[crop_id] for crop_id in crop_ids]
        indices = evenly_spaced_indices(len(crops), configuration.crops_per_page)
        return tuple((index, crops[index]) for index in indices)

    def progress(self, configuration: ResolvedConfiguration) -> "RunProgress":
        """Per-method completed counts and page completion, computed from replayed telemetry."""
        per_method = {method: 0 for method in RECOGNITION_METHOD_IDS}
        per_method_failed = {method: 0 for method in RECOGNITION_METHOD_IDS}
        pages_segmented = 0
        pages_completed = 0
        crops_known = 0
        for page in configuration.pages:
            selected = self.selected_crops_for_page(configuration, page)
            if selected is None:
                continue
            pages_segmented += 1
            crops_known += len(selected)
            page_done = True
            for index, crop in selected:
                for method in RECOGNITION_METHOD_IDS:
                    identity = self.recognition_identity(
                        configuration,
                        page_id=page.page_id,
                        crop_index=index,
                        crop_hash=crop.hash,
                        method_id=method,
                    )
                    if identity.method_run_id in self.completed_method_run_ids:
                        per_method[method] += 1
                        if self.method_run_outcomes.get(identity.method_run_id) != "succeeded":
                            per_method_failed[method] += 1
                    else:
                        page_done = False
            if page_done:
                pages_completed += 1
        return RunProgress(
            run_id=self.paths.run_id,
            experiment_run_id=self.experiment_run_id,
            started_at=self.started_at,
            sealed=self.sealed,
            seal_kind=self.seal_kind,
            pages_total=configuration.page_count,
            pages_segmented=pages_segmented,
            pages_completed=pages_completed,
            crops_expected=configuration.total_recognition_tasks,
            crops_known=crops_known,
            completed_by_method=per_method,
            failed_by_method=per_method_failed,
            event_count=self.event_count,
        )

    def rebuild_progress_cache(self, configuration: ResolvedConfiguration) -> "RunProgress":
        """Regenerates the derived display cache from telemetry. The proof that the cache is a
        cache: this method never reads it."""
        progress = self.progress(configuration)
        write_progress_cache(self.paths, progress)
        return progress


@dataclass(frozen=True)
class RunProgress:
    """A run's counts, always derived -- never accumulated in memory across a stop."""

    run_id: str
    experiment_run_id: str | None
    started_at: str | None
    sealed: bool
    seal_kind: str | None
    pages_total: int
    pages_segmented: int
    pages_completed: int
    crops_expected: int
    crops_known: int
    completed_by_method: Mapping[str, int]
    failed_by_method: Mapping[str, int]
    event_count: int

    @property
    def finished(self) -> bool:
        return self.pages_completed == self.pages_total and self.pages_total > 0

    def to_json(self) -> dict:
        return {
            "schema": PROGRESS_CACHE_SCHEMA,
            "run_id": self.run_id,
            "experiment_run_id": self.experiment_run_id,
            "started_at": self.started_at,
            "sealed": self.sealed,
            "seal_kind": self.seal_kind,
            "pages_total": self.pages_total,
            "pages_segmented": self.pages_segmented,
            "pages_completed": self.pages_completed,
            "crops_expected": self.crops_expected,
            "crops_known": self.crops_known,
            "completed_by_method": dict(self.completed_by_method),
            "failed_by_method": dict(self.failed_by_method),
            "event_count": self.event_count,
            "note": (
                "DERIVED CACHE, NOT A SOURCE OF TRUTH. Every number here is recomputed from "
                "events.jsonl by RunState.progress(); delete this file and nothing is lost."
            ),
        }


def write_progress_cache(paths: RunPaths, progress: RunProgress) -> None:
    paths.directory.mkdir(parents=True, exist_ok=True)
    paths.progress_cache.write_text(
        json.dumps(progress.to_json(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def read_progress_cache(paths: RunPaths) -> dict | None:
    """Reads the display cache if it is present and parseable, `None` otherwise.

    A corrupt cache returns `None` rather than raising: the caller's fallback is to recompute from
    telemetry, which is always correct, so a damaged cache must never be able to stop a run.
    """
    if not paths.progress_cache.is_file():
        return None
    try:
        payload = json.loads(paths.progress_cache.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    return payload if payload.get("schema") == PROGRESS_CACHE_SCHEMA else None


def compare_configurations(
    recorded: ResolvedConfiguration, current: ResolvedConfiguration
) -> dict[str, tuple[str, str]]:
    """Every named field on which the two configurations disagree, `{name: (recorded, current)}`.

    Empty means the run may proceed. Anything else is a refusal, and the caller must show these
    lines rather than a generic message.
    """
    recorded_fields = recorded.comparable_fields()
    current_fields = current.comparable_fields()
    names = list(dict.fromkeys([*recorded_fields, *current_fields]))
    return {
        name: (recorded_fields.get(name, "<absent>"), current_fields.get(name, "<absent>"))
        for name in names
        if recorded_fields.get(name) != current_fields.get(name)
    }


def assert_resumable(state: RunState, current: ResolvedConfiguration) -> None:
    """The refuse-to-silently-continue gate, run before a resume does any work.

    Three independent refusals, each with its own message:

    1. the run recorded no configuration at all (its manifest never landed) -- unreconstructable;
    2. the recorded and current configurations differ in any compared field -- including the dataset
       content fingerprint, so a changed page file is caught as a dataset mismatch specifically;
    3. its completed evidence does not reconstruct (raised out of `selected_crops_for_page`).

    Hash-chain validation and corrupt-record isolation already happened in `RunState.load`, which is
    the only supported way to obtain a `RunState` from disk.
    """
    if state.recorded_configuration is None:
        raise RunStateError(
            f"run {state.paths.run_id!r} has no recorded configuration in its telemetry "
            "(no ReproducibilityManifestRecorded carrying it). Its completed work cannot be shown "
            "to belong to any particular configuration, so it will not be resumed."
        )
    differences = compare_configurations(state.recorded_configuration, current)
    if differences:
        raise ConfigurationMismatchError(differences)
    for page in current.pages:
        state.selected_crops_for_page(current, page)
