#!/usr/bin/env python
"""ArchiveTrust HTR Reliability Test -- the interactive Run / Stop / Resume harness.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/reliability_test.py

Four choices, no flags, no run ids to remember. Start it, press 1, walk away; press `S` when you
need the machine back; start it again whenever and press 2. The benchmark picks up exactly where the
durable telemetry log says it left off.

**Everything this script decides, it decides by reading the run's append-only telemetry log.** The
`progress-cache.json` beside it exists only so this menu paints instantly; delete it and every
number below is identical, recomputed from the events. See `htr/screening/run_state.py`.

**What it runs.** Real Florence-2 line detection over the approved 60-page stratified sample from
`dataset-rgb/`, ten evenly-spaced line crops per page, each crop handed byte-identically to real
SATRN and real Florence-2, with the reference-free reliability heuristics recorded per output.
No accuracy metric is computed -- this corpus has no ground truth. Expect this to take hours:
SATRN reloads its full model in a fresh subprocess on every call (~9.7 s/crop, measured, expected,
documented in `satrn-repetition-diagnostic.md`). That cost is precisely why stopping and resuming
has to work.

**Stopping.** While a run is executing, press `S` -- no Enter needed. The crop in flight finishes,
its real outcome is recorded, and you are returned to this menu with the process still alive.
Ctrl+C does exactly the same thing through exactly the same code. A task interrupted mid-call is
never recorded as complete; it is simply re-executed next time.

The `ARCHIVETRUST_RELIABILITY_PROFILE=fixture` environment variable selects a two-page, two-crop
configuration used only to demonstrate this machinery against the real models in minutes
(`scripts/reliability_fixture_demo.py`). It hashes differently from the real run, so fixture
evidence can never be resumed into, or mistaken for, the benchmark.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from archivetrust.htr.persistence import DurableHtrResearchStore  # noqa: E402
from archivetrust.htr.screening.run_configuration import (  # noqa: E402
    RECOGNITION_METHOD_IDS,
    ResolvedConfiguration,
    active_profile,
    resolve_configuration,
)
from archivetrust.htr.screening.run_report import write_report  # noqa: E402
from archivetrust.htr.screening.run_state import (  # noqa: E402
    RunPaths,
    RunState,
    RunStateError,
    allocate_run_id,
    assert_resumable,
    find_unfinished_run,
    latest_run,
    runs_root,
)
from archivetrust.htr.screening.runner import (  # noqa: E402
    ReliabilityRunner,
    RunEvent,
    StopSignal,
    build_adapters,
    last_completed_point,
    summarize_pending,
)
from archivetrust.htr.segmentation import (  # noqa: E402
    Florence2LineDetectorAdapter,
    SegmentationService,
)
from archivetrust.infrastructure.storage.telemetry_sink import FileTelemetrySink  # noqa: E402

METHOD_LABELS = {"satrn": "SATRN", "florence2_htr": "Florence-2"}


class KeyboardStopWatcher:
    """Non-blocking `S` polling via `msvcrt` -- Windows-only, which is this environment.

    Runs on a daemon thread rather than being polled inline, for one reason that matters: a SATRN
    call blocks for ~10 seconds, and a user who presses `S` during it should have the keystroke
    *consumed* at that moment rather than leaking into the shell afterwards. The runner still only
    acts on the flag at a safe boundary; this thread just makes sure the keypress is never missed.

    Degrades honestly: where `msvcrt` is unavailable the watcher reports that `S` will not work and
    Ctrl+C remains the stop, rather than pretending to listen.
    """

    def __init__(self, signal: StopSignal) -> None:
        self._signal = signal
        self._thread: threading.Thread | None = None
        self._running = False
        try:
            import msvcrt  # noqa: PLC0415

            self._msvcrt = msvcrt
        except ImportError:
            self._msvcrt = None

    @property
    def available(self) -> bool:
        return self._msvcrt is not None

    def __enter__(self) -> "KeyboardStopWatcher":
        if self._msvcrt is not None:
            self._running = True
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _poll(self) -> None:
        while self._running:
            try:
                while self._msvcrt.kbhit():
                    key = self._msvcrt.getwch()
                    if key.lower() == "s":
                        self._signal.request("user pressed S")
            except Exception:  # noqa: BLE001 -- a console read must never kill a running benchmark
                return
            time.sleep(0.05)


def _elapsed(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


_find_unfinished = find_unfinished_run
_latest_run = latest_run
"""Both predicates live in `htr/screening/run_state.py`, where they are unit-tested; this script is
deliberately only the terminal in front of them. "Unfinished" is defined there: an
`ExperimentRunStarted` with no `ExperimentRunCompleted` and no `ExperimentRunFailed`."""


def print_status(configuration: ResolvedConfiguration, root: Path) -> None:
    found = _find_unfinished(root) or _latest_run(root)
    if found is None:
        print("Status: No run started yet")
        print(f"        {configuration.page_count} pages configured, "
              f"{configuration.crops_per_page} crops per page, "
              f"{configuration.total_recognition_tasks} crops per method when complete")
        return
    paths, state = found
    reference = state.recorded_configuration or configuration
    progress = state.progress(reference)
    print(
        f"Status: {progress.pages_completed} of {progress.pages_total} pages completed"
        f"   [run {paths.run_id}"
        + ("" if not state.sealed else f", {state.seal_kind}")
        + "]"
    )
    for method in RECOGNITION_METHOD_IDS:
        label = METHOD_LABELS.get(method, method)
        print(
            f"{label}: {progress.completed_by_method.get(method, 0)} of "
            f"{progress.crops_expected} crops completed"
        )


def print_configuration(configuration: ResolvedConfiguration, paths: RunPaths) -> None:
    print()
    print("Resolved configuration")
    print("-" * 72)
    print(f"  profile                     {configuration.profile}")
    print(f"  dataset                     {configuration.dataset_root_relative_path}/ "
          f"({configuration.corpus_page_count} pages, digest "
          f"{configuration.corpus_digest_sha256[:16]}...)")
    print(f"  sample                      {configuration.page_count} pages, "
          f"{configuration.sampling_version}")
    print(f"  crops per page              {configuration.crops_per_page} "
          f"({configuration.crop_selection_rule})")
    print(f"  methods                     {', '.join(configuration.method_ids)}")
    for method, revision in sorted(configuration.model_revisions.items()):
        print(f"    {method:<24}  {revision}")
    print(f"  segmentation                {configuration.segmentation_adapter_name}")
    print(f"                              {configuration.segmentation_configuration_hash[:40]}...")
    print(f"  RGB normalization           v{configuration.preprocessing_version} "
          f"({configuration.preprocessing_configuration_hash[:24]}...)")
    print(f"  reliability heuristics      v{configuration.reliability_heuristics_version} "
          f"({configuration.reliability_thresholds_hash[:24]}...)")
    print(f"  configuration hash          {configuration.configuration_hash}")
    print(f"  telemetry                   {paths.events}")
    print(f"  crops / blobs               {paths.crops}")
    print("-" * 72)
    print("  No accuracy metric is computed: this corpus has no ground truth.")
    print("  Segmentation is a Florence-2-family model -- the recorded, un-neutralized confound.")


def _confirm(prompt: str) -> bool:
    try:
        answer = input(f"{prompt} [y/N]: ").strip().lower()
    except EOFError:
        return False
    return answer in {"y", "yes"}


def _make_runner(
    configuration: ResolvedConfiguration, paths: RunPaths, observer
) -> tuple[ReliabilityRunner, object]:
    paths.directory.mkdir(parents=True, exist_ok=True)
    sink = FileTelemetrySink(paths.events, blob_dir=paths.blobs)
    store = DurableHtrResearchStore.open(sink, actor_id="reliability-test-harness")
    detector = Florence2LineDetectorAdapter()
    segmentation = SegmentationService(
        adapter=detector, store=store, crop_directory=paths.crops
    )
    runner = ReliabilityRunner(
        repo_root=REPO_ROOT,
        paths=paths,
        configuration=configuration,
        store=store,
        segmentation_service=segmentation,
        adapters=build_adapters(),
        observer=observer,
    )
    return runner, store


def _execute(configuration: ResolvedConfiguration, paths: RunPaths, state: RunState) -> None:
    signal = StopSignal()
    started = time.monotonic()

    def observer(event: RunEvent) -> None:
        if event.kind == "segmenting":
            print(f"\nPage {event.page_index}/{event.page_total} -- {event.message}")
            return
        if event.kind != "recognizing" or event.progress is None:
            return
        progress = event.progress
        print()
        print(f"Page {event.page_index}/{event.page_total}")
        print(f"Crop {event.crop_index}/{event.crop_total}")
        for method in RECOGNITION_METHOD_IDS:
            print(
                f"{METHOD_LABELS.get(method, method)}: "
                f"{progress.completed_by_method.get(method, 0)}/{progress.crops_expected}"
            )
        print(f"Elapsed: {_elapsed(time.monotonic() - started)}")
        print()
        print("Press S to stop safely.")

    runner, _store = _make_runner(configuration, paths, observer)
    runner.ensure_run_registered(state)
    with KeyboardStopWatcher(signal) as watcher:
        if not watcher.available:
            print("(msvcrt unavailable: press Ctrl+C to stop safely -- same guarantees.)")
        outcome = runner.run(state, signal)

    print()
    print("=" * 72)
    if outcome.stopped:
        print(f"STOPPED SAFELY -- {outcome.stop_reason}")
    elif outcome.completed:
        print("RUN COMPLETE")
    else:
        print("RUN PAUSED (no work executed)")
    progress = outcome.progress
    if progress is not None:
        print(f"  completed : {progress.pages_completed}/{progress.pages_total} pages")
        for method in RECOGNITION_METHOD_IDS:
            print(
                f"              {METHOD_LABELS.get(method, method)} "
                f"{progress.completed_by_method.get(method, 0)}/{progress.crops_expected} crops "
                f"({progress.failed_by_method.get(method, 0)} recorded as failures)"
            )
    pending_pages, pending = summarize_pending(state, configuration)
    print(f"  remaining : {pending_pages} pages with work left")
    for method, count in pending.items():
        print(f"              {METHOD_LABELS.get(method, method)} {count} crops")
    print(f"  executed this session: {outcome.tasks_executed} tasks "
          f"({outcome.tasks_skipped} already complete, skipped)")
    print(f"  elapsed this session : {_elapsed(outcome.elapsed_seconds)}")
    print(f"  telemetry hash chain : {'verified' if outcome.chain_verified else 'NOT VERIFIED'}")
    for failure in outcome.failures[:10]:
        print(f"  ! {failure}")
    report = write_report(paths)
    print(f"  report               : {report}")
    print("=" * 72)


def do_run(configuration: ResolvedConfiguration, root: Path) -> None:
    unfinished = _find_unfinished(root)
    if unfinished is not None:
        paths, _state = unfinished
        print()
        print(f"An unfinished run already exists: {paths.run_id}")
        print("Choose 2 (Resume) to continue it. A new run is refused while one is unfinished, so")
        print("no run can be silently abandoned halfway and no telemetry file is ever overwritten.")
        return
    run_id = allocate_run_id(root)
    paths = RunPaths.for_run(root, run_id)
    print_configuration(configuration, paths)
    print()
    print(f"New run id: {run_id}")
    if not _confirm("Start this run?"):
        print("Not started.")
        return
    _execute(configuration, paths, RunState(paths=paths))


def do_resume(configuration: ResolvedConfiguration, root: Path) -> None:
    unfinished = _find_unfinished(root)
    if unfinished is None:
        print()
        print("No unfinished run to resume. Choose 1 to start one.")
        return
    paths, state = unfinished
    try:
        assert_resumable(state, configuration)
    except RunStateError as error:
        print()
        print("REFUSING TO RESUME")
        print("-" * 72)
        print(error)
        return
    progress = state.progress(configuration)
    pending_pages, pending = summarize_pending(state, configuration)
    print()
    print(f"Resuming run {paths.run_id}")
    print("-" * 72)
    print(f"  started              {state.started_at}")
    print(f"  pages prepared       {progress.pages_segmented}/{progress.pages_total} segmented, "
          f"{progress.pages_completed} fully complete")
    for method in RECOGNITION_METHOD_IDS:
        print(
            f"  {METHOD_LABELS.get(method, method):<20} "
            f"{progress.completed_by_method.get(method, 0)}/{progress.crops_expected} crops "
            f"completed, {pending[method]} pending"
        )
    print(f"  last safe completion {last_completed_point(state, configuration) or 'none'}")
    print(f"  telemetry events     {progress.event_count}")
    print(f"  configuration        matches ({configuration.configuration_hash[:32]}...)")
    print("-" * 72)
    if pending_pages == 0:
        print("Every task is already complete; this run only needs sealing.")
    if not _confirm("Continue this run?"):
        print("Not resumed.")
        return
    _execute(configuration, paths, state)


def do_status(configuration: ResolvedConfiguration, root: Path) -> None:
    print()
    print_status(configuration, root)
    found = _find_unfinished(root) or _latest_run(root)
    if found is None:
        return
    paths, state = found
    reference = state.recorded_configuration or configuration
    pending_pages, pending = summarize_pending(state, reference)
    print(f"  run directory        {paths.directory}")
    print(f"  telemetry events     {state.event_count}")
    print(f"  remaining            {pending_pages} pages, "
          + ", ".join(
              f"{METHOD_LABELS.get(m, m)} {c}" for m, c in pending.items()
          ))
    if state.recorded_configuration is not None:
        differs = (
            state.recorded_configuration.configuration_hash != configuration.configuration_hash
        )
        print(
            "  configuration        "
            + ("DIFFERS from current -- resume would refuse" if differs else "matches current")
        )


def main() -> int:
    print()
    print("ArchiveTrust HTR Reliability Test")
    print()
    profile = active_profile()
    if profile != "full":
        print(f"!! profile = {profile!r} (not the real benchmark)")
        print()
    try:
        configuration = resolve_configuration(REPO_ROOT, profile=profile)
    except Exception as error:  # noqa: BLE001 -- the message is the whole point
        print("Cannot resolve the run configuration:")
        print(f"  {error}")
        return 1
    root = runs_root(REPO_ROOT)

    while True:
        print()
        try:
            print_status(configuration, root)
        except RunStateError as error:
            print("Status unavailable:")
            print(f"  {error}")
        print()
        print("1. Run")
        print("2. Resume")
        print("3. Show status")
        print("0. Exit")
        print()
        try:
            choice = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if choice == "1":
            do_run(configuration, root)
        elif choice == "2":
            do_resume(configuration, root)
        elif choice == "3":
            do_status(configuration, root)
        elif choice == "0":
            return 0
        else:
            print("Choose 1, 2, 3 or 0.")


if __name__ == "__main__":
    raise SystemExit(main())
