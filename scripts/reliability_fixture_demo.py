#!/usr/bin/env python
"""Real-model demonstration of the reliability harness's stop/resume guarantee.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/reliability_fixture_demo.py

Not a unit test and not a mock. This runs **real Florence-2 line detection** on two real pages from
`dataset-rgb/`, and hands the resulting byte-identical crops to **real SATRN** (its actual isolated
subprocess, reloading its actual checkpoint per call) and **real Florence-2**. Then it:

1. starts a fresh run and stops safely after a fixed number of genuinely completed crops;
2. **exits that process entirely** -- phases 1 and 2 are separate OS processes, spawned by this
   script, so "resume after process termination" is demonstrated literally rather than simulated by
   dropping an object reference;
3. starts again, discovers the unfinished run with no run id typed, resumes it, and completes it;
4. regenerates the report from the durable telemetry alone.

Between the phases it prints the per-method completed counts read back from telemetry, and the
number of real adapter calls each phase made, which together are the proof that no completed work
was repeated: phase 2's call count is exactly the number of tasks phase 1 left pending.

Runs under the `fixture` profile (two pages, two crops per page), whose `configuration_hash` differs
from the real 60-page benchmark's -- so nothing here can be resumed into, or mistaken for, the real
run. Its telemetry lives under `fixture-demo/`, never under `full-run/`.

**This script does not, and must not, run the real 60-page benchmark.**
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from archivetrust.htr.persistence import DurableHtrResearchStore  # noqa: E402
from archivetrust.htr.screening.run_configuration import (  # noqa: E402
    RECOGNITION_METHOD_IDS,
    resolve_configuration,
    screening_directory,
)
from archivetrust.htr.screening.run_report import build_report, write_report  # noqa: E402
from archivetrust.htr.screening.run_state import (  # noqa: E402
    RunPaths,
    RunState,
    allocate_run_id,
    assert_resumable,
    discover_runs,
    read_progress_cache,
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

DEMO_ROOT = screening_directory(REPO_ROOT) / "fixture-demo"
STOP_AFTER_CROPS = 3
"""Stop after this many genuinely completed real recognition tasks. Three, not one: it must land
mid-page, with one method ahead of the other, so the resume that follows is asymmetric."""


class CountingAdapter:
    """Wraps a real adapter and counts the calls it actually made in this process."""

    def __init__(self, inner, name: str) -> None:
        self._inner = inner
        self.name = name
        self.calls = 0

    def recognize(self, recognition_input):
        self.calls += 1
        return self._inner.recognize(recognition_input)


def _counts(state: RunState, configuration) -> dict[str, int]:
    progress = state.progress(configuration)
    return {method: progress.completed_by_method.get(method, 0) for method in RECOGNITION_METHOD_IDS}


def _build(configuration, paths, observer, adapters):
    paths.directory.mkdir(parents=True, exist_ok=True)
    sink = FileTelemetrySink(paths.events, blob_dir=paths.blobs)
    store = DurableHtrResearchStore.open(sink, actor_id="reliability-fixture-demo")
    detector = Florence2LineDetectorAdapter()
    segmentation = SegmentationService(adapter=detector, store=store, crop_directory=paths.crops)
    return ReliabilityRunner(
        repo_root=REPO_ROOT,
        paths=paths,
        configuration=configuration,
        store=store,
        segmentation_service=segmentation,
        adapters=adapters,
        observer=observer,
    )


def _observer(signal: StopSignal | None, limit: int | None):
    completed = {"n": 0}

    def observe(event: RunEvent) -> None:
        if event.kind == "segmenting":
            print(f"    [segmenting] page {event.page_index}/{event.page_total}", flush=True)
        elif event.kind == "recognized":
            completed["n"] += 1
            print(f"    [{completed['n']:>2}] {event.message}", flush=True)
            if signal is not None and limit is not None and completed["n"] >= limit:
                print(f"    -- simulating the S keypress after {limit} real crops", flush=True)
                signal.request("user pressed S")

    return observe


def phase_start() -> int:
    configuration = resolve_configuration(REPO_ROOT, profile="fixture")
    run_id = allocate_run_id(DEMO_ROOT)
    paths = RunPaths.for_run(DEMO_ROOT, run_id)
    print(f"PHASE 1  (pid {__import__('os').getpid()})  new run: {run_id}")
    print(f"  configuration {configuration.configuration_hash}")
    print(f"  pages         {configuration.page_count}, crops/page {configuration.crops_per_page}")
    for page in configuration.pages:
        print(f"    {page.page_id[:28]}  {page.width}x{page.height}  {page.automatic_category}")

    signal = StopSignal()
    adapters = {
        name: CountingAdapter(adapter, name) for name, adapter in build_adapters().items()
    }
    runner = _build(configuration, paths, _observer(signal, STOP_AFTER_CROPS), adapters)
    state = RunState.load(paths)
    runner.ensure_run_registered(state)
    outcome = runner.run(state, signal)

    print(f"  stopped={outcome.stopped} reason={outcome.stop_reason!r} "
          f"completed={outcome.completed}")
    print(f"  real adapter calls this process: "
          + ", ".join(f"{n}={a.calls}" for n, a in adapters.items()))
    print(f"  chain verified: {outcome.chain_verified}")
    print(f"  completed by method (from telemetry): {_counts(state, configuration)}")
    print(f"  last safe completion point: {last_completed_point(state, configuration)}")
    (DEMO_ROOT / "demo-phase1.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "completed_by_method": _counts(state, configuration),
                "adapter_calls": {n: a.calls for n, a in adapters.items()},
                "stopped": outcome.stopped,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


def phase_resume() -> int:
    configuration = resolve_configuration(REPO_ROOT, profile="fixture")
    unfinished = [p for p in discover_runs(DEMO_ROOT) if RunState.load(p).unfinished]
    if not unfinished:
        print("PHASE 2  no unfinished run discovered -- nothing to resume")
        return 1
    paths = unfinished[-1]
    print(f"PHASE 2  (pid {__import__('os').getpid()})  discovered unfinished run: {paths.run_id}")

    state = RunState.load(paths)
    assert_resumable(state, configuration)
    before = _counts(state, configuration)
    pending_pages, pending = summarize_pending(state, configuration)
    print(f"  reconstructed from telemetry alone: {state.event_count} events")
    print(f"  completed before resume: {before}")
    print(f"  pending: {pending_pages} pages, {pending}")
    print(f"  last safe completion point: {last_completed_point(state, configuration)}")

    adapters = {
        name: CountingAdapter(adapter, name) for name, adapter in build_adapters().items()
    }
    runner = _build(configuration, paths, _observer(None, None), adapters)
    runner.ensure_run_registered(state)
    outcome = runner.run(state, StopSignal())

    after = _counts(state, configuration)
    calls = {n: a.calls for n, a in adapters.items()}
    print(f"  completed after resume: {after}")
    print(f"  real adapter calls this process: {calls}")
    print(f"  tasks skipped as already complete: {outcome.tasks_skipped}")
    print(f"  completed={outcome.completed} chain verified={outcome.chain_verified}")

    expected = {m: after[m] - before[m] for m in RECOGNITION_METHOD_IDS}
    ok = calls == expected
    print(f"  PROOF nothing was repeated: calls {calls} == newly completed {expected} -> {ok}")
    (DEMO_ROOT / "demo-phase2.json").write_text(
        json.dumps(
            {
                "run_id": paths.run_id,
                "completed_before": before,
                "completed_after": after,
                "adapter_calls": calls,
                "newly_completed": expected,
                "no_work_repeated": ok,
                "tasks_skipped": outcome.tasks_skipped,
                "completed": outcome.completed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0 if ok and outcome.completed else 1


def phase_report() -> int:
    configuration = resolve_configuration(REPO_ROOT, profile="fixture")
    runs = discover_runs(DEMO_ROOT)
    paths = runs[-1]
    print(f"PHASE 3  regenerating the report for {paths.run_id} from telemetry alone")
    paths.progress_cache.unlink(missing_ok=True)
    print(f"  progress cache deleted; read_progress_cache -> {read_progress_cache(paths)}")
    state = RunState.load(paths)
    rebuilt = state.rebuild_progress_cache(configuration)
    print(f"  rebuilt from events: pages {rebuilt.pages_completed}/{rebuilt.pages_total}, "
          f"{dict(rebuilt.completed_by_method)}")
    report_path = write_report(paths)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    print(f"  report: {report_path}")
    print(f"  sealed={report['sealed']} pages_segmented="
          f"{report['segmentation']['pages_segmented']} "
          f"lines={report['segmentation']['total_lines_detected']} "
          f"crops={report['segmentation']['total_crops_written']}")
    for method, summary in sorted(report["per_method"].items()):
        print(
            f"  {method:<14} runs={summary['runs_recorded']} ok={summary['succeeded']} "
            f"fail={summary['failed']} "
            f"median_wall={summary['timing_seconds']['median']} "
            f"distinct_ratio={summary['repeated_output_across_inputs']['distinct_ratio']}"
        )
    print(f"  {report['no_accuracy_metric']}")
    return 0


def orchestrate() -> int:
    print("=" * 78)
    print("REAL-MODEL STOP / RESUME DEMONSTRATION -- fixture profile, two real pages")
    print("=" * 78)
    for phase in ("start", "resume", "report"):
        print()
        print("-" * 78)
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), phase],
            cwd=str(REPO_ROOT),
            env={**__import__("os").environ, "PYTHONPATH": "src"},
        )
        if completed.returncode != 0:
            print(f"phase {phase!r} exited {completed.returncode}")
            return completed.returncode
        print(f"-- phase {phase!r} process exited cleanly (rc=0)")
    print()
    print("=" * 78)
    phase1 = json.loads((DEMO_ROOT / "demo-phase1.json").read_text(encoding="utf-8"))
    phase2 = json.loads((DEMO_ROOT / "demo-phase2.json").read_text(encoding="utf-8"))
    print(f"phase 1 stopped after {sum(phase1['completed_by_method'].values())} real crops "
          f"({phase1['adapter_calls']} adapter calls)")
    print(f"phase 2 resumed in a NEW process, made {phase2['adapter_calls']} adapter calls, "
          f"skipped {phase2['tasks_skipped']} already-complete tasks")
    print(f"no work repeated: {phase2['no_work_repeated']}; run completed: {phase2['completed']}")
    print("=" * 78)
    return 0 if phase2["no_work_repeated"] and phase2["completed"] else 1


def main(argv: list[str]) -> int:
    phase = argv[1] if len(argv) > 1 else None
    if phase == "start":
        return phase_start()
    if phase == "resume":
        return phase_resume()
    if phase == "report":
        return phase_report()
    if phase is None:
        return orchestrate()
    print(f"unknown phase {phase!r}; use start | resume | report, or no argument to run all three")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
