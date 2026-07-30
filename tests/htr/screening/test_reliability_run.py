"""The reliability harness's guarantees, one test per guarantee.

The properties under test are the ones a user is being asked to trust when they walk away from a
multi-hour benchmark and come back tomorrow:

* a task is complete only when the durable log says so;
* a task interrupted mid-call is never complete;
* SATRN and Florence-2 resume independently;
* a changed configuration or a changed dataset refuses instead of mixing;
* the progress cache is genuinely disposable;
* resuming appends no duplicate events.
"""

from __future__ import annotations

import json

import pytest

from archivetrust.htr.screening.run_configuration import (
    FLORENCE2_METHOD_ID,
    SATRN_METHOD_ID,
    evenly_spaced_indices,
)
from archivetrust.htr.screening.run_report import build_report
from archivetrust.htr.screening.run_state import (
    ConfigurationMismatchError,
    IntegrityError,
    RunPaths,
    RunState,
    RunStateError,
    allocate_run_id,
    compare_configurations,
    discover_runs,
    find_unfinished_run,
    latest_run,
    read_progress_cache,
)
from archivetrust.htr.screening.runner import StopSignal, summarize_pending
from archivetrust.htr.screening.task_identity import TaskIdentity, crop_slot_id
from archivetrust.infrastructure.storage.integrity import verify_hash_chain_sidecar
from archivetrust.infrastructure.storage.telemetry_sink import FileTelemetrySink
from tests.htr.screening._reliability_fakes import (
    FakeRecognizer,
    build_configuration,
    detector_facade,
    make_runner,
    write_page,
)


@pytest.fixture
def repo(tmp_path):
    return tmp_path


@pytest.fixture
def configuration(repo):
    return build_configuration(repo, page_count=2, crops_per_page=2)


@pytest.fixture
def paths(repo):
    return RunPaths.for_run(repo / "runs", "reliability-test-1")


def _adapters(**kwargs):
    return {
        SATRN_METHOD_ID: kwargs.get("satrn") or FakeRecognizer(SATRN_METHOD_ID),
        FLORENCE2_METHOD_ID: kwargs.get("florence2") or FakeRecognizer(FLORENCE2_METHOD_ID),
    }


def _run(repo, paths, configuration, adapters, *, state=None, signal=None, pages=2):
    runner, store, sink = make_runner(
        repo_root=repo,
        paths=paths,
        configuration=configuration,
        adapters=adapters,
        facade=detector_facade(pages),
    )
    state = state if state is not None else RunState.load(paths)
    runner.ensure_run_registered(state)
    outcome = runner.run(state, signal or StopSignal())
    return outcome, state, sink


# -- task identity ---------------------------------------------------------------------------------


def test_task_identity_tuple_is_the_specified_eight_fields():
    identity = TaskIdentity(
        experiment_version_id="ev",
        method_id="satrn",
        model_revision="rev",
        page_or_crop_id="page#crop0",
        input_hash="crop_abc",
        preprocessing_version="1.0.0",
        sampling_version="sampling/v1",
        configuration_hash="cfg",
    )
    assert identity.identity_tuple() == (
        "ev",
        "satrn",
        "rev",
        "page#crop0",
        "crop_abc",
        "1.0.0",
        "sampling/v1",
        "cfg",
    )
    assert identity.task_key.startswith("task_")
    assert identity.method_run_id == "method_run_" + identity.task_key.split("_", 1)[1]


def test_task_identity_is_deterministic_and_field_sensitive():
    base = dict(
        experiment_version_id="ev",
        method_id="satrn",
        model_revision="rev",
        page_or_crop_id="page#crop0",
        input_hash="crop_abc",
        preprocessing_version="1.0.0",
        sampling_version="sampling/v1",
        configuration_hash="cfg",
    )
    assert TaskIdentity(**base).task_key == TaskIdentity(**base).task_key
    for field in base:
        changed = {**base, field: base[field] + "-changed"}
        assert TaskIdentity(**changed).task_key != TaskIdentity(**base).task_key, field


def test_crop_slot_id_does_not_depend_on_the_random_crop_id():
    assert crop_slot_id("page_x", 3) == "page_x#crop3"


def test_evenly_spaced_indices_never_fabricates_a_line():
    assert evenly_spaced_indices(0, 10) == ()
    assert evenly_spaced_indices(3, 10) == (0, 1, 2)
    assert len(evenly_spaced_indices(40, 10)) == 10


# -- starting a run ---------------------------------------------------------------------------------


def test_starting_a_new_run_records_scaffolding_and_completes(repo, paths, configuration):
    outcome, state, sink = _run(repo, paths, configuration, _adapters())

    assert outcome.completed is True
    assert outcome.stopped is False
    assert outcome.progress.pages_completed == 2
    assert outcome.progress.completed_by_method[SATRN_METHOD_ID] == 4
    assert outcome.progress.completed_by_method[FLORENCE2_METHOD_ID] == 4
    assert outcome.chain_verified is True

    kinds = {type(event).__name__ for event in sink.all_events()}
    assert {
        "ExperimentRunStarted",
        "ReproducibilityManifestRecorded",
        "SegmentationRunCompleted",
        "MethodRunStarted",
        "MethodRunCompleted",
        "ExperimentRunCompleted",
    } <= kinds
    assert state.sealed is True


def test_allocate_run_id_never_returns_an_existing_directory(repo):
    root = repo / "runs"
    first = allocate_run_id(root)
    (root / first).mkdir(parents=True)
    second = allocate_run_id(root)
    assert second != first
    assert not (root / second).exists()


def test_a_second_run_never_writes_into_the_first_runs_telemetry(repo, configuration):
    root = repo / "runs"
    first = RunPaths.for_run(root, allocate_run_id(root))
    _run(repo, first, configuration, _adapters())
    before = first.events.read_bytes()

    second_id = allocate_run_id(root)
    second = RunPaths.for_run(root, second_id)
    assert second.events != first.events
    _run(repo, second, configuration, _adapters())
    assert first.events.read_bytes() == before
    assert {p.run_id for p in discover_runs(root)} == {first.run_id, second_id}


# -- safe stop --------------------------------------------------------------------------------------


def test_safe_stop_after_a_crop_leaves_a_consistent_resumable_state(repo, paths, configuration):
    signal = StopSignal()
    satrn = FakeRecognizer(SATRN_METHOD_ID)

    class StopAfterTwo(FakeRecognizer):
        def recognize(self, recognition_input):
            result = super().recognize(recognition_input)
            if len(self.calls) == 2:
                signal.request("test requested stop")
            return result

    florence2 = StopAfterTwo(FLORENCE2_METHOD_ID)
    outcome, state, sink = _run(
        repo, paths, configuration, _adapters(satrn=satrn, florence2=florence2), signal=signal
    )

    assert outcome.stopped is True
    assert outcome.completed is False
    assert outcome.stop_reason == "test requested stop"
    # The crop in flight finished and was recorded; nothing beyond it ran.
    assert len(satrn.calls) == 2
    assert len(florence2.calls) == 2
    assert outcome.progress.completed_by_method[SATRN_METHOD_ID] == 2
    assert outcome.progress.completed_by_method[FLORENCE2_METHOD_ID] == 2
    assert outcome.chain_verified is True
    assert verify_hash_chain_sidecar(paths.events).ok

    reloaded = RunState.load(paths)
    assert reloaded.unfinished is True
    pending_pages, pending = summarize_pending(reloaded, configuration)
    assert pending_pages == 1
    assert pending[SATRN_METHOD_ID] == 2


def test_keyboard_interrupt_routes_through_the_same_safe_stop(repo, paths, configuration):
    satrn = FakeRecognizer(SATRN_METHOD_ID, raise_on={1}, exception=KeyboardInterrupt)
    outcome, state, sink = _run(repo, paths, configuration, _adapters(satrn=satrn))

    assert outcome.stopped is True
    assert outcome.stop_reason == "keyboard interrupt (Ctrl+C)"
    assert outcome.completed is False
    # One SATRN crop completed before the interrupt; the interrupted one did not.
    assert outcome.progress.completed_by_method[SATRN_METHOD_ID] == 1
    assert outcome.chain_verified is True
    assert RunState.load(paths).unfinished is True


def test_a_task_interrupted_mid_call_is_never_recorded_as_completed(repo, paths, configuration):
    """The structural guarantee: nothing is appended for a task until the call returns."""
    satrn = FakeRecognizer(SATRN_METHOD_ID, raise_on={0}, exception=KeyboardInterrupt)
    outcome, state, sink = _run(repo, paths, configuration, _adapters(satrn=satrn))

    assert satrn.calls  # it really was called
    assert outcome.progress.completed_by_method[SATRN_METHOD_ID] == 0

    replayed = RunState.load(paths)
    assert replayed.completed_method_run_ids == set()
    # And not even a started marker: the MethodRun could not be constructed without the result.
    kinds = [str(event.kind) for event in FileTelemetrySink(paths.events).all_events()]
    assert not any("MethodRunStarted" in kind for kind in kinds)


def test_an_unexpected_exception_is_not_swallowed_as_a_stop(repo, paths, configuration):
    satrn = FakeRecognizer(SATRN_METHOD_ID, raise_on={0}, exception=RuntimeError)
    with pytest.raises(RuntimeError):
        _run(repo, paths, configuration, _adapters(satrn=satrn))


# -- telemetry durability ---------------------------------------------------------------------------


def test_telemetry_is_flushed_and_chained_after_every_crop(repo, paths, configuration):
    sizes: list[int] = []
    signal = StopSignal()

    class Watching(FakeRecognizer):
        def recognize(self, recognition_input):
            result = super().recognize(recognition_input)
            sizes.append(paths.events.stat().st_size)
            return result

    _run(repo, paths, configuration, _adapters(satrn=Watching(SATRN_METHOD_ID)), signal=signal)
    # Strictly growing between calls: every task's events are on disk before the next one starts.
    assert sizes == sorted(sizes)
    assert len(set(sizes)) == len(sizes)
    assert verify_hash_chain_sidecar(paths.events).ok


def test_resuming_appends_no_duplicate_events(repo, paths, configuration):
    _run(repo, paths, configuration, _adapters())
    events_after_first = paths.events.read_text(encoding="utf-8")
    lines_before = len(events_after_first.splitlines())

    # A second full pass over an already-complete run: every task is skipped, nothing is appended.
    state = RunState.load(paths)
    outcome, _state, _sink = _run(repo, paths, configuration, _adapters(), state=state)
    assert outcome.tasks_executed == 0
    assert outcome.tasks_skipped == 8
    assert len(paths.events.read_text(encoding="utf-8").splitlines()) == lines_before
    assert verify_hash_chain_sidecar(paths.events).ok


# -- resume -----------------------------------------------------------------------------------------


def test_resume_after_process_termination_reconstructs_from_telemetry_alone(
    repo, paths, configuration
):
    """Literally: stop, throw away every in-memory object, reopen from the file, continue."""
    signal = StopSignal()

    class StopAfterOne(FakeRecognizer):
        def recognize(self, recognition_input):
            result = super().recognize(recognition_input)
            signal.request("stop")
            return result

    first_satrn = StopAfterOne(SATRN_METHOD_ID)
    outcome, state, sink = _run(
        repo, paths, configuration, _adapters(satrn=first_satrn), signal=signal
    )
    assert outcome.stopped
    completed_before = dict(outcome.progress.completed_by_method)
    del state, sink, outcome

    # A brand-new process would have exactly this: a path, and nothing else.
    reloaded = RunState.load(paths)
    assert reloaded.progress(configuration).completed_by_method == completed_before

    second_satrn = FakeRecognizer(SATRN_METHOD_ID)
    second_florence2 = FakeRecognizer(FLORENCE2_METHOD_ID)
    outcome2, state2, _sink2 = _run(
        repo,
        paths,
        configuration,
        _adapters(satrn=second_satrn, florence2=second_florence2),
        state=reloaded,
    )
    assert outcome2.completed is True
    # Nothing already recorded was executed a second time.
    assert outcome2.tasks_skipped == sum(completed_before.values())
    assert len(second_satrn.calls) == 4 - completed_before[SATRN_METHOD_ID]
    assert len(second_florence2.calls) == 4 - completed_before[FLORENCE2_METHOD_ID]


def test_completed_satrn_work_is_skipped_on_resume(repo, paths, configuration):
    _run(repo, paths, configuration, _adapters())
    satrn = FakeRecognizer(SATRN_METHOD_ID)
    outcome, _state, _sink = _run(
        repo, paths, configuration, _adapters(satrn=satrn), state=RunState.load(paths)
    )
    assert satrn.calls == []
    assert outcome.tasks_skipped == 8


def test_completed_florence2_work_is_skipped_on_resume(repo, paths, configuration):
    _run(repo, paths, configuration, _adapters())
    florence2 = FakeRecognizer(FLORENCE2_METHOD_ID)
    _run(repo, paths, configuration, _adapters(florence2=florence2), state=RunState.load(paths))
    assert florence2.calls == []


def test_methods_resume_independently_when_only_one_completed_a_crop(repo, paths, configuration):
    """The asymmetric case: Florence-2 finished crop X, SATRN did not. Only SATRN reruns for X."""
    signal = StopSignal()

    # SATRN is called first for each crop; interrupt it on its second call, which leaves crop 2 with
    # Florence-2 complete for crop 1 only -- an intentionally lopsided state.
    class SatrnStopsOnSecond(FakeRecognizer):
        def recognize(self, recognition_input):
            if len(self.calls) == 1:
                self.calls.append(recognition_input.input_crop_id)
                signal.request("stop before recording")
                raise KeyboardInterrupt("interrupted")
            return super().recognize(recognition_input)

    outcome, state, _sink = _run(
        repo,
        paths,
        configuration,
        _adapters(satrn=SatrnStopsOnSecond(SATRN_METHOD_ID)),
        signal=signal,
    )
    progress = outcome.progress
    assert progress.completed_by_method[SATRN_METHOD_ID] == 1
    assert progress.completed_by_method[FLORENCE2_METHOD_ID] == 1

    reloaded = RunState.load(paths)
    page = configuration.pages[0]
    selected = reloaded.selected_crops_for_page(configuration, page)
    first_index, first_crop = selected[0]
    second_index, second_crop = selected[1]

    def _done(method, index, crop):
        identity = reloaded.recognition_identity(
            configuration,
            page_id=page.page_id,
            crop_index=index,
            crop_hash=crop.hash,
            method_id=method,
        )
        return identity.method_run_id in reloaded.completed_method_run_ids

    assert _done(SATRN_METHOD_ID, first_index, first_crop)
    assert _done(FLORENCE2_METHOD_ID, first_index, first_crop)
    assert not _done(SATRN_METHOD_ID, second_index, second_crop)
    assert not _done(FLORENCE2_METHOD_ID, second_index, second_crop)

    # Now make the state genuinely asymmetric for the *second* crop by completing Florence-2 only.
    florence2_only = FakeRecognizer(FLORENCE2_METHOD_ID)
    stop_after_florence2 = StopSignal()

    class SatrnRefuses(FakeRecognizer):
        def recognize(self, recognition_input):
            stop_after_florence2.request("stop")
            raise KeyboardInterrupt("interrupted again")

    _run(
        repo,
        paths,
        configuration,
        {SATRN_METHOD_ID: SatrnRefuses(SATRN_METHOD_ID), FLORENCE2_METHOD_ID: florence2_only},
        state=reloaded,
        signal=stop_after_florence2,
    )

    third = RunState.load(paths)
    # Florence-2 is ahead of SATRN on the same crop -- the exact asymmetry under test.
    assert third.progress(configuration).completed_by_method[FLORENCE2_METHOD_ID] == 1
    assert third.progress(configuration).completed_by_method[SATRN_METHOD_ID] == 1

    satrn_final = FakeRecognizer(SATRN_METHOD_ID)
    florence2_final = FakeRecognizer(FLORENCE2_METHOD_ID)
    outcome_final, _state, _sink = _run(
        repo,
        paths,
        configuration,
        _adapters(satrn=satrn_final, florence2=florence2_final),
        state=third,
    )
    assert outcome_final.completed is True
    # Each method only ran the crops *it* was missing -- never the other's completed work.
    assert len(satrn_final.calls) == 4 - 1
    assert len(florence2_final.calls) == 4 - 1


# -- the progress cache is disposable ---------------------------------------------------------------


def test_progress_is_rebuilt_from_telemetry_when_the_cache_is_deleted(repo, paths, configuration):
    _run(repo, paths, configuration, _adapters())
    assert paths.progress_cache.is_file()
    cached = read_progress_cache(paths)

    paths.progress_cache.unlink()
    assert read_progress_cache(paths) is None

    rebuilt = RunState.load(paths).rebuild_progress_cache(configuration)
    assert paths.progress_cache.is_file()
    assert rebuilt.to_json()["completed_by_method"] == cached["completed_by_method"]
    assert rebuilt.to_json()["pages_completed"] == cached["pages_completed"]


def test_a_corrupt_progress_cache_does_not_break_status(repo, paths, configuration):
    _run(repo, paths, configuration, _adapters())
    paths.progress_cache.write_text("{not json", encoding="utf-8")
    assert read_progress_cache(paths) is None
    assert RunState.load(paths).progress(configuration).pages_completed == 2


# -- refusals ---------------------------------------------------------------------------------------


def test_resume_refuses_a_changed_configuration(repo, paths, configuration):
    from archivetrust.htr.screening.run_state import assert_resumable

    signal = StopSignal()
    signal.request("stop immediately")
    _run(repo, paths, configuration, _adapters(), signal=signal)

    changed = build_configuration(repo, page_count=2, crops_per_page=2).model_copy(
        update={"reliability_thresholds_hash": "reliability_thresholds_DIFFERENT"}
    )
    state = RunState.load(paths)
    with pytest.raises(ConfigurationMismatchError) as raised:
        assert_resumable(state, changed)
    assert "reliability thresholds" in str(raised.value)
    assert "configuration hash" in str(raised.value)


def test_resume_refuses_a_changed_model_revision(repo, paths, configuration):
    from archivetrust.htr.screening.run_state import assert_resumable

    signal = StopSignal()
    signal.request("stop immediately")
    _run(repo, paths, configuration, _adapters(), signal=signal)
    changed = configuration.model_copy(
        update={"model_revisions": {SATRN_METHOD_ID: "satrn-r2", FLORENCE2_METHOD_ID: "flo-r1"}}
    )
    with pytest.raises(ConfigurationMismatchError) as raised:
        assert_resumable(RunState.load(paths), changed)
    assert f"model revision [{SATRN_METHOD_ID}]" in str(raised.value)


def test_resume_refuses_a_changed_dataset(repo, paths, configuration):
    from archivetrust.htr.screening.run_state import assert_resumable

    signal = StopSignal()
    signal.request("stop immediately")
    _run(repo, paths, configuration, _adapters(), signal=signal)

    # The page file's bytes change -> the sampled-page content fingerprint changes.
    write_page(repo / configuration.pages[0].relative_path, seed=99)
    changed = build_configuration(repo, page_count=2, crops_per_page=2)
    changed = changed.model_copy(
        update={
            "pages": (
                configuration.pages[0].model_copy(update={"content_hash": "page_image_changed"}),
                configuration.pages[1],
            )
        }
    )
    with pytest.raises(ConfigurationMismatchError) as raised:
        assert_resumable(RunState.load(paths), changed)
    assert "sampled-page content fingerprint" in str(raised.value)


def test_resume_refuses_when_the_hash_chain_does_not_verify(repo, paths, configuration):
    _run(repo, paths, configuration, _adapters())
    lines = paths.events.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[1])
    record["actor_id"] = "tampered"
    lines[1] = json.dumps(record, separators=(",", ":"), ensure_ascii=False)
    paths.events.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(IntegrityError) as raised:
        RunState.load(paths)
    assert "hash chain" in str(raised.value)


def test_resume_refuses_when_a_recorded_crop_cannot_be_reconstructed(repo, paths, configuration):
    from archivetrust.htr.screening.run_state import assert_resumable

    _run(repo, paths, configuration, _adapters())
    state = RunState.load(paths)
    # Drop one crop from the reconstructed set, as a truncated or partially-written log would.
    run_id = next(iter(state.completed_segmentation_run_ids))
    state.crops_by_id.pop(state.segmentation_crop_ids[run_id][0])
    with pytest.raises(IntegrityError) as raised:
        assert_resumable(state, configuration)
    assert "cannot be reconstructed cleanly" in str(raised.value)


def test_resume_refuses_a_run_with_no_recorded_configuration(repo, paths, configuration):
    from archivetrust.htr.screening.run_state import assert_resumable

    _run(repo, paths, configuration, _adapters())
    state = RunState.load(paths)
    state.recorded_configuration = None
    with pytest.raises(RunStateError) as raised:
        assert_resumable(state, configuration)
    assert "no recorded configuration" in str(raised.value)


def test_compare_configurations_reports_every_differing_field(repo, configuration):
    other = configuration.model_copy(update={"crops_per_page": 5})
    differences = compare_configurations(configuration, other)
    assert "crops per page" in differences
    assert differences["crops per page"] == ("2", "5")
    assert compare_configurations(configuration, configuration) == {}


# -- status and reporting ----------------------------------------------------------------------------


def test_no_unfinished_run_is_discovered_before_anything_has_started(repo):
    assert find_unfinished_run(repo / "runs") is None
    assert latest_run(repo / "runs") is None


def test_a_stopped_run_is_discovered_as_unfinished_with_no_run_id_typed(repo, paths, configuration):
    signal = StopSignal()
    signal.request("stop immediately")
    _run(repo, paths, configuration, _adapters(), signal=signal)

    found = find_unfinished_run(paths.directory.parent)
    assert found is not None
    discovered_paths, discovered_state = found
    assert discovered_paths.run_id == paths.run_id
    assert discovered_state.unfinished is True


def test_a_completed_run_is_not_discovered_as_unfinished(repo, paths, configuration):
    """This is what makes 'Run' refuse a new run only while one is genuinely unfinished."""
    _run(repo, paths, configuration, _adapters())
    assert find_unfinished_run(paths.directory.parent) is None
    found = latest_run(paths.directory.parent)
    assert found is not None and found[1].sealed is True


def test_an_unfinished_run_with_no_pending_work_only_needs_sealing(repo, paths, configuration):
    """The 'finished but unsealed' case: the process died between the last crop and the seal."""
    _run(repo, paths, configuration, _adapters())
    # Strip the terminal marker, exactly as a kill one instruction earlier would have left it.
    lines = paths.events.read_text(encoding="utf-8").splitlines()
    kept = [
        line for line in lines if json.loads(line).get("kind") != "ExperimentRunCompleted"
    ]
    assert len(kept) == len(lines) - 1
    paths.events.write_text("\n".join(kept) + "\n", encoding="utf-8")
    paths.events.with_name(paths.events.name + ".chain.jsonl").unlink()

    state = RunState.load(paths, verify_chain=False)
    assert state.unfinished is True
    pending_pages, pending = summarize_pending(state, configuration)
    assert pending_pages == 0
    assert pending == {SATRN_METHOD_ID: 0, FLORENCE2_METHOD_ID: 0}

    satrn = FakeRecognizer(SATRN_METHOD_ID)
    outcome, sealed_state, _sink = _run(
        repo, paths, configuration, _adapters(satrn=satrn), state=state
    )
    assert satrn.calls == []
    assert outcome.completed is True
    assert sealed_state.sealed is True


def test_status_before_any_run_reports_no_run_rather_than_zeros(repo, configuration):
    assert discover_runs(repo / "runs") == ()
    state = RunState.load(RunPaths.for_run(repo / "runs", "nothing"))
    assert state.exists is False
    assert state.unfinished is False


def test_status_counts_come_from_telemetry(repo, paths, configuration):
    _run(repo, paths, configuration, _adapters())
    progress = RunState.load(paths).progress(configuration)
    assert progress.pages_total == 2
    assert progress.pages_completed == 2
    assert progress.crops_expected == 4
    assert progress.completed_by_method == {SATRN_METHOD_ID: 4, FLORENCE2_METHOD_ID: 4}
    assert progress.finished is True


def test_a_recorded_failure_is_counted_as_completed_but_reported_as_a_failure(
    repo, paths, configuration
):
    satrn = FakeRecognizer(SATRN_METHOD_ID, fail_on={0})
    outcome, _state, _sink = _run(repo, paths, configuration, _adapters(satrn=satrn))
    assert outcome.completed is True
    assert outcome.progress.completed_by_method[SATRN_METHOD_ID] == 4
    assert outcome.progress.failed_by_method[SATRN_METHOD_ID] == 1
    assert any("failed on crop" in failure for failure in outcome.failures)


def test_report_is_regenerated_from_telemetry_alone(repo, paths, configuration):
    _run(repo, paths, configuration, _adapters())
    sink = FileTelemetrySink(paths.events, blob_dir=paths.blobs)
    report = build_report(sink.all_events(), run_id=paths.run_id)

    assert report["run_id"] == paths.run_id
    assert report["sealed"] is True
    assert report["segmentation"]["pages_segmented"] == 2
    assert report["per_method"][SATRN_METHOD_ID]["runs_recorded"] == 4
    assert report["per_method"][SATRN_METHOD_ID]["succeeded"] == 4
    assert report["per_method"][FLORENCE2_METHOD_ID]["succeeded"] == 4
    assert report["configuration_hash"] == configuration.configuration_hash
    assert "No CER, WER or accuracy" in report["no_accuracy_metric"]
    # Distinct outputs over distinct inputs -- the aggregate heuristic, computed from the log.
    assert report["per_method"][SATRN_METHOD_ID]["repeated_output_across_inputs"]["outputs"] == 4
