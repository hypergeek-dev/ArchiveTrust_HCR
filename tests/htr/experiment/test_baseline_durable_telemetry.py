"""Real-execution proof that the baseline experiment's evidence outlives the process that made it.

The live counterpart of `tests/htr/persistence/test_real_baseline_reconstruction.py`. That module
replays the *committed* log of the 2026-07-30 run, which keeps the demonstration available in the
default suite on any machine. This one runs the baseline for real -- real SATRN GPU inference, real
Florence-2 GPU inference, real Transkribus PAGE XML parsing -- writes it to a real
`FileTelemetrySink` under `tmp_path`, then destroys every in-memory object and reconstructs from that
file alone. Only this version can fail if `run_baseline_comparison`'s *wiring* to the durable store
regresses, which is why both exist.

Marked `real_model` (registered in `pyproject.toml`), same convention as
`tests/htr/experiment/test_baseline_execution.py` -- skipped, not failed, when either local method's
environment is missing.

The real run happens **once**, in a module-scoped fixture. The sibling
`test_baseline_execution.py` deliberately re-runs per test because each of its tests is an independent
assertion about a fresh run; here every test interrogates one run's durable log, and re-running two
GPU models per assertion would buy nothing.
"""

from __future__ import annotations

import gc
import json

import pytest

from archivetrust.application.htr_journal import HtrJournal, causation_chain
from archivetrust.htr.corpus.models import InputCrop
from archivetrust.htr.experiment.baseline_execution import (
    CER_WER_METRIC_NAMES,
    DEFAULT_LINE_FIXTURE_IMAGE,
    build_research_report,
    build_research_report_from_store,
    evidence_from_events,
    run_baseline_comparison,
)
from archivetrust.htr.persistence import DurableHtrResearchStore
from archivetrust.htr.research_store import HtrResearchStore
from archivetrust.infrastructure.storage.telemetry_sink import FileTelemetrySink
from archivetrust.providers.florence2_htr.adapter import METHOD_ID as FLORENCE2_METHOD_ID
from archivetrust.providers.florence2_htr.facade import florence2_dependencies_available
from archivetrust.providers.satrn.adapter import METHOD_ID as SATRN_METHOD_ID
from archivetrust.providers.satrn.facade import satrn_python_available
from archivetrust.providers.transkribus.adapter import METHOD_ID as TRANSKRIBUS_METHOD_ID

pytestmark = pytest.mark.real_model

_satrn_ok, _satrn_message = satrn_python_available()
_florence2_ok, _florence2_message = florence2_dependencies_available()
requires_local_methods = pytest.mark.skipif(
    not (_satrn_ok and _florence2_ok),
    reason=f"satrn_available={_satrn_ok} ({_satrn_message}); florence2_available={_florence2_ok} ({_florence2_message})",
)


@pytest.fixture(scope="module", name="durable_run")
def _durable_run(tmp_path_factory):
    """One real baseline execution into a real file-backed telemetry sink.

    Returns everything the tests need *as plain values plus a path* -- deliberately not the store, so
    no test can accidentally reach the live projection instead of the reconstructed one.
    """
    if not (_satrn_ok and _florence2_ok):
        pytest.skip("local HTR method environments unavailable")

    directory = tmp_path_factory.mktemp("baseline_durable")
    path = directory / "htr_research_events.jsonl"

    store = DurableHtrResearchStore(FileTelemetrySink(path))
    assert store.is_durable, "this test is meaningless against a non-durable sink"
    result = run_baseline_comparison(store=store)

    # The live report, for the live-vs-reconstructed comparison below.
    live_report_metadata = json.loads(
        json.dumps(build_research_report(result).metadata, ensure_ascii=False)
    )
    live = {
        "controlled_run_id": result.controlled_run.experiment_run_id,
        "end_to_end_run_id": result.end_to_end_run.experiment_run_id,
        "crop_id": result.shared_crop.crop_id,
        "crop_hash": result.shared_crop.hash,
        "ground_truth_text": result.ground_truth_text,
        "satrn_raw": result.satrn.transcript.raw_text,
        "satrn_normalized": result.satrn.transcript.normalized_text,
        "satrn_cer": result.satrn.metrics.character_error_rate_normalized,
        "satrn_wer": result.satrn.metrics.word_error_rate_normalized,
        "satrn_method_run_id": result.satrn.method_run.method_run_id,
        "satrn_flags": {f.category for f in result.satrn.reliability_failures},
        "florence2_raw": result.florence2.transcript.raw_text,
        "florence2_parsed": result.florence2.transcript.parsed_text,
        "florence2_cer": result.florence2.metrics.character_error_rate_normalized,
        "florence2_wer": result.florence2.metrics.word_error_rate_normalized,
        "florence2_method_run_id": result.florence2.method_run.method_run_id,
        "transkribus_method_run_id": result.transkribus.method_run.method_run_id,
        "transkribus_parsed": result.transkribus.transcript.parsed_text,
        "manifest_id": result.manifest.manifest_id,
        "method_run_started_event_ids": dict(result.method_run_started_event_ids),
        "report_metadata": live_report_metadata,
        "events_on_disk": len(list(store.sink.all_events())),
    }

    # Destroy every in-memory object. Nothing below may reach the store, its sink, or the result.
    store_id = id(store)
    del store
    del result
    gc.collect()

    return path, live, store_id


@pytest.fixture(name="reconstructed")
def _reconstructed(durable_run):
    """A brand-new `FileTelemetrySink` over the same path, replayed into a fresh projection."""
    path, live, store_id = durable_run
    events = list(FileTelemetrySink(path).all_events())
    store = HtrJournal().replay(events)
    assert isinstance(store, HtrResearchStore)
    assert id(store) != store_id
    return store, events, live


@requires_local_methods
def test_the_real_run_wrote_a_real_file_with_its_hash_chain(durable_run):
    path, live, _ = durable_run
    assert path.exists()
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == live["events_on_disk"] > 50
    assert (path.parent / "htr_research_events.jsonl.chain.jsonl").exists()

    kinds = {json.loads(line)["kind"] for line in lines}
    for expected in (
        "ResearchProjectCreated",
        "DatasetCreated",
        "CollectionCreated",
        "DatasetVersionCreated",
        "ExperimentCreated",
        "ExperimentVersionCreated",
        "ExperimentRunStarted",
        "ExperimentRunCompleted",
        "PageRegistered",
        "RegionDetected",
        "TextLineDetected",
        "InputCropCreated",
        "GroundTruthTextRecorded",
        "MethodRunStarted",
        "MethodRunCompleted",
        "MethodRunFailed",
        "EvidenceCreated",
        "RawMethodResultRecorded",
        "ParsedMethodResultRecorded",
        "NormalizedMethodResultRecorded",
        "MetricDefinitionRegistered",
        "MetricCalculated",
        "ReliabilityIssueClassified",
        "ReproducibilityManifestRecorded",
    ):
        assert expected in kinds, f"{expected} was never written by the real baseline run"


@requires_local_methods
def test_both_controlled_method_runs_reconstruct_with_the_same_input_crop(reconstructed):
    """The follow-up's required demonstration, on a run whose process is gone."""
    store, _events, live = reconstructed

    satrn = store.method_run(live["satrn_method_run_id"])
    florence2 = store.method_run(live["florence2_method_run_id"])
    assert satrn is not None and florence2 is not None
    assert satrn.method_id == SATRN_METHOD_ID
    assert florence2.method_id == FLORENCE2_METHOD_ID

    assert satrn.input_crop_id == florence2.input_crop_id == live["crop_id"]
    assert satrn.experiment_run_id == florence2.experiment_run_id == live["controlled_run_id"]

    crop = store.input_crop(satrn.input_crop_id)
    assert crop is not None
    assert crop.hash == live["crop_hash"]
    assert crop.hash == InputCrop.compute_hash(DEFAULT_LINE_FIXTURE_IMAGE.read_bytes())


@requires_local_methods
def test_real_transcriptions_and_metrics_reconstruct_to_the_values_the_live_run_measured(
    reconstructed,
):
    store, _events, live = reconstructed

    satrn_transcript = store.transcript(live["satrn_method_run_id"])
    assert satrn_transcript.raw_text == live["satrn_raw"]
    assert satrn_transcript.normalized_text == live["satrn_normalized"]
    assert satrn_transcript.raw_text.strip() != ""

    florence2_transcript = store.transcript(live["florence2_method_run_id"])
    assert florence2_transcript.raw_text == live["florence2_raw"]
    assert florence2_transcript.parsed_text == live["florence2_parsed"]
    assert florence2_transcript.parsed_text.strip() != ""

    crop = store.input_crop(live["crop_id"])
    assert store.ground_truth_for_line(crop.text_line_id) == live["ground_truth_text"]

    def _metrics(method_run_id):
        out = {}
        for result in store.metric_results(method_run_id=method_run_id):
            definition = store.metric_definition(result.metric_definition_id)
            assert definition is not None
            out[definition.name] = result.value
        return out

    satrn = _metrics(live["satrn_method_run_id"])
    florence2 = _metrics(live["florence2_method_run_id"])
    assert satrn["character_error_rate_normalized"] == pytest.approx(live["satrn_cer"])
    assert satrn["word_error_rate_normalized"] == pytest.approx(live["satrn_wer"])
    assert florence2["character_error_rate_normalized"] == pytest.approx(live["florence2_cer"])
    assert florence2["word_error_rate_normalized"] == pytest.approx(live["florence2_wer"])
    assert satrn["execution_time_ms"] > 0

    # Reliability classifications, and the manifest, came back too.
    assert {
        failure.category for failure in store.failures(method_run_id=live["satrn_method_run_id"])
    } == live["satrn_flags"]
    assert [m.manifest_id for m in store.manifests(experiment_run_id=live["controlled_run_id"])] == [
        live["manifest_id"]
    ]


@requires_local_methods
def test_the_causal_chain_of_the_real_run_is_walkable_by_pointer_alone(reconstructed):
    _store, events, live = reconstructed

    chain = causation_chain(
        events, from_event_id=live["method_run_started_event_ids"][FLORENCE2_METHOD_ID]
    )
    assert [event.kind.value for event in chain][:5] == [
        "MethodRunStarted",
        "RawMethodResultRecorded",
        "ParsedMethodResultRecorded",
        "NormalizedMethodResultRecorded",
        "MetricCalculated",
    ]
    for parent, child in zip(chain, chain[1:]):
        assert child.causation_id == parent.event_id


@requires_local_methods
def test_transkribus_stays_out_of_the_controlled_correlation_and_has_no_cer(reconstructed):
    """"No invalid Transkribus CER/WER", on a real run's durable record."""
    store, events, live = reconstructed

    transkribus = store.method_run(live["transkribus_method_run_id"])
    assert transkribus.method_id == TRANSKRIBUS_METHOD_ID
    assert transkribus.input_crop_id is None
    assert transkribus.experiment_run_id == live["end_to_end_run_id"]
    assert live["end_to_end_run_id"] != live["controlled_run_id"]

    controlled_ids = {e.event_id for e in events if e.correlation_id == live["controlled_run_id"]}
    end_to_end_ids = {e.event_id for e in events if e.correlation_id == live["end_to_end_run_id"]}
    assert controlled_ids and end_to_end_ids
    assert not (controlled_ids & end_to_end_ids)

    for event in events:
        if getattr(event, "method_run_id", None) == transkribus.method_run_id:
            assert event.correlation_id == live["end_to_end_run_id"]

    names = set()
    for result in store.metric_results(method_run_id=transkribus.method_run_id):
        definition = store.metric_definition(result.metric_definition_id)
        names.add(definition.name if definition is not None else result.metric_definition_id)
    assert not (names & CER_WER_METRIC_NAMES)

    # Preserved, not discarded.
    assert store.transcript(transkribus.method_run_id).parsed_text == live["transkribus_parsed"]


@requires_local_methods
def test_the_report_regenerates_identically_from_the_reconstructed_projection(reconstructed):
    """Report generation reads durable records only -- proven by regenerating it after a replay."""
    store, events, live = reconstructed

    rebuilt = build_research_report_from_store(
        store,
        controlled_experiment_run_id=live["controlled_run_id"],
        end_to_end_experiment_run_id=live["end_to_end_run_id"],
        evidence_by_id=evidence_from_events(events),
    )
    rebuilt_metadata = json.loads(json.dumps(rebuilt.metadata, ensure_ascii=False))
    assert rebuilt_metadata == live["report_metadata"]
