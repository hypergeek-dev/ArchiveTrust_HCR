"""The follow-up's final required demonstration, on REAL baseline data rather than synthetic fixtures.

`test_read_model_reconstruction.py` proves the persistence layer works using `_fixtures.py`'s
deliberately synthetic entities. That is the right scope for a persistence-layer test, and its module
docstring says so -- but it leaves the demonstration one step short of what this follow-up actually
asks for:

    Hash-verified input crop -> SATRN and Florence-2 controlled MethodRuns -> raw method outputs ->
    parsed outputs -> CER and WER -> reliability classifications

reconstructed from durable telemetry alone. This module does that against the committed telemetry log
of the real 2026-07-30 baseline run (`docs/experiments/baseline-comparison/htr_research_events.jsonl`)
-- real SATRN GPU inference, real Florence-2 GPU inference, real Transkribus PAGE XML parsing, real
CER/WER against the real upstream Riksarkivet ground truth.

**Why it reads a committed artifact instead of re-running inference.** The demonstration this module
makes is about *reconstruction*, not about inference: re-running two GPU models to then throw their
in-memory results away and read a file would prove nothing extra about replay, while making the
proof unavailable on any machine without a GPU. The real-execution counterpart -- run for real, then
destroy and reconstruct within one test -- is
`tests/htr/experiment/test_baseline_durable_telemetry.py`, marked `real_model`. Both exist on purpose:
this one keeps the demonstration in the default suite, that one keeps it honest about a live run.

**The log is copied to `tmp_path` before being opened.** `FileTelemetrySink` is an append-capable
object that creates sibling directories on construction, and a test must not mutate a committed
research artifact. Copying the bytes changes nothing about what is being proven: the reconstruction
still runs from that file's contents and from nothing else -- no in-memory store crosses over, and
nothing is deserialized as a whole-store blob (which the follow-up brief is explicit does not count).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from archivetrust.application.htr_journal import HtrJournal, causation_chain
from archivetrust.htr.corpus.models import InputCrop
from archivetrust.htr.experiment.baseline_execution import (
    CER_WER_METRIC_NAMES,
    DEFAULT_LINE_FIXTURE_IMAGE,
    build_research_report_from_store,
    evidence_from_events,
)
from archivetrust.htr.research_store import HtrResearchStore
from archivetrust.infrastructure.storage.telemetry_sink import FileTelemetrySink
from archivetrust.providers.florence2_htr.adapter import METHOD_ID as FLORENCE2_METHOD_ID
from archivetrust.providers.satrn.adapter import METHOD_ID as SATRN_METHOD_ID
from archivetrust.providers.transkribus.adapter import METHOD_ID as TRANSKRIBUS_METHOD_ID

REPO_ROOT = Path(__file__).resolve().parents[3]
BASELINE_DIR = REPO_ROOT / "docs" / "experiments" / "baseline-comparison"
TELEMETRY_PATH = BASELINE_DIR / "htr_research_events.jsonl"
REPORT_PATH = BASELINE_DIR / "research_report.json"


@pytest.fixture(name="replayed")
def _replayed(tmp_path):
    """The real baseline run, reconstructed by `HtrJournal.replay` from its committed log alone.

    Returns `(store, events)`. The store is a plain `HtrResearchStore` -- replay's output -- built
    with no reference to the process that originally wrote the log.
    """
    if not TELEMETRY_PATH.exists():
        pytest.fail(
            f"the committed baseline telemetry log is missing: {TELEMETRY_PATH}\n"
            "It is a tracked research artifact, not generated output. Regenerate it with\n"
            "  PYTHONPATH=src .venv/Scripts/python.exe scripts/run_baseline_comparison.py --fresh\n"
            "(real GPU inference), and see docs/experiments/baseline-comparison/README.md."
        )
    copied = tmp_path / "htr_research_events.jsonl"
    shutil.copyfile(TELEMETRY_PATH, copied)

    events = list(FileTelemetrySink(copied).all_events())
    store = HtrJournal().replay(events)
    assert isinstance(store, HtrResearchStore)
    return store, events


def _runs(store: HtrResearchStore):
    """The controlled and end-to-end `ExperimentRun`s, identified by `is_end_to_end` rather than by a
    hardcoded id -- so this suite keeps working against a regenerated log (whose ids are freshly
    minted) and so the labelling itself is what gets asserted."""
    runs = store.experiment_runs()
    controlled = [run for run in runs if not run.is_end_to_end]
    end_to_end = [run for run in runs if run.is_end_to_end]
    assert len(controlled) == 1, f"expected exactly one controlled run, got {len(controlled)}"
    assert len(end_to_end) == 1, f"expected exactly one end-to-end run, got {len(end_to_end)}"
    return controlled[0], end_to_end[0]


def _metrics_by_name(store: HtrResearchStore, *, method_run_id: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for result in store.metric_results(method_run_id=method_run_id):
        definition = store.metric_definition(result.metric_definition_id)
        assert definition is not None, (
            f"MetricResult {result.metric_result_id} references MetricDefinition "
            f"{result.metric_definition_id}, which the log never registered -- a replayed metric "
            "with no resolvable name is an unreadable metric"
        )
        values[definition.name] = result.value
    return values


# --- The required demonstration, end to end -----------------------------------------------------


def test_both_controlled_method_runs_are_visible_and_share_one_hash_verified_input_crop(replayed):
    """Hash-verified input crop -> SATRN and Florence-2 controlled MethodRuns.

    The load-bearing assertion of this whole follow-up: after the process that ran the experiment is
    gone, the durable log alone still shows that two independent recognizers read *the same bytes*.
    """
    store, _ = replayed
    controlled, _end_to_end = _runs(store)

    method_runs = {run.method_id: run for run in store.method_runs(experiment_run_id=controlled.experiment_run_id)}
    assert set(method_runs) == {SATRN_METHOD_ID, FLORENCE2_METHOD_ID}, (
        "the controlled run must hold exactly the two local recognizers' MethodRuns"
    )

    satrn = method_runs[SATRN_METHOD_ID]
    florence2 = method_runs[FLORENCE2_METHOD_ID]

    # Both reference the identical InputCrop id -- reconstructed, not remembered.
    assert satrn.input_crop_id is not None
    assert satrn.input_crop_id == florence2.input_crop_id

    crop = store.input_crop(satrn.input_crop_id)
    assert crop is not None, "the shared InputCrop must be reconstructable from the log"

    # And that crop's stored content address still matches the real fixture's bytes on disk today,
    # recomputed here rather than trusted -- so the log cannot be describing a different image.
    assert crop.hash == InputCrop.compute_hash(DEFAULT_LINE_FIXTURE_IMAGE.read_bytes())
    assert crop.hash.startswith("crop_")


def test_raw_parsed_and_normalized_outputs_are_reconstructed_as_four_distinct_stages(replayed):
    """-> raw method outputs -> parsed outputs, kept distinct rather than collapsed into "the text"."""
    store, _ = replayed
    controlled, _ = _runs(store)
    by_method = {run.method_id: run for run in store.method_runs(experiment_run_id=controlled.experiment_run_id)}

    satrn_transcript = store.transcript(by_method[SATRN_METHOD_ID].method_run_id)
    florence2_transcript = store.transcript(by_method[FLORENCE2_METHOD_ID].method_run_id)
    assert satrn_transcript is not None and florence2_transcript is not None

    # SATRN's real output, reconstructed from the log.
    assert satrn_transcript.raw_text == "till den 23 Januarii"
    assert satrn_transcript.normalized_text == "till den 23 Januarii"
    # SATRN genuinely has no separate parsed stage (its adapter emits one text), so the honest record
    # is an absent stage -- never the raw text duplicated into `parsed_text` to fill the column.
    assert satrn_transcript.parsed_text is None

    # Florence-2 has all three, and its raw stage really is the undecoded special-token form.
    assert florence2_transcript.raw_text == (
        "</s><s>Be Kindsf. SAGE och Doth Hoon Minnes Coon Pilla gonger werff</s>"
    )
    assert florence2_transcript.parsed_text == (
        "Be Kindsf. SAGE och Doth Hoon Minnes Coon Pilla gonger werff"
    )
    assert florence2_transcript.normalized_text == florence2_transcript.parsed_text

    # `reviewed_text` stays None: nobody reviewed this run, and that must remain distinguishable from
    # "a reviewer agreed with the machine" (MethodRunTranscript's own docstring).
    assert satrn_transcript.reviewed_text is None
    assert florence2_transcript.reviewed_text is None


def test_ground_truth_and_cer_wer_are_reconstructed_with_their_real_measured_values(replayed):
    """-> CER and WER, against the real upstream Riksarkivet ground truth.

    The exact numbers this run measured are asserted, not merely bounds-checked: a reconstruction that
    returned *some* plausible float would satisfy `0 <= cer <= 1` while proving nothing.
    """
    store, _ = replayed
    controlled, _ = _runs(store)
    by_method = {run.method_id: run for run in store.method_runs(experiment_run_id=controlled.experiment_run_id)}

    # The ground truth is reached by walking the durable chain, not by reading the fixture:
    # MethodRun.input_crop_id -> InputCrop.text_line_id -> ground truth for that line.
    crop = store.input_crop(by_method[SATRN_METHOD_ID].input_crop_id)
    reference = store.ground_truth_for_line(crop.text_line_id)
    assert reference == "bekiendt. Säger och deth hon Minnes hoon Tuå gånger waritt"

    satrn = _metrics_by_name(store, method_run_id=by_method[SATRN_METHOD_ID].method_run_id)
    florence2 = _metrics_by_name(store, method_run_id=by_method[FLORENCE2_METHOD_ID].method_run_id)

    assert satrn["character_error_rate_raw"] == pytest.approx(0.7931034482758621)
    assert satrn["character_error_rate_normalized"] == pytest.approx(0.7931034482758621)
    assert satrn["word_error_rate_raw"] == pytest.approx(1.0)
    assert satrn["word_error_rate_normalized"] == pytest.approx(1.0)

    assert florence2["character_error_rate_raw"] == pytest.approx(0.43103448275862066)
    assert florence2["character_error_rate_normalized"] == pytest.approx(0.43103448275862066)
    assert florence2["word_error_rate_normalized"] == pytest.approx(0.9)

    # Operational metrics survived too, and are real measured values rather than fabricated zeros.
    assert satrn["execution_time_ms"] > 0
    assert satrn["gpu_memory_mb"] > 0
    assert florence2["execution_time_ms"] > 0


def test_reliability_classifications_are_reconstructed(replayed):
    """-> reliability classifications, the last link of the required chain.

    SATRN's near-total miss on this line is *classified*, durably: the log carries the two flags
    `htr/evaluation/failures.py::classify_reliability` produced, and Florence-2's clean run carries
    none -- so "no flags" and "flags not recorded" stay distinguishable.
    """
    store, events = replayed
    controlled, _ = _runs(store)
    by_method = {run.method_id: run for run in store.method_runs(experiment_run_id=controlled.experiment_run_id)}

    satrn_flags = {
        failure.category
        for failure in store.failures(method_run_id=by_method[SATRN_METHOD_ID].method_run_id)
    }
    assert satrn_flags == {"confidence_calibration_disagreement", "omitted_text"}
    assert store.failures(method_run_id=by_method[FLORENCE2_METHOD_ID].method_run_id) == ()

    # The same conclusions are also on the stream under the semantically correct evaluation-layer
    # kind, which this baseline run is the first producer of.
    classified = [event for event in events if event.kind.value == "ReliabilityIssueClassified"]
    assert {event.classification for event in classified} == satrn_flags


def test_the_whole_chain_is_walkable_by_causation_pointer_alone(replayed):
    """The chain is a real DAG in the log, not an inference from append order.

    Walked with `causation_chain`, which consults only `causation_id -> event_id` -- no timestamp, no
    file position, no shared domain id.
    """
    store, events = replayed
    controlled, _ = _runs(store)
    by_method = {run.method_id: run for run in store.method_runs(experiment_run_id=controlled.experiment_run_id)}

    started = {
        event.method_run_id: event.event_id
        for event in events
        if event.kind.value == "MethodRunStarted"
    }

    florence2_chain = causation_chain(
        events, from_event_id=started[by_method[FLORENCE2_METHOD_ID].method_run_id]
    )
    assert [event.kind.value for event in florence2_chain] == [
        "MethodRunStarted",
        "RawMethodResultRecorded",
        "ParsedMethodResultRecorded",
        "NormalizedMethodResultRecorded",
        "MetricCalculated",
        "ReproducibilityManifestRecorded",
        "ExperimentRunCompleted",
    ]

    # SATRN's chain is honestly one link shorter -- it has no parsed stage, and no event was invented
    # to make the two methods' chains look alike.
    satrn_chain = causation_chain(
        events, from_event_id=started[by_method[SATRN_METHOD_ID].method_run_id]
    )
    assert [event.kind.value for event in satrn_chain] == [
        "MethodRunStarted",
        "RawMethodResultRecorded",
        "NormalizedMethodResultRecorded",
        "MetricCalculated",
    ]

    for chain in (florence2_chain, satrn_chain):
        for parent, child in zip(chain, chain[1:]):
            assert child.causation_id == parent.event_id, (
                f"{child.kind.value} does not point at {parent.kind.value} via causation_id"
            )


def test_the_reproducibility_manifest_survived_with_its_real_captured_environment(replayed):
    store, _ = replayed
    controlled, _ = _runs(store)

    manifests = store.manifests(experiment_run_id=controlled.experiment_run_id)
    assert len(manifests) == 1
    manifest = manifests[0]

    assert manifest.git_commit is not None
    assert manifest.software_environment["cuda_available"] == "True"
    assert manifest.software_environment["torch"].startswith("2.")
    assert manifest.hardware_environment["gpu_name"]
    assert manifest.pipeline_configuration_hash is not None
    # Terminal run state, not the started one: the log's ExperimentRunCompleted advanced it.
    assert controlled.completed_at is not None


# --- The Transkribus chain stays separate --------------------------------------------------------


def test_transkribus_is_not_part_of_the_controlled_runs_correlation(replayed):
    """One of this follow-up's named testing requirements, asserted on the durable record.

    Two `ExperimentRun`s means two `correlation_id`s, and no event may belong to both -- which is what
    makes the separation a property of the log rather than of a naming convention.
    """
    store, events = replayed
    controlled, end_to_end = _runs(store)

    assert controlled.experiment_run_id != end_to_end.experiment_run_id

    controlled_ids = {
        event.event_id for event in events if event.correlation_id == controlled.experiment_run_id
    }
    end_to_end_ids = {
        event.event_id for event in events if event.correlation_id == end_to_end.experiment_run_id
    }
    assert controlled_ids and end_to_end_ids
    assert not (controlled_ids & end_to_end_ids)

    # And the Transkribus MethodRun's own events carry the end-to-end correlation, never the
    # controlled one.
    transkribus_runs = store.method_runs(experiment_run_id=end_to_end.experiment_run_id)
    assert [run.method_id for run in transkribus_runs] == [TRANSKRIBUS_METHOD_ID]
    transkribus_run_id = transkribus_runs[0].method_run_id
    transkribus_events = [
        event
        for event in events
        if getattr(event, "method_run_id", None) == transkribus_run_id
    ]
    assert transkribus_events
    for event in transkribus_events:
        assert event.correlation_id == end_to_end.experiment_run_id, (
            f"{event.kind.value} for the Transkribus method run carries correlation_id "
            f"{event.correlation_id!r}, which is not the end-to-end run's"
        )

    # It is page-level-only: no input crop, so it was never in the hash-matched set.
    assert transkribus_runs[0].input_crop_id is None


def test_no_invalid_transkribus_cer_or_wer_exists_anywhere_in_the_log(replayed):
    """"No invalid Transkribus CER/WER" -- asserted against every recorded metric, not just the report.

    `baseline_template.py`'s `exclusion_criteria` explains why: the fixture's transcribed content is a
    different, unrelated Swedish court-record phrase, so a CER between it and the controlled
    fixture's ground truth would compare two unrelated texts. This asserts the durable log actually
    obeys that, and that the absence is a genuine absence rather than a null written over a computed
    value.
    """
    store, events = replayed
    _controlled, end_to_end = _runs(store)
    transkribus_run = store.method_runs(experiment_run_id=end_to_end.experiment_run_id)[0]

    recorded = _metrics_by_name(store, method_run_id=transkribus_run.method_run_id)
    assert not (set(recorded) & CER_WER_METRIC_NAMES), (
        f"the log records CER/WER {sorted(set(recorded) & CER_WER_METRIC_NAMES)} for Transkribus, "
        "which has no ground-truth correspondence to the controlled fixture"
    )

    # Stronger: no MetricCalculated event anywhere in the stream names this method run together with
    # a CER/WER definition -- so nothing was computed and then merely left out of the projection.
    cer_wer_definition_ids = {
        event.metric_definition.metric_definition_id
        for event in events
        if event.kind.value == "MetricDefinitionRegistered"
        and event.metric_definition.name in CER_WER_METRIC_NAMES
    }
    assert cer_wer_definition_ids
    for event in events:
        if event.kind.value != "MetricCalculated":
            continue
        if event.metric_result.method_run_id != transkribus_run.method_run_id:
            continue
        assert event.metric_result.metric_definition_id not in cer_wer_definition_ids

    # But its real, non-empty parsed output *is* preserved -- excluded from the controlled comparison
    # is not the same as discarded.
    transcript = store.transcript(transkribus_run.method_run_id)
    assert transcript is not None
    assert transcript.parsed_text is not None and transcript.parsed_text.strip() != ""


# --- The committed report agrees with the committed log ------------------------------------------


def test_the_committed_report_is_reproducible_from_the_committed_log(replayed):
    """The proof that report generation really is durable-record-sourced.

    Regenerates the report from the replayed projection plus the stream's `EvidenceCreated` records,
    and asserts its metadata matches the committed `research_report.json` field for field. If
    `build_research_report_from_store` were quietly reading anything from a live process, this could
    not pass from a replay.
    """
    store, events = replayed
    controlled, end_to_end = _runs(store)

    rebuilt = build_research_report_from_store(
        store,
        controlled_experiment_run_id=controlled.experiment_run_id,
        end_to_end_experiment_run_id=end_to_end.experiment_run_id,
        evidence_by_id=evidence_from_events(events),
    )

    committed = json.loads(REPORT_PATH.read_text(encoding="utf-8"))["report"]
    # Round-tripped through JSON so tuples/floats compare on the same footing as the stored file.
    rebuilt_metadata = json.loads(json.dumps(rebuilt.metadata, ensure_ascii=False))

    assert rebuilt_metadata == committed["metadata"]
    assert rebuilt.results_summary == committed["results_summary"]
    assert rebuilt.methodology == committed["methodology"]
    assert rebuilt.dataset_description == committed["dataset_description"]
    assert list(rebuilt.experiment_run_ids) == committed["experiment_run_ids"]
    assert list(rebuilt.reproducibility_manifest_refs) == committed["reproducibility_manifest_refs"]
    # `report_id` is minted per generation and `generated_at` comes from the run, so only the latter
    # is expected to match; a differing report_id is correct, not a failure.
    assert rebuilt.generated_at == committed["generated_at"]
    assert rebuilt.report_id != committed["report_id"]
