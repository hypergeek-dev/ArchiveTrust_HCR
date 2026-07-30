"""The knowledge records are checked against the run they claim to describe -- not merely parsed.

`src/archivetrust/htr/knowledge/baseline_knowledge.py` pins the 2026-07-30 baseline run's ids and
measured values as module constants, so the observations stay statements about *that* run rather than
about whatever a log file happens to contain later. That pinning is only worth anything if something
verifies it, which is this module's job: every id it references must exist in the committed run log,
and every number it quotes must equal the value that log already carries.

Also proven here: the two committed streams -- `htr_research_events.jsonl` (the run) and
`htr_knowledge_events.jsonl` (the knowledge) -- replay *together* into one projection, so an
observation's evidence links resolve against the very entities the run recorded, and the causal DAG
walks across the file boundary.

Both logs are copied to `tmp_path` before being opened: `FileTelemetrySink` is append-capable and
creates sibling directories on construction, and a test must not mutate a committed research artifact
(the same discipline `tests/htr/persistence/test_real_baseline_reconstruction.py` states).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from archivetrust.application.htr_journal import HtrJournal, causation_chain
from archivetrust.htr.knowledge import baseline_knowledge as bk
from archivetrust.htr.knowledge.models import (
    EvidenceReferenceKind,
    FindingStatus,
    ObservationType,
)
from archivetrust.infrastructure.storage.telemetry_sink import FileTelemetrySink

REPO_ROOT = Path(__file__).resolve().parents[3]
BASELINE_DIR = REPO_ROOT / "docs" / "experiments" / "baseline-comparison"
RUN_LOG = BASELINE_DIR / "htr_research_events.jsonl"
KNOWLEDGE_LOG = BASELINE_DIR / "htr_knowledge_events.jsonl"

OBSERVATION_NAMES = (
    "satrn_omitted_text",
    "satrn_confidence_disagreement",
    "florence2_relative_result",
    "gpu_memory_variability",
    "transkribus_comparability",
)
"""The keys `baseline_candidate_findings` expects, in `OBSERVATION_BUILDERS`' order."""


def _named_observations() -> dict[str, object]:
    """Freshly-built observations keyed by name -- what `baseline_candidate_findings` takes.

    Deliberately not the registered ones from the committed log: these tests check the *builders*, so
    they must not depend on the artifact they help produce.
    """
    return dict(zip(OBSERVATION_NAMES, bk.baseline_observations(), strict=True))


@pytest.fixture(name="run_records")
def _run_records() -> dict[str, object]:
    """The committed run log, read as raw JSON -- deliberately not through the event classes.

    Reading the raw lines means this fixture cannot be fooled by a model default: if an id below is
    absent from the file's actual bytes, it is absent, full stop.
    """
    with RUN_LOG.open(encoding="utf-8") as handle:
        events = [json.loads(line) for line in handle if line.strip()]

    event_ids = {event["event_id"] for event in events}
    metric_values: dict[str, float] = {}
    metric_names: dict[str, str] = {}
    definitions = {
        event["metric_definition"]["metric_definition_id"]: event["metric_definition"]["name"]
        for event in events
        if event["kind"] == "MetricDefinitionRegistered"
    }
    for event in events:
        if event["kind"] == "MetricCalculated":
            result = event["metric_result"]
            metric_values[result["metric_result_id"]] = result["value"]
            metric_names[result["metric_result_id"]] = definitions[result["metric_definition_id"]]

    return {
        "events": events,
        "event_ids": event_ids,
        "metric_values": metric_values,
        "metric_names": metric_names,
        "method_run_ids": {
            event["method_run"]["method_run_id"]
            for event in events
            if event["kind"] == "MethodRunStarted"
        },
        "experiment_run_ids": {
            event["experiment_run"]["experiment_run_id"]
            for event in events
            if event["kind"] == "ExperimentRunStarted"
        },
        "failure_record_ids": {
            event["failure_record"]["failure_record_id"]
            for event in events
            if event["kind"] == "MethodRunFailed"
        },
        "evidence_ids": {
            event["evidence"]["evidence_id"]
            for event in events
            if event["kind"] == "EvidenceCreated"
        },
        "crop_ids": {
            event["input_crop"]["crop_id"] for event in events if event["kind"] == "InputCropCreated"
        },
        "text_line_ids": {
            event["text_line"]["text_line_id"]
            for event in events
            if event["kind"] == "TextLineDetected"
        },
        "page_ids": {event["page"]["page_id"] for event in events if event["kind"] == "PageRegistered"},
    }


@pytest.fixture(name="replayed")
def _replayed(tmp_path):
    """One projection built from both committed streams, replayed off copies."""
    for source in (RUN_LOG, KNOWLEDGE_LOG):
        shutil.copy2(source, tmp_path / source.name)
    run_events = list(FileTelemetrySink(tmp_path / RUN_LOG.name).all_events())
    knowledge_events = list(FileTelemetrySink(tmp_path / KNOWLEDGE_LOG.name).all_events())
    return HtrJournal().replay([*run_events, *knowledge_events]), [*run_events, *knowledge_events]


# -- The pinned constants really describe the committed run ----------------------------------------


def test_every_pinned_entity_id_exists_in_the_committed_run_log(run_records):
    assert bk.CONTROLLED_RUN_ID in run_records["experiment_run_ids"]
    assert bk.END_TO_END_RUN_ID in run_records["experiment_run_ids"]
    for method_run_id in (
        bk.SATRN_METHOD_RUN_ID,
        bk.FLORENCE2_METHOD_RUN_ID,
        bk.TRANSKRIBUS_METHOD_RUN_ID,
    ):
        assert method_run_id in run_records["method_run_ids"]
    assert bk.INPUT_CROP_ID in run_records["crop_ids"]
    assert bk.TEXT_LINE_ID in run_records["text_line_ids"]
    assert bk.TRANSKRIBUS_PAGE_ID in run_records["page_ids"]
    assert bk.SATRN_EVIDENCE_ID in run_records["evidence_ids"]
    assert bk.FLORENCE2_EVIDENCE_ID in run_records["evidence_ids"]
    assert bk.SATRN_OMITTED_TEXT_FAILURE_ID in run_records["failure_record_ids"]
    assert bk.SATRN_CONFIDENCE_DISAGREEMENT_FAILURE_ID in run_records["failure_record_ids"]


def test_every_pinned_event_id_exists_in_the_committed_run_log(run_records):
    pinned = {
        name: value
        for name, value in vars(bk).items()
        if name.endswith("_EVENT_ID") and isinstance(value, str)
    }
    assert pinned, "expected pinned event-id constants to exist"
    missing = {name: value for name, value in pinned.items() if value not in run_records["event_ids"]}
    assert not missing, f"pinned event ids absent from the committed run log: {missing}"


def test_every_quoted_metric_value_equals_the_committed_one(run_records):
    """The numbers in the observations are quoted, never recomputed -- so they must match exactly."""
    values = run_records["metric_values"]
    names = run_records["metric_names"]
    expected = {
        bk.SATRN_CER_NORMALIZED_METRIC_ID: (
            "character_error_rate_normalized",
            bk.SATRN_CER_NORMALIZED,
        ),
        bk.SATRN_WER_NORMALIZED_METRIC_ID: ("word_error_rate_normalized", bk.SATRN_WER_NORMALIZED),
        bk.SATRN_CHARACTER_DELETIONS_METRIC_ID: (
            "character_deletions",
            bk.SATRN_CHARACTER_DELETIONS,
        ),
        bk.SATRN_WORD_DELETIONS_METRIC_ID: ("word_deletions", bk.SATRN_WORD_DELETIONS),
        bk.SATRN_GPU_MEMORY_METRIC_ID: ("gpu_memory_mb", bk.SATRN_GPU_MEMORY_MEASURED_MIB),
        bk.FLORENCE2_CER_NORMALIZED_METRIC_ID: (
            "character_error_rate_normalized",
            bk.FLORENCE2_CER_NORMALIZED,
        ),
        bk.FLORENCE2_WER_NORMALIZED_METRIC_ID: (
            "word_error_rate_normalized",
            bk.FLORENCE2_WER_NORMALIZED,
        ),
        bk.FLORENCE2_GPU_MEMORY_METRIC_ID: (
            "gpu_memory_mb",
            bk.FLORENCE2_GPU_MEMORY_MEASURED_MIB,
        ),
    }
    for metric_result_id, (metric_name, quoted) in expected.items():
        assert names[metric_result_id] == metric_name
        assert values[metric_result_id] == quoted, (
            f"{metric_result_id} is {values[metric_result_id]} in the log but baseline_knowledge "
            f"quotes {quoted}"
        )
    assert values[bk.SATRN_EXACT_WORD_ACCURACY_METRIC_ID] == pytest.approx(0.0)


def test_the_quoted_texts_and_classifications_match_the_committed_log(run_records):
    events = run_records["events"]
    by_id = {event["event_id"]: event for event in events}

    assert by_id[bk.SATRN_RAW_EVENT_ID]["text"] == bk.SATRN_OUTPUT_TEXT
    assert by_id[bk.FLORENCE2_PARSED_EVENT_ID]["text"] == bk.FLORENCE2_PARSED_TEXT
    assert by_id[bk.GROUND_TRUTH_EVENT_ID]["text"] == bk.GROUND_TRUTH_TEXT
    assert bk.TRANSKRIBUS_FIRST_LINE_TEXT in by_id[bk.TRANSKRIBUS_PARSED_EVENT_ID]["text"]

    assert by_id[bk.SATRN_OMITTED_TEXT_EVENT_ID]["classification"] == "omitted_text"
    assert by_id[bk.SATRN_OMITTED_TEXT_EVENT_ID]["detail"] == bk.SATRN_OMITTED_TEXT_DETAIL
    disagreement = by_id[bk.SATRN_CONFIDENCE_DISAGREEMENT_EVENT_ID]
    assert disagreement["classification"] == "confidence_calibration_disagreement"
    assert disagreement["detail"] == bk.SATRN_CONFIDENCE_DISAGREEMENT_DETAIL

    satrn_evidence = next(
        event["evidence"]
        for event in events
        if event["kind"] == "EvidenceCreated"
        and event["evidence"]["evidence_id"] == bk.SATRN_EVIDENCE_ID
    )
    assert satrn_evidence["provider_confidence"] == bk.SATRN_REPORTED_CONFIDENCE
    assert satrn_evidence["model_revision"] == bk.SATRN_MODEL_REVISION


def test_satrn_really_has_no_parsed_stage_in_the_committed_log(run_records):
    """`SATRN_PARSED_EVENT_ID is None` is a claim about the log, so it is checked against the log."""
    parsed_for_satrn = [
        event
        for event in run_records["events"]
        if event["kind"] == "ParsedMethodResultRecorded"
        and event.get("method_run_id") == bk.SATRN_METHOD_RUN_ID
    ]
    assert parsed_for_satrn == [], (
        "the omitted-text observation states there is no ParsedMethodResultRecorded for SATRN; if "
        "one appeared, the observation would need to link it"
    )
    assert bk.SATRN_PARSED_EVENT_ID is None


def test_no_transkribus_cer_or_wer_is_quoted_anywhere_in_the_knowledge(run_records):
    """The comparability finding rests on the *absence* of a metric, so nothing may invent one.

    Two halves, and both matter. First: the committed run log genuinely contains no CER/WER for the
    Transkribus method run (checked by owner, so a renamed metric could not slip past). Second: the
    comparability observation and finding reference no metric result at all, so neither smuggles in a
    number that was never computed.
    """
    error_rate_owners = {
        event["metric_result"]["method_run_id"]
        for event in run_records["events"]
        if event["kind"] == "MetricCalculated"
        and run_records["metric_names"][event["metric_result"]["metric_result_id"]].startswith(
            ("character_error", "word_error")
        )
    }
    assert error_rate_owners == {bk.SATRN_METHOD_RUN_ID, bk.FLORENCE2_METHOD_RUN_ID}, (
        "only the two controlled method runs may own a CER/WER in the committed log"
    )

    observation = bk.transkribus_comparability_observation()
    assert observation.evidence_ids(EvidenceReferenceKind.METRIC_RESULT) == (), (
        "the Transkribus observation must reference no metric result -- there is no id for a metric "
        "that was never computed"
    )
    findings = bk.baseline_candidate_findings(
        {**_named_observations(), "transkribus_comparability": observation}
    )
    assert findings["transkribus_not_comparable"].supporting_metrics == ()


# -- The committed knowledge artifact ---------------------------------------------------------------


def test_the_committed_knowledge_log_replays_into_five_observations_and_five_findings(replayed):
    store, _ = replayed
    observations = store.research_observations()
    assert len(observations) == 5
    assert {o.observation_type for o in observations} == {
        ObservationType.MODEL_LIMITATION,
        ObservationType.CONFIDENCE_ANOMALY,
        ObservationType.UNEXPECTED_METHOD_DISAGREEMENT,
        ObservationType.REPRODUCIBILITY_ANOMALY,
        ObservationType.EXPERIMENT_VALIDITY_BOUNDARY,
    }
    assert len(store.findings()) == 5


def test_the_committed_findings_carry_the_statuses_the_documentation_quotes(replayed):
    store, _ = replayed
    statuses = sorted(f.review_status.value for f in store.findings())
    assert statuses == [
        "Candidate",
        "Candidate",
        "Candidate",
        "Disputed",
        "Provisionally supported",
    ], (
        "three of the four specified findings stay Candidate; the Transkribus one reached "
        "Provisionally supported and the environment-reproducibility one is Disputed -- and no "
        "finding is Supported, because no second run exists to reproduce anything in"
    )
    assert store.findings(review_status="Supported") == (), (
        "a Supported finding would mean reproduction evidence from a second run, which this "
        "repository does not have"
    )


def test_every_run_scoped_evidence_reference_resolves_against_the_replayed_run(replayed):
    """The observations' evidence links are not decorative: each resolves to a real entity.

    Checked against the *replayed* projection rather than the raw file, so this also proves the two
    streams reconstruct into one graph -- an observation recorded in the knowledge log points at a
    metric result that only the run log carries, and after replay both are in the same store.
    """
    store, _ = replayed
    resolvers = {
        EvidenceReferenceKind.METRIC_RESULT: lambda i: any(
            m.metric_result_id == i for m in store.metric_results()
        ),
        EvidenceReferenceKind.METHOD_RUN: lambda i: store.method_run(i) is not None,
        EvidenceReferenceKind.EXPERIMENT_RUN: lambda i: store.experiment_run(i) is not None,
        EvidenceReferenceKind.INPUT_CROP: lambda i: store.input_crop(i) is not None,
        EvidenceReferenceKind.RELIABILITY_CLASSIFICATION: lambda i: any(
            f.failure_record_id == i for f in store.failures()
        ),
    }
    checked = 0
    for observation in store.research_observations():
        for ref in observation.supporting_evidence:
            resolver = resolvers.get(ref.kind)
            if resolver is None:
                continue
            assert resolver(ref.reference_id), (
                f"{observation.observation_id} references {ref.kind.value} {ref.reference_id}, which "
                "does not exist in the replayed projection of the run it claims to describe"
            )
            checked += 1
    assert checked >= 25, f"expected many resolvable references, only checked {checked}"


def test_a_causation_chain_walks_from_a_run_event_into_the_knowledge_stream(replayed):
    """The causal DAG crosses the file boundary, by pointer alone.

    Starting at the `ReliabilityIssueClassified` event in the *run* log, the chain reaches the
    `ResearchObservationCreated` in the *knowledge* log, then the `CandidateFindingCreated` it caused.
    No timestamp, no append order, no shared domain id is consulted -- `causation_chain` walks
    `causation_id -> event_id` only.
    """
    _, events = replayed
    chain = causation_chain(events, from_event_id=bk.SATRN_OMITTED_TEXT_EVENT_ID)
    kinds = [event.kind.value for event in chain]
    assert kinds[0] == "ReliabilityIssueClassified"
    assert "ResearchObservationCreated" in kinds
    assert "CandidateFindingCreated" in kinds
    assert kinds.index("ResearchObservationCreated") < kinds.index("CandidateFindingCreated"), (
        "an observation must precede the finding it caused -- a finding never auto-derives from an "
        "event"
    )


def test_the_disputed_finding_keeps_both_sides_readable(replayed):
    store, _ = replayed
    disputed = store.findings(review_status="Disputed")
    assert len(disputed) == 1
    finding = disputed[0]
    assert len(finding.contradictory_evidence) == 1
    contradiction = finding.contradictory_evidence[0]

    cited = store.research_observation(contradiction.source_id)
    assert cited is not None, (
        "the observation the contradiction cites must remain independently readable -- disputing a "
        "finding never removes the record that disputed it"
    )
    assert cited.observation_type is ObservationType.REPRODUCIBILITY_ANOMALY
    assert cited.unverified_hypothesis is not None
    assert [r.to_status.value for r in finding.revision_history] == ["Under review", "Disputed"]
    assert finding.superseded_by is None


def test_no_finding_in_the_committed_artifact_is_superseded(replayed):
    """Stated as a test so the honest gap cannot quietly disappear.

    `docs/knowledge-lifecycle.md` records that no real `Superseded` example exists because this
    repository has one baseline run. If a second run ever produces a superseding finding, this test
    fails and the documentation gets updated with it -- which is the point.
    """
    store, _ = replayed
    assert store.findings(review_status="Superseded") == ()
    assert all(f.superseded_by is None for f in store.findings())


# -- Phrasing discipline ----------------------------------------------------------------------------


def test_no_observation_or_finding_claims_one_method_is_better_than_another():
    """The follow-up is explicit that the relative result must never be phrased as a ranking.

    Checked mechanically because it is exactly the kind of wording that creeps back in during an edit.
    """
    forbidden = (
        "is better",
        "better than",
        "outperform",
        "superior",
        "beats",
        "worse than",
        "is worse",
        "wins",
        "the best",
    )
    named = _named_observations()
    observations = tuple(named.values())
    findings = bk.baseline_candidate_findings(named)
    texts: list[tuple[str, str]] = []
    for observation in observations:
        texts.append((observation.observation_id, observation.title))
        texts.append((observation.observation_id, observation.description))
    for finding in findings.values():
        texts.append((finding.finding_id, finding.statement))
        texts.extend((finding.finding_id, limitation) for limitation in finding.limitations)

    offenders = [
        (owner, phrase, text)
        for owner, text in texts
        for phrase in forbidden
        if phrase in text.casefold() and "no general ranking" not in text.casefold()
        and "says nothing about which method is better" not in text.casefold()
    ]
    assert not offenders, f"ranking language found in scoped records: {offenders}"


def test_the_relative_result_is_phrased_as_the_follow_ups_own_example():
    observation = bk.florence2_relative_result_observation()
    title = observation.title
    assert title.startswith("Florence-2 produced lower CER and WER than SATRN")
    assert "one shared line crop" in title
    assert bk.EXPERIMENT_VERSION_ID in title, "the exact experiment version must be in the title"


def test_the_gpu_memory_explanation_is_held_as_an_unverified_hypothesis_not_a_fact():
    observation = bk.gpu_memory_variability_observation()
    assert observation.unverified_hypothesis is not None
    assert "UNVERIFIED" in observation.unverified_hypothesis
    assert "HAS NOT MEASURED" in observation.unverified_hypothesis
    assert "allocator" not in observation.description.casefold(), (
        "the candidate cause must not appear among the factual fields, where it would be "
        "indistinguishable from the two measured figures beside it"
    )
    assert str(bk.FLORENCE2_GPU_MEMORY_MEASURED_MIB) in observation.description
    assert "3983" in observation.description


def test_the_transkribus_observation_is_typed_as_a_validity_boundary_not_a_performance_claim():
    observation = bk.transkribus_comparability_observation()
    assert observation.observation_type is ObservationType.EXPERIMENT_VALIDITY_BOUNDARY
    assert "not_a_method_performance_observation" in observation.tags
    for word in ("accuracy of", "performs well", "performs poorly"):
        assert word not in observation.description.casefold()


def test_every_finding_states_the_n_equals_1_limitation_and_no_transcription_convention():
    for name, finding in bk.baseline_candidate_findings(_named_observations()).items():
        assert finding.limitations, name
        assert any("N=1" in limitation for limitation in finding.limitations), name
        assert any(
            "No general ranking of methods" in limitation for limitation in finding.limitations
        ), name
        assert finding.transcription_convention is None, (
            f"{name}: no versioned TranscriptionConvention exists for this run's ground truth, so "
            "the field must be None rather than a fabricated id"
        )
        assert finding.review_status is FindingStatus.CANDIDATE
        assert finding.reviewer is None
        assert finding.supporting_observations, name
