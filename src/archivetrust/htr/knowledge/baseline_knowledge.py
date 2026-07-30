"""The real research knowledge extracted from the committed 2026-07-30 baseline run.

**Every id and every number in this module was read out of
`docs/experiments/baseline-comparison/htr_research_events.jsonl`** -- the committed, hash-chained log
of one real execution (real SATRN GPU inference, real Florence-2 GPU inference, real Transkribus PAGE
XML import, real CER/WER against the real upstream Riksarkivet ground truth). Nothing here is a
placeholder, and nothing is recomputed: the metric values are quoted from the `MetricCalculated`
events that already carry them, and the reliability classifications from the
`ReliabilityIssueClassified` events that already carry theirs.
`tests/htr/knowledge/test_baseline_knowledge.py` re-verifies every id and value against that log, so
a drift between this module and the run it describes fails a test rather than sitting undetected.

**Why the ids are module constants rather than parsed from the log at import time.** These
observations are statements about *one specific historical run*. Parsing "the log" would silently
re-point them at whatever run a future file happens to contain, which is the opposite of what a
scoped observation means. The constants pin them; the test proves the pinning is accurate.

**The knowledge stream is a separate file from the run's log.** These records are appended to
`docs/experiments/baseline-comparison/htr_knowledge_events.jsonl`, not to
`htr_research_events.jsonl`, for two reasons. First, that log is a committed research artifact whose
own README documents it as exactly "100 append-only telemetry events, 25 distinct kinds" and whose
reconstruction is asserted by `tests/htr/persistence/test_real_baseline_reconstruction.py`; appending
to it would falsify that description of the run. Second, `docs/architecture/htr-telemetry.md` §7
already establishes "two streams, one mechanism" as this codebase's answer when two event populations
have different readers. The `causation_id`s below therefore point *across* files into the run's log,
which every `EvidenceReference` records via its `stream` field, and
`HtrJournal().replay(chain(run_events, knowledge_events))` reconstructs the whole graph together --
proven by `test_observations_resolve_against_the_replayed_run`.
"""

from __future__ import annotations

from archivetrust.htr.knowledge.models import (
    ContradictionSourceKind,
    ContradictoryEvidence,
    EvidenceReference,
    EvidenceReferenceKind,
    FindingConfidence,
    FindingStatus,
    HypothesisRelationship,
    ObservationConfidence,
    ObservationType,
    ResearchFinding,
    ResearchObservation,
    ResearchScope,
    ScopeUnit,
)

# -- Real ids from the committed run --------------------------------------------------------------

RUN_STREAM = "docs/experiments/baseline-comparison/htr_research_events.jsonl"
"""The stream every `EvidenceReference` below resolves against."""

KNOWLEDGE_STREAM = "docs/experiments/baseline-comparison/htr_knowledge_events.jsonl"

PROJECT_ID = "research_project_e5b85512e64140cfa6ecbbe7692001ff"
DATASET_ID = "dataset_014079548fb34741b8b3334bbb7587f7"
DATASET_VERSION_ID = "dataset_version_35afef34f88d4d9a8a8bc4ed0295dc6e"
EXPERIMENT_ID = "experiment_fa667e22afe241b8b27f9aa3998edb0f"
EXPERIMENT_VERSION_ID = "experiment_version_088916c194ad4aecbb5b8578cb405dc3"

CONTROLLED_RUN_ID = "experiment_run_30ba2bcc18a14c06af0f9ca291442cf8"
"""`is_end_to_end=false` -- the hash-matched, line-level comparison."""
END_TO_END_RUN_ID = "experiment_run_30ef72f1ca494a8e8eebf0b4fde33d73"
"""`is_end_to_end=true` -- Transkribus's page-level run, deliberately outside the controlled set."""

TEXT_LINE_ID = "text_line_aa805ef122b54d1e961e0f6ec11e266e"
INPUT_CROP_ID = "input_crop_8f36c0e77f914386ad917c09d5ef9947"
INPUT_CROP_HASH = "crop_e59f301d0763fab60e0141b8d984e88adc764b955c34e6b9490a316cf3a48261"

GROUND_TRUTH_TEXT = "bekiendt. Säger och deth hon Minnes hoon Tuå gånger waritt"
GROUND_TRUTH_EVENT_ID = "event_cc9e4261bca841c69565b4e30552c977"

SATRN_METHOD_ID = "satrn"
SATRN_METHOD_RUN_ID = "method_run_5bd842b5b850442daa0d701b63c3e545"
SATRN_MODEL_REVISION = "a40c7093232eaa47a83ce6469fc4abd033486bdc"
SATRN_EVIDENCE_ID = "evidence_82cd7c1a5005828f106c305d381ab97ba735941e56219ff47abdb345697b1aa7"
SATRN_OUTPUT_TEXT = "till den 23 Januarii"
SATRN_REPORTED_CONFIDENCE = 0.6666051723062992
SATRN_RAW_EVENT_ID = "event_b53e520f7a8745c4ab7c32772e07a888"
SATRN_NORMALIZED_EVENT_ID = "event_5c24fe2d39da4dfd9e8feb1f0772816a"
SATRN_PARSED_EVENT_ID = None
"""**There is no `ParsedMethodResultRecorded` for SATRN in the run's log, and this records that
honestly rather than pointing at the normalized event instead.** The SATRN adapter has no separate
parsed stage, so no such event was emitted (README: "SATRN emits no `ParsedMethodResultRecorded`
because it has no separate parsed stage"). An `EvidenceReference` to a parsed stage that does not
exist would be an unresolvable id."""
SATRN_CER_NORMALIZED_METRIC_ID = "metric_result_40f23f7723b54c6f92ab4d2d14c6b85a"
SATRN_CER_NORMALIZED_EVENT_ID = "event_69f2a7b3ebd94930b1788dd1cdf72640"
SATRN_CER_NORMALIZED = 0.7931034482758621
SATRN_WER_NORMALIZED_METRIC_ID = "metric_result_16be0b552e5844c7b1a9f1db0fffd031"
SATRN_WER_NORMALIZED_EVENT_ID = "event_464494d41b624082baaffe2070dc8b71"
SATRN_WER_NORMALIZED = 1.0
SATRN_CHARACTER_DELETIONS_METRIC_ID = "metric_result_fc94177ea53e4a969b8a1fc5907aabc1"
SATRN_CHARACTER_DELETIONS = 38.0
SATRN_WORD_DELETIONS_METRIC_ID = "metric_result_3b006aa6bce2425c8826cc9f260b7eda"
SATRN_WORD_DELETIONS = 6.0
SATRN_EXACT_WORD_ACCURACY_METRIC_ID = "metric_result_d462f57778f84cbc97cacc3805aa56a0"
SATRN_OMITTED_TEXT_EVENT_ID = "event_aca5554290d149fe85d9332ec32e3802"
SATRN_OMITTED_TEXT_FAILURE_ID = "failure_record_c17788ac55564906921a9d3a393183b8"
SATRN_OMITTED_TEXT_DETAIL = "6 deleted words out of 10 reference words"
SATRN_CONFIDENCE_DISAGREEMENT_EVENT_ID = "event_b8cb85b68ae94a41811c737c85470e6e"
SATRN_CONFIDENCE_DISAGREEMENT_FAILURE_ID = "failure_record_544c359478124a9cb401b57798641c14"
SATRN_CONFIDENCE_DISAGREEMENT_DETAIL = (
    "reported confidence 0.667 vs. measured similarity 0.207 against ground truth "
    "(disagreement 0.460 > threshold 0.35)"
)

FLORENCE2_METHOD_ID = "florence2_htr"
FLORENCE2_METHOD_RUN_ID = "method_run_e3209c63bb90458ea8d7a0673dd5c706"
FLORENCE2_MODEL_REVISION = "994f47e8a0e8d77cb2e11528665efd07a855c3af"
FLORENCE2_EVIDENCE_ID = "evidence_6f80d565e3bf614b65066302960d35615f2e3df2e529987ba8f888fb28c76cbc"
FLORENCE2_PARSED_TEXT = "Be Kindsf. SAGE och Doth Hoon Minnes Coon Pilla gonger werff"
FLORENCE2_RAW_EVENT_ID = "event_f392d2960f5145cb8e41840df67bf981"
FLORENCE2_PARSED_EVENT_ID = "event_f76ae5e0525e4d51ae9624a757135cb4"
FLORENCE2_NORMALIZED_EVENT_ID = "event_2cfd82d7004c4a37a468a09a908e9b87"
FLORENCE2_CER_NORMALIZED_METRIC_ID = "metric_result_25048768c1024dbf8bf9b0e1c91f598e"
FLORENCE2_CER_NORMALIZED_EVENT_ID = "event_ee1c4a31acef40849146a053a9120182"
FLORENCE2_CER_NORMALIZED = 0.43103448275862066
FLORENCE2_WER_NORMALIZED_METRIC_ID = "metric_result_ddbf925f834f4c1483f5708384e9a0b7"
FLORENCE2_WER_NORMALIZED_EVENT_ID = "event_113b8c0cda8c42d0868af5a84b36306d"
FLORENCE2_WER_NORMALIZED = 0.9
FLORENCE2_GPU_MEMORY_METRIC_ID = "metric_result_6b3a191d07074ddebf21e2bf9ef27894"
FLORENCE2_GPU_MEMORY_EVENT_ID = "event_2301532626da4c4da50acb91957ad9ea"
FLORENCE2_GPU_MEMORY_MEASURED_MIB = 1210.64111328125
FLORENCE2_GPU_MEMORY_README_MB = 3983.0
"""`src/archivetrust/providers/florence2_htr/README.md`: "Peak GPU memory: ~3983 MB". A real,
committed measurement from a real earlier CUDA session -- which has **no** durable telemetry record,
because it predates the persistence this project now has. That absence is part of what the
GPU-memory observation records."""
FLORENCE2_README_REF = "src/archivetrust/providers/florence2_htr/README.md#peak-gpu-memory"

SATRN_GPU_MEMORY_METRIC_ID = "metric_result_11954d828f2742059dfd81a324347008"
SATRN_GPU_MEMORY_MEASURED_MIB = 438.78173828125
SATRN_GPU_MEMORY_README_MB = 439.0
"""`src/archivetrust/providers/satrn/README.md`: "Peak GPU memory: ~439 MB" -- which *did* reproduce
(438.78 MiB measured). Recorded here because it is the control that makes the Florence-2 divergence a
real anomaly rather than a general property of GPU-memory measurement in this repository."""

TRANSKRIBUS_METHOD_ID = "transkribus_swedish_lion_1"
TRANSKRIBUS_METHOD_RUN_ID = "method_run_fa544c4adb9d44d89439c8e85d1b8565"
TRANSKRIBUS_MODEL_VERSION = "Swedish Lion I - v3"
TRANSKRIBUS_METHOD_RUN_EVENT_ID = "event_ee5448b140e14320b256028fb86fee03"
TRANSKRIBUS_PARSED_EVENT_ID = "event_cd394580328a4189a005cf8aea842da5"
TRANSKRIBUS_NORMALIZED_EVENT_ID = "event_ec65ed1b1a1746959b91965739a2a1a7"
TRANSKRIBUS_PAGE_ID = "page_68aa5147bfd544d9816ecb5e9f3f0d74"
TRANSKRIBUS_FIRST_LINE_TEXT = "Anno 1712 den 3 Januarii holltes ting"
TRANSKRIBUS_FIXTURE_REF = "tests/fixtures/transkribus/sample_page.xml"
BASELINE_TEMPLATE_REF = "src/archivetrust/htr/experiment/baseline_template.py#exclusion_criteria"

NORMALIZATION_PROFILE = (
    "evaluation.metrics.normalize_text v1 (Unicode NFC composition + whitespace-run collapse, "
    "no case folding)"
)
"""Quoted from `MetricDefinition.description` for `character_error_rate_normalized` v1
(`metric_definition_d7340dde39be4f0f8ca3c150a90c2845`). Named in every scope below because a CER is
not comparable across normalization profiles, so a scope that omits it is under-specified."""

EXTRACTED_AT = "2026-07-30T03:30:00+00:00"
"""When this knowledge was extracted from the run's log -- after the run (which ended
`2026-07-30T00:39:48.109810+00:00`), not during it. Passed rather than read from a clock so the
committed knowledge artifact is byte-reproducible."""

EXTRACTION_COMPONENT = "htr.knowledge.baseline_knowledge"
"""`author_or_source_component` for all five observations. An automated extraction step reading
durable records, honestly named as one -- not a researcher's byline."""


# -- Scopes --------------------------------------------------------------------------------------


def _controlled_scope(
    *,
    method_ids: tuple[str, ...],
    model_version_ids: tuple[str, ...],
    method_run_ids: tuple[str, ...],
) -> ResearchScope:
    """The controlled comparison's scope: one hash-verified line crop, one run, one configuration."""
    return ResearchScope(
        experiment_id=EXPERIMENT_ID,
        experiment_version_id=EXPERIMENT_VERSION_ID,
        experiment_run_ids=(CONTROLLED_RUN_ID,),
        unit_of_analysis=ScopeUnit.LINE_CROP,
        covered_unit_ids=(INPUT_CROP_ID,),
        dataset_id=DATASET_ID,
        dataset_version_id=DATASET_VERSION_ID,
        method_ids=method_ids,
        model_version_ids=model_version_ids,
        method_run_ids=method_run_ids,
        normalization_profile=NORMALIZATION_PROFILE,
        ground_truth_ref=TEXT_LINE_ID,
        configuration_ref=EXPERIMENT_VERSION_ID,
    )


SATRN_SCOPE = _controlled_scope(
    method_ids=(SATRN_METHOD_ID,),
    model_version_ids=(SATRN_MODEL_REVISION,),
    method_run_ids=(SATRN_METHOD_RUN_ID,),
)

COMPARISON_SCOPE = _controlled_scope(
    method_ids=(SATRN_METHOD_ID, FLORENCE2_METHOD_ID),
    model_version_ids=(SATRN_MODEL_REVISION, FLORENCE2_MODEL_REVISION),
    method_run_ids=(SATRN_METHOD_RUN_ID, FLORENCE2_METHOD_RUN_ID),
)

FLORENCE2_ENVIRONMENT_SCOPE = _controlled_scope(
    method_ids=(FLORENCE2_METHOD_ID,),
    model_version_ids=(FLORENCE2_MODEL_REVISION,),
    method_run_ids=(FLORENCE2_METHOD_RUN_ID,),
)

TRANSKRIBUS_SCOPE = ResearchScope(
    experiment_id=EXPERIMENT_ID,
    experiment_version_id=EXPERIMENT_VERSION_ID,
    experiment_run_ids=(END_TO_END_RUN_ID,),
    unit_of_analysis=ScopeUnit.PAGE,
    covered_unit_ids=(TRANSKRIBUS_PAGE_ID,),
    dataset_id=DATASET_ID,
    dataset_version_id=DATASET_VERSION_ID,
    method_ids=(TRANSKRIBUS_METHOD_ID,),
    model_version_ids=(TRANSKRIBUS_MODEL_VERSION,),
    method_run_ids=(TRANSKRIBUS_METHOD_RUN_ID,),
    normalization_profile=NORMALIZATION_PROFILE,
    ground_truth_ref=None,
    configuration_ref=BASELINE_TEMPLATE_REF,
)
"""The end-to-end run's scope. `ground_truth_ref=None` is the honest value: this page-level fixture
has no ground truth at all, which is the whole substance of the comparability observation."""


def _ref(
    kind: EvidenceReferenceKind,
    reference_id: str,
    note: str | None = None,
    *,
    stream: str | None = RUN_STREAM,
) -> EvidenceReference:
    return EvidenceReference(kind=kind, reference_id=reference_id, note=note, stream=stream)


# -- The five real observations -------------------------------------------------------------------


def satrn_omitted_text_observation(*, at: str = EXTRACTED_AT) -> ResearchObservation:
    """Observation 1: SATRN omitted most of the reference text on the shared crop.

    Typed `MODEL_LIMITATION` rather than `RECURRING_RECOGNITION_FAILURE`: exactly one occurrence is
    recorded, and "recurring" would assert a recurrence no evidence in this repository establishes.
    """
    return ResearchObservation.create(
        observation_type=ObservationType.MODEL_LIMITATION,
        title=(
            "SATRN omitted 6 of 10 reference words on input crop "
            f"{INPUT_CROP_ID} in experiment run {CONTROLLED_RUN_ID}"
        ),
        description=(
            f"On the one hash-verified shared line crop ({INPUT_CROP_ID}, content hash "
            f"{INPUT_CROP_HASH}), SATRN at model revision {SATRN_MODEL_REVISION} produced "
            f"{SATRN_OUTPUT_TEXT!r} against the reference {GROUND_TRUTH_TEXT!r}. Measured against "
            f"that reference under {NORMALIZATION_PROFILE}: CER {SATRN_CER_NORMALIZED}, WER "
            f"{SATRN_WER_NORMALIZED} (every word wrong), exact word accuracy 0.0, "
            f"{SATRN_CHARACTER_DELETIONS:.0f} character deletions and "
            f"{SATRN_WORD_DELETIONS:.0f} word deletions. "
            f"htr/evaluation/failures.py::classify_reliability independently flagged the run "
            f"omitted_text: {SATRN_OMITTED_TEXT_DETAIL!r}. The output is not a partial or degraded "
            "reading of the line -- it has no lexical overlap with it. Recorded for this run and this "
            "sample only; no recurrence is established by any record in this repository. Note that "
            "SATRN emitted no ParsedMethodResultRecorded event: the adapter has no separate parsed "
            "stage, so the raw and normalized stages are the only two text stages that exist to link."
        ),
        scope=SATRN_SCOPE,
        supporting_evidence=(
            _ref(EvidenceReferenceKind.EXPERIMENT_RUN, CONTROLLED_RUN_ID),
            _ref(EvidenceReferenceKind.METHOD_RUN, SATRN_METHOD_RUN_ID),
            _ref(EvidenceReferenceKind.INPUT_CROP, INPUT_CROP_ID, f"content hash {INPUT_CROP_HASH}"),
            _ref(
                EvidenceReferenceKind.TELEMETRY_EVENT,
                SATRN_RAW_EVENT_ID,
                f"RawMethodResultRecorded: {SATRN_OUTPUT_TEXT!r}",
            ),
            _ref(
                EvidenceReferenceKind.TELEMETRY_EVENT,
                SATRN_NORMALIZED_EVENT_ID,
                "NormalizedMethodResultRecorded -- SATRN's only other text stage; there is no "
                "ParsedMethodResultRecorded for this method run",
            ),
            _ref(
                EvidenceReferenceKind.GROUND_TRUTH_TEXT,
                GROUND_TRUTH_EVENT_ID,
                f"GroundTruthTextRecorded for {TEXT_LINE_ID}: {GROUND_TRUTH_TEXT!r}",
            ),
            _ref(
                EvidenceReferenceKind.METRIC_RESULT,
                SATRN_CER_NORMALIZED_METRIC_ID,
                f"character_error_rate_normalized = {SATRN_CER_NORMALIZED}",
            ),
            _ref(
                EvidenceReferenceKind.METRIC_RESULT,
                SATRN_WER_NORMALIZED_METRIC_ID,
                f"word_error_rate_normalized = {SATRN_WER_NORMALIZED}",
            ),
            _ref(
                EvidenceReferenceKind.METRIC_RESULT,
                SATRN_CHARACTER_DELETIONS_METRIC_ID,
                f"character_deletions = {SATRN_CHARACTER_DELETIONS}",
            ),
            _ref(
                EvidenceReferenceKind.METRIC_RESULT,
                SATRN_WORD_DELETIONS_METRIC_ID,
                f"word_deletions = {SATRN_WORD_DELETIONS}",
            ),
            _ref(
                EvidenceReferenceKind.RELIABILITY_CLASSIFICATION,
                SATRN_OMITTED_TEXT_FAILURE_ID,
                f"omitted_text -- {SATRN_OMITTED_TEXT_DETAIL}",
            ),
            _ref(
                EvidenceReferenceKind.TELEMETRY_EVENT,
                SATRN_OMITTED_TEXT_EVENT_ID,
                "ReliabilityIssueClassified carrying the omitted_text classification",
            ),
            _ref(EvidenceReferenceKind.EVIDENCE_RECORD, SATRN_EVIDENCE_ID),
        ),
        source_experiment_id=EXPERIMENT_ID,
        source_experiment_run_id=CONTROLLED_RUN_ID,
        affected_method=SATRN_METHOD_ID,
        affected_model_version=SATRN_MODEL_REVISION,
        affected_dataset_id=DATASET_ID,
        affected_dataset_version_id=DATASET_VERSION_ID,
        affected_document_or_segment_ids=(TEXT_LINE_ID, INPUT_CROP_ID),
        author_or_source_component=EXTRACTION_COMPONENT,
        creation_timestamp=at,
        tags=("omitted_text", "n=1", "single_occurrence", "controlled_comparison", "swedish_17c"),
        observation_confidence=ObservationConfidence.HIGH,
        unverified_hypothesis=None,
    )


def satrn_confidence_disagreement_observation(*, at: str = EXTRACTED_AT) -> ResearchObservation:
    """Observation 2: SATRN's reported confidence did not track its accuracy on this sample.

    `scope` binds this to one run and one crop, and `ResearchScope.sample_size` is derived from
    `covered_unit_ids` rather than asserted, so every rendering of this observation states `1
    line_crop` whether or not the reader is paying attention. There is no field on `ResearchScope`
    that could express "SATRN's confidence is miscalibrated in general", which is the point.
    """
    return ResearchObservation.create(
        observation_type=ObservationType.CONFIDENCE_ANOMALY,
        title=(
            f"SATRN reported confidence {SATRN_REPORTED_CONFIDENCE:.4f} on a transcription with CER "
            f"{SATRN_CER_NORMALIZED:.4f} and WER {SATRN_WER_NORMALIZED} -- one crop, one run"
        ),
        description=(
            f"For method run {SATRN_METHOD_RUN_ID} in experiment run {CONTROLLED_RUN_ID}, on the "
            f"single crop {INPUT_CROP_ID}, SATRN's own reported confidence scalar was "
            f"{SATRN_REPORTED_CONFIDENCE} (recorded as Evidence.provider_confidence on "
            f"{SATRN_EVIDENCE_ID}) while the measured accuracy of the same output against ground "
            f"truth was CER {SATRN_CER_NORMALIZED} and WER {SATRN_WER_NORMALIZED}. "
            f"classify_reliability flagged confidence_calibration_disagreement: "
            f"{SATRN_CONFIDENCE_DISAGREEMENT_DETAIL}. "
            "This is a statement about this run and this sample. It is not a claim about SATRN's "
            "confidence calibration, which one sample cannot measure -- and it must not be read as "
            "one: the scope covers exactly one line crop in exactly one experiment run at exactly one "
            "model revision, and no record in this repository extends it further."
        ),
        scope=SATRN_SCOPE,
        supporting_evidence=(
            _ref(EvidenceReferenceKind.EXPERIMENT_RUN, CONTROLLED_RUN_ID),
            _ref(EvidenceReferenceKind.METHOD_RUN, SATRN_METHOD_RUN_ID),
            _ref(
                EvidenceReferenceKind.EVIDENCE_RECORD,
                SATRN_EVIDENCE_ID,
                f"provider_confidence = {SATRN_REPORTED_CONFIDENCE} (the model's own scalar)",
            ),
            _ref(
                EvidenceReferenceKind.METRIC_RESULT,
                SATRN_CER_NORMALIZED_METRIC_ID,
                f"character_error_rate_normalized = {SATRN_CER_NORMALIZED}",
            ),
            _ref(
                EvidenceReferenceKind.METRIC_RESULT,
                SATRN_WER_NORMALIZED_METRIC_ID,
                f"word_error_rate_normalized = {SATRN_WER_NORMALIZED}",
            ),
            _ref(
                EvidenceReferenceKind.METRIC_RESULT,
                SATRN_EXACT_WORD_ACCURACY_METRIC_ID,
                "exact_word_accuracy = 0.0",
            ),
            _ref(
                EvidenceReferenceKind.RELIABILITY_CLASSIFICATION,
                SATRN_CONFIDENCE_DISAGREEMENT_FAILURE_ID,
                f"confidence_calibration_disagreement -- {SATRN_CONFIDENCE_DISAGREEMENT_DETAIL}",
            ),
            _ref(
                EvidenceReferenceKind.TELEMETRY_EVENT,
                SATRN_CONFIDENCE_DISAGREEMENT_EVENT_ID,
                "ReliabilityIssueClassified carrying the confidence_calibration_disagreement "
                "classification",
            ),
            _ref(EvidenceReferenceKind.INPUT_CROP, INPUT_CROP_ID),
            _ref(EvidenceReferenceKind.GROUND_TRUTH_TEXT, GROUND_TRUTH_EVENT_ID),
        ),
        source_experiment_id=EXPERIMENT_ID,
        source_experiment_run_id=CONTROLLED_RUN_ID,
        affected_method=SATRN_METHOD_ID,
        affected_model_version=SATRN_MODEL_REVISION,
        affected_dataset_id=DATASET_ID,
        affected_dataset_version_id=DATASET_VERSION_ID,
        affected_document_or_segment_ids=(TEXT_LINE_ID, INPUT_CROP_ID),
        author_or_source_component=EXTRACTION_COMPONENT,
        creation_timestamp=at,
        tags=(
            "confidence_calibration_disagreement",
            "n=1",
            "this_run_and_this_sample_only",
            "not_a_calibration_claim",
        ),
        observation_confidence=ObservationConfidence.HIGH,
        unverified_hypothesis=None,
    )


def florence2_relative_result_observation(*, at: str = EXTRACTED_AT) -> ResearchObservation:
    """Observation 3: the two recognizers' measured error rates on one shared crop.

    The title and description are phrased as the follow-up's own worked example -- "Florence-2
    produced lower CER and WER than SATRN on one shared line crop in baseline experiment version X
    under configuration Y" -- and deliberately never as "Florence-2 is better than SATRN". The
    difference is not cosmetic: the first is a measurement, the second is a generalization from N=1.
    """
    return ResearchObservation.create(
        observation_type=ObservationType.UNEXPECTED_METHOD_DISAGREEMENT,
        title=(
            f"Florence-2 produced lower CER and WER than SATRN on one shared line crop "
            f"({INPUT_CROP_ID}) in baseline experiment version {EXPERIMENT_VERSION_ID} under "
            f"configuration {EXPERIMENT_VERSION_ID}"
        ),
        description=(
            f"On the one shared, byte-identical, hash-verified line crop {INPUT_CROP_ID} (content "
            f"hash {INPUT_CROP_HASH}), in experiment run {CONTROLLED_RUN_ID} of experiment version "
            f"{EXPERIMENT_VERSION_ID}, against the reference {GROUND_TRUTH_TEXT!r} and under "
            f"{NORMALIZATION_PROFILE}: Florence-2 at model revision {FLORENCE2_MODEL_REVISION} "
            f"produced {FLORENCE2_PARSED_TEXT!r} with CER {FLORENCE2_CER_NORMALIZED} and WER "
            f"{FLORENCE2_WER_NORMALIZED}, while SATRN at model revision {SATRN_MODEL_REVISION} "
            f"produced {SATRN_OUTPUT_TEXT!r} with CER {SATRN_CER_NORMALIZED} and WER "
            f"{SATRN_WER_NORMALIZED}. Both values are lower for Florence-2 on this crop under these "
            "exact model revisions, this exact ground truth, this exact experiment configuration and "
            "these exact normalization rules. Change any one of those five and this observation says "
            "nothing about the result. It is a measurement on one sample, not a ranking of the two "
            "methods, and no aggregate over more than this one crop exists in this repository."
        ),
        scope=COMPARISON_SCOPE,
        supporting_evidence=(
            _ref(EvidenceReferenceKind.EXPERIMENT_RUN, CONTROLLED_RUN_ID),
            _ref(
                EvidenceReferenceKind.INPUT_CROP,
                INPUT_CROP_ID,
                f"one shared crop, content hash {INPUT_CROP_HASH} verified identical for both "
                "method runs",
            ),
            _ref(EvidenceReferenceKind.METHOD_RUN, FLORENCE2_METHOD_RUN_ID),
            _ref(EvidenceReferenceKind.METHOD_RUN, SATRN_METHOD_RUN_ID),
            _ref(
                EvidenceReferenceKind.METRIC_RESULT,
                FLORENCE2_CER_NORMALIZED_METRIC_ID,
                f"Florence-2 character_error_rate_normalized = {FLORENCE2_CER_NORMALIZED}",
            ),
            _ref(
                EvidenceReferenceKind.METRIC_RESULT,
                FLORENCE2_WER_NORMALIZED_METRIC_ID,
                f"Florence-2 word_error_rate_normalized = {FLORENCE2_WER_NORMALIZED}",
            ),
            _ref(
                EvidenceReferenceKind.METRIC_RESULT,
                SATRN_CER_NORMALIZED_METRIC_ID,
                f"SATRN character_error_rate_normalized = {SATRN_CER_NORMALIZED}",
            ),
            _ref(
                EvidenceReferenceKind.METRIC_RESULT,
                SATRN_WER_NORMALIZED_METRIC_ID,
                f"SATRN word_error_rate_normalized = {SATRN_WER_NORMALIZED}",
            ),
            _ref(
                EvidenceReferenceKind.TELEMETRY_EVENT,
                FLORENCE2_PARSED_EVENT_ID,
                f"ParsedMethodResultRecorded: {FLORENCE2_PARSED_TEXT!r}",
            ),
            _ref(
                EvidenceReferenceKind.TELEMETRY_EVENT,
                SATRN_NORMALIZED_EVENT_ID,
                f"NormalizedMethodResultRecorded: {SATRN_OUTPUT_TEXT!r}",
            ),
            _ref(EvidenceReferenceKind.GROUND_TRUTH_TEXT, GROUND_TRUTH_EVENT_ID),
            _ref(EvidenceReferenceKind.EVIDENCE_RECORD, FLORENCE2_EVIDENCE_ID),
            _ref(EvidenceReferenceKind.EVIDENCE_RECORD, SATRN_EVIDENCE_ID),
        ),
        source_experiment_id=EXPERIMENT_ID,
        source_experiment_run_id=CONTROLLED_RUN_ID,
        affected_method=None,
        affected_model_version=None,
        affected_dataset_id=DATASET_ID,
        affected_dataset_version_id=DATASET_VERSION_ID,
        affected_document_or_segment_ids=(TEXT_LINE_ID, INPUT_CROP_ID),
        author_or_source_component=EXTRACTION_COMPONENT,
        creation_timestamp=at,
        tags=(
            "n=1",
            "one_shared_line_crop",
            "not_a_method_ranking",
            "controlled_comparison",
            "hash_verified_input",
        ),
        observation_confidence=ObservationConfidence.HIGH,
        unverified_hypothesis=None,
    )


def gpu_memory_variability_observation(*, at: str = EXTRACTED_AT) -> ResearchObservation:
    """Observation 4: two different Florence-2 peak-GPU-memory figures across separate sessions.

    The one observation in this module with a populated `unverified_hypothesis`. Everything in
    `description` is a figure quoted from a committed record; the allocator explanation lives in the
    separate field because this project has not measured it. Keeping the two apart is structural, not
    stylistic: an exporter, a UI or a reader can tell which is which without parsing prose.
    """
    return ResearchObservation.create(
        observation_type=ObservationType.REPRODUCIBILITY_ANOMALY,
        title=(
            "Florence-2 peak GPU memory on the shared baseline crop differs by ~3.3x between two "
            f"documented sessions ({FLORENCE2_GPU_MEMORY_README_MB:.0f} MB documented vs. "
            f"{FLORENCE2_GPU_MEMORY_MEASURED_MIB:.2f} MiB measured)"
        ),
        description=(
            "Two committed records in this repository state different peak-GPU-memory figures for "
            f"Florence-2 on the same fixture. src/archivetrust/providers/florence2_htr/README.md "
            f"documents ~{FLORENCE2_GPU_MEMORY_README_MB:.0f} MB from an earlier CUDA session. The "
            f"2026-07-30 run (experiment run {CONTROLLED_RUN_ID}, method run "
            f"{FLORENCE2_METHOD_RUN_ID}) measured {FLORENCE2_GPU_MEMORY_MEASURED_MIB} MiB, recorded "
            f"as metric result {FLORENCE2_GPU_MEMORY_METRIC_ID} and as Evidence.gpu_memory_mb on "
            f"{FLORENCE2_EVIDENCE_ID}. Both figures are real measurements; the earlier session has no "
            "durable telemetry record at all, which is itself part of this observation -- it predates "
            "the persistence this project now has, so there is no way to compare the two runs' "
            "environments beyond what each wrote down. The same comparison for SATRN did NOT diverge: "
            f"its README documents ~{SATRN_GPU_MEMORY_README_MB:.0f} MB and this run measured "
            f"{SATRN_GPU_MEMORY_MEASURED_MIB} MiB (metric result "
            f"{SATRN_GPU_MEMORY_METRIC_ID}), so the divergence is specific to Florence-2 rather than a "
            "general property of how this repository measures GPU memory. Both methods' text outputs "
            "reproduced exactly across the two sessions, so this is an operational/reproducibility "
            "observation about resource measurement, not about recognition output."
        ),
        scope=FLORENCE2_ENVIRONMENT_SCOPE,
        supporting_evidence=(
            _ref(EvidenceReferenceKind.EXPERIMENT_RUN, CONTROLLED_RUN_ID),
            _ref(EvidenceReferenceKind.METHOD_RUN, FLORENCE2_METHOD_RUN_ID),
            _ref(
                EvidenceReferenceKind.METRIC_RESULT,
                FLORENCE2_GPU_MEMORY_METRIC_ID,
                f"gpu_memory_mb = {FLORENCE2_GPU_MEMORY_MEASURED_MIB} (2026-07-30 run)",
            ),
            _ref(
                EvidenceReferenceKind.TELEMETRY_EVENT,
                FLORENCE2_GPU_MEMORY_EVENT_ID,
                "MetricCalculated carrying the measured figure",
            ),
            _ref(
                EvidenceReferenceKind.EVIDENCE_RECORD,
                FLORENCE2_EVIDENCE_ID,
                f"Evidence.gpu_memory_mb = {FLORENCE2_GPU_MEMORY_MEASURED_MIB}, "
                "execution_device='cuda', torch 2.13.0+cu130",
            ),
            _ref(
                EvidenceReferenceKind.EXTERNAL_DOCUMENT,
                FLORENCE2_README_REF,
                f"'Peak GPU memory: ~{FLORENCE2_GPU_MEMORY_README_MB:.0f} MB' -- the earlier "
                "session, which has no durable telemetry record",
                stream=None,
            ),
            _ref(
                EvidenceReferenceKind.METRIC_RESULT,
                SATRN_GPU_MEMORY_METRIC_ID,
                f"SATRN control: gpu_memory_mb = {SATRN_GPU_MEMORY_MEASURED_MIB}, against a "
                f"documented ~{SATRN_GPU_MEMORY_README_MB:.0f} MB -- reproduced",
            ),
        ),
        source_experiment_id=EXPERIMENT_ID,
        source_experiment_run_id=CONTROLLED_RUN_ID,
        affected_method=FLORENCE2_METHOD_ID,
        affected_model_version=FLORENCE2_MODEL_REVISION,
        affected_dataset_id=DATASET_ID,
        affected_dataset_version_id=DATASET_VERSION_ID,
        affected_document_or_segment_ids=(INPUT_CROP_ID,),
        author_or_source_component=EXTRACTION_COMPONENT,
        creation_timestamp=at,
        tags=(
            "reproducibility",
            "operational",
            "gpu_memory",
            "environment_not_fully_captured",
            "outputs_reproduced_exactly",
        ),
        observation_confidence=ObservationConfidence.MODERATE,
        unverified_hypothesis=(
            "UNVERIFIED. One candidate explanation is CUDA caching-allocator state: "
            "torch.cuda.max_memory_allocated (or reserved) reflects allocator history within a "
            "process, so a session that had already run other models, or that used a different torch "
            "build (2.13.0+cu130 here vs. whatever the earlier session used -- not recorded), can "
            "report a substantially different peak for identical work. THIS PROJECT HAS NOT MEASURED "
            "THAT. No controlled experiment varying allocator state exists, the earlier session's "
            "torch version and process history were never captured, and this explanation is therefore "
            "a hypothesis to test, not a cause to cite. Testing it would need two runs in this "
            "repository's own durable persistence with the environment fields populated for both."
        ),
    )


def transkribus_comparability_observation(*, at: str = EXTRACTED_AT) -> ResearchObservation:
    """Observation 5: the Transkribus fixture is outside the controlled comparison's boundary.

    Typed `EXPERIMENT_VALIDITY_BOUNDARY` and tagged `not_a_method_performance_observation`. This is a
    statement about what the experiment can measure, and it says nothing whatever about how
    Transkribus performs -- the fixture it ran on has no ground truth, so no accuracy claim about it
    is even constructible.
    """
    return ResearchObservation.create(
        observation_type=ObservationType.EXPERIMENT_VALIDITY_BOUNDARY,
        title=(
            f"The Transkribus fixture {TRANSKRIBUS_FIXTURE_REF} does not correspond to the controlled "
            "ground-truth line and was correctly excluded from CER/WER"
        ),
        description=(
            f"Method run {TRANSKRIBUS_METHOD_RUN_ID} (Transkribus {TRANSKRIBUS_MODEL_VERSION}) ran on "
            f"the separate end-to-end experiment run {END_TO_END_RUN_ID} with input_crop_id = null, "
            f"page-level only, parsing {TRANSKRIBUS_FIXTURE_REF}. That fixture transcribes a "
            f"different, unrelated Swedish court-record passage -- its first line is "
            f"{TRANSKRIBUS_FIRST_LINE_TEXT!r} -- with no established line-to-line correspondence to "
            f"the controlled reference {GROUND_TRUTH_TEXT!r}. No CER and no WER exists for it "
            "anywhere in the run's log: not a null written over a computed value, but a metric that "
            "was never computed, because a CER between two unrelated texts would measure nothing. The "
            "exclusion is declared in advance by the experiment's own exclusion_criteria "
            f"({BASELINE_TEMPLATE_REF}) rather than applied after seeing results. "
            "THIS IS NOT A METHOD-PERFORMANCE OBSERVATION. It records where the controlled "
            "comparison's boundary lies. Nothing here says anything about Transkribus's recognition "
            "accuracy, and nothing could: the fixture it read has no ground truth at all, and is "
            "itself hand-authored rather than genuine vendor output "
            "(tests/fixtures/transkribus/README.md)."
        ),
        scope=TRANSKRIBUS_SCOPE,
        supporting_evidence=(
            _ref(EvidenceReferenceKind.EXPERIMENT_RUN, END_TO_END_RUN_ID, "is_end_to_end = true"),
            _ref(
                EvidenceReferenceKind.METHOD_RUN,
                TRANSKRIBUS_METHOD_RUN_ID,
                "input_crop_id = null -- page-level, never bound to the shared crop",
            ),
            _ref(
                EvidenceReferenceKind.TELEMETRY_EVENT,
                TRANSKRIBUS_METHOD_RUN_EVENT_ID,
                "MethodRunStarted carrying the MethodRun with input_crop_id = null",
            ),
            _ref(
                EvidenceReferenceKind.TELEMETRY_EVENT,
                TRANSKRIBUS_PARSED_EVENT_ID,
                f"ParsedMethodResultRecorded, first line {TRANSKRIBUS_FIRST_LINE_TEXT!r}",
            ),
            _ref(
                EvidenceReferenceKind.TELEMETRY_EVENT,
                TRANSKRIBUS_NORMALIZED_EVENT_ID,
                "NormalizedMethodResultRecorded -- the last text stage; no MetricCalculated follows "
                "it for CER or WER",
            ),
            _ref(
                EvidenceReferenceKind.GROUND_TRUTH_TEXT,
                GROUND_TRUTH_EVENT_ID,
                "the controlled reference this fixture does not correspond to",
            ),
            _ref(EvidenceReferenceKind.INPUT_CROP, INPUT_CROP_ID, "the controlled crop, not read by "
                 "this method run"),
            _ref(
                EvidenceReferenceKind.EXTERNAL_DOCUMENT,
                BASELINE_TEMPLATE_REF,
                "the pre-declared exclusion criterion",
                stream=None,
            ),
            _ref(
                EvidenceReferenceKind.EXTERNAL_DOCUMENT,
                TRANSKRIBUS_FIXTURE_REF,
                "the fixture itself -- hand-authored, not genuine vendor output",
                stream=None,
            ),
        ),
        source_experiment_id=EXPERIMENT_ID,
        source_experiment_run_id=END_TO_END_RUN_ID,
        affected_method=TRANSKRIBUS_METHOD_ID,
        affected_model_version=TRANSKRIBUS_MODEL_VERSION,
        affected_dataset_id=DATASET_ID,
        affected_dataset_version_id=DATASET_VERSION_ID,
        affected_document_or_segment_ids=(TRANSKRIBUS_PAGE_ID,),
        author_or_source_component=EXTRACTION_COMPONENT,
        creation_timestamp=at,
        tags=(
            "experiment_validity",
            "comparison_boundary",
            "not_a_method_performance_observation",
            "excluded_from_controlled_set",
            "no_ground_truth_exists",
        ),
        observation_confidence=ObservationConfidence.HIGH,
        unverified_hypothesis=None,
    )


OBSERVATION_BUILDERS = (
    satrn_omitted_text_observation,
    satrn_confidence_disagreement_observation,
    florence2_relative_result_observation,
    gpu_memory_variability_observation,
    transkribus_comparability_observation,
)
"""The five in the order they are registered. Ordered deliberately: the two SATRN observations come
before the comparison observation that cites SATRN's numbers, so a reader of the log meets each fact
before the record that builds on it."""


def baseline_observations(*, at: str = EXTRACTED_AT) -> tuple[ResearchObservation, ...]:
    """All five, freshly constructed (each call mints new observation ids)."""
    return tuple(builder(at=at) for builder in OBSERVATION_BUILDERS)


# -- Causation: which run event supplied each observation's evidence ------------------------------

OBSERVATION_CAUSING_EVENT: dict[str, str] = {
    "satrn_omitted_text": SATRN_OMITTED_TEXT_EVENT_ID,
    "satrn_confidence_disagreement": SATRN_CONFIDENCE_DISAGREEMENT_EVENT_ID,
    "florence2_relative_result": FLORENCE2_CER_NORMALIZED_EVENT_ID,
    "gpu_memory_variability": FLORENCE2_GPU_MEMORY_EVENT_ID,
    "transkribus_comparability": TRANSKRIBUS_NORMALIZED_EVENT_ID,
}
"""`causation_id` for each observation's `ResearchObservationCreated`, keyed by the short name used in
`register_baseline_knowledge`.

Each is the event in the run's log that supplied the *decisive* piece of evidence -- the
`ReliabilityIssueClassified` for the two reliability observations, the `MetricCalculated` for the
comparison and the GPU-memory one, and for the Transkribus observation the
`NormalizedMethodResultRecorded` that is its last text stage: the absence of any `MetricCalculated`
after it *is* the evidence, so the event that should have caused one is the honest pointer.

An observation typically rests on ten or more records; `causation_id` is single-valued, so the full
list lives in `supporting_evidence` where it can be plural and typed. Choosing one decisive event for
the causal edge is a judgement, and naming it in a table here rather than burying it in a call site is
what makes that judgement reviewable."""


# -- The four real candidate findings --------------------------------------------------------------

SHARED_LIMITATIONS = (
    "N=1: one line crop, one experiment run, one checkpoint per method. No aggregate over more "
    "than this crop exists in this repository.",
    "No general ranking of methods can be inferred from this scope.",
    "The ground truth is a single external annotation from the upstream published dataset -- no "
    "blind dual annotation and no adjudication was performed for it.",
    "No versioned TranscriptionConvention record exists for this ground truth, so the "
    "transcription rules the reference was authored under are not themselves versioned here.",
    "Metrics hold only under the normalization profile named in scope; a different profile could "
    "produce different values from the same texts.",
)
"""Limitations every finding from this run carries. `ResearchFinding.limitations` is non-empty by
construction, and on an N=1 run the limitations are most of the content -- so they are stated once
here and shared rather than paraphrased four times and drifting."""

RESEARCH_QUESTION = (
    "How do SATRN (Riksarkivet), Florence-2 (fine-tuned OCR checkpoint) and Transkribus Swedish "
    "Lion I compare on Swedish historical handwriting recognition, under a controlled line-level "
    "comparison on byte-identical input crops and an end-to-end page-level comparison?"
)
"""Quoted from the experiment version's own `pipeline_configuration_ref.research_question`."""

FINDING_AUTHOR = "htr.knowledge.baseline_knowledge"
"""Author of all four candidate findings: the same automated extraction step that wrote the
observations. Honest -- these are machine-drafted candidates awaiting a human, which is exactly what
`Candidate` status means. A human name appears only on a `FindingRevision`, once a human actually
reviews one."""


def _finding(
    *,
    statement: str,
    scope: ResearchScope,
    supporting_observations: tuple[str, ...],
    supporting_metrics: tuple[str, ...],
    affected_methods: tuple[str, ...],
    affected_model_versions: tuple[str, ...],
    confidence_level: FindingConfidence,
    extra_limitations: tuple[str, ...] = (),
    hypothesis_relationship: HypothesisRelationship = (
        HypothesisRelationship.NO_HYPOTHESIS_ASSERTED
    ),
    at: str = EXTRACTED_AT,
) -> ResearchFinding:
    return ResearchFinding.create(
        statement=statement,
        scope=scope,
        supporting_observations=supporting_observations,
        supporting_metrics=supporting_metrics,
        supporting_experiments=(EXPERIMENT_ID,),
        limitations=(*SHARED_LIMITATIONS, *extra_limitations),
        confidence_level=confidence_level,
        author=FINDING_AUTHOR,
        creation_date=at,
        status=FindingStatus.CANDIDATE,
        research_question=RESEARCH_QUESTION,
        hypothesis_relationship=hypothesis_relationship,
        affected_datasets=(DATASET_ID,),
        affected_dataset_versions=(DATASET_VERSION_ID,),
        affected_document_types=("swedish_court_record_line",),
        affected_handwriting_periods=("17th_century_swedish",),
        affected_methods=affected_methods,
        affected_model_versions=affected_model_versions,
        transcription_convention=None,
    )


def baseline_candidate_findings(
    observations: dict[str, ResearchObservation], *, at: str = EXTRACTED_AT
) -> dict[str, ResearchFinding]:
    """The four specified candidate findings, plus the fifth this repository's own data supports.

    Every one links to its `ResearchObservation`(s) by id via `supporting_observations` and does not
    re-derive evidence independently -- the observation already names the metric results and events,
    and duplicating that list here would create two evidence trails that could disagree.

    `observations` maps the short names in `OBSERVATION_CAUSING_EVENT` to registered observations, so
    the ids linked here are the ones actually in the log rather than freshly minted ones.

    **The fifth finding.** The follow-up specifies four; this returns five. The extra one --
    `florence2_environment_reproducible` -- exists because the workflow demonstration needs a
    genuinely disputable claim, and inventing a second experiment run to manufacture one is exactly
    what the brief forbids. This claim is real, is asserted by a committed record in this repository
    (the Florence-2 adapter README's documented peak GPU memory), and is genuinely contradicted by
    another committed record (this run's measured figure). Disputing it is therefore grounded in real
    data rather than a constructed scenario. It is labelled as an addition rather than passed off as
    one of the four.
    """
    return {
        "florence2_lower_error_rates": _finding(
            statement=(
                "On the single controlled baseline line, Florence-2 produced lower CER and WER than "
                f"SATRN (CER {FLORENCE2_CER_NORMALIZED} vs. {SATRN_CER_NORMALIZED}; WER "
                f"{FLORENCE2_WER_NORMALIZED} vs. {SATRN_WER_NORMALIZED}), on input crop "
                f"{INPUT_CROP_ID} in experiment version {EXPERIMENT_VERSION_ID}, experiment run "
                f"{CONTROLLED_RUN_ID}, at model revisions {FLORENCE2_MODEL_REVISION} and "
                f"{SATRN_MODEL_REVISION} respectively, under {NORMALIZATION_PROFILE}."
            ),
            scope=COMPARISON_SCOPE,
            supporting_observations=(
                observations["florence2_relative_result"].observation_id,
                observations["satrn_omitted_text"].observation_id,
            ),
            supporting_metrics=(
                FLORENCE2_CER_NORMALIZED_METRIC_ID,
                FLORENCE2_WER_NORMALIZED_METRIC_ID,
                SATRN_CER_NORMALIZED_METRIC_ID,
                SATRN_WER_NORMALIZED_METRIC_ID,
            ),
            affected_methods=(FLORENCE2_METHOD_ID, SATRN_METHOD_ID),
            affected_model_versions=(FLORENCE2_MODEL_REVISION, SATRN_MODEL_REVISION),
            confidence_level=FindingConfidence.LOW,
            extra_limitations=(
                "Says nothing about which method is better at Swedish historical HTR. Two error "
                "rates on one crop are two measurements, not a ranking.",
                "The two methods' reported confidence values are not comparable to each other: "
                "SATRN's is the model's own scalar, Florence-2's is a proxy exp(sequences_scores). "
                "This finding rests on CER/WER only, not on either confidence.",
            ),
            at=at,
        ),
        "satrn_omitted_reference_text": _finding(
            statement=(
                "On the single controlled baseline line, SATRN omitted a substantial portion of the "
                f"reference text: {SATRN_WORD_DELETIONS:.0f} of 10 reference words and "
                f"{SATRN_CHARACTER_DELETIONS:.0f} characters were deleted, on input crop "
                f"{INPUT_CROP_ID} in experiment run {CONTROLLED_RUN_ID} at model revision "
                f"{SATRN_MODEL_REVISION}, flagged omitted_text by "
                "htr/evaluation/failures.py::classify_reliability."
            ),
            scope=SATRN_SCOPE,
            supporting_observations=(observations["satrn_omitted_text"].observation_id,),
            supporting_metrics=(
                SATRN_CHARACTER_DELETIONS_METRIC_ID,
                SATRN_WORD_DELETIONS_METRIC_ID,
                SATRN_CER_NORMALIZED_METRIC_ID,
                SATRN_WER_NORMALIZED_METRIC_ID,
            ),
            affected_methods=(SATRN_METHOD_ID,),
            affected_model_versions=(SATRN_MODEL_REVISION,),
            confidence_level=FindingConfidence.MODERATE,
            extra_limitations=(
                "One occurrence. 'Omission' here is the measured edit-distance category on this "
                "sample, not an established behaviour of the checkpoint.",
                "The output has no lexical overlap with the line, so 'omitted' understates it -- but "
                "'omitted_text' is the classification the evaluation engine actually recorded, and "
                "this finding does not substitute a different word for it.",
            ),
            at=at,
        ),
        "satrn_confidence_not_aligned": _finding(
            statement=(
                "The SATRN confidence value on this sample was not aligned with the observed "
                f"transcription accuracy: reported confidence {SATRN_REPORTED_CONFIDENCE} against "
                f"measured CER {SATRN_CER_NORMALIZED} and WER {SATRN_WER_NORMALIZED} on input crop "
                f"{INPUT_CROP_ID} in experiment run {CONTROLLED_RUN_ID}, flagged "
                "confidence_calibration_disagreement."
            ),
            scope=SATRN_SCOPE,
            supporting_observations=(
                observations["satrn_confidence_disagreement"].observation_id,
            ),
            supporting_metrics=(
                SATRN_CER_NORMALIZED_METRIC_ID,
                SATRN_WER_NORMALIZED_METRIC_ID,
                SATRN_EXACT_WORD_ACCURACY_METRIC_ID,
            ),
            affected_methods=(SATRN_METHOD_ID,),
            affected_model_versions=(SATRN_MODEL_REVISION,),
            confidence_level=FindingConfidence.MODERATE,
            extra_limitations=(
                "A statement about one confidence value on one sample. Calibration is a "
                "distributional property and one sample cannot measure it.",
                "The disagreement threshold (0.35) and the similarity measure are "
                "classify_reliability's own parameters; a different threshold would not have flagged "
                "the run.",
            ),
            at=at,
        ),
        "transkribus_not_comparable": _finding(
            statement=(
                "The current Transkribus fixture cannot be included in the controlled recognizer "
                "comparison because it does not correspond to the shared ground-truth line: method "
                f"run {TRANSKRIBUS_METHOD_RUN_ID} ran page-level with input_crop_id = null on "
                f"{TRANSKRIBUS_FIXTURE_REF}, whose content is an unrelated passage, and no CER or WER "
                "was computed for it anywhere in the run's log."
            ),
            scope=TRANSKRIBUS_SCOPE,
            supporting_observations=(observations["transkribus_comparability"].observation_id,),
            supporting_metrics=(),
            affected_methods=(TRANSKRIBUS_METHOD_ID,),
            affected_model_versions=(TRANSKRIBUS_MODEL_VERSION,),
            confidence_level=FindingConfidence.HIGH,
            extra_limitations=(
                "A statement about experiment validity, not about method performance. It says "
                "nothing about Transkribus's accuracy, which this run did not and could not measure.",
                "Scoped to the *current* fixture. A genuine vendor export of the controlled line "
                "would be a different fixture and this finding would not apply to it.",
                "supporting_metrics is deliberately empty: the evidence is the *absence* of a "
                "CER/WER metric, and there is no metric id for a metric that was never computed.",
            ),
            at=at,
        ),
        "florence2_environment_reproducible": _finding(
            statement=(
                "Florence-2's peak-GPU-memory measurement on the shared baseline line crop is "
                "reproducible across sessions: the adapter README documents "
                f"~{FLORENCE2_GPU_MEMORY_README_MB:.0f} MB for this fixture, and the controlled run "
                f"{CONTROLLED_RUN_ID} re-measures the same quantity for method run "
                f"{FLORENCE2_METHOD_RUN_ID} at model revision {FLORENCE2_MODEL_REVISION}."
            ),
            scope=FLORENCE2_ENVIRONMENT_SCOPE,
            supporting_observations=(observations["gpu_memory_variability"].observation_id,),
            supporting_metrics=(FLORENCE2_GPU_MEMORY_METRIC_ID,),
            affected_methods=(FLORENCE2_METHOD_ID,),
            affected_model_versions=(FLORENCE2_MODEL_REVISION,),
            confidence_level=FindingConfidence.LOW,
            extra_limitations=(
                "The earlier session has no durable telemetry record, so its torch version, device "
                "state and process history are unknown and the two measurements cannot be compared "
                "under matched conditions.",
                "This is the reproducibility claim implicit in documenting a resource measurement in "
                "a README at all, made explicit here so it can be examined rather than assumed.",
            ),
            at=at,
        ),
    }


def gpu_memory_contradiction(
    *,
    observation_id: str,
    recorded_by: str,
    at: str = EXTRACTED_AT,
) -> ContradictoryEvidence:
    """The real contradiction that disputes `florence2_environment_reproducible`.

    Grounded entirely in two committed records of this repository -- the adapter README's
    ~3983 MB and the run's measured 1210.64 MiB -- rather than in a hypothetical second run. Both
    records survive the dispute untouched: this entity is *appended* to the disputed finding, and the
    observation it cites remains independently readable with its own evidence intact.
    """
    return ContradictoryEvidence.create(
        source_kind=ContradictionSourceKind.RESEARCH_OBSERVATION,
        source_id=observation_id,
        description=(
            "The two figures differ by a factor of ~3.3 and are therefore not reproductions of one "
            f"another: {FLORENCE2_GPU_MEMORY_README_MB:.0f} MB documented in "
            f"{FLORENCE2_README_REF} against {FLORENCE2_GPU_MEMORY_MEASURED_MIB} MiB measured as "
            f"metric result {FLORENCE2_GPU_MEMORY_METRIC_ID} in run {CONTROLLED_RUN_ID}. The SATRN "
            f"control reproduced to within 0.3 MB over the same pair of sessions "
            f"({SATRN_GPU_MEMORY_README_MB:.0f} MB documented vs. "
            f"{SATRN_GPU_MEMORY_MEASURED_MIB} MiB measured, metric result "
            f"{SATRN_GPU_MEMORY_METRIC_ID}), so the divergence cannot be attributed to the "
            "measurement method in general. Whether the earlier figure, the later figure, or both "
            "are environment-dependent is not established -- which is precisely why the finding is "
            "Disputed rather than Rejected."
        ),
        recorded_by=recorded_by,
        recorded_at=at,
        evidence_refs=(
            _ref(
                EvidenceReferenceKind.METRIC_RESULT,
                FLORENCE2_GPU_MEMORY_METRIC_ID,
                f"gpu_memory_mb = {FLORENCE2_GPU_MEMORY_MEASURED_MIB}",
            ),
            _ref(
                EvidenceReferenceKind.EXTERNAL_DOCUMENT,
                FLORENCE2_README_REF,
                f"~{FLORENCE2_GPU_MEMORY_README_MB:.0f} MB",
                stream=None,
            ),
            _ref(
                EvidenceReferenceKind.METRIC_RESULT,
                SATRN_GPU_MEMORY_METRIC_ID,
                "the SATRN control, which did reproduce",
            ),
            _ref(
                EvidenceReferenceKind.RESEARCH_OBSERVATION,
                observation_id,
                "the observation recording both figures and holding the allocator explanation as an "
                "unverified hypothesis",
                stream=KNOWLEDGE_STREAM,
            ),
        ),
    )
