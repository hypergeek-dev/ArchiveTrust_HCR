"""Verifies `htr/knowledge/satrn_repetition_knowledge.py` against the real committed artifacts.

Same discipline as `test_baseline_knowledge.py`: the knowledge module states facts about a specific
historical run, and this test re-reads the artifacts those facts came from so a drift between the
two fails a test rather than sitting undetected. Nothing here re-runs inference; it reads the
committed diagnostic JSON, the committed smoke-test results, and the committed knowledge stream.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from archivetrust.application.htr_journal import HtrJournal
from archivetrust.htr.knowledge.models import (
    EvidenceReferenceKind,
    ObservationConfidence,
    ObservationReviewStatus,
    ObservationType,
)
from archivetrust.htr.knowledge.satrn_repetition_knowledge import (
    ALL_CROP_IDS,
    CROPS_IN_SAMPLE,
    DISTINCT_OUTPUT_CROP_IDS,
    FLORENCE2_DISTINCT_OUTPUTS,
    REPEATED_CROP_IDS,
    REPEATED_CROP_SHA256,
    REPEATED_GROUP_CONFIDENCES,
    REPEATED_OUTPUT,
    SATRN_DISTINCT_OUTPUTS,
    SATRN_METHOD_ID,
    SATRN_MODEL_REVISION,
    SECOND_REPEATED_CROP_IDS,
    satrn_output_repetition_observation,
    verify_against_diagnostic_report,
)
from archivetrust.infrastructure.storage.telemetry_sink import FileTelemetrySink

REPO_ROOT = Path(__file__).resolve().parents[3]
SCREENING = REPO_ROOT / "docs" / "experiments" / "technical-reliability-screening"
DIAGNOSTIC_JSON = SCREENING / "diagnostic" / "satrn_repetition_diagnostic.json"
KNOWLEDGE_STREAM = SCREENING / "diagnostic" / "htr_knowledge_events.jsonl"
SMOKE_RESULTS = SCREENING / "smoke-test" / "smoke_test_results.json"

pytestmark = pytest.mark.skipif(
    not DIAGNOSTIC_JSON.exists(),
    reason=(
        "the diagnostic artifact is gitignored alongside the smoke test it analyses; this test "
        "verifies the knowledge module against it wherever it is present"
    ),
)


@pytest.fixture(scope="module")
def report() -> dict:
    return json.loads(DIAGNOSTIC_JSON.read_text(encoding="utf-8"))


def _observation():
    return satrn_output_repetition_observation(
        experiment_id="experiment_test",
        experiment_version_id="experiment_version_test",
        experiment_run_id="experiment_run_test",
        dataset_id="dataset_test",
        dataset_version_id="dataset_version_test",
    )


def test_knowledge_module_has_not_drifted_from_the_diagnostic_artifact(report: dict) -> None:
    """The single most important assertion here -- `verify_against_diagnostic_report` recomputes
    every crop id, confidence, hash prefix, ratio and preprocessing constant from the artifact."""
    assert verify_against_diagnostic_report(report) == ()


def test_the_six_repeated_crops_are_the_ones_the_smoke_test_recorded() -> None:
    """Resolved against the *smoke test's own* results, not the diagnostic -- so the diagnostic
    cannot be the only thing vouching for which crops repeated."""
    results = json.loads(SMOKE_RESULTS.read_text(encoding="utf-8"))
    crops = [crop for page in results["pages"] for crop in page.get("recognition_crops", [])]
    assert len(crops) == CROPS_IN_SAMPLE
    repeated = tuple(c["crop_id"] for c in crops if c["satrn"]["text"] == REPEATED_OUTPUT)
    assert repeated == REPEATED_CROP_IDS
    assert len(set(c["satrn"]["text"] for c in crops)) == SATRN_DISTINCT_OUTPUTS
    assert len(set(c["florence2"]["text"] for c in crops)) == FLORENCE2_DISTINCT_OUTPUTS


def test_every_crop_group_is_disjoint_and_together_covers_the_sample() -> None:
    groups = (REPEATED_CROP_IDS, SECOND_REPEATED_CROP_IDS, DISTINCT_OUTPUT_CROP_IDS)
    combined = [crop_id for group in groups for crop_id in group]
    assert len(combined) == len(set(combined)) == CROPS_IN_SAMPLE
    assert set(combined) == set(ALL_CROP_IDS)


def test_the_repeated_group_carries_one_distinct_confidence_per_crop() -> None:
    """The evidence that most directly excludes a cached/stale/cross-wired result: six identical
    strings, six different scores."""
    assert len(REPEATED_GROUP_CONFIDENCES) == len(REPEATED_CROP_IDS)
    assert set(REPEATED_GROUP_CONFIDENCES) == set(REPEATED_CROP_IDS)
    assert len(set(REPEATED_GROUP_CONFIDENCES.values())) == len(REPEATED_CROP_IDS)


def test_repeated_crop_hashes_are_all_distinct_full_length_sha256() -> None:
    digests = set(REPEATED_CROP_SHA256.values())
    assert len(digests) == len(REPEATED_CROP_IDS)
    assert all(len(d) == 64 and set(d) <= set("0123456789abcdef") for d in digests)


def test_observation_is_scoped_bounded_and_attributed_to_the_pinned_checkpoint() -> None:
    observation = _observation()
    assert observation.observation_type is ObservationType.MODEL_LIMITATION
    assert observation.observation_confidence is ObservationConfidence.HIGH
    assert observation.review_status is ObservationReviewStatus.UNREVIEWED
    assert observation.affected_method == SATRN_METHOD_ID
    assert observation.affected_model_version == SATRN_MODEL_REVISION
    # The scope is finite and derived, never asserted.
    assert observation.scope.sample_size == CROPS_IN_SAMPLE
    assert observation.scope.covered_unit_ids == ALL_CROP_IDS
    # Both methods are named with an exact revision each (ResearchScope enforces the pairing).
    assert len(observation.scope.method_ids) == len(observation.scope.model_version_ids) == 2
    assert SATRN_MODEL_REVISION in observation.scope.model_version_ids


def test_every_repeated_crop_is_cited_as_typed_evidence() -> None:
    observation = _observation()
    cited = observation.evidence_ids(EvidenceReferenceKind.INPUT_CROP)
    for crop_id in REPEATED_CROP_IDS + DISTINCT_OUTPUT_CROP_IDS:
        assert crop_id in cited
    assert observation.evidence_ids(EvidenceReferenceKind.EXTERNAL_DOCUMENT)


def test_the_causal_explanation_is_quarantined_in_unverified_hypothesis() -> None:
    """The description must contain only what the records show. The aspect-ratio/binarization story
    is a guess and belongs in `unverified_hypothesis`, which is exactly the separation that field
    exists to enforce."""
    observation = _observation()
    assert observation.unverified_hypothesis is not None
    assert "NOT established" in observation.unverified_hypothesis
    for guess_word in ("binarized", "keep_ratio", "prior"):
        assert guess_word in observation.unverified_hypothesis
        assert guess_word not in observation.description


def test_the_observation_does_not_recommend_excluding_satrn() -> None:
    """The screening verdict belongs to a `ScreeningPolicy`/`ScreeningDecision` pair that
    design-audit.md §3.3 records as not yet existing. An observation must not smuggle one in."""
    observation = _observation()
    text = f"{observation.title} {observation.description}".casefold()
    # Phrases, not bare substrings: the description legitimately says the *integration-defect
    # hypotheses* were "excluded", which a naive `"exclude" not in text` would flag.
    for verdict_phrase in (
        "exclude satrn",
        "satrn should be excluded",
        "suspend satrn",
        "drop satrn",
        "satrn is unfit",
        "not viable",
        "disqualif",
        "remove satrn",
    ):
        assert verdict_phrase not in text
    assert "remains in scope" in observation.description
    assert "asserts nothing about SATRN's suitability" in observation.description


def test_all_five_diagnostic_passes_reproduced_the_repetition(report: dict) -> None:
    determinism = report["determinism"]
    assert determinism["pass_A_reproduced_smoke_text_for_all_15"]
    assert determinism["pass_B_reproduced_repeat_for_all_6"]
    assert determinism["pass_C_reversed_reproduced_repeat_for_all_6"]
    assert determinism["pass_D_isolated_reproduced_repeat_for_all_6"]
    assert determinism["pass_E_reference_reproduced_repeat_for_all_6"]


def test_the_diagnostic_excluded_the_integration_defect_hypotheses(report: dict) -> None:
    checks = report["integration_checks"]
    assert checks["every_call_got_a_fresh_child_process"]
    assert checks["distinct_child_pids_pass_A"] == checks["calls_in_pass_A"] == CROPS_IN_SAMPLE
    assert checks["no_temp_file_in_argv"]
    assert checks["argv_always_pointed_at_source_crop"]
    assert checks["input_bytes_unchanged_by_call"]
    assert checks["input_hash_matches_manifest"]
    assert checks["distinct_model_revisions"] == [SATRN_MODEL_REVISION]
    assert checks["distinct_confidences_within_repeated_group"] == len(REPEATED_CROP_IDS)


def test_the_reference_path_reproduced_the_constancy_without_archivetrust(report: dict) -> None:
    """Pass E ran mmocr's own inferencer in one process with one model load. Identical constancy
    there is what makes "integration defect" untenable."""
    adapter_side = report["constancy"]["smoke_test_original"]
    reference_side = report["constancy"]["pass_E_reference_mmocr"]
    assert reference_side == adapter_side
    assert reference_side["distinct_outputs"] == SATRN_DISTINCT_OUTPUTS
    assert reference_side["most_repeated_output"] == REPEATED_OUTPUT
    assert reference_side["most_repeated_count"] == len(REPEATED_CROP_IDS)


@pytest.mark.skipif(
    not KNOWLEDGE_STREAM.exists(), reason="observation stream not present in this checkout"
)
def test_the_registered_observation_replays_from_its_durable_stream() -> None:
    replayed = HtrJournal().replay(FileTelemetrySink(KNOWLEDGE_STREAM).all_events())
    observations = replayed.research_observations()
    assert len(observations) == 1
    stored = observations[0]
    assert stored.affected_method == SATRN_METHOD_ID
    assert stored.affected_model_version == SATRN_MODEL_REVISION
    assert stored.scope.covered_unit_ids == ALL_CROP_IDS
    assert stored.scope.sample_size == CROPS_IN_SAMPLE
    assert stored.source_experiment_run_id in stored.scope.experiment_run_ids
