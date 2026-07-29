"""Reliability classification + aggregation tests (docs/htr-migration-plan.md Stage 9).

Method-failure-passthrough tests reuse **real** adapter failure paths -- SATRN's `cuda_oom` (via
the same fake-facade seam `tests/providers/satrn/test_adapter_contract.py` uses) and Transkribus's
real `malformed_page.xml` fixture parsed by the real parser -- rather than inventing synthetic
failures, per the task brief's explicit instruction to reuse real failure paths where they exist.
The heuristic reliability checks (hallucination, omission, repetition, etc.) use small synthetic
strings engineered to isolate one behavior each, labeled as such; the confidence-calibration check
uses SATRN's real documented confidence (0.6666, `providers/satrn/README.md`) against the real
fixture ground truth.
"""

from __future__ import annotations

from pathlib import Path

from archivetrust.htr.evaluation.failures import (
    MethodRunEvaluationEntry,
    ReliabilityFlag,
    aggregate_method_run_metrics,
    classify_reliability,
)
from archivetrust.providers.htr_adapter import RecognitionInput
from archivetrust.providers.satrn.adapter import build_failure_record as satrn_build_failure_record
from archivetrust.providers.satrn.facade import SatrnWorkerResult
from archivetrust.providers.transkribus.adapter import TranskribusAdapter
from archivetrust.providers.transkribus.adapter import build_failure_record as transkribus_build_failure_record

TRANSKRIBUS_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "transkribus"

GROUND_TRUTH_TEXT = (
    Path(__file__).resolve().parents[2] / "fixtures" / "htr" / "trolldomskommissionen_sample_line.txt"
).read_text(encoding="utf-8").strip()
SATRN_REAL_OUTPUT = "till den 23 Januarii"
SATRN_REAL_CONFIDENCE = 0.6666  # providers/satrn/README.md "Measured real-inference behavior"


class _FakeSatrnFacade:
    """Mirrors `tests/providers/satrn/test_adapter_contract.py::_FakeFacade` -- a real
    `SatrnAdapter.recognize()` call through the adapter's real failure-handling code path, only
    the subprocess boundary is faked (no GPU/subprocess needed to exercise the real classification
    logic)."""

    def __init__(self, result: dict) -> None:
        self._result = result

    def run_inference(self, *, image_path, device_request, model_id, revision):
        return SatrnWorkerResult(self._result)


def _real_satrn_cuda_oom_failure():
    from archivetrust.providers.satrn.adapter import SatrnAdapter

    adapter = SatrnAdapter(
        facade=_FakeSatrnFacade({"ok": False, "category": "cuda_oom", "message": "CUDA out of memory."})
    )
    result = adapter.recognize(RecognitionInput(input_crop_id="/tmp/line.jpg"))
    failure = satrn_build_failure_record(result, method_run_id="method_run_satrn_1")
    assert failure is not None
    return failure


def _real_transkribus_malformed_xml_failure():
    adapter = TranskribusAdapter()
    configuration = {"export_file_path": str(TRANSKRIBUS_FIXTURES / "malformed_page.xml")}
    result = adapter.recognize(RecognitionInput(configuration=configuration))
    failure = transkribus_build_failure_record(result, method_run_id="method_run_transkribus_1")
    assert failure is not None
    return failure


# --- Method failure passthrough: real adapter failure paths, unchanged ------------------------


def test_real_satrn_cuda_oom_failure_passes_through_unchanged():
    real_failure = _real_satrn_cuda_oom_failure()
    assert real_failure.category == "cuda_oom"
    classified = classify_reliability(
        method_run_id="method_run_satrn_1",
        reference_text=GROUND_TRUTH_TEXT,
        raw_text=None,
        parsed_text=None,
        normalized_text=None,
        adapter_failure=real_failure,
    )
    assert classified == (real_failure,)  # identical object, not reclassified/rewrapped


def test_real_transkribus_malformed_xml_failure_passes_through_unchanged():
    real_failure = _real_transkribus_malformed_xml_failure()
    assert real_failure.category == "malformed_xml"
    classified = classify_reliability(
        method_run_id="method_run_transkribus_1",
        reference_text=None,
        raw_text=None,
        parsed_text=None,
        normalized_text=None,
        adapter_failure=real_failure,
    )
    assert classified == (real_failure,)


# --- Confidence-calibration disagreement: real SATRN confidence vs. real measured accuracy -----


def test_satrn_real_confidence_disagrees_with_real_measured_accuracy():
    """SATRN reports 0.6666 confidence (a real measured decode probability) on a line whose real
    output barely resembles the real ground truth (measured normalized similarity ~0.207,
    `test_recognition.py`) -- a genuine calibration failure, not a synthetic scenario."""
    flags = classify_reliability(
        method_run_id="method_run_satrn_1",
        reference_text=GROUND_TRUTH_TEXT,
        raw_text=SATRN_REAL_OUTPUT,
        parsed_text=SATRN_REAL_OUTPUT,
        normalized_text=SATRN_REAL_OUTPUT,
        reported_confidence=SATRN_REAL_CONFIDENCE,
    )
    categories = {f.category for f in flags}
    assert ReliabilityFlag.CONFIDENCE_CALIBRATION_DISAGREEMENT.value in categories


def test_well_calibrated_confidence_on_an_exact_match_is_not_flagged():
    flags = classify_reliability(
        method_run_id="method_run_1",
        reference_text="katt",
        raw_text="katt",
        parsed_text="katt",
        normalized_text="katt",
        reported_confidence=0.95,
    )
    assert flags == ()


# --- Empty / malformed output --------------------------------------------------------------


def test_empty_output_flagged_and_short_circuits_other_checks():
    flags = classify_reliability(
        method_run_id="method_run_1",
        reference_text="x",
        raw_text=None,
        parsed_text=None,
        normalized_text="   ",
    )
    assert len(flags) == 1
    assert flags[0].category == ReliabilityFlag.EMPTY_OUTPUT.value


def test_malformed_output_flagged_using_florence2s_real_raw_special_tokens_misused_as_parsed_text():
    """Florence-2's real raw decoder output (`providers/florence2_htr/README.md`'s "Raw decoder
    output" row, special tokens intact) is genuinely expected on `raw_output` -- but if it ever
    leaked into the *parsed*/*normalized* stage (a parsing-step bug), this must be flagged."""
    leaked_raw_text = "</s><s>Be Kindsf. SAGE och Doth Hoon Minnes Coon Pilla gonger werff</s>"
    flags = classify_reliability(
        method_run_id="method_run_1",
        reference_text=None,
        raw_text=leaked_raw_text,
        parsed_text=leaked_raw_text,
        normalized_text=leaked_raw_text,
    )
    categories = {f.category for f in flags}
    assert ReliabilityFlag.MALFORMED_OUTPUT.value in categories


def test_unsupported_normalization_flagged_when_normalized_text_is_not_nfc_plus_strip_of_parsed():
    flags = classify_reliability(
        method_run_id="method_run_1",
        reference_text=None,
        raw_text=None,
        parsed_text="Hej  där",
        normalized_text="HEJ DÄR",  # uppercased -- not explainable by NFC + strip alone
    )
    categories = {f.category for f in flags}
    assert ReliabilityFlag.UNSUPPORTED_NORMALIZATION.value in categories


def test_normalize_transcriptions_own_nfc_plus_strip_convention_is_not_flagged():
    flags = classify_reliability(
        method_run_id="method_run_1",
        reference_text=None,
        raw_text=None,
        parsed_text="  Hejdå  \n",
        normalized_text="Hejdå",  # exactly unicodedata.normalize("NFC", ...).strip()
    )
    assert flags == ()


# --- Line-break invention / loss -------------------------------------------------------------


def test_invented_line_breaks_flagged():
    flags = classify_reliability(
        method_run_id="method_run_1",
        reference_text="line1\nline2",
        raw_text=None,
        parsed_text=None,
        normalized_text="line1\nline1b\nline2",
    )
    categories = {f.category for f in flags}
    assert ReliabilityFlag.INVENTED_LINE_BREAKS.value in categories


def test_lost_line_breaks_flagged():
    flags = classify_reliability(
        method_run_id="method_run_1",
        reference_text="line1\nline2\nline3",
        raw_text=None,
        parsed_text=None,
        normalized_text="line1",
    )
    categories = {f.category for f in flags}
    assert ReliabilityFlag.LOST_LINE_BREAKS.value in categories


# --- Hallucination / omission / truncation (synthetic, engineered) ---------------------------


def test_hallucinated_text_flagged_when_hypothesis_is_mostly_invented_words():
    flags = classify_reliability(
        method_run_id="method_run_1",
        reference_text="en katt sitter",
        raw_text=None,
        parsed_text=None,
        normalized_text="en katt sitter helt still pa den gamla trasiga stolen i koket",
    )
    categories = {f.category for f in flags}
    assert ReliabilityFlag.HALLUCINATED_TEXT.value in categories


def test_omitted_text_flagged_when_most_of_the_reference_is_missing():
    flags = classify_reliability(
        method_run_id="method_run_1",
        reference_text="en katt sitter helt still pa stolen",
        raw_text=None,
        parsed_text=None,
        normalized_text="en katt",
    )
    categories = {f.category for f in flags}
    assert ReliabilityFlag.OMITTED_TEXT.value in categories


def test_output_truncation_flagged_when_hypothesis_is_a_short_exact_prefix():
    flags = classify_reliability(
        method_run_id="method_run_1",
        reference_text="en katt sitter helt still",
        raw_text=None,
        parsed_text=None,
        normalized_text="en katt sitter",
    )
    categories = {f.category for f in flags}
    assert ReliabilityFlag.OUTPUT_TRUNCATION.value in categories


# --- Reading-order error (same words, wrong order) --------------------------------------------


def test_reading_order_error_flagged_when_words_match_but_sequence_is_reversed():
    flags = classify_reliability(
        method_run_id="method_run_1",
        reference_text="alpha beta gamma delta epsilon",
        raw_text=None,
        parsed_text=None,
        normalized_text="epsilon delta gamma beta alpha",
    )
    categories = {f.category for f in flags}
    assert ReliabilityFlag.READING_ORDER_ERROR.value in categories


# --- Repeated text (degenerate repetition loop) -----------------------------------------------


def test_repeated_text_flagged_on_a_degenerate_word_repetition_loop():
    flags = classify_reliability(
        method_run_id="method_run_1",
        reference_text="hej där",
        raw_text=None,
        parsed_text=None,
        normalized_text="hej hej hej hej där",
    )
    categories = {f.category for f in flags}
    assert ReliabilityFlag.REPEATED_TEXT.value in categories


# --- Aggregation: failures are visible in the aggregate, never silently excluded --------------


def test_aggregate_includes_failed_runs_visibly_not_dropped():
    """The brief's core requirement: a failed run must change the reported all-in mean and must
    be counted, not silently excluded from the aggregate's denominator."""
    entries = (
        MethodRunEvaluationEntry(
            method_run_id="run_1", method_id="satrn", outcome="succeeded",
            character_error_rate_normalized=0.1,
        ),
        MethodRunEvaluationEntry(
            method_run_id="run_2", method_id="satrn", outcome="succeeded",
            character_error_rate_normalized=0.3,
        ),
        MethodRunEvaluationEntry(
            method_run_id="run_3", method_id="satrn", outcome="failed",
            failure_category="cuda_oom",
        ),
    )
    aggregates = aggregate_method_run_metrics(entries)
    assert len(aggregates) == 1
    aggregate = aggregates[0]
    assert aggregate.total_runs == 3
    assert aggregate.succeeded_runs == 2
    assert aggregate.failed_runs == 1
    assert aggregate.mean_cer_normalized_succeeded_only == 0.2  # (0.1 + 0.3) / 2
    # The failed run is folded in as CER 1.0 -- this must differ from the succeeded-only mean,
    # proving the failure is visible in the aggregate rather than dropped from it.
    assert aggregate.mean_cer_normalized_all_including_failures != aggregate.mean_cer_normalized_succeeded_only
    assert aggregate.mean_cer_normalized_all_including_failures == (0.1 + 0.3 + 1.0) / 3
    assert aggregate.failure_categories == {"cuda_oom": 1}


def test_aggregate_groups_by_method_id_independently():
    entries = (
        MethodRunEvaluationEntry(
            method_run_id="run_1", method_id="satrn", outcome="succeeded",
            character_error_rate_normalized=0.2,
        ),
        MethodRunEvaluationEntry(
            method_run_id="run_2", method_id="florence2_htr", outcome="failed",
            failure_category="malformed_output",
        ),
    )
    aggregates = {a.method_id: a for a in aggregate_method_run_metrics(entries)}
    assert aggregates["satrn"].failed_runs == 0
    assert aggregates["florence2_htr"].succeeded_runs == 0
    assert aggregates["florence2_htr"].mean_cer_normalized_all_including_failures == 1.0
    assert aggregates["florence2_htr"].mean_cer_normalized_succeeded_only is None


def test_aggregate_with_no_entries_for_a_method_is_simply_absent_not_a_crash():
    assert aggregate_method_run_metrics(()) == ()
