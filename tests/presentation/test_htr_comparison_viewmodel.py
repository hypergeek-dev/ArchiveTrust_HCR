"""Comparison ViewModel: side-by-side rows, real metrics, and the five never-collapsed stages."""

from __future__ import annotations

from archivetrust.htr.evaluation.recognition import compute_recognition_metrics
from archivetrust.htr.research_store import MethodRunTranscript
from archivetrust.presentation.htr_comparison_viewmodel import STAGE_ORDER, ComparisonViewModel
from tests.presentation._htr_fixtures import SYNTHETIC_FIXTURE_LINE_0, build_fixture_corpus


def _cell(row, method_id: str):
    return next(cell for cell in row.cells if cell.method_id == method_id)


def test_line_row_has_one_cell_per_method_that_ran_on_that_crop() -> None:
    corpus = build_fixture_corpus()
    row = ComparisonViewModel(corpus.store).line_comparison(corpus.line_0_id)

    assert row is not None
    assert {cell.method_id for cell in row.cells} == {
        "satrn",
        "florence2_htr",
        "transkribus_swedish_lion_1",
    }
    assert row.ground_truth == SYNTHETIC_FIXTURE_LINE_0
    assert row.page_id == corpus.page_id


def test_all_five_stages_are_always_present_as_labeled_entries() -> None:
    """Even when a stage has no text, it appears with `present=False` -- the surface must never
    silently omit a stage, because an omitted stage is indistinguishable from an empty one."""
    corpus = build_fixture_corpus()
    row = ComparisonViewModel(corpus.store).line_comparison(corpus.line_0_id)

    for cell in row.cells:
        assert tuple(stage.stage for stage in cell.stages) == STAGE_ORDER
        assert all(stage.label for stage in cell.stages)


def test_a_missing_stage_is_never_back_filled_from_an_adjacent_stage() -> None:
    """Florence-2's fixture run has no reviewed text and was not canonically selected. Both stages
    must read as absent, not inherit the normalized text."""
    corpus = build_fixture_corpus()
    row = ComparisonViewModel(corpus.store).line_comparison(corpus.line_0_id)
    cell = _cell(row, "florence2_htr")

    by_stage = {stage.stage: stage for stage in cell.stages}
    assert by_stage["normalized"].present is True
    assert by_stage["normalized"].text == "till den 23 Januari"
    assert by_stage["reviewed"].present is False
    assert by_stage["reviewed"].text is None
    assert by_stage["canonical"].present is False
    assert by_stage["canonical"].text is None


def test_raw_and_parsed_stay_distinct_when_parsing_actually_changed_the_text() -> None:
    """Florence-2's raw output carries its decoder's special tokens; the parsed stage does not.
    Collapsing the two would hide exactly the parsing step a researcher needs to audit."""
    corpus = build_fixture_corpus()
    row = ComparisonViewModel(corpus.store).line_comparison(corpus.line_0_id)
    by_stage = {stage.stage: stage for stage in _cell(row, "florence2_htr").stages}

    assert by_stage["raw"].text == "</s><s>till den 23 Januari</s>"
    assert by_stage["parsed"].text == "till den 23 Januari"
    assert by_stage["raw"].text != by_stage["parsed"].text


def test_reviewed_stage_carries_the_reviewer_as_its_attribution_not_the_method() -> None:
    corpus = build_fixture_corpus()
    row = ComparisonViewModel(corpus.store).line_comparison(corpus.line_0_id)
    by_stage = {stage.stage: stage for stage in _cell(row, "satrn").stages}

    assert by_stage["reviewed"].present is True
    assert by_stage["reviewed"].attribution == "reviewer_a"
    assert by_stage["raw"].attribution == "SATRN (Riksarkivet)"


def test_canonical_stage_appears_only_on_the_method_run_the_strategy_selected() -> None:
    corpus = build_fixture_corpus()
    row = ComparisonViewModel(corpus.store).line_comparison(corpus.line_0_id)

    selected = [cell.method_id for cell in row.cells if cell.selected_as_canonical]
    assert selected == ["satrn"]

    satrn_canonical = {s.stage: s for s in _cell(row, "satrn").stages}["canonical"]
    assert satrn_canonical.present is True
    assert satrn_canonical.attribution == "Human-approved selection"
    assert row.canonical_strategy_label == "Human-approved selection"


def test_metrics_match_the_recognition_engine_exactly_and_are_not_recomputed_here() -> None:
    corpus = build_fixture_corpus()
    row = ComparisonViewModel(corpus.store).line_comparison(corpus.line_0_id)
    cell = _cell(row, "florence2_htr")

    expected = compute_recognition_metrics(SYNTHETIC_FIXTURE_LINE_0, "till den 23 Januari")
    assert cell.character_error_rate_normalized == expected.character_error_rate_normalized
    assert cell.character_error_rate_raw == expected.character_error_rate_raw
    assert cell.word_error_rate_normalized == expected.word_error_rate_normalized
    assert cell.exact_match_normalized is False
    assert cell.edit_ops is not None
    assert cell.edit_ops.char_deletions == expected.char_edits_normalized.deletions
    assert cell.edit_ops.word_substitutions == expected.word_edits_normalized.substitutions


def test_an_exactly_correct_method_scores_zero_error_and_exact_match() -> None:
    corpus = build_fixture_corpus()
    row = ComparisonViewModel(corpus.store).line_comparison(corpus.line_0_id)
    cell = _cell(row, "satrn")

    assert cell.character_error_rate_normalized == 0.0
    assert cell.word_error_rate_normalized == 0.0
    assert cell.exact_match_normalized is True


def test_metrics_are_none_without_ground_truth_never_zero() -> None:
    """A CER of 0.0 means "perfect"; a CER of None means "there was nothing to compare against".
    Conflating them would report an unmeasured method as flawless."""
    corpus = build_fixture_corpus()
    store = corpus.store
    line = store.text_line(corpus.line_0_id)
    # Rebuild a store without ground truth rather than mutating the fixture's.
    from archivetrust.htr.research_store import HtrResearchStore

    bare = HtrResearchStore()
    bare.register_region(store.region(line.region_id))
    bare.register_text_line(line)
    bare.register_input_crop(store.input_crop(corpus.crop_a_id))
    run = store.method_run(corpus.satrn_run_line_0)
    bare.register_method_run(run)
    bare.register_transcript(store.transcript(corpus.satrn_run_line_0))

    row = ComparisonViewModel(bare).line_comparison(corpus.line_0_id)
    cell = _cell(row, "satrn")
    assert cell.character_error_rate_normalized is None
    assert cell.edit_ops is None


def test_a_failed_run_is_shown_with_its_reason_not_dropped_from_the_row() -> None:
    """`docs/htr-domain-design.md` §1: FailureRecord is preserved, never excluded."""
    corpus = build_fixture_corpus()
    row = ComparisonViewModel(corpus.store).line_comparison(corpus.line_1_id)

    failed = _cell(row, "florence2_htr")
    assert failed.failed is True
    assert failed.outcome_label == "Failed"
    assert failed.failure_reasons == (
        "CUDA out of memory while loading the fine-tuned checkpoint",
    )
    # ...and it is still a cell in the row, beside the method that succeeded.
    assert {cell.method_id for cell in row.cells} == {"satrn", "florence2_htr"}


def test_one_shared_crop_hash_marks_the_row_a_controlled_comparison() -> None:
    corpus = build_fixture_corpus()
    row = ComparisonViewModel(corpus.store).line_comparison(corpus.line_0_id)

    assert row.source_crop_hashes == (corpus.crop_a_hash,)
    assert row.controlled_comparison is True


def test_confidence_and_time_are_none_unless_actually_supplied() -> None:
    corpus = build_fixture_corpus()

    without = ComparisonViewModel(corpus.store).line_comparison(corpus.line_0_id)
    assert _cell(without, "satrn").confidence is None
    assert _cell(without, "satrn").processing_time_ms is None

    with_data = ComparisonViewModel(
        corpus.store,
        run_confidences={corpus.satrn_run_line_0: 0.6666},
        run_times_ms={corpus.satrn_run_line_0: 1310.0},
    ).line_comparison(corpus.line_0_id)
    assert _cell(with_data, "satrn").confidence == 0.6666
    assert _cell(with_data, "satrn").processing_time_ms == 1310.0


def test_metrics_fall_back_through_parsed_to_raw_when_later_stages_are_absent() -> None:
    """A method with only a raw stage must still be measurable, rather than silently scoring None
    because normalization never ran."""
    corpus = build_fixture_corpus()
    corpus.store._transcripts[corpus.transkribus_run_line_0] = MethodRunTranscript(  # noqa: SLF001
        method_run_id=corpus.transkribus_run_line_0,
        raw_text="till den 23 Januarii",
    )

    row = ComparisonViewModel(corpus.store).line_comparison(corpus.line_0_id)
    cell = _cell(row, "transkribus_swedish_lion_1")
    by_stage = {s.stage: s for s in cell.stages}
    assert by_stage["normalized"].present is False
    assert cell.character_error_rate_normalized == 0.0


def test_page_comparison_returns_every_line_in_reading_order() -> None:
    corpus = build_fixture_corpus()
    rows = ComparisonViewModel(corpus.store).page_comparison(corpus.page_id)

    assert [row.reading_order_index for row in rows] == [0, 1]


def test_legacy_method_runs_are_flagged_so_they_are_never_read_as_htr_baselines() -> None:
    corpus = build_fixture_corpus()
    vm = ComparisonViewModel(corpus.store, legacy_method_ids=frozenset({"satrn"}))
    row = vm.line_comparison(corpus.line_0_id)

    assert _cell(row, "satrn").is_legacy_method is True
    assert _cell(row, "florence2_htr").is_legacy_method is False


def test_unknown_line_returns_none() -> None:
    corpus = build_fixture_corpus()
    assert ComparisonViewModel(corpus.store).line_comparison("nope") is None
