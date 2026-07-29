"""Research dashboard ViewModel: real counts, honest absences, legacy runs held out."""

from __future__ import annotations

from archivetrust.htr.experiment.models import ExperimentRun, MethodRun
from archivetrust.htr.research_store import HtrResearchStore
from archivetrust.presentation.htr_dashboard_viewmodel import ResearchDashboardViewModel
from archivetrust.presentation.htr_review_center_viewmodel import ReviewCenterViewModel
from archivetrust.review.blind_review.assignment import create_blind_review_pair
from archivetrust.review.blind_review.store import BlindReviewStore
from archivetrust.review.htr_models import ReviewSubmission
from tests.presentation._htr_fixtures import build_fixture_corpus

COMPUTED_AT = "2026-07-29T12:00:00Z"


def test_corpus_counts_come_from_the_registered_entities() -> None:
    corpus = build_fixture_corpus()
    snapshot = ResearchDashboardViewModel(corpus.store).snapshot(computed_at=COMPUTED_AT)

    assert [p.name for p in snapshot.projects] == ["Swedish Historical HTR"]
    assert snapshot.projects[0].dataset_count == 1
    assert snapshot.projects[0].experiment_count == 1
    assert snapshot.dataset_count == 1
    assert snapshot.dataset_version_count == 1
    assert snapshot.documents_in_corpus == 1
    assert snapshot.pages_in_corpus == 1
    assert snapshot.regions_in_corpus == 1
    assert snapshot.text_lines_in_corpus == 2
    assert snapshot.lines_with_ground_truth == 2
    assert snapshot.method_run_count == 5


def test_experiment_status_reflects_completed_versus_in_progress_runs() -> None:
    corpus = build_fixture_corpus()
    vm = ResearchDashboardViewModel(corpus.store)

    row = vm.snapshot(computed_at=COMPUTED_AT).experiments[0]
    assert row.run_count == 1
    assert row.completed_run_count == 1
    assert row.in_progress_run_count == 0
    assert row.status == "1 run(s) complete"
    assert row.latest_run_completed_at == "2026-07-03T01:30:00Z"

    corpus.store.register_experiment_run(
        ExperimentRun.create(
            experiment_version_id=corpus.experiment_version_id,
            is_end_to_end=True,
            started_at="2026-07-04T00:00:00Z",
        )
    )
    updated = vm.snapshot(computed_at=COMPUTED_AT).experiments[0]
    assert updated.run_count == 2
    assert updated.in_progress_run_count == 1
    assert updated.status == "1 run(s) in progress"


def test_an_experiment_with_no_runs_says_so_rather_than_showing_zero_percent() -> None:
    from archivetrust.htr.experiment.models import Experiment

    corpus = build_fixture_corpus()
    corpus.store.register_experiment(
        Experiment.create(
            name="Not started yet",
            research_project_id=corpus.project_id,
            created_at="2026-07-05T00:00:00Z",
        )
    )
    rows = ResearchDashboardViewModel(corpus.store).snapshot(computed_at=COMPUTED_AT).experiments
    fresh = next(row for row in rows if row.name == "Not started yet")

    assert fresh.status == "No runs yet"
    assert fresh.latest_run_id is None


def test_method_failures_count_failed_runs_into_the_total_not_out_of_it() -> None:
    corpus = build_fixture_corpus()
    snapshot = ResearchDashboardViewModel(corpus.store).snapshot(computed_at=COMPUTED_AT)
    by_method = {row.method_id: row for row in snapshot.method_failures}

    assert by_method["florence2_htr"].total_runs == 2
    assert by_method["florence2_htr"].succeeded_runs == 1
    assert by_method["florence2_htr"].failed_runs == 1
    assert by_method["florence2_htr"].failure_categories == {"cuda_oom": 1}
    assert by_method["florence2_htr"].method_display_name == "Florence-2 (vlm-htr line OCR)"

    assert by_method["satrn"].failed_runs == 0
    assert by_method["satrn"].failure_categories == {}


def test_legacy_ocr_runs_are_held_out_of_aggregates_and_counted_separately() -> None:
    """`docs/htr-domain-design.md` §8: legacy runs are never in HTR baseline statistics, and never
    invisible either."""
    corpus = build_fixture_corpus()
    corpus.store.register_method_run(
        MethodRun.create(
            experiment_run_id=corpus.experiment_run_id,
            method_id="docling",
            evidence_id="evidence_legacy",
            outcome="succeeded",
            started_at="2026-06-01T00:00:00Z",
            input_crop_id=corpus.crop_a_id,
        )
    )
    snapshot = ResearchDashboardViewModel(corpus.store).snapshot(computed_at=COMPUTED_AT)

    assert snapshot.legacy_method_runs == 1
    assert snapshot.method_run_count == 5  # unchanged: the legacy run is not counted as HTR
    assert "docling" not in {row.method_id for row in snapshot.method_failures}


def test_transkribus_external_imports_are_listed_with_their_vendor_accuracy_kept_apart() -> None:
    corpus = build_fixture_corpus()
    snapshot = ResearchDashboardViewModel(corpus.store).snapshot(computed_at=COMPUTED_AT)

    assert len(snapshot.external_imports) == 1
    row = snapshot.external_imports[0]
    assert row.export_format == "page_xml"
    assert row.imported_by == "researcher_1"
    assert row.transkribus_document_id == "12345"
    assert row.vendor_reported_accuracy == 0.94
    # It is reported on its own field, never merged into a method_failures/CER figure.
    assert not hasattr(snapshot.method_failures[0], "vendor_reported_accuracy")


def test_review_figures_are_none_with_a_reason_when_no_review_store_is_wired() -> None:
    """A dashboard showing `0 unresolved disagreements` when it has no review store at all would
    be a lie of omission."""
    corpus = build_fixture_corpus()
    snapshot = ResearchDashboardViewModel(corpus.store).snapshot(computed_at=COMPUTED_AT)

    assert snapshot.reviewed_item_count is None
    assert snapshot.unresolved_disagreement_count is None
    assert snapshot.pending_review_count is None
    assert snapshot.review_status_note is not None
    assert "unknown rather than zero" in snapshot.review_status_note


def test_a_wired_review_store_with_no_targets_says_that_distinctly() -> None:
    corpus = build_fixture_corpus()
    snapshot = ResearchDashboardViewModel(
        corpus.store, review_center=ReviewCenterViewModel(BlindReviewStore())
    ).snapshot(computed_at=COMPUTED_AT)

    assert snapshot.reviewed_item_count is None
    assert "no review targets are registered" in snapshot.review_status_note


def test_real_review_figures_are_read_from_the_blind_review_store() -> None:
    corpus = build_fixture_corpus()
    review_store = BlindReviewStore()
    a1, b1 = create_blind_review_pair(
        target_ref=corpus.line_0_id,
        reviewer_a_ref="reviewer_a",
        reviewer_b_ref="reviewer_b",
        convention_id="c",
        convention_version=1,
        assigned_at="2026-07-29T10:00:00Z",
    )
    review_store.register_blind_pair(a1, b1)
    a2, b2 = create_blind_review_pair(
        target_ref=corpus.line_1_id,
        reviewer_a_ref="reviewer_a",
        reviewer_b_ref="reviewer_b",
        convention_id="c",
        convention_version=1,
        assigned_at="2026-07-29T10:00:00Z",
    )
    review_store.register_blind_pair(a2, b2)

    center = ReviewCenterViewModel(review_store)
    for assignment, value, illegible in (
        (a1, "till den 23 Januarii", False),
        (b1, "till den 23 Januarii", False),
        (a2, None, True),
        (b2, "waritt i Stockholm", False),
    ):
        center.submit(
            ReviewSubmission.create(
                assignment_id=assignment.assignment_id,
                reviewer_ref=assignment.reviewer_ref,
                submitted_value=value,
                submitted_at="2026-07-29T11:00:00Z",
                illegible=illegible,
            )
        )

    snapshot = ResearchDashboardViewModel(
        corpus.store,
        review_center=center,
        review_target_refs=(corpus.line_0_id, corpus.line_1_id),
    ).snapshot(computed_at=COMPUTED_AT)

    assert snapshot.reviewed_item_count == 2
    assert snapshot.unresolved_disagreement_count == 1  # the illegibility mismatch
    assert snapshot.pending_review_count == 0
    assert snapshot.excluded_from_benchmark_count == 0
    assert snapshot.review_status_note is None


def test_an_empty_store_produces_an_empty_but_valid_snapshot() -> None:
    snapshot = ResearchDashboardViewModel(HtrResearchStore()).snapshot(computed_at=COMPUTED_AT)

    assert snapshot.projects == ()
    assert snapshot.experiments == ()
    assert snapshot.method_run_count == 0
    assert snapshot.latest_completed_runs == ()


def test_experiment_run_detail_names_the_comparison_mode_and_the_manifest() -> None:
    corpus = build_fixture_corpus()
    detail = ResearchDashboardViewModel(corpus.store).experiment_run_detail(
        corpus.experiment_run_id
    )
    joined = " | ".join(detail)

    assert "Controlled (every method read the same input crops)" in joined
    assert "SATRN (Riksarkivet): Succeeded" in joined
    assert "Florence-2 (vlm-htr line OCR): Failed" in joined
    assert "0951e91" in joined


def test_unknown_experiment_run_detail_reports_the_absence() -> None:
    corpus = build_fixture_corpus()
    detail = ResearchDashboardViewModel(corpus.store).experiment_run_detail("nope")
    assert "not registered" in detail[0]
