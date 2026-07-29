"""Offscreen Qt construction tests for the HTR research pages (Stage 11).

Same convention as `tests/clients/test_qt_app.py`: the offscreen platform is selected before any
`QApplication` exists, and PySide6 is an optional `[gui]` extra so the module skips entirely when it
is absent.

These assert that each page constructs against a real `AppContext`, binds its real ViewModel, and
renders honest content in both the empty and populated cases. They deliberately do NOT assert
anything about appearance -- no display is available in this environment, so pixel layout is not
verifiable here and is not claimed to be.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from archivetrust.clients.desktop_v2.htr_pages import (  # noqa: E402
    NOTHING_REGISTERED,
    ComparisonPage,
    DatasetsPage,
    EvidenceChainPage,
    ExperimentsPage,
    MethodsPage,
    ResearchOverviewPage,
    ResearchReportExporter,
    ReviewCenterPage,
)
from archivetrust.composition import AppContext  # noqa: E402
from archivetrust.presentation.desktop_v2_pages import DesktopV2Page  # noqa: E402
from archivetrust.presentation.htr_experiment_builder_viewmodel import (  # noqa: E402
    EvaluationMode,
)
from archivetrust.review.blind_review.assignment import create_blind_review_pair  # noqa: E402
from archivetrust.review.htr_models import ReviewSubmission  # noqa: E402
from tests.presentation._htr_fixtures import build_fixture_corpus  # noqa: E402


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


def _context_with_corpus(tmp_path) -> AppContext:
    """A real `AppContext` whose research store holds the shared fixture corpus."""
    context = AppContext(deployment_root=tmp_path)
    corpus = build_fixture_corpus()
    context._htr_research_store = corpus.store  # noqa: SLF001
    context._fixture = corpus  # type: ignore[attr-defined]  # test-only handle on the ids
    return context


def test_every_research_page_constructs_against_an_empty_store(qapp, tmp_path) -> None:
    """A fresh launch has no HTR entities registered; no page may crash or render blank."""
    context = AppContext(deployment_root=tmp_path)
    for page_class in (
        ResearchOverviewPage,
        MethodsPage,
        DatasetsPage,
        ExperimentsPage,
        ComparisonPage,
        EvidenceChainPage,
        ReviewCenterPage,
    ):
        page = page_class(context)
        assert page is not None


def test_research_overview_shows_real_counts_from_the_store(qapp, tmp_path) -> None:
    from PySide6.QtWidgets import QLabel

    context = _context_with_corpus(tmp_path)
    page = ResearchOverviewPage(context)
    text = " ".join(label.text() for label in page.findChildren(QLabel))

    assert "Research projects" in text
    assert "Unresolved disagreements" in text
    # The review store is wired but has no targets -- that must be stated, not shown as zero.
    assert "no review targets are registered" in text


def test_empty_overview_states_that_nothing_is_registered(qapp, tmp_path) -> None:
    from PySide6.QtWidgets import QLabel

    context = AppContext(deployment_root=tmp_path)
    page = ResearchOverviewPage(context)
    text = " ".join(label.text() for label in page.findChildren(QLabel))

    assert NOTHING_REGISTERED in text


def test_methods_page_lists_the_three_real_methods_without_probing(qapp, tmp_path) -> None:
    from PySide6.QtWidgets import QLabel, QTableWidget

    context = AppContext(deployment_root=tmp_path)
    page = MethodsPage(context)

    cells = []
    for table in page.findChildren(QTableWidget):
        for row in range(table.rowCount()):
            for column in range(table.columnCount()):
                item = table.item(row, column)
                if item is not None:
                    cells.append(item.text())
    joined = " | ".join(cells)

    assert "SATRN (Riksarkivet)" in joined
    assert "Florence-2 (vlm-htr line OCR)" in joined
    assert "Transkribus Swedish Lion I" in joined
    # Unprobed environments read as "Not probed", never as "No".
    assert "Not probed" in joined

    labels = " ".join(label.text() for label in page.findChildren(QLabel))
    assert "Known limitations" in labels
    assert "not that a method is unavailable" in labels


def test_methods_page_probes_environments_only_when_asked(qapp, tmp_path) -> None:
    context = AppContext(deployment_root=tmp_path)
    page = MethodsPage(context)
    assert page._probe is False  # noqa: SLF001

    page._on_probe()  # noqa: SLF001
    assert page._probe is True  # noqa: SLF001


def test_datasets_page_renders_the_corpus_tree_and_line_detail(qapp, tmp_path) -> None:
    context = _context_with_corpus(tmp_path)
    corpus = context._fixture  # type: ignore[attr-defined]
    page = DatasetsPage(context)

    assert page._tree.topLevelItemCount() == 1  # noqa: SLF001
    root = page._tree.topLevelItem(0)  # noqa: SLF001
    assert root.text(0) == "Swedish Historical HTR"

    # Expand down to the lines, exercising the lazy child loader at every level:
    # project -> dataset -> version -> collection -> document -> page -> region -> text_line.
    node = root
    for expected_kind in (
        "dataset",
        "dataset_version",
        "collection",
        "document",
        "page",
        "region",
        "text_line",
    ):
        page._tree.expandItem(node)  # noqa: SLF001
        assert node.childCount() >= 1
        node = node.child(0)
        from PySide6.QtCore import Qt

        assert node.data(0, Qt.ItemDataRole.UserRole + 1) == expected_kind

    page._on_clicked(node)  # noqa: SLF001
    detail = page._detail.text()  # noqa: SLF001
    assert "Ground truth: till den 23 Januarii" in detail
    assert "SATRN (Riksarkivet)" in detail
    assert "byte-identical" in detail
    assert corpus.line_0_id  # the fixture id exists; the tree reached it


def test_empty_datasets_page_says_so_rather_than_showing_an_empty_tree(qapp, tmp_path) -> None:
    context = AppContext(deployment_root=tmp_path)
    page = DatasetsPage(context)

    assert page._tree.topLevelItemCount() == 1  # noqa: SLF001
    assert page._tree.topLevelItem(0).text(1) == NOTHING_REGISTERED  # noqa: SLF001


def test_comparison_page_renders_all_five_stages_per_method(qapp, tmp_path) -> None:
    context = _context_with_corpus(tmp_path)
    page = ComparisonPage(context)

    assert page._page_box.count() == 1  # noqa: SLF001
    assert page._tree.topLevelItemCount() == 2  # noqa: SLF001  (two lines)

    line = page._tree.topLevelItem(0)  # noqa: SLF001
    assert "Controlled comparison" in line.text(2)

    method_items = [line.child(i) for i in range(line.childCount())]
    satrn = next(item for item in method_items if item.text(0) == "SATRN (Riksarkivet)")
    stage_labels = [satrn.child(i).text(0) for i in range(satrn.childCount())]
    for expected in ("Raw output", "Parsed", "Normalized", "Human-reviewed", "Canonical"):
        assert expected in stage_labels


def test_comparison_page_shows_a_failed_run_with_its_reason(qapp, tmp_path) -> None:
    context = _context_with_corpus(tmp_path)
    page = ComparisonPage(context)

    line = page._tree.topLevelItem(1)  # noqa: SLF001  (line 1 has the failed Florence-2 run)
    method_items = [line.child(i) for i in range(line.childCount())]
    florence = next(
        item for item in method_items if item.text(0) == "Florence-2 (vlm-htr line OCR)"
    )
    assert florence.text(1) == "Failed"
    children = [florence.child(i).text(1) for i in range(florence.childCount())]
    assert any("CUDA out of memory" in text for text in children)
    # A stage that was never produced says so rather than being blank.
    assert "Not produced at this stage" in children


def test_evidence_chain_page_renders_the_full_chain_for_a_method_run(qapp, tmp_path) -> None:
    context = _context_with_corpus(tmp_path)
    page = EvidenceChainPage(context)

    assert page._run_box.count() == 5  # noqa: SLF001
    entries = [page._chain.item(i).text() for i in range(page._chain.count())]  # noqa: SLF001
    joined = " ".join(entries)

    assert "method_run" in joined
    assert "input_crop" in joined
    assert "project: Swedish Historical HTR" in joined
    assert "Complete chain" in page._status.text()  # noqa: SLF001


def test_empty_evidence_chain_page_states_the_absence(qapp, tmp_path) -> None:
    context = AppContext(deployment_root=tmp_path)
    page = EvidenceChainPage(context)

    assert page._run_box.count() == 0  # noqa: SLF001
    assert page._status.text() == NOTHING_REGISTERED  # noqa: SLF001


def test_experiments_page_builds_a_real_experiment_into_the_store(qapp, tmp_path) -> None:
    context = _context_with_corpus(tmp_path)
    store = context.htr_research_store
    before = len(store.experiments())

    page = ExperimentsPage(context)
    assert page._project.count() == 1  # noqa: SLF001
    assert page._dataset_version.count() == 1  # noqa: SLF001

    page._name.setText("Qt-built baseline")  # noqa: SLF001
    page._segmentation.setText("segmentation_baseline_v1")  # noqa: SLF001
    page._method_boxes[0].setChecked(True)  # noqa: SLF001
    assert page._create.isEnabled()  # noqa: SLF001

    page._on_create()  # noqa: SLF001
    assert len(store.experiments()) == before + 1
    assert any(e.name == "Qt-built baseline" for e in store.experiments())
    assert "Created experiment" in page._status.text()  # noqa: SLF001


def test_experiments_page_disables_create_and_lists_the_reasons(qapp, tmp_path) -> None:
    context = _context_with_corpus(tmp_path)
    page = ExperimentsPage(context)

    assert page._create.isEnabled() is False  # noqa: SLF001
    assert "An experiment needs a name." in page._issues.text()  # noqa: SLF001


def test_experiments_page_end_to_end_mode_drops_the_segmentation_requirement(
    qapp, tmp_path
) -> None:
    context = _context_with_corpus(tmp_path)
    page = ExperimentsPage(context)
    page._name.setText("End-to-end run")  # noqa: SLF001
    page._method_boxes[0].setChecked(True)  # noqa: SLF001
    assert page._create.isEnabled() is False  # noqa: SLF001

    index = page._mode.findData(EvaluationMode.END_TO_END_METHOD_SEGMENTATION)
    page._mode.setCurrentIndex(index)  # noqa: SLF001
    assert page._create.isEnabled() is True  # noqa: SLF001


def test_review_center_shows_only_the_selected_reviewers_own_queue(qapp, tmp_path) -> None:
    context = AppContext(deployment_root=tmp_path)
    a, b = create_blind_review_pair(
        target_ref="line_1",
        reviewer_a_ref="reviewer_a",
        reviewer_b_ref="reviewer_b",
        convention_id="c",
        convention_version=1,
        assigned_at="2026-07-29T10:00:00Z",
    )
    context.blind_review_store.register_blind_pair(a, b)

    page = ReviewCenterPage(context)
    page._reviewer.setText("reviewer_a")  # noqa: SLF001
    rows = [page._queue.item(i).text() for i in range(page._queue.count())]  # noqa: SLF001

    assert len(rows) == 1
    assert "reviewer_a" in rows[0]
    assert "reviewer_b" not in rows[0]


def test_review_center_hides_the_other_reading_until_both_have_submitted(qapp, tmp_path) -> None:
    context = AppContext(deployment_root=tmp_path)
    a, b = create_blind_review_pair(
        target_ref="line_1",
        reviewer_a_ref="reviewer_a",
        reviewer_b_ref="reviewer_b",
        convention_id="c",
        convention_version=1,
        assigned_at="2026-07-29T10:00:00Z",
    )
    context.blind_review_store.register_blind_pair(a, b)
    context.review_center_viewmodel().submit(
        ReviewSubmission.create(
            assignment_id=b.assignment_id,
            reviewer_ref="reviewer_b",
            submitted_value="waritt i Stockholm",
            submitted_at="2026-07-29T11:00:00Z",
        )
    )

    page = ReviewCenterPage(context)
    page._reviewer.setText("reviewer_a")  # noqa: SLF001
    page._queue.setCurrentRow(0)  # noqa: SLF001
    page._on_selected(page._queue.currentItem())  # noqa: SLF001

    detail = page._detail.toPlainText()  # noqa: SLF001
    assert "still pending" in detail
    assert "waritt i Stockholm" not in detail  # reviewer B's reading has not leaked


def test_review_center_shows_disagreements_once_both_have_submitted(qapp, tmp_path) -> None:
    context = AppContext(deployment_root=tmp_path)
    a, b = create_blind_review_pair(
        target_ref="line_1",
        reviewer_a_ref="reviewer_a",
        reviewer_b_ref="reviewer_b",
        convention_id="c",
        convention_version=1,
        assigned_at="2026-07-29T10:00:00Z",
    )
    context.blind_review_store.register_blind_pair(a, b)
    vm = context.review_center_viewmodel()
    vm.submit(
        ReviewSubmission.create(
            assignment_id=a.assignment_id,
            reviewer_ref="reviewer_a",
            submitted_value="waritt i Stockholm den tjugonde",
            submitted_at="2026-07-29T11:00:00Z",
        )
    )
    vm.submit(
        ReviewSubmission.create(
            assignment_id=b.assignment_id,
            reviewer_ref="reviewer_b",
            submitted_value="warit uti Goeteborg den tjugonde",
            submitted_at="2026-07-29T11:05:00Z",
        )
    )

    page = ReviewCenterPage(context)
    page._reviewer.setText("reviewer_a")  # noqa: SLF001
    page._queue.setCurrentRow(0)  # noqa: SLF001
    page._on_selected(page._queue.currentItem())  # noqa: SLF001

    detail = page._detail.toPlainText()  # noqa: SLF001
    assert "Disagreements:" in detail
    assert "Goeteborg" in detail


def test_review_center_refuses_a_blank_exclusion_reason_and_shows_why(qapp, tmp_path) -> None:
    context = AppContext(deployment_root=tmp_path)
    a, b = create_blind_review_pair(
        target_ref="line_1",
        reviewer_a_ref="reviewer_a",
        reviewer_b_ref="reviewer_b",
        convention_id="c",
        convention_version=1,
        assigned_at="2026-07-29T10:00:00Z",
    )
    context.blind_review_store.register_blind_pair(a, b)

    page = ReviewCenterPage(context)
    page._reviewer.setText("reviewer_a")  # noqa: SLF001
    page._queue.setCurrentRow(0)  # noqa: SLF001
    page._rationale.setText("")  # noqa: SLF001
    page._on_exclude()  # noqa: SLF001

    assert "reason is required" in page._status.text()  # noqa: SLF001
    assert context.blind_review_store.exclusion_for("line_1") is None

    page._rationale.setText("The source scan is corrupt below the fold.")  # noqa: SLF001
    page._on_exclude()  # noqa: SLF001
    assert context.blind_review_store.exclusion_for("line_1") is not None


def test_review_center_states_the_correction_time_gap(qapp, tmp_path) -> None:
    context = AppContext(deployment_root=tmp_path)
    page = ReviewCenterPage(context)
    assert "correction time is not tracked" in page._status.text()  # noqa: SLF001


def test_report_exporter_writes_json_and_csv(qapp, tmp_path) -> None:
    from archivetrust.research.reports.export import report_from_json
    from archivetrust.research.reports.models import ResearchReport

    context = AppContext(deployment_root=tmp_path)
    report = ResearchReport.create(
        title="Swedish Historical HTR Baseline Comparison",
        generated_at="2026-07-29T12:00:00Z",
        experiment_run_ids=("experiment_run_1",),
    )
    exporter = ResearchReportExporter(context)

    json_path = exporter.export_to_path(report, str(tmp_path / "r.json"), export_format="json")
    csv_path = exporter.export_to_path(report, str(tmp_path / "r.csv"), export_format="csv")

    assert report_from_json(json_path.read_text(encoding="utf-8")) == report
    assert "experiment_run_1" in csv_path.read_text(encoding="utf-8")


def test_research_pages_are_grouped_ahead_of_the_operational_ones(qapp) -> None:
    from archivetrust.presentation.desktop_v2_shell_viewmodel import DesktopV2ShellViewModel

    shell = DesktopV2ShellViewModel()
    pages = list(shell.pages)
    research = shell.research_pages()
    operational = shell.operational_pages()

    assert len(research) == 7
    assert set(research) | set(operational) == set(DesktopV2Page)
    assert not set(research) & set(operational)
    # Every research page comes before every operational one in rail order.
    assert max(pages.index(p) for p in research) < min(pages.index(p) for p in operational)
