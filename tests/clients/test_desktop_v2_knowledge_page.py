"""Offscreen Qt construction tests for the Research Knowledge page (the eighth research surface).

Same convention as `tests/clients/test_desktop_v2_htr_pages.py`: the offscreen platform is selected
before any `QApplication` exists, and PySide6 is an optional `[gui]` extra so the module skips entirely
when it is absent.

These assert content and structure -- which rows exist, which text is present, what selecting a row
produces. They deliberately assert **nothing** about appearance: there is no display in the environment
this page was built in, so pixel layout is not verifiable here and is not claimed to be.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from archivetrust.clients.desktop_v2.app import MainWindowV2  # noqa: E402
from archivetrust.clients.desktop_v2.htr_knowledge_page import (  # noqa: E402
    EVIDENCE_PANEL_PLACEHOLDER,
    NOTHING_OBSERVED,
    ResearchKnowledgePage,
)
from archivetrust.composition import AppContext  # noqa: E402
from archivetrust.htr.knowledge.baseline_knowledge import (  # noqa: E402
    EXTRACTED_AT,
    SATRN_METHOD_ID,
)
from archivetrust.htr.knowledge.models import ObservationType  # noqa: E402
from archivetrust.htr.knowledge.registration import (  # noqa: E402
    demonstrate_review_workflow,
    register_baseline_knowledge,
    register_feedback_loop,
)
from archivetrust.htr.persistence import DurableHtrResearchStore  # noqa: E402
from archivetrust.infrastructure.storage.telemetry_sink import (  # noqa: E402
    InMemoryTelemetrySink,
)
from archivetrust.presentation.desktop_v2_pages import DesktopV2Page  # noqa: E402
from archivetrust.presentation.display_names import observation_type_label  # noqa: E402

REVIEWER = "hypergeek-dev"


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


def _context_with_knowledge(tmp_path) -> AppContext:
    """A real `AppContext` whose research store holds the real baseline knowledge, its two real
    transitions, and the real research question drafted from the confidence-anomaly observation."""
    context = AppContext(deployment_root=tmp_path)
    store = DurableHtrResearchStore(InMemoryTelemetrySink())
    knowledge = register_baseline_knowledge(store, at=EXTRACTED_AT)
    demonstrate_review_workflow(store, knowledge, reviewer=REVIEWER, at=EXTRACTED_AT)
    loop = register_feedback_loop(
        store,
        observation=knowledge.observations["satrn_confidence_disagreement"],
        finding=knowledge.findings["satrn_confidence_not_aligned"],
        created_by=REVIEWER,
        at=EXTRACTED_AT,
        caused_by=knowledge.finding_event_ids["satrn_confidence_not_aligned"],
    )
    context._htr_research_store = store  # noqa: SLF001
    context._knowledge = knowledge  # type: ignore[attr-defined]  # test-only handle
    context._loop = loop  # type: ignore[attr-defined]
    return context


def _section(page: ResearchKnowledgePage, prefix: str):
    tree = page._tree  # noqa: SLF001
    for index in range(tree.topLevelItemCount()):
        item = tree.topLevelItem(index)
        if item.text(0).startswith(prefix):
            return item
    raise AssertionError(
        f"no section heading starting {prefix!r}; found "
        f"{[tree.topLevelItem(i).text(0) for i in range(tree.topLevelItemCount())]}"
    )


def _descendants(item):
    for index in range(item.childCount()):
        child = item.child(index)
        yield child
        yield from _descendants(child)


# -- Registration in the shell ---------------------------------------------------------------------


def test_the_shell_registers_the_knowledge_page_and_resolves_it(qapp, tmp_path) -> None:
    """The eighth research page is in the exhaustive `_PAGE_CLASSES` table, has a rail entry, and
    builds -- the "fails loudly rather than falling through" pattern working for a new page."""
    window = MainWindowV2(AppContext(deployment_root=tmp_path))

    assert window._stack.count() == len(DesktopV2Page)  # noqa: SLF001
    assert DesktopV2Page.RESEARCH_KNOWLEDGE.title == "Research Knowledge"
    index = list(window._shell.pages).index(DesktopV2Page.RESEARCH_KNOWLEDGE)  # noqa: SLF001
    assert isinstance(window._stack.widget(index), ResearchKnowledgePage)  # noqa: SLF001

    window._shell.navigate_to(DesktopV2Page.RESEARCH_KNOWLEDGE)  # noqa: SLF001
    assert window._stack.currentIndex() == index  # noqa: SLF001
    assert window._rail.currentRow() == index  # noqa: SLF001


def test_the_knowledge_page_is_the_last_research_surface_in_rail_order(qapp) -> None:
    from archivetrust.presentation.desktop_v2_shell_viewmodel import DesktopV2ShellViewModel

    shell = DesktopV2ShellViewModel()
    research = shell.research_pages()

    assert DesktopV2Page.RESEARCH_KNOWLEDGE.is_research_surface
    assert research[-1] is DesktopV2Page.RESEARCH_KNOWLEDGE


# -- Empty state -----------------------------------------------------------------------------------


def test_an_empty_store_says_nothing_is_observed_rather_than_showing_an_empty_table(
    qapp, tmp_path
) -> None:
    page = ResearchKnowledgePage(AppContext(deployment_root=tmp_path))

    assert page._tree.topLevelItemCount() > 0  # noqa: SLF001 - the headings still render
    observations = _section(page, "Observations")
    assert observations.text(0) == "Observations (0)"
    # The empty child carries the recorded reason, not a blank.
    assert "extracted from durable" in observations.child(0).text(0)
    assert page._detail.toPlainText() == EVIDENCE_PANEL_PLACEHOLDER  # noqa: SLF001
    assert NOTHING_OBSERVED  # the constant a populated-then-emptied page renders


# -- Populated content -----------------------------------------------------------------------------


def test_the_page_renders_the_real_observations_with_their_type_labels(qapp, tmp_path) -> None:
    page = ResearchKnowledgePage(_context_with_knowledge(tmp_path))
    observations = _section(page, "Observations")

    assert observations.text(0) == "Observations (5)"
    labels = {observations.child(i).text(1) for i in range(observations.childCount())}
    assert observation_type_label(ObservationType.CONFIDENCE_ANOMALY) in labels
    assert observation_type_label(ObservationType.REPRODUCIBILITY_ANOMALY) in labels
    # Every observation states its sample size in the scope column.
    scopes = {observations.child(i).text(2) for i in range(observations.childCount())}
    assert scopes <= {"1 line_crop", "1 page"}


def test_findings_are_grouped_by_their_real_statuses_with_researcher_facing_labels(
    qapp, tmp_path
) -> None:
    page = ResearchKnowledgePage(_context_with_knowledge(tmp_path))

    assert _section(page, "Candidate findings").text(0).endswith("(3)")
    assert _section(page, "Provisionally supported findings").text(0).endswith("(1)")
    assert _section(page, "Disputed findings").text(0).endswith("(1)")
    assert _section(page, "Supported findings").text(0).endswith("(0)")

    provisional = _section(page, "Provisionally supported findings").child(0)
    assert provisional.text(1) == "Provisionally supported (not reproduced)"


def test_an_empty_status_section_shows_the_recorded_reason_it_is_empty(qapp, tmp_path) -> None:
    """"Supported (0)" beside an unexplained blank invites the reader to assume a bug. The reason is
    the lifecycle rule, and it is rendered."""
    page = ResearchKnowledgePage(_context_with_knowledge(tmp_path))
    supported = _section(page, "Supported findings")

    assert supported.childCount() == 1
    assert "reproduction evidence" in supported.child(0).text(0)


def test_the_unverified_hypothesis_is_its_own_labelled_row_not_part_of_the_description(
    qapp, tmp_path
) -> None:
    page = ResearchKnowledgePage(_context_with_knowledge(tmp_path))
    observations = _section(page, "Observations")
    anomaly = next(
        observations.child(i)
        for i in range(observations.childCount())
        if observations.child(i).text(1)
        == observation_type_label(ObservationType.REPRODUCIBILITY_ANOMALY)
    )
    rows = {child.text(0): child.text(1) for child in _descendants(anomaly)}

    assert "UNVERIFIED hypothesis (not a fact)" in rows
    assert "allocator" in rows["UNVERIFIED hypothesis (not a fact)"]
    assert "allocator" not in rows["Description"]


def test_a_finding_states_that_it_has_not_been_reproduced(qapp, tmp_path) -> None:
    page = ResearchKnowledgePage(_context_with_knowledge(tmp_path))
    provisional = _section(page, "Provisionally supported findings").child(0)
    rows = {child.text(0): child.text(1) for child in _descendants(provisional)}

    assert "has not been shown to hold in any other experiment run" in (
        rows["Reproduced in another run"]
    )


def test_a_disputed_finding_shows_both_its_history_and_the_contradiction(qapp, tmp_path) -> None:
    page = ResearchKnowledgePage(_context_with_knowledge(tmp_path))
    disputed = _section(page, "Disputed findings").child(0)
    texts = [child.text(0) for child in _descendants(disputed)]

    assert "Candidate (not yet reviewed) -> Under review" in texts
    assert "Under review -> Disputed" in texts
    assert any(text.startswith("Contradicted by research_observation") for text in texts)
    assert any(text.startswith("Limitations (") for text in texts)
    # The claim itself is still on the row, not replaced by the contradiction.
    assert "reproducible across sessions" in disputed.text(0)


def test_the_research_question_shows_its_hypothesis_and_that_the_draft_never_ran(
    qapp, tmp_path
) -> None:
    page = ResearchKnowledgePage(_context_with_knowledge(tmp_path))
    questions = _section(page, "Open research questions")

    assert questions.text(0).endswith("(1)")
    question = questions.child(0)
    assert "confidence" in question.text(0)
    assert question.text(1) == "Being investigated"
    rows = {child.text(0): (child.text(1), child.text(2)) for child in _descendants(question)}
    assert "Why it was asked" in rows
    assert "Raised by" in rows
    assert "Hypothesis" in rows
    assert "Refuted if" in rows
    assert rows["Drafted experiment"][1] == "Drafted only -- never executed"


def test_the_pattern_table_says_a_single_occurrence_is_not_a_recurrence(qapp, tmp_path) -> None:
    from PySide6.QtWidgets import QLabel, QTableWidget

    page = ResearchKnowledgePage(_context_with_knowledge(tmp_path))
    labels = " ".join(label.text() for label in page.findChildren(QLabel))
    assert "Recorded patterns, grouped by observation type" in labels

    tables = page.findChildren(QTableWidget)
    assert len(tables) == 1
    pattern_table = tables[0]
    headers = [
        pattern_table.horizontalHeaderItem(c).text()
        for c in range(pattern_table.columnCount())
    ]
    assert headers == [
        "Pattern",
        "Occurrences",
        "Distinct runs",
        "Methods",
        "Recurrence established",
        "Shared tags",
    ]
    recurrence_column = headers.index("Recurrence established")
    verdicts = {
        pattern_table.item(r, recurrence_column).text()
        for r in range(pattern_table.rowCount())
    }
    assert pattern_table.rowCount() == 3
    assert verdicts == {"No -- a single occurrence is not a recurrence"}


# -- Filters ----------------------------------------------------------------------------------------


def test_a_method_filter_really_narrows_the_rendered_rows(qapp, tmp_path) -> None:
    page = ResearchKnowledgePage(_context_with_knowledge(tmp_path))
    assert _section(page, "Observations").text(0) == "Observations (5)"

    box = page._filters["method_id"]  # noqa: SLF001
    index = box.findData(SATRN_METHOD_ID)
    assert index >= 0, "the method combobox is populated from what is actually registered"
    box.setCurrentIndex(index)

    assert _section(page, "Observations").text(0) == "Observations (3)"
    assert "A filter is active" in page._notes.text()  # noqa: SLF001

    page._on_clear_filters()  # noqa: SLF001
    assert _section(page, "Observations").text(0) == "Observations (5)"
    assert "A filter is active" not in page._notes.text()  # noqa: SLF001


def test_the_tag_filter_narrows_by_a_real_recorded_tag(qapp, tmp_path) -> None:
    page = ResearchKnowledgePage(_context_with_knowledge(tmp_path))

    page._tag.setText("not_a_calibration_claim")  # noqa: SLF001
    assert _section(page, "Observations").text(0) == "Observations (1)"

    page._tag.setText("no_such_tag")  # noqa: SLF001
    assert _section(page, "Observations").text(0) == "Observations (0)"


def test_an_observation_type_filter_offers_every_one_of_the_fourteen_types(qapp, tmp_path) -> None:
    """Populated from the enum, so a new member appears without an edit here or in the page."""
    page = ResearchKnowledgePage(_context_with_knowledge(tmp_path))
    box = page._filters["observation_type"]  # noqa: SLF001

    # Fourteen types plus the "Any" entry.
    assert box.count() == len(ObservationType) + 1
    assert box.itemText(0) == "Any"
    for member in ObservationType:
        assert box.findData(member.value) >= 0


# -- Evidence panel ---------------------------------------------------------------------------------


def test_selecting_an_evidence_reference_shows_its_target_and_navigation_answer(
    qapp, tmp_path
) -> None:
    page = ResearchKnowledgePage(_context_with_knowledge(tmp_path))
    observations = _section(page, "Observations")
    first = observations.child(0)
    evidence_group = next(
        child for child in _descendants(first) if child.text(0).startswith("Evidence (")
    )
    reference = evidence_group.child(0)

    page._on_item_clicked(reference, 0)  # noqa: SLF001
    shown = page._detail.toPlainText()  # noqa: SLF001

    assert shown != EVIDENCE_PANEL_PLACEHOLDER
    assert "Reference id:" in shown
    assert "Resolvable from this surface:" in shown
    assert "Target entity:" in shown
    assert "Page that can show it:" in shown


def test_an_unopenable_reference_says_so_in_the_row_and_in_the_panel(qapp, tmp_path) -> None:
    """A telemetry-event reference is durable but is not indexed by this read model. The row is marked
    and the panel says which page could show it -- none."""
    page = ResearchKnowledgePage(_context_with_knowledge(tmp_path))
    observations = _section(page, "Observations")
    unopenable = next(
        child
        for child in _descendants(observations)
        if child.text(1) == "Telemetry event"
        and child.text(2) == "not openable from this surface"
    )

    page._on_item_clicked(unopenable, 0)  # noqa: SLF001
    shown = page._detail.toPlainText()  # noqa: SLF001

    assert "Resolvable from this surface: no" in shown
    assert "Page that can show it: none in this application" in shown


def test_clicking_a_non_evidence_row_leaves_the_panel_alone(qapp, tmp_path) -> None:
    page = ResearchKnowledgePage(_context_with_knowledge(tmp_path))
    heading = _section(page, "Observations")

    page._on_item_clicked(heading, 0)  # noqa: SLF001
    assert page._detail.toPlainText() == EVIDENCE_PANEL_PLACEHOLDER  # noqa: SLF001


def test_the_page_offers_no_way_to_change_the_knowledge_it_displays(qapp, tmp_path) -> None:
    """A reader, not an editor. Promoting a finding goes through `lifecycle.py`, and a button here
    would be a second, un-audited path past it."""
    from PySide6.QtWidgets import QPushButton

    page = ResearchKnowledgePage(_context_with_knowledge(tmp_path))
    buttons = {button.text() for button in page.findChildren(QPushButton)}

    assert buttons == {"Clear filters"}
