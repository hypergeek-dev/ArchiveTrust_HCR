"""The Research Knowledge page -- the eighth HTR research surface.

A separate module from `htr_pages.py`, which was already 1116 lines of seven pages. That file's own
docstring gives the precedent for splitting when a file stops being about one thing ("that file was
already 1697 lines of operational workflow before this stage ... Splitting keeps each file about one
thing"), and this page is one page with a filter bar, four sections and an evidence panel. Its
conventions are imported from `htr_pages.py` rather than restated, so the two files cannot drift on
what an empty state says.

Like every other research page it is a thin Qt binding: it constructs
`presentation/htr_knowledge_viewmodel.py::ResearchKnowledgeViewModel` from the `AppContext`, calls
`snapshot()`, and renders. It decides no status, groups no pattern, resolves no evidence reference and
formats no domain value -- all of that is the ViewModel's, which is what keeps this surface testable
without a display.

**What is deliberately scoped down, and why.**

*Evidence links show, they do not navigate.* Selecting an evidence reference fills the detail panel
with everything the ViewModel resolved for it -- kind, id, resolved label, detail, the target entity
kind and id, the page that could show it, and the full evidence-chain breadcrumb. It does **not**
switch the shell to that page and pre-select that entity. Cross-page navigation with target
pre-selection would need a shell-level "navigate to page X showing entity Y" channel that
`DesktopV2ShellViewModel` does not have (it carries a page, not a page-plus-subject), and inventing one
here would put navigation state in a page. `EvidenceLink.target_page` already carries the answer, so
the wiring is a small, separate change when a shell channel exists. What the panel shows is the whole
resolved reference, not a stub.

*No editing.* Nothing on this page raises a question, promotes a finding, or accepts an observation.
Those go through `htr/knowledge/lifecycle.py` and the durable store, and a page that could promote a
finding would be a second, un-audited path past the transition function. The page is a reader.

*Appearance is unverified.* There is no display in the environment this was built in. Every assertion
in `tests/clients/test_desktop_v2_knowledge_page.py` is about content and structure -- which rows
exist, which text is present, what selecting a row produces. Nothing here claims the layout looks
right.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from archivetrust.clients.desktop.ui import (
    TONE_ATTENTION,
    TONE_DANGER,
    TONE_NEUTRAL,
    card,
    empty_state,
    muted,
    page_header,
    scroll_page,
    section_title,
    stat_row,
    stat_tile,
    table,
)
from archivetrust.clients.desktop_v2.htr_pages import NOTHING_REGISTERED
from archivetrust.composition import AppContext
from archivetrust.presentation.display_names import (
    finding_status_label,
    knowledge_confidence_label,
    observation_type_label,
    short_ref,
)
from archivetrust.presentation.htr_knowledge_viewmodel import (
    EvidenceLink,
    KnowledgeFilter,
    ObservationRow,
    ResearchKnowledgeSnapshot,
)
from archivetrust.htr.knowledge.models import (
    FindingConfidence,
    FindingStatus,
    ObservationType,
)

_EVIDENCE_ROLE = Qt.ItemDataRole.UserRole
_ROW_KIND_ROLE = Qt.ItemDataRole.UserRole + 1

NOTHING_OBSERVED = (
    "No research observation or finding is recorded for this workspace. Observations are extracted "
    "from durable telemetry by an explicit step -- they are never derived automatically from a run -- "
    "so an empty page means nobody has run that extraction here, not that the runs went unrecorded."
)

ANY_CHOICE = "Any"
"""The filter comboboxes' first entry, carrying `None`. Spelled "Any" rather than left blank so an
unfiltered control is visibly unfiltered."""

EVIDENCE_PANEL_PLACEHOLDER = (
    "Select an evidence reference under an observation or finding to see what it points at."
)

_ATTENTION_STATUSES = frozenset({FindingStatus.DISPUTED.value, FindingStatus.UNDER_REVIEW.value})


def _status_tone(status: str) -> str:
    """Which tile tone a finding status gets. `Disputed`/`Under review` draw attention because both
    mean an unresolved human question; `Rejected` is the danger tone; everything else is neutral. A
    `Supported` count is deliberately *not* toned as good news -- it is a count, and a green tile next
    to a number a researcher must interpret would editorialise."""
    if status == FindingStatus.REJECTED.value:
        return TONE_DANGER
    if status in _ATTENTION_STATUSES:
        return TONE_ATTENTION
    return TONE_NEUTRAL


class ResearchKnowledgePage(QWidget):
    """Observations, findings, research questions and derived failure patterns, with filters and
    evidence references."""

    def __init__(self, context: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._context = context

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 24)
        root.setSpacing(12)
        root.addWidget(
            page_header(
                "Research Knowledge",
                "What has been observed, what is claimed, what is disputed, and what nobody has "
                "settled yet. Every row states the scope it holds within.",
            )
        )

        self._filters: dict[str, QComboBox] = {}
        root.addWidget(self._build_filter_bar())

        body = QWidget()
        split = QHBoxLayout(body)
        split.setContentsMargins(0, 0, 0, 0)

        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Record", "Status", "Scope"])
        self._tree.setColumnWidth(0, 420)
        self._tree.setColumnWidth(1, 200)
        self._tree.itemClicked.connect(self._on_item_clicked)
        split.addWidget(self._tree, stretch=3)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(section_title("Evidence reference"))
        self._detail = QTextEdit()
        self._detail.setReadOnly(True)
        self._detail.setPlainText(EVIDENCE_PANEL_PLACEHOLDER)
        right_layout.addWidget(self._detail, stretch=1)
        split.addWidget(right, stretch=2)
        root.addWidget(body, stretch=1)

        self._summary = QWidget()
        self._summary_layout = QVBoxLayout(self._summary)
        self._summary_layout.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self._summary)

        self._notes = muted("")
        self._notes.setWordWrap(True)
        root.addWidget(self._notes)

        self.refresh()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().showEvent(event)
        self.refresh()

    # -- Filter bar -------------------------------------------------------------------------------

    def _build_filter_bar(self) -> QWidget:
        """One combobox per filter dimension the ViewModel accepts, plus a free-text tag box.

        The enum-backed controls (observation type, finding status, confidence) are populated from the
        enums themselves rather than from a hand-written list, so a new member appears here without an
        edit. The id-backed controls (project, dataset version, experiment, method, model version,
        collection) are populated in `refresh` from what is actually registered -- offering a filter
        value that matches nothing would be a control that only ever produces an empty table.
        """
        bar = QWidget()
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(0, 0, 0, 0)

        for key, label, choices in (
            ("observation_type", "Observation type", _enum_choices(ObservationType, observation_type_label)),
            ("status", "Finding status", _enum_choices(FindingStatus, finding_status_label)),
            ("confidence", "Confidence", _enum_choices(FindingConfidence, knowledge_confidence_label)),
        ):
            layout.addWidget(QLabel(label))
            box = QComboBox()
            box.addItem(ANY_CHOICE, None)
            for value, text in choices:
                box.addItem(text, value)
            box.currentIndexChanged.connect(lambda _index: self.refresh())
            layout.addWidget(box)
            self._filters[key] = box

        for key, label in (
            ("project_id", "Project"),
            ("dataset_version_id", "Dataset version"),
            ("collection_id", "Collection"),
            ("experiment_id", "Experiment"),
            ("method_id", "Method"),
            ("model_version_id", "Model version"),
        ):
            layout.addWidget(QLabel(label))
            box = QComboBox()
            box.currentIndexChanged.connect(lambda _index: self.refresh())
            layout.addWidget(box)
            self._filters[key] = box

        layout.addWidget(QLabel("Tag"))
        self._tag = QLineEdit()
        self._tag.setPlaceholderText("e.g. n=1")
        self._tag.textChanged.connect(lambda _text: self.refresh())
        layout.addWidget(self._tag)

        clear = QPushButton("Clear filters")
        clear.clicked.connect(self._on_clear_filters)
        layout.addWidget(clear)
        layout.addStretch(1)
        return bar

    def _reload_id_choices(self) -> None:
        """Repopulates the id-backed comboboxes from the observations and findings actually present,
        preserving the current selection where it still exists."""
        vm = self._context.research_knowledge_viewmodel()
        observations = vm.observations()
        findings = vm.findings()
        available: dict[str, set[str]] = {
            "project_id": {row.project_id for row in observations if row.project_id}
            | {row.project_id for row in findings if row.project_id},
            "dataset_version_id": {
                row.dataset_version_id for row in observations if row.dataset_version_id
            }
            | {v for row in findings for v in row.affected_dataset_versions},
            "collection_id": {c for row in observations for c in row.collection_ids}
            | {c for row in findings for c in row.collection_ids},
            "experiment_id": {row.experiment_id for row in observations}
            | {row.experiment_id for row in findings},
            "method_id": {m for row in observations for m in row.method_ids}
            | {m for row in findings for m in row.affected_methods},
            "model_version_id": {m for row in observations for m in row.model_version_ids}
            | {m for row in findings for m in row.affected_model_versions},
        }
        for key, values in available.items():
            box = self._filters[key]
            current = box.currentData()
            box.blockSignals(True)
            box.clear()
            box.addItem(ANY_CHOICE, None)
            for value in sorted(values):
                box.addItem(short_ref(value, prefix=28), value)
            if current is not None:
                index = box.findData(current)
                if index >= 0:
                    box.setCurrentIndex(index)
            box.blockSignals(False)

    def _current_filter(self) -> KnowledgeFilter:
        return KnowledgeFilter(
            **{key: box.currentData() for key, box in self._filters.items()},
            tag=self._tag.text().strip() or None,
        )

    def _on_clear_filters(self) -> None:
        for box in self._filters.values():
            box.blockSignals(True)
            box.setCurrentIndex(0)
            box.blockSignals(False)
        self._tag.blockSignals(True)
        self._tag.clear()
        self._tag.blockSignals(False)
        self.refresh()

    # -- Rendering --------------------------------------------------------------------------------

    def refresh(self) -> None:
        self._reload_id_choices()
        snapshot = self._context.research_knowledge_viewmodel().snapshot(
            filter=self._current_filter()
        )
        self._render_tree(snapshot)
        self._render_summary(snapshot)
        self._render_notes(snapshot)

    def _render_tree(self, snapshot: ResearchKnowledgeSnapshot) -> None:
        self._tree.clear()
        sections = (
            ("Observations", snapshot.recent_observations, self._observation_item),
            (
                "What these comparisons cannot measure",
                snapshot.comparison_boundary_observations,
                self._observation_item,
            ),
            ("Candidate findings (not yet reviewed)", snapshot.candidate_findings, self._finding_item),
            ("Findings under review", snapshot.findings_under_review, self._finding_item),
            (
                "Provisionally supported findings",
                snapshot.provisionally_supported_findings,
                self._finding_item,
            ),
            ("Supported findings", snapshot.supported_findings, self._finding_item),
            ("Disputed findings", snapshot.disputed_findings, self._finding_item),
            ("Superseded findings", snapshot.superseded_findings, self._finding_item),
            ("Reproduced findings", snapshot.reproduced_findings, self._finding_item),
            ("Open research questions", snapshot.unresolved_research_questions, self._question_item),
            ("Answered research questions", snapshot.answered_research_questions, self._question_item),
        )
        notes = dict(snapshot.empty_field_notes)
        for title, rows, builder in sections:
            header = QTreeWidgetItem([f"{title} ({len(rows)})", "", ""])
            self._tree.addTopLevelItem(header)
            if not rows:
                QTreeWidgetItem(
                    header,
                    [
                        notes.get(_field_name_for(title), "Nothing in this category."),
                        "",
                        "",
                    ],
                )
                continue
            for row in rows:
                header.addChild(builder(row))
        self._tree.expandToDepth(0)

    def _observation_item(self, row: ObservationRow) -> QTreeWidgetItem:
        item = QTreeWidgetItem(
            [
                row.title,
                row.observation_type_label,
                f"{row.sample_size} {row.unit_of_analysis}"
                f"{'' if row.sample_size == 1 else 's'}",
            ]
        )
        item.setData(0, _ROW_KIND_ROLE, "observation")
        QTreeWidgetItem(item, ["Scope", row.scope_description, ""])
        QTreeWidgetItem(item, ["Description", row.description, ""])
        QTreeWidgetItem(
            item,
            [
                "Observer confidence",
                row.observation_confidence_label,
                row.review_status_label,
            ],
        )
        if row.tags:
            QTreeWidgetItem(item, ["Tags", ", ".join(row.tags), ""])
        if row.unverified_hypothesis is not None:
            # Rendered as its own labelled row, never merged into the description: everything in the
            # description is quoted from a durable record and this is not.
            QTreeWidgetItem(
                item, ["UNVERIFIED hypothesis (not a fact)", row.unverified_hypothesis, ""]
            )
        for question_id in row.raised_question_ids:
            QTreeWidgetItem(item, ["Raised research question", question_id, ""])
        self._attach_evidence(item, row.evidence)
        return item

    def _finding_item(self, row) -> QTreeWidgetItem:
        item = QTreeWidgetItem(
            [
                row.statement,
                row.review_status_label,
                f"{row.sample_size} {row.unit_of_analysis}"
                f"{'' if row.sample_size == 1 else 's'}",
            ]
        )
        item.setData(0, _ROW_KIND_ROLE, "finding")
        QTreeWidgetItem(item, ["Scope", row.scope_description, ""])
        QTreeWidgetItem(
            item,
            [
                "Author / reviewer",
                f"{row.author} / {row.reviewer or 'nobody has reviewed this'}",
                row.confidence_label,
            ],
        )
        if row.hypothesis_relationship_label is not None:
            QTreeWidgetItem(item, ["Hypothesis", row.hypothesis_relationship_label, ""])
        QTreeWidgetItem(
            item,
            [
                "Reproduced in another run",
                "Yes: " + ", ".join(row.reproducing_run_ids)
                if row.reproduced
                else "No -- this claim has not been shown to hold in any other experiment run",
                "",
            ],
        )
        limitations = QTreeWidgetItem(item, [f"Limitations ({len(row.limitations)})", "", ""])
        for limitation in row.limitations:
            QTreeWidgetItem(limitations, ["", limitation, ""])
        observations = QTreeWidgetItem(
            item, [f"Rests on observations ({len(row.supporting_observation_ids)})", "", ""]
        )
        for observation_id in row.supporting_observation_ids:
            QTreeWidgetItem(observations, ["", observation_id, ""])
        for metric_id in row.supporting_metric_ids:
            QTreeWidgetItem(item, ["Supporting metric", metric_id, ""])
        for revision in row.revisions:
            revision_item = QTreeWidgetItem(
                item,
                [
                    f"{revision.from_status_label} -> {revision.to_status_label}",
                    revision.actor,
                    revision.revised_at,
                ],
            )
            QTreeWidgetItem(revision_item, ["Reasoning", revision.reasoning, ""])
            self._attach_evidence(revision_item, revision.evidence)
        for contradiction in row.contradictions:
            contradiction_item = QTreeWidgetItem(
                item,
                [
                    "Contradicted by " + contradiction.source_kind,
                    contradiction.recorded_by,
                    contradiction.recorded_at,
                ],
            )
            QTreeWidgetItem(contradiction_item, ["", contradiction.description, ""])
            self._attach_evidence(contradiction_item, contradiction.evidence)
        if row.superseded_by is not None:
            QTreeWidgetItem(item, ["Superseded by", row.superseded_by, ""])
        for question_id in row.raised_question_ids:
            QTreeWidgetItem(item, ["Raised research question", question_id, ""])
        return item

    def _question_item(self, row) -> QTreeWidgetItem:
        item = QTreeWidgetItem([row.statement, row.status_label, ""])
        item.setData(0, _ROW_KIND_ROLE, "research_question")
        QTreeWidgetItem(item, ["Why it was asked", row.motivation, ""])
        if row.originating_summary is not None:
            QTreeWidgetItem(item, ["Raised by", row.originating_summary, ""])
        for hypothesis in row.hypotheses:
            hypothesis_item = QTreeWidgetItem(item, ["Hypothesis", hypothesis.statement, ""])
            QTreeWidgetItem(
                hypothesis_item, ["Refuted if", hypothesis.falsification_criterion, ""]
            )
        if row.created_experiment_id is not None:
            QTreeWidgetItem(
                item,
                [
                    "Drafted experiment",
                    row.created_experiment_name or row.created_experiment_id,
                    "Executed"
                    if row.drafted_experiment_has_runs
                    else "Drafted only -- never executed",
                ],
            )
            QTreeWidgetItem(
                item, ["Drafted experiment version", row.created_experiment_version_id or "", ""]
            )
        if row.answered_by_finding_id is not None:
            QTreeWidgetItem(item, ["Answered by finding", row.answered_by_finding_id, ""])
        self._attach_evidence(item, row.evidence)
        return item

    def _attach_evidence(
        self, parent: QTreeWidgetItem, evidence: tuple[EvidenceLink, ...]
    ) -> None:
        if not evidence:
            return
        resolved = sum(1 for link in evidence if link.resolved)
        group = QTreeWidgetItem(
            parent,
            [
                f"Evidence ({len(evidence)} references, {resolved} openable here)",
                "",
                "",
            ],
        )
        for link in evidence:
            child = QTreeWidgetItem(
                group,
                [
                    link.label,
                    link.kind_label,
                    "" if link.resolved else "not openable from this surface",
                ],
            )
            child.setData(0, _EVIDENCE_ROLE, link.model_dump())
            child.setData(0, _ROW_KIND_ROLE, "evidence")

    def _render_summary(self, snapshot: ResearchKnowledgeSnapshot) -> None:
        while self._summary_layout.count():
            widget = self._summary_layout.takeAt(0).widget()
            if widget is not None:
                widget.setParent(None)
        if snapshot.observation_count == 0 and snapshot.finding_count == 0:
            self._summary_layout.addWidget(
                empty_state(
                    "No research knowledge recorded",
                    NOTHING_OBSERVED if not snapshot.filtered else NOTHING_REGISTERED,
                )
            )
            return
        self._summary_layout.addWidget(
            stat_row(
                [
                    stat_tile(str(snapshot.observation_count), "Observations"),
                    stat_tile(str(snapshot.finding_count), "Findings"),
                    stat_tile(
                        str(len(snapshot.disputed_findings)),
                        "Disputed",
                        _status_tone(FindingStatus.DISPUTED.value),
                    ),
                    stat_tile(
                        str(len(snapshot.unresolved_research_questions)),
                        "Open questions",
                    ),
                    stat_tile(str(len(snapshot.reproduced_findings)), "Reproduced"),
                ]
            )
        )
        self._summary_layout.addWidget(
            card(
                "Recorded patterns, grouped by observation type",
                table(
                    [
                        "Pattern",
                        "Occurrences",
                        "Distinct runs",
                        "Methods",
                        "Recurrence established",
                        "Shared tags",
                    ],
                    [
                        [
                            row.observation_type_label,
                            str(row.occurrence_count),
                            str(row.distinct_experiment_run_count),
                            ", ".join(row.affected_method_labels) or "-",
                            "Yes"
                            if row.recurrence_established
                            else "No -- a single occurrence is not a recurrence",
                            ", ".join(row.shared_tags) or "-",
                        ]
                        for row in snapshot.recurring_failure_patterns
                    ],
                ),
            )
        )

    def _render_notes(self, snapshot: ResearchKnowledgeSnapshot) -> None:
        lines = []
        if snapshot.filtered:
            lines.append(
                "A filter is active: the counts and tables above describe the filtered subset, not "
                "everything recorded."
            )
        lines.extend(f"{name}: {reason}" for name, reason in snapshot.empty_field_notes)
        self._notes.setText("\n".join(lines))

    # -- Evidence panel ---------------------------------------------------------------------------

    def _on_item_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        payload = item.data(0, _EVIDENCE_ROLE)
        if payload is None:
            return
        link = EvidenceLink.model_validate(payload)
        lines = [
            f"{link.kind_label}",
            f"Reference id: {link.reference_id}",
            "",
            link.label,
        ]
        if link.detail:
            lines.extend(["", link.detail])
        if link.note:
            lines.extend(["", f"Note recorded with the reference: {link.note}"])
        if link.stream:
            lines.extend(["", f"Durable stream: {link.stream}"])
        lines.extend(
            [
                "",
                f"Resolvable from this surface: {'yes' if link.resolved else 'no'}",
                f"Target entity: {link.target_kind or 'none'} {link.target_id or ''}".rstrip(),
                f"Page that can show it: {link.target_page or 'none in this application'}",
            ]
        )
        if link.breadcrumb:
            lines.extend(["", "Evidence chain from the target upward:"])
            lines.extend(
                f"  {position}. {hop}" for position, hop in enumerate(link.breadcrumb, start=1)
            )
        self._detail.setPlainText("\n".join(lines))


def _enum_choices(enum_class, labeller) -> tuple[tuple[str, str], ...]:
    return tuple((member.value, labeller(member)) for member in enum_class)


_FIELD_NAME_BY_SECTION_TITLE = {
    "Observations": "recent_observations",
    "What these comparisons cannot measure": "comparison_boundary_observations",
    "Candidate findings (not yet reviewed)": "candidate_findings",
    "Findings under review": "findings_under_review",
    "Provisionally supported findings": "provisionally_supported_findings",
    "Supported findings": "supported_findings",
    "Disputed findings": "disputed_findings",
    "Superseded findings": "superseded_findings",
    "Reproduced findings": "reproduced_findings",
    "Open research questions": "unresolved_research_questions",
    "Answered research questions": "answered_research_questions",
}
"""Maps each rendered section heading back to the snapshot field it shows, so an empty section can
render that field's recorded reason from `EMPTY_FIELD_NOTES` instead of a generic blank. A table rather
than a derived string so a renamed heading is a visible edit here."""


def _field_name_for(section_title: str) -> str:
    return _FIELD_NAME_BY_SECTION_TITLE.get(section_title, "")


__all__ = [
    "EVIDENCE_PANEL_PLACEHOLDER",
    "NOTHING_OBSERVED",
    "ResearchKnowledgePage",
]
