"""Desktop v2 HTR research pages (docs/htr-migration-plan.md Stage 11).

A separate module from `pages.py` deliberately: that file was already 1697 lines of operational
workflow before this stage, and the research surfaces have a different backing layer (the
`presentation/htr_*_viewmodel.py` family over `htr/research_store.py`) than the operational pages'
telemetry read-models. Splitting keeps each file about one thing.

Every page here is a thin Qt binding: it constructs its ViewModel from the `AppContext`, calls it,
and renders. No page computes a metric, decides a status, or formats a domain value itself -- if a
rule about what to show lives anywhere, it lives in the ViewModel, which is what keeps the research
interface testable without a display.

**Honest empty states.** These surfaces are backed by an in-memory research store that is empty on a
fresh launch (see `htr/research_store.py`). Every page therefore renders an explicit "nothing
registered yet" message rather than an empty table that could be mistaken for a load failure.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
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
from archivetrust.composition import AppContext
from archivetrust.presentation.display_names import short_ref
from archivetrust.presentation.htr_experiment_builder_viewmodel import (
    EvaluationMode,
    FailurePolicy,
)
from archivetrust.presentation.htr_review_center_viewmodel import CORRECTION_TIME_GAP

_NODE_ROLE = Qt.ItemDataRole.UserRole
_KIND_ROLE = Qt.ItemDataRole.UserRole + 1

NOTHING_REGISTERED = (
    "No HTR research entities are registered in this session yet. Research projects, datasets, "
    "experiments and method runs are held in memory for now, so this is empty on every fresh "
    "launch rather than showing placeholder data."
)


def _yes_no(value: bool | None, *, unknown: str = "Not probed") -> str:
    """Three-state rendering. `None` never renders as "No" -- an unprobed capability and an absent
    one are different facts everywhere else in this codebase, and must stay different here."""
    if value is None:
        return unknown
    return "Yes" if value else "No"


def _number(value: int | None, *, unknown: str = "Unknown") -> str:
    return unknown if value is None else str(value)


def _rate(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


class _RefreshablePage(QWidget):
    """Shared plumbing: rebuild the body on show and on an explicit refresh, so a page that was
    opened before its data existed is never stale."""

    def __init__(self, context: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._context = context
        self._root = QVBoxLayout(self)
        self._root.setContentsMargins(0, 0, 0, 0)
        self._body: QWidget | None = None
        # Build once at construction: the shell builds every page up front and shows only the
        # current one, so a page whose body existed only after its first `showEvent` would be
        # empty for any caller that inspects it before it is navigated to.
        self.refresh()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().showEvent(event)
        self.refresh()

    def refresh(self) -> None:
        if self._body is not None:
            self._root.removeWidget(self._body)
            self._body.setParent(None)
        self._body = self.build_body()
        self._root.addWidget(self._body)

    def build_body(self) -> QWidget:  # pragma: no cover - subclasses always override
        raise NotImplementedError


class ResearchOverviewPage(_RefreshablePage):
    """The landing dashboard: projects, datasets, experiments, corpus size, review state, method
    failures, latest completed runs, and active Transkribus imports."""

    def build_body(self) -> QWidget:
        vm = self._context.research_dashboard_viewmodel()
        snapshot = vm.snapshot(computed_at="")

        sections: list[QWidget] = [
            page_header(
                "Research Overview",
                "The current state of this Historical HTR research centre: what is registered, "
                "what has run, and what still needs a human decision.",
            ),
            stat_row(
                [
                    stat_tile(str(len(snapshot.projects)), "Research projects"),
                    stat_tile(str(snapshot.dataset_count), "Datasets"),
                    stat_tile(str(snapshot.dataset_version_count), "Dataset versions"),
                    stat_tile(str(len(snapshot.experiments)), "Experiments"),
                ]
            ),
            stat_row(
                [
                    stat_tile(str(snapshot.documents_in_corpus), "Documents"),
                    stat_tile(str(snapshot.pages_in_corpus), "Pages"),
                    stat_tile(str(snapshot.text_lines_in_corpus), "Text lines"),
                    stat_tile(str(snapshot.lines_with_ground_truth), "Lines with ground truth"),
                ]
            ),
            stat_row(
                [
                    stat_tile(
                        _number(snapshot.reviewed_item_count), "Reviewed items"
                    ),
                    stat_tile(
                        _number(snapshot.unresolved_disagreement_count),
                        "Unresolved disagreements",
                        TONE_ATTENTION
                        if snapshot.unresolved_disagreement_count
                        else TONE_NEUTRAL,
                    ),
                    stat_tile(_number(snapshot.pending_review_count), "Pending review"),
                    stat_tile(
                        _number(snapshot.excluded_from_benchmark_count),
                        "Excluded from benchmark",
                    ),
                ]
            ),
        ]

        if snapshot.review_status_note:
            sections.append(muted(snapshot.review_status_note))

        if not snapshot.projects:
            sections.append(empty_state("Nothing registered yet", NOTHING_REGISTERED))

        sections.append(
            card(
                "Experiments",
                table(
                    ["Experiment", "Versions", "Runs", "Complete", "Status"],
                    [
                        [
                            row.name,
                            str(row.version_count),
                            str(row.run_count),
                            str(row.completed_run_count),
                            row.status,
                        ]
                        for row in snapshot.experiments
                    ],
                ),
            )
        )
        sections.append(
            card(
                "Latest completed runs",
                table(
                    ["Experiment", "Completed"],
                    [
                        [row.name, row.latest_run_completed_at or "-"]
                        for row in snapshot.latest_completed_runs
                    ],
                ),
            )
        )
        sections.append(
            card(
                "Method runs and failures",
                table(
                    ["Method", "Total", "Succeeded", "Failed", "Failure categories"],
                    [
                        [
                            row.method_display_name,
                            str(row.total_runs),
                            str(row.succeeded_runs),
                            str(row.failed_runs),
                            ", ".join(
                                f"{name} x{count}"
                                for name, count in sorted(row.failure_categories.items())
                            )
                            or "None",
                        ]
                        for row in snapshot.method_failures
                    ],
                ),
            )
        )
        sections.append(
            card(
                "External imports (Transkribus)",
                table(
                    ["Format", "Imported by", "Imported at", "Transkribus doc", "Vendor accuracy"],
                    [
                        [
                            row.export_format,
                            row.imported_by,
                            row.imported_at,
                            row.transkribus_document_id or "Not stated in the export",
                            (
                                f"{row.vendor_reported_accuracy:.2f} (vendor-reported; never "
                                "compared against our own CER/WER)"
                                if row.vendor_reported_accuracy is not None
                                else "Not stated in the export"
                            ),
                        ]
                        for row in snapshot.external_imports
                    ],
                ),
            )
        )

        if snapshot.legacy_method_runs:
            sections.append(
                muted(
                    f"{snapshot.legacy_method_runs} retired-OCR-method run(s) are recorded and "
                    "readable, and are held out of every figure above."
                )
            )

        return scroll_page(sections)


class MethodsPage(_RefreshablePage):
    """The three real HTR methods, their declared capabilities, and their known limitations."""

    def __init__(self, context: AppContext, parent: QWidget | None = None) -> None:
        self._probe = False
        super().__init__(context, parent)

    def build_body(self) -> QWidget:
        vm = self._context.method_overview_viewmodel()
        rows = vm.method_rows(probe_environment=self._probe)

        probe_button = QPushButton(
            "Re-check environments" if self._probe else "Check method environments"
        )
        probe_button.clicked.connect(self._on_probe)

        sections: list[QWidget] = [
            page_header(
                "HTR Methods",
                "Each method reports its own identity, capabilities and limitations. Nothing on "
                "this page is a hand-written summary.",
            ),
            probe_button,
        ]

        if not self._probe:
            sections.append(
                muted(
                    "Environments have not been checked this session. 'Not probed' below means "
                    "exactly that, not that a method is unavailable."
                )
            )

        errors = self._context.htr_method_adapter_errors()
        if errors:
            sections.append(
                card("Methods that could not be constructed", muted("\n".join(errors)))
            )

        sections.append(
            card(
                "Registered methods",
                table(
                    [
                        "Method",
                        "Vendor",
                        "Model revision",
                        "Adapter",
                        "Execution",
                        "Environment",
                        "Healthy",
                        "Device",
                        "Runs",
                        "Failed",
                    ],
                    [
                        [
                            row.display_name,
                            row.vendor,
                            row.model_revision,
                            row.adapter_version or "Not declared",
                            row.execution_location,
                            _yes_no(row.environment_valid),
                            _yes_no(row.healthy),
                            row.device or "Never run here",
                            str(row.total_runs),
                            str(row.failed_runs),
                        ]
                        for row in rows
                    ],
                ),
            )
        )

        for row in rows:
            detail = QWidget()
            layout = QVBoxLayout(detail)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(
                table(
                    ["Capability", "Supported"],
                    [[cap.label, _yes_no(cap.supported)] for cap in row.capabilities],
                )
            )
            layout.addWidget(section_title("Known limitations"))
            if row.known_limitations:
                for limitation in row.known_limitations:
                    layout.addWidget(muted(f"- {limitation}"))
            else:
                layout.addWidget(muted("This method's adapter documents no known limitations."))
            if row.last_failure_reason:
                layout.addWidget(
                    muted(f"Last failure ({row.last_failure_at}): {row.last_failure_reason}")
                )
            if row.environment_messages:
                layout.addWidget(section_title("Environment"))
                for message in row.environment_messages:
                    layout.addWidget(muted(message))
            sections.append(card(row.display_name, detail))

        return scroll_page(sections)

    def _on_probe(self) -> None:
        self._probe = True
        self.refresh()


class DatasetsPage(QWidget):
    """Read-only corpus tree: project -> dataset -> version -> collection -> document -> page ->
    region -> line -> crop, with a detail panel for the selected line."""

    def __init__(self, context: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._context = context

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 24)
        root.setSpacing(12)
        root.addWidget(
            page_header(
                "Datasets",
                "Navigate the registered corpus. This view is read-only: it never edits the "
                "corpus it displays.",
            )
        )

        body = QWidget()
        split = QHBoxLayout(body)
        split.setContentsMargins(0, 0, 0, 0)

        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Corpus", "Detail"])
        self._tree.setColumnWidth(0, 340)
        self._tree.itemExpanded.connect(self._on_expanded)
        self._tree.itemClicked.connect(self._on_clicked)
        split.addWidget(self._tree, stretch=3)

        self._detail = muted("Select a text line to see its ground truth and method outputs.")
        self._detail.setWordWrap(True)
        self._detail.setAlignment(Qt.AlignmentFlag.AlignTop)
        split.addWidget(self._detail, stretch=2)
        root.addWidget(body, stretch=1)

        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh)
        root.addWidget(refresh)

        self.refresh()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().showEvent(event)
        self.refresh()

    def refresh(self) -> None:
        vm = self._context.dataset_explorer_viewmodel()
        self._tree.clear()
        projects = vm.projects()
        if not projects:
            placeholder = QTreeWidgetItem(["No research project registered", NOTHING_REGISTERED])
            self._tree.addTopLevelItem(placeholder)
            return
        for node in projects:
            self._tree.addTopLevelItem(self._item(node))

    @staticmethod
    def _item(node) -> QTreeWidgetItem:
        item = QTreeWidgetItem([node.label, node.detail])
        item.setData(0, _NODE_ROLE, node.node_id)
        item.setData(0, _KIND_ROLE, node.kind)
        if node.child_count is None or node.child_count > 0:
            # A placeholder child makes the node expandable; it is replaced on first expand.
            item.addChild(QTreeWidgetItem(["Loading...", ""]))
        return item

    def _on_expanded(self, item: QTreeWidgetItem) -> None:
        if item.childCount() != 1 or item.child(0).text(0) != "Loading...":
            return
        item.takeChildren()
        for node in self._children_of(item.data(0, _KIND_ROLE), item.data(0, _NODE_ROLE)):
            item.addChild(self._item(node))

    def _children_of(self, kind: str, node_id: str) -> tuple:
        vm = self._context.dataset_explorer_viewmodel()
        return {
            "project": vm.datasets,
            "dataset": vm.dataset_versions,
            "dataset_version": vm.collections,
            "collection": vm.documents,
            "document": vm.pages,
            "page": vm.regions,
            "region": vm.text_lines,
            "text_line": vm.input_crops,
        }.get(kind, lambda _id: ())(node_id)

    def _on_clicked(self, item: QTreeWidgetItem) -> None:
        if item.data(0, _KIND_ROLE) != "text_line":
            return
        detail = self._context.dataset_explorer_viewmodel().line_detail(item.data(0, _NODE_ROLE))
        if detail is None:
            self._detail.setText("This line is no longer registered.")
            return
        shared = {
            True: "Yes - every crop for this line is byte-identical",
            False: "No - this line has more than one distinct crop, so method runs on it are "
            "not a controlled comparison",
            None: "No crop registered yet (this line has not been segmented)",
        }[detail.shared_crop_verified]
        self._detail.setText(
            "\n".join(
                [
                    f"Line {detail.reading_order_index}",
                    f"Ground truth: {detail.ground_truth or 'None recorded'}",
                    f"Methods that produced output: "
                    f"{', '.join(detail.method_outputs) or 'None'}",
                    f"Review status: {detail.review_status}",
                    f"Shared input crop: {shared}",
                    f"Crop hashes: {', '.join(short_ref(h, prefix=18) for h in detail.crop_hashes) or 'None'}",
                ]
            )
        )


class ExperimentsPage(QWidget):
    """Experiment builder: select project, dataset version, methods, segmentation, evaluation mode,
    metrics, convention, hardware, failure policy and report template, then build the experiment."""

    def __init__(self, context: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._context = context
        self._vm = context.experiment_builder_viewmodel()

        self._name = QLineEdit()
        self._name.setPlaceholderText("e.g. SATRN vs Florence-2 on 1670s court records")
        self._name.textChanged.connect(lambda text: self._vm.set_field("name", text))

        self._project = QComboBox()
        self._project.currentIndexChanged.connect(self._on_project_changed)
        self._dataset_version = QComboBox()
        self._dataset_version.currentIndexChanged.connect(self._on_dataset_version_changed)

        self._method_boxes: list[QCheckBox] = []
        methods = QWidget()
        methods_layout = QVBoxLayout(methods)
        methods_layout.setContentsMargins(0, 0, 0, 0)
        for method_id, label in self._vm.method_choices():
            box = QCheckBox(label)
            box.toggled.connect(
                lambda _checked=False, chosen=method_id: self._vm.toggle_method(chosen)
                or self._sync()
            )
            methods_layout.addWidget(box)
            self._method_boxes.append(box)

        self._segmentation = QLineEdit()
        self._segmentation.setPlaceholderText("Segmentation configuration reference")
        self._segmentation.textChanged.connect(
            lambda text: self._set("segmentation_configuration_ref", text or None)
        )

        self._mode = QComboBox()
        for mode, label in self._vm.evaluation_mode_choices():
            self._mode.addItem(label, mode)
        self._mode.currentIndexChanged.connect(
            lambda _index: self._set(
                "evaluation_mode", EvaluationMode(self._mode.currentData())
            )
        )

        self._metrics = QListWidget()
        self._metrics.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        for metric_id in self._vm.metric_choices():
            self._metrics.addItem(QListWidgetItem(metric_id))
        self._metrics.itemSelectionChanged.connect(self._on_metrics_changed)

        self._convention_id = QLineEdit()
        self._convention_id.setPlaceholderText("Transcription convention id")
        self._convention_id.textChanged.connect(
            lambda text: self._set("transcription_convention_id", text or None)
        )
        self._convention_version = QSpinBox()
        self._convention_version.setRange(0, 999)
        self._convention_version.valueChanged.connect(
            lambda value: self._set(
                "transcription_convention_version", value if value > 0 else None
            )
        )

        self._hardware = QLineEdit()
        self._hardware.setPlaceholderText("Hardware profile, e.g. rtx_3070_cuda")
        self._hardware.textChanged.connect(
            lambda text: self._set("hardware_profile", text or None)
        )

        self._failure_policy = QComboBox()
        for policy, label in self._vm.failure_policy_choices():
            self._failure_policy.addItem(label, policy)
        self._failure_policy.currentIndexChanged.connect(
            lambda _index: self._set(
                "failure_policy", FailurePolicy(self._failure_policy.currentData())
            )
        )

        self._report_template = QLineEdit()
        self._report_template.setPlaceholderText("Report template, e.g. swedish_historical_htr_baseline")
        self._report_template.textChanged.connect(
            lambda text: self._set("report_template", text or None)
        )

        self._sample_size = QSpinBox()
        self._sample_size.setRange(0, 100000)
        self._sample_size.valueChanged.connect(
            lambda value: self._set("sample_size", value if value > 0 else None)
        )
        self._sample_seed = QSpinBox()
        self._sample_seed.setRange(0, 100000)
        self._sample_seed.valueChanged.connect(
            lambda value: self._set("sample_seed", value if value > 0 else None)
        )

        self._issues = muted("")
        self._status = muted("")
        self._create = QPushButton("Create experiment")
        self._create.clicked.connect(self._on_create)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(
            scroll_page(
                [
                    page_header(
                        "Experiments",
                        "Compose a comparison. Every selection is validated as you make it; the "
                        "reasons you cannot create it yet are listed below.",
                    ),
                    card("Name", self._name),
                    card("Research project", self._project),
                    card("Dataset version", self._dataset_version),
                    card("Methods", methods),
                    card("Segmentation configuration", self._segmentation),
                    card("Evaluation mode", self._mode),
                    card("Metrics", self._metrics),
                    card("Transcription convention id", self._convention_id),
                    card("Transcription convention version (0 = none)", self._convention_version),
                    card("Hardware profile", self._hardware),
                    card("Failure policy", self._failure_policy),
                    card("Report template", self._report_template),
                    card("Sample size (0 = whole dataset version)", self._sample_size),
                    card("Sample seed (0 = none)", self._sample_seed),
                    self._issues,
                    self._create,
                    self._status,
                ]
            )
        )

        self._vm.validation_issues.subscribe(self._on_issues, immediate=True)
        self._vm.can_build.subscribe(self._create.setEnabled, immediate=True)
        self.refresh()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().showEvent(event)
        self.refresh()

    def refresh(self) -> None:
        """Reloads the project/dataset-version choices from the store without discarding what the
        researcher has already typed."""
        store = self._context.htr_research_store
        self._project.blockSignals(True)
        self._project.clear()
        for project in store.projects():
            self._project.addItem(project.name, project.project_id)
        self._project.blockSignals(False)
        self._on_project_changed(self._project.currentIndex())

    def _set(self, field: str, value: object) -> None:
        self._vm.set_field(field, value)
        self._sync()

    def _sync(self) -> None:
        self._vm.revalidate()

    def _on_project_changed(self, _index: int) -> None:
        project_id = self._project.currentData()
        self._vm.set_field("research_project_id", project_id)
        store = self._context.htr_research_store
        self._dataset_version.blockSignals(True)
        self._dataset_version.clear()
        if project_id is not None:
            for dataset in store.datasets(project_id=project_id):
                for version in store.dataset_versions(dataset_id=dataset.dataset_id):
                    self._dataset_version.addItem(
                        f"{dataset.name} v{version.version}", version.dataset_version_id
                    )
        self._dataset_version.blockSignals(False)
        self._on_dataset_version_changed(self._dataset_version.currentIndex())

    def _on_dataset_version_changed(self, _index: int) -> None:
        self._vm.set_field("dataset_version_id", self._dataset_version.currentData())

    def _on_metrics_changed(self) -> None:
        selected = {item.text() for item in self._metrics.selectedItems()}
        for metric_id in self._vm.metric_choices():
            chosen = metric_id in selected
            if chosen != (metric_id in self._vm.metric_definition_ids):
                self._vm.toggle_metric(metric_id)

    def _on_issues(self, issues) -> None:
        self._issues.setText(
            "\n".join(f"- {issue.message}" for issue in issues)
            or "Every required selection is made."
        )

    def _on_create(self) -> None:
        try:
            built = self._vm.build(created_at="")
        except ValueError as exc:
            self._status.setText(str(exc))
            return
        store = self._context.htr_research_store
        store.register_experiment(built.experiment)
        store.register_experiment_version(built.experiment_version)
        self._status.setText(
            f"Created experiment {built.experiment.name} "
            f"(version {built.experiment_version.version})."
        )


class ComparisonPage(QWidget):
    """Side-by-side per line, with all five result stages kept distinct."""

    def __init__(self, context: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._context = context

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 24)
        root.setSpacing(12)
        root.addWidget(
            page_header(
                "Comparison",
                "One line at a time, every method beside every other, with raw, parsed, "
                "normalized, reviewed and canonical shown as separate stages.",
            )
        )

        selector = QWidget()
        selector_layout = QHBoxLayout(selector)
        selector_layout.setContentsMargins(0, 0, 0, 0)
        selector_layout.addWidget(QLabel("Page"))
        self._page_box = QComboBox()
        self._page_box.currentIndexChanged.connect(self._on_page_changed)
        selector_layout.addWidget(self._page_box, stretch=1)
        root.addWidget(selector)

        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Line / method / stage", "Text", "Metrics"])
        self._tree.setColumnWidth(0, 320)
        self._tree.setColumnWidth(1, 380)
        root.addWidget(self._tree, stretch=1)

        self._note = muted("")
        root.addWidget(self._note)
        self.refresh()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().showEvent(event)
        self.refresh()

    def refresh(self) -> None:
        store = self._context.htr_research_store
        current = self._page_box.currentData()
        self._page_box.blockSignals(True)
        self._page_box.clear()
        for page in store.pages():
            self._page_box.addItem(
                f"{short_ref(page.archive_object_ref, prefix=16)} page {page.page_number}",
                page.page_id,
            )
        if current is not None:
            index = self._page_box.findData(current)
            if index >= 0:
                self._page_box.setCurrentIndex(index)
        self._page_box.blockSignals(False)
        self._on_page_changed(self._page_box.currentIndex())

    def _on_page_changed(self, _index: int) -> None:
        self._tree.clear()
        page_id = self._page_box.currentData()
        if page_id is None:
            self._note.setText(NOTHING_REGISTERED)
            return
        rows = self._context.htr_comparison_viewmodel().page_comparison(page_id)
        if not rows:
            self._note.setText("This page has no segmented lines registered yet.")
            return
        self._note.setText("")
        for row in rows:
            self._tree.addTopLevelItem(self._line_item(row))

    def _line_item(self, row) -> QTreeWidgetItem:
        controlled = {
            True: "Controlled comparison (one shared crop)",
            False: "NOT a controlled comparison - the methods read different crops",
            None: "No input crop registered",
        }[row.controlled_comparison]
        item = QTreeWidgetItem(
            [
                f"Line {row.reading_order_index}",
                f"Ground truth: {row.ground_truth or 'None recorded'}",
                controlled,
            ]
        )
        if row.canonical_text is not None:
            QTreeWidgetItem(
                item,
                [
                    "Canonical selection",
                    row.canonical_text,
                    row.canonical_strategy_label or "",
                ],
            )
        if row.reviewer_decision:
            QTreeWidgetItem(item, ["Reviewer decision", row.reviewer_decision, ""])

        for cell in row.cells:
            metrics = (
                f"CER {_rate(cell.character_error_rate_normalized)} / "
                f"WER {_rate(cell.word_error_rate_normalized)}"
                if cell.character_error_rate_normalized is not None
                else "No ground truth to measure against"
            )
            if cell.confidence is not None:
                metrics += f" | confidence {cell.confidence:.4f}"
            else:
                metrics += " | confidence not reported"
            if cell.processing_time_ms is not None:
                metrics += f" | {cell.processing_time_ms:.0f} ms"
            headline = cell.method_display_name
            if cell.is_legacy_method:
                headline += " [legacy]"
            method_item = QTreeWidgetItem([headline, cell.outcome_label, metrics])

            for reason in cell.failure_reasons:
                QTreeWidgetItem(method_item, ["Failure", reason, ""])
            for stage in cell.stages:
                QTreeWidgetItem(
                    method_item,
                    [
                        stage.label,
                        stage.text if stage.present else "Not produced at this stage",
                        stage.attribution or "",
                    ],
                )
            if cell.edit_ops is not None:
                QTreeWidgetItem(
                    method_item,
                    [
                        "Edit operations",
                        f"chars: {cell.edit_ops.char_substitutions} sub / "
                        f"{cell.edit_ops.char_insertions} ins / {cell.edit_ops.char_deletions} del",
                        f"words: {cell.edit_ops.word_substitutions} sub / "
                        f"{cell.edit_ops.word_insertions} ins / {cell.edit_ops.word_deletions} del",
                    ],
                )
            item.addChild(method_item)
        return item


class EvidenceChainPage(QWidget):
    """The `docs/htr-domain-design.md` §4 chain for one method run, rendered as breadcrumbs."""

    def __init__(self, context: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._context = context

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 24)
        root.setSpacing(12)
        root.addWidget(
            page_header(
                "Evidence Chain",
                "Every result traced back through its stages, method run, model version, input "
                "crop, segmentation, page, document, collection, dataset and project.",
            )
        )

        selector = QWidget()
        selector_layout = QHBoxLayout(selector)
        selector_layout.setContentsMargins(0, 0, 0, 0)
        selector_layout.addWidget(QLabel("Method run"))
        self._run_box = QComboBox()
        self._run_box.currentIndexChanged.connect(self._on_run_changed)
        selector_layout.addWidget(self._run_box, stretch=1)
        root.addWidget(selector)

        self._status = muted("")
        root.addWidget(self._status)

        self._chain = QListWidget()
        root.addWidget(self._chain, stretch=1)
        self.refresh()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().showEvent(event)
        self.refresh()

    def refresh(self) -> None:
        from archivetrust.presentation.display_names import method_label

        current = self._run_box.currentData()
        self._run_box.blockSignals(True)
        self._run_box.clear()
        for run in self._context.htr_research_store.method_runs():
            self._run_box.addItem(
                f"{method_label(run.method_id)} - {short_ref(run.method_run_id, prefix=16)}",
                run.method_run_id,
            )
        if current is not None:
            index = self._run_box.findData(current)
            if index >= 0:
                self._run_box.setCurrentIndex(index)
        self._run_box.blockSignals(False)
        self._on_run_changed(self._run_box.currentIndex())

    def _on_run_changed(self, _index: int) -> None:
        self._chain.clear()
        run_id = self._run_box.currentData()
        if run_id is None:
            self._status.setText(NOTHING_REGISTERED)
            return
        chain = self._context.htr_evidence_chain_viewmodel().chain_for_method_run(run_id)
        self._status.setText(
            "Complete chain to the research project."
            if chain.complete
            else (
                f"Chain stops at {short_ref(chain.broken_at, prefix=24)}: that reference is not "
                "registered."
                if chain.broken_at
                else "Chain is incomplete: this run has no shared input crop to trace upward from."
            )
        )
        for position, link in enumerate(chain.links, start=1):
            marker = "" if link.resolved else "  [unresolved]"
            self._chain.addItem(
                QListWidgetItem(
                    f"{position}. {link.kind}: {link.label}{marker}"
                    + (f"\n      {link.detail}" if link.detail else "")
                )
            )


class ReviewCenterPage(QWidget):
    """Blind dual review: a reviewer's own queue, disagreement display, adjudication, exclusion,
    and workload."""

    def __init__(self, context: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._context = context

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 24)
        root.setSpacing(12)
        root.addWidget(
            page_header(
                "Review Center",
                "Two independent readings of the same line, compared only after both are "
                "submitted, adjudicated when they diverge too far.",
            )
        )

        self._reviewer = QLineEdit(context.reviewer_ref)
        self._reviewer.textChanged.connect(lambda _text: self.refresh())
        root.addWidget(card("Reviewer", self._reviewer))

        body = QWidget()
        split = QHBoxLayout(body)
        split.setContentsMargins(0, 0, 0, 0)

        self._queue = QListWidget()
        self._queue.itemClicked.connect(self._on_selected)
        split.addWidget(self._queue, stretch=2)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self._detail = QTextEdit()
        self._detail.setReadOnly(True)
        right_layout.addWidget(self._detail, stretch=1)

        self._rationale = QLineEdit()
        self._rationale.setPlaceholderText("Adjudication rationale (required, cannot be blank)")
        right_layout.addWidget(self._rationale)
        self._resolved = QLineEdit()
        self._resolved.setPlaceholderText("Adjudicated reading")
        right_layout.addWidget(self._resolved)

        actions = QWidget()
        actions_layout = QHBoxLayout(actions)
        actions_layout.setContentsMargins(0, 0, 0, 0)
        self._adjudicate = QPushButton("Record adjudication")
        self._adjudicate.clicked.connect(self._on_adjudicate)
        actions_layout.addWidget(self._adjudicate)
        self._exclude = QPushButton("Exclude with reason")
        self._exclude.clicked.connect(self._on_exclude)
        actions_layout.addWidget(self._exclude)
        actions_layout.addStretch(1)
        right_layout.addWidget(actions)

        self._status = muted(CORRECTION_TIME_GAP)
        right_layout.addWidget(self._status)
        split.addWidget(right, stretch=3)
        root.addWidget(body, stretch=1)

        self._workload = muted("")
        root.addWidget(self._workload)
        self.refresh()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().showEvent(event)
        self.refresh()

    def refresh(self) -> None:
        vm = self._context.review_center_viewmodel()
        reviewer = self._reviewer.text().strip() or self._context.reviewer_ref
        self._queue.clear()
        for row in vm.reviewer_queue(reviewer):
            item = QListWidgetItem(
                f"{short_ref(row.target_ref, prefix=20)} - {row.role} - {row.status}"
            )
            item.setData(_NODE_ROLE, row.target_ref)
            self._queue.addItem(item)
        if self._queue.count() == 0:
            self._queue.addItem(
                QListWidgetItem("No review assignments for this reviewer in this session.")
            )
        self._workload.setText(
            " | ".join(
                f"{w.reviewer_ref}: {w.open_count} open, {w.submitted_count} submitted"
                for w in vm.all_workloads()
            )
            or "No reviewers have assignments in this session."
        )

    def _selected_target(self) -> str | None:
        item = self._queue.currentItem()
        return item.data(_NODE_ROLE) if item is not None else None

    def _on_selected(self, item: QListWidgetItem) -> None:
        target_ref = item.data(_NODE_ROLE)
        if target_ref is None:
            return
        state = self._context.review_center_viewmodel().target_state(target_ref, computed_at="")
        if state is None:
            self._detail.setPlainText(
                "This target is still pending: it does not yet have both blind submissions, so "
                "there is nothing to compare. The other reviewer's reading stays hidden until "
                "you have submitted your own."
            )
            return
        lines = [
            f"Status: {state.status_label}",
            f"Ready for benchmark: {'yes' if state.ready_for_benchmark else 'no'}",
            f"Resolved value: {state.benchmark_value or 'none yet'}",
        ]
        if state.character_error_rate_normalized is not None:
            lines.append(
                f"Normalized CER between the two readings: "
                f"{state.character_error_rate_normalized:.4f}"
            )
        if state.disagreements:
            lines.append("")
            lines.append("Disagreements:")
            for span in state.disagreements:
                lines.append(
                    f"  [{span.op}] A: {span.reviewer_a_text or '(nothing)'}"
                    f"  <->  B: {span.reviewer_b_text or '(nothing)'}"
                )
        if state.adjudicated:
            lines.append("")
            lines.append(f"Adjudicated by {state.adjudicator_ref}: {state.adjudication_rationale}")
        if state.excluded:
            lines.append("")
            lines.append(f"Excluded by {state.excluded_by}: {state.exclusion_reason}")
        self._detail.setPlainText("\n".join(lines))

    def _on_adjudicate(self) -> None:
        target_ref = self._selected_target()
        if target_ref is None:
            self._status.setText("Select a review target first.")
            return
        try:
            self._context.review_center_viewmodel().adjudicate_target(
                target_ref=target_ref,
                adjudicator_ref=self._reviewer.text().strip(),
                resolved_value=self._resolved.text() or None,
                rationale=self._rationale.text(),
                adjudicated_at="",
            )
        except Exception as exc:  # noqa: BLE001 - the refusal reason is exactly what to show
            self._status.setText(str(exc))
            return
        self._status.setText("Adjudication recorded.")
        self.refresh()

    def _on_exclude(self) -> None:
        target_ref = self._selected_target()
        if target_ref is None:
            self._status.setText("Select a review target first.")
            return
        try:
            self._context.review_center_viewmodel().exclude_target(
                target_ref=target_ref,
                reason=self._rationale.text(),
                excluded_by=self._reviewer.text().strip(),
                excluded_at="",
            )
        except Exception as exc:  # noqa: BLE001 - the refusal reason is exactly what to show
            self._status.setText(str(exc))
            return
        self._status.setText("Excluded from the benchmark, with the reason recorded.")
        self.refresh()


class ResearchReportExporter:
    """The JSON/CSV export action the research pages offer. A plain class, not a widget, so the
    export decision stays testable without a file dialog."""

    def __init__(self, context: AppContext) -> None:
        self._context = context

    @staticmethod
    def export_to_path(report, path: str, *, export_format: str):
        from pathlib import Path

        from archivetrust.research.reports.export import write_report

        return write_report(report, Path(path), export_format=export_format)

    def choose_path(self, parent: QWidget) -> str | None:  # pragma: no cover - needs a dialog
        path, _filter = QFileDialog.getSaveFileName(
            parent, "Export research report", "", "JSON (*.json);;CSV (*.csv)"
        )
        return path or None


__all__ = [
    "ComparisonPage",
    "DatasetsPage",
    "EvidenceChainPage",
    "ExperimentsPage",
    "MethodsPage",
    "NOTHING_REGISTERED",
    "ResearchOverviewPage",
    "ResearchReportExporter",
    "ReviewCenterPage",
]
