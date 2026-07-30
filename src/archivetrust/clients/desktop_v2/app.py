"""ArchiveTrust Desktop v2 shell -- the HTR Research Center desktop client.

Since `docs/htr-migration-plan.md` Stage 11 the rail leads with the research surfaces (overview,
methods, datasets, experiments, comparison, evidence chain, review center, from
`htr_pages.py`, plus research knowledge, from `htr_knowledge_page.py`) and follows with the retained
operational surfaces (acquisition, processing, release, administration, from `pages.py`). The split
lives on `DesktopV2Page.is_research_surface`, not in this file's layout code.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QStackedWidget,
    QWidget,
)

from archivetrust.composition import AppContext
from archivetrust.clients.desktop.ui import STYLESHEET
from archivetrust.clients.desktop_v2.htr_knowledge_page import ResearchKnowledgePage
from archivetrust.clients.desktop_v2.htr_pages import (
    ComparisonPage,
    DatasetsPage,
    EvidenceChainPage,
    ExperimentsPage,
    MethodsPage,
    ResearchOverviewPage,
    ReviewCenterPage,
)
from archivetrust.clients.desktop_v2.pages import (
    AdministrationPage,
    DocumentsPage,
    EvidencePage,
    EvaluationApprovalPage,
    OutputsPage,
    ProcessingPage,
    SourcesPage,
    WorkQueuePage,
    WorkspacesPage,
)
from archivetrust.presentation.desktop_v2_pages import DesktopV2Page
from archivetrust.presentation.desktop_v2_shell_viewmodel import DesktopV2ShellViewModel


class MainWindowV2(QMainWindow):
    """The single supported desktop window. A thin binding: it owns navigation wiring and nothing
    else -- which page exists, and what each shows, are the shell ViewModel's and each page's
    ViewModel's decisions respectively."""

    def __init__(self, context: AppContext) -> None:
        super().__init__()
        self._context = context
        self._shell = DesktopV2ShellViewModel()
        self.setWindowTitle("ArchiveTrust HTR Research Center")
        self.setStyleSheet(STYLESHEET)

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._rail = QListWidget()
        self._rail.setObjectName("Nav")
        self._rail.setFixedWidth(220)
        for page in self._shell.pages:
            item = QListWidgetItem(page.title)
            if not page.is_research_surface:
                # Retained operational surfaces are still reachable, just visibly a second group,
                # so the research work is what the rail leads with.
                item.setToolTip("Operational surface (acquisition, processing, release)")
            self._rail.addItem(item)
        self._rail.currentRowChanged.connect(self._on_rail_changed)
        layout.addWidget(self._rail)

        self._stack = QStackedWidget()
        for page in self._shell.pages:
            self._stack.addWidget(self._build_page(page))
        layout.addWidget(self._stack, stretch=1)

        self.setCentralWidget(central)
        self._shell.current_page.subscribe(self._on_page_changed, immediate=True)

    _PAGE_CLASSES = {
        DesktopV2Page.RESEARCH_OVERVIEW: ResearchOverviewPage,
        DesktopV2Page.METHODS: MethodsPage,
        DesktopV2Page.DATASETS: DatasetsPage,
        DesktopV2Page.EXPERIMENTS: ExperimentsPage,
        DesktopV2Page.COMPARISON: ComparisonPage,
        DesktopV2Page.RESEARCH_EVIDENCE: EvidenceChainPage,
        DesktopV2Page.REVIEW_CENTER: ReviewCenterPage,
        DesktopV2Page.RESEARCH_KNOWLEDGE: ResearchKnowledgePage,
        DesktopV2Page.WORKSPACES: WorkspacesPage,
        DesktopV2Page.SOURCES: SourcesPage,
        DesktopV2Page.PROCESSING: ProcessingPage,
        DesktopV2Page.WORK_QUEUE: WorkQueuePage,
        DesktopV2Page.DOCUMENTS: DocumentsPage,
        DesktopV2Page.EVIDENCE: EvidencePage,
        DesktopV2Page.EVALUATION: EvaluationApprovalPage,
        DesktopV2Page.OUTPUTS: OutputsPage,
        DesktopV2Page.ADMINISTRATION: AdministrationPage,
    }
    """One entry per `DesktopV2Page`. A table rather than an if-chain so a page added to the enum
    without a widget fails loudly here instead of silently falling through to Administration, which
    is what the previous `return AdministrationPage(...)` default did."""

    def _build_page(self, page: DesktopV2Page) -> QWidget:
        page_class = self._PAGE_CLASSES.get(page)
        if page_class is None:
            raise KeyError(f"No widget is registered for navigation page {page.value!r}")
        return page_class(self._context)

    def _on_rail_changed(self, row: int) -> None:
        if 0 <= row < len(self._shell.pages):
            self._shell.navigate_to(self._shell.pages[row])

    def _on_page_changed(self, page: DesktopV2Page) -> None:
        index = self._shell.pages.index(page)
        self._stack.setCurrentIndex(index)
        self._rail.setCurrentRow(index)
