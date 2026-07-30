"""Desktop v2 navigation.

Page labels are researcher-work labels, not backend module names.

**Stage 11 additions.** The HTR research surfaces (`RESEARCH_OVERVIEW`, `METHODS`, `DATASETS`,
`EXPERIMENTS`, `COMPARISON`, `RESEARCH_EVIDENCE`, `REVIEW_CENTER`) are added ahead of the retained
operational pages, because a research deployment lands on the dashboard, not on a workspace picker.
The operational pages (workspaces, sources, processing, work queue, documents, evidence, evaluation
approval, outputs, administration) are retained unchanged: they administer the acquisition/
processing/release machinery `docs/htr-transformation-audit.md` classifies as "retain, extend", and
deleting them would remove working functionality this stage does not replace.
"""

from __future__ import annotations

from enum import Enum


class DesktopV2Page(str, Enum):
    # -- HTR research surfaces (Stage 11) ------------------------------------------------------
    RESEARCH_OVERVIEW = "research_overview"
    METHODS = "methods"
    DATASETS = "datasets"
    EXPERIMENTS = "experiments"
    COMPARISON = "comparison"
    RESEARCH_EVIDENCE = "research_evidence"
    REVIEW_CENTER = "review_center"
    RESEARCH_KNOWLEDGE = "research_knowledge"
    """Added with the research-knowledge frontend phase. Last of the research surfaces in rail order,
    because it is the only one whose content is *produced* by the others: an observation is extracted
    from what the experiment and comparison surfaces show, and a finding is what the review center's
    work makes defensible."""

    # -- Retained operational surfaces ---------------------------------------------------------
    WORKSPACES = "workspaces"
    SOURCES = "sources"
    PROCESSING = "processing"
    WORK_QUEUE = "work_queue"
    DOCUMENTS = "documents"
    EVIDENCE = "evidence"
    EVALUATION = "evaluation"
    OUTPUTS = "outputs"
    ADMINISTRATION = "administration"

    @property
    def title(self) -> str:
        return {
            DesktopV2Page.RESEARCH_OVERVIEW: "Research Overview",
            DesktopV2Page.METHODS: "HTR Methods",
            DesktopV2Page.DATASETS: "Datasets",
            DesktopV2Page.EXPERIMENTS: "Experiments",
            DesktopV2Page.COMPARISON: "Comparison",
            DesktopV2Page.RESEARCH_EVIDENCE: "Evidence Chain",
            DesktopV2Page.REVIEW_CENTER: "Review Center",
            DesktopV2Page.RESEARCH_KNOWLEDGE: "Research Knowledge",
            DesktopV2Page.WORKSPACES: "Workspaces",
            DesktopV2Page.SOURCES: "Sources",
            DesktopV2Page.PROCESSING: "Processing",
            DesktopV2Page.WORK_QUEUE: "Work Queue",
            DesktopV2Page.DOCUMENTS: "Documents",
            DesktopV2Page.EVIDENCE: "Document Evidence",
            DesktopV2Page.EVALUATION: "Evaluation Approval",
            DesktopV2Page.OUTPUTS: "Releases / Outputs",
            DesktopV2Page.ADMINISTRATION: "Administration",
        }[self]

    @property
    def is_research_surface(self) -> bool:
        """Whether this page belongs to the HTR research interface (Stage 11) as opposed to the
        retained operational machinery. Lets the shell group the navigation rail without the View
        hardcoding which page is which."""
        return self in _RESEARCH_PAGES


_RESEARCH_PAGES = frozenset(
    {
        DesktopV2Page.RESEARCH_OVERVIEW,
        DesktopV2Page.METHODS,
        DesktopV2Page.DATASETS,
        DesktopV2Page.EXPERIMENTS,
        DesktopV2Page.COMPARISON,
        DesktopV2Page.RESEARCH_EVIDENCE,
        DesktopV2Page.REVIEW_CENTER,
        DesktopV2Page.RESEARCH_KNOWLEDGE,
    }
)
