"""Release WS2: the v2 Evidence page — the canonical → observation → evidence trail, provider
invocation outcomes labeled as what they are, and the Administration Diagnostics tab.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from archivetrust.composition import AppContext  # noqa: E402
from archivetrust.domain.confidence.models import ComparisonClassification  # noqa: E402
from archivetrust.domain.shared.ids import new_id  # noqa: E402
from archivetrust.domain.telemetry.events import (  # noqa: E402
    ProviderFailureCategory,
    ProviderInvocationOutcome,
    ProviderObservationAttempted,
    stamp_recorded_at,
)
from tests.review._helpers import emit_slot, heading  # noqa: E402


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


def _context_with_document(tmp_path) -> AppContext:
    context = AppContext(deployment_root=tmp_path)
    emit_slot(
        context.telemetry,  # type: ignore[arg-type]
        document_ref="doc1",
        canonical_payload=heading("Municipal record"),
        provider_payloads=(("docling", heading("Municipal record")),),
        classification=ComparisonClassification.UNCORROBORATED_SINGLE_SOURCE,
    )
    context.telemetry.append(
        stamp_recorded_at(
            ProviderObservationAttempted(
                event_id=new_id("event"),
                document_ref="doc1",
                provider_id="tesseract_layoutparser",
                provider_version="5.x",
                invocation_id=new_id("invocation"),
                outcome=ProviderInvocationOutcome.FAILED,
                failure_category=ProviderFailureCategory.EXCEPTION,
                failure_reason="tesseract exited with code 1",
            )
        )
    )
    return context


def _tree_texts(tree) -> list[str]:
    texts = []

    def walk(item):
        texts.append(f"{item.text(0)} | {item.text(1)}")
        for i in range(item.childCount()):
            walk(item.child(i))

    for i in range(tree.topLevelItemCount()):
        walk(tree.topLevelItem(i))
    return texts


def test_evidence_page_traces_canonical_to_evidence_and_labels_failures(qapp, tmp_path):
    from archivetrust.clients.desktop_v2.pages import EvidencePage

    context = _context_with_document(tmp_path)
    page = EvidencePage(context)

    assert page._documents.count() == 1  # noqa: SLF001
    page.show_document("doc1")

    texts = _tree_texts(page._tree)  # noqa: SLF001
    joined = "\n".join(texts)
    # The chain: canonical result -> contributing observation -> evidence.
    assert "Municipal record" in joined
    assert "Observation by" in joined
    assert "Evidence " in joined
    # Honest labels: uncorroborated classification and a real provider failure.
    assert "Single source" in joined
    assert "FAILED (exception): tesseract exited with code 1" in joined


def test_evidence_page_appears_in_v2_shell_navigation(qapp, tmp_path):
    from archivetrust.clients.desktop_v2.app import MainWindowV2
    from archivetrust.presentation.desktop_v2_pages import DesktopV2Page

    context = _context_with_document(tmp_path)
    window = MainWindowV2(context)
    assert DesktopV2Page.EVIDENCE in window._shell.pages  # noqa: SLF001


def test_administration_diagnostics_tab_reports_pipeline_and_stream_integrity(qapp, tmp_path):
    from archivetrust.clients.desktop_v2.pages import AdministrationPage

    from PySide6.QtWidgets import QTabWidget

    context = _context_with_document(tmp_path)
    page = AdministrationPage(context)
    tab_widget = page.findChild(QTabWidget)
    labels = [tab_widget.tabText(i) for i in range(tab_widget.count())]
    assert "Diagnostics" in labels
