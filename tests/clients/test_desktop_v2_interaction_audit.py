"""Human-paced Desktop v2 interaction audit.

This test intentionally drives the real Desktop v2 Qt shell and pages, using a disposable
workspace and exactly one generated PDF fixture. Provider execution is deterministic: a small
injected Docling client returns one reviewable observation with real PDF-point geometry.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("ARCHIVETRUST_DESKTOP_V2", "1")

pytest.importorskip("PySide6")

from PySide6.QtCore import QEventLoop, QPoint, QRectF, Qt  # noqa: E402
from PySide6.QtGui import QGuiApplication  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QGraphicsRectItem  # noqa: E402

from archivetrust.application.progress import ProcessingProgressKind  # noqa: E402
from archivetrust.composition import AppContext  # noqa: E402
from archivetrust.clients.desktop_v2.app import MainWindowV2  # noqa: E402
from archivetrust.clients.desktop_v2.pages import ProcessingPage, SourcesPage, WorkQueuePage  # noqa: E402
from archivetrust.domain.telemetry.events import HumanCorrectionApplied, ReviewOutcomeRecorded  # noqa: E402
from archivetrust.presentation.desktop_v2_pages import DesktopV2Page  # noqa: E402
from tests.infrastructure.rendering._minimal_pdf import write_minimal_pdf  # noqa: E402


ARTIFACT_DIR = Path("test-artifacts/desktop-v2-interaction")


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


class AuditRecorder:
    def __init__(self, window: MainWindowV2) -> None:
        self.window = window
        self.metadata: dict[str, object] = {
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "qt_platform": os.environ.get("QT_QPA_PLATFORM"),
                "desktop_v2_env": os.environ.get("ARCHIVETRUST_DESKTOP_V2"),
            },
            "screenshots": [],
            "checkpoints": [],
            "warnings": [],
            "exceptions": [],
        }

    def capture(self, name: str, **properties: object) -> None:
        QApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 50)
        path = ARTIFACT_DIR / name
        assert self.window.grab().save(str(path))
        screen = QGuiApplication.primaryScreen()
        dpi = screen.logicalDotsPerInch() if screen is not None else None
        size = self.window.size()
        self.metadata["screenshots"].append(
            {
                "file": str(path.as_posix()),
                "window_size": [size.width(), size.height()],
                "dpi": dpi,
                "properties": properties,
            }
        )

    def checkpoint(self, name: str, **properties: object) -> None:
        self.metadata["checkpoints"].append({"name": name, **properties})

    def write(self) -> None:
        (ARTIFACT_DIR / "metadata.json").write_text(
            json.dumps(self.metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )


def _wait_until(condition, *, timeout_ms: int = 5000, interval_ms: int = 25) -> None:
    deadline = time.monotonic() + timeout_ms / 1000
    while time.monotonic() < deadline:
        QApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, interval_ms)
        if condition():
            return
        QTest.qWait(interval_ms)
    raise AssertionError("Timed out waiting for expected UI/application state")


def _install_single_pdf_provider(context: AppContext) -> None:
    """Registers a minimal, real `ProviderAdapter` producing two Observations with real PDF-point
    geometry (docs/htr-migration-plan.md Stage 5 -- the real Docling adapter this helper used to
    install was deleted along with the rest of the OCR-era provider architecture; the geometry
    highlighting behavior this test audits is generic to any `ProviderAdapter`, not Docling-
    specific, so a minimal fake adapter preserves the same test intent)."""
    import json

    from archivetrust.domain.evidence.models import BoundingBox, Evidence, Precision, ProcessingStage
    from archivetrust.domain.ontology.base import Observation
    from archivetrust.domain.ontology.payloads import HeadingPayload, ParagraphPayload
    from archivetrust.providers.base import (
        DeploymentLocality,
        DeploymentProfile,
        DeterministicProviderAdapter,
        Introspectability,
        ProviderAttempt,
        ProviderRegistration,
        ProviderRunResult,
        Reproducibility,
    )

    class _FakePdfAdapter(DeterministicProviderAdapter):
        registration = ProviderRegistration(
            provider_id="fake-pdf",
            reproducibility=Reproducibility.DETERMINISTIC,
            deployment_locality=DeploymentLocality.IN_PROCESS,
            introspectability=Introspectability.OPEN,
        )

        def observe(self, *, document_ref: str, invocation_id: str, source: object) -> ProviderRunResult:
            def _make(payload, text: str, bbox: tuple[float, float, float, float]):
                evidence = Evidence.create(
                    provider=self.provider_id,
                    provider_version="1.0",
                    raw_output=json.dumps({"text": text}),
                    processing_stage=ProcessingStage.RAW,
                    page=1,
                    bounding_box=BoundingBox(
                        x0=bbox[0], y0=bbox[1], x1=bbox[2], y1=bbox[3], precision=Precision.PIXEL_ACCURATE
                    ),
                )
                observation = Observation.from_evidence(
                    provider_id=self.provider_id, provider_version="1.0", payload=payload, evidence=(evidence,)
                )
                return evidence, observation

            heading_evidence, heading_observation = _make(
                HeadingPayload(text="Municipal record", level=1), "Municipal record", (40.0, 240.0, 260.0, 280.0)
            )
            paragraph_evidence, paragraph_observation = _make(
                ParagraphPayload(text="Approved for review"), "Approved for review", (40.0, 180.0, 300.0, 220.0)
            )
            return ProviderRunResult(
                attempt=ProviderAttempt(
                    provider_id=self.provider_id, provider_version="1.0", invocation_id=invocation_id
                ),
                evidence=(heading_evidence, paragraph_evidence),
                observations=(heading_observation, paragraph_observation),
            )

    adapter = _FakePdfAdapter()
    context.provider_registry._adapters[(adapter.provider_id, DeploymentProfile.SINGLE_PASS)] = adapter  # noqa: SLF001
    context.provider_config_store.set(  # type: ignore[union-attr]
        context.provider_config_store.get(adapter.provider_id).model_copy(update={"enabled": True})
    )
    context.provider_availability[adapter.provider_id] = True


def _page(window: MainWindowV2, page: DesktopV2Page):
    index = window._shell.pages.index(page)  # noqa: SLF001
    return window._stack.widget(index)  # noqa: SLF001


def _navigate(window: MainWindowV2, page: DesktopV2Page) -> None:
    index = window._shell.pages.index(page)  # noqa: SLF001
    item = window._rail.item(index)  # noqa: SLF001
    rect = window._rail.visualItemRect(item)  # noqa: SLF001
    QTest.mouseClick(window._rail.viewport(), Qt.MouseButton.LeftButton, pos=rect.center())  # noqa: SLF001
    _wait_until(lambda: window._stack.currentIndex() == index)  # noqa: SLF001


def _all_label_and_button_text(root) -> str:
    from PySide6.QtWidgets import QLabel, QPushButton

    parts = [widget.text() for widget in root.findChildren(QLabel)]
    parts.extend(widget.text() for widget in root.findChildren(QPushButton))
    return " ".join(parts)


def _highlight_rects(page: WorkQueuePage) -> list[QRectF]:
    return [
        item.rect()
        for item in page._scene.items()  # noqa: SLF001
        if isinstance(item, QGraphicsRectItem)
        and item.pen().color().name().lower() == "#d06767"
        and item.rect().width() > 0
        and item.rect().height() > 0
    ]


def test_desktop_v2_one_document_operator_workflow_visual_and_cognitive_audit(qapp, tmp_path) -> None:
    if ARTIFACT_DIR.exists():
        resolved = ARTIFACT_DIR.resolve()
        assert resolved.parts[-2:] == ("test-artifacts", "desktop-v2-interaction")
        shutil.rmtree(resolved)
    ARTIFACT_DIR.mkdir(parents=True)

    context = AppContext(deployment_root=tmp_path / "deployment", persistent_telemetry=False)
    context.current_workspace = context.workspace_store.rename(  # type: ignore[assignment]
        context.current_workspace.id, "Desktop v2 disposable interaction workspace"
    )
    _install_single_pdf_provider(context)

    fixture = tmp_path / "fixtures" / "municipal-record.pdf"
    fixture.parent.mkdir()
    write_minimal_pdf(fixture, num_pages=1, width=320, height=320)

    window = MainWindowV2(context)
    window.resize(1440, 920)
    window.show()
    _wait_until(lambda: window.isVisible())
    recorder = AuditRecorder(window)

    try:
        recorder.capture(
            "01-launch.png",
            current_page=window._shell.current_page.value.value,  # noqa: SLF001
            nav_items=[window._rail.item(i).text() for i in range(window._rail.count())],  # noqa: SLF001
        )
        # Asserted against the enum, not a literal: the rail gained the seven HTR research
        # surfaces in docs/htr-migration-plan.md Stage 11, and will change again.
        assert window._rail.count() == len(DesktopV2Page)  # noqa: SLF001

        _navigate(window, DesktopV2Page.WORKSPACES)
        recorder.capture(
            "02-workspace.png",
            selected_workspace=context.current_workspace.name,
            workspace_id=context.current_workspace.id,
        )
        assert "disposable interaction workspace" in context.current_workspace.name

        _navigate(window, DesktopV2Page.SOURCES)
        source_page = _page(window, DesktopV2Page.SOURCES)
        assert isinstance(source_page, SourcesPage)
        recorder.capture(
            "03-sources-empty.png",
            pending_count=len(context.acquisition_manager.pending),
            configured_sources=len(context.acquisition_manager.sources()),
        )

        source_page.import_paths((str(fixture),))
        _wait_until(lambda: len(context.acquisition_manager.pending) == 1)
        imported = context.acquisition_manager.pending[0]
        assert imported.original_filename == fixture.name
        assert context.acquisition_manager.sources()[0].source_id == "manual-import"
        recorder.capture(
            "04-document-imported.png",
            pending_count=len(context.acquisition_manager.pending),
            selected_document=imported.id,
            source_folder_watchers=len([s for s in context.acquisition_manager.sources() if s.kind.value == "folder_watch"]),
        )

        _navigate(window, DesktopV2Page.PROCESSING)
        processing_page = _page(window, DesktopV2Page.PROCESSING)
        assert isinstance(processing_page, ProcessingPage)
        processing_page._refresh()  # noqa: SLF001
        assert processing_page._process_button.isEnabled()  # noqa: SLF001
        assert len(context.acquisition_manager.pending) == 1
        recorder.capture(
            "05-processing-ready.png",
            pending_count=len(context.acquisition_manager.pending),
            enabled_providers=sorted(context.enabled_provider_ids()),
        )

        QTest.mouseClick(processing_page._process_button, Qt.MouseButton.LeftButton, pos=QPoint(10, 10))  # noqa: SLF001
        _wait_until(lambda: processing_page._worker is not None)  # noqa: SLF001
        recorder.capture(
            "06-processing-running.png",
            current_document=processing_page._current_document.text(),  # noqa: SLF001
            progress=processing_page._progress.text(),  # noqa: SLF001
        )
        _wait_until(lambda: processing_page._worker is None, timeout_ms=10000)  # noqa: SLF001
        progress_events = context.processing_progress.events()
        started = [e for e in progress_events if e.kind == ProcessingProgressKind.DOCUMENT_STARTED]
        completed = [e for e in progress_events if e.kind == ProcessingProgressKind.DOCUMENT_COMPLETED]
        assert len(started) == 1
        assert len(completed) == 1
        assert context.acquisition_manager.pending == []
        recorder.capture(
            "07-processing-complete.png",
            documents_started=len(started),
            documents_completed=len(completed),
            pending_count=len(context.acquisition_manager.pending),
        )

        _navigate(window, DesktopV2Page.WORK_QUEUE)
        queue_page = _page(window, DesktopV2Page.WORK_QUEUE)
        assert isinstance(queue_page, WorkQueuePage)
        queue_page.refresh()
        _wait_until(lambda: len(queue_page._items) == 1)  # noqa: SLF001
        item = queue_page._items[0]  # noqa: SLF001
        row_text = queue_page._list.item(0).text()  # noqa: SLF001
        assert "Operational review" in row_text
        assert "{" not in row_text and "}" not in row_text
        assert "AdaptiveQueueEntry" not in row_text
        recorder.capture(
            "08-work-queue.png",
            queue_size_before=len(queue_page._items),  # noqa: SLF001
            selected_review_item=item.entry.document_ref,
            visible_item=row_text,
        )

        row_rect = queue_page._list.visualItemRect(queue_page._list.item(0))  # noqa: SLF001
        QTest.mouseClick(queue_page._list.viewport(), Qt.MouseButton.LeftButton, pos=row_rect.center())  # noqa: SLF001
        _wait_until(lambda: queue_page._packet is not None)  # noqa: SLF001
        assert queue_page._canvas.isVisible()  # noqa: SLF001
        assert queue_page._scene.items()  # noqa: SLF001
        assert "Recognised text" in queue_page._document_caption.text()  # noqa: SLF001
        recorder.capture(
            "09-review-item-selected.png",
            selected_review_item=item.entry.document_ref,
            canonical_result=queue_page._current.text(),  # noqa: SLF001
            alternatives=queue_page._candidates.count(),  # noqa: SLF001
        )

        rects = _highlight_rects(queue_page)
        assert rects
        page_rect = queue_page._scene.itemsBoundingRect()  # noqa: SLF001
        for rect in rects:
            assert page_rect.contains(rect)
        before_resize = [(r.x(), r.y(), r.width(), r.height()) for r in rects]
        window.resize(1200, 820)
        QApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 100)
        after_resize = [(r.x(), r.y(), r.width(), r.height()) for r in _highlight_rects(queue_page)]
        assert before_resize == after_resize
        queue_page._canvas.scale(1.25, 1.25)  # noqa: SLF001
        after_zoom = [(r.x(), r.y(), r.width(), r.height()) for r in _highlight_rects(queue_page)]
        assert before_resize == after_zoom
        recorder.capture(
            "10-pdf-and-overlays.png",
            page_number=1,
            rendered_page_dimensions=[320, 320],
            overlay_rectangles=[list(values) for values in before_resize],
            overlay_count=len(rects),
        )

        QTest.mouseClick(queue_page._evidence_toggle, Qt.MouseButton.LeftButton, pos=QPoint(10, 10))  # noqa: SLF001
        _wait_until(lambda: queue_page._evidence.isVisible())  # noqa: SLF001
        evidence_text = " ".join(queue_page._evidence.item(i).text() for i in range(queue_page._evidence.count()))  # noqa: SLF001
        assert "Municipal record" in evidence_text or "Approved for review" in evidence_text
        assert "{" not in evidence_text and "}" not in evidence_text
        recorder.capture("11-evidence-expanded.png", evidence_items=queue_page._evidence.count())  # noqa: SLF001

        before_outcomes = sum(isinstance(event, ReviewOutcomeRecorded) for event in context.telemetry.all_events())
        before_corrections = sum(isinstance(event, HumanCorrectionApplied) for event in context.telemetry.all_events())
        QTest.mouseClick(queue_page._skip, Qt.MouseButton.LeftButton, pos=QPoint(10, 10))  # noqa: SLF001
        _wait_until(
            lambda: sum(isinstance(event, ReviewOutcomeRecorded) for event in context.telemetry.all_events())
            == before_outcomes + 1
        )
        assert sum(isinstance(event, HumanCorrectionApplied) for event in context.telemetry.all_events()) == before_corrections
        assert len(queue_page._items) == 1  # noqa: SLF001 - Skip is deferred, so the item remains eligible.
        assert queue_page._packet is None  # noqa: SLF001
        assert not queue_page._scene.items()  # noqa: SLF001
        recorder.capture(
            "12-after-decision.png",
            queue_size_after=len(queue_page._items),  # noqa: SLF001
            review_outcomes_recorded=before_outcomes + 1,
        )

        for page in (
            DesktopV2Page.DOCUMENTS,
            DesktopV2Page.OUTPUTS,
            DesktopV2Page.ADMINISTRATION,
        ):
            _navigate(window, page)
            text = _all_label_and_button_text(_page(window, page))
            assert "{" not in text and "}" not in text
            assert "Traceback" not in text
            recorder.checkpoint(page.value, visible_text=text[:500])
    except Exception as exc:
        recorder.metadata["exceptions"].append(repr(exc))
        raise
    finally:
        recorder.write()
        window.close()
