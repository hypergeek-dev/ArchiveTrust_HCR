from __future__ import annotations

import json
from pathlib import Path

from archivetrust.composition import AppContext
from archivetrust.core import HeadlessCoreService, WorkspaceSummary
from archivetrust.domain.confidence.models import ComparisonClassification
from tests.review._helpers import emit_slot, heading


def _context_with_reviewable_slot(tmp_path: Path) -> AppContext:
    context = AppContext(deployment_root=tmp_path)
    emit_slot(
        context.telemetry,  # type: ignore[arg-type]
        document_ref="doc1",
        canonical_payload=heading("A"),
        provider_payloads=(("docling", heading("A")), ("qwen", heading("B"))),
        classification=ComparisonClassification.CONTESTED,
    )
    return context


def test_app_context_exposes_headless_core_service(tmp_path: Path) -> None:
    context = _context_with_reviewable_slot(tmp_path)
    service = context.core_service()

    assert isinstance(service, HeadlessCoreService)
    assert isinstance(service.workspace(), WorkspaceSummary)
    assert service.workspace().workspace_id == context.current_workspace.id


def test_core_service_review_queue_matches_processing_viewmodel(tmp_path: Path) -> None:
    context = _context_with_reviewable_slot(tmp_path)
    service = context.core_service()

    assert service.review_queue() == context.processing_viewmodel().review_queue()
    assert service.processing_overview().documents_awaiting_review == 1


def test_core_service_exposes_document_lifecycle_and_quality_projections(tmp_path: Path) -> None:
    service = _context_with_reviewable_slot(tmp_path).core_service()

    documents = service.documents()
    assert len(documents) == 1
    assert documents[0].document_ref == "doc1"
    assert service.review_decision_stats().terminal_decisions_recorded == 0
    assert service.confidence_distribution()


def test_core_service_exports_workspace_json_without_qt(tmp_path: Path) -> None:
    service = _context_with_reviewable_slot(tmp_path).core_service()

    payload = json.loads(service.export_workspace_json())

    assert payload["export_schema"] == "archivetrust.canonical_document.v2"
    assert payload["workspace"]["id"] == service.workspace().workspace_id
    assert "documents" in payload


def test_core_service_module_does_not_import_qt_or_desktop_pages() -> None:
    source = Path("src/archivetrust/core/service.py").read_text(encoding="utf-8")

    assert "PySide6" not in source
    assert "clients.desktop.pages" not in source
