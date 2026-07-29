from __future__ import annotations

from archivetrust.application.progress import (
    InMemoryProcessingProgressSink,
    ProcessingProgressEvent,
    ProcessingProgressKind,
    utc_now_iso,
)
from archivetrust.presentation.document_lifecycle import (
    LifecycleState,
    document_lifecycle_details,
    processing_run_history,
)
from tests.review._helpers import emit_slot, heading


def test_lifecycle_marks_imported_document_as_waiting(tmp_path) -> None:
    from archivetrust.composition import AppContext

    context = AppContext(deployment_root=tmp_path)
    sample = tmp_path / "waiting.html"
    sample.write_text("<html><body>Waiting</body></html>", encoding="utf-8")
    context.acquisition_manager_viewmodel().import_files((str(sample),))

    details = document_lifecycle_details(
        processing=context.processing_viewmodel(),
        acquisition_manager=context.acquisition_manager,
        telemetry_source=context.telemetry,
        processing_progress=context.processing_progress,
    )

    assert len(details) == 1
    assert details[0].display_name == "waiting.html"
    assert details[0].state is LifecycleState.WAITING
    assert details[0].next_action == "Process queue"


def test_lifecycle_marks_reviewable_document_as_needing_attention(tmp_path) -> None:
    from archivetrust.composition import AppContext
    from archivetrust.domain.confidence.models import ComparisonClassification

    context = AppContext(deployment_root=tmp_path)
    emit_slot(
        context.telemetry,
        document_ref="doc1",
        canonical_payload=heading("A"),
        provider_payloads=(("docling", heading("A")),),
        classification=ComparisonClassification.UNCORROBORATED_SINGLE_SOURCE,
    )

    detail = document_lifecycle_details(
        processing=context.processing_viewmodel(),
        acquisition_manager=context.acquisition_manager,
        telemetry_source=context.telemetry,
        processing_progress=context.processing_progress,
    )[0]

    assert detail.state is LifecycleState.NEEDS_ATTENTION
    assert detail.next_action == "Open work queue"
    assert detail.canonical_history


def test_lifecycle_detects_interrupted_processing_run(tmp_path) -> None:
    from archivetrust.composition import AppContext

    context = AppContext(deployment_root=tmp_path)
    sample = tmp_path / "interrupted.html"
    sample.write_text("<html><body>Interrupted</body></html>", encoding="utf-8")
    archive_object = context.acquisition_manager_viewmodel().import_files((str(sample),))[0]
    context.processing_progress.record(
        ProcessingProgressEvent(
            kind=ProcessingProgressKind.RUN_STARTED,
            run_id="run1",
            recorded_at=utc_now_iso(),
            provider_ids=("docling",),
            pending_count=1,
        )
    )
    context.processing_progress.record(
        ProcessingProgressEvent(
            kind=ProcessingProgressKind.DOCUMENT_STARTED,
            run_id="run1",
            recorded_at=utc_now_iso(),
            archive_object_id=archive_object.id,
            original_filename=archive_object.original_filename,
        )
    )

    detail = document_lifecycle_details(
        processing=context.processing_viewmodel(),
        acquisition_manager=context.acquisition_manager,
        telemetry_source=context.telemetry,
        processing_progress=context.processing_progress,
    )[0]

    assert detail.state is LifecycleState.INTERRUPTED
    assert detail.next_action == "Resume processing"
    assert detail.processing_runs[0].state == "Interrupted"


def test_run_history_summarizes_completed_failed_and_paused_runs() -> None:
    progress = InMemoryProcessingProgressSink()
    progress.record(
        ProcessingProgressEvent(
            kind=ProcessingProgressKind.RUN_STARTED,
            run_id="run1",
            recorded_at=utc_now_iso(),
            provider_ids=("docling", "tesseract_layoutparser"),
            pending_count=2,
        )
    )
    progress.record(
        ProcessingProgressEvent(
            kind=ProcessingProgressKind.DOCUMENT_COMPLETED,
            run_id="run1",
            recorded_at=utc_now_iso(),
            archive_object_id="a1",
            original_filename="one.pdf",
            had_failure=False,
            documents_done=1,
        )
    )
    progress.record(
        ProcessingProgressEvent(
            kind=ProcessingProgressKind.RUN_PAUSED,
            run_id="run1",
            recorded_at=utc_now_iso(),
            paused_reason="Docling is unavailable.",
        )
    )

    row = processing_run_history(progress)[0]

    assert row.state == "Paused"
    assert row.documents_total == 2
    assert row.documents_completed == 1
    assert row.failure_reason == "Docling is unavailable."
    # Historical telemetry naming a deleted OCR provider stays readable, and is labeled as retired
    # rather than presented as a current HTR method (docs/htr-domain-design.md section 8).
    assert "Docling (retired OCR method)" in row.providers
    assert "Tesseract + LayoutParser (retired OCR method)" in row.providers
