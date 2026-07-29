"""Headless tests for the Center ViewModels over the realistic demo archive.

No PySide6 — these verify the operational/analytics data the Qt Center views render is correct and
reproducible, before any Qt is involved. They also assert the demo data is realistic (Swedish
municipal records), not a programming example.
"""

from __future__ import annotations

import pytest

from archivetrust.clients.desktop.demo_data import seed_realistic_archive
from archivetrust.infrastructure.storage.telemetry_sink import InMemoryTelemetrySink
from archivetrust.learning.review.sink import InMemoryReviewInteractionSink
from archivetrust.presentation.evidence_explorer_viewmodel import EvidenceExplorerViewModel
from archivetrust.presentation.evolution_viewmodel import EvolutionCenterViewModel
from archivetrust.presentation.operations_viewmodel import ProcessingCenterViewModel
from archivetrust.presentation.quality_viewmodel import QualityCenterViewModel


@pytest.fixture
def seeded() -> tuple[InMemoryTelemetrySink, InMemoryReviewInteractionSink]:
    sink = InMemoryTelemetrySink()
    interactions = InMemoryReviewInteractionSink()
    seed_realistic_archive(sink, interactions)
    return sink, interactions


def test_demo_data_is_realistic_not_a_programming_example(seeded) -> None:
    sink, _ = seeded
    texts = " ".join(
        e.canonical_observation.payload.model_dump_json()
        for e in sink.all_events()
        if type(e).__name__ == "CanonicalDecisionCreated"
    )
    assert "Kommunfullmäktige" in texts or "Byggnadsnämnden" in texts
    assert "stormy night" not in texts


def test_processing_overview_reflects_the_archive(seeded) -> None:
    sink, _ = seeded
    vm = ProcessingCenterViewModel(sink)
    overview = vm.overview()
    assert overview.documents_processed >= 3
    assert overview.observations_captured > 0
    assert overview.failures >= 1  # one provider rejection was seeded
    assert overview.corrections_applied >= 1
    # The queue is ordered most-uncertain-first.
    queue = vm.queue()
    assert queue[0].uncertainties >= queue[-1].uncertainties


def test_processing_center_without_workspace_or_acquisition_manager_returns_honest_defaults(seeded) -> None:
    # ROADMAP.md §5.13: these fields are optional so the ViewModel stays constructible wherever a
    # Workspace isn't relevant -- verifies the honest-empty-state contract, not fabricated data.
    sink, _ = seeded
    vm = ProcessingCenterViewModel(sink)
    assert vm.current_workspace_summary() is None
    assert vm.input_sources_status() == ()
    assert vm.documents_waiting() == 0
    assert vm.documents_processing() == 0
    assert vm.storage_usage() == vm.storage_usage()  # zeroed, stable


def test_processing_center_with_acquisition_manager_reports_real_state(tmp_path) -> None:
    from archivetrust.acquisition.events import AcquisitionTelemetrySink
    from archivetrust.acquisition.manager import AcquisitionManager
    from archivetrust.acquisition.manual_import import ManualImportSource
    from archivetrust.infrastructure.storage.telemetry_sink import InMemoryTelemetrySink as Sink
    from archivetrust.runtime.provider_profiles import ProviderProfileName
    from archivetrust.workspace.layout import WorkspaceLayout
    from archivetrust.workspace.models import Workspace

    layout = WorkspaceLayout(root=tmp_path / "ws").ensure()
    manager = AcquisitionManager("ws1", layout, AcquisitionTelemetrySink())
    manual = ManualImportSource()
    manager.add_source(manual)
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    (incoming / "a.pdf").write_bytes(b"content")
    manual.import_files(
        (incoming / "a.pdf",), workspace_id="ws1", layout=layout, known_hashes=manager._known_hashes,
        sink=manager._sink,
    )
    workspace = Workspace(id="ws1", name="Research Dataset", processing_profile=ProviderProfileName.RESEARCH)

    vm = ProcessingCenterViewModel(Sink(), workspace=workspace, acquisition_manager=manager)
    summary = vm.current_workspace_summary()
    assert summary is not None and summary.name == "Research Dataset"
    sources = vm.input_sources_status()
    assert len(sources) == 1 and sources[0].documents_imported == 1


def test_recovery_status_is_none_without_a_progress_stream(seeded) -> None:
    sink, _ = seeded
    vm = ProcessingCenterViewModel(sink)
    assert vm.recovery_status() is None


def test_recovery_status_reports_a_clean_finished_run(tmp_path) -> None:
    from archivetrust.application.progress import (
        FileProcessingProgressSink,
        ProcessingProgressEvent,
        ProcessingProgressKind,
        utc_now_iso,
    )
    from archivetrust.infrastructure.storage.telemetry_sink import InMemoryTelemetrySink as Sink

    progress = FileProcessingProgressSink(tmp_path / "processing.jsonl")
    for kind, fields in [
        (ProcessingProgressKind.RUN_STARTED, {"pending_count": 1}),
        (ProcessingProgressKind.DOCUMENT_STARTED, {"archive_object_id": "a1", "original_filename": "one.pdf"}),
        (
            ProcessingProgressKind.DOCUMENT_COMPLETED,
            {"archive_object_id": "a1", "original_filename": "one.pdf", "had_failure": False, "documents_done": 1},
        ),
        (ProcessingProgressKind.RUN_FINISHED, {"documents_done": 1, "cancelled": False}),
    ]:
        progress.record(ProcessingProgressEvent(kind=kind, run_id="run1", recorded_at=utc_now_iso(), **fields))

    vm = ProcessingCenterViewModel(Sink(), processing_progress=progress)
    status = vm.recovery_status()
    assert status is not None
    assert status.was_interrupted is False
    assert status.documents_completed == 1
    assert status.interrupted_document is None


def test_recovery_status_reports_an_interrupted_run(tmp_path) -> None:
    """Production Incident, 2026-07-14: a run that stalled after `DocumentProcessingStarted` for
    one document, with no matching completion and no `ProcessingRunFinished`, must be reported as
    interrupted, naming that exact document -- the failure shape of the 2026-07-14 incident."""
    from archivetrust.application.progress import (
        FileProcessingProgressSink,
        ProcessingProgressEvent,
        ProcessingProgressKind,
        utc_now_iso,
    )
    from archivetrust.infrastructure.storage.telemetry_sink import InMemoryTelemetrySink as Sink

    progress = FileProcessingProgressSink(tmp_path / "processing.jsonl")
    for kind, fields in [
        (ProcessingProgressKind.RUN_STARTED, {"pending_count": 2}),
        (ProcessingProgressKind.DOCUMENT_STARTED, {"archive_object_id": "a1", "original_filename": "one.pdf"}),
        (
            ProcessingProgressKind.DOCUMENT_COMPLETED,
            {"archive_object_id": "a1", "original_filename": "one.pdf", "had_failure": False, "documents_done": 1},
        ),
        (ProcessingProgressKind.DOCUMENT_STARTED, {"archive_object_id": "a2", "original_filename": "two.pdf"}),
        # process dies here -- no DOCUMENT_COMPLETED, no RUN_FINISHED
    ]:
        progress.record(ProcessingProgressEvent(kind=kind, run_id="run1", recorded_at=utc_now_iso(), **fields))

    vm = ProcessingCenterViewModel(Sink(), processing_progress=progress)
    status = vm.recovery_status()
    assert status is not None
    assert status.was_interrupted is True
    assert status.documents_completed == 1
    assert status.last_completed_document == "one.pdf"
    assert status.interrupted_document == "two.pdf"
    assert status.interrupted_at is not None


def test_recovery_status_surfaces_paused_reason_from_a_run_paused_event(tmp_path) -> None:
    """Provider Health Guard (Phase 22): distinguishes "gracefully paused, here's why" from an
    unexplained crash -- both are `was_interrupted=True`, but only the guard-paused case has a
    `paused_reason`."""
    from archivetrust.application.progress import (
        FileProcessingProgressSink,
        ProcessingProgressEvent,
        ProcessingProgressKind,
        utc_now_iso,
    )
    from archivetrust.infrastructure.storage.telemetry_sink import InMemoryTelemetrySink as Sink
    from archivetrust.presentation.operations_viewmodel import PipelineState

    progress = FileProcessingProgressSink(tmp_path / "processing.jsonl")
    for kind, fields in [
        (ProcessingProgressKind.RUN_STARTED, {"pending_count": 2}),
        (
            ProcessingProgressKind.RUN_PAUSED,
            {
                "paused_provider_id": "paddleocr-vl",
                "paused_reason": "Required provider(s) unavailable -- paddleocr-vl: not currently reachable",
                "recovery_attempted": True,
                "recovery_succeeded": False,
            },
        ),
    ]:
        progress.record(ProcessingProgressEvent(kind=kind, run_id="run1", recorded_at=utc_now_iso(), **fields))

    vm = ProcessingCenterViewModel(Sink(), processing_progress=progress)
    status = vm.recovery_status()
    assert status is not None
    assert status.was_interrupted is True
    assert status.paused_reason == "Required provider(s) unavailable -- paddleocr-vl: not currently reachable"

    health = vm.pipeline_health()
    assert health.state == PipelineState.PAUSED
    assert health.reason == status.paused_reason


def test_recovery_status_has_no_paused_reason_for_an_unexplained_interruption(tmp_path) -> None:
    """Regression proof: the pre-existing crash-recovery path (no RUN_PAUSED event) is unchanged
    -- `paused_reason` stays `None` and `pipeline_health()` still reports RECOVERING, not PAUSED."""
    from archivetrust.application.progress import (
        FileProcessingProgressSink,
        ProcessingProgressEvent,
        ProcessingProgressKind,
        utc_now_iso,
    )
    from archivetrust.infrastructure.storage.telemetry_sink import InMemoryTelemetrySink as Sink
    from archivetrust.presentation.operations_viewmodel import PipelineState

    progress = FileProcessingProgressSink(tmp_path / "processing.jsonl")
    progress.record(
        ProcessingProgressEvent(
            kind=ProcessingProgressKind.RUN_STARTED, run_id="run1", recorded_at=utc_now_iso(), pending_count=1
        )
    )

    vm = ProcessingCenterViewModel(Sink(), processing_progress=progress)
    status = vm.recovery_status()
    assert status is not None
    assert status.was_interrupted is True
    assert status.paused_reason is None
    assert vm.pipeline_health().state == PipelineState.RECOVERING


def test_throughput_is_empty_without_a_progress_stream(seeded) -> None:
    sink, _ = seeded
    vm = ProcessingCenterViewModel(sink)
    stats = vm.throughput()
    assert stats.documents_measured == 0
    assert stats.average_seconds_per_document is None
    assert stats.documents_per_minute is None
    assert stats.estimated_seconds_remaining is None


def test_throughput_averages_real_recorded_durations(tmp_path) -> None:
    """Phase 34: throughput must come from `DocumentProcessingCompleted.duration_seconds` (the
    real, `time.monotonic()`-measured wall-clock time `QueueWorker` itself recorded), not any
    reconstruction from domain telemetry timestamps."""
    from archivetrust.application.progress import (
        FileProcessingProgressSink,
        ProcessingProgressEvent,
        ProcessingProgressKind,
        utc_now_iso,
    )
    from archivetrust.infrastructure.storage.telemetry_sink import InMemoryTelemetrySink as Sink

    progress = FileProcessingProgressSink(tmp_path / "processing.jsonl")
    for kind, fields in [
        (ProcessingProgressKind.RUN_STARTED, {"pending_count": 3}),
        (ProcessingProgressKind.DOCUMENT_STARTED, {"archive_object_id": "a1", "original_filename": "one.pdf"}),
        (
            ProcessingProgressKind.DOCUMENT_COMPLETED,
            {"archive_object_id": "a1", "original_filename": "one.pdf", "duration_seconds": 10.0, "documents_done": 1},
        ),
        (ProcessingProgressKind.DOCUMENT_STARTED, {"archive_object_id": "a2", "original_filename": "two.pdf"}),
        (
            ProcessingProgressKind.DOCUMENT_COMPLETED,
            {"archive_object_id": "a2", "original_filename": "two.pdf", "duration_seconds": 20.0, "documents_done": 2},
        ),
        (ProcessingProgressKind.RUN_FINISHED, {"documents_done": 2, "cancelled": False}),
    ]:
        progress.record(ProcessingProgressEvent(kind=kind, run_id="run1", recorded_at=utc_now_iso(), **fields))

    vm = ProcessingCenterViewModel(Sink(), processing_progress=progress)
    stats = vm.throughput()
    assert stats.documents_measured == 2
    assert stats.average_seconds_per_document == 15.0  # (10.0 + 20.0) / 2, not some reconstructed span


def test_throughput_excludes_an_interrupted_attempt_never_retried_in_this_run(tmp_path) -> None:
    """The 2026-07-14/Phase-33 failure shape: a document is started but the process dies before it
    completes. It must never contribute to the average -- there is no duration to measure, and the
    old first_seen/last_seen approach would have (wrongly) measured the gap until whenever a later
    session finally finished it. Only real completions count."""
    from archivetrust.application.progress import (
        FileProcessingProgressSink,
        ProcessingProgressEvent,
        ProcessingProgressKind,
        utc_now_iso,
    )
    from archivetrust.infrastructure.storage.telemetry_sink import InMemoryTelemetrySink as Sink

    progress = FileProcessingProgressSink(tmp_path / "processing.jsonl")
    for kind, fields in [
        (ProcessingProgressKind.RUN_STARTED, {"pending_count": 2}),
        (ProcessingProgressKind.DOCUMENT_STARTED, {"archive_object_id": "a1", "original_filename": "one.pdf"}),
        (
            ProcessingProgressKind.DOCUMENT_COMPLETED,
            {"archive_object_id": "a1", "original_filename": "one.pdf", "duration_seconds": 5.0, "documents_done": 1},
        ),
        (ProcessingProgressKind.DOCUMENT_STARTED, {"archive_object_id": "a2", "original_filename": "two.pdf"}),
        # process dies here -- no DOCUMENT_COMPLETED for a2, no RUN_FINISHED
    ]:
        progress.record(ProcessingProgressEvent(kind=kind, run_id="run1", recorded_at=utc_now_iso(), **fields))

    vm = ProcessingCenterViewModel(Sink(), processing_progress=progress)
    stats = vm.throughput()
    assert stats.documents_measured == 1
    assert stats.average_seconds_per_document == 5.0


def test_throughput_includes_a_retried_documents_eventual_real_duration_only(tmp_path) -> None:
    """A document interrupted in one run and successfully retried in a later one (a fresh run_id,
    the exact shape Phase 33 found in real telemetry) must contribute only the retry's own real
    duration -- never the wall-clock gap between the abandoned attempt and the retry."""
    from archivetrust.application.progress import (
        FileProcessingProgressSink,
        ProcessingProgressEvent,
        ProcessingProgressKind,
        utc_now_iso,
    )
    from archivetrust.infrastructure.storage.telemetry_sink import InMemoryTelemetrySink as Sink

    progress = FileProcessingProgressSink(tmp_path / "processing.jsonl")
    for kind, fields, run_id in [
        (ProcessingProgressKind.RUN_STARTED, {"pending_count": 1}, "run1"),
        (ProcessingProgressKind.DOCUMENT_STARTED, {"archive_object_id": "a1", "original_filename": "one.pdf"}, "run1"),
        # process dies here mid-document -- no DOCUMENT_COMPLETED for run1's attempt
        # ... a long time later, a fresh session retries it ...
        (ProcessingProgressKind.RUN_STARTED, {"pending_count": 1}, "run2"),
        (ProcessingProgressKind.DOCUMENT_STARTED, {"archive_object_id": "a1", "original_filename": "one.pdf"}, "run2"),
        (
            ProcessingProgressKind.DOCUMENT_COMPLETED,
            {"archive_object_id": "a1", "original_filename": "one.pdf", "duration_seconds": 25.0, "documents_done": 1},
            "run2",
        ),
        (ProcessingProgressKind.RUN_FINISHED, {"documents_done": 1, "cancelled": False}, "run2"),
    ]:
        progress.record(ProcessingProgressEvent(kind=kind, run_id=run_id, recorded_at=utc_now_iso(), **fields))

    vm = ProcessingCenterViewModel(Sink(), processing_progress=progress)
    stats = vm.throughput()
    assert stats.documents_measured == 1
    assert stats.average_seconds_per_document == 25.0  # the real retry duration, not any inter-session gap


def test_throughput_ignores_human_review_activity_entirely(tmp_path) -> None:
    """Phase 33's second confirmed mechanism: a human reviewing a document long after it finished
    must never move the reported throughput -- review is a different telemetry stream this method
    never reads at all now, not merely a filtered-out value."""
    from archivetrust.domain.telemetry.events import HumanCorrectionSubmitted
    from archivetrust.application.progress import (
        FileProcessingProgressSink,
        ProcessingProgressEvent,
        ProcessingProgressKind,
        utc_now_iso,
    )
    from archivetrust.infrastructure.storage.telemetry_sink import InMemoryTelemetrySink as Sink

    progress = FileProcessingProgressSink(tmp_path / "processing.jsonl")
    for kind, fields in [
        (ProcessingProgressKind.RUN_STARTED, {"pending_count": 1}),
        (ProcessingProgressKind.DOCUMENT_STARTED, {"archive_object_id": "a1", "original_filename": "one.pdf"}),
        (
            ProcessingProgressKind.DOCUMENT_COMPLETED,
            {"archive_object_id": "a1", "original_filename": "one.pdf", "duration_seconds": 8.0, "documents_done": 1},
        ),
        (ProcessingProgressKind.RUN_FINISHED, {"documents_done": 1, "cancelled": False}),
    ]:
        progress.record(ProcessingProgressEvent(kind=kind, run_id="run1", recorded_at=utc_now_iso(), **fields))

    sink = Sink()
    vm = ProcessingCenterViewModel(sink, processing_progress=progress)
    before = vm.throughput()

    # A human corrects something on this same document, "long" after processing finished -- this
    # must not be able to move throughput()'s numbers, since it no longer reads this stream at all.
    sink.append(
        HumanCorrectionSubmitted(
            event_id="evt1", document_ref="doc_a1", correction_id="c1",
            target_canonical_observation_id="co1", category="other", action="accept",
            raw_ai_output="text", reviewer_ref="reviewer", submitted_at=utc_now_iso(),
            review_duration_seconds=2.0,
        )
    )
    after = vm.throughput()
    assert after == before
    assert after.average_seconds_per_document == 8.0


def test_throughput_skips_historical_completions_with_no_duration_recorded(tmp_path) -> None:
    """`duration_seconds` is `None` for `DocumentProcessingCompleted` events recorded before that
    field existed (Observability milestone, 2026-07-13) -- these must be skipped, never treated as
    a real zero-second completion (which would wrongly drag the average down)."""
    from archivetrust.application.progress import (
        FileProcessingProgressSink,
        ProcessingProgressEvent,
        ProcessingProgressKind,
        utc_now_iso,
    )
    from archivetrust.infrastructure.storage.telemetry_sink import InMemoryTelemetrySink as Sink

    progress = FileProcessingProgressSink(tmp_path / "processing.jsonl")
    for kind, fields in [
        (ProcessingProgressKind.RUN_STARTED, {"pending_count": 2}),
        # A historical completion with no duration_seconds at all (pre-dates that field).
        (
            ProcessingProgressKind.DOCUMENT_COMPLETED,
            {"archive_object_id": "old1", "original_filename": "old.pdf", "documents_done": 1},
        ),
        (ProcessingProgressKind.DOCUMENT_STARTED, {"archive_object_id": "a1", "original_filename": "one.pdf"}),
        (
            ProcessingProgressKind.DOCUMENT_COMPLETED,
            {"archive_object_id": "a1", "original_filename": "one.pdf", "duration_seconds": 12.0, "documents_done": 2},
        ),
    ]:
        progress.record(ProcessingProgressEvent(kind=kind, run_id="run1", recorded_at=utc_now_iso(), **fields))

    vm = ProcessingCenterViewModel(Sink(), processing_progress=progress)
    stats = vm.throughput()
    assert stats.documents_measured == 1  # only the one with a real duration
    assert stats.average_seconds_per_document == 12.0


def test_throughput_eta_uses_documents_waiting_and_the_measured_average(tmp_path) -> None:
    from archivetrust.acquisition.archive_object import ArchiveObject
    from archivetrust.acquisition.events import AcquisitionTelemetrySink
    from archivetrust.acquisition.manager import AcquisitionManager
    from archivetrust.application.progress import (
        FileProcessingProgressSink,
        ProcessingProgressEvent,
        ProcessingProgressKind,
        utc_now_iso,
    )
    from archivetrust.infrastructure.storage.telemetry_sink import InMemoryTelemetrySink as Sink
    from archivetrust.workspace.layout import WorkspaceLayout

    progress = FileProcessingProgressSink(tmp_path / "processing.jsonl")
    for kind, fields in [
        (ProcessingProgressKind.RUN_STARTED, {"pending_count": 3}),
        (ProcessingProgressKind.DOCUMENT_STARTED, {"archive_object_id": "a1", "original_filename": "one.pdf"}),
        (
            ProcessingProgressKind.DOCUMENT_COMPLETED,
            {"archive_object_id": "a1", "original_filename": "one.pdf", "duration_seconds": 30.0, "documents_done": 1},
        ),
    ]:
        progress.record(ProcessingProgressEvent(kind=kind, run_id="run1", recorded_at=utc_now_iso(), **fields))

    layout = WorkspaceLayout(root=tmp_path / "ws").ensure()
    manager = AcquisitionManager("ws1", layout, AcquisitionTelemetrySink())
    for name in ("b.pdf", "c.pdf"):
        manager.pending.append(
            ArchiveObject(
                id=f"archive_object_{name}", workspace_id="ws1", source_id="manual",
                original_filename=name, content_hash=f"hash-{name}", byte_size=10,
                mime_type="application/pdf", storage_path=name,
            )
        )

    vm = ProcessingCenterViewModel(Sink(), acquisition_manager=manager, processing_progress=progress)
    stats = vm.throughput()
    assert stats.average_seconds_per_document == 30.0
    assert vm.documents_waiting() == 2
    assert stats.estimated_seconds_remaining == 60.0  # 2 waiting * 30.0s average
    assert stats.documents_per_minute == pytest.approx(2.0)


def test_gpu_usage_is_none_without_a_gpu_resource_manager(seeded) -> None:
    sink, _ = seeded
    vm = ProcessingCenterViewModel(sink)
    assert vm.gpu_usage() is None


def test_gpu_usage_reflects_active_reservations(seeded) -> None:
    from archivetrust.runtime.gpu_resource_manager import GPUResourceManager

    class _FakeGpuInfo:
        def total_memory_bytes(self, gpu_device=None):
            return 8_000_000_000

        def used_memory_bytes(self, gpu_device=None):
            return 0

    sink, _ = seeded
    manager = GPUResourceManager(gpu_info=_FakeGpuInfo())
    manager.reserve(runtime_key="paddleocr-vl", model_id="PaddlePaddle/PaddleOCR-VL")

    vm = ProcessingCenterViewModel(sink, gpu_resource_manager=manager)
    usage = vm.gpu_usage()
    assert usage is not None
    assert usage.total_bytes == 8_000_000_000
    assert usage.reserved_bytes > 0
    assert usage.reservations == ("PaddlePaddle/PaddleOCR-VL",)


def test_pipeline_health_is_recovering_when_interrupted(tmp_path) -> None:
    from archivetrust.application.progress import (
        FileProcessingProgressSink,
        ProcessingProgressEvent,
        ProcessingProgressKind,
        utc_now_iso,
    )
    from archivetrust.infrastructure.storage.telemetry_sink import InMemoryTelemetrySink as Sink
    from archivetrust.presentation.operations_viewmodel import PipelineState

    progress = FileProcessingProgressSink(tmp_path / "processing.jsonl")
    progress.record(
        ProcessingProgressEvent(
            kind=ProcessingProgressKind.RUN_STARTED, run_id="run1", recorded_at=utc_now_iso(), pending_count=1
        )
    )
    progress.record(
        ProcessingProgressEvent(
            kind=ProcessingProgressKind.DOCUMENT_STARTED,
            run_id="run1", recorded_at=utc_now_iso(),
            archive_object_id="a1", original_filename="one.pdf",
        )
    )

    vm = ProcessingCenterViewModel(Sink(), processing_progress=progress)
    health = vm.pipeline_health()
    assert health.state == PipelineState.RECOVERING


def test_pipeline_health_is_idle_with_no_documents(seeded) -> None:
    from archivetrust.presentation.operations_viewmodel import PipelineState

    sink, _ = seeded
    vm = ProcessingCenterViewModel(sink)
    health = vm.pipeline_health()
    assert health.state in (PipelineState.IDLE, PipelineState.WARNING)


def test_quality_center_reuses_analytics(seeded) -> None:
    sink, interactions = seeded
    vm = QualityCenterViewModel(sink, interactions)
    assert vm.observation_survival()  # per-type survival present
    assert vm.human_effort().session_count >= 1  # seeded review sessions
    assert sum(b.count for b in vm.confidence_distribution()) > 0
    assert vm.correction_breakdown().total >= 1


def test_evolution_center_surfaces_candidates_and_alignment(seeded) -> None:
    sink, _ = seeded
    vm = EvolutionCenterViewModel(sink)
    # The VLM-only signature and single-provider metadata are real single-source concepts.
    subjects = {c.subject for c in vm.candidates()}
    assert "handwritten_note" in subjects or "metadata" in subjects
    alignment = vm.alignment_summary()
    assert alignment.attempts > 0
    assert alignment.unaligned_observations >= 1  # the VLM-only signature


def test_evidence_explorer_reconstructs_the_trust_chain(seeded) -> None:
    sink, _ = seeded
    vm = EvidenceExplorerViewModel(sink)
    documents = vm.documents()
    assert len(documents) >= 3
    trace = vm.document_trace(documents[0])
    assert trace.canonical_traces
    # Every canonical fact traces to at least one contributor with backing evidence.
    first = trace.canonical_traces[0]
    assert first.contributors
    assert first.contributors[0].evidence
