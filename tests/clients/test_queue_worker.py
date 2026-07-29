"""`QueueWorker` tests — offscreen Qt, but `run()` is called directly (never `.start()`), so these
run synchronously and headless while still exercising the real `QThread`/`Signal` machinery.
Proves the Operational Completion milestone's core acceptance criterion: dropping a real file into
an Acquisition Source and pressing Process Queue moves it through the *unchanged* Trust Engine
without any Python-shell call — the whole `AppContext` is real, only the file is a tiny real HTML
document exercising the actually-installed `docling` in this environment (or, if `docling` is not
installed elsewhere, its adapter simply reports zero Observations without raising -- an available
adapter and an empty pipeline are both handled).
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from archivetrust.composition import AppContext  # noqa: E402


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


def _seed_manual_import(context: AppContext, tmp_path, filename: str, content: bytes) -> None:
    incoming = tmp_path / "incoming"
    incoming.mkdir(exist_ok=True)
    path = incoming / filename
    path.write_bytes(content)
    from archivetrust.acquisition.manual_import import ManualImportSource

    manual = ManualImportSource()
    context.acquisition_manager.add_source(manual)
    context.acquisition_manager.import_files(manual, (path,))


def _install_fake_docling_provider(context: AppContext) -> None:
    """Registers a minimal, real `ProviderAdapter` under `provider_id="docling"` so a queue run
    actually succeeds (docs/htr-migration-plan.md Stage 5 -- the real Docling adapter this module's
    tests used to rely on being actually-installed-or-honestly-empty was deleted along with the
    rest of the OCR-era provider architecture). Registered under the literal id `"docling"`,
    matching every `ProviderHealthGuard(required_provider_ids=frozenset({"docling"}))` already
    written throughout this file, so those tests need no further change."""
    import json

    from archivetrust.domain.evidence.models import Evidence, ProcessingStage
    from archivetrust.domain.ontology.base import Observation
    from archivetrust.domain.ontology.payloads import HeadingPayload
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

    class _FakeDoclingAdapter(DeterministicProviderAdapter):
        registration = ProviderRegistration(
            provider_id="docling",
            reproducibility=Reproducibility.DETERMINISTIC,
            deployment_locality=DeploymentLocality.IN_PROCESS,
            introspectability=Introspectability.OPEN,
        )

        def observe(self, *, document_ref: str, invocation_id: str, source: object) -> ProviderRunResult:
            evidence = Evidence.create(
                provider=self.provider_id,
                provider_version="1.0",
                raw_output=json.dumps({"text": "Kommunfullmaktige"}),
                processing_stage=ProcessingStage.RAW,
            )
            observation = Observation.from_evidence(
                provider_id=self.provider_id,
                provider_version="1.0",
                payload=HeadingPayload(text="Kommunfullmaktige", level=1),
                evidence=(evidence,),
            )
            return ProviderRunResult(
                attempt=ProviderAttempt(
                    provider_id=self.provider_id, provider_version="1.0", invocation_id=invocation_id
                ),
                evidence=(evidence,),
                observations=(observation,),
            )

    adapter = _FakeDoclingAdapter()
    context.provider_registry._adapters[(adapter.provider_id, DeploymentProfile.SINGLE_PASS)] = adapter  # noqa: SLF001
    context.provider_config_store.set(  # type: ignore[union-attr]
        context.provider_config_store.get(adapter.provider_id).model_copy(update={"enabled": True})
    )
    context.provider_availability[adapter.provider_id] = True


def test_process_queue_drains_pending_and_reaches_a_canonical_document(qapp, tmp_path) -> None:
    context = AppContext(deployment_root=tmp_path)
    _install_fake_docling_provider(context)
    _seed_manual_import(
        context, tmp_path, "protokoll.html",
        b"<html><body><h1>Kommunfullmaktige</h1><p>Beslut.</p></body></html>",
    )
    context.provider_config_store.set(
        context.provider_config_store.get("tesseract_layoutparser").model_copy(update={"enabled": False})
    )
    assert len(context.acquisition_manager.pending) == 1

    worker = context.queue_worker()
    events = {"started": [], "completed": [], "progress": [], "activity": [], "finished": False}
    worker.document_started.connect(lambda name: events["started"].append(name))
    worker.document_completed.connect(lambda name, failed: events["completed"].append((name, failed)))
    worker.queue_progress.connect(lambda done, total: events["progress"].append((done, total)))
    worker.activity.connect(lambda line: events["activity"].append(line))
    worker.run_finished.connect(lambda: events.__setitem__("finished", True))

    worker.run()  # synchronous, headless -- proves the sequencing logic without a real thread

    assert events["started"] == ["protokoll.html"]
    assert events["completed"] == [("protokoll.html", False)]
    assert events["progress"][-1] == (1, 1)
    assert events["finished"] is True
    assert context.acquisition_manager.pending == []
    assert any("Completed protokoll.html" in line for line in events["activity"])

    overview = context.processing_viewmodel().overview()
    assert overview.documents_processed == 1


def test_a_processing_failure_does_not_stop_the_queue(qapp, tmp_path) -> None:
    context = AppContext(deployment_root=tmp_path)
    _seed_manual_import(context, tmp_path, "a.html", b"<html><body><p>A</p></body></html>")
    _seed_manual_import(context, tmp_path, "b.html", b"<html><body><p>B</p></body></html>")
    # Disabling every provider forces run_pipeline's "at least one invocation" requirement to fail
    # for every document -- proving a failure on one document does not abort the remaining ones.
    for provider_id in ("docling", "tesseract_layoutparser"):
        context.provider_config_store.set(
            context.provider_config_store.get(provider_id).model_copy(update={"enabled": False})
        )

    worker = context.queue_worker()
    completed = []
    worker.document_completed.connect(lambda name, failed: completed.append((name, failed)))
    worker.run()

    assert completed == [("a.html", True), ("b.html", True)]
    assert context.acquisition_manager.pending == []
    # Regression test: the "Failed" tile must never read 0 while every document actually failed --
    # these failures never reach EvidenceRejected telemetry (run_pipeline raised before any
    # provider ran), so ProcessingCenterViewModel.documents_failed() must fold in queue_failure_count.
    assert context.queue_failure_count == 2
    assert context.processing_viewmodel().documents_failed() == 2


def test_pause_and_cancel_flags_are_respected_between_documents(qapp, tmp_path) -> None:
    context = AppContext(deployment_root=tmp_path)
    _seed_manual_import(context, tmp_path, "a.html", b"<html><body><p>A</p></body></html>")
    _seed_manual_import(context, tmp_path, "b.html", b"<html><body><p>B</p></body></html>")

    worker = context.queue_worker()
    worker.cancel()  # cancel before any document is processed

    worker.run()

    # Cancelling immediately means nothing was processed; both remain pending.
    assert len(context.acquisition_manager.pending) == 2


def test_running_provider_ids_are_cleared_after_a_run(qapp, tmp_path) -> None:
    context = AppContext(deployment_root=tmp_path)
    _seed_manual_import(context, tmp_path, "a.html", b"<html><body><p>A</p></body></html>")

    worker = context.queue_worker()
    worker.run()

    assert context.running_provider_ids == set()


def test_unhealthy_required_provider_pauses_before_the_first_document(qapp, tmp_path) -> None:
    """Provider Health Guard (Phase 22): if a required provider is unavailable, processing must
    stop before the next document begins -- zero documents processed, nothing popped from the
    queue, and the pause is recorded durably (not just inferred from a missing RUN_FINISHED)."""
    from archivetrust.acquisition.provider_health_guard import ProviderHealthGuard
    from archivetrust.application.progress import ProcessingProgressKind, ProcessingRunTerminalState
    from archivetrust.clients.desktop.queue_worker import QueueWorker

    context = AppContext(deployment_root=tmp_path)
    _seed_manual_import(context, tmp_path, "a.html", b"<html><body><p>A</p></body></html>")
    _seed_manual_import(context, tmp_path, "b.html", b"<html><body><p>B</p></body></html>")

    always_unhealthy = ProviderHealthGuard(required_provider_ids=frozenset({"docling"}), probes={"docling": lambda: False})
    worker = QueueWorker(context, health_guard=always_unhealthy)
    alerts = []
    worker.provider_health_alert.connect(alerts.append)
    completed = []
    worker.document_completed.connect(lambda name, failed: completed.append(name))

    worker.run()

    assert completed == []
    assert len(context.acquisition_manager.pending) == 2
    assert len(alerts) == 1
    assert "docling" in alerts[0]

    events = context.processing_progress.events()
    kinds = [e.kind for e in events]
    assert kinds == [
        ProcessingProgressKind.RUN_STARTED,
        ProcessingProgressKind.RUN_PAUSED,
        ProcessingProgressKind.RUN_FINISHED,
    ]
    paused_event = events[-2]
    assert paused_event.paused_provider_id == "docling"
    assert paused_event.paused_reason is not None
    assert paused_event.recovery_attempted is False
    assert paused_event.recovery_succeeded is None
    assert events[-1].run_terminal_state is ProcessingRunTerminalState.INTERRUPTED


def test_provider_becoming_unhealthy_mid_run_stops_before_the_next_document(qapp, tmp_path) -> None:
    from archivetrust.acquisition.provider_health_guard import ProviderHealthGuard
    from archivetrust.application.progress import ProcessingProgressKind, ProcessingRunTerminalState
    from archivetrust.clients.desktop.queue_worker import QueueWorker

    context = AppContext(deployment_root=tmp_path)
    _seed_manual_import(context, tmp_path, "a.html", b"<html><body><p>A</p></body></html>")
    _seed_manual_import(context, tmp_path, "b.html", b"<html><body><p>B</p></body></html>")

    calls = {"n": 0}

    def flaky_probe() -> bool:
        calls["n"] += 1
        return calls["n"] == 1  # healthy for document 1's check, unhealthy from then on

    guard = ProviderHealthGuard(required_provider_ids=frozenset({"docling"}), probes={"docling": flaky_probe})
    worker = QueueWorker(context, health_guard=guard)
    completed = []
    worker.document_completed.connect(lambda name, failed: completed.append(name))

    worker.run()

    assert len(completed) == 1
    assert len(context.acquisition_manager.pending) == 1

    events = context.processing_progress.events()
    assert ProcessingProgressKind.RUN_PAUSED in [e.kind for e in events]
    assert events[-1].kind is ProcessingProgressKind.RUN_FINISHED
    assert events[-1].run_terminal_state is ProcessingRunTerminalState.INTERRUPTED


def test_healthy_providers_leave_existing_behavior_unchanged(qapp, tmp_path) -> None:
    """Regression proof: a run where every required provider stays healthy behaves exactly as it
    did before the Provider Health Guard existed -- all documents processed, RUN_FINISHED
    recorded, no health alert, no RUN_PAUSED."""
    from archivetrust.acquisition.provider_health_guard import ProviderHealthGuard
    from archivetrust.application.progress import ProcessingProgressKind
    from archivetrust.clients.desktop.queue_worker import QueueWorker

    context = AppContext(deployment_root=tmp_path)
    _install_fake_docling_provider(context)
    _seed_manual_import(
        context, tmp_path, "protokoll.html",
        b"<html><body><h1>Kommunfullmaktige</h1><p>Beslut.</p></body></html>",
    )
    context.provider_config_store.set(
        context.provider_config_store.get("tesseract_layoutparser").model_copy(update={"enabled": False})
    )

    always_healthy = ProviderHealthGuard(required_provider_ids=frozenset({"docling"}), probes={"docling": lambda: True})
    worker = QueueWorker(context, health_guard=always_healthy)
    alerts = []
    worker.provider_health_alert.connect(alerts.append)
    completed = []
    worker.document_completed.connect(lambda name, failed: completed.append((name, failed)))

    worker.run()

    assert completed == [("protokoll.html", False)]
    assert context.acquisition_manager.pending == []
    assert alerts == []
    kinds = [e.kind for e in context.processing_progress.events()]
    assert ProcessingProgressKind.RUN_FINISHED in kinds
    assert ProcessingProgressKind.RUN_PAUSED not in kinds


def test_run_paused_event_round_trips_through_the_durable_file_sink(qapp, tmp_path) -> None:
    from archivetrust.acquisition.provider_health_guard import ProviderHealthGuard
    from archivetrust.application.progress import FileProcessingProgressSink, ProcessingProgressKind
    from archivetrust.clients.desktop.queue_worker import QueueWorker

    context = AppContext(deployment_root=tmp_path)
    _seed_manual_import(context, tmp_path, "a.html", b"<html><body><p>A</p></body></html>")

    guard = ProviderHealthGuard(required_provider_ids=frozenset({"docling"}), probes={"docling": lambda: False})
    worker = QueueWorker(context, health_guard=guard)
    worker.run()

    layout = context._current_layout
    reopened = FileProcessingProgressSink(layout.telemetry_dir / "processing.jsonl")
    paused = next(e for e in reopened.events() if e.kind == ProcessingProgressKind.RUN_PAUSED)
    assert paused.paused_provider_id == "docling"
    assert paused.paused_reason is not None
    assert paused.recovery_attempted is False
    assert paused.recovery_succeeded is None


def test_run_records_a_durable_progress_stream(qapp, tmp_path) -> None:
    """Production Hardening Review (2026-07-13): the domain stream flushes a document's events only
    when the document *completes*, so before the progress stream existed a process dying
    mid-document left no persisted trace of which document was in flight or which providers the run
    was using. The queue worker must record run-started (with the actual provider set),
    document-started *before* processing, document-completed, and run-finished."""
    from archivetrust.application.progress import ProcessingProgressKind

    context = AppContext(deployment_root=tmp_path)
    _install_fake_docling_provider(context)
    _seed_manual_import(
        context, tmp_path, "protokoll.html",
        b"<html><body><h1>Kommunfullmaktige</h1><p>Beslut.</p></body></html>",
    )

    worker = context.queue_worker()
    worker.run()

    events = context.processing_progress.events()
    kinds = [e.kind for e in events]
    assert kinds[0] == ProcessingProgressKind.RUN_STARTED
    assert kinds[-1] == ProcessingProgressKind.RUN_FINISHED
    assert ProcessingProgressKind.DOCUMENT_STARTED in kinds
    assert ProcessingProgressKind.DOCUMENT_COMPLETED in kinds

    run_started = events[0]
    assert run_started.pending_count == 1
    assert run_started.provider_ids == tuple(sorted(context.enabled_provider_ids()))

    started = next(e for e in events if e.kind == ProcessingProgressKind.DOCUMENT_STARTED)
    assert started.original_filename == "protokoll.html"
    completed = next(e for e in events if e.kind == ProcessingProgressKind.DOCUMENT_COMPLETED)
    assert completed.archive_object_id == started.archive_object_id
    assert completed.had_failure is False

    finished = events[-1]
    assert finished.documents_done == 1
    assert finished.cancelled is False

    # Durable: the stream is reconstructible from the Workspace's own telemetry directory alone.
    from archivetrust.application.progress import FileProcessingProgressSink

    layout = context._current_layout
    reopened = FileProcessingProgressSink(layout.telemetry_dir / "processing.jsonl")
    assert [e.kind for e in reopened.events()] == kinds
