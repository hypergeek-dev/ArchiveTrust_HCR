"""Recovery verification (Production Incident Recovery, 2026-07-14) — end-to-end proof, over a
real `AppContext` and the real Trust Engine (never mocked), that a Workspace reopened after a
crash resumes correctly: completed work is skipped, the document that was in flight when the
process died is reprocessed, untouched documents continue normally, and nothing durable ends up
duplicated (archive objects, processing-progress completions, or canonical documents).

Deliberately does **not** touch the real preserved 2026-07-13/2026-07-14 incident workspaces under
`archivetrust_data/` (`incident-2026-07-13-stalled-run`, `incident-2026-07-14-benchmark-stall`
memory records) — those are forensic evidence, not a test fixture. Instead this reconstructs the
same *shape* of interruption (a durable `DocumentProcessingStarted` with no matching
`DocumentProcessingCompleted`, on an otherwise-normal multi-document run) synthetically, against a
disposable `tmp_path` deployment root, exactly as `AcquisitionManager`'s own resume test does one
layer down (`tests/acquisition/test_manager.py`).
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from archivetrust.composition import AppContext  # noqa: E402


def _seed_manual_import(context: AppContext, tmp_path, filename: str, content: bytes):
    incoming = tmp_path / "incoming"
    incoming.mkdir(exist_ok=True)
    path = incoming / filename
    path.write_bytes(content)
    from archivetrust.acquisition.manual_import import ManualImportSource

    manual = ManualImportSource()
    context.acquisition_manager.add_source(manual)
    return context.acquisition_manager.import_files(manual, (path,))[0]


def _install_fake_docling_provider(context: AppContext) -> None:
    """Registers a minimal, real `ProviderAdapter` under `provider_id="docling"` so a real
    `processing_service().process(...)` call actually produces Evidence/Observations (docs/htr-
    migration-plan.md Stage 5 -- the real Docling adapter this recovery-verification test used to
    exercise was deleted along with the rest of the OCR-era provider architecture; this test's own
    point -- recovery bookkeeping across a simulated crash -- is orthogonal to which recognizer ran,
    so a minimal fake adapter preserves the same test intent). Must be called again on every fresh
    `AppContext` (registration is in-memory, not persisted across the "crash" reopen this test
    simulates)."""
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
                raw_output=json.dumps({"text": document_ref}),
                processing_stage=ProcessingStage.RAW,
            )
            observation = Observation.from_evidence(
                provider_id=self.provider_id,
                provider_version="1.0",
                payload=HeadingPayload(text=document_ref, level=1),
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


def test_reopened_workspace_resumes_without_losing_or_duplicating_completed_work(tmp_path) -> None:
    from archivetrust.application.progress import (
        ProcessingProgressEvent,
        ProcessingProgressKind,
        utc_now_iso,
    )

    deployment_root = tmp_path / "deployment"
    context1 = AppContext(deployment_root=deployment_root)
    workspace = context1.create_workspace("Recovery Verification")
    _install_fake_docling_provider(context1)

    doc_a = _seed_manual_import(context1, tmp_path, "a.html", b"<html><body><p>Document A.</p></body></html>")
    doc_b = _seed_manual_import(context1, tmp_path, "b.html", b"<html><body><p>Document B.</p></body></html>")
    doc_c = _seed_manual_import(context1, tmp_path, "c.html", b"<html><body><p>Document C.</p></body></html>")
    assert len(context1.acquisition_manager.pending) == 3

    run_id = "recovery_verification_run"
    context1.processing_progress.record(
        ProcessingProgressEvent(
            kind=ProcessingProgressKind.RUN_STARTED, run_id=run_id, recorded_at=utc_now_iso(),
            provider_ids=tuple(sorted(context1.enabled_provider_ids())), pending_count=3,
        )
    )

    # Document A: processed for real, through the unchanged Trust Engine, and its completion is
    # durably recorded -- exactly what 478 documents in the 2026-07-14 incident actually did.
    context1.processing_progress.record(
        ProcessingProgressEvent(
            kind=ProcessingProgressKind.DOCUMENT_STARTED, run_id=run_id, recorded_at=utc_now_iso(),
            archive_object_id=doc_a.id, original_filename=doc_a.original_filename,
        )
    )
    context1.processing_service().process(doc_a)
    context1.processing_progress.record(
        ProcessingProgressEvent(
            kind=ProcessingProgressKind.DOCUMENT_COMPLETED, run_id=run_id, recorded_at=utc_now_iso(),
            archive_object_id=doc_a.id, original_filename=doc_a.original_filename,
            had_failure=False, documents_done=1,
        )
    )

    # Document B: started, then the process dies -- no completion, no run-finished. This is the
    # exact durable shape the 2026-07-14 incident's document #479 left behind.
    context1.processing_progress.record(
        ProcessingProgressEvent(
            kind=ProcessingProgressKind.DOCUMENT_STARTED, run_id=run_id, recorded_at=utc_now_iso(),
            archive_object_id=doc_b.id, original_filename=doc_b.original_filename,
        )
    )
    # Document C: never started at all -- documents #480-1000 in the real incident.

    # -- "crash": drop context1, reopen a fresh AppContext over the same durable deployment root --
    del context1
    context2 = AppContext(deployment_root=deployment_root)
    context2.open_workspace(workspace.id)
    _install_fake_docling_provider(context2)

    recovery = context2.processing_viewmodel().recovery_status()
    assert recovery is not None
    assert recovery.was_interrupted is True
    assert recovery.documents_completed == 1
    assert recovery.last_completed_document == "a.html"
    assert recovery.interrupted_document == "b.html"

    pending_filenames = {obj.original_filename for obj in context2.acquisition_manager.pending}
    assert pending_filenames == {"b.html", "c.html"}  # completed work is skipped, nothing is lost

    worker = context2.queue_worker()
    completed = []
    worker.document_completed.connect(lambda name, failed: completed.append((name, failed)))
    worker.run()

    assert {name for name, _failed in completed} == {"b.html", "c.html"}
    assert context2.acquisition_manager.pending == []

    # -- no duplication anywhere durable --
    from archivetrust.acquisition.events import FileAcquisitionTelemetrySink, ArchiveObjectRegistered
    from archivetrust.application.progress import FileProcessingProgressSink
    from archivetrust.infrastructure.storage.telemetry_sink import FileTelemetrySink

    layout = context2._current_layout  # noqa: SLF001 -- reading back the durable files this test wrote
    acquisition_events = FileAcquisitionTelemetrySink(layout.telemetry_dir / "acquisition.jsonl").all_events()
    registered = [e for e in acquisition_events if isinstance(e, ArchiveObjectRegistered)]
    assert len(registered) == 3  # a, b, c each registered exactly once, never re-registered on resume

    progress_events = FileProcessingProgressSink(layout.telemetry_dir / "processing.jsonl").events()
    completed_ids = [
        e.archive_object_id for e in progress_events if e.kind.value == "DocumentProcessingCompleted"
    ]
    assert len(completed_ids) == len(set(completed_ids)) == 3  # one completion per document, no dupes

    domain_events = FileTelemetrySink(layout.telemetry_dir / "events.jsonl").all_events()
    canonical_created = [e for e in domain_events if type(e).__name__ == "CanonicalDocumentCreated"]
    documents_with_canonicals = {e.document_ref for e in canonical_created}
    assert len(canonical_created) == len(documents_with_canonicals)  # never more than one per document
