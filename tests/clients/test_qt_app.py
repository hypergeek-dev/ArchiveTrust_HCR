"""Offscreen Qt construction tests for the desktop client.

Skipped entirely if PySide6 is not installed (the client is an optional `[gui]` extra). When it is
installed, these run under the Qt "offscreen" platform — no display needed — and actually construct
the v2 shell (the sole desktop client since the v1 retirement, release WS2) and its pages against
the ViewModels, verifying the thin Qt binding constructs, binds,
and forwards reviewer gestures correctly.
"""

from __future__ import annotations

import os

import pytest

# The offscreen platform must be selected before QApplication is created.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from archivetrust.composition import AppContext, build_demo_context  # noqa: E402
from archivetrust.clients.desktop_v2.app import MainWindowV2  # noqa: E402
from archivetrust.domain.confidence.models import ComparisonClassification  # noqa: E402
from archivetrust.domain.telemetry.events import HumanCorrectionApplied, ReviewOutcomeRecorded  # noqa: E402
from archivetrust.presentation.desktop_v2_pages import DesktopV2Page  # noqa: E402
from archivetrust.presentation.pages import AppPage  # noqa: E402
from archivetrust.review.sampling.intent import ReviewIntent  # noqa: E402
from tests.review._helpers import emit_document_snapshot, emit_slot, heading  # noqa: E402


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


def _review_document(context_docs) -> tuple[str, str]:
    first = context_docs[0]
    return first.document_ref, first.archive_object_ref


def _install_reviewable_fake_provider(context: AppContext) -> None:
    """Registers a minimal, real `ProviderAdapter` (docs/htr-migration-plan.md Stage 5 -- the real
    Docling adapter this helper used to install was deleted along with the rest of the OCR-era
    provider architecture) so `test_desktop_v2_end_to_end_operator_flow` can exercise a genuine
    process-queue -> WorkspaceProcessingService -> ProviderAdapter.observe() -> WorkQueuePage flow
    without depending on a deleted, provider-specific adapter. Mirrors
    `tests/providers/test_conformance.py::_PassingAdapter`'s minimal-but-real construction."""
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

    class _FakeReviewableAdapter(DeterministicProviderAdapter):
        registration = ProviderRegistration(
            provider_id="fake-reviewable",
            reproducibility=Reproducibility.DETERMINISTIC,
            deployment_locality=DeploymentLocality.IN_PROCESS,
            introspectability=Introspectability.OPEN,
        )

        def observe(self, *, document_ref: str, invocation_id: str, source: object) -> ProviderRunResult:
            evidence = Evidence.create(
                provider=self.provider_id,
                provider_version="1.0",
                raw_output=json.dumps({"text": "Municipal record"}),
                processing_stage=ProcessingStage.RAW,
            )
            observation = Observation.from_evidence(
                provider_id=self.provider_id,
                provider_version="1.0",
                payload=HeadingPayload(text="Municipal record", level=1),
                evidence=(evidence,),
            )
            return ProviderRunResult(
                attempt=ProviderAttempt(
                    provider_id=self.provider_id, provider_version="1.0", invocation_id=invocation_id
                ),
                evidence=(evidence,),
                observations=(observation,),
            )

    adapter = _FakeReviewableAdapter()
    context.provider_registry._adapters[(adapter.provider_id, DeploymentProfile.SINGLE_PASS)] = adapter  # noqa: SLF001
    context.provider_config_store.set(  # type: ignore[union-attr]
        context.provider_config_store.get(adapter.provider_id).model_copy(update={"enabled": True})
    )
    context.provider_availability[adapter.provider_id] = True



def test_desktop_launch_always_uses_v2(qapp: QApplication, monkeypatch) -> None:
    """The v1 shell is retired (release WS2): --demo and normal launches both construct the v2
    window; no environment variable resurrects v1."""
    from archivetrust.clients.desktop import main as desktop_main
    import archivetrust.clients.desktop_v2.app as desktop_v2_app

    created: list[str] = []

    class _V2Window:
        def __init__(self, *args, **kwargs) -> None:
            created.append("v2")

        def resize(self, *_args) -> None:
            pass

        def show(self) -> None:
            pass

    monkeypatch.setattr(desktop_main, "load_local_env", lambda: False)
    monkeypatch.setattr(QApplication, "exec", lambda self: 0)
    monkeypatch.setattr(desktop_v2_app, "MainWindowV2", _V2Window)
    monkeypatch.setenv("ARCHIVETRUST_DESKTOP_V1", "1")  # must have no effect anymore
    assert desktop_main.main(["--demo"]) == 0
    assert created == ["v2"]


def test_desktop_v2_shell_hosts_every_navigation_page(qapp: QApplication, tmp_path) -> None:
    """One widget per `DesktopV2Page`, no more and no fewer -- asserted against the enum rather
    than a hardcoded count, so adding a page cannot leave a rail entry with no widget behind it.
    Since docs/htr-migration-plan.md Stage 11 this includes the seven HTR research surfaces."""
    context = AppContext(deployment_root=tmp_path)
    window = MainWindowV2(context)
    assert window._stack.count() == len(DesktopV2Page)  # noqa: SLF001
    assert window._rail.count() == len(DesktopV2Page)  # noqa: SLF001
    titles = {window._rail.item(i).text() for i in range(window._rail.count())}  # noqa: SLF001
    assert titles == {page.title for page in DesktopV2Page}
    assert DesktopV2Page.WORK_QUEUE.title in titles
    assert DesktopV2Page.EVALUATION.title in titles
    assert DesktopV2Page.RESEARCH_OVERVIEW.title in titles
    assert DesktopV2Page.REVIEW_CENTER.title in titles


def test_desktop_v2_opens_on_the_research_overview(qapp: QApplication, tmp_path) -> None:
    """A Historical HTR research centre lands on the research state, not the workspace picker."""
    context = AppContext(deployment_root=tmp_path)
    window = MainWindowV2(context)
    assert window._shell.current_page.value is DesktopV2Page.RESEARCH_OVERVIEW  # noqa: SLF001
    assert window._rail.currentRow() == 0  # noqa: SLF001


def test_desktop_v2_evaluation_page_is_blinded_and_separate(qapp: QApplication, tmp_path) -> None:
    from PySide6.QtWidgets import QLabel, QPushButton

    from archivetrust.clients.desktop_v2.pages import EvaluationApprovalPage
    from archivetrust.domain.evidence.models import BoundingBox, Precision
    from archivetrust.evaluation.workflow import EvaluationAction

    context = AppContext(deployment_root=tmp_path, reviewer_ref="reviewer_a")
    context.evaluation_approval_service.create_assignment(
        archive_object_ref="archive_object_eval",
        content_hash="sha256:eval",
        page=1,
        region_geometry=BoundingBox(
            x0=10, y0=20, x1=200, y1=60, precision=Precision.PIXEL_ACCURATE
        ),
        observation_type="paragraph",
        task_type="transcription_approval",
        scope="paragraph",
        proposed_value="KommunfullmÃ¤ktige beslutar",
        annotation_method="visual transcription from source crop",
        campaign_id="campaign_1",
        sampling_stratum_id="paragraphs",
        legal_basis="internal municipal evaluation authorization",
        sampling_basis="campaign protocol section 4 stratified sample",
        reviewer_refs=("reviewer_a", "reviewer_b"),
    )

    page = EvaluationApprovalPage(context)
    widgets = page.findChildren(QLabel) + page.findChildren(QPushButton)
    visible_text = " ".join(widget.text() for widget in widgets).lower()
    assert "KommunfullmÃ¤ktige beslutar" == page._answer.text()  # noqa: SLF001
    assert {button.text() for button in page._action_buttons} == {  # noqa: SLF001
        "Accept",
        "Correct",
        "Illegible",
        "Reject segment",
        "Boundary incorrect",
        "Cannot determine",
    }
    for forbidden in (
        "provider identity",
        "canonical identity",
        "provider confidence",
        "canonical confidence",
        "agreement count",
        "previous reviewer",
        "expected result",
    ):
        assert forbidden not in visible_text

    page._submit(EvaluationAction.ACCEPT)  # noqa: SLF001
    records = context.evaluation_approval_service.annotations.all_records()
    assert len(records) == 1
    assert records[0].source == "independent_evaluation_approval"


def test_desktop_v2_workspaces_page_exposes_workspace_lifecycle(qapp: QApplication, tmp_path) -> None:
    from archivetrust.clients.desktop_v2.pages import WorkspacesPage, _WORKSPACE_ROLE

    context = AppContext(deployment_root=tmp_path)
    page = WorkspacesPage(context)
    assert page._list.count() == 1  # noqa: SLF001

    second = context.create_workspace("Second Workspace")
    page.refresh()
    assert page._list.count() == 2  # noqa: SLF001

    for index in range(page._list.count()):  # noqa: SLF001
        if page._list.item(index).data(_WORKSPACE_ROLE) == second.id:  # noqa: SLF001
            page._list.setCurrentRow(index)  # noqa: SLF001
            break
    page._on_open()  # noqa: SLF001
    assert context.current_workspace.id == second.id


def test_desktop_v2_documents_and_outputs_show_truthful_lifecycle(qapp: QApplication, tmp_path) -> None:
    from archivetrust.clients.desktop_v2.pages import DocumentsPage, OutputsPage

    context = AppContext(deployment_root=tmp_path)
    emit_slot(
        context.telemetry,  # type: ignore[arg-type]
        document_ref="doc1",
        canonical_payload=heading("A"),
        provider_payloads=(("docling", heading("A")),),
        classification=ComparisonClassification.UNCORROBORATED_SINGLE_SOURCE,
    )

    documents = DocumentsPage(context)
    assert documents._list.count() == 1  # noqa: SLF001
    assert "Review required" in documents._detail.text()  # noqa: SLF001

    outputs = OutputsPage(context)
    text = " ".join(
        label.text()
        for label in outputs.findChildren(type(documents._detail))
        if hasattr(label, "text")
    )
    assert "JSON export" in text
    assert "PAGE-XML" in text


def test_desktop_v2_outputs_exports_json_file(qapp: QApplication, tmp_path) -> None:
    import json
    from archivetrust.clients.desktop_v2.pages import OutputsPage
    from archivetrust.domain.document.canonical_document import CanonicalDocument
    from archivetrust.domain.graph.reconciled_graph import ReconciledObservationGraph
    from archivetrust.domain.telemetry.events import CanonicalDocumentCreated, stamp_recorded_at
    from archivetrust.domain.shared.ids import new_id

    context = AppContext(deployment_root=tmp_path)
    canonical = emit_slot(
        context.telemetry,  # type: ignore[arg-type]
        document_ref="doc1",
        canonical_payload=heading("A"),
        provider_payloads=(("docling", heading("A")),),
        classification=ComparisonClassification.UNCORROBORATED_SINGLE_SOURCE,
    )
    document = CanonicalDocument.assemble(
        reconciled_graph=ReconciledObservationGraph(
            reconciliation_sequence=canonical.reconciliation_sequence,
            canonical_observations=(canonical,),
        ),
        archive_object_ref="doc1",
        reassembly_trigger="test_export",
    )
    context.telemetry.append(
        stamp_recorded_at(
            CanonicalDocumentCreated(event_id=new_id("event"), document_ref="doc1", canonical_document=document)
        )
    )

    page = OutputsPage(context)
    output_path = tmp_path / "output.json"
    page.export_json_to_path(str(output_path))

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["export_schema"] == "archivetrust.canonical_document.v2"
    assert payload["documents"][0]["canonical_document"]["document_snapshot_id"] == document.document_snapshot_id
    assert "Exported JSON" in page._status.text()  # noqa: SLF001

    release_path = tmp_path / "release"
    page.export_release_to_path(str(release_path))
    release_manifest = json.loads((release_path / "release_manifest.json").read_text(encoding="utf-8"))
    assert release_manifest["documents"][0]["document_snapshot_id"] == document.document_snapshot_id
    assert (release_path / "interop" / "mets.xml").exists()
    assert "Exported archival release" in page._status.text()  # noqa: SLF001


def test_desktop_v2_administration_hosts_real_admin_surfaces(qapp: QApplication, tmp_path) -> None:
    from PySide6.QtWidgets import QTabWidget
    from archivetrust.clients.desktop_v2.pages import AdministrationPage

    context = AppContext(deployment_root=tmp_path)
    page = AdministrationPage(context)
    tabs = page.findChild(QTabWidget)
    assert tabs is not None
    labels = {tabs.tabText(index) for index in range(tabs.count())}
    assert {"System Health", "Providers", "Settings", "Policies", "Calibration"} <= labels


def test_desktop_v2_sources_imports_through_real_acquisition_path(qapp: QApplication, tmp_path) -> None:
    from archivetrust.clients.desktop_v2.pages import SourcesPage

    context = AppContext(deployment_root=tmp_path)
    page = SourcesPage(context)
    sample = tmp_path / "source.html"
    sample.write_text("<html><body><p>Source</p></body></html>", encoding="utf-8")

    page.import_paths((str(sample),))

    assert len(context.acquisition_manager.pending) == 1
    assert any(
        type(event).__name__ == "ArchiveObjectRegistered"
        for event in context.acquisition_telemetry.events_for_workspace(context.current_workspace.id)
    )


def test_desktop_v2_sources_reports_duplicate_imports_honestly(qapp: QApplication, tmp_path) -> None:
    from archivetrust.clients.desktop_v2.pages import SourcesPage
    from archivetrust.presentation.desktop_v2_viewmodels import source_workflow_summary

    context = AppContext(deployment_root=tmp_path)
    page = SourcesPage(context)
    sample = tmp_path / "duplicate.html"
    sample.write_text("<html><body><p>Same</p></body></html>", encoding="utf-8")

    page.import_paths((str(sample),))
    page.import_paths((str(sample),))
    summary = source_workflow_summary(
        manager=context.acquisition_manager,
        viewmodel=context.acquisition_manager_viewmodel(),
        acquisition_events=context.acquisition_telemetry.events_for_workspace(context.current_workspace.id),
    )

    assert summary.imported_documents == 1
    assert summary.duplicate_discoveries == 1


def test_desktop_v2_processing_starts_real_queue_worker_and_updates_status(
    qapp: QApplication, tmp_path
) -> None:
    from archivetrust.clients.desktop_v2.pages import ProcessingPage

    context = AppContext(deployment_root=tmp_path)
    sample = tmp_path / "process.html"
    sample.write_text("<html><body><h1>Protocol</h1><p>Decision.</p></body></html>", encoding="utf-8")
    context.acquisition_manager_viewmodel().import_files((str(sample),))
    page = ProcessingPage(context)

    assert page._process_button.isEnabled()  # noqa: SLF001
    page._on_process_queue()  # noqa: SLF001
    assert page._worker is not None  # noqa: SLF001
    # First conversion pays docling's full import (torch/RapidOCR, ~12 s cold) now that the
    # availability probe no longer imports it at workspace open (release WS5).
    page._worker.wait(60_000)  # noqa: SLF001

    assert context.acquisition_manager.pending == []
    assert context.processing_viewmodel().overview().documents_processed == 1
    assert "No document" in page._current_document.text()  # noqa: SLF001


def test_desktop_v2_work_queue_uses_adaptive_coordinator_budget_zero_operational_only(
    qapp: QApplication, tmp_path, monkeypatch
) -> None:
    from archivetrust.clients.desktop_v2.pages import WorkQueuePage

    monkeypatch.delenv("ARCHIVETRUST_ADAPTIVE_REVIEW", raising=False)
    context = AppContext(deployment_root=tmp_path)
    emit_slot(
        context.telemetry,  # type: ignore[arg-type]
        document_ref="doc1",
        canonical_payload=heading("A"),
        provider_payloads=(("docling", heading("A")),),
        classification=ComparisonClassification.UNCORROBORATED_SINGLE_SOURCE,
    )

    page = WorkQueuePage(context)

    assert page._items  # noqa: SLF001
    assert all(item.entry.intent is ReviewIntent.OPERATIONAL for item in page._items)  # noqa: SLF001


def test_desktop_v2_work_queue_shows_sampled_calibration_intent_without_raw_metrics(
    qapp: QApplication, tmp_path, monkeypatch
) -> None:
    from archivetrust.clients.desktop_v2.pages import WorkQueuePage

    monkeypatch.setenv("ARCHIVETRUST_ADAPTIVE_REVIEW", "1")
    monkeypatch.setenv("ARCHIVETRUST_CALIBRATION_SAMPLE_BUDGET", "1")
    context = AppContext(deployment_root=tmp_path)
    emit_slot(
        context.telemetry,  # type: ignore[arg-type]
        document_ref="doc1",
        canonical_payload=heading("A"),
        provider_payloads=(("docling", heading("A")),),
        classification=ComparisonClassification.UNCORROBORATED_SINGLE_SOURCE,
    )

    page = WorkQueuePage(context)
    calibration_items = [item for item in page._items if item.entry.intent is ReviewIntent.CALIBRATION]  # noqa: SLF001

    assert calibration_items
    assert "Calibration review" in calibration_items[0].intent_label
    assert "expected to improve confidence estimates" in calibration_items[0].explanation
    combined = " ".join([page._list.item(i).text() for i in range(page._list.count())])  # noqa: SLF001
    assert "random_sampling" not in combined
    assert "Wilson" not in combined


def test_desktop_v2_review_decision_preserves_correction_and_outcome_telemetry_and_refreshes_queue(
    qapp: QApplication, tmp_path
) -> None:
    from archivetrust.clients.desktop_v2.pages import WorkQueuePage

    context = AppContext(deployment_root=tmp_path)
    emit_slot(
        context.telemetry,  # type: ignore[arg-type]
        document_ref="doc1",
        canonical_payload=heading("A"),
        provider_payloads=(("docling", heading("A")),),
        classification=ComparisonClassification.UNCORROBORATED_SINGLE_SOURCE,
    )
    emit_document_snapshot(context.telemetry, document_ref="doc1")  # type: ignore[arg-type]
    page = WorkQueuePage(context)
    assert len(page._items) == 1  # noqa: SLF001
    page.open_item(page._items[0])  # noqa: SLF001

    before_corrections = sum(isinstance(event, HumanCorrectionApplied) for event in context.telemetry.all_events())
    before_outcomes = sum(isinstance(event, ReviewOutcomeRecorded) for event in context.telemetry.all_events())
    page._candidates.setCurrentRow(0)  # noqa: SLF001
    page._on_accept()  # noqa: SLF001

    assert sum(isinstance(event, HumanCorrectionApplied) for event in context.telemetry.all_events()) == before_corrections + 1
    assert sum(isinstance(event, ReviewOutcomeRecorded) for event in context.telemetry.all_events()) == before_outcomes + 1
    assert page._items == ()  # noqa: SLF001


def test_desktop_v2_work_queue_renders_original_document_with_highlight_scene(
    qapp: QApplication, tmp_path
) -> None:
    from archivetrust.clients.desktop_v2.pages import WorkQueuePage

    context = AppContext(deployment_root=tmp_path)
    emit_slot(
        context.telemetry,  # type: ignore[arg-type]
        document_ref="doc1",
        canonical_payload=heading("A"),
        provider_payloads=(("docling", heading("A")),),
        classification=ComparisonClassification.UNCORROBORATED_SINGLE_SOURCE,
    )
    emit_document_snapshot(context.telemetry, document_ref="doc1")  # type: ignore[arg-type]
    page = WorkQueuePage(context)
    page.open_item(page._items[0])  # noqa: SLF001

    # Without acquisition history there is no archive object to validate against; do not draw a
    # misleading precise box on a placeholder page.
    assert len(page._scene.items()) == 1  # noqa: SLF001
    assert "could not be validated" in page._document_caption.text()  # noqa: SLF001


def test_desktop_v2_sampled_decision_updates_sampling_log(qapp: QApplication, tmp_path, monkeypatch) -> None:
    from archivetrust.clients.desktop_v2.pages import WorkQueuePage

    monkeypatch.setenv("ARCHIVETRUST_ADAPTIVE_REVIEW", "1")
    monkeypatch.setenv("ARCHIVETRUST_CALIBRATION_SAMPLE_BUDGET", "1")
    context = AppContext(deployment_root=tmp_path, persistent_telemetry=False)
    emit_slot(
        context.telemetry,  # type: ignore[arg-type]
        document_ref="doc1",
        canonical_payload=heading("A"),
        provider_payloads=(("docling", heading("A")),),
        classification=ComparisonClassification.UNCORROBORATED_SINGLE_SOURCE,
    )
    emit_document_snapshot(context.telemetry, document_ref="doc1")  # type: ignore[arg-type]
    page = WorkQueuePage(context)
    sampled = next(item for item in page._items if item.entry.intent is ReviewIntent.CALIBRATION)  # noqa: SLF001
    page.open_item(sampled)
    page._candidates.setCurrentRow(0)  # noqa: SLF001
    page._on_accept()  # noqa: SLF001

    assert any(decision.correction_id is not None for decision in context.sampling_log.all_decisions())


def test_desktop_v2_end_to_end_operator_flow(qapp: QApplication, tmp_path) -> None:
    from archivetrust.clients.desktop_v2.pages import ProcessingPage, SourcesPage, WorkQueuePage

    context = AppContext(deployment_root=tmp_path)
    _install_reviewable_fake_provider(context)
    source = SourcesPage(context)
    sample = tmp_path / "end_to_end.html"
    sample.write_text("<html><body><h1>Municipal record</h1><p>Approved.</p></body></html>", encoding="utf-8")
    source.import_paths((str(sample),))

    processing = ProcessingPage(context)
    processing._on_process_queue()  # noqa: SLF001
    processing._worker.wait(10_000)  # noqa: SLF001

    queue = WorkQueuePage(context)
    assert queue._items  # noqa: SLF001
    queue.open_item(queue._items[0])  # noqa: SLF001
    queue._candidates.setCurrentRow(0)  # noqa: SLF001
    before = sum(isinstance(event, ReviewOutcomeRecorded) for event in context.telemetry.all_events())
    queue._on_accept()  # noqa: SLF001

    assert sum(isinstance(event, ReviewOutcomeRecorded) for event in context.telemetry.all_events()) == before + 1
    assert queue._items == ()  # noqa: SLF001


def _two_document_scenario(context: AppContext) -> tuple[str, str]:
    """Two documents, one uncertainty (a contested heading) each -- the smallest scenario that
    exercises cross-document queue behaviour (Phase 23: Review Queue Consistency)."""
    from archivetrust.domain.confidence.models import ComparisonClassification

    from tests.review._helpers import emit_slot, heading

    for doc_ref in ("doc_a", "doc_b"):
        emit_slot(
            context.telemetry,  # type: ignore[arg-type]
            document_ref=doc_ref,
            canonical_payload=heading("A"),
            provider_payloads=(("docling", heading("A")), ("qwen", heading("B"))),
            classification=ComparisonClassification.CONTESTED,
        )
    return "doc_a", "doc_b"


def test_settings_execution_mode_change_survives_qt_enum_marshalling(qapp: QApplication, tmp_path) -> None:
    # Regression test: PySide6 unwraps a str-subclassed Enum stored via addItem(label, data) back
    # to a plain str on retrieval; the settings page must re-coerce it, not pass the raw str
    # through to a ViewModel that expects the Enum type.
    from archivetrust.clients.desktop.pages.settings import GeneralSettingsPanel
    from archivetrust.runtime.provider_config import ExecutionMode

    context = AppContext(deployment_root=tmp_path)
    panel = GeneralSettingsPanel(context)
    panel._execution_box.setCurrentIndex(  # noqa: SLF001
        panel._execution_box.findData(ExecutionMode.GPU_ONLY)  # noqa: SLF001
    )
    provider = context.general_settings_viewmodel().available_providers()
    if provider:
        context.general_settings_viewmodel().set_primary_vision_provider(provider[0])
        panel._on_execution_changed(panel._execution_box.currentIndex())  # noqa: SLF001
        assert context.general_settings_viewmodel().execution_mode() == ExecutionMode.GPU_ONLY


def test_provider_manager_view_lists_registered_providers(qapp: QApplication, tmp_path) -> None:
    from archivetrust.clients.desktop.pages.provider_manager import ProviderManagerView

    context, _documents = build_demo_context(deployment_root=tmp_path)
    view = ProviderManagerView(context)
    assert view is not None  # constructs without error over the real provider registry


def test_workspace_wizard_creates_a_workspace_end_to_end(qapp: QApplication, tmp_path) -> None:
    from archivetrust.clients.desktop.workspace_wizard import WorkspaceWizard
    from archivetrust.presentation.workspace_wizard_viewmodel import WizardStep

    context = AppContext(deployment_root=tmp_path)
    wizard = WorkspaceWizard(context)
    wizard._name_field.setText("Legal Archive")  # noqa: SLF001
    while wizard._vm.step != WizardStep.REVIEW:  # noqa: SLF001
        wizard._on_next()  # noqa: SLF001
    wizard._on_next()  # noqa: SLF001 -- Create Workspace
    assert wizard.was_completed()
    assert context.current_workspace.name == "Legal Archive"
