"""Desktop v2 operator workflow pages."""

from __future__ import annotations

import json
import string

from PySide6.QtCore import QRectF, QThread, QTimer, Qt
from PySide6.QtGui import QBrush, QColor, QPen, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QComboBox,
    QInputDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from archivetrust.composition import AppContext
from archivetrust.clients.desktop.pages.provider_manager import ProviderManagerView
from archivetrust.clients.desktop.pages.settings import SettingsView
from archivetrust.clients.desktop.review_document_viewer import ReviewDocumentViewer
from archivetrust.clients.desktop.pdf_render import PageRenderError, render_pdf_page
from archivetrust.clients.desktop.ui import (
    TONE_ATTENTION,
    TONE_DANGER,
    TONE_NEUTRAL,
    card,
    muted,
    page_header,
    scroll_page,
    section_title,
    stat_row,
    stat_tile,
    table,
)
from archivetrust.clients.desktop.workspace_wizard import WorkspaceWizard
from archivetrust.admin.identity import AdminAction, Permission
from archivetrust.application.export.json_export import exportable_document_count, write_workspace_json
from archivetrust.application.export.release import write_export_release
from archivetrust.presentation.desktop_v2_viewmodels import (
    DocumentLifecycleSummary,
    WorkQueueItem,
    document_lifecycle_summaries,
    source_workflow_summary,
    work_queue_items,
)
from archivetrust.presentation.display_names import profile_label, provider_label, runtime_label
from archivetrust.presentation.document_lifecycle import (
    DocumentLifecycleDetail,
    document_lifecycle_details,
    processing_run_history,
)
from archivetrust.presentation.time_format import format_timestamp
from archivetrust.review.sampling.intent import ReviewIntent
from archivetrust.review.service import ReviewAction
from archivetrust.evaluation.workflow import AdjudicationTask, EvaluationAction, EvaluationTask

_ENTRY_ROLE = Qt.ItemDataRole.UserRole
_CANDIDATE_ROLE = Qt.ItemDataRole.UserRole + 1
_WORKSPACE_ROLE = Qt.ItemDataRole.UserRole + 2


class _EvaluationSourceViewer(QWidget):
    """Blinded source crop: original page plus only the assigned segment boundary."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.canvas = QGraphicsView(self.scene)
        self.canvas.setMinimumHeight(360)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

    def render_task(self, task: EvaluationTask | AdjudicationTask, archive_path) -> None:
        self.scene.clear()
        page_rect = QRectF(0, 0, 612, 792)
        scale = 1.0
        if archive_path is not None and archive_path.exists():
            try:
                image, width_points, _height_points = render_pdf_page(archive_path, task.page)
                item = QGraphicsPixmapItem(QPixmap.fromImage(image))
                self.scene.addItem(item)
                page_rect = item.boundingRect()
                scale = image.width() / width_points if width_points else 1.0
            except PageRenderError:
                self.scene.addRect(page_rect, QPen(QColor("#33404f")), QBrush(QColor("#e9ecf2")))
        else:
            self.scene.addRect(page_rect, QPen(QColor("#33404f")), QBrush(QColor("#e9ecf2")))
        box = task.region_geometry
        boundary = QRectF(
            box.x0 * scale,
            box.y0 * scale,
            (box.x1 - box.x0) * scale,
            (box.y1 - box.y0) * scale,
        )
        pen = QPen(QColor("#c84c4c"))
        pen.setWidth(3)
        self.scene.addRect(boundary, pen)
        self.canvas.fitInView(boundary if not boundary.isEmpty() else page_rect, Qt.AspectRatioMode.KeepAspectRatio)

    def show_empty(self) -> None:
        self.scene.clear()
        self.scene.addText("No evaluation chunk is assigned to this reviewer.")


class EvaluationApprovalPage(QWidget):
    """One bounded, source-only evaluation judgment at a time."""

    def __init__(self, context: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._context = context
        self._service = context.evaluation_approval_service
        self._task: EvaluationTask | AdjudicationTask | None = None
        self._adjudicating = False

        self._instructions = muted("")
        self._chunk = QLabel("")
        self._viewer = _EvaluationSourceViewer()
        self._answer = QLineEdit()
        self._answer.setPlaceholderText("Transcribe only the highlighted segment")
        self._status = muted("")

        actions = QWidget()
        action_row = QHBoxLayout(actions)
        action_row.setContentsMargins(0, 0, 0, 0)
        self._action_buttons: list[QPushButton] = []
        for label, action in (
            ("Accept", EvaluationAction.ACCEPT),
            ("Correct", EvaluationAction.CORRECT),
            ("Illegible", EvaluationAction.ILLEGIBLE),
            ("Reject segment", EvaluationAction.REJECT_SEGMENT),
            ("Boundary incorrect", EvaluationAction.BOUNDARY_INCORRECT),
            ("Cannot determine", EvaluationAction.CANNOT_DETERMINE),
        ):
            button = QPushButton(label)
            button.clicked.connect(lambda _checked=False, chosen=action: self._submit(chosen))
            action_row.addWidget(button)
            self._action_buttons.append(button)
        self._adjudicate_button = QPushButton("Record adjudication")
        self._adjudicate_button.clicked.connect(self._submit_adjudication)
        action_row.addWidget(self._adjudicate_button)
        action_row.addStretch(1)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(
            scroll_page(
                [
                    page_header(
                        "Evaluation Approval",
                        "Independent accuracy reference work. This does not change an operational document.",
                    ),
                    self._instructions,
                    self._chunk,
                    self._viewer,
                    card("Your reading", self._answer),
                    actions,
                    self._status,
                ]
            )
        )
        self.refresh()

    def refresh(self) -> None:
        if self._service is None:
            self._set_empty("Evaluation storage is unavailable.")
            return
        task = self._service.next_task(self._context.reviewer_ref)
        self._adjudicating = False
        if task is None:
            task = self._service.next_adjudication(self._context.reviewer_ref)
            self._adjudicating = task is not None
        self._task = task
        if task is None:
            self._set_empty("No evaluation or adjudication chunk is assigned.")
            return
        self._instructions.setText(task.instructions)
        self._chunk.setText(
            f"Page {task.page} Â· {task.scope.replace('_', ' ')} Â· {task.observation_type.replace('_', ' ')}"
        )
        self._viewer.render_task(task, self._context.resolve_archive_path(task.archive_object_ref))
        if isinstance(task, AdjudicationTask):
            left = task.reviewer_answers[0] or "[no transcription]"
            right = task.reviewer_answers[1] or "[no transcription]"
            self._answer.setPlaceholderText(f"Source A: {left} | Source B: {right}")
            self._answer.clear()
        else:
            self._answer.setText(task.proposed_value or "")
        for button in self._action_buttons:
            button.setVisible(not self._adjudicating)
        self._adjudicate_button.setVisible(self._adjudicating)
        self._status.setText("")

    def _set_empty(self, message: str) -> None:
        self._task = None
        self._instructions.setText(message)
        self._chunk.setText("")
        self._answer.clear()
        self._viewer.show_empty()
        for button in self._action_buttons:
            button.setEnabled(False)
        self._adjudicate_button.setVisible(False)

    def _submit(self, action: EvaluationAction) -> None:
        if not isinstance(self._task, EvaluationTask) or self._service is None:
            return
        try:
            if self._context.authenticated_session is not None:
                self._context.require_permission(Permission.EVALUATION_ANNOTATE)
            self._service.submit(
                assignment_id=self._task.assignment_id,
                reviewer_ref=self._context.reviewer_ref,
                action=action,
                submitted_value=self._answer.text().strip() or None,
            )
            self._context.audit_action(
                AdminAction.EVALUATION_ANNOTATED,
                target_ref=self._task.assignment_id,
                resulting_state_ref=action.value,
            )
        except (ValueError, PermissionError) as exc:
            self._status.setText(str(exc))
            return
        self.refresh()

    def _submit_adjudication(self) -> None:
        if not isinstance(self._task, AdjudicationTask) or self._service is None:
            return
        value = self._answer.text().strip()
        if not value:
            self._status.setText("Adjudication requires a source-supported reading.")
            return
        if self._context.authenticated_session is not None:
            self._context.require_permission(Permission.EVALUATION_ADJUDICATE)
        self._service.adjudicate(
            assignment_id=self._task.assignment_id,
            adjudicator_ref=self._context.reviewer_ref,
            submitted_value=value,
            method="visual adjudication from source crop",
        )
        self._context.audit_action(
            AdminAction.EVALUATION_ADJUDICATED,
            target_ref=self._task.assignment_id,
            resulting_state_ref="adjudicated",
        )
        self.refresh()


class WorkspacesPage(QWidget):
    def __init__(self, context: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._context = context
        self._vm = context.workspace_manager_viewmodel()

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        actions = QWidget()
        row = QHBoxLayout(actions)
        row.setContentsMargins(0, 0, 0, 0)
        self._create_button = QPushButton("Create Workspace...")
        self._create_button.setObjectName("Primary")
        self._create_button.clicked.connect(self._on_create)
        self._open_button = QPushButton("Open")
        self._open_button.clicked.connect(self._on_open)
        self._rename_button = QPushButton("Rename...")
        self._rename_button.clicked.connect(self._on_rename)
        self._clone_button = QPushButton("Clone...")
        self._clone_button.clicked.connect(self._on_clone)
        self._archive_button = QPushButton("Archive")
        self._archive_button.clicked.connect(self._on_archive)
        self._delete_button = QPushButton("Delete...")
        self._delete_button.clicked.connect(self._on_delete)
        self._refresh_button = QPushButton("Refresh")
        self._refresh_button.clicked.connect(self.refresh)
        for button in (
            self._create_button,
            self._open_button,
            self._rename_button,
            self._clone_button,
            self._archive_button,
            self._delete_button,
            self._refresh_button,
        ):
            row.addWidget(button)
        row.addStretch(1)

        self._list = QListWidget()
        self._list.itemDoubleClicked.connect(lambda _item: self._on_open())
        self._summary = muted("")

        root.addWidget(
            scroll_page(
                [
                    page_header("Workspaces", "Choose the archival project currently being operated."),
                    actions,
                    self._summary,
                    card("Workspaces", self._list),
                    muted("Sources are managed from the Sources page, not from workspace management."),
                ]
            )
        )
        self.refresh()

    def refresh(self) -> None:
        self._vm = self._context.workspace_manager_viewmodel()
        rows = self._vm.workspaces()
        current = next((row for row in rows if row.is_current), None)
        self._summary.setText(
            f"{len(rows)} workspace(s). Current: {current.name if current is not None else 'none'}."
        )
        selected = self._selected_id()
        self._list.clear()
        restore_row = 0
        for index, row in enumerate(rows):
            marker = "* " if row.is_current else "  "
            status = "" if row.status == "active" else f" [{row.status}]"
            description = f" - {row.description}" if row.description else ""
            item = QListWidgetItem(f"{marker}{row.name}{status} - {profile_label(row.processing_profile)}{description}")
            item.setData(_WORKSPACE_ROLE, row.id)
            self._list.addItem(item)
            if row.id == selected:
                restore_row = index
        if self._list.count():
            self._list.setCurrentRow(restore_row)

    def _selected_id(self) -> str | None:
        item = self._list.currentItem()
        return str(item.data(_WORKSPACE_ROLE)) if item is not None else None

    def _on_create(self) -> None:
        wizard = WorkspaceWizard(self._context, self)
        if wizard.exec():
            self.refresh()

    def _on_open(self) -> None:
        workspace_id = self._selected_id()
        if workspace_id is not None:
            self._vm.open(workspace_id)
            self.refresh()

    def _on_rename(self) -> None:
        workspace_id = self._selected_id()
        if workspace_id is None:
            return
        name, ok = QInputDialog.getText(self, "Rename Workspace", "New name:")
        if ok and name.strip():
            self._vm.rename(workspace_id, name.strip())
            self.refresh()

    def _on_clone(self) -> None:
        workspace_id = self._selected_id()
        if workspace_id is None:
            return
        name, ok = QInputDialog.getText(self, "Clone Workspace", "Name for the clone:")
        if ok and name.strip():
            self._vm.clone(workspace_id, name.strip())
            self.refresh()

    def _on_archive(self) -> None:
        workspace_id = self._selected_id()
        if workspace_id is not None:
            self._vm.archive(workspace_id)
            self.refresh()

    def _on_delete(self) -> None:
        workspace_id = self._selected_id()
        if workspace_id is None:
            return
        if self._context.account is None:
            self._vm.delete(workspace_id)
            self.refresh()
            return
        workspace = next(row for row in self._vm.workspaces() if row.id == workspace_id)
        confirmation, ok = QInputDialog.getText(
            self,
            "Permanently delete Workspace",
            f"Type the exact Workspace name to delete all of its data:\n{workspace.name}",
        )
        if not ok:
            return
        reason, ok = QInputDialog.getText(self, "Deletion reason", "Reason for permanent deletion:")
        if not ok:
            return
        try:
            self._vm.delete_confirmed(
                workspace_id,
                confirmation=confirmation,
                actor=self._context.account,
                reason=reason,
            )
        except (ValueError, PermissionError) as exc:
            QMessageBox.warning(self, "Workspace not deleted", str(exc))
            return
        self.refresh()


class DocumentsPage(QWidget):
    def __init__(self, context: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._context = context
        self._rows: tuple[DocumentLifecycleDetail, ...] = ()

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        self._list = QListWidget()
        self._list.currentRowChanged.connect(self._on_selected)
        self._overview = QLabel("Select a document.")
        self._overview.setWordWrap(True)
        self._detail = self._overview
        self._processing = QLabel("")
        self._processing.setWordWrap(True)
        self._review = QLabel("")
        self._review.setWordWrap(True)
        self._canonical = QLabel("")
        self._canonical.setWordWrap(True)
        self._technical = QLabel("")
        self._technical.setWordWrap(True)
        self._tabs = QTabWidget()
        self._tabs.addTab(self._overview, "Overview")
        self._tabs.addTab(self._processing, "Processing")
        self._tabs.addTab(self._review, "Review")
        self._tabs.addTab(self._canonical, "Canonical")
        self._tabs.addTab(self._technical, "Technical")
        self._refresh_button = QPushButton("Refresh")
        self._refresh_button.clicked.connect(self.refresh)

        body = QWidget()
        row = QHBoxLayout(body)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self._list, stretch=2)
        row.addWidget(card("Lifecycle", self._tabs), stretch=3)

        root.addWidget(
            scroll_page(
                [
                    page_header("Documents", "Originals, processing state, review state, and output readiness."),
                    self._refresh_button,
                    body,
                ]
            )
        )
        self.refresh()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().showEvent(event)
        self.refresh()

    def refresh(self) -> None:
        vm = self._context.processing_viewmodel()
        self._rows = document_lifecycle_details(
            processing=vm,
            acquisition_manager=self._context.acquisition_manager,
            telemetry_source=self._context.telemetry,
            processing_progress=self._context.processing_progress,
        )
        current = self._list.currentRow()
        self._list.clear()
        for row in self._rows:
            item = QListWidgetItem(
                f"{row.display_name} - {row.state.value} - {row.next_action}"
            )
            self._list.addItem(item)
        if self._rows:
            self._list.setCurrentRow(current if 0 <= current < len(self._rows) else 0)
        else:
            self._set_empty_detail("No documents have been imported or processed in this Workspace.")

    def _on_selected(self, row_index: int) -> None:
        if not (0 <= row_index < len(self._rows)):
            return
        row = self._rows[row_index]
        self._overview.setText(
            "\n".join(
                [
                    f"Document: {row.display_name}",
                    f"State: {row.state.value}",
                    f"Review required: {'yes' if row.state.value == 'Needs attention' else 'no'}",
                    f"Next action: {row.next_action}",
                    f"Source: {row.source_status}",
                    f"Canonical snapshot: {'available' if row.canonical_snapshot_available else 'not available'}",
                    f"Output: {row.output_status}",
                    *[f"Warning: {warning}" for warning in row.warnings],
                ]
            )
        )
        self._processing.setText(_document_processing_text(row))
        self._review.setText(_document_review_text(row))
        self._canonical.setText(_document_canonical_text(row))
        self._technical.setText(
            "\n".join(
                [
                    f"Document reference: {row.document_ref}",
                    f"Archive reference: {row.archive_object_ref or 'not linked'}",
                    f"Technical reference: {row.technical_reference}",
                ]
            )
        )

    def _set_empty_detail(self, text: str) -> None:
        self._overview.setText(text)
        self._processing.setText("")
        self._review.setText("")
        self._canonical.setText("")
        self._technical.setText("")


class OutputsPage(QWidget):
    def __init__(self, context: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._context = context
        self._body = QWidget()
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(0, 0, 0, 0)
        self._body_layout.setSpacing(16)
        self._refresh_button = QPushButton("Refresh")
        self._refresh_button.clicked.connect(self.refresh)
        self._export_json_button = QPushButton("Export JSON...")
        self._export_json_button.clicked.connect(self._on_export_json)
        self._export_release_button = QPushButton("Export archival release...")
        self._export_release_button.setObjectName("Primary")
        self._export_release_button.clicked.connect(self._on_export_release)
        self._status = muted("")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(
            scroll_page(
                [
                    page_header("Releases / Outputs", "What has been produced, and what is still missing."),
                    _button_row(
                        self._refresh_button,
                        self._export_json_button,
                        self._export_release_button,
                    ),
                    self._status,
                    self._body,
                ]
            )
        )
        self.refresh()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().showEvent(event)
        self.refresh()

    def refresh(self) -> None:
        _clear_layout(self._body_layout)
        rows = document_lifecycle_summaries(
            processing=self._context.processing_viewmodel(),
            acquisition_manager=self._context.acquisition_manager,
        )
        snapshots = [row for row in rows if row.canonical_snapshot_available]
        exportable_count = exportable_document_count(self._context.telemetry)
        self._export_json_button.setEnabled(exportable_count > 0)
        self._export_release_button.setEnabled(exportable_count > 0)
        self._body_layout.addWidget(
            stat_row(
                [
                    stat_tile(str(len(snapshots)), "Canonical snapshots"),
                    stat_tile(str(exportable_count), "JSON exportable"),
                    stat_tile(str(len(rows) - len(snapshots)), "Pending or unavailable"),
                ]
            )
        )
        self._body_layout.addWidget(
            muted(
                "Archival release export writes the same current canonical versions to JSON, "
                "PAGE-XML, ALTO, and METS with a versioned manifest and integrity records. "
                "Unresolved content is counted explicitly; PDF derivative export is not available."
            )
        )
        table_rows = [
            [
                row.display_name,
                "Available" if row.canonical_snapshot_available else "Not available",
                row.output_state,
            ]
            for row in rows
        ]
        self._body_layout.addWidget(
            card("Output status", table(["Document", "Canonical snapshot", "Export status"], table_rows))
        )

    def _on_export_json(self) -> None:
        path, _filter = QFileDialog.getSaveFileName(self, "Export canonical documents", "archivetrust-export.json", "JSON files (*.json)")
        if path:
            self.export_json_to_path(path)

    def export_json_to_path(self, path: str) -> None:
        if self._context.authenticated_session is not None:
            self._context.require_permission(Permission.EXPORT)
        if exportable_document_count(self._context.telemetry) == 0:
            self._status.setText("No canonical snapshots are available to export.")
            return
        write_workspace_json(
            path,
            workspace=self._context.current_workspace,
            telemetry_source=self._context.telemetry,
            acquisition_manager=self._context.acquisition_manager,
        )
        self._context.audit_action(
            AdminAction.EXPORT_CREATED,
            target_ref=str(path),
            target_label="workspace JSON export",
            resulting_state_ref=str(path),
        )
        self._status.setText(f"Exported JSON to {path}.")

    def _on_export_release(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose archival release directory")
        if path:
            self.export_release_to_path(path)

    def export_release_to_path(self, path: str) -> None:
        if self._context.authenticated_session is not None:
            self._context.require_permission(Permission.EXPORT)
        if exportable_document_count(self._context.telemetry) == 0:
            self._status.setText("No canonical snapshots are available to release.")
            return
        release = write_export_release(
            path,
            workspace=self._context.current_workspace,
            telemetry_source=self._context.telemetry,
            acquisition_manager=self._context.acquisition_manager,
            configuration_dir=self._context._current_layout.config_dir,  # noqa: SLF001
        )
        self._context.audit_action(
            AdminAction.EXPORT_CREATED,
            target_ref=release.manifest.release_id,
            target_label="archival release",
            resulting_state_ref=str(path),
        )
        self._status.setText(
            f"Exported archival release {release.manifest.release_id} to {path} "
            f"({release.manifest.release_status})."
        )


class AdministrationPage(QWidget):
    def __init__(self, context: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._context = context
        tabs = QTabWidget()
        tabs.addTab(self._system_health(), "System Health")
        tabs.addTab(self._diagnostics(), "Diagnostics")
        tabs.addTab(ProviderManagerView(context), "Providers")
        tabs.addTab(SettingsView(context), "Settings")
        tabs.addTab(self._policy_placeholder(), "Policies")
        tabs.addTab(self._calibration_placeholder(), "Calibration")

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 24)
        root.addWidget(
            page_header(
                "Administration",
                "Provider configuration, policy placeholders, calibration status, system health, and settings.",
            )
        )
        root.addWidget(tabs, stretch=1)

    def _system_health(self) -> QWidget:
        vm = self._context.processing_viewmodel()
        providers = self._context.provider_manager_viewmodel().entries()
        rows = [
            [
                provider_label(entry.provider_id),
                "Enabled" if entry.enabled else "Disabled",
                entry.status,
                runtime_label(entry.runtime_kind),
            ]
            for entry in providers
        ]
        return scroll_page(
            [
                stat_row(
                    [
                        stat_tile(str(len(providers)), "Registered providers"),
                        stat_tile(str(vm.documents_waiting()), "Waiting documents"),
                        stat_tile(str(vm.documents_failed()), "Failures", TONE_DANGER if vm.documents_failed() else TONE_NEUTRAL),
                    ]
                ),
                card("Provider health", table(["Provider", "State", "Health", "Runtime"], rows)),
            ]
        )

    def _diagnostics(self) -> QWidget:
        """Answers the release's diagnostic questions from real state, never guesses: why is
        processing not starting (provider availability + reasons), which provider failed, is the
        GPU/runtime available, is a recovery active, and did the stream itself take damage
        (corrupt telemetry records, isolated by the sink)."""
        vm = self._context.processing_viewmodel()
        sections: list[QWidget] = []

        health = vm.pipeline_health()
        sections.append(card("Pipeline state", muted(f"{health.state.value}: {health.reason}")))

        recovery = vm.recovery_status()
        if recovery is not None and recovery.was_interrupted:
            paused = f" Paused by the health guard: {recovery.paused_reason}." if recovery.paused_reason else ""
            in_flight = (
                f" while processing {recovery.interrupted_document}" if recovery.interrupted_document else ""
            )
            sections.append(
                card(
                    "Interrupted work",
                    muted(
                        f"The most recent run was interrupted{in_flight}.{paused} "
                        f"{recovery.documents_completed} document(s) had completed and are preserved; "
                        f"{recovery.documents_remaining} remain pending — reprocess the queue to resume."
                    ),
                )
            )
        else:
            sections.append(card("Interrupted work", muted("No interrupted run detected.")))

        gpu = vm.gpu_usage()
        if gpu is not None and gpu.reservations:
            total = f" of {gpu.total_bytes / 1e9:.1f} GB" if gpu.total_bytes is not None else ""
            gpu_text = (
                f"{gpu.reserved_bytes / 1e9:.1f} GB reserved{total} for {', '.join(gpu.reservations)}"
            )
        else:
            gpu_text = "No GPU reservation is active (deterministic providers run on CPU)."
        sections.append(card("GPU / runtime", muted(gpu_text)))

        availability_rows = [
            [
                provider_label(provider_id),
                "Available" if available else "Unavailable",
            ]
            for provider_id, available in sorted(self._context.provider_availability.items())
        ]
        if availability_rows:
            sections.append(card("Provider availability", table(["Provider", "State"], availability_rows)))

        corrupt = getattr(self._context.telemetry, "corrupt_records", ())
        sections.append(
            card(
                "Telemetry stream integrity",
                muted(
                    "No corrupt records in this Workspace's event stream."
                    if not corrupt
                    else f"{len(corrupt)} corrupt record(s) isolated (first at line "
                    f"{corrupt[0].line_number}: {corrupt[0].reason}). Unaffected history remains readable."
                ),
            )
        )
        return scroll_page(sections)

    @staticmethod
    def _policy_placeholder() -> QWidget:
        return scroll_page(
            [
                card(
                    "Policies",
                    muted(
                        "Policy administration is not implemented in F2. Current processing and review "
                        "policies remain the versioned policies wired through the Trust Engine."
                    ),
                )
            ]
        )

    @staticmethod
    def _calibration_placeholder() -> QWidget:
        return scroll_page(
            [
                card(
                    "Calibration Administration",
                    muted(
                        "Calibration sampling can be enabled for review selection, but F2 does not expose "
                        "operator-facing maturity controls or automatic acceptance."
                    ),
                )
            ]
        )


class SourcesPage(QWidget):
    def __init__(self, context: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._context = context
        self._vm = context.acquisition_manager_viewmodel()

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        actions = QWidget()
        row = QHBoxLayout(actions)
        row.setContentsMargins(0, 0, 0, 0)
        self._import_button = QPushButton("Import Files...")
        self._import_button.setObjectName("Primary")
        self._import_button.clicked.connect(self._on_import_files)
        self._add_folder_button = QPushButton("Add Watched Folder...")
        self._add_folder_button.clicked.connect(self._on_add_folder)
        self._scan_button = QPushButton("Scan Sources")
        self._scan_button.clicked.connect(self._on_scan)
        self._refresh_button = QPushButton("Refresh")
        self._refresh_button.clicked.connect(self._refresh)
        for button in (self._import_button, self._add_folder_button, self._scan_button, self._refresh_button):
            row.addWidget(button)
        row.addStretch(1)

        self._body = QWidget()
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(0, 0, 0, 0)
        self._body_layout.setSpacing(16)

        root.addWidget(
            scroll_page(
                [
                    page_header(
                        "Sources",
                        "Where documents come from, and what happened when they were imported.",
                    ),
                    actions,
                    self._body,
                ]
            )
        )
        self._refresh()

    def import_paths(self, paths: tuple[str, ...]) -> None:
        if paths:
            self._vm.import_files(paths)
            self._refresh()

    def add_folder_path(self, path: str) -> None:
        if path:
            self._vm.add_folder_watch(path)
            self._refresh()

    def _on_import_files(self) -> None:
        paths, _filter = QFileDialog.getOpenFileNames(self, "Select files to import")
        self.import_paths(tuple(paths))

    def _on_add_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select folder to watch")
        self.add_folder_path(path)

    def _on_scan(self) -> None:
        self._vm.trigger_scan()
        self._refresh()

    def _refresh(self) -> None:
        _clear_layout(self._body_layout)
        events = self._context.acquisition_telemetry.events_for_workspace(self._context.current_workspace.id)
        summary = source_workflow_summary(
            manager=self._context.acquisition_manager,
            viewmodel=self._vm,
            acquisition_events=events,
        )
        self._body_layout.addWidget(
            stat_row(
                [
                    stat_tile(str(summary.configured_sources), "Configured sources"),
                    stat_tile(str(summary.pending_documents), "Waiting for processing"),
                    stat_tile(str(summary.imported_documents), "Archived originals"),
                    stat_tile(
                        str(summary.duplicate_discoveries),
                        "Duplicates found",
                        TONE_ATTENTION if summary.duplicate_discoveries else TONE_NEUTRAL,
                    ),
                    stat_tile(
                        str(summary.failures),
                        "Source failures",
                        TONE_DANGER if summary.failures else TONE_NEUTRAL,
                    ),
                ]
            )
        )
        if summary.source_configuration_note:
            self._body_layout.addWidget(muted(summary.source_configuration_note))

        source_rows = [
            [
                row.label,
                "Included" if row.enabled else "Paused",
                row.status,
                format_timestamp(row.last_scan_at),
                str(row.documents_imported),
                row.last_error or "",
            ]
            for row in self._vm.configured_sources()
        ]
        self._body_layout.addWidget(
            card(
                "Configured sources",
                table(["Source", "State", "Health", "Last checked", "Documents", "Issue"], source_rows),
            )
        )
        activity_rows = [[row.kind, row.detail, row.outcome] for row in summary.recent_activity]
        self._body_layout.addWidget(
            card("Recent acquisition activity", table(["Event", "Document/source", "Outcome"], activity_rows))
        )


class ProcessingPage(QWidget):
    def __init__(self, context: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._context = context
        self._worker = None
        self._worker_timer = QTimer(self)
        self._worker_timer.setInterval(500)
        self._worker_timer.timeout.connect(self._poll_process_worker)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        controls = QWidget()
        row = QHBoxLayout(controls)
        row.setContentsMargins(0, 0, 0, 0)
        self._process_button = QPushButton("Process Queue")
        self._process_button.setObjectName("Primary")
        self._process_button.clicked.connect(self._on_process_queue)
        self._pause_button = QPushButton("Pause")
        self._pause_button.clicked.connect(self._on_pause)
        self._resume_button = QPushButton("Resume")
        self._resume_button.clicked.connect(self._on_resume)
        self._cancel_button = QPushButton("Cancel")
        self._cancel_button.clicked.connect(self._on_cancel)
        self._refresh_button = QPushButton("Refresh")
        self._refresh_button.clicked.connect(self._refresh)
        for button in (self._process_button, self._pause_button, self._resume_button, self._cancel_button, self._refresh_button):
            row.addWidget(button)
        row.addStretch(1)

        self._current_document = muted("No document is processing.")
        self._progress = QProgressBar()
        self._progress.setValue(0)
        self._body = QWidget()
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(0, 0, 0, 0)
        self._body_layout.setSpacing(16)

        root.addWidget(
            scroll_page(
                [
                    page_header("Processing", "What is happening to my documents?"),
                    controls,
                    self._current_document,
                    self._progress,
                    self._body,
                ]
            )
        )
        self._refresh()

    def _refresh(self) -> None:
        vm = self._context.processing_viewmodel()
        overview = vm.overview()
        health = vm.pipeline_health(is_running=self._worker is not None)
        throughput = vm.throughput()
        self._process_button.setEnabled(self._worker is None and len(self._context.acquisition_manager.pending) > 0)
        self._pause_button.setEnabled(self._worker is not None)
        self._resume_button.setEnabled(self._worker is not None)
        self._cancel_button.setEnabled(self._worker is not None)

        _clear_layout(self._body_layout)
        self._body_layout.addWidget(
            stat_row(
                [
                    stat_tile(health.state.value.title(), "Status"),
                    stat_tile(str(vm.documents_waiting()), "Waiting"),
                    stat_tile(str(overview.documents_processed), "Complete"),
                    stat_tile(str(overview.documents_awaiting_review), "Needs attention"),
                    stat_tile(
                        str(vm.documents_failed()),
                        "Failed",
                        TONE_DANGER if vm.documents_failed() else TONE_NEUTRAL,
                    ),
                ]
            )
        )
        eta = "Not enough data yet"
        if throughput.estimated_seconds_remaining is not None:
            eta = f"{throughput.estimated_seconds_remaining:.0f}s"
        self._body_layout.addWidget(muted(f"{health.reason} ETA: {eta}."))

        rows = [
            [
                item.display_name,
                item.processing_state,
                "Yes" if item.original_archived else "No",
                "Yes" if item.canonical_snapshot_available else "No",
                item.review_state,
                item.output_state,
            ]
            for item in document_lifecycle_summaries(
                processing=vm,
                acquisition_manager=self._context.acquisition_manager,
                telemetry_source=self._context.telemetry,
            )
        ]
        self._body_layout.addWidget(
            card(
                "Document status",
                table(["Document", "State", "Original archived", "Canonical", "Review", "Output"], rows),
            )
        )

        details = [
            [provider_label(h.provider_id), str(h.invocation_count), str(h.failed_invocations), str(h.no_observation_invocations)]
            for h in (vm.provider_execution_live() if self._worker is not None else vm.provider_execution())
        ]
        self._body_layout.addWidget(
            card("Details on demand", table(["Reading engine", "Runs", "Failed", "No content"], details))
        )

        run_rows = [
            [
                run.started_at,
                run.state,
                ", ".join(run.providers) or "-",
                f"{run.documents_completed} / {run.documents_total}",
                str(run.documents_failed),
                run.failure_reason or "",
            ]
            for run in processing_run_history(self._context.processing_progress)
        ]
        self._body_layout.addWidget(
            card("Run history", table(["Started", "State", "Reading engines", "Completed", "Failed", "Issue"], run_rows))
        )

    def _on_process_queue(self) -> None:
        if self._worker is not None or not self._context.acquisition_manager.pending:
            return
        if self._context.authenticated_session is not None:
            self._context.require_permission(Permission.PROCESSING_CONTROL)
        if self._context.use_process_worker:
            self._worker = self._context.local_worker_client().start_process_queue()
            self._context.audit_action(
                AdminAction.PROCESSING_STARTED,
                target_ref=self._context.current_workspace.id,
                target_label=self._context.current_workspace.name,
                resulting_state_ref="worker-command-queued",
            )
            self._current_document.setText("Processing continues in the local worker service.")
            self._worker_timer.start()
            self._refresh()
            return
        worker = self._context.queue_worker()
        worker.document_started.connect(self._on_document_started)
        worker.document_completed.connect(self._on_document_completed)
        worker.queue_progress.connect(self._on_queue_progress)
        worker.run_finished.connect(self._on_run_finished)
        worker.provider_health_alert.connect(self._on_health_alert)
        self._worker = worker
        self._refresh()
        worker.start(QThread.LowPriority)

    def _on_pause(self) -> None:
        if self._worker is not None:
            self._worker.pause()

    def _on_resume(self) -> None:
        if self._worker is not None:
            self._worker.resume()

    def _on_cancel(self) -> None:
        if self._worker is not None:
            if self._context.authenticated_session is not None:
                self._context.require_permission(Permission.PROCESSING_CONTROL)
            self._worker.cancel()
            self._context.audit_action(
                AdminAction.PROCESSING_CANCELLED,
                target_ref=self._context.current_workspace.id,
                target_label=self._context.current_workspace.name,
                resulting_state_ref="cancellation-requested",
            )

    def _on_document_started(self, filename: str) -> None:
        self._current_document.setText(f"Processing: {filename}")

    def _on_document_completed(self, _filename: str, _had_failure: bool) -> None:
        self._refresh()

    def _on_queue_progress(self, done: int, total: int) -> None:
        self._progress.setMaximum(max(total, 1))
        self._progress.setValue(done)
        self._progress.setFormat(f"{done} / {total} documents")

    def _on_health_alert(self, reason: str) -> None:
        self._current_document.setText(reason)

    def _on_run_finished(self) -> None:
        self._worker = None
        self._current_document.setText("No document is processing.")
        self._refresh()

    def _poll_process_worker(self) -> None:
        if self._worker is None or not self._context.use_process_worker:
            self._worker_timer.stop()
            return
        result = self._worker.result()
        if result is None or result.status.value in ("queued", "running"):
            return
        workspace_id = self._context.current_workspace.id
        self._context.open_workspace(workspace_id)
        self._worker = None
        self._worker_timer.stop()
        if result.status.value == "failed":
            self._current_document.setText(f"Worker failed: {result.error}")
        else:
            self._current_document.setText(
                f"Worker completed {result.documents_done} / {result.documents_total} documents."
            )
        self._refresh()


class WorkQueuePage(QWidget):
    def __init__(self, context: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._context = context
        self._items: tuple[WorkQueueItem, ...] = ()
        self._active: WorkQueueItem | None = None
        self._packet = None

        root = QHBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 24)
        root.setSpacing(16)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(section_title("Work Queue"))
        left_layout.addWidget(muted("Where human attention is needed now."))
        self._lifecycle = muted("")
        self._lifecycle.setWordWrap(True)
        left_layout.addWidget(self._lifecycle)
        self._filter = QComboBox()
        self._filter.currentTextChanged.connect(lambda _text: self._populate_list())
        left_layout.addWidget(self._filter)
        self._list = QListWidget()
        self._list.itemClicked.connect(self._on_item_clicked)
        left_layout.addWidget(self._list, stretch=1)
        self._refresh_button = QPushButton("Refresh")
        self._refresh_button.clicked.connect(self.refresh)
        left_layout.addWidget(self._refresh_button)
        root.addWidget(left, stretch=1)

        document = QWidget()
        document_layout = QVBoxLayout(document)
        document_layout.setContentsMargins(0, 0, 0, 0)
        document_layout.addWidget(section_title("Original document"))
        self._document_viewer = ReviewDocumentViewer(
            initial_caption="Select a work item to see the original page and highlighted evidence."
        )
        self._document_caption = self._document_viewer.caption
        self._scene = self._document_viewer.scene
        self._canvas = self._document_viewer.canvas
        document_layout.addWidget(self._document_viewer, stretch=1)
        root.addWidget(document, stretch=2)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self._why = muted("Select a work item.")
        self._current = QLabel("")
        self._current.setWordWrap(True)
        self._candidates = QListWidget()
        self._evidence_toggle = QPushButton("Show evidence details")
        self._evidence_toggle.clicked.connect(self._toggle_evidence)
        self._evidence = QListWidget()
        self._evidence.hide()
        self._accept = QPushButton("Accept selected")
        self._accept.setObjectName("Primary")
        self._accept.clicked.connect(self._on_accept)
        edit_row = QHBoxLayout()
        self._edit = QLineEdit()
        self._edit.setPlaceholderText("...or type a corrected reading")
        self._correct = QPushButton("Correct")
        self._correct.clicked.connect(self._on_correct)
        edit_row.addWidget(self._edit, stretch=1)
        edit_row.addWidget(self._correct)
        decision_row = QHBoxLayout()
        self._reject = QPushButton("Reject")
        self._reject.clicked.connect(lambda: self._submit(ReviewAction.REJECT))
        self._illegible = QPushButton("Illegible")
        self._illegible.clicked.connect(lambda: self._submit(ReviewAction.ILLEGIBLE))
        self._ambiguous = QPushButton("Mark ambiguous")
        self._ambiguous.clicked.connect(lambda: self._submit(ReviewAction.MARK_AMBIGUOUS))
        self._further = QPushButton("Request further review")
        self._further.clicked.connect(lambda: self._submit(ReviewAction.REQUEST_FURTHER_REVIEW))
        self._skip = QPushButton("Skip")
        self._skip.clicked.connect(lambda: self._submit(ReviewAction.SKIP))
        for button in (
            self._reject,
            self._illegible,
            self._ambiguous,
            self._further,
            self._skip,
        ):
            decision_row.addWidget(button)

        for widget in (
            self._why,
            section_title("Current result"),
            self._current,
            section_title("Available alternatives"),
            self._candidates,
            self._evidence_toggle,
            self._evidence,
            self._accept,
        ):
            right_layout.addWidget(widget)
        right_layout.addLayout(edit_row)
        right_layout.addLayout(decision_row)
        right_layout.addStretch(1)
        root.addWidget(right, stretch=2)

        self.refresh()
        self._set_review_enabled(False)

    def refresh(self) -> None:
        self._items = work_queue_items(
            coordinator=self._context.adaptive_review_coordinator,
            processing=self._context.processing_viewmodel(),
            acquisition_manager=self._context.acquisition_manager,
        )
        self._refresh_lifecycle()
        self._refresh_filter()
        self._populate_list()

    def _refresh_lifecycle(self) -> None:
        """Packet-lifecycle summary derived from the durable stream (F4): dispatched/open/closed
        counts, plus how many recorded decisions predate packet tracking entirely — labeled as
        legacy history, never counted as closures and never rewritten.
        """
        from datetime import timedelta

        from archivetrust.review.closure import review_closure_status

        events = tuple(self._context.telemetry.all_events())
        status = review_closure_status(events, horizon=timedelta(hours=24))
        self._open_slot_pairs = {
            (p.semantic_slot_id, p.canonical_observation_id) for p in status.open_packets
        }
        parts = [
            f"Packets: {max(0, len(self._items) - status.open_count)} pending, "
            f"{status.opened_count} claimed/in progress, {status.closed_count} closed."
        ]
        terminal_counts: dict[str, int] = {}
        for packet in status.closed_packets:
            label = packet.closure_kind.value.replace("_", " ")
            terminal_counts[label] = terminal_counts.get(label, 0) + 1
        if terminal_counts:
            parts.append(
                "History: "
                + ", ".join(
                    f"{label} {count}" for label, count in sorted(terminal_counts.items())
                )
                + "."
            )
        if status.aged_open_count:
            parts.append(f"{status.aged_open_count} open past the 24h horizon.")
        if status.legacy_incomplete_count:
            parts.append(
                f"{status.legacy_incomplete_count} historical decision(s) predate packet tracking "
                "(legacy incomplete, not rewritten)."
            )
        self._lifecycle.setText(" ".join(parts))

    def _refresh_filter(self) -> None:
        current = self._filter.currentText()
        labels = ["All"] + sorted({item.intent_label for item in self._items})
        self._filter.blockSignals(True)
        self._filter.clear()
        self._filter.addItems(labels)
        if current in labels:
            self._filter.setCurrentText(current)
        self._filter.blockSignals(False)

    def _populate_list(self) -> None:
        selected = self._filter.currentText() or "All"
        self._list.clear()
        for index, item in enumerate(self._items):
            if selected != "All" and item.intent_label != selected:
                continue
            impact = f" - {item.impact_label}" if item.impact_label else ""
            slot_pair = (item.entry.semantic_slot_id, item.entry.canonical_observation_id)
            in_progress = " [in progress]" if slot_pair in getattr(self, "_open_slot_pairs", set()) else ""
            row = QListWidgetItem(
                f"{item.intent_label}: {item.document_label} - {item.observation_type}{impact}{in_progress}"
            )
            row.setData(_ENTRY_ROLE, index)
            self._list.addItem(row)

    def _on_item_clicked(self, row: QListWidgetItem) -> None:
        self.open_item(self._items[row.data(_ENTRY_ROLE)])

    def open_item(self, item: WorkQueueItem) -> None:
        self._active = item
        self._packet = self._context.adaptive_review_coordinator.open_packet(
            item.entry,
            archive_object_ref=item.archive_object_ref,
        )
        self._why.setText(f"{item.intent_label}. {item.explanation}")
        self._current.setText(self._packet.current_value or "(no text value)")
        self._edit.clear()
        self._candidates.clear()
        self._evidence.clear()
        for idx, candidate in enumerate(self._packet.candidates):
            label = f"{_source_label(idx)}: {candidate.value or '(structural)'}"
            candidate_row = QListWidgetItem(label)
            candidate_row.setData(_CANDIDATE_ROLE, (candidate.observation_id, candidate.value))
            self._candidates.addItem(candidate_row)
            for evidence in candidate.evidence:
                self._evidence.addItem(f"{_source_label(idx)}: {_human_evidence_text(evidence.raw_output)}")
        self._draw_document()
        self._append_geometry_diagnostics()
        self._set_review_enabled(True)

    def _set_review_enabled(self, enabled: bool) -> None:
        for widget in (
            self._candidates,
            self._evidence_toggle,
            self._accept,
            self._edit,
            self._correct,
            self._reject,
            self._illegible,
            self._ambiguous,
            self._further,
            self._skip,
        ):
            widget.setEnabled(enabled)

    def _toggle_evidence(self) -> None:
        self._evidence.setVisible(not self._evidence.isVisible())
        self._evidence_toggle.setText("Hide evidence details" if self._evidence.isVisible() else "Show evidence details")

    def _on_accept(self) -> None:
        if self._packet is None:
            return
        selected = self._candidates.currentItem()
        if selected is None:
            return
        _observation_id, value = selected.data(_CANDIDATE_ROLE)
        if value is None or value == self._packet.current_value:
            self._submit(ReviewAction.ACCEPT_PROVIDER)
        else:
            self._submit(ReviewAction.MANUAL_EDIT, corrected_output=value)

    def _on_correct(self) -> None:
        value = self._edit.text().strip()
        if value:
            self._submit(ReviewAction.MANUAL_EDIT, corrected_output=value)

    def _submit(self, action: ReviewAction, *, corrected_output: str | None = None) -> None:
        if self._active is None or self._packet is None:
            return
        if self._context.authenticated_session is not None:
            self._context.require_permission(Permission.REVIEW)
        kwargs = {"entry": self._active.entry, "packet": self._packet, "action": action}
        if corrected_output is not None:
            kwargs["corrected_output"] = corrected_output
        entry = self._active.entry
        self._context.adaptive_review_coordinator.submit_decision(**kwargs)
        # Verify closure from the durable stream itself, never from an in-memory assumption (F4):
        # the operator sees either the recorded closure or an honest "still open".
        closure = self._context.adaptive_review_coordinator.closure_for(entry)
        self._context.audit_action(
            AdminAction.REVIEW_DECIDED,
            target_ref=(closure.packet_id if closure is not None else entry.semantic_slot_id),
            resulting_state_ref=action.value,
        )
        still_open = self._context.adaptive_review_coordinator.open_dispatch_for(entry)
        if closure is not None and still_open is None:
            confirmation = (
                f"Decision recorded; packet {closure.packet_id} closed "
                f"({closure.closure_kind.value}) in durable telemetry."
            )
        else:
            confirmation = (
                "Decision recorded, but no packet closure was found — the packet may still be "
                "open. Check the lifecycle summary."
            )
        self._active = None
        self._packet = None
        self._why.setText(confirmation + " Select another work item.")
        self._current.setText("")
        self._candidates.clear()
        self._evidence.clear()
        self._scene.clear()
        self._document_caption.setText("Select a work item to see the original page and highlighted evidence.")
        self._set_review_enabled(False)
        self.refresh()

    def _first_page_geometry(self):
        return ReviewDocumentViewer._first_page_geometry(self._packet) if self._packet is not None else None

    def _draw_document(self) -> None:
        if self._packet is None:
            self._document_viewer.show_empty_state()
            return
        archive_path = self._context.resolve_archive_path(self._packet.archive_object_ref)
        self._document_viewer.render_packet(self._packet, archive_path=archive_path, source_label=_source_label)

    def _append_geometry_diagnostics(self) -> None:
        results = getattr(self._document_viewer, "validation_results", {})
        for evidence_id, result in results.items():
            self._evidence.addItem(
                f"Geometry: {result.status.value} ({result.reason_code}) - {result.explanation}"
            )


class EvidencePage(QWidget):
    """The full trust chain, navigable (release WS2, ported from the v1 Evidence Explorer):

        Canonical result → contributing observations → evidence → provider invocation

    Built on the framework-independent `EvidenceExplorerViewModel` (the same replayed-telemetry
    reconstruction the v1 page used — nothing here recomputes a provider). Absences are labeled,
    never blank: a provider that failed, a provider that ran and observed nothing, rejected
    evidence, missing confidence, and unaligned observations are each named as what they are.
    """

    def __init__(self, context: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._context = context
        self._vm = None

        root = QHBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 24)
        root.setSpacing(16)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(section_title("Evidence"))
        left_layout.addWidget(muted("Every canonical claim, traced to the evidence that produced it."))
        self._documents = QListWidget()
        self._documents.itemClicked.connect(self._on_document_selected)
        left_layout.addWidget(self._documents, stretch=1)
        self._refresh_button = QPushButton("Refresh")
        self._refresh_button.clicked.connect(self.refresh)
        left_layout.addWidget(self._refresh_button)
        root.addWidget(left, stretch=1)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self._summary = muted("Select a document to trace its trust chain.")
        self._summary.setWordWrap(True)
        right_layout.addWidget(self._summary)
        from PySide6.QtWidgets import QTreeWidget

        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Trust chain", "Detail"])
        self._tree.setColumnWidth(0, 420)
        right_layout.addWidget(self._tree, stretch=1)
        root.addWidget(right, stretch=3)

        self.refresh()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().showEvent(event)
        self.refresh()

    def refresh(self) -> None:
        from archivetrust.presentation.evidence_explorer_viewmodel import EvidenceExplorerViewModel

        self._vm = EvidenceExplorerViewModel(
            self._context.telemetry,
            current_state=self._context.read_model.current_state,
        )
        selected = self._selected_ref()
        self._documents.clear()
        for ref in self._vm.documents():
            row = QListWidgetItem(self._document_label(ref))
            row.setData(_ENTRY_ROLE, ref)
            self._documents.addItem(row)
            if ref == selected:
                self._documents.setCurrentItem(row)

    def _document_label(self, document_ref: str) -> str:
        manager = self._context.acquisition_manager
        archive_object = manager.archive_object_by_ref(document_ref) if manager is not None else None
        return archive_object.original_filename if archive_object is not None else document_ref

    def _selected_ref(self) -> str | None:
        item = self._documents.currentItem()
        return item.data(_ENTRY_ROLE) if item is not None else None

    def _on_document_selected(self, item: QListWidgetItem) -> None:
        self.show_document(item.data(_ENTRY_ROLE))

    def show_document(self, document_ref: str) -> None:
        from PySide6.QtWidgets import QTreeWidgetItem

        trace = self._vm.document_trace(document_ref)
        self._summary.setText(
            f"{self._document_label(document_ref)} — {trace.evidence_count} evidence records, "
            f"{trace.observation_count} observations ({trace.unaligned_count} unaligned), "
            f"{len(trace.canonical_traces)} canonical results, "
            f"{len(trace.excluded_pairs)} structurally excluded candidate pairs."
        )
        self._tree.clear()

        invocations = QTreeWidgetItem(["Provider invocations", ""])
        for attempt in self._provider_attempts(document_ref):
            QTreeWidgetItem(invocations, attempt)
        self._tree.addTopLevelItem(invocations)

        for canonical in trace.canonical_traces:
            headline = f"{canonical.observation_type}: {canonical.value or '(no text value)'}"
            detail_parts = [self._classification_label(canonical.classification)]
            detail_parts.append(
                f"confidence {canonical.canonical_confidence:.2f}"
                if canonical.canonical_confidence is not None
                else "no confidence recorded"
            )
            if canonical.was_corrected:
                detail_parts.append("human-corrected")
            node = QTreeWidgetItem([headline, "; ".join(detail_parts)])
            if canonical.reconciliation_basis_code:
                QTreeWidgetItem(node, ["Decision basis", canonical.reconciliation_basis_code])
            if canonical.alignment_rationale:
                QTreeWidgetItem(node, ["Alignment rationale", canonical.alignment_rationale])
            for contributor in canonical.contributors:
                contributor_node = QTreeWidgetItem(
                    node,
                    [
                        f"Observation by {provider_label(contributor.provider_id)}",
                        f"{contributor.value or '(structural)'} — {contributor.alignment_outcome}",
                    ],
                )
                if contributor.mapping_table_entry_id:
                    QTreeWidgetItem(
                        contributor_node, ["Mapping entry", contributor.mapping_table_entry_id]
                    )
                for evidence in contributor.evidence:
                    confidence = (
                        f"provider confidence {evidence.provider_confidence:.2f}"
                        if evidence.provider_confidence is not None
                        else "no provider confidence reported"
                    )
                    QTreeWidgetItem(
                        contributor_node,
                        [
                            f"Evidence {evidence.evidence_id[:24]}…",
                            f"{confidence} — {_human_evidence_text(evidence.raw_output)[:200]}",
                        ],
                    )
            self._tree.addTopLevelItem(node)

        if trace.excluded_pairs:
            excluded = QTreeWidgetItem(
                ["Structurally excluded candidate pairs", f"{len(trace.excluded_pairs)} pairs"]
            )
            for pair in trace.excluded_pairs[:200]:
                QTreeWidgetItem(
                    excluded,
                    [
                        f"{pair.candidate_observation_id[:20]}… vs {pair.compared_against_observation_id[:20]}…",
                        f"{pair.excluding_mechanism} ({pair.basis_code})",
                    ],
                )
            if len(trace.excluded_pairs) > 200:
                QTreeWidgetItem(excluded, ["…", f"{len(trace.excluded_pairs) - 200} more pairs not shown"])
            self._tree.addTopLevelItem(excluded)

        if trace.examining_findings:
            findings = QTreeWidgetItem(["Research findings examining this Workspace", ""])
            for title in trace.examining_findings:
                QTreeWidgetItem(findings, [title, ""])
            self._tree.addTopLevelItem(findings)

    def _provider_attempts(self, document_ref: str) -> list[list[str]]:
        from archivetrust.domain.telemetry.events import (
            EvidenceRejected,
            ProviderObservationAttempted,
            ProviderInvocationOutcome,
        )

        rows: list[list[str]] = []
        for event in self._context.telemetry.events_for_document(document_ref):
            if isinstance(event, ProviderObservationAttempted):
                if event.outcome is ProviderInvocationOutcome.FAILED:
                    category = event.failure_category.value if event.failure_category else "uncategorized"
                    detail = f"FAILED ({category}): {event.failure_reason or 'no reason recorded'}"
                elif event.outcome is ProviderInvocationOutcome.NO_OBSERVATIONS:
                    detail = "ran and observed nothing (not a failure)"
                elif event.outcome is None:
                    detail = "outcome not recorded (attempt predates outcome telemetry)"
                else:
                    detail = event.outcome.value
                rows.append([f"{provider_label(event.provider_id)} {event.provider_version}", detail])
            elif isinstance(event, EvidenceRejected):
                rows.append(
                    [f"{provider_label(event.provider_id)} (evidence rejected)", event.reason or ""]
                )
        if not rows:
            rows.append(["No provider invocations recorded", "this document has no attempt telemetry"])
        return rows

    @staticmethod
    def _classification_label(classification: str) -> str:
        return {
            "contested": "Disagreement between sources",
            "corroborated": "Corroborated by multiple sources",
            "uncorroborated_single_source": "Single source — uncorroborated",
        }.get(classification, classification)


def _clear_layout(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.setParent(None)


def _button_row(*buttons: QPushButton) -> QWidget:
    host = QWidget()
    row = QHBoxLayout(host)
    row.setContentsMargins(0, 0, 0, 0)
    for button in buttons:
        row.addWidget(button)
    row.addStretch(1)
    return host


def _human_evidence_text(raw_output: str) -> str:
    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError:
        return raw_output
    if not isinstance(parsed, dict):
        return raw_output
    text = parsed.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    label = parsed.get("label")
    if isinstance(label, str) and label.strip():
        return label.strip().replace("_", " ").title()
    return "Captured evidence"


def _source_label(index: int) -> str:
    return f"Source {string.ascii_uppercase[index]}" if index < 26 else f"Source {index + 1}"


def _document_processing_text(row: DocumentLifecycleDetail) -> str:
    sections = [
        "Processing runs:",
        *[
            (
                f"{run.started_at}: {run.state}"
                f" ({run.duration})"
                f"{' - ' + run.failure_reason if run.failure_reason else ''}"
            )
            for run in row.processing_runs
        ],
        "",
        "Reading engine attempts:",
        *[
            (
                f"{attempt.recorded_at}: {attempt.provider} {attempt.outcome}"
                f" ({attempt.observations if attempt.observations is not None else 'unknown'} observations)"
                f"{' - ' + attempt.failure_reason if attempt.failure_reason else ''}"
            )
            for attempt in row.provider_attempts
        ],
    ]
    return "\n".join(sections) if row.processing_runs or row.provider_attempts else "No processing runs recorded."


def _document_review_text(row: DocumentLifecycleDetail) -> str:
    if not row.review_history:
        return "No review decisions recorded."
    return "\n".join(
        f"{item.recorded_at}: {item.action} - {item.outcome}"
        f"{' (' + item.correction_ref + ')' if item.correction_ref else ''}"
        for item in row.review_history
    )


def _document_canonical_text(row: DocumentLifecycleDetail) -> str:
    if not row.canonical_history:
        return "No canonical history recorded."
    return "\n".join(f"{item.recorded_at}: {item.summary}" for item in row.canonical_history)
