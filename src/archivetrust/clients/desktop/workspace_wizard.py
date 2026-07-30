"""Workspace Wizard (ROADMAP.md §5.13.1) — the professional first-run / new-Workspace flow: Name →
Profile → Providers → Models → Input Sources → Output Location → Review → Create → Launch Processing
Center. Binds to `WorkspaceWizardViewModel`; validation errors are shown as a plain status label,
never raised past this widget (mirrors `FirstLaunchWizard`'s convention).
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from archivetrust.composition import AppContext
from archivetrust.clients.desktop.ui import STYLESHEET, muted, page_header, section_title
from archivetrust.presentation.display_names import profile_label, provider_label
from archivetrust.presentation.first_launch_viewmodel import VISION_PROVIDER_CHOICES
from archivetrust.presentation.workspace_wizard_viewmodel import (
    InvalidWizardStateError,
    WizardStep,
)
from archivetrust.runtime.provider_profiles import ProviderProfileName


class WorkspaceWizard(QDialog):
    """Returns via `exec()`; call `was_completed()` to know whether a Workspace was created."""

    def __init__(self, context: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("ArchiveTrust — New Workspace")
        self.setStyleSheet(STYLESHEET)
        self.setMinimumWidth(560)
        self._vm = context.workspace_wizard_viewmodel()
        self._completed = False

        layout = QVBoxLayout(self)
        layout.addWidget(
            page_header(
                "Create a Workspace",
                "A Workspace is a complete archival project — its own providers, models, "
                "telemetry, and storage, independent of every other Workspace.",
            )
        )

        self._stack = QStackedWidget()
        self._name_field = QLineEdit()
        self._description_field = QLineEdit()
        self._stack.addWidget(self._build_name_step())
        self._profile_box = QComboBox()
        self._stack.addWidget(self._build_profile_step())
        self._provider_checks: dict[str, QCheckBox] = {}
        self._stack.addWidget(self._build_providers_step())
        self._model_field = QLineEdit()
        self._stack.addWidget(self._build_models_step())
        self._manual_check = QCheckBox("Manual Import (drag & drop / import files)")
        self._folder_field = QLineEdit()
        self._stack.addWidget(self._build_input_sources_step())
        self._stack.addWidget(self._build_output_location_step())
        self._review_label = QLabel()
        self._review_label.setWordWrap(True)
        self._stack.addWidget(self._build_review_step())
        layout.addWidget(self._stack)

        self._status = muted("")
        layout.addWidget(self._status)

        nav = QWidget()
        nav_layout = QVBoxLayout(nav)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        button_row = QWidget()
        row_layout = QHBoxLayout(button_row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        self._back_button = QPushButton("Back")
        self._back_button.clicked.connect(self._on_back)
        self._next_button = QPushButton("Next")
        self._next_button.setObjectName("Primary")
        self._next_button.clicked.connect(self._on_next)
        row_layout.addWidget(self._back_button)
        row_layout.addStretch(1)
        row_layout.addWidget(self._next_button)
        nav_layout.addWidget(button_row)
        layout.addWidget(nav)

        self._sync_step()

    # -- step builders -----------------------------------------------------------------------

    def _build_name_step(self) -> QWidget:
        host = QWidget()
        form = QFormLayout(host)
        form.addRow("Workspace name", self._name_field)
        self._description_field.setPlaceholderText("e.g. Kommunfullmäktige protokoll 1900–1970")
        form.addRow("Description", self._description_field)
        return host

    def _build_profile_step(self) -> QWidget:
        host = QWidget()
        form = QFormLayout(host)
        for profile in self._vm.available_profiles():
            self._profile_box.addItem(profile.label, profile.name)
        form.addRow("Processing Profile", self._profile_box)
        return host

    def _build_providers_step(self) -> QWidget:
        """Every checkbox comes from `WorkspaceWizardViewModel.available_providers()` (Provider
        Discovery milestone, 2026-07-13) — enumerated from the same provider registry/catalog
        every other surface reads, grouped by `WizardProviderChoice.group`, never a hardcoded id
        list. A provider reporting `available=False` is still shown (never hidden), disabled, with
        its `unavailable_reason` as a muted line underneath — so an operator sees *why* Surya (or
        any other provider) can't be selected right now instead of wondering why it's absent.
        """
        host = QWidget()
        layout = QVBoxLayout(host)

        grouped: dict[str, list] = {}
        for choice in self._vm.available_providers():
            grouped.setdefault(choice.group, []).append(choice)

        for group in sorted(grouped):
            layout.addWidget(section_title(group))
            for choice in grouped[group]:
                check = QCheckBox(choice.label)
                check.setEnabled(choice.available)
                check.setChecked(choice.available and choice.provider_id in self._vm.enabled_provider_ids)
                if not choice.available and choice.unavailable_reason:
                    check.setToolTip(choice.unavailable_reason)
                self._provider_checks[choice.provider_id] = check
                layout.addWidget(check)
                if not choice.available and choice.unavailable_reason:
                    reason = muted(choice.unavailable_reason)
                    reason.setWordWrap(True)
                    layout.addWidget(reason)
        layout.addStretch(1)
        return host

    def _build_models_step(self) -> QWidget:
        host = QWidget()
        layout = QVBoxLayout(host)
        form_host = QWidget()
        form = QFormLayout(form_host)

        # Vision Provider selector (Multi-Provider Activation milestone) -- reads
        # VISION_PROVIDER_CHOICES so a future provider needs no wizard change.
        #
        # **The row is omitted entirely when the catalog is empty** (2026-07-30 residual cleanup),
        # which it now is: the three entries it used to hold named adapters deleted in migration
        # Stage 5, and `composition.py::_VISION_PROVIDER_ADAPTER_FACTORIES` has no factory for any
        # provider, so nothing a combo could offer is activatable. An empty combo labelled "Vision
        # Provider" would imply a choice exists and has none; no row states the truth. The loop is
        # kept rather than deleted so re-populating the catalog re-shows the row with no UI change.
        self._model_provider_box = QComboBox()
        for provider_id, label, _default_model, _runtime_kind in VISION_PROVIDER_CHOICES:
            self._model_provider_box.addItem(label, provider_id)
        if self._model_provider_box.count() > 0:
            self._model_provider_box.currentIndexChanged.connect(self._on_model_provider_changed)
            form.addRow("Vision Provider", self._model_provider_box)

        self._model_field.setPlaceholderText("Leave blank to skip")
        # Blank by default now that `DEFAULT_VISION_MODEL_ID` is `None`: pre-filling a deleted
        # provider's repository made every wizard-created Workspace persist a binding for a provider
        # that cannot exist. An operator with a real model to bind still types it here.
        self._model_field.setText(self._vm.model_identifier or "")
        self._model_field.textChanged.connect(self._on_model_field_changed)
        form.addRow("Hugging Face model", self._model_field)
        layout.addWidget(form_host)
        # Clearly communicated, not a silent gap (Multi-Provider Activation milestone, Workspace
        # Binding): skipping this step is a real, workspace-scoped choice with a consequence -- the
        # operator should see that consequence here, not discover it later as an absent Provider
        # Manager row.
        self._model_status = muted("")
        layout.addWidget(self._model_status)
        self._on_model_field_changed(self._model_field.text())
        return host

    def _on_model_provider_changed(self, index: int) -> None:
        # Bounds-guarded: only connected when the catalog is non-empty, but a catalog that shrinks
        # between construction and a signal must not raise IndexError into a wizard.
        if not 0 <= index < len(VISION_PROVIDER_CHOICES):
            return
        _provider_id, _label, default_model, _runtime_kind = VISION_PROVIDER_CHOICES[index]
        self._model_field.setText(default_model)

    def _on_model_field_changed(self, text: str) -> None:
        # The "blank" message used to name Docling and Tesseract as still-running fallbacks; both
        # adapters were deleted in migration Stage 5, so it promised a capability that no longer
        # exists (2026-07-30 residual cleanup). It now says what is true: the three HTR methods are
        # not configured here at all, and their real readiness is on the HTR Methods page.
        if text.strip():
            self._model_status.setText("A primary reading engine will be configured for this Workspace.")
        else:
            self._model_status.setText(
                "No general vision reading engine will be bound in this Workspace. The three HTR "
                "recognition methods (SATRN, Florence-2, Transkribus) are not configured here — see "
                "the HTR Methods page for each one's real environment status."
            )

    def _build_input_sources_step(self) -> QWidget:
        host = QWidget()
        layout = QVBoxLayout(host)
        self._manual_check.setChecked(True)
        layout.addWidget(self._manual_check)
        folder_row = QWidget()
        form = QFormLayout(folder_row)
        self._folder_field.setPlaceholderText("Leave blank to skip Folder Watch")
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._on_browse_folder)
        folder_line = QWidget()
        folder_line_layout = QHBoxLayout(folder_line)
        folder_line_layout.setContentsMargins(0, 0, 0, 0)
        folder_line_layout.addWidget(self._folder_field)
        folder_line_layout.addWidget(browse)
        form.addRow("Watched folder", folder_line)
        layout.addWidget(folder_row)
        return host

    def _build_output_location_step(self) -> QWidget:
        host = QWidget()
        layout = QVBoxLayout(host)
        layout.addWidget(
            muted(
                "ArchiveTrust manages each Workspace's storage layout automatically so archive "
                "objects, telemetry, and derived artifacts stay together and remain isolated from "
                "every other Workspace."
            )
        )
        return host

    def _build_review_step(self) -> QWidget:
        host = QWidget()
        layout = QVBoxLayout(host)
        layout.addWidget(self._review_label)
        return host

    # -- navigation ----------------------------------------------------------------------------

    def _sync_step(self) -> None:
        self._stack.setCurrentIndex(list(WizardStep).index(self._vm.step))
        self._back_button.setEnabled(self._vm.step != WizardStep.NAME)
        self._next_button.setText("Create Workspace" if self._vm.step == WizardStep.REVIEW else "Next")
        if self._vm.step == WizardStep.REVIEW:
            self._sync_review_step_state()
            review = self._vm.review()
            self._review_label.setText(
                f"<b>{review.name}</b><br>{review.description}<br><br>"
                f"Processing profile: {profile_label(review.processing_profile)}<br>"
                f"Reading engines: {_provider_list(review.enabled_provider_ids)}<br>"
                f"Model: {review.model_identifier or '-'}<br>"
                f"Watched folder: {review.watched_folder or '-'}<br>"
                f"Storage: {review.output_location}"
            )

    def _sync_review_step_state(self) -> None:
        self._vm.name = self._name_field.text()
        self._vm.description = self._description_field.text()
        # Qt unwraps a str-subclassed Enum stored via addItem(label, data) back to a plain str --
        # re-coerce rather than assume currentData() preserves the Python type (see
        # FirstLaunchWizard._on_confirm's identical note).
        self._vm.processing_profile = ProviderProfileName(self._profile_box.currentData())
        self._vm.enabled_provider_ids = {
            provider_id for provider_id, check in self._provider_checks.items() if check.isChecked()
        }
        self._vm.model_identifier = self._model_field.text().strip() or None
        self._vm.model_provider_id = str(self._model_provider_box.currentData())
        self._vm.add_manual_import = self._manual_check.isChecked()
        self._vm.watched_folder = self._folder_field.text().strip() or None

    def _on_browse_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select folder to watch")
        if path:
            self._folder_field.setText(path)

    def _on_back(self) -> None:
        self._sync_review_step_state()
        self._vm.back()
        self._sync_step()

    def _on_next(self) -> None:
        self._sync_review_step_state()
        if self._vm.step == WizardStep.REVIEW:
            self._on_create()
            return
        try:
            self._vm.advance()
        except InvalidWizardStateError as exc:
            self._status.setText(str(exc))
            self._status.setStyleSheet("color: #d06767;")
            return
        self._status.setText("")
        self._sync_step()

    def _on_create(self) -> None:
        try:
            self._vm.create()
        except InvalidWizardStateError as exc:
            self._status.setText(str(exc))
            self._status.setStyleSheet("color: #d06767;")
            return
        self._completed = True
        self.accept()

    def was_completed(self) -> bool:
        return self._completed


def _provider_list(provider_ids: tuple[str, ...]) -> str:
    if not provider_ids:
        return "-"
    return ", ".join(provider_label(provider_id) for provider_id in provider_ids)
