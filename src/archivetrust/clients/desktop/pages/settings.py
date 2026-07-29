"""Settings view — General (Part 8) and, per provider, Advanced (Part 9).

Deliberately layered: the simple `GeneralSettingsPanel` (Primary Vision Provider / Execution /
Processing Mode) is what most operators need. A per-provider "Advanced settings…" link opens
`AdvancedProviderSettingsPanel`, which renders only the setting groups that provider's
capabilities actually support — never a one-size-fits-all form. Both bind to their ViewModels and
hold no configuration logic themselves.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from archivetrust.composition import AppContext
from archivetrust.clients.desktop.ui import card, muted, page_header, scroll_page, section_title
from archivetrust.presentation.display_names import provider_label
from archivetrust.runtime.provider_config import ExecutionMode, ProcessingMode


class GeneralSettingsPanel(QWidget):
    def __init__(self, context: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._vm = context.general_settings_viewmodel()

        form_host = QWidget()
        form = QFormLayout(form_host)

        self._provider_box = QComboBox()
        for provider_id in self._vm.available_providers():
            self._provider_box.addItem(provider_label(provider_id), provider_id)
        current = self._vm.primary_vision_provider()
        if current is not None:
            index = self._provider_box.findData(current)
            if index >= 0:
                self._provider_box.setCurrentIndex(index)
        self._provider_box.currentIndexChanged.connect(self._on_provider_changed)
        form.addRow("Primary reading engine", self._provider_box)

        self._execution_box = QComboBox()
        for mode in ExecutionMode:
            self._execution_box.addItem(mode.value.replace("_", " ").title(), mode)
        self._select(self._execution_box, self._vm.execution_mode())
        self._execution_box.currentIndexChanged.connect(self._on_execution_changed)
        form.addRow("Execution", self._execution_box)

        self._processing_box = QComboBox()
        for mode in ProcessingMode:
            self._processing_box.addItem(mode.value.replace("_", " ").title(), mode)
        self._select(self._processing_box, self._vm.processing_mode())
        self._processing_box.currentIndexChanged.connect(self._on_processing_changed)
        form.addRow("Processing Mode", self._processing_box)

        general_card = card("General", form_host)
        note = muted(
            "This is the simple settings surface: three controls. Per-provider tuning "
            "lives in Provider Manager advanced settings, and only shows what that "
            "provider actually supports."
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(general_card)
        layout.addWidget(note)

    @staticmethod
    def _select(box: QComboBox, value) -> None:
        index = box.findData(value)
        if index >= 0:
            box.setCurrentIndex(index)

    def _on_provider_changed(self, index: int) -> None:
        provider_id = self._provider_box.itemData(index)
        if provider_id:
            self._vm.set_primary_vision_provider(provider_id)

    def _on_execution_changed(self, index: int) -> None:
        # Qt's QVariant marshalling unwraps a `str`-subclassed Enum back to a plain `str` -- see
        # first_launch_wizard.py's identical fix, found by an actual offscreen-launched run.
        self._vm.set_execution_mode(ExecutionMode(self._execution_box.itemData(index)))

    def _on_processing_changed(self, index: int) -> None:
        self._vm.set_processing_mode(ProcessingMode(self._processing_box.itemData(index)))


class AdvancedProviderSettingsPanel(QWidget):
    """One provider's full settings surface (Part 9), gated by its capabilities."""

    def __init__(self, context: AppContext, provider_id: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        vm = context.advanced_provider_settings_viewmodel(provider_id)

        form_host = QWidget()
        form = QFormLayout(form_host)
        for field in vm.visible_fields():
            form.addRow(f"{field.group} — {field.label}", QLabel(field.value))

        if not vm.visible_fields():
            body: QWidget = muted("This provider exposes no configurable runtime settings.")
        else:
            body = form_host

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(card(f"Advanced settings - {provider_label(provider_id)}", body))


class SettingsView(QWidget):
    """The Settings Center: General on the left navigation, one Advanced page per provider,
    honest about what is read-only in this build (Part 8, Part 9).
    """

    def __init__(self, context: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._context = context

        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 20, 24, 24)
        outer.setSpacing(12)
        outer.addWidget(page_header("Settings", "Operator configuration for this workstation."))

        nav_and_pages = QWidget()
        row = QHBoxLayout(nav_and_pages)
        self._nav = QListWidget()
        self._nav.setMaximumWidth(220)
        self._nav.addItem("General")
        for provider_id in context.general_settings_viewmodel().available_providers():
            self._nav.addItem(f"Advanced: {provider_label(provider_id)}")
        self._nav.currentRowChanged.connect(self._on_nav_changed)
        row.addWidget(self._nav)

        self._stack = QStackedWidget()
        self._stack.addWidget(scroll_page([GeneralSettingsPanel(context)]))
        for provider_id in context.general_settings_viewmodel().available_providers():
            self._stack.addWidget(scroll_page([AdvancedProviderSettingsPanel(context, provider_id)]))
        row.addWidget(self._stack, stretch=1)

        outer.addWidget(nav_and_pages, stretch=1)
        self._nav.setCurrentRow(0)

    def _on_nav_changed(self, row: int) -> None:
        if 0 <= row < self._stack.count():
            self._stack.setCurrentIndex(row)
