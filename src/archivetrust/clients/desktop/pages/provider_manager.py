"""Provider Manager view (Part 5). Displays installed providers with version, backend, model,
runtime, status, health, and capabilities. Binds to `ProviderManagerViewModel`; holds no logic of
its own — enabling/disabling a provider is forwarded straight to the ViewModel.
"""

from __future__ import annotations

from PySide6.QtWidgets import QVBoxLayout, QWidget

from archivetrust.composition import AppContext
from archivetrust.clients.desktop.ui import card, page_header, scroll_page, table
from archivetrust.presentation.display_names import provider_label, runtime_label


def _fmt_seconds(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 1000:.0f} ms"


class ProviderManagerView(QWidget):
    def __init__(self, context: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._context = context
        vm = context.provider_manager_viewmodel()
        entries = vm.entries()

        rows = [
            [
                provider_label(e.provider_id),
                "Yes" if e.enabled else "No",
                e.backend,
                runtime_label(e.runtime_kind),
                e.status,
                e.version or "-",
                _fmt_seconds(e.health.average_inference_seconds),
            ]
            for e in entries
        ]
        providers_card = card(
            "Installed providers",
            table(
                ["Provider", "Enabled", "Backend", "Runtime", "Status", "Model revision", "Avg. inference"],
                rows,
            ),
        )

        capability_rows = [
            [
                provider_label(e.provider_id),
                "Yes" if e.capabilities.has_device_selection else "-",
                "Yes" if e.capabilities.has_precision_selection else "-",
                "Yes" if e.capabilities.has_prompt_management else "-",
                "Yes" if e.capabilities.has_determinism_controls else "-",
                "Yes" if e.capabilities.has_grounding else "-",
            ]
            for e in entries
        ]
        capabilities_card = card(
            "Capabilities (only what each provider actually supports is offered in Advanced Settings)",
            table(["Provider", "Device", "Precision", "Prompt", "Determinism", "Grounding"], capability_rows),
        )

        page = scroll_page(
            [
                page_header(
                    "Provider Manager",
                    "Every registered document-understanding provider, deterministic and "
                    "probabilistic alike, behind one uniform interface.",
                ),
                providers_card,
                capabilities_card,
            ]
        )
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.addWidget(page)
