from __future__ import annotations

from archivetrust.presentation.pages import AppPage
from archivetrust.presentation.shell_viewmodel import ShellViewModel


def test_all_nine_centers_are_present_in_order() -> None:
    vm = ShellViewModel()
    # Operational Hardening milestone, Priority 15: reordered to match the operator's mental model
    # and the pipeline's own shape. Workspace Manager leads (everything is Workspace-scoped) and
    # Acquisition is promoted to a first-class page ("get documents in" was previously reachable
    # only via a dialog nested inside Workspace Manager).
    assert vm.pages == (
        AppPage.WORKSPACE_MANAGER,
        AppPage.ACQUISITION_MANAGER,
        AppPage.PROCESSING_CENTER,
        AppPage.REVIEW_CENTER,
        AppPage.QUALITY_CENTER,
        AppPage.EVIDENCE_EXPLORER,
        AppPage.EVOLUTION_CENTER,
        AppPage.PROVIDER_MANAGER,
        AppPage.SETTINGS,
    )


def test_default_page_is_workspace_manager() -> None:
    assert ShellViewModel().current_page.value is AppPage.WORKSPACE_MANAGER


def test_navigation_updates_observable() -> None:
    vm = ShellViewModel()
    seen: list[AppPage] = []
    vm.current_page.subscribe(seen.append)
    vm.navigate_to(AppPage.QUALITY_CENTER)
    vm.navigate_to(AppPage.SETTINGS)
    assert seen == [AppPage.QUALITY_CENTER, AppPage.SETTINGS]
    assert vm.current_page.value is AppPage.SETTINGS


def test_page_titles_are_human_readable() -> None:
    assert AppPage.EVIDENCE_EXPLORER.title == "Evidence Explorer"
