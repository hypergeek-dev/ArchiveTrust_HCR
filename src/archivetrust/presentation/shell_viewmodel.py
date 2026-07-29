"""The shell / navigation ViewModel (First Goal: navigation + window management).

Holds which center is currently shown and drives navigation between them. Framework-independent:
the Qt shell (Increment 3) binds a navigation rail and a page stack to this ViewModel's
`current_page` observable; it contains no navigation logic of its own.
"""

from __future__ import annotations

from archivetrust.presentation.observable import Observable
from archivetrust.presentation.pages import AppPage


class ShellViewModel:
    """Navigation state for the application shell. `current_page` is observable so the View's page
    stack and rail selection update together whenever navigation happens, from wherever it is
    triggered (rail click, keyboard shortcut, or a deep-link from another page).
    """

    def __init__(self, *, initial_page: AppPage = AppPage.WORKSPACE_MANAGER) -> None:
        self.pages: tuple[AppPage, ...] = tuple(AppPage)
        self.current_page: Observable[AppPage] = Observable(initial_page)

    def navigate_to(self, page: AppPage) -> None:
        self.current_page.value = page
