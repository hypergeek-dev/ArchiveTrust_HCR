"""Navigation state for the Desktop v2 shell.

Since Stage 11 the default landing page is the research overview rather than the workspace picker:
this is a Historical HTR research centre, so the first thing a researcher sees is the state of their
projects, datasets and experiments.
"""

from __future__ import annotations

from archivetrust.presentation.desktop_v2_pages import DesktopV2Page
from archivetrust.presentation.observable import Observable


class DesktopV2ShellViewModel:
    """Which page the shell is showing, and the ordered page list its rail renders."""

    def __init__(
        self, *, initial_page: DesktopV2Page = DesktopV2Page.RESEARCH_OVERVIEW
    ) -> None:
        self.pages: tuple[DesktopV2Page, ...] = tuple(DesktopV2Page)
        self.current_page: Observable[DesktopV2Page] = Observable(initial_page)

    def navigate_to(self, page: DesktopV2Page) -> None:
        self.current_page.value = page

    def research_pages(self) -> tuple[DesktopV2Page, ...]:
        """The HTR research surfaces, in rail order."""
        return tuple(page for page in self.pages if page.is_research_surface)

    def operational_pages(self) -> tuple[DesktopV2Page, ...]:
        """The retained acquisition/processing/release surfaces, in rail order."""
        return tuple(page for page in self.pages if not page.is_research_surface)
