"""The navigable centers of the Human Review application (the First Goal's page set).

Shared between the shell ViewModel (navigation state) and the Qt Views (which page widget to show),
so the set of pages is defined once and both layers agree. The application built here is the
**Architect Client** of `HUMAN_REVIEW_SPECIFICATION.md` §16.2 — "the complete operational platform"
that includes a review surface alongside the other Learning Platform centers — not the narrow
Reviewer Client of §16.1.
"""

from __future__ import annotations

from enum import Enum


class AppPage(str, Enum):
    """One navigable center. Order is display order in the navigation rail.

    Ordered to match the operator's actual mental model and the pipeline's own shape (Operational
    Hardening milestone, Priority 15): Workspace -> Acquisition -> Processing -> Review -> Quality
    -> Evidence Explorer -> Evolution -> Provider Manager -> Settings. Everything is
    Workspace-scoped (nothing works until a Workspace exists), so Workspace Manager leads rather
    than sitting sixth; Acquisition -- "get documents in," the single most frequent operator act --
    is promoted to a first-class page instead of being reachable only through a dialog nested
    inside Workspace Manager.
    """

    WORKSPACE_MANAGER = "workspace_manager"
    """Create/open/rename/clone/archive/delete/switch Workspaces (ROADMAP.md §5.13, Revision 5) —
    leads the rail because every other page is scoped to whichever Workspace is open."""
    ACQUISITION_MANAGER = "acquisition_manager"
    """Configured Acquisition Sources, their health, and Import Files/Add Folder Watch/Scan Now
    (ROADMAP.md §5.13.2) — promoted to first-class (Priority 15); previously reachable only via a
    dialog opened from Workspace Manager."""
    PROCESSING_CENTER = "processing_center"
    """Processing is ArchiveTrust's primary workflow, and review is an *exception* workflow entered
    intentionally (HUMAN_REVIEW_SPECIFICATION.md §1.1, §9.2) — the operational picture, not a
    single uncertainty, once a Workspace has documents to process."""
    REVIEW_CENTER = "review_center"
    QUALITY_CENTER = "quality_center"
    EVIDENCE_EXPLORER = "evidence_explorer"
    EVOLUTION_CENTER = "evolution_center"
    PROVIDER_MANAGER = "provider_manager"
    """Installed providers, their version/backend/model/runtime/status/health/capabilities
    (Provider Runtime Abstraction milestone) — an administration surface, ordered near Settings."""
    SETTINGS = "settings"

    @property
    def title(self) -> str:
        """Human-readable title for the navigation rail and window."""
        return {
            AppPage.WORKSPACE_MANAGER: "Workspace Manager",
            AppPage.ACQUISITION_MANAGER: "Acquisition",
            AppPage.PROCESSING_CENTER: "Processing Center",
            AppPage.REVIEW_CENTER: "Review Center",
            AppPage.QUALITY_CENTER: "Quality Center",
            AppPage.EVIDENCE_EXPLORER: "Evidence Explorer",
            AppPage.EVOLUTION_CENTER: "Evolution Center",
            AppPage.PROVIDER_MANAGER: "Provider Manager",
            AppPage.SETTINGS: "Settings",
        }[self]
