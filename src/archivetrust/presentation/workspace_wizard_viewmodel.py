"""Workspace Wizard ViewModel (ROADMAP.md §5.13.1) — the professional first-run/new-Workspace flow:
Name → Profile → Providers → Models → Input Sources → Output Location → Review → Create.

Nothing is persisted until `create()` (the Review step's action) — the operator can step back and
forth freely beforehand. `create()` then drives the same real objects a hand-built Workspace would:
`WorkspaceStore.create`, `ProviderConfigurationStore`, and (reusing, not duplicating,
`FirstLaunchViewModel`'s existing binding logic — decision 5) a Model Registry binding, plus any
chosen Acquisition Sources.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from archivetrust.acquisition.folder_watch import FolderWatchConfig, FolderWatchSource
from archivetrust.acquisition.manual_import import ManualImportSource
from archivetrust.presentation.first_launch_viewmodel import (
    DEFAULT_VISION_MODEL_ID,
    DEFAULT_VISION_PROVIDER_ID,
    FirstLaunchViewModel,
    SetupSourceKind,
    runtime_kind_for_provider,
)
from archivetrust.runtime.provider_config import ProviderConfigurationStore
from archivetrust.runtime.provider_profiles import ProviderProfile, ProviderProfileName, default_profiles
from archivetrust.workspace.models import Workspace
from archivetrust.workspace.store import WorkspaceStore


class WizardProviderChoice(BaseModel):
    """One selectable provider in the Workspace Wizard's Providers step (Provider Discovery
    milestone, 2026-07-13). Assembled by `AppContext.available_wizard_providers()` from the same
    sources every other provider-facing surface uses (`ProviderRegistry.all()` for eagerly-
    registered deterministic providers, the Vision Provider catalog for probabilistic ones) —
    never from a list of provider ids hand-maintained inside the wizard itself. Adding a new
    provider (a registry registration, or a `VISION_PROVIDER_CHOICES` entry — both already
    mandatory for the provider to function at all) is what makes it appear here; no wizard code
    changes are needed.
    """

    model_config = ConfigDict(frozen=True)

    provider_id: str
    label: str
    description: str
    group: str
    """A human-facing category ("Deterministic (Layout)", "Deterministic (OCR)", "Vision-Language
    Models", ...) — derived from the provider's own registered `Reproducibility` axis by default,
    so a future provider groups sensibly with zero configuration; overridable per-provider only
    for the finer distinctions (Layout vs. OCR) that axis alone can't express."""
    available: bool
    unavailable_reason: str | None
    """Set exactly when `available` is `False` — why (a missing optional dependency, an absent
    runtime), never left for the operator to guess (mirrors `ProviderManagerEntry.status`'s
    Unavailable/Configuration Error vocabulary)."""
    default_enabled: bool


class WizardContext(Protocol):
    """The slice of `AppContext` the wizard needs, expressed locally so `presentation` never
    imports `clients` (`tests/review/test_separation.py`)."""

    current_workspace: Workspace | None
    workspace_store: WorkspaceStore

    def create_workspace(self, name: str, description: str = "") -> Workspace: ...
    def open_workspace(self, workspace_id: str) -> Workspace: ...
    def first_launch_viewmodel(self) -> FirstLaunchViewModel: ...
    def activate_configured_vision_provider(self) -> None: ...
    def available_wizard_providers(self) -> tuple[WizardProviderChoice, ...]: ...

    @property
    def provider_config_store(self) -> ProviderConfigurationStore: ...

    @property
    def acquisition_manager(self): ...

    @property
    def _current_layout(self): ...


class WizardStep(str, Enum):
    NAME = "name"
    PROFILE = "profile"
    PROVIDERS = "providers"
    MODELS = "models"
    INPUT_SOURCES = "input_sources"
    OUTPUT_LOCATION = "output_location"
    REVIEW = "review"


STEP_ORDER: tuple[WizardStep, ...] = tuple(WizardStep)


class InvalidWizardStateError(ValueError):
    """A user-facing configuration problem — a View catches this and shows its message, never a
    traceback, mirroring `FirstLaunchViewModel`'s own `InvalidSetupInputError` convention."""


class WizardReview(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    processing_profile: str
    enabled_provider_ids: tuple[str, ...]
    model_identifier: str | None
    watched_folder: str | None
    output_location: str


class WorkspaceWizardViewModel:
    def __init__(self, *, store: WorkspaceStore, context: WizardContext) -> None:
        self._store = store
        self._context = context
        self.step: WizardStep = WizardStep.NAME

        self.name: str = ""
        self.description: str = ""
        self.processing_profile: ProviderProfileName = ProviderProfileName.ARCHIVE
        # Derived from the same dynamic provider list the Providers step renders (Provider
        # Discovery milestone, 2026-07-13) -- never a second, independently-hand-maintained id set
        # that could drift from it (the previous `{"docling", "tesseract_layoutparser"}` literal
        # happened to match `WIZARD_PROVIDER_CHOICES`'s `default_enabled` intent by coincidence,
        # but nothing enforced that the two stayed in sync).
        self.enabled_provider_ids: set[str] = {
            choice.provider_id for choice in self.available_providers() if choice.default_enabled
        }
        # Pre-filled with the default Vision Provider (Multi-Provider Activation milestone): a new
        # Workspace automatically inherits PaddleOCR-VL unless the operator clears the Models step
        # field (the existing "leave blank to skip" affordance is exactly the override mechanism --
        # nothing new needed there). Existing Workspaces are never touched by this default: it only
        # affects what a *new* WorkspaceWizardViewModel starts with, never a saved binding.
        self.model_identifier: str | None = DEFAULT_VISION_MODEL_ID
        self.model_provider_id: str = DEFAULT_VISION_PROVIDER_ID
        # No separate `model_runtime_kind` field (Runtime Architecture Completion milestone, Part
        # 2/11): which runtime a provider needs is that provider's own property, derived from
        # `model_provider_id` via `runtime_kind_for_provider()` at `create()` time below -- never
        # an independent wizard choice.
        self.add_manual_import: bool = True
        self.watched_folder: str | None = None

    # -- step navigation -----------------------------------------------------------------------

    def can_advance(self) -> bool:
        if self.step == WizardStep.NAME:
            return bool(self.name.strip())
        return True

    def advance(self) -> None:
        if not self.can_advance():
            raise InvalidWizardStateError("A Workspace name is required before continuing.")
        index = STEP_ORDER.index(self.step)
        if index + 1 < len(STEP_ORDER):
            self.step = STEP_ORDER[index + 1]

    def back(self) -> None:
        index = STEP_ORDER.index(self.step)
        if index > 0:
            self.step = STEP_ORDER[index - 1]

    def available_profiles(self) -> tuple[ProviderProfile, ...]:
        return default_profiles()

    def available_providers(self) -> tuple[WizardProviderChoice, ...]:
        """The Providers step's checkbox list (Provider Discovery milestone, 2026-07-13) — reads
        `AppContext.available_wizard_providers()`, which enumerates the same
        `ProviderRegistry`/Vision Provider catalog every other provider-facing surface uses.
        Holds no provider id itself; a provider newly registered there appears here automatically.
        """
        return self._context.available_wizard_providers()

    def review(self) -> WizardReview:
        return WizardReview(
            name=self.name,
            description=self.description,
            processing_profile=self.processing_profile.value,
            enabled_provider_ids=tuple(sorted(self.enabled_provider_ids)),
            model_identifier=self.model_identifier,
            watched_folder=self.watched_folder,
            output_location="Managed automatically by ArchiveTrust.",
        )

    # -- creation --------------------------------------------------------------------------------

    def create(self) -> Workspace:
        if not self.name.strip():
            raise InvalidWizardStateError("A Workspace name is required.")

        workspace = self._context.create_workspace(self.name.strip(), description=self.description.strip())
        self._store.set_processing_profile(workspace.id, self.processing_profile)
        self._context.open_workspace(workspace.id)  # reload so current_workspace reflects the profile

        for choice in self.available_providers():
            provider_id = choice.provider_id
            enabled = choice.available and provider_id in self.enabled_provider_ids
            self._context.provider_config_store.set(
                self._context.provider_config_store.get(provider_id).model_copy(update={"enabled": enabled})
            )

        if self.model_identifier:
            self._context.first_launch_viewmodel().complete_setup(
                source_kind=SetupSourceKind.HUGGING_FACE,
                identifier=self.model_identifier,
                runtime_kind=runtime_kind_for_provider(self.model_provider_id),
                provider_id=self.model_provider_id,
            )
            # Root-cause fix (Production Incident, 2026-07-13): `complete_setup` above *binds* the
            # vision provider but registers no adapter -- and this method's last `open_workspace`
            # call (whose tail is the only other activation site) ran *before* the binding existed.
            # A wizard-created Workspace therefore ran its entire first queue without the vision
            # provider the operator just configured, silently, until the next app restart.
            try:
                self._context.activate_configured_vision_provider()
            except Exception as exc:  # noqa: BLE001 -- surfaced to the operator, never a traceback
                raise InvalidWizardStateError(
                    "The Workspace was created, but its vision provider could not be activated: "
                    f"{exc}"
                ) from exc

        if self.add_manual_import:
            self._context.acquisition_manager.add_source(ManualImportSource())
        if self.watched_folder:
            self._context.acquisition_manager.add_source(
                FolderWatchSource(FolderWatchConfig(path=Path(self.watched_folder)))
            )

        return self._context.current_workspace
