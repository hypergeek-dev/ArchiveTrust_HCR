"""First Launch ViewModel (Part 12): drives the setup wizard shown when no model is configured.

Never surfaces a Python traceback (Part 12's explicit requirement) — every method here returns a
result value or raises one of the small, named errors below, which a View renders as plain text.
On completion, this ViewModel binds a logical model into a `ModelRegistry` and persists the
resulting `ProviderConfiguration`/`PromptLibrary` to the deployment's `config/` directory, so the
next launch skips the wizard (`needs_first_launch()` returns `False` once a model is installed).

**The Vision Provider selection this wizard used to offer is gone (2026-07-30 residual cleanup).**

Stage 11 flagged and deliberately did not fix a real defect here: `VISION_PROVIDER_CHOICES` listed
`paddleocr-vl`, `qwen2.5-vl` and `surya` as selectable "vision reading engines" after all three
adapters were deleted in migration Stage 5, so a first-time researcher was offered three choices
nothing could activate -- and `paddleocr-vl` was the *default*, pre-filled into the Workspace Wizard's
model field. That flag was carried in `docs/htr-repository-cleanup.md`'s "Residual gaps honestly
carried forward" through two subsequent phases.

It is now resolved by **removing the selection**, not by re-populating it. The decisive fact is in
`composition.py`: `_VISION_PROVIDER_ADAPTER_FACTORIES` is an **empty dict**, so even a correct
`VISION_PROVIDER_CHOICES` entry could not produce a registered adapter --
`activate_configured_vision_provider` constructs the runtime, finds no factory, and returns. A catalog
entry is only meaningful when a matching factory exists, which is now the documented precondition on
`VISION_PROVIDER_CHOICES` itself.

**What replaces it, for an HTR research centre.** The three real methods are `HtrMethodAdapter`s and
are not configured through the Model Registry at all -- SATRN runs in an isolated venv, Florence-2
loads its own checkpoint through `transformers`, Transkribus imports a file. There is nothing for a
first-launch wizard to *bind*. What a first-launch flow can usefully do is tell the researcher whether
each method's environment is actually ready, which is what `htr_method_readiness()` below does by
calling each adapter's real `validate_environment()`. It reports; it configures nothing, because there
is nothing configurable.

The generic model-binding half of this wizard is retained unchanged and still serves a real purpose:
`ModelRegistry` bindings are how any future runtime-backed method would be resolved, and
`admin/qualification.py` and the Settings page both still administer them.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from archivetrust.runtime.deployment_layout import DeploymentLayout
from archivetrust.runtime.model_manager import scan_installed_models
from archivetrust.runtime.model_registry import LogicalModelBinding, ModelRegistry
from archivetrust.runtime.models import ModelDescriptor, ModelSource, ModelSourceKind

PRIMARY_VISION_PROVIDER = "Primary Vision Provider"

SECONDARY_VISION_PROVIDER = "Secondary Vision Provider"
"""A second, independent vision-provider binding slot (P4-P10 Prequalification, 2026-07-20).

Never surfaced by any wizard step; it exists solely for admin/automation callers (the qualification
runner) that must bind more than one vision-capable provider concurrently to satisfy a
`required_providers` list naming both -- something a single binding cannot express. See
`AppContext.bind_additional_vision_provider` and `activate_configured_vision_provider`.

Retained rather than deleted with the provider catalog: it names a *slot*, not a provider, and the
slot mechanism is generic. It currently has nothing to hold, which is a fact about
`_VISION_PROVIDER_ADAPTER_FACTORIES` being empty, not about this constant.
"""

NO_VISION_PROVIDER = ""
"""The `provider_id` a binding carries when no vision-provider adapter backs it.

Empty rather than a plausible-looking id, and an explicitly named constant rather than a bare `""`, so
`LogicalModelBinding(provider_id=NO_VISION_PROVIDER)` reads as a deliberate statement -- *this binding
resolves a model, and activates no `ProviderAdapter`* -- instead of looking like a value someone forgot
to fill in. `_VISION_PROVIDER_ADAPTER_FACTORIES.get("")` is `None`, so
`activate_configured_vision_provider` correctly registers nothing for it.
"""

DEFAULT_VISION_PROVIDER_ID = NO_VISION_PROVIDER
"""Formerly `"paddleocr-vl"`, whose adapter was deleted in migration Stage 5.

**A default naming a deleted provider is worse than a selectable option naming one**, which is why this
changed rather than only `VISION_PROVIDER_CHOICES`: an operator pressing through the Workspace Wizard
without touching anything used to get a persisted binding for a provider that cannot exist, and a
Provider Manager row for it.

Kept under its existing name because four modules import it (`composition.py`,
`presentation/workspace_wizard_viewmodel.py`, `admin/qualification.py`, and the Qt wizard); renaming it
would be a mechanical churn across a wizard this pass is otherwise narrowing, not widening.
"""

DEFAULT_VISION_MODEL_ID: str | None = None
"""Formerly `"PaddlePaddle/PaddleOCR-VL"`, pre-filled into the Workspace Wizard's model field.

`None` means "nothing pre-filled", which `WorkspaceWizardViewModel.create()` already handles as its
documented "leave blank to skip" path -- so a new Workspace now binds no vision model instead of
binding a deleted provider's. Typed `str | None` because
`WorkspaceWizardViewModel.model_identifier` is already `str | None`.
"""

DEFAULT_VISION_RUNTIME_KIND = "transformers"
"""Formerly `"vllm"`. Changed because `runtime/vllm_runtime.py` was **deleted** in migration Stage 5
along with the two providers that used it, so `"vllm"` names a runtime this build cannot construct --
`composition.py::_vllm_runtime_factory_with_budget` and its availability probe are both documented
placeholders that always report unavailable.

`"transformers"` is the only runtime kind with a real registered factory
(`composition.py::_register_transformers_runtime_factory`), so it is the only honest fallback for a
`provider_id` this module knows nothing about. See `runtime_kind_for_provider`, and note that
`declared_runtime_kind_for_provider` exists precisely so a caller that must not fall back does not.
"""

VISION_PROVIDER_CHOICES: tuple[tuple[str, str, str, str], ...] = ()
"""`(provider_id, display_label, default_model_identifier, runtime_kind)` -- **deliberately empty.**

It previously listed `paddleocr-vl`, `qwen2.5-vl` and `surya`. All three adapters were deleted in
migration Stage 5 (`docs/htr-repository-cleanup.md`'s providers table) and the entries were left
behind, so the wizard offered three choices nothing could activate. Stage 11 flagged this and left it;
this is the fix.

**The precondition for adding an entry back**, so the same defect cannot recur silently: an entry here
is activatable only if `composition.py::_VISION_PROVIDER_ADAPTER_FACTORIES` has a factory for the same
`provider_id`. That dict is currently empty. `tests/presentation/test_first_launch_viewmodel.py::
test_every_offered_vision_provider_has_an_adapter_factory` asserts the relationship in both directions,
so a future entry added without a factory (or a factory removed from under an entry) fails a test
rather than silently offering a dead choice again.

Empty, not deleted: the *catalog mechanism* is correct and generic -- the wizard reading a table rather
than hardcoding a pair is what made this defect a one-line fix instead of a wizard rewrite. Nothing in
this codebase breaks on an empty tuple; every consumer iterates it.
"""


def declared_runtime_kind_for_provider(provider_id: str) -> str | None:
    """The runtime `VISION_PROVIDER_CHOICES` declares for `provider_id`, or `None` if it declares
    none.

    The non-defaulting counterpart to `runtime_kind_for_provider`, added with the catalog emptying
    because the difference stopped being academic. A caller that *corrects* stored state must not treat
    "I have no declaration for this provider" as "the default is correct":
    `composition.py::_activate_vision_binding` self-heals a persisted binding's `runtime_kind` toward
    the catalog's declaration, and with an empty catalog the defaulting lookup would have rewritten
    every binding's runtime to `DEFAULT_VISION_RUNTIME_KIND` -- silently retargeting, say, a deliberate
    `openai_compatible` binding. That heal now runs only where a declaration actually exists.
    """
    for choice_provider_id, _label, _default_model, runtime_kind in VISION_PROVIDER_CHOICES:
        if choice_provider_id == provider_id:
            return runtime_kind
    return None


def runtime_kind_for_provider(provider_id: str) -> str:
    """The runtime a given Vision Provider requires -- looked up, never chosen by the operator
    (Part 2: "providers declare only id, name, path, runtime"; Part 11: "do NOT expand the
    ingestion UI"). Falls back to `DEFAULT_VISION_RUNTIME_KIND` for a `provider_id` the catalog does
    not declare, rather than raising -- `complete_setup` below still lets a caller override
    explicitly.

    With `VISION_PROVIDER_CHOICES` empty this always returns the fallback. That is correct for its
    callers (they need *some* runtime kind to probe or bind) and wrong for a caller correcting stored
    state -- see `declared_runtime_kind_for_provider`.
    """
    declared = declared_runtime_kind_for_provider(provider_id)
    return declared if declared is not None else DEFAULT_VISION_RUNTIME_KIND


def default_model_id_for_provider(provider_id: str) -> str | None:
    """The model a given Vision Provider serves -- looked up, mirroring `runtime_kind_for_provider`
    (Production Investigation, Settings UI / stale config, 2026-07-12: found a persisted binding
    naming `provider_id="paddleocr-vl"` but a `descriptor.model_id` still pointing at
    `"Qwen/Qwen2.5-VL-7B-Instruct"` -- a leftover, self-inconsistent pairing, not a deliberate
    operator choice, since `provider_id` and `model_id` must always name the same provider).
    `None` for a `provider_id` the catalog does not declare -- a future/enterprise provider's
    model identity is not this table's business to guess. Always `None` while the catalog is empty,
    which correctly disables the `model_id` half of `_activate_vision_binding`'s self-heal.
    """
    for choice_provider_id, _label, default_model, _runtime_kind in VISION_PROVIDER_CHOICES:
        if choice_provider_id == provider_id:
            return default_model
    return None


# -- HTR method readiness: what a first launch can usefully report in an HTR research centre -------


class HtrMethodReadiness(BaseModel):
    """One HTR method's identity plus the result of its **real** `validate_environment()` call.

    Qt-independent, like every ViewModel type in this package, and a plain value object -- a View
    renders it, it renders nothing itself.
    """

    model_config = ConfigDict(frozen=True)

    method_id: str
    method_name: str
    vendor: str
    model_revision: str
    """The pinned checkpoint the adapter reports, never `"main"`/`"latest"`. Transkribus honestly
    reports an unpinned string, which is why this is `str` and not a hash."""
    ready: bool
    messages: tuple[str, ...]
    """The adapter's own diagnostic messages, verbatim. `EnvironmentValidation` guarantees these are
    populated whenever `valid=False`, so a `ready=False` row is never a bare failure."""
    local_execution_supported: bool
    external_upload_required: bool
    """Both read from `get_capabilities()`, because "is this ready?" means different things for a
    method that loads GPU weights and one that parses an export file, and a first-launch reader needs
    to know which they are looking at."""


def htr_method_readiness(adapters: tuple) -> tuple[HtrMethodReadiness, ...]:
    """Calls `get_metadata()`, `get_capabilities()` and `validate_environment()` on each adapter.

    `adapters` is injected (`AppContext.htr_method_adapters()` in production, fakes in tests) rather
    than constructed here -- this module has no business importing three provider packages, and a
    ViewModel that builds its own collaborators is not testable without them.

    **`validate_environment()` is the only outward-reaching call**, and it is the one the brief names.
    For the three real adapters it is cheap and side-effect-free: SATRN probes `torch` importability
    and its isolated venv's interpreter path, Florence-2 probes `transformers`/`torch` importability,
    Transkribus stats a configured import directory. None loads weights, opens a socket, or starts a
    process. An adapter that raised is reported as `ready=False` with the exception text rather than
    propagating -- a first-launch screen must never show a traceback (Part 12), and one broken method
    must not hide the other two.
    """
    rows: list[HtrMethodReadiness] = []
    for adapter in adapters:
        metadata = adapter.get_metadata()
        capabilities = adapter.get_capabilities()
        try:
            validation = adapter.validate_environment()
            ready = validation.valid
            messages = tuple(validation.messages)
        except Exception as exc:  # noqa: BLE001 -- reported, never raised into a wizard
            ready = False
            messages = (f"validate_environment() raised {type(exc).__name__}: {exc}",)
        rows.append(
            HtrMethodReadiness(
                method_id=metadata.method_id,
                method_name=metadata.method_name,
                vendor=metadata.vendor,
                model_revision=metadata.model_revision,
                ready=ready,
                messages=messages,
                local_execution_supported=capabilities.local_execution_supported,
                external_upload_required=capabilities.external_upload_required,
            )
        )
    return tuple(rows)


class SetupSourceKind(str, Enum):
    """The four options Part 12 lists."""

    BROWSE_LOCAL = "browse_local"
    HUGGING_FACE = "hugging_face"
    ENTERPRISE_REPOSITORY = "enterprise_repository"
    NETWORK_LOCATION = "network_location"

    def to_model_source_kind(self) -> ModelSourceKind:
        return {
            SetupSourceKind.BROWSE_LOCAL: ModelSourceKind.LOCAL_PATH,
            SetupSourceKind.HUGGING_FACE: ModelSourceKind.HUGGING_FACE,
            SetupSourceKind.ENTERPRISE_REPOSITORY: ModelSourceKind.ENTERPRISE_REPOSITORY,
            SetupSourceKind.NETWORK_LOCATION: ModelSourceKind.NETWORK_LOCATION,
        }[self]


class InvalidSetupInputError(ValueError):
    """A user-facing configuration problem — a View catches this and shows its message, never a
    traceback (Part 12)."""


class SetupResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    logical_name: str
    model_id: str
    runtime_kind: str
    provider_id: str


def needs_first_launch(layout: DeploymentLayout, model_registry: ModelRegistry | None = None) -> bool:
    """`True` when nothing is configured yet — the wizard-triggering condition (Part 12).

    Checked two ways, since a configured model may have no on-disk footprint at all (a Hugging
    Face or network-location source has nothing under `models_dir` to scan): a local model
    installed under `layout.models_dir`, **or** a persisted "Primary Vision Provider" binding
    (`model_registry.bindings_path()`) from a prior First Launch. Scanning disk alone would show
    the wizard again on every restart for any non-local source, which is exactly the "configure
    once" experience Part 12 asks for.
    """
    if len(scan_installed_models(layout)) > 0:
        return False
    if model_registry is not None and any(
        b.logical_name == PRIMARY_VISION_PROVIDER for b in model_registry.bindings()
    ):
        return False
    return True


class FirstLaunchViewModel:
    def __init__(
        self,
        *,
        layout: DeploymentLayout,
        model_registry: ModelRegistry,
        htr_method_adapters: tuple = (),
    ) -> None:
        self._layout = layout
        self._registry = model_registry
        self._htr_method_adapters = htr_method_adapters
        """Injected, and defaulted to empty so every existing caller and test constructs unchanged.
        `AppContext.first_launch_viewmodel()` passes the three real adapters."""

    def needs_setup(self) -> bool:
        return needs_first_launch(self._layout, self._registry)

    def htr_method_readiness(self) -> tuple[HtrMethodReadiness, ...]:
        """What the three real HTR methods report about their own environments, right now.

        The HTR-relevant thing a first launch can offer (see the module docstring): the methods are
        not bound through the Model Registry and there is nothing to configure, so this reports
        rather than configures. Empty when no adapters were injected -- an empty list is the honest
        answer to "which methods are ready?" when nothing was supplied, not an error.
        """
        return htr_method_readiness(self._htr_method_adapters)

    def complete_setup(
        self,
        *,
        source_kind: SetupSourceKind,
        identifier: str,
        runtime_kind: str | None = None,
        revision: str | None = None,
        display_name: str | None = None,
        provider_id: str = DEFAULT_VISION_PROVIDER_ID,
    ) -> SetupResult:
        """Validates and saves the operator's choice; binds it as the "Primary Vision Provider"
        logical model. Raises `InvalidSetupInputError` (never a bare exception) on bad input.

        `provider_id` (Multi-Provider Activation milestone) records *which* registered
        `ProviderAdapter` this binding activates. It now defaults to `NO_VISION_PROVIDER` (`""`),
        because `VISION_PROVIDER_CHOICES` is empty and `_VISION_PROVIDER_ADAPTER_FACTORIES` has no
        factories -- so the honest default is "this binding activates no adapter". A caller that has a
        real, activatable provider id passes it explicitly, exactly as before.

        `runtime_kind` (Runtime Architecture Completion milestone) is optional -- when omitted, it
        is derived from `provider_id` via `runtime_kind_for_provider()`, since which runtime a
        provider needs is that provider's own property, never an independent operator choice (Part
        2/11). A caller may still pass it explicitly (existing tests, and any advanced/enterprise
        caller pointing a provider at a non-default runtime deployment).
        """
        identifier = identifier.strip()
        if not identifier:
            raise InvalidSetupInputError("A model location or repository identifier is required.")

        model_source_kind = source_kind.to_model_source_kind()
        if model_source_kind == ModelSourceKind.LOCAL_PATH:
            candidate = Path(identifier)
            if not candidate.exists():
                raise InvalidSetupInputError(f"No such local path: {identifier}")
            identifier = candidate.name if candidate.is_dir() and candidate.parent == self._layout.models_dir else identifier

        resolved_runtime_kind = runtime_kind or runtime_kind_for_provider(provider_id)
        model_id = identifier if model_source_kind != ModelSourceKind.LOCAL_PATH else Path(identifier).name
        descriptor = ModelDescriptor(
            model_id=model_id,
            display_name=display_name or model_id,
            source=ModelSource(kind=model_source_kind, identifier=identifier, revision=revision),
            runtime_kind=resolved_runtime_kind,
        )
        self._registry.bind(
            LogicalModelBinding(
                logical_name=PRIMARY_VISION_PROVIDER, descriptor=descriptor, provider_id=provider_id
            )
        )
        self._registry.save_bindings()  # survives a restart -- see needs_first_launch()
        return SetupResult(
            logical_name=PRIMARY_VISION_PROVIDER, model_id=model_id, runtime_kind=resolved_runtime_kind,
            provider_id=provider_id,
        )
