"""First Launch ViewModel (Part 12): drives the setup wizard shown when no model is configured.

Never surfaces a Python traceback (Part 12's explicit requirement) — every method here returns a
result value or raises one of the small, named errors below, which a View renders as plain text.
On completion, this ViewModel binds a logical model into a `ModelRegistry` and persists the
resulting `ProviderConfiguration`/`PromptLibrary` to the deployment's `config/` directory, so the
next launch skips the wizard (`needs_first_launch()` returns `False` once a model is installed).

.. warning::

   **STALE AFTER THE HTR TRANSFORMATION -- flagged, not silently left (Stage 11 terminology
   sweep).** Every provider in `VISION_PROVIDER_CHOICES` below (`paddleocr-vl`, `qwen2.5-vl`,
   `surya`) had its adapter **deleted** in migration Stage 5
   (`docs/htr-repository-cleanup.md`'s providers table). Verified at the time of this note:
   `AppContext.provider_registry` now registers zero adapters, so this wizard offers a first-time
   researcher three choices that cannot be activated by anything.

   This was **not** repaired during Stage 11 on purpose. Stage 11's scope is the research
   interface (`presentation/htr_*_viewmodel.py`, `clients/desktop_v2/htr_pages.py`); deciding what
   a first-launch flow should offer in an HTR research centre -- whether it configures HTR methods
   at all, given SATRN/Florence-2 need model weights and Transkribus needs no setup -- is a
   first-launch redesign, and guessing at it here would replace one wrong flow with another.

   The HTR methods themselves are *not* configured through this path: they are
   `HtrMethodAdapter`s, surfaced by `AppContext.htr_method_adapters()` and the HTR Methods page,
   which reports each one's real environment validity on request.
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

The First Launch wizard only ever offers one choice (`VISION_PROVIDER_CHOICES` above, Part 11: "do
NOT expand the ingestion UI") -- this constant is never surfaced there and no operator-facing wizard
step creates a binding under this name. It exists solely for admin/automation callers (the
qualification runner) that must run more than one vision-capable provider (e.g. `paddleocr-vl` and
`surya`) concurrently to satisfy a `required_providers` list naming both -- something a single-choice
wizard binding cannot express. See `AppContext.bind_additional_vision_provider` (composition.py) and
`activate_configured_vision_provider`, which now activates every entry in `_VISION_PROVIDER_LOGICAL_NAMES`
rather than only the primary one.
"""

DEFAULT_VISION_PROVIDER_ID = "paddleocr-vl"
DEFAULT_VISION_MODEL_ID = "PaddlePaddle/PaddleOCR-VL"
DEFAULT_VISION_RUNTIME_KIND = "vllm"
"""The default Vision Provider (Multi-Provider Activation milestone) — a ~0.96B-parameter model
(~1.9GB in bf16), chosen specifically because it fits comfortably on a modest single GPU. Its
runtime is `"vllm"`, not `"transformers"` (Runtime Architecture Completion milestone): live
investigation (`docs/PADDLEOCR_VL_RUNTIME_COMPATIBILITY_INVESTIGATION.md`,
`docs/PADDLEOCR_VL_VLLM_LIVE_VERIFICATION.md`) found PaddleOCR-VL genuinely broken under the plain
`transformers` library across every installed version tried, but confirmed working end-to-end
through a self-managed vLLM server on this exact reference hardware. Qwen remains fully supported
and selectable on `"transformers"` — this only changes which provider/runtime pair a first-time
operator is guided toward. `DEFAULT_VISION_MODEL_ID` resolves automatically (`VLLMRuntime` pulls
and serves it via Docker on first use); no filesystem path or runtime choice is ever required for
the default path (Part 3/11: "the operator shall never have to configure ... docker images")."""

VISION_PROVIDER_CHOICES: tuple[tuple[str, str, str, str], ...] = (
    (DEFAULT_VISION_PROVIDER_ID, "PaddleOCR-VL (recommended — lightweight, ~2GB)", DEFAULT_VISION_MODEL_ID, DEFAULT_VISION_RUNTIME_KIND),
    ("qwen2.5-vl", "Qwen2.5-VL (large — requires a GPU with ≥16GB VRAM)", "Qwen/Qwen2.5-VL-7B-Instruct", "transformers"),
    ("surya", "Surya (Docker/vLLM-served OCR)", "datalab-to/surya-ocr-2", "vllm"),
)
"""`(provider_id, display_label, default_model_identifier, runtime_kind)` — the wizard's Vision
Provider combo reads this, never a hardcoded pair, so a future provider only needs an entry here
plus a matching `activate_configured_vision_provider()` dispatch case (composition.py), not a
wizard rewrite. `runtime_kind` (Runtime Architecture Completion milestone) is a property of the
*provider*, never a separate operator choice (Part 2/11) -- `runtime_kind_for_provider()` below is
how a caller looks it up instead of exposing a runtime selector in any wizard."""


def runtime_kind_for_provider(provider_id: str) -> str:
    """The runtime a given Vision Provider requires -- looked up, never chosen by the operator
    (Part 2: "providers declare only id, name, path, runtime"; Part 11: "do NOT expand the
    ingestion UI"). Falls back to `DEFAULT_VISION_RUNTIME_KIND` for a `provider_id` not in
    `VISION_PROVIDER_CHOICES` (e.g. a future/enterprise provider configured outside this wizard)
    rather than raising -- `complete_setup` below still lets a caller override explicitly.
    """
    for choice_provider_id, _label, _default_model, runtime_kind in VISION_PROVIDER_CHOICES:
        if choice_provider_id == provider_id:
            return runtime_kind
    return DEFAULT_VISION_RUNTIME_KIND


def default_model_id_for_provider(provider_id: str) -> str | None:
    """The model a given Vision Provider serves -- looked up, mirroring `runtime_kind_for_provider`
    (Production Investigation, Settings UI / stale config, 2026-07-12: found a persisted binding
    naming `provider_id="paddleocr-vl"` but a `descriptor.model_id` still pointing at
    `"Qwen/Qwen2.5-VL-7B-Instruct"` -- a leftover, self-inconsistent pairing, not a deliberate
    operator choice, since `provider_id` and `model_id` must always name the same provider).
    `None` for a `provider_id` not in `VISION_PROVIDER_CHOICES` -- a future/enterprise provider's
    model identity is not this table's business to guess.
    """
    for choice_provider_id, _label, default_model, _runtime_kind in VISION_PROVIDER_CHOICES:
        if choice_provider_id == provider_id:
            return default_model
    return None


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
    def __init__(self, *, layout: DeploymentLayout, model_registry: ModelRegistry) -> None:
        self._layout = layout
        self._registry = model_registry

    def needs_setup(self) -> bool:
        return needs_first_launch(self._layout, self._registry)

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
        `ProviderAdapter` this binding activates — defaults to the new default Vision Provider
        (PaddleOCR-VL), but any caller selecting a different entry from `VISION_PROVIDER_CHOICES`
        (e.g. Qwen2.5-VL) passes its id explicitly.

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
