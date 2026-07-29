"""Per-provider configuration and capability gating (Parts 5, 8, 9).

`ProviderConfiguration` is what General Settings (Part 8, a handful of fields) and Advanced Provider
Settings (Part 9, everything) both edit. `ProviderSettingCapabilities` states which of Part 9's
setting groups actually apply to one provider/runtime pairing, so a settings UI never offers a
control that provider cannot honor (Part 9: "do not force every provider to share identical
settings") -- e.g. Tesseract has no device/precision/prompt notion at all; Docling has no prompt or
grounding-confidence notion; only a probabilistic, prompt-driven VLM provider exposes every group.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict

from archivetrust.providers.base import DeploymentProfile
from archivetrust.runtime.contracts import DeviceSelection, Precision
from archivetrust.runtime.provider_profiles import ProviderProfileName


class OutputFormat(str, Enum):
    JSON = "json"
    MARKDOWN = "markdown"
    PLAIN_TEXT = "plain_text"


class ExecutionMode(str, Enum):
    """General Settings' "Execution" control (Part 8) -- a friendlier alias over the same
    `DeviceSelection` every provider already uses, so the simple and advanced settings pages agree
    on one underlying value rather than maintaining two competing device concepts."""

    AUTOMATIC = "automatic"
    GPU_ONLY = "gpu_only"
    CPU_ONLY = "cpu_only"

    def to_device_selection(self) -> DeviceSelection:
        return DeviceSelection(self.value)


class ProcessingMode(str, Enum):
    """General Settings' "Processing Mode" control (Part 8) -- maps onto a Provider Profile so the
    simple settings page never exposes precision/batch/parallelism directly."""

    BALANCED = "balanced"
    FAST = "fast"
    MAXIMUM_QUALITY = "maximum_quality"

    def to_profile_name(self) -> ProviderProfileName:
        return {
            ProcessingMode.BALANCED: ProviderProfileName.ARCHIVE,
            ProcessingMode.FAST: ProviderProfileName.FAST_REVIEW,
            ProcessingMode.MAXIMUM_QUALITY: ProviderProfileName.MAXIMUM_QUALITY,
        }[self]


class ProviderSettingCapabilities(BaseModel):
    """Which Advanced Provider Settings groups (Part 9) apply to one provider. A settings page reads
    this before rendering a single control -- it is the enforcement mechanism for "only expose
    settings actually supported by each provider," not a convention left to each page to remember.
    """

    model_config = ConfigDict(frozen=True)

    has_device_selection: bool
    has_precision_selection: bool
    has_batching: bool
    has_image_processing: bool
    has_prompt_management: bool
    has_determinism_controls: bool
    has_grounding: bool
    """Bounding boxes / coordinates / confidence, per Part 9 -- only meaningful for a provider whose
    Evidence carries a Bounding Box (ROADMAP.md S4.2)."""
    supported_output_formats: tuple[OutputFormat, ...]


PROBABILISTIC_VISION_CAPABILITIES = ProviderSettingCapabilities(
    has_device_selection=True,
    has_precision_selection=True,
    has_batching=True,
    has_image_processing=True,
    has_prompt_management=True,
    has_determinism_controls=True,
    has_grounding=True,
    supported_output_formats=(OutputFormat.JSON, OutputFormat.MARKDOWN, OutputFormat.PLAIN_TEXT),
)
"""What a runtime-backed vision-language provider (Qwen2.5-VL, GLM-OCR, PaddleOCR-VL, ...)
exposes -- every Part 9 group applies."""

DETERMINISTIC_LOCAL_CAPABILITIES = ProviderSettingCapabilities(
    has_device_selection=False,
    has_precision_selection=False,
    has_batching=False,
    has_image_processing=False,
    has_prompt_management=False,
    has_determinism_controls=False,
    has_grounding=True,
    supported_output_formats=(OutputFormat.PLAIN_TEXT,),
)
"""What a classical, non-runtime-backed deterministic provider (Tesseract+LayoutParser, Docling)
exposes -- no device/precision/prompt/determinism notion applies to a CPU-bound, non-model
pipeline; grounding is still meaningful since these providers do emit bounding boxes."""


class ProviderConfiguration(BaseModel):
    """Everything an operator can set for one provider, across both the simple (Part 8) and
    advanced (Part 9) settings surfaces. Fields with no meaning for a given provider (per its
    `ProviderSettingCapabilities`) are simply never read by that provider's bridge -- they still
    exist here so one configuration type serves every provider uniformly, keeping the settings
    store simple (Part 8: "keep the default experience simple").
    """

    model_config = ConfigDict(frozen=True)

    provider_id: str
    enabled: bool = True
    profile: ProviderProfileName = ProviderProfileName.ARCHIVE

    # Device / execution (Part 8, Part 9 "Device")
    device: DeviceSelection = DeviceSelection.AUTOMATIC
    precision: Precision = Precision.AUTOMATIC

    # Performance (Part 9)
    batch_size: int = 1
    parallel_documents: int = 1
    worker_threads: int = 1

    # Image processing (Part 9)
    image_resolution: int | None = None
    max_tokens: int | None = None

    # Prompt management (Part 9/10)
    prompt_template_name: str = "default"
    prompt_template_version: int = 1

    # Determinism (Part 9)
    deterministic: bool = False
    seed: int | None = None

    # Grounding (Part 9)
    emit_bounding_boxes: bool = True
    emit_confidence: bool = True

    # Output (Part 9)
    output_format: OutputFormat = OutputFormat.JSON

    # Structured Pipeline deployment profile (Phase 31) -- read only by the
    # `paddleocr-vl`/`STRUCTURED_PIPELINE` adapter's construction; every other provider (and the
    # `SINGLE_PASS` profile of the same provider_id) never reads these, per this class's own stated
    # "fields with no meaning for a given provider are simply never read" convention. Defaulted so
    # every configuration persisted before this phase remains valid unchanged (Phase 20 precedent).
    recognize_furniture_regions: bool = False
    """Whether page-furniture regions (header/footer/page-number/decorative, per Phase 30's measured
    label vocabulary) are still sent to the VLM for recognition, instead of only receiving a
    geometry-only Observation. `False` matches Phase 30's measured-cheaper default (skip furniture)."""
    structured_recognition_concurrency: int | None = None
    """Max concurrent single-region recognition requests the Structured Pipeline profile issues
    (Phase 30: independent concurrent requests against vLLM's own continuous batching, never
    client-side multi-image batching). `None` defers to the adapter's own conservative default."""
    deployment_profile: DeploymentProfile = DeploymentProfile.SINGLE_PASS
    """Which deployment profile `composition.py` should register for this `provider_id` (Phase 31)
    -- e.g. `"paddleocr-vl"`'s existing whole-page adapter (`SINGLE_PASS`, unchanged default) vs. its
    Structured Pipeline adapter (`STRUCTURED_PIPELINE`). Only one profile is ever the *active,
    registered* adapter for a given `provider_id` at a time in this phase's composition wiring --
    `ProviderRegistry` itself supports holding both simultaneously (keyed on
    `(provider_id, deployment_profile)`), but `ProviderConfigurationStore`/`provider_health_trackers`/
    `enabled_provider_ids()` are all still keyed by bare `provider_id` elsewhere in this codebase, so
    concurrently running both profiles for the same provider is a real, larger change (per-profile
    settings/health/enablement) intentionally out of this phase's scope, not silently assumed away."""

    def with_effective_profile_marker(self) -> "ProviderConfiguration":
        """Returns `self` with `profile` set to `CUSTOM` if this configuration's values differ from
        the named profile's own preset -- so a settings UI never claims "Fast Review" is active
        once an operator has hand-edited a value out from under it (Part 7)."""
        from archivetrust.runtime.provider_profiles import default_profiles

        if self.profile == ProviderProfileName.CUSTOM:
            return self
        preset = next((p for p in default_profiles() if p.name == self.profile), None)
        if preset is None:
            return self
        matches = (
            self.device == preset.device
            and self.precision == preset.precision
            and self.batch_size == preset.batch_size
            and self.parallel_documents == preset.parallel_documents
            and self.deterministic == preset.deterministic
        )
        return self if matches else self.model_copy(update={"profile": ProviderProfileName.CUSTOM})


class ProviderConfigurationStore:
    """An in-memory (optionally file-backed) collection of `ProviderConfiguration`s, one per
    provider -- the General/Advanced Settings pages' single source of truth.
    """

    def __init__(self) -> None:
        self._configs: dict[str, ProviderConfiguration] = {}

    def set(self, configuration: ProviderConfiguration) -> None:
        self._configs[configuration.provider_id] = configuration

    def get(self, provider_id: str) -> ProviderConfiguration:
        return self._configs.get(provider_id, ProviderConfiguration(provider_id=provider_id))

    def all(self) -> tuple[ProviderConfiguration, ...]:
        return tuple(self._configs.values())

    def primary_vision_provider(self) -> str | None:
        """The provider currently bound to the logical "Primary Vision Provider" identifier (Part
        3, Part 8): the first enabled provider configured, in insertion order. `None` if no
        provider is enabled -- never guessed.
        """
        for config in self._configs.values():
            if config.enabled:
                return config.provider_id
        return None
