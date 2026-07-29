"""Settings ViewModels — General (Part 8) and Advanced Provider Settings (Part 9).

`GeneralSettingsViewModel` exposes exactly the three controls Part 8 asks for: Primary Vision
Provider, Execution, Processing Mode. `AdvancedProviderSettingsViewModel` exposes everything Part 9
lists, but reads only the fields a given provider's `ProviderSettingCapabilities` says apply — a
View binds to `.visible_groups` before rendering a single control, so an incapable provider is never
shown a setting it cannot honor (Part 9: "do not force every provider to share identical settings").
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from archivetrust.providers.registry import ProviderRegistry
from archivetrust.runtime.contracts import DeviceSelection, Precision
from archivetrust.runtime.provider_config import (
    ExecutionMode,
    OutputFormat,
    ProcessingMode,
    ProviderConfiguration,
    ProviderConfigurationStore,
    ProviderSettingCapabilities,
)
from archivetrust.runtime.provider_profiles import ProviderProfileName


class GeneralSettingsViewModel:
    """The simple, operator-facing settings page (Part 8). Deliberately narrow: three controls,
    nothing else — advanced tuning lives one level down, per provider (Part 9).
    """

    def __init__(self, *, provider_registry: ProviderRegistry, configuration_store: ProviderConfigurationStore) -> None:
        self._providers = provider_registry
        self._config = configuration_store

    def available_providers(self) -> tuple[str, ...]:
        return tuple(sorted(a.provider_id for a in self._providers.all()))

    def primary_vision_provider(self) -> str | None:
        return self._config.primary_vision_provider()

    def set_primary_vision_provider(self, provider_id: str) -> None:
        # Exactly one provider is "primary" (Part 8) -- disable every other, enable the chosen one.
        for adapter in self._providers.all():
            current = self._config.get(adapter.provider_id)
            self._config.set(current.model_copy(update={"enabled": adapter.provider_id == provider_id}))

    def execution_mode(self) -> ExecutionMode:
        provider_id = self.primary_vision_provider()
        if provider_id is None:
            return ExecutionMode.AUTOMATIC
        device = self._config.get(provider_id).device
        return ExecutionMode(device.value)

    def set_execution_mode(self, mode: ExecutionMode) -> None:
        provider_id = self.primary_vision_provider()
        if provider_id is None:
            return
        current = self._config.get(provider_id)
        self._config.set(current.model_copy(update={"device": mode.to_device_selection()}))

    def processing_mode(self) -> ProcessingMode:
        provider_id = self.primary_vision_provider()
        if provider_id is None:
            return ProcessingMode.BALANCED
        profile = self._config.get(provider_id).profile
        return {
            ProviderProfileName.ARCHIVE: ProcessingMode.BALANCED,
            ProviderProfileName.FAST_REVIEW: ProcessingMode.FAST,
            ProviderProfileName.MAXIMUM_QUALITY: ProcessingMode.MAXIMUM_QUALITY,
        }.get(profile, ProcessingMode.BALANCED)

    def set_processing_mode(self, mode: ProcessingMode) -> None:
        provider_id = self.primary_vision_provider()
        if provider_id is None:
            return
        current = self._config.get(provider_id)
        self._config.set(current.model_copy(update={"profile": mode.to_profile_name()}))


class SettingField(BaseModel):
    """One editable field the Advanced Provider Settings view can render — group name plus current
    value, so a generic View can lay these out without hardcoding per-provider knowledge."""

    model_config = ConfigDict(frozen=True)

    group: str
    label: str
    value: str


class AdvancedProviderSettingsViewModel:
    """One provider's full configuration surface (Part 9), gated by its
    `ProviderSettingCapabilities`."""

    def __init__(self, *, provider_id: str, configuration_store: ProviderConfigurationStore, capabilities: ProviderSettingCapabilities) -> None:
        self._provider_id = provider_id
        self._config = configuration_store
        self.capabilities = capabilities

    def configuration(self) -> ProviderConfiguration:
        return self._config.get(self._provider_id)

    def _update(self, **fields: object) -> None:
        current = self.configuration()
        self._config.set(current.model_copy(update=fields))

    # -- Device / Precision --------------------------------------------------------------------

    def set_device(self, device: DeviceSelection) -> None:
        self._update(device=device)

    def set_precision(self, precision: Precision) -> None:
        self._update(precision=precision)

    # -- Performance ----------------------------------------------------------------------------

    def set_batch_size(self, batch_size: int) -> None:
        self._update(batch_size=max(1, batch_size))

    def set_parallel_documents(self, count: int) -> None:
        self._update(parallel_documents=max(1, count))

    def set_worker_threads(self, count: int) -> None:
        self._update(worker_threads=max(1, count))

    # -- Image processing -----------------------------------------------------------------------

    def set_image_resolution(self, pixels: int | None) -> None:
        self._update(image_resolution=pixels)

    def set_max_tokens(self, tokens: int | None) -> None:
        self._update(max_tokens=tokens)

    # -- Prompt management ------------------------------------------------------------------------

    def set_prompt_template(self, name: str, version: int) -> None:
        self._update(prompt_template_name=name, prompt_template_version=version)

    # -- Determinism ------------------------------------------------------------------------------

    def set_deterministic(self, *, deterministic: bool, seed: int | None = None) -> None:
        self._update(deterministic=deterministic, seed=seed)

    # -- Grounding --------------------------------------------------------------------------------

    def set_grounding(self, *, bounding_boxes: bool, confidence: bool) -> None:
        self._update(emit_bounding_boxes=bounding_boxes, emit_confidence=confidence)

    # -- Output -----------------------------------------------------------------------------------

    def set_output_format(self, output_format: OutputFormat) -> None:
        self._update(output_format=output_format)

    # -- Display ----------------------------------------------------------------------------------

    def visible_fields(self) -> tuple[SettingField, ...]:
        """Only the groups this provider's capabilities actually support (Part 9's core rule) —
        a generic settings View iterates this instead of hardcoding a fixed field list."""
        config = self.configuration()
        fields: list[SettingField] = []
        if self.capabilities.has_device_selection:
            fields.append(SettingField(group="Device", label="Device", value=config.device.value))
        if self.capabilities.has_precision_selection:
            fields.append(SettingField(group="Precision", label="Precision", value=config.precision.value))
        if self.capabilities.has_batching:
            fields.append(SettingField(group="Performance", label="Batch size", value=str(config.batch_size)))
            fields.append(
                SettingField(group="Performance", label="Parallel documents", value=str(config.parallel_documents))
            )
            fields.append(SettingField(group="Performance", label="Worker threads", value=str(config.worker_threads)))
        if self.capabilities.has_image_processing:
            fields.append(
                SettingField(group="Image Processing", label="Image resolution", value=str(config.image_resolution or "auto"))
            )
            fields.append(SettingField(group="Image Processing", label="Max tokens", value=str(config.max_tokens or "auto")))
        if self.capabilities.has_prompt_management:
            fields.append(
                SettingField(
                    group="Prompt", label="Template",
                    value=f"{config.prompt_template_name}@{config.prompt_template_version}",
                )
            )
        if self.capabilities.has_determinism_controls:
            fields.append(
                SettingField(group="Determinism", label="Deterministic mode", value=str(config.deterministic))
            )
            if config.deterministic:
                fields.append(SettingField(group="Determinism", label="Seed", value=str(config.seed)))
        if self.capabilities.has_grounding:
            fields.append(SettingField(group="Grounding", label="Bounding boxes", value=str(config.emit_bounding_boxes)))
            fields.append(SettingField(group="Grounding", label="Confidence", value=str(config.emit_confidence)))
        fields.append(
            SettingField(group="Output", label="Format", value=config.output_format.value)
        )  # every provider has an output shape, even if just plain text
        return tuple(fields)
