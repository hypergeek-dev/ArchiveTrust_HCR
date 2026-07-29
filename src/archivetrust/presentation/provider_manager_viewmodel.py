"""Provider Manager ViewModel (Part 5).

Assembles one row per registered provider from three independent sources — the existing
`ProviderRegistry` (Trust Engine identity: `provider_id`, the three registration axes), the
`ProviderConfigurationStore` (Part 8/9: enabled, profile), and a per-provider `ProviderHealthTracker`
(Part 6) — without modifying any of them. This is the "combine, don't own" pattern the whole
Provider Manager rests on: none of the three sources knows the other two exist.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from archivetrust.providers.base import ProviderAdapter
from archivetrust.providers.registry import ProviderRegistry
from archivetrust.runtime.model_registry import ModelRegistry, UnknownLogicalModelError
from archivetrust.runtime.provider_config import (
    DETERMINISTIC_LOCAL_CAPABILITIES,
    PROBABILISTIC_VISION_CAPABILITIES,
    ProviderConfigurationStore,
    ProviderSettingCapabilities,
)
from archivetrust.runtime.provider_health import ProviderHealthSnapshot, ProviderHealthTracker


class ProviderManagerEntry(BaseModel):
    """One row: everything Part 5 asks the Provider Manager to display for one provider."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    provider_id: str
    installed: bool
    enabled: bool
    version: str | None
    """The provider's registration identity; providers don't carry a bare version string beyond
    what a runtime/Evidence reports at inference time (S5.5) -- shown as the health snapshot's
    `model_revision` once available, `None` before any inference has run."""
    backend: str
    """"deterministic" / "probabilistic" (the provider's registered Reproducibility axis, S5.5) —
    provider-identity information, distinct from the runtime `backend` string in `health`."""
    runtime_kind: str | None
    """Which `InferenceRuntime` this provider is currently bound to via the Model Registry, if
    any — `None` for a provider with no runtime binding (e.g. Docling, Tesseract)."""
    status: str
    health: ProviderHealthSnapshot
    capabilities: ProviderSettingCapabilities


def capabilities_for_provider(adapter: ProviderAdapter) -> ProviderSettingCapabilities:
    # Grounded in the provider's own registered Reproducibility axis (S5.5) -- never a name-based
    # special case (Guiding Principle 3): a Probabilistic provider is prompt/runtime-driven and
    # gets the full Part 9 surface; a Deterministic provider does not.
    from archivetrust.providers.base import Reproducibility

    if adapter.registration.reproducibility == Reproducibility.PROBABILISTIC:
        return PROBABILISTIC_VISION_CAPABILITIES
    return DETERMINISTIC_LOCAL_CAPABILITIES


def _status_for(
    *, adapter: ProviderAdapter, enabled: bool, available: bool, running: bool, ready: bool
) -> str:
    """The five-way status vocabulary the Operational Completion milestone asks for. Deterministic
    providers (Docling, Tesseract) have no separate runtime warm-up step -- once their real client
    is available, they are simply Ready; probabilistic providers additionally require the runtime
    to actually be `ready` (a real model bound and warmed up), so "available but never bound/warmed
    up" reports as a Configuration Error rather than a false "Ready."
    """
    if not enabled:
        return "Disabled"
    if not available:
        return "Unavailable"
    if running:
        return "Running"
    from archivetrust.providers.base import Reproducibility

    if adapter.registration.reproducibility == Reproducibility.PROBABILISTIC and not ready:
        return "Configuration Error"
    return "Ready"


class ProviderManagerViewModel:
    def __init__(
        self,
        *,
        provider_registry: ProviderRegistry,
        configuration_store: ProviderConfigurationStore,
        health_trackers: dict[str, ProviderHealthTracker],
        model_registry: ModelRegistry | None = None,
        availability: dict[str, bool] | None = None,
        running_provider_ids: set[str] | None = None,
    ) -> None:
        self._providers = provider_registry
        self._config = configuration_store
        self._health = health_trackers
        self._models = model_registry
        self._availability = availability or {}
        self._running = running_provider_ids or set()

    def entries(self) -> tuple[ProviderManagerEntry, ...]:
        rows = []
        for adapter in self._providers.all():
            config = self._config.get(adapter.provider_id)
            tracker = self._health.get(adapter.provider_id)
            snapshot = tracker.snapshot() if tracker is not None else ProviderHealthTracker(
                adapter.provider_id
            ).snapshot()

            runtime_kind = None
            if self._models is not None:
                try:
                    runtime_kind = self._models.resolve(adapter.provider_id).descriptor.runtime_kind
                except UnknownLogicalModelError:
                    runtime_kind = None

            rows.append(
                ProviderManagerEntry(
                    provider_id=adapter.provider_id,
                    installed=snapshot.installed,
                    enabled=config.enabled,
                    version=snapshot.model_revision,
                    backend=adapter.registration.reproducibility.value,
                    runtime_kind=runtime_kind,
                    status=_status_for(
                        adapter=adapter,
                        enabled=config.enabled,
                        available=self._availability.get(adapter.provider_id, False),
                        running=adapter.provider_id in self._running,
                        ready=snapshot.ready,
                    ),
                    health=snapshot,
                    capabilities=capabilities_for_provider(adapter),
                )
            )
        rows.sort(key=lambda r: r.provider_id)
        return tuple(rows)

    def set_enabled(self, provider_id: str, enabled: bool) -> None:
        current = self._config.get(provider_id)
        self._config.set(current.model_copy(update={"enabled": enabled}))
