"""ProviderRegistry (ROADMAP.md S5.7: "providers register through a ProviderRegistry. Adding a
fourth provider must require zero changes to the Evidence Model, Observation Layer, Comparison
Engine, or Canonical Document Model.") Deliberately trivial -- a keyed lookup, nothing more. Any
provider-specific behavior belongs in that provider's own adapter module, never here.
"""

from __future__ import annotations

from archivetrust.providers.base import DeploymentProfile, ProviderAdapter


class DuplicateProviderError(KeyError):
    """Raised when two adapters register the same `(provider_id, deployment_profile)` pair --
    silently overwriting a registration would make "which adapter actually ran" ambiguous,
    undermining Evidence Traceability at the source. The same `provider_id` registered under two
    *different* `DeploymentProfile`s is not a collision (Phase 31) -- that is the whole point of the
    profile axis: one vendor system, more than one deployment shape, each independently registered.
    """


class UnknownProviderError(KeyError):
    pass


class ReservedProviderIdentityError(ValueError):
    """Evaluation-reference identities can never enter production provider composition."""


_RESERVED_EVALUATION_IDENTITIES = frozenset({"ground_truth", "evaluation_reference"})


class ProviderRegistry:
    """Keyed on `(provider_id, deployment_profile)` (Phase 31), not bare `provider_id` -- widened so
    one provider (e.g. PaddleOCR-VL) can register more than one deployment profile (raw single-pass
    vs. structured pipeline) without inventing a second, synthetic `provider_id` to dodge the
    uniqueness constraint. Every pre-Phase-31 caller that only ever knew a bare `provider_id` keeps
    working unchanged: `get()`'s `deployment_profile` parameter defaults to `SINGLE_PASS`, which is
    exactly what every provider registered before this phase already is.
    """

    def __init__(self) -> None:
        self._adapters: dict[tuple[str, DeploymentProfile], ProviderAdapter] = {}

    def register(self, adapter: ProviderAdapter) -> None:
        if adapter.provider_id in _RESERVED_EVALUATION_IDENTITIES:
            raise ReservedProviderIdentityError(
                f"{adapter.provider_id!r} is reserved for non-provider evaluation data"
            )
        key = (adapter.provider_id, adapter.registration.deployment_profile)
        if key in self._adapters:
            raise DuplicateProviderError(key)
        self._adapters[key] = adapter

    def get(
        self, provider_id: str, deployment_profile: DeploymentProfile = DeploymentProfile.SINGLE_PASS
    ) -> ProviderAdapter:
        try:
            return self._adapters[(provider_id, deployment_profile)]
        except KeyError:
            raise UnknownProviderError((provider_id, deployment_profile)) from None

    def all(self) -> tuple[ProviderAdapter, ...]:
        return tuple(self._adapters.values())
