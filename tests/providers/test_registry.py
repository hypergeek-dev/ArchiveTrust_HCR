from __future__ import annotations

import pytest

from archivetrust.providers.base import (
    DeploymentLocality,
    DeterministicProviderAdapter,
    Introspectability,
    ProviderAttempt,
    ProviderRegistration,
    ProviderRunResult,
    Reproducibility,
)
from archivetrust.providers.registry import (
    DuplicateProviderError,
    ProviderRegistry,
    ReservedProviderIdentityError,
    UnknownProviderError,
)


class _StubAdapter(DeterministicProviderAdapter):
    registration = ProviderRegistration(
        provider_id="stub",
        reproducibility=Reproducibility.DETERMINISTIC,
        deployment_locality=DeploymentLocality.IN_PROCESS,
        introspectability=Introspectability.OPEN,
    )

    def observe(self, *, document_ref: str, invocation_id: str, source: object) -> ProviderRunResult:
        attempt = ProviderAttempt(provider_id=self.provider_id, provider_version="1.0", invocation_id=invocation_id)
        return ProviderRunResult(attempt=attempt)


def test_register_and_get():
    registry = ProviderRegistry()
    adapter = _StubAdapter()
    registry.register(adapter)
    assert registry.get("stub") is adapter


def test_duplicate_registration_rejected():
    registry = ProviderRegistry()
    registry.register(_StubAdapter())
    with pytest.raises(DuplicateProviderError):
        registry.register(_StubAdapter())


def test_unknown_provider_raises():
    registry = ProviderRegistry()
    with pytest.raises(UnknownProviderError):
        registry.get("nonexistent")


def test_all_returns_every_registered_adapter():
    registry = ProviderRegistry()
    adapter = _StubAdapter()
    registry.register(adapter)
    assert registry.all() == (adapter,)


@pytest.mark.parametrize("provider_id", ("ground_truth", "evaluation_reference"))
def test_evaluation_reference_identities_are_rejected_from_provider_composition(provider_id):
    class _ContaminatingAdapter(_StubAdapter):
        registration = _StubAdapter.registration.model_copy(update={"provider_id": provider_id})

    with pytest.raises(ReservedProviderIdentityError, match="non-provider evaluation data"):
        ProviderRegistry().register(_ContaminatingAdapter())
