"""Provider adapter conformance checks for SDK/CI use."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict

from archivetrust.providers.base import ProbabilisticProviderAdapter, ProviderAdapter


class ConformanceStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"


class ProviderConformanceCheck(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    status: ConformanceStatus
    detail: str


class ProviderConformanceReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider_id: str
    provider_version: str | None
    passed: bool
    checks: tuple[ProviderConformanceCheck, ...]

    @property
    def failures(self) -> tuple[ProviderConformanceCheck, ...]:
        return tuple(check for check in self.checks if check.status is ConformanceStatus.FAIL)


def run_provider_conformance(
    adapter: ProviderAdapter,
    *,
    document_ref: str,
    invocation_id: str,
    source: Any,
) -> ProviderConformanceReport:
    """Run the core provider contract checks without raising on failures."""
    checks: list[ProviderConformanceCheck] = []
    checks.extend(_registration_checks(adapter))

    result = adapter.observe(document_ref=document_ref, invocation_id=invocation_id, source=source)
    provider_version = result.attempt.provider_version

    _check(
        checks,
        "attempt-provider",
        result.attempt.provider_id == adapter.provider_id,
        f"attempt provider_id={result.attempt.provider_id!r}, adapter provider_id={adapter.provider_id!r}",
    )
    _check(
        checks,
        "attempt-invocation",
        result.attempt.invocation_id == invocation_id,
        f"attempt invocation_id={result.attempt.invocation_id!r}, expected {invocation_id!r}",
    )
    _check(checks, "attempt-version", bool(result.attempt.provider_version), "provider_version must be non-empty")

    evidence_ids = {evidence.evidence_id for evidence in result.evidence}
    for index, evidence in enumerate(result.evidence):
        _check(
            checks,
            f"evidence-{index}-provider",
            evidence.provider == adapter.provider_id,
            f"evidence provider={evidence.provider!r}, adapter provider_id={adapter.provider_id!r}",
        )
        _check(
            checks,
            f"evidence-{index}-version",
            evidence.provider_version == result.attempt.provider_version,
            (
                f"evidence provider_version={evidence.provider_version!r}, "
                f"attempt provider_version={result.attempt.provider_version!r}"
            ),
        )
        if isinstance(adapter, ProbabilisticProviderAdapter):
            _check(
                checks,
                f"evidence-{index}-prompt",
                evidence.prompt is not None and evidence.prompt != "",
                "probabilistic provider evidence must carry prompt provenance",
            )

    for index, observation in enumerate(result.observations):
        _check(
            checks,
            f"observation-{index}-provider",
            observation.provider_id == adapter.provider_id,
            f"observation provider_id={observation.provider_id!r}, adapter provider_id={adapter.provider_id!r}",
        )
        _check(
            checks,
            f"observation-{index}-version",
            observation.provider_version == result.attempt.provider_version,
            (
                f"observation provider_version={observation.provider_version!r}, "
                f"attempt provider_version={result.attempt.provider_version!r}"
            ),
        )
        _check(
            checks,
            f"observation-{index}-evidence-closure",
            set(observation.evidence_ids) <= evidence_ids,
            "observation evidence_ids must refer to evidence returned by the same invocation",
        )
        _check(
            checks,
            f"observation-{index}-scope",
            observation.scope is not None,
            "F5 requires every observation to carry a scope measurement",
        )

    passed = all(check.status is ConformanceStatus.PASS for check in checks)
    return ProviderConformanceReport(
        provider_id=adapter.provider_id,
        provider_version=provider_version,
        passed=passed,
        checks=tuple(checks),
    )


def _registration_checks(adapter: ProviderAdapter) -> tuple[ProviderConformanceCheck, ...]:
    checks: list[ProviderConformanceCheck] = []
    registration = adapter.registration
    _check(
        checks,
        "registration-provider-id",
        bool(registration.provider_id) and registration.provider_id == adapter.provider_id,
        "registration provider_id must be non-empty and match adapter.provider_id",
    )
    _check(checks, "registration-reproducibility", registration.reproducibility is not None, "axis must be declared")
    _check(checks, "registration-locality", registration.deployment_locality is not None, "axis must be declared")
    _check(checks, "registration-introspectability", registration.introspectability is not None, "axis must be declared")
    _check(checks, "registration-profile", registration.deployment_profile is not None, "deployment profile must be declared")
    _check(checks, "input-kind", adapter.input_kind is not None, "provider input kind must be declared")
    return tuple(checks)


def _check(checks: list[ProviderConformanceCheck], name: str, condition: bool, detail: str) -> None:
    checks.append(
        ProviderConformanceCheck(
            name=name,
            status=ConformanceStatus.PASS if condition else ConformanceStatus.FAIL,
            detail=detail,
        )
    )
