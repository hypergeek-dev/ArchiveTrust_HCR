from __future__ import annotations

import json

from archivetrust.domain.evidence.models import Evidence, ProcessingStage
from archivetrust.domain.ontology.base import Observation
from archivetrust.domain.ontology.payloads import ParagraphPayload
from archivetrust.providers.base import (
    DeploymentLocality,
    DeterministicProviderAdapter,
    Introspectability,
    ProviderAttempt,
    ProviderRegistration,
    ProviderRunResult,
    Reproducibility,
)
from archivetrust.providers.conformance import ConformanceStatus, run_provider_conformance


class _PassingAdapter(DeterministicProviderAdapter):
    registration = ProviderRegistration(
        provider_id="passing",
        reproducibility=Reproducibility.DETERMINISTIC,
        deployment_locality=DeploymentLocality.IN_PROCESS,
        introspectability=Introspectability.OPEN,
    )

    def observe(self, *, document_ref: str, invocation_id: str, source: object) -> ProviderRunResult:
        evidence = Evidence.create(
            provider=self.provider_id,
            provider_version="1.0",
            raw_output=json.dumps({"text": "hello"}),
            processing_stage=ProcessingStage.RAW,
        )
        observation = Observation.from_evidence(
            provider_id=self.provider_id,
            provider_version="1.0",
            payload=ParagraphPayload(text="hello"),
            evidence=(evidence,),
        )
        return ProviderRunResult(
            attempt=ProviderAttempt(provider_id=self.provider_id, provider_version="1.0", invocation_id=invocation_id),
            evidence=(evidence,),
            observations=(observation,),
        )


class _BadAttemptAdapter(_PassingAdapter):
    registration = ProviderRegistration(
        provider_id="bad-attempt",
        reproducibility=Reproducibility.DETERMINISTIC,
        deployment_locality=DeploymentLocality.IN_PROCESS,
        introspectability=Introspectability.OPEN,
    )

    def observe(self, *, document_ref: str, invocation_id: str, source: object) -> ProviderRunResult:
        result = super().observe(document_ref=document_ref, invocation_id=invocation_id, source=source)
        return result.model_copy(
            update={
                "attempt": ProviderAttempt(
                    provider_id="someone-else",
                    provider_version="",
                    invocation_id="wrong",
                )
            }
        )


def test_conformance_passes_for_valid_adapter() -> None:
    report = run_provider_conformance(
        _PassingAdapter(), document_ref="doc-1", invocation_id="inv-1", source="source"
    )

    assert report.passed is True
    assert report.failures == ()


def test_conformance_reports_failures_without_raising() -> None:
    report = run_provider_conformance(
        _BadAttemptAdapter(), document_ref="doc-1", invocation_id="inv-1", source="source"
    )

    assert report.passed is False
    assert {failure.name for failure in report.failures} >= {
        "attempt-provider",
        "attempt-invocation",
        "attempt-version",
        "observation-0-version",
        "evidence-0-version",
    }
    assert all(failure.status is ConformanceStatus.FAIL for failure in report.failures)
