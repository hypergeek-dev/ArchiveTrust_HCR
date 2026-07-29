from __future__ import annotations

import pytest
from pydantic import ValidationError

from archivetrust.domain.evidence.models import Evidence, ProcessingStage
from archivetrust.domain.graph.provider_graph import ProviderObservationGraph
from archivetrust.domain.ontology.base import Observation
from archivetrust.domain.ontology.payloads import HeadingPayload, ParagraphPayload


def _evidence(provider: str, version: str, text: str) -> Evidence:
    return Evidence.create(
        provider=provider, provider_version=version, raw_output=text, processing_stage=ProcessingStage.OCR
    )


def test_graph_accepts_observations_from_one_provider_invocation():
    heading = Observation.from_evidence(
        provider_id="docling",
        provider_version="1.0",
        payload=HeadingPayload(text="Ch 1", level=1),
        evidence=(_evidence("docling", "1.0", "Ch 1"),),
    )
    para = Observation.from_evidence(
        provider_id="docling",
        provider_version="1.0",
        payload=ParagraphPayload(text="body"),
        evidence=(_evidence("docling", "1.0", "body"),),
        parent_observations=(heading.observation_id,),
    )
    heading = heading.model_copy(update={"child_observations": (para.observation_id,)})

    graph = ProviderObservationGraph(
        provider_id="docling",
        provider_version="1.0",
        invocation_id="run-1",
        observations=(heading, para),
    )
    assert len(graph.observations) == 2


def test_graph_rejects_observations_from_a_different_provider():
    docling_obs = Observation.from_evidence(
        provider_id="docling",
        provider_version="1.0",
        payload=HeadingPayload(text="Ch 1", level=1),
        evidence=(_evidence("docling", "1.0", "Ch 1"),),
    )
    tesseract_obs = Observation.from_evidence(
        provider_id="tesseract",
        provider_version="5.3",
        payload=ParagraphPayload(text="body"),
        evidence=(_evidence("tesseract", "5.3", "body"),),
    )
    with pytest.raises(ValidationError):
        ProviderObservationGraph(
            provider_id="docling",
            provider_version="1.0",
            invocation_id="run-1",
            observations=(docling_obs, tesseract_obs),
        )


def test_graph_rejects_dangling_references():
    docling_obs = Observation.from_evidence(
        provider_id="docling",
        provider_version="1.0",
        payload=HeadingPayload(text="Ch 1", level=1),
        evidence=(_evidence("docling", "1.0", "Ch 1"),),
        child_observations=("observation_does_not_exist",),
    )
    with pytest.raises(ValidationError):
        ProviderObservationGraph(
            provider_id="docling",
            provider_version="1.0",
            invocation_id="run-1",
            observations=(docling_obs,),
        )


def test_graph_rejects_cycles():
    a = Observation.from_evidence(
        provider_id="docling",
        provider_version="1.0",
        payload=HeadingPayload(text="A", level=1),
        evidence=(_evidence("docling", "1.0", "A"),),
    )
    b = Observation.from_evidence(
        provider_id="docling",
        provider_version="1.0",
        payload=HeadingPayload(text="B", level=1),
        evidence=(_evidence("docling", "1.0", "B"),),
    )
    a = a.model_copy(update={"child_observations": (b.observation_id,)})
    b = b.model_copy(update={"child_observations": (a.observation_id,)})
    with pytest.raises(ValidationError):
        ProviderObservationGraph(
            provider_id="docling",
            provider_version="1.0",
            invocation_id="run-1",
            observations=(a, b),
        )
