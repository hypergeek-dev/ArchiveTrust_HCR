from __future__ import annotations

import pytest

from archivetrust.domain.confidence.models import (
    ComparisonClassification,
    ComparisonConfidence,
)
from archivetrust.domain.evidence.models import Evidence, ProcessingStage
from archivetrust.domain.ontology.base import Observation
from archivetrust.domain.ontology.payloads import HeadingPayload, ParagraphPayload


@pytest.fixture
def heading_evidence() -> Evidence:
    return Evidence.create(
        provider="docling",
        provider_version="1.0",
        raw_output="Chapter 1: Origins",
        processing_stage=ProcessingStage.LAYOUT_ANALYSIS,
        page=1,
        provider_confidence=0.92,
    )


@pytest.fixture
def heading_observation(heading_evidence: Evidence) -> Observation:
    return Observation.from_evidence(
        provider_id="docling",
        provider_version="1.0",
        payload=HeadingPayload(text="Chapter 1: Origins", level=1),
        evidence=(heading_evidence,),
    )


@pytest.fixture
def paragraph_evidence() -> Evidence:
    return Evidence.create(
        provider="tesseract",
        provider_version="5.3",
        raw_output="It was a dark and stormy night.",
        processing_stage=ProcessingStage.OCR,
        page=1,
        provider_confidence=0.75,
    )


@pytest.fixture
def paragraph_observation(paragraph_evidence: Evidence) -> Observation:
    return Observation.from_evidence(
        provider_id="tesseract",
        provider_version="5.3",
        payload=ParagraphPayload(text="It was a dark and stormy night."),
        evidence=(paragraph_evidence,),
    )


@pytest.fixture
def single_source_confidence() -> ComparisonConfidence:
    return ComparisonConfidence(
        classification=ComparisonClassification.UNCORROBORATED_SINGLE_SOURCE,
        magnitude=None,
        basis="only one provider attempted this slot",
    )
