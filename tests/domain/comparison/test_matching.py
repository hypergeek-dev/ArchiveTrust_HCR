from __future__ import annotations

from archivetrust.domain.comparison.capability_matrix import (
    Capability,
    CapabilityMatrix,
    CapabilityMatrixEntry,
)
from archivetrust.domain.comparison.clustering import Cluster
from archivetrust.domain.comparison.matching import MatchCardinality, match_cluster
from archivetrust.domain.evidence.models import Evidence, ProcessingStage
from archivetrust.domain.ontology.base import Observation
from archivetrust.domain.ontology.payloads import HeadingPayload
from archivetrust.domain.ontology.types import ObservationType


def _obs(provider: str, text: str) -> Observation:
    ev = Evidence.create(provider=provider, provider_version="1.0", raw_output=text, processing_stage=ProcessingStage.OCR)
    return Observation.from_evidence(provider_id=provider, provider_version="1.0", payload=HeadingPayload(text=text, level=1), evidence=(ev,))


def _cluster(*observations: Observation) -> Cluster:
    return Cluster(
        cluster_id="cluster-1",
        observation_type=ObservationType.HEADING,
        page=1,
        member_observation_ids=tuple(o.observation_id for o in observations),
        clustering_basis="test",
    )


def _matrix(*, tesseract_capable: bool) -> CapabilityMatrix:
    entries = [
        CapabilityMatrixEntry(provider_id="docling", provider_version="1.0", observation_type=ObservationType.HEADING, capability=Capability.NATIVE)
    ]
    if tesseract_capable:
        entries.append(
            CapabilityMatrixEntry(provider_id="tesseract", provider_version="1.0", observation_type=ObservationType.HEADING, capability=Capability.PARTIAL)
        )
    return CapabilityMatrix(matrix_version=1, entries=tuple(entries))


def test_one_to_one_when_provider_contributes_exactly_one_observation():
    docling_obs = _obs("docling", "Chapter 1")
    cluster = _cluster(docling_obs)
    observations_by_id = {docling_obs.observation_id: docling_obs}
    contributions = match_cluster(cluster, observations_by_id, (("docling", "1.0"),), _matrix(tesseract_capable=False))
    assert contributions[0].cardinality == MatchCardinality.ONE_TO_ONE


def test_many_to_one_when_provider_over_segments():
    docling_a = _obs("docling", "Chapter")
    docling_b = _obs("docling", "1")
    cluster = _cluster(docling_a, docling_b)
    observations_by_id = {docling_a.observation_id: docling_a, docling_b.observation_id: docling_b}
    contributions = match_cluster(cluster, observations_by_id, (("docling", "1.0"),), _matrix(tesseract_capable=False))
    assert contributions[0].cardinality == MatchCardinality.MANY_TO_ONE


def test_missing_when_capable_provider_contributes_nothing():
    docling_obs = _obs("docling", "Chapter 1")
    cluster = _cluster(docling_obs)
    observations_by_id = {docling_obs.observation_id: docling_obs}
    contributions = match_cluster(
        cluster, observations_by_id, (("docling", "1.0"), ("tesseract", "1.0")), _matrix(tesseract_capable=True)
    )
    tesseract_contribution = next(c for c in contributions if c.provider_id == "tesseract")
    assert tesseract_contribution.cardinality == MatchCardinality.MISSING


def test_not_attempted_when_provider_has_no_capability_for_this_type():
    docling_obs = _obs("docling", "Chapter 1")
    cluster = _cluster(docling_obs)
    observations_by_id = {docling_obs.observation_id: docling_obs}
    contributions = match_cluster(
        cluster, observations_by_id, (("docling", "1.0"), ("tesseract", "1.0")), _matrix(tesseract_capable=False)
    )
    tesseract_contribution = next(c for c in contributions if c.provider_id == "tesseract")
    assert tesseract_contribution.cardinality == MatchCardinality.NOT_ATTEMPTED


def test_missing_and_not_attempted_are_distinguishable_constitution_article_18():
    # The load-bearing invariant: a capability gap must never look identical to unexplained
    # silence.
    docling_obs = _obs("docling", "Chapter 1")
    cluster = _cluster(docling_obs)
    observations_by_id = {docling_obs.observation_id: docling_obs}
    contributions = match_cluster(
        cluster, observations_by_id, (("docling", "1.0"), ("tesseract", "1.0")), _matrix(tesseract_capable=True)
    )
    tesseract_contribution = next(c for c in contributions if c.provider_id == "tesseract")
    assert tesseract_contribution.cardinality == MatchCardinality.MISSING
    assert tesseract_contribution.cardinality != MatchCardinality.NOT_ATTEMPTED
