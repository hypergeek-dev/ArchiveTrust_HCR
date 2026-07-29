from __future__ import annotations

from archivetrust.domain.comparison.capability_matrix import (
    Capability,
    CapabilityMatrix,
    CapabilityMatrixEntry,
)
from archivetrust.domain.comparison.clustering import Cluster
from archivetrust.domain.comparison.policy import ReconciliationPolicy
from archivetrust.domain.comparison.structural import reconcile_structure
from archivetrust.domain.evidence.models import Evidence, ProcessingStage
from archivetrust.domain.ontology.base import Observation
from archivetrust.domain.ontology.payloads import HeadingPayload, ParagraphPayload
from archivetrust.domain.ontology.types import ObservationType


def _policy() -> ReconciliationPolicy:
    return ReconciliationPolicy(policy_version=1, edge_accept_threshold=0.5)


def _matrix() -> CapabilityMatrix:
    return CapabilityMatrix(
        matrix_version=1,
        entries=(
            CapabilityMatrixEntry(
                provider_id="docling", provider_version="1.0", observation_type=ObservationType.HEADING, capability=Capability.NATIVE
            ),
        ),
    )


def _obs(provider: str, payload, text: str, child_ids: tuple[str, ...] = ()) -> Observation:
    ev = Evidence.create(provider=provider, provider_version="1.0", raw_output=text, processing_stage=ProcessingStage.OCR)
    return Observation.from_evidence(
        provider_id=provider, provider_version="1.0", payload=payload, evidence=(ev,), child_observations=child_ids
    )


def test_edge_elected_when_capability_weighted_support_exceeds_threshold():
    para = _obs("docling", ParagraphPayload(text="body"), "body")
    heading = _obs("docling", HeadingPayload(text="Ch 1", level=1), "Ch 1", child_ids=(para.observation_id,))

    heading_cluster = Cluster(
        cluster_id="cluster_heading",
        observation_type=ObservationType.HEADING,
        page=1,
        member_observation_ids=(heading.observation_id,),
        clustering_basis="test",
    )
    para_cluster = Cluster(
        cluster_id="cluster_para",
        observation_type=ObservationType.PARAGRAPH,
        page=1,
        member_observation_ids=(para.observation_id,),
        clustering_basis="test",
    )

    result = reconcile_structure(
        (heading_cluster, para_cluster),
        {heading.observation_id: heading, para.observation_id: para},
        _matrix(),
        _policy(),
    )

    assert len(result.edges) == 1
    edge = result.edges[0]
    assert edge.elected
    assert edge.parent_cluster_id == "cluster_heading"
    assert edge.child_cluster_id == "cluster_para"


def test_single_source_structural_claim_is_uncorroborated_not_elected_confidently():
    para = _obs("docling", ParagraphPayload(text="body"), "body")
    heading = _obs("docling", HeadingPayload(text="Ch 1", level=1), "Ch 1", child_ids=(para.observation_id,))
    heading_cluster = Cluster(
        cluster_id="cluster_heading", observation_type=ObservationType.HEADING, page=1,
        member_observation_ids=(heading.observation_id,), clustering_basis="test",
    )
    para_cluster = Cluster(
        cluster_id="cluster_para", observation_type=ObservationType.PARAGRAPH, page=1,
        member_observation_ids=(para.observation_id,), clustering_basis="test",
    )
    result = reconcile_structure(
        (heading_cluster, para_cluster), {heading.observation_id: heading, para.observation_id: para}, _matrix(), _policy()
    )
    edge = result.edges[0]
    assert edge.comparison_confidence.classification.value == "uncorroborated_single_source"
    assert edge.comparison_confidence.magnitude is None


def test_cycle_is_broken_deterministically_and_logged():
    obs_a = _obs("docling", HeadingPayload(text="A", level=1), "A")
    obs_b = _obs("docling", HeadingPayload(text="B", level=1), "B", child_ids=(obs_a.observation_id,))
    obs_a = obs_a.model_copy(update={"child_observations": (obs_b.observation_id,)})

    cluster_a = Cluster(
        cluster_id="cluster_a", observation_type=ObservationType.HEADING, page=1,
        member_observation_ids=(obs_a.observation_id,), clustering_basis="test",
    )
    cluster_b = Cluster(
        cluster_id="cluster_b", observation_type=ObservationType.HEADING, page=1,
        member_observation_ids=(obs_b.observation_id,), clustering_basis="test",
    )
    result = reconcile_structure(
        (cluster_a, cluster_b), {obs_a.observation_id: obs_a, obs_b.observation_id: obs_b}, _matrix(), _policy()
    )
    assert len(result.dropped_for_cycle) == 1
    elected = [e for e in result.edges if e.elected]
    assert len(elected) == 1


def test_reading_order_aggregation_is_deterministic():
    child_1 = _obs("docling", ParagraphPayload(text="first"), "first")
    child_2 = _obs("docling", ParagraphPayload(text="second"), "second")
    parent = _obs(
        "docling", HeadingPayload(text="Ch 1", level=1), "Ch 1",
        child_ids=(child_1.observation_id, child_2.observation_id),
    )
    parent_cluster = Cluster(
        cluster_id="cluster_parent", observation_type=ObservationType.HEADING, page=1,
        member_observation_ids=(parent.observation_id,), clustering_basis="test",
    )
    cluster_1 = Cluster(
        cluster_id="cluster_1", observation_type=ObservationType.PARAGRAPH, page=1,
        member_observation_ids=(child_1.observation_id,), clustering_basis="test",
    )
    cluster_2 = Cluster(
        cluster_id="cluster_2", observation_type=ObservationType.PARAGRAPH, page=1,
        member_observation_ids=(child_2.observation_id,), clustering_basis="test",
    )
    observations_by_id = {
        parent.observation_id: parent, child_1.observation_id: child_1, child_2.observation_id: child_2
    }
    result = reconcile_structure((parent_cluster, cluster_1, cluster_2), observations_by_id, _matrix(), _policy())
    assert result.reading_order["cluster_parent"] == ("cluster_1", "cluster_2")
