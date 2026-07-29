"""Phase 13 (`ROADMAP_TELEMETRY_STANDARD.md`): the Adversarial Architecture Audit's falsification,
re-expressed as a permanent, CI-gated regression (Constitution Article 31).

The Adversarial Audit (`docs/ADVERSARIAL_AUDIT_WHAT_DID_WE_REMOVE_2026-07-14.md`) falsified the RCDP
counterfactual's proposed geometry-precision guard: 60% of a random sample of what it would
eliminate carried unique, high-value PaddleOCR-VL content -- a single whole-page transcript with no
bounding-box geometry, clustered via `ClusteringBasisCode.ORDINAL_POSITION_FALLBACK` against a
pixel-accurate Docling fragment on the same page. The guard was correctly never merged into
`clustering.py` (`ARCHITECTURE_TELEMETRY_STANDARD.md` §1.1, category 2/3) -- it exists only as an
alternate `AlignmentService` in `scripts/rcdp_ordinal_geometry_counterfactual.py`.

Article 31 requires this stay a *standing* check, not a one-time audit: "no future
clustering/comparison change may ship without this check passing." Re-running the full falsification
against the real ~977MB corpus is not CI-appropriate (it needs `archivetrust_data/`, which CI does
not have, and takes minutes even locally). This test instead encodes the exact mechanism as a fast,
deterministic fixture: one Docling Observation with pixel-accurate geometry and one PaddleOCR-VL
Observation with none, same type and page, each the only Observation its provider contributed to
that pool -- precisely the shape the real corpus's eliminated packets had. Today's real,
production `ClusteringAlignmentService` clusters them together via `ORDINAL_POSITION_FALLBACK`,
never excludes the pair. If a future change adds the geometry-precision guard (or any mechanism with
the same effect) to `clustering.py`, this test fails -- forcing an explicit decision (satisfy Article
31's audit for the new mechanism, or don't ship it), rather than silently shipping the change the
Adversarial Audit already found removes real information.
"""

from __future__ import annotations

from archivetrust.domain.alignment.service import ClusteringAlignmentService
from archivetrust.domain.comparison.clustering import ClusteringBasisCode
from archivetrust.domain.comparison.policy import ReconciliationPolicy

from tests.domain.alignment._helpers import box, evidence, observation, paragraph

DOC = "doc-1"


def test_geometry_precision_guard_is_not_shipped_docling_paddleocr_vl_still_cluster():
    docling_evidence = evidence("docling", "Section 3.2 Results", box(10, 10, 200, 30))
    docling_obs = observation("docling", paragraph("Section 3.2 Results"), docling_evidence)

    paddle_evidence = evidence(
        "paddleocr-vl", "Section 3.2 Results — full page transcript with additional detail", None
    )
    paddle_obs = observation(
        "paddleocr-vl",
        paragraph("Section 3.2 Results — full page transcript with additional detail"),
        paddle_evidence,
    )

    evidence_by_id = {docling_evidence.evidence_id: docling_evidence, paddle_evidence.evidence_id: paddle_evidence}
    result = ClusteringAlignmentService().align(
        (docling_obs, paddle_obs), evidence_by_id, ReconciliationPolicy(policy_version=1)
    )

    assert result.excluded_pairs == (), (
        "a geometry-precision guard appears to have shipped: this Docling/PaddleOCR-VL pair was "
        "structurally excluded, exactly what the Adversarial Audit falsified as a general-purpose "
        "fix (60% of a real sample carried unique content this would make permanently "
        "unreachable). Do not ship this change without first satisfying Article 31's audit "
        "requirement for it, recorded as a new AuditConducted event."
    )
    clusters_with_both = [
        c
        for c in result.clusters
        if docling_obs.observation_id in c.member_observation_ids
        and paddle_obs.observation_id in c.member_observation_ids
    ]
    assert len(clusters_with_both) == 1
    edge = clusters_with_both[0].affinity_edges[0]
    assert edge.basis_code == ClusteringBasisCode.ORDINAL_POSITION_FALLBACK
