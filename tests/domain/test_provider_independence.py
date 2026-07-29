"""Contract test for Constitution Article 20 / Article 9 (S3.6): nothing downstream of the
Comparison Engine's output may carry a provider_id or other provider-native field.
"""

from __future__ import annotations

from archivetrust.domain.canonical.observation import CanonicalObservation
from archivetrust.domain.document.canonical_document import CanonicalDocument
from archivetrust.domain.graph.reconciled_graph import ReconciledObservationGraph

PROVIDER_LEAK_FIELD_NAMES = {"provider_id", "provider_version", "provider", "native_label_source"}


def test_canonical_observation_carries_no_provider_identity_field():
    assert set(CanonicalObservation.model_fields) & PROVIDER_LEAK_FIELD_NAMES == set()


def test_reconciled_observation_graph_carries_no_provider_identity_field():
    assert set(ReconciledObservationGraph.model_fields) & PROVIDER_LEAK_FIELD_NAMES == set()


def test_canonical_document_carries_no_provider_identity_field():
    assert set(CanonicalDocument.model_fields) & PROVIDER_LEAK_FIELD_NAMES == set()
