from __future__ import annotations

import pytest

from archivetrust.domain.evidence.models import Evidence, ProcessingStage
from archivetrust.domain.ontology.base import Observation
from archivetrust.domain.ontology.migrations import (
    MigrationNotRegisteredError,
    MigrationRegistry,
)
from archivetrust.domain.ontology.payloads import HeadingPayload
from archivetrust.domain.ontology.types import ObservationType
from archivetrust.domain.shared.versioning import CURRENT_SCHEMA_VERSION


@pytest.fixture
def evidence() -> Evidence:
    return Evidence.create(
        provider="docling", provider_version="1.0", raw_output="x", processing_stage=ProcessingStage.OCR
    )


def _at_version(observation: Observation, ontology_version: int) -> Observation:
    return observation.model_copy(update={"ontology_version": ontology_version})


def test_migrate_raises_loudly_when_no_path_registered(evidence: Evidence):
    observation = Observation.from_evidence(
        provider_id="docling",
        provider_version="1.0",
        payload=HeadingPayload(text="Ch 1", level=1),
        evidence=(evidence,),
        ontology_version=0,
        schema_version=CURRENT_SCHEMA_VERSION,
    )
    registry = MigrationRegistry()
    with pytest.raises(MigrationNotRegisteredError):
        registry.migrate(observation)


def test_migrate_applies_a_registered_migration_path(evidence: Evidence):
    # Simulates HEADING going through one ontology revision (version 0 -> 1). The registered
    # migration is exercised end to end: an Observation stored at the older version is replayed
    # forward to CURRENT_ONTOLOGY_VERSION without manual intervention.
    observation = Observation.from_evidence(
        provider_id="docling",
        provider_version="1.0",
        payload=HeadingPayload(text="Ch 1", level=1),
        evidence=(evidence,),
        ontology_version=0,
        schema_version=CURRENT_SCHEMA_VERSION,
    )

    def heading_v0_to_v1(obs: Observation) -> Observation:
        return obs.model_copy(update={"ontology_version": obs.ontology_version + 1})

    registry = MigrationRegistry()
    registry.register(ObservationType.HEADING, 0, heading_v0_to_v1)

    migrated = registry.migrate(observation)
    assert migrated.ontology_version == 1
    assert migrated.payload == observation.payload


def test_migrate_chains_multiple_registered_steps(evidence: Evidence):
    observation = Observation.from_evidence(
        provider_id="docling",
        provider_version="1.0",
        payload=HeadingPayload(text="Ch 1", level=1),
        evidence=(evidence,),
        ontology_version=0,
        schema_version=CURRENT_SCHEMA_VERSION,
    )
    registry = MigrationRegistry()
    registry.register(
        ObservationType.HEADING, 0, lambda obs: obs.model_copy(update={"ontology_version": 1})
    )
    from archivetrust.domain.shared import versioning

    # Simulate a future ontology bump so a two-step chain is meaningfully exercised.
    original_current = versioning.CURRENT_ONTOLOGY_VERSION
    try:
        import archivetrust.domain.ontology.migrations as migrations_module

        migrations_module.CURRENT_ONTOLOGY_VERSION = 2  # type: ignore[attr-defined]
        registry.register(
            ObservationType.HEADING, 1, lambda obs: obs.model_copy(update={"ontology_version": 2})
        )
        migrated = registry.migrate(observation)
        assert migrated.ontology_version == 2
    finally:
        migrations_module.CURRENT_ONTOLOGY_VERSION = original_current  # type: ignore[attr-defined]
