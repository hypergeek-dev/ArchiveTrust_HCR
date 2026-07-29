"""The ontology-version migration registry (ROADMAP.md S5.4, Constitution Article 19).

A migration is a pure function `Observation<ontology_version=N> -> Observation<ontology_version=N+1>`
registered against `(observation_type, from_version)`. Replay must apply registered migrations
when reconstructing history recorded under an older ontology version, and must fail loudly -- never
silently drop data -- if no migration path is registered for the version gap it encounters.
"""

from __future__ import annotations

from collections.abc import Callable

from archivetrust.domain.ontology.base import Observation
from archivetrust.domain.ontology.types import ObservationType
from archivetrust.domain.shared.versioning import CURRENT_ONTOLOGY_VERSION

Migration = Callable[[Observation], Observation]


class MigrationNotRegisteredError(RuntimeError):
    """Raised when replay encounters an Observation whose ontology_version has no registered
    migration path forward. Per ROADMAP.md S5.4, this must fail loudly, never silently drop data.
    """

    def __init__(self, observation_type: ObservationType, from_version: int) -> None:
        super().__init__(
            f"No migration registered for {observation_type.value!r} from ontology_version "
            f"{from_version} -- replay cannot proceed without either a registered migration or "
            "an explicit 'cannot migrate, requires re-observation' marker"
        )
        self.observation_type = observation_type
        self.from_version = from_version


class MigrationRegistry:
    """Holds registered migrations and applies the full chain needed to bring an Observation up
    to `CURRENT_ONTOLOGY_VERSION`.
    """

    def __init__(self) -> None:
        self._migrations: dict[tuple[ObservationType, int], Migration] = {}

    def register(
        self, observation_type: ObservationType, from_version: int, migration: Migration
    ) -> None:
        self._migrations[(observation_type, from_version)] = migration

    def migrate(self, observation: Observation) -> Observation:
        """Applies registered migrations repeatedly until `observation.ontology_version` equals
        `CURRENT_ONTOLOGY_VERSION`. Raises `MigrationNotRegisteredError` if a required step is
        missing -- this is what makes "predates a type's redefinition" distinguishable from
        "malformed" during replay (MILESTONE1_DOMAIN_MODEL.md S1.8, ROADMAP.md S5.4).
        """
        current = observation
        while current.ontology_version < CURRENT_ONTOLOGY_VERSION:
            key = (current.observation_type, current.ontology_version)
            migration = self._migrations.get(key)
            if migration is None:
                raise MigrationNotRegisteredError(current.observation_type, current.ontology_version)
            current = migration(current)
        return current
