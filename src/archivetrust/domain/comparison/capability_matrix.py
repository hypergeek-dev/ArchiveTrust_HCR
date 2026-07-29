"""The machine-readable Capability Matrix (MILESTONE4_COMPARISON_ENGINE.md S1, S18 item 3).

Formalizes `docs/investigation/CAPABILITY_MATRIX.md`'s Native/Derived/Partial/No ratings as a
versioned engine input. The engine weights structural and coverage claims **by capability tag,
never by provider name** (C6) -- this type is what makes that lookup possible without any
comparison code special-casing a provider.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict

from archivetrust.domain.ontology.types import ObservationType


class Capability(str, Enum):
    """Native/Derived/Partial/No, per `docs/investigation/CAPABILITY_MATRIX.md`'s format."""

    NATIVE = "native"
    DERIVED = "derived"
    PARTIAL = "partial"
    NO = "no"


class CapabilityMatrixEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider_id: str
    provider_version: str
    observation_type: ObservationType
    capability: Capability


class CapabilityMatrix(BaseModel):
    """Must carry the `(provider, provider_version)` it was validated against
    (`ARCHITECTURE_READINESS_REVIEW.md` S5.3) -- enforced here by keying every entry on both,
    never provider_id alone, so a provider_version bump requires an explicit new entry rather
    than silently inheriting an old rating.
    """

    model_config = ConfigDict(frozen=True)

    matrix_version: int
    entries: tuple[CapabilityMatrixEntry, ...]

    def capability_for(
        self, *, provider_id: str, provider_version: str, observation_type: ObservationType
    ) -> Capability:
        """Defaults to `NO` for any `(provider, provider_version, type)` combination with no
        entry -- an unrated provider/type pair must never be silently treated as capable.
        """
        for entry in self.entries:
            if (
                entry.provider_id == provider_id
                and entry.provider_version == provider_version
                and entry.observation_type == observation_type
            ):
                return entry.capability
        return Capability.NO
