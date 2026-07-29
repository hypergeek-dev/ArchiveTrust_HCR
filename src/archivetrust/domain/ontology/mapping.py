"""The Semantic Contract Mapping Table (Constitution Article 28,
`ARCHITECTURE_TELEMETRY_STANDARD.md` §1.1/Article 28).

Every provider adapter's inline `native_label -> ObservationType` conditional -- previously a bare
`dict[str, str]` (or, for `qwen_vl`/`surya`, an unstructured if/elif chain) private to that
provider's `importer.py` -- is a semantic contract: a claim about what a provider's own vocabulary
*means* in terms of the Canonical Observation Ontology. This module gives that claim a named,
versioned, queryable identity, closing the exact gap the Semantic Contract Audit found
(`tesseract_layoutparser`'s unconditional `"Text"` -> `PARAGRAPH` mapping, invisible to telemetry
and discoverable only by reading adapter source).

**Behavior-preserving by design (mirrors Phase 3/4's own discipline):** this module changes
*visibility*, never *decision* -- each provider's `MAPPING_TABLE` is built to reproduce exactly the
same `native_label -> ObservationType` assignment its previous ad hoc dict/if-elif chain already
made; payload *construction* (which fields, defaults, fallback text) is untouched, since a mapping
table records only which ontology concept a label means, not how to build that concept's payload.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from archivetrust.domain.ontology.types import ObservationType


class MappingTableEntry(BaseModel):
    """One `native_label -> ObservationType` claim, named and independently identifiable
    (Constitution Article 28) -- the exact granularity the Semantic Contract Audit needed and did
    not have: `ObservationMapped.ontology_version` alone could say "which set of Observation types
    existed," never "was this specific label always meant to mean this specific type."
    """

    model_config = ConfigDict(frozen=True)

    entry_id: str
    """Stable, human-readable identity (`f"{provider_id}:{native_label}"`, or
    `f"{provider_id}:fallback"` for the catch-all entry) -- never a random id, since this entry's
    identity is exactly the (provider, label) pair it describes, not an event instance."""
    native_label: str | None
    """The provider's own vocabulary term. `None` only for a provider whose native artifact
    carries no label at all (e.g. `paddleocr_vl`'s plain-text transcription) -- the single entry
    representing "no label; always this type," never a guessed or invented label."""
    observation_type: ObservationType
    fallback: bool = False
    """True for the one entry per provider representing "any other/undocumented label" (Article 6:
    Full Exposure -- an adapter's `LayoutRegionPayload` catch-all, so nothing a provider emits is
    ever silently discarded for lacking a named mapping). At most one fallback entry per table."""


class MappingTable(BaseModel):
    """One provider's complete, versioned mapping table (Constitution Article 28). A version bump
    is required whenever an entry's *meaning* changes (a label starts mapping to a different
    `ObservationType`), even if the Canonical Observation Ontology itself does not change (Article
    19's independent-versioning discipline, applied one layer inward: the mapping from provider
    vocabulary to ontology concept has its own version axis, independent of both
    `ontology_version` and telemetry's `schema_version`).
    """

    model_config = ConfigDict(frozen=True)

    provider_id: str
    table_version: int
    entries: tuple[MappingTableEntry, ...]

    def entry_for(self, native_label: str | None) -> MappingTableEntry:
        """The entry governing this label, or the table's fallback entry if none names it
        explicitly. Raises if a table has no fallback and no entry matches -- every real provider
        table in this codebase defines one (Article 6), so this is a defensive check against a
        malformed table, never an expected runtime path.
        """
        for entry in self.entries:
            if not entry.fallback and entry.native_label == native_label:
                return entry
        for entry in self.entries:
            if entry.fallback:
                return entry
        raise KeyError(
            f"MappingTable for {self.provider_id!r} has no entry and no fallback for "
            f"native_label={native_label!r} -- every provider table must define a fallback "
            "(Constitution Article 6: nothing a provider emits is ever silently discarded)"
        )
