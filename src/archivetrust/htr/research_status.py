"""Which HTR methods are in scope for the *current* research phase (docs/loghi-integration-audit.md
§4: no such concept existed before this module).

**Deliberately separate from `MethodCapabilities`.** A method's capabilities (what it can produce) and
its research status (whether the current research phase is actively comparing it) are different
questions that change on different timelines -- a capability is a fact about an adapter's
implementation; a research status is a fact about which comparison the project is running right now.
Overloading a capability flag to also mean "in scope" would make a future capability change silently
also change scope, and vice versa. See `providers/htr_capability_matrix.py`'s generated "Research
status" section, which renders this module's data in a table separate from the capability table for
the same reason.

**Versioned like `ExperimentVersion`/`ReviewPacketClosureKind`-adjacent status types elsewhere in this
codebase**, not a bare dict: a `ResearchPhase` is an immutable, dated declaration of which methods were
in scope, so "SATRN was active until phase 1 archived it" remains answerable rather than being
overwritten in place. `CURRENT_RESEARCH_PHASE` is version 1 -- not because nothing came before it in
time (the three-method reliability benchmark obviously did), but because no phase before this one was
ever *declared* through this model; status was purely implicit (every constructed adapter was, in
effect, active) until now. `supersedes=None` records that honestly rather than inventing a synthetic
phase 0.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, model_validator

from archivetrust.domain.shared.ids import new_id


class MethodResearchStatus(str, Enum):
    """What role a method plays in the *current* research phase. Distinct from
    `EnvironmentValidation.valid` (can it run here, right now) and from any capability flag (what it
    can produce when it does run) -- this only answers "is the current research phase comparing it."
    """

    ACTIVE = "active"
    """In scope for the current research phase's comparisons; selectable when building a new
    experiment."""
    INACTIVE = "inactive"
    """Not part of the current research phase, and never was part of a phase now being retired --
    simply out of current scope. Distinct from ARCHIVED_FROM_CURRENT_PHASE below."""
    ARCHIVED_FROM_CURRENT_PHASE = "archived_from_current_phase"
    """Was active in a prior, now-retired research phase (its evidence is sealed, historical, and
    fully replayable) and is deliberately excluded from the phase that superseded it. Not a judgment
    that the method is worse -- a scoping decision about what the project is comparing next."""
    EXPERIMENTAL = "experimental"
    """Registered but not yet a first-class comparison subject -- e.g. a method under evaluation
    before a decision to make it ACTIVE or leave it INACTIVE."""
    UNAVAILABLE = "unavailable"
    """No explicit status was ever recorded for this method_id. Never the default for a method that
    *has* an entry -- see `status_for()` -- only the honest fallback for one that has none."""


class MethodResearchStatusEntry(BaseModel):
    """One method's status within one `ResearchPhase`."""

    model_config = ConfigDict(frozen=True)

    method_id: str
    status: MethodResearchStatus
    reason: str
    """Always populated -- a status with no stated reason is exactly the kind of silent classification
    Constitution Article 6 (Full Exposure) forbids elsewhere in this codebase; the same discipline
    applies here."""


class ResearchPhase(BaseModel):
    """An immutable, dated declaration of every method's research status at one point in time. A
    status change is always a new `ResearchPhase` with an incremented `version` and `supersedes` set
    to the prior phase's id -- never an edit in place, mirroring `ExperimentVersion`'s "supersede,
    don't overwrite" rule (`htr/experiment/models.py`)."""

    model_config = ConfigDict(frozen=True)

    research_phase_id: str
    version: int
    name: str
    description: str
    entries: tuple[MethodResearchStatusEntry, ...]
    created_at: str
    supersedes: str | None = None

    @model_validator(mode="after")
    def _validate(self) -> "ResearchPhase":
        if self.version < 1:
            raise ValueError("ResearchPhase.version must be >= 1")
        method_ids = [entry.method_id for entry in self.entries]
        if len(method_ids) != len(set(method_ids)):
            raise ValueError("ResearchPhase.entries must not repeat a method_id")
        return self

    @classmethod
    def create(
        cls,
        *,
        version: int,
        name: str,
        description: str,
        entries: tuple[MethodResearchStatusEntry, ...],
        created_at: str,
        supersedes: str | None = None,
    ) -> "ResearchPhase":
        return cls(
            research_phase_id=new_id("research_phase"),
            version=version,
            name=name,
            description=description,
            entries=entries,
            created_at=created_at,
            supersedes=supersedes,
        )

    def status_for(self, method_id: str) -> MethodResearchStatusEntry:
        """This phase's entry for `method_id`, or an honest `UNAVAILABLE` fallback -- never a crash,
        never a silent default to `ACTIVE`. A method absent from a phase's `entries` was simply never
        classified under it, which is a fact worth reporting, not an error worth raising: a new
        adapter added to composition without an accompanying `ResearchPhase` update should surface as
        "unavailable, no status recorded" in the UI, not take down the method overview page.
        """
        for entry in self.entries:
            if entry.method_id == method_id:
                return entry
        return MethodResearchStatusEntry(
            method_id=method_id,
            status=MethodResearchStatus.UNAVAILABLE,
            reason=(
                f"No research-phase status recorded for method_id={method_id!r} in phase "
                f"{self.research_phase_id!r} ({self.name!r}, version {self.version})"
            ),
        )

    def active_method_ids(self) -> tuple[str, ...]:
        return tuple(
            entry.method_id for entry in self.entries if entry.status is MethodResearchStatus.ACTIVE
        )


CURRENT_RESEARCH_PHASE = ResearchPhase.create(
    version=1,
    name="Swedish Lion I vs. Loghi",
    description=(
        "Swedish Lion I (local TrOCR, method_id=swedish_lion) and Loghi are the active comparison "
        "pair for in-domain/cross-domain technical comparison on Swedish and Dutch historical "
        "handwritten pages. SATRN and Florence-2 are archived from this phase: their evidence from "
        "the completed technical reliability screening benchmark "
        "(docs/experiments/technical-reliability-screening/) remains sealed, historical, and fully "
        "replayable, but neither participates in new experiments created under this phase. "
        "Transkribus Swedish Lion I (method_id=transkribus_swedish_lion_1) was never part of the "
        "benchmark this phase supersedes and is marked inactive rather than archived -- it remains "
        "available, just out of scope for this phase's active pair."
    ),
    created_at="2026-08-01T00:00:00Z",
    supersedes=None,
    entries=(
        MethodResearchStatusEntry(
            method_id="swedish_lion",
            status=MethodResearchStatus.ACTIVE,
            reason="Active method for the Lion-vs-Loghi research phase (local TrOCR execution).",
        ),
        MethodResearchStatusEntry(
            method_id="loghi",
            status=MethodResearchStatus.ACTIVE,
            reason="Active method for the Lion-vs-Loghi research phase (local containerized pipeline).",
        ),
        MethodResearchStatusEntry(
            method_id="satrn",
            status=MethodResearchStatus.ARCHIVED_FROM_CURRENT_PHASE,
            reason=(
                "Completed its role in the sealed technical reliability screening benchmark "
                "(reliability-2026-07-31); excluded from the new active comparison, evidence remains "
                "readable and replayable."
            ),
        ),
        MethodResearchStatusEntry(
            method_id="florence2_htr",
            status=MethodResearchStatus.ARCHIVED_FROM_CURRENT_PHASE,
            reason=(
                "Completed its role in the sealed technical reliability screening benchmark "
                "(reliability-2026-07-31); excluded from the new active comparison, evidence remains "
                "readable and replayable."
            ),
        ),
        MethodResearchStatusEntry(
            method_id="transkribus_swedish_lion_1",
            status=MethodResearchStatus.INACTIVE,
            reason=(
                "Distinct adapter from the active swedish_lion method (external manual Transkribus "
                "workflow, not local execution); never part of the retired reliability benchmark, so "
                "not archived -- simply out of scope for this phase's active pair. Remains available "
                "for its own workflows."
            ),
        ),
    ),
)
"""The concrete, current declaration this module's callers read. Superseding this (a future research
phase) means constructing a new `ResearchPhase` with `version=2`, `supersedes=CURRENT_RESEARCH_PHASE.
research_phase_id`, and rebinding this name -- never editing the entries tuple above in place."""


def status_for(method_id: str) -> MethodResearchStatusEntry:
    """Convenience wrapper over `CURRENT_RESEARCH_PHASE.status_for` -- the call site most of this
    codebase actually wants (composition, capability matrix, method overview), all of which care about
    "the current phase," not "some phase."""
    return CURRENT_RESEARCH_PHASE.status_for(method_id)


def active_method_ids() -> tuple[str, ...]:
    return CURRENT_RESEARCH_PHASE.active_method_ids()
