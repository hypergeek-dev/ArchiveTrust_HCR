"""`htr/research_status.py` -- the versioned method research-status model."""

from __future__ import annotations

import pytest

from archivetrust.htr.research_status import (
    CURRENT_RESEARCH_PHASE,
    MethodResearchStatus,
    MethodResearchStatusEntry,
    ResearchPhase,
    active_method_ids,
    status_for,
)


def test_swedish_lion_and_loghi_are_active() -> None:
    assert status_for("swedish_lion").status is MethodResearchStatus.ACTIVE
    assert status_for("loghi").status is MethodResearchStatus.ACTIVE


def test_satrn_and_florence2_are_archived_from_current_phase() -> None:
    assert status_for("satrn").status is MethodResearchStatus.ARCHIVED_FROM_CURRENT_PHASE
    assert status_for("florence2_htr").status is MethodResearchStatus.ARCHIVED_FROM_CURRENT_PHASE


def test_transkribus_is_inactive_not_archived() -> None:
    """Distinct from `satrn`/`florence2_htr`: it was never part of the retired benchmark, so it is
    `INACTIVE`, not `ARCHIVED_FROM_CURRENT_PHASE`."""
    assert status_for("transkribus_swedish_lion_1").status is MethodResearchStatus.INACTIVE


def test_active_method_ids_is_exactly_the_lion_loghi_pair() -> None:
    assert set(active_method_ids()) == {"swedish_lion", "loghi"}


def test_unknown_method_id_is_unavailable_not_a_crash() -> None:
    entry = status_for("some_future_method_nobody_registered")
    assert entry.status is MethodResearchStatus.UNAVAILABLE
    assert "some_future_method_nobody_registered" in entry.reason


def test_every_entry_has_a_non_empty_reason() -> None:
    for entry in CURRENT_RESEARCH_PHASE.entries:
        assert entry.reason.strip()


def test_research_phase_rejects_duplicate_method_ids() -> None:
    with pytest.raises(ValueError):
        ResearchPhase.create(
            version=1,
            name="broken",
            description="d",
            created_at="2026-01-01T00:00:00Z",
            entries=(
                MethodResearchStatusEntry(method_id="x", status=MethodResearchStatus.ACTIVE, reason="r"),
                MethodResearchStatusEntry(method_id="x", status=MethodResearchStatus.INACTIVE, reason="r2"),
            ),
        )


def test_research_phase_rejects_version_below_one() -> None:
    with pytest.raises(ValueError):
        ResearchPhase.create(
            version=0, name="n", description="d", created_at="2026-01-01T00:00:00Z", entries=()
        )


def test_current_phase_is_version_one_with_no_supersession() -> None:
    """No prior phase was ever explicitly declared through this model -- status was purely implicit
    before this integration -- so version 1 with `supersedes=None` is the honest starting point, not
    a synthesized phase 0."""
    assert CURRENT_RESEARCH_PHASE.version == 1
    assert CURRENT_RESEARCH_PHASE.supersedes is None


def test_status_for_is_a_pure_lookup_never_mutates() -> None:
    first = status_for("loghi")
    second = status_for("loghi")
    assert first == second
