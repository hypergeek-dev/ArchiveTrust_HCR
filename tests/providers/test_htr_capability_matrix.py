"""`docs/CAPABILITY_MATRIX_HTR.md` is checked against the live adapters, not proofread.

The document this replaces (`docs/CAPABILITY_MATRIX.md`) went stale in exactly the way a hand-maintained
capability table always does: two providers were deleted from the codebase, their rating rows stayed,
and no test failed. These tests are the mechanism that makes that impossible for the HTR matrix -- the
brief's *"Generate or validate the human-readable matrix from those capabilities where practical"*, taken
as "validate on every test run".

Four different kinds of drift are covered, because "the table matches" alone is not enough:

1. the tables disagreeing with the adapters (either direction);
2. a method appearing in `composition.py` but not in the matrix, or vice versa;
3. a new field on `MethodCapabilities` that the generated table would silently omit;
4. a capability that differs across methods with no prose explaining why -- the specific failure this
   document's second half exists to prevent, and the one a byte-comparison cannot catch.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from archivetrust.providers.htr_adapter import MethodCapabilities
from archivetrust.providers.htr_capability_matrix import (
    CAPABILITY_FLAGS,
    GENERATED_BEGIN,
    GENERATED_END,
    HTR_ADAPTER_CLASSES,
    MatrixRegionError,
    build_htr_adapters,
    regenerate_document,
    render_generated_region,
    split_document,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = REPO_ROOT / "docs" / "CAPABILITY_MATRIX_HTR.md"


@pytest.fixture(scope="module")
def matrix_text() -> str:
    return MATRIX_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def adapters() -> tuple:
    return build_htr_adapters()


def test_the_generated_region_matches_what_the_adapters_report(matrix_text, adapters) -> None:
    """The central assertion: byte-for-byte, in both directions.

    A flag flipped in an adapter, a checkpoint repinned, a method's display name changed, or a
    hand-edit to a table all fail here. The failure message is the diff itself, because "the capability
    matrix is out of date" is useless without knowing which cell.
    """
    _prefix, committed, _suffix = split_document(matrix_text)
    expected = render_generated_region(adapters)
    assert committed == expected, (
        "docs/CAPABILITY_MATRIX_HTR.md's generated region has drifted from the adapters'\n"
        "get_capabilities()/get_metadata(). Re-run:\n"
        "  PYTHONPATH=src .venv/Scripts/python.exe scripts/generate_capability_matrix.py\n"
        "(if the adapters are right), or fix the adapter (if the document was right).\n\n"
        f"--- committed ---\n{committed}\n\n--- adapters report ---\n{expected}"
    )


def test_regeneration_is_idempotent_and_leaves_the_prose_untouched(matrix_text, adapters) -> None:
    """The generator must never eat the human-written half. Regenerating an already-current document
    changes nothing at all, and the prose sections survive verbatim."""
    once = regenerate_document(matrix_text, adapters)
    twice = regenerate_document(once, adapters)
    assert once == twice
    assert once == matrix_text
    for marker in (
        "## What the flags do *not* say",
        "the three numbers are not comparable",
        "the flag cannot express the truth",
        "## Model revisions in full",
    ):
        assert marker in once


def test_the_matrix_covers_exactly_the_methods_composition_builds() -> None:
    """`HTR_ADAPTER_CLASSES` duplicates `composition.py::_build_htr_adapters`'s list on purpose (so a
    doc renderer does not import the composition root). This is the assertion that makes the
    duplication safe: a fourth method added to one and not the other fails here."""
    from archivetrust.composition import AppContext

    composition_adapters = AppContext._build_htr_adapters()[0]
    composition_ids = {a.get_metadata().method_id for a in composition_adapters}
    matrix_ids = {a.get_metadata().method_id for a in build_htr_adapters()}
    assert matrix_ids == composition_ids
    assert len(HTR_ADAPTER_CLASSES) == len(composition_adapters) == 5


def test_active_htr_method_adapters_is_exactly_the_lion_loghi_pair() -> None:
    """`AppContext.active_htr_method_adapters()` -- the method-selection surface for new experiments --
    must never include an archived or inactive method, and must include both active ones."""
    from archivetrust.composition import AppContext

    ctx = AppContext.__new__(AppContext)  # composition-root construction is heavy; this method needs
    # no other state, so a bare instance is sufficient (mirrors how _build_htr_adapters is a staticmethod)
    active = ctx.active_htr_method_adapters()
    assert {a.get_metadata().method_id for a in active} == {"swedish_lion", "loghi"}


def test_experiment_builder_available_method_ids_only_offers_active_methods() -> None:
    from archivetrust.composition import AppContext

    ctx = AppContext.__new__(AppContext)
    vm = ctx.experiment_builder_viewmodel()
    offered_ids = {method_id for method_id, _label in vm.method_choices()}
    assert offered_ids == {"swedish_lion", "loghi"}
    assert "satrn" not in offered_ids
    assert "florence2_htr" not in offered_ids


def test_every_capability_flag_has_a_column() -> None:
    """Read off `MethodCapabilities` rather than restated: a flag added to the model without a
    `CAPABILITY_FLAGS` entry would otherwise be absent from the matrix with nothing complaining."""
    model_fields = set(MethodCapabilities.model_fields)
    documented = {field_name for field_name, _heading in CAPABILITY_FLAGS}
    assert documented == model_fields, (
        f"MethodCapabilities fields not in the generated matrix: {sorted(model_fields - documented)}; "
        f"matrix rows with no such field: {sorted(documented - model_fields)}"
    )


def test_every_flag_row_names_its_field_and_covers_every_method(matrix_text, adapters) -> None:
    """Structural, not just textual: each flag has exactly one row, and each row has one cell per
    method."""
    _prefix, region, _suffix = split_document(matrix_text)
    for field_name, _heading in CAPABILITY_FLAGS:
        rows = [line for line in region.splitlines() if f"(`{field_name}`)" in line]
        assert len(rows) == 1, f"{field_name} appears in {len(rows)} generated rows, expected 1"
        cells = [c.strip() for c in rows[0].strip().strip("|").split("|")]
        assert len(cells) == 1 + len(adapters), rows[0]
        assert all(cell in {"yes", "no"} for cell in cells[1:]), rows[0]


def test_the_prose_documents_every_capability_asymmetry(matrix_text, adapters) -> None:
    """A capability that is not the same across all three methods is a comparison boundary, and this
    document's entire second half exists to explain those. A byte-comparison of the generated tables
    cannot catch a *new* asymmetry arriving with no prose about it; this can.

    Checks only that the field name is discussed somewhere outside the generated region -- it cannot
    judge whether the discussion is any good, and does not pretend to.
    """
    _prefix, region, suffix = split_document(matrix_text)
    prose = matrix_text.replace(region, "")
    capabilities = [a.get_capabilities() for a in adapters]
    for field_name, _heading in CAPABILITY_FLAGS:
        values = {getattr(c, field_name) for c in capabilities}
        if len(values) > 1:
            assert f"`{field_name}`" in prose, (
                f"{field_name} differs across the three methods (a comparison boundary) but no prose "
                "outside the generated region discusses it"
            )


def test_the_confidence_non_comparability_caveat_is_present(matrix_text) -> None:
    """Special-cased because it is the one caveat where the flags agree (`yes`/`yes`/`yes`) and are
    therefore *most* misleading -- the asymmetry test above cannot catch it, since there is no
    asymmetry to catch. The 2026-07-30 baseline is the concrete counter-example and is cited by
    number."""
    assert "not comparable" in matrix_text
    assert "0.66661" in matrix_text and "0.24904" in matrix_text
    assert "sequences_scores" in matrix_text


def test_the_matrix_names_every_pinned_model_revision_in_full(matrix_text, adapters) -> None:
    """The generated identity table truncates long revisions; the prose table must carry them whole,
    or a reader cannot reproduce a run from this document."""
    for adapter in adapters:
        revision = adapter.get_metadata().model_revision
        assert revision in matrix_text, f"{revision!r} is truncated everywhere in the matrix"


def test_the_superseded_ocr_matrix_points_forward_to_this_one() -> None:
    """The old document must not be a dead end: `docs/CAPABILITY_MATRIX.md`'s supersession notice used
    to say a standalone HTR matrix "has not yet been built". It now names this file."""
    old = (REPO_ROOT / "docs" / "CAPABILITY_MATRIX.md").read_text(encoding="utf-8")
    assert "CAPABILITY_MATRIX_HTR.md" in old
    assert "not yet been built" not in old


def test_loghi_appears_exactly_once(matrix_text) -> None:
    """The brief's explicit requirement -- exactly one row per row-per-method table (identity,
    research status) and exactly one column in the flags-as-rows capability table, never duplicated."""
    _prefix, region, _suffix = split_document(matrix_text)
    for table_heading in (
        "### Method identity, as each adapter reports it",
        "### Research status (current phase)",
    ):
        start = region.index(table_heading)
        end = region.index("###", start + 1) if "###" in region[start + 1 :] else len(region)
        table_slice = region[start:end]
        assert table_slice.count("| `loghi` |") == 1, table_heading

    capability_header = next(
        line for line in region.splitlines() if line.startswith("| Capability |")
    )
    assert capability_header.count(" loghi ") == 1


def test_research_status_table_covers_exactly_the_adapters_and_matches_active_set(adapters) -> None:
    """The generated "Research status" table must name every adapter the matrix documents (archived
    ones included -- historical honesty) and the active subset must be exactly `swedish_lion`/`loghi`,
    never more, never fewer."""
    from archivetrust.htr.research_status import MethodResearchStatus, status_for

    all_ids = {a.get_metadata().method_id for a in adapters}
    active_ids = {mid for mid in all_ids if status_for(mid).status is MethodResearchStatus.ACTIVE}
    archived_ids = {
        mid for mid in all_ids if status_for(mid).status is MethodResearchStatus.ARCHIVED_FROM_CURRENT_PHASE
    }
    assert active_ids == {"swedish_lion", "loghi"}
    assert archived_ids == {"satrn", "florence2_htr"}
    # every method the matrix documents has an explicit (never UNAVAILABLE-by-omission) status
    for method_id in all_ids:
        assert status_for(method_id).status is not MethodResearchStatus.UNAVAILABLE


def test_archived_methods_still_render_in_every_generated_table(matrix_text) -> None:
    """Archiving a method from the current research phase must never remove it from this document --
    historical capability records stay readable."""
    _prefix, region, _suffix = split_document(matrix_text)
    for method_id in ("satrn", "florence2_htr"):
        assert region.count(f"`{method_id}`") >= 2  # identity table + capability header, at minimum


def test_split_document_refuses_a_malformed_region() -> None:
    """Every marker failure mode silently keeps a stale table, so each is refused rather than
    guessed."""
    with pytest.raises(MatrixRegionError):
        split_document("no markers at all")
    with pytest.raises(MatrixRegionError):
        split_document(f"{GENERATED_BEGIN}\nx\n{GENERATED_END}\n{GENERATED_BEGIN}\ny\n{GENERATED_END}")
    with pytest.raises(MatrixRegionError):
        split_document(f"{GENERATED_BEGIN}\nonly a begin marker")
