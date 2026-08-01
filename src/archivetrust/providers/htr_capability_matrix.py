"""Renders `docs/CAPABILITY_MATRIX_HTR.md`'s capability tables from the adapters themselves.

**The adapters' `get_capabilities()` / `get_metadata()` are the source of truth.** The brief is explicit
about this -- *"The adapters' `get_capabilities()` implementations should remain the machine-readable
source of truth. Generate or validate the human-readable matrix from those capabilities where
practical."* -- and the previous `docs/CAPABILITY_MATRIX.md` is the cautionary example: it carried a
hand-maintained `(provider_id, provider_version, observation_type)` rating table for two providers that
no longer exist in the codebase, and nothing failed when they were deleted.

So the human-readable tables in `docs/CAPABILITY_MATRIX_HTR.md` are **generated**, not written, and the
generated region is delimited by `GENERATED_BEGIN`/`GENERATED_END` markers.
`tests/providers/test_htr_capability_matrix.py` re-renders from the live adapters and asserts the
document's region matches byte-for-byte, so any drift -- a flag flipped in an adapter, a checkpoint
repinned, a table hand-edited, a method added or removed -- fails a test rather than quietly leaving a
document that lies. `scripts/generate_capability_matrix.py --check` is the same assertion from a CLI.

Only the *tables* are generated. Everything outside the markers is prose a human wrote about what the
flags mean and where they are misleading -- e.g. that Transkribus's `external_upload_required=False` is
"the least-misleading available value" for output that originated in an external service this adapter
never calls. Generating that would be impossible, and mechanically regenerating a whole document would
delete it.

This module is deliberately in `providers/` rather than in `scripts/`: it reads the adapter Protocol and
nothing else, it is imported by a test, and `scripts/` is not an importable package.
"""

from __future__ import annotations

GENERATED_BEGIN = "<!-- BEGIN GENERATED FROM ADAPTERS -- do not edit by hand -->"
GENERATED_END = "<!-- END GENERATED FROM ADAPTERS -->"

HTR_ADAPTER_CLASSES = (
    ("archivetrust.providers.satrn.adapter", "SatrnAdapter"),
    ("archivetrust.providers.florence2_htr.adapter", "Florence2Adapter"),
    ("archivetrust.providers.transkribus.adapter", "TranskribusAdapter"),
    ("archivetrust.providers.swedish_lion.adapter", "SwedishLionAdapter"),
    ("archivetrust.providers.loghi.adapter", "LoghiAdapter"),
)
"""The five real HTR methods, in the order `composition.py::_build_htr_adapters` lists them.

Duplicated from that method rather than imported from it on purpose: importing `composition` here would
pull the entire application composition root (and PySide6-adjacent modules) into a documentation
renderer and into the test that drives it. The duplication is asserted away --
`tests/providers/test_htr_capability_matrix.py::test_the_matrix_covers_exactly_the_methods_composition_builds`
fails if the two lists ever disagree, so a fourth method added to one and not the other surfaces
immediately.
"""

CAPABILITY_FLAGS = (
    ("confidence_supported", "Confidence"),
    ("geometry_supported", "Geometry"),
    ("line_level_supported", "Line-level input"),
    ("page_level_supported", "Page-level input"),
    ("local_execution_supported", "Runs locally"),
    ("external_upload_required", "Requires external upload"),
    ("image_color_normalization_required", "Requires RGB normalization"),
)
"""`(field_name, column_heading)` for every field on `MethodCapabilities`.

Derived from the model in the test (`test_every_capability_flag_has_a_column`), not trusted: a flag
added to `MethodCapabilities` without a column here would otherwise be silently absent from the
generated matrix, which is precisely the drift this module exists to prevent.
"""


def build_htr_adapters() -> tuple:
    """The three real adapters, constructed with no arguments.

    Construction is inert for all three (no model load, no network, no subprocess -- see
    `composition.py::htr_method_adapters`'s docstring), which is what makes generating a document from
    them cheap enough to assert on every test run. An import or construction failure propagates: a
    capability matrix that silently omitted a method it could not construct would be worse than no
    matrix.
    """
    import importlib

    adapters = []
    for module_name, class_name in HTR_ADAPTER_CLASSES:
        module = importlib.import_module(module_name)
        adapters.append(getattr(module, class_name)())
    return tuple(adapters)


def _flag(value: bool) -> str:
    return "yes" if value else "no"


def render_identity_table(adapters: tuple) -> str:
    """One row per method: `method_id`, name, vendor, pinned `model_revision`.

    `model_revision` is included in the *generated* region specifically so a repinned checkpoint breaks
    the matrix test. A capability matrix that names no checkpoint describes a method in the abstract;
    every measured claim in this repository is bound to a revision.
    """
    lines = [
        "| `method_id` | Method | Vendor | `model_revision` (pinned) |",
        "|---|---|---|---|",
    ]
    for adapter in adapters:
        metadata = adapter.get_metadata()
        revision = metadata.model_revision
        # A long honest prose revision (Transkribus reports one) is truncated in this table and stated
        # in full in the prose below it -- never silently shortened without saying so.
        if len(revision) > 60:
            revision = revision[:57] + "..."
        lines.append(
            f"| `{metadata.method_id}` | {metadata.method_name} | {metadata.vendor} | "
            f"`{revision}` |"
        )
    return "\n".join(lines)


def render_capability_table(adapters: tuple) -> str:
    """The capability matrix proper: one row per flag, one column per method.

    Flags as rows rather than columns because there are six flags and three methods, and because the
    question a reader arrives with is "which methods give me geometry?" -- a row read across.
    """
    metadata = [adapter.get_metadata() for adapter in adapters]
    capabilities = [adapter.get_capabilities() for adapter in adapters]

    header = "| Capability | " + " | ".join(m.method_id for m in metadata) + " |"
    divider = "|---|" + "---|" * len(metadata)
    lines = [header, divider]
    for field_name, heading in CAPABILITY_FLAGS:
        cells = " | ".join(_flag(getattr(c, field_name)) for c in capabilities)
        lines.append(f"| {heading} (`{field_name}`) | {cells} |")
    return "\n".join(lines)


def render_research_status_table(adapters: tuple) -> str:
    """One row per method: its status in the *current* research phase (`htr/research_status.py`),
    and why.

    **Deliberately a separate table from `render_capability_table`, never merged into it.** Research
    status and capability are different concepts (module docstring of `htr/research_status.py`): a
    method's capability flags describe what its adapter can produce and change only when the adapter
    changes; its research status describes whether the current research phase is comparing it and
    changes when the phase does. Rendering them in one table would make a future status change look
    like a capability change in every diff.
    """
    from archivetrust.htr.research_status import status_for

    lines = [
        "| `method_id` | Research status | Reason |",
        "|---|---|---|",
    ]
    for adapter in adapters:
        method_id = adapter.get_metadata().method_id
        entry = status_for(method_id)
        lines.append(f"| `{method_id}` | `{entry.status.value}` | {entry.reason} |")
    return "\n".join(lines)


def render_generated_region(adapters: tuple) -> str:
    """Everything between the markers, markers included.

    One function so the document, the generator CLI and the drift test all produce the same bytes --
    a test that re-implemented the rendering would test itself.
    """
    return "\n".join(
        (
            GENERATED_BEGIN,
            "",
            "### Method identity, as each adapter reports it",
            "",
            render_identity_table(adapters),
            "",
            "### Capability flags, as each adapter reports them",
            "",
            render_capability_table(adapters),
            "",
            "### Research status (current phase)",
            "",
            render_research_status_table(adapters),
            "",
            GENERATED_END,
        )
    )


class MatrixRegionError(ValueError):
    """Raised when a document has no generated region, or a malformed one. Named rather than a bare
    `ValueError` so a caller can distinguish "this document is not generated-matrix-shaped" from "the
    generated content drifted"."""


def split_document(text: str) -> tuple[str, str, str]:
    """`(prefix, generated region including markers, suffix)`.

    Refuses a document with a missing, duplicated or inverted marker pair rather than guessing, because
    every failure mode here silently produces a document that keeps a stale table.
    """
    if text.count(GENERATED_BEGIN) != 1 or text.count(GENERATED_END) != 1:
        raise MatrixRegionError(
            f"expected exactly one {GENERATED_BEGIN!r} and one {GENERATED_END!r}; found "
            f"{text.count(GENERATED_BEGIN)} and {text.count(GENERATED_END)}"
        )
    start = text.index(GENERATED_BEGIN)
    end = text.index(GENERATED_END) + len(GENERATED_END)
    if end <= start:
        raise MatrixRegionError("the generated-region end marker precedes its begin marker")
    return text[:start], text[start:end], text[end:]


def regenerate_document(text: str, adapters: tuple) -> str:
    """`text` with its generated region replaced by freshly-rendered content. Prose is untouched."""
    prefix, _current, suffix = split_document(text)
    return prefix + render_generated_region(adapters) + suffix
