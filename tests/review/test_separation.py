"""Enforces the dependency direction of the Human Review System (HUMAN_REVIEW_SPECIFICATION.md §0
Frozen Constraint 1, §3; ROADMAP_V2.md LP-1).

The `review` package spans the Dual Architecture boundary and may import both the Trust Engine and
the Learning Platform. The one-directional rule is: nothing *below* it may import it. The Trust
Engine must not import `review` (nor `learning`), and the Learning Platform must not import
`review` — so the boundary-spanner can be deleted without disturbing either side it composes.
"""

from __future__ import annotations

import re
from pathlib import Path

import archivetrust

SRC_ROOT = Path(archivetrust.__file__).resolve().parent
# `runtime` is a foundational, provider-agnostic AI-infrastructure package (Provider Runtime
# Abstraction milestone) -- it imports nothing from archivetrust itself and sits alongside `domain`
# as something the Trust Engine (via providers/qwen_vl/runtime_backend.py) depends on. It is
# scanned here for the same reason `domain` is: it must never import an upper layer.
TRUST_ENGINE_DIRS = ("domain", "providers", "application", "infrastructure", "runtime")

# `workspace`/`acquisition` are the Workspace & Acquisition Management layer (ROADMAP.md §5.13,
# Revision 5; Constitution Article 25) -- they sit immediately above the Trust Engine (Acquisition
# calls `application.pipeline.run_pipeline` once an Archive Object is registered) but immediately
# below `learning`: the Learning Platform learns from telemetry, not acquisition mechanics, so it
# must stay as ignorant of acquisition as the Trust Engine is (Article 25 extends Article 20's
# provider-independence discipline one stage upstream).
ACQUISITION_LAYER_DIRS = ("workspace", "acquisition")

# The layers above the Trust Engine, from lowest to highest. Each may import those below it; none
# below may import those above (ROADMAP_V2.md LP-1; HUMAN_REVIEW_SPECIFICATION.md §0).
_UPPER_LAYERS = ("workspace", "acquisition", "learning", "review", "presentation", "clients")


def _import_pattern(package: str) -> re.Pattern[str]:
    return re.compile(rf"(from|import)\s+archivetrust\.{package}\b")


def _python_files(subdir: str) -> list[Path]:
    directory = SRC_ROOT / subdir
    return list(directory.rglob("*.py")) if directory.exists() else []


def _importers(scanned_subdir: str, forbidden_package: str) -> list[Path]:
    pattern = _import_pattern(forbidden_package)
    return [
        path.relative_to(SRC_ROOT)
        for path in _python_files(scanned_subdir)
        if pattern.search(path.read_text(encoding="utf-8"))
    ]


def test_trust_engine_never_imports_any_upper_layer() -> None:
    offenders = {}
    for subdir in TRUST_ENGINE_DIRS:
        for package in _UPPER_LAYERS:
            hits = _importers(subdir, package)
            if hits:
                offenders.setdefault(package, []).extend(hits)
    assert not offenders, (
        "Trust Engine modules must never import an upper layer (learning/review/presentation/"
        f"clients) — ROADMAP_V2.md LP-1, HUMAN_REVIEW_SPECIFICATION.md §0; offenders: {offenders}"
    )


def test_layer_dependencies_run_only_downward() -> None:
    # workspace/acquisition may not import learning/review/presentation/clients; learning may not
    # import review/presentation/clients; review may not import presentation/clients; presentation
    # may not import clients. Each lower layer stays ignorant of the layers that consume it (HR-11:
    # the boundary-spanner and ViewModels can be deleted from below).
    forbidden_above = {
        "workspace": ("learning", "review", "presentation", "clients"),
        "acquisition": ("learning", "review", "presentation", "clients"),
        "learning": ("review", "presentation", "clients"),
        "review": ("presentation", "clients"),
        "presentation": ("clients",),
    }
    offenders = {}
    for lower, above in forbidden_above.items():
        for higher in above:
            hits = _importers(lower, higher)
            if hits:
                offenders.setdefault(f"{lower}->{higher}", []).extend(hits)
    assert not offenders, f"Upward imports detected (must run only downward): {offenders}"


def test_review_package_composes_both_sides() -> None:
    # Sanity: the boundary-spanner imports and exposes its service surface, proving it can reach
    # both the Trust Engine correction contract and the Learning Platform telemetry stream.
    from archivetrust.review.service import ReviewAction, ReviewService

    assert hasattr(ReviewService, "submit_decision")
    assert hasattr(ReviewService, "open_document")
    assert ReviewAction.SKIP in ReviewAction
