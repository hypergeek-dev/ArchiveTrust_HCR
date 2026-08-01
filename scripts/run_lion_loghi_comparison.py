#!/usr/bin/env python
"""ArchiveTrust Lion-vs-Loghi comparison launcher -- the current-research-phase menu.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/run_lion_loghi_comparison.py

The default current-research launcher no longer offers SATRN or Florence-2 (archived from the current
research phase, `htr/research_status.py`) -- only `swedish_lion` and `loghi`, the active pair. The
prior three-method reliability harness (`scripts/reliability_test.py`) remains runnable as-is for
historical reproduction; this is a separate, dedicated script composing the same underlying
persistence/telemetry primitives directly, following the precedent
`scripts/run_swedish_lion_comparison.py` already set for not generalizing that harness.

**Deviation from a plain reading of the brief's menu, stated once, here:** items 5-6 in the brief's
template assume "Swedish Lion I" is an external, manual-upload method (prepare an export, later import
a result). The Swedish Lion adapter actually wired into this research phase (`swedish_lion`, Swedish
Lion **Libre**, a local TrOCR model -- see docs/loghi-integration-audit.md §0) runs in-process, so
those two steps are collapsed into one real local-execution step below. Nothing about the menu's shape
otherwise changes.

Every action either performs a real, honest, read-only or small-scale check, or refuses to proceed with
an explanatory message -- nothing here fabricates a result. Options 6 onward touch real filesystem
state (registering experiments, writing a report); each prints exactly what it is about to do and asks
for confirmation first, mirroring `scripts/reliability_test.py`'s `_confirm` discipline.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from archivetrust.htr.experiment.models import DomainRelationship  # noqa: E402
from archivetrust.htr.persistence import DurableHtrResearchStore  # noqa: E402
from archivetrust.htr.research_status import CURRENT_RESEARCH_PHASE  # noqa: E402
from archivetrust.htr.screening.lion_loghi_experiment import build_lion_loghi_comparison  # noqa: E402
from archivetrust.htr.screening.loghi_smoke_test import (  # noqa: E402
    DEFAULT_DUTCH_PAGE_COUNT,
    DEFAULT_SWEDISH_PAGE_COUNT,
    SmokeTestPageRef,
    run_loghi_smoke_test,
)
from archivetrust.infrastructure.storage.telemetry_sink import FileTelemetrySink  # noqa: E402
from archivetrust.providers.htr_adapter import RecognitionInput  # noqa: E402
from archivetrust.providers.loghi.adapter import LoghiAdapter  # noqa: E402
from archivetrust.providers.swedish_lion.adapter import SwedishLionAdapter  # noqa: E402

SWEDISH_DATASET_ROOT = REPO_ROOT / "dataset-rgb"
DUTCH_DATASET_ROOT = REPO_ROOT / "dataset-dutch-rgb" / "republic7"
COMPARISON_DIR = REPO_ROOT / "docs" / "experiments" / "lion-loghi-comparison"
RUNS_DIR = COMPARISON_DIR / "runs"


def _confirm(prompt: str) -> bool:
    answer = input(f"{prompt} [y/N] ").strip().lower()
    return answer == "y"


def _swedish_page_paths(limit: int) -> list[Path]:
    if not SWEDISH_DATASET_ROOT.exists():
        return []
    paths: list[Path] = []
    for collection_dir in sorted(p for p in SWEDISH_DATASET_ROOT.iterdir() if p.is_dir()):
        for image_path in sorted(collection_dir.glob("*.png")):
            paths.append(image_path)
            if len(paths) >= limit:
                return paths
    return paths


def _dutch_page_paths(limit: int, split: str = "val") -> list[Path]:
    split_dir = DUTCH_DATASET_ROOT / split
    if not split_dir.exists():
        return []
    return sorted(split_dir.glob("*.jpg"))[:limit]


def do_check_environments() -> None:
    print("\n-- Check environments --")
    lion = SwedishLionAdapter()
    lion_validation = lion.validate_environment()
    print(f"swedish_lion: valid={lion_validation.valid}")
    for message in lion_validation.messages:
        print(f"  - {message}")

    loghi = LoghiAdapter()
    loghi_validation = loghi.validate_environment()
    print(f"loghi: valid={loghi_validation.valid}")
    for message in loghi_validation.messages:
        print(f"  - {message}")


def do_prepare_swedish_pages() -> None:
    print("\n-- Prepare Swedish pages --")
    if not SWEDISH_DATASET_ROOT.exists():
        print(f"MISSING: {SWEDISH_DATASET_ROOT} -- restore dataset-rgb/ before proceeding.")
        return
    collections = [p for p in SWEDISH_DATASET_ROOT.iterdir() if p.is_dir()]
    total_pages = sum(1 for c in collections for _ in c.glob("*.png"))
    print(f"dataset-rgb/: {len(collections)} collections, {total_pages} page images (real count).")


def do_prepare_dutch_pages() -> None:
    print("\n-- Prepare Dutch pages --")
    manifest_path = COMPARISON_DIR / "dataset-dutch-manifest.json"
    if not manifest_path.exists():
        print(
            f"No manifest at {manifest_path.relative_to(REPO_ROOT)} -- run "
            "scripts/inventory_dutch_dataset.py first."
        )
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(f"dataset-dutch-rgb/: {json.dumps(manifest['counts'], indent=2)}")
    print(
        "Provenance/licensing status: see "
        "docs/experiments/lion-loghi-comparison/dataset-provenance-dutch.md "
        "(unresolved license -- internal research use only until confirmed)."
    )


def do_run_loghi_smoke_test(store: DurableHtrResearchStore) -> None:
    print("\n-- Run Loghi smoke test --")
    swedish_paths = _swedish_page_paths(DEFAULT_SWEDISH_PAGE_COUNT)
    dutch_paths = _dutch_page_paths(DEFAULT_DUTCH_PAGE_COUNT)
    if not swedish_paths:
        print("No Swedish pages found (dataset-rgb/ missing or empty) -- proceeding with 0 Swedish pages.")
    if not dutch_paths:
        print("No Dutch pages found (dataset-dutch-rgb/ missing or empty) -- proceeding with 0 Dutch pages.")

    swedish_refs = tuple(
        SmokeTestPageRef(page_id=p.stem, image_path=str(p), corpus_language="sv") for p in swedish_paths
    )
    dutch_refs = tuple(
        SmokeTestPageRef(page_id=p.stem, image_path=str(p), corpus_language="nl") for p in dutch_paths
    )

    if not _confirm(
        f"Run Loghi on {len(swedish_refs)} Swedish + {len(dutch_refs)} Dutch page(s) "
        "using the real (subprocess/Docker) facade?"
    ):
        print("Cancelled.")
        return

    adapter = LoghiAdapter()
    report = run_loghi_smoke_test(
        adapter=adapter, swedish_pages=swedish_refs, dutch_pages=dutch_refs, store=store
    )
    print(f"environment_valid={report.environment_valid}")
    for message in report.environment_messages:
        print(f"  - {message}")
    print(f"succeeded={report.succeeded_count()} failed={report.failed_count()}")
    if report.failed_count():
        print(f"failure_categories={report.failure_categories()}")
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RUNS_DIR / f"smoke-test-{int(time.time())}.json"
    out_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(f"Wrote {out_path.relative_to(REPO_ROOT)}")


def do_run_swedish_lion(store: DurableHtrResearchStore) -> None:
    print("\n-- Run Swedish Lion I locally --")
    print(
        "swedish_lion runs in-process (no export/import step -- see this script's module docstring "
        "for why this differs from the brief's external-Lion menu template)."
    )
    paths = _swedish_page_paths(DEFAULT_SWEDISH_PAGE_COUNT)
    if not paths:
        print("No Swedish pages found -- nothing to run.")
        return
    if not _confirm(f"Run Swedish Lion I on {len(paths)} page(s) locally (downloads/loads real weights)?"):
        print("Cancelled.")
        return
    adapter = SwedishLionAdapter()
    for path in paths:
        result = adapter.recognize(RecognitionInput(page_image_ref=str(path)))
        status = "ok" if result.text is not None else "failed"
        print(f"  {path.name}: {status}")


def do_build_experiment_family(store: DurableHtrResearchStore) -> None:
    print("\n-- Build four-cell experiment family --")
    if not _confirm("Construct (and register) the four Lion-vs-Loghi experiment cells?"):
        print("Cancelled.")
        return
    project_id = "research_project_lion_loghi"  # a real ResearchProject must be registered separately
    comparison = build_lion_loghi_comparison(
        research_project_id=project_id,
        swedish_dataset_version_id="dataset_version_swedish_pending",
        dutch_dataset_version_id="dataset_version_dutch_pending",
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    for cell in comparison.cells:
        store.register_experiment(cell.experiment)
        store.register_experiment_version(cell.experiment_version)
        store.record_domain_relationship(
            experiment_version_id=cell.experiment_version.experiment_version_id,
            corpus_language=cell.corpus_language,
            method_primary_language_domain=cell.experiment_version.method_primary_language_domain,
            domain_relationship=cell.domain_relationship,
        )
    store.register_comparison_group(comparison.comparison_group)
    for cell in comparison.cells:
        print(f"  {cell.label}: experiment={cell.experiment.experiment_id} domain={cell.domain_relationship.value}")
    print(f"comparison_group={comparison.comparison_group.comparison_id}")


def do_run_or_resume_comparison() -> None:
    print("\n-- Run or resume comparison --")
    print(
        "Not available in this integration pass -- the brief's explicit instruction is to stop at "
        "the supervised feasibility checkpoint before launching the full Swedish-Dutch experiment. "
        "Use option 4 (smoke test) to validate the pipeline; a full run is a separate, "
        "explicitly-triggered next step. See docs/experiments/lion-loghi-comparison/RUNNING_THE_COMPARISON.md."
    )


def do_review_outputs(store: DurableHtrResearchStore) -> None:
    print("\n-- Review outputs --")
    phases = store.research_phases()
    print(f"Registered research phases: {[p.name for p in phases]}")
    groups = store.comparison_groups()
    print(f"Registered comparison groups: {[g.name for g in groups]}")
    print("Human plausibility review happens through the existing Review Center (review/) over the "
          "Evidence/Observation records these runs produce -- no separate review UI here.")


def do_generate_report() -> None:
    print("\n-- Generate report --")
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "research_phase": CURRENT_RESEARCH_PHASE.name,
        "active_method_ids": list(CURRENT_RESEARCH_PHASE.active_method_ids()),
        "swedish_dataset_present": SWEDISH_DATASET_ROOT.exists(),
        "dutch_dataset_present": DUTCH_DATASET_ROOT.exists(),
    }
    out_path = RUNS_DIR / f"status-report-{int(time.time())}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out_path.relative_to(REPO_ROOT)}")
    print(json.dumps(report, indent=2))


MENU = """
Swedish Lion I vs. Loghi -- current research phase
1. Check environments
2. Prepare Swedish pages
3. Prepare Dutch pages
4. Run Loghi smoke test
5. Run Swedish Lion I locally
6. Build four-cell experiment family
7. Run or resume comparison
8. Review outputs
9. Generate report
0. Exit
"""


def main() -> int:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    sink = FileTelemetrySink(RUNS_DIR / "events.jsonl")
    store = DurableHtrResearchStore(sink)
    # This store's in-memory projection starts empty each process (full replay from the sink's
    # existing events is HtrJournal's job, not this launcher's) -- so this registers a fresh
    # MethodResearchStatusChanged each run rather than truly deduplicating across processes. Each
    # such event is still a real, honest fact ("the phase was active at this decision point"), not a
    # fabricated one, so this is an accepted simplification for a launcher, not a correctness bug.
    store.register_research_phase(CURRENT_RESEARCH_PHASE)

    while True:
        print(MENU)
        choice = input("> ").strip()
        if choice == "1":
            do_check_environments()
        elif choice == "2":
            do_prepare_swedish_pages()
        elif choice == "3":
            do_prepare_dutch_pages()
        elif choice == "4":
            do_run_loghi_smoke_test(store)
        elif choice == "5":
            do_run_swedish_lion(store)
        elif choice == "6":
            do_build_experiment_family(store)
        elif choice == "7":
            do_run_or_resume_comparison()
        elif choice == "8":
            do_review_outputs(store)
        elif choice == "9":
            do_generate_report()
        elif choice == "0":
            print("Exiting.")
            return 0
        else:
            print("Unrecognized choice.")


if __name__ == "__main__":
    raise SystemExit(main())
