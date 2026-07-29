"""F3 validation campaign (release Workstream 4): a real processing run over real PDFs, through
the exact production queue path (`acquisition/queue_runner.run_queue` — the same code the desktop
Process Queue button executes), against a dedicated Workspace, measuring telemetry volume under
the batched `CandidateExcludedBatch` encoding.

Usage:
    python scripts/run_f3_validation_campaign.py --docs 5            # probe
    python scripts/run_f3_validation_campaign.py --docs 500          # full campaign
    python scripts/run_f3_validation_campaign.py --docs 0 --report   # report only

Re-runnable: acquisition dedupes by content hash, and only still-pending Archive Objects are
processed, so successive invocations continue the same campaign rather than restarting it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

CORPUS = REPO / "archive_test"
WORKSPACE_NAME = "F3 Validation Campaign 2026-07-16"


def open_campaign_context():
    from archivetrust.clients.desktop.composition import AppContext

    context = AppContext(
        deployment_root=REPO / "archivetrust_data", auto_create_default_workspace=False
    )
    existing = next((w for w in context.workspace_store.list() if w.name == WORKSPACE_NAME), None)
    if existing is not None:
        context.open_workspace(existing.id)
    else:
        context.create_workspace(WORKSPACE_NAME, description="Release WS4: batched-exclusion validation run")
    return context


def ensure_imported(context, limit: int) -> None:
    from archivetrust.acquisition.manual_import import ManualImportSource

    pdfs = sorted(CORPUS.glob("*.pdf"))[:limit]
    source = next(
        (s for s in context.acquisition_manager.sources() if s.source_id == "manual-import"),
        None,
    ) or ManualImportSource()
    imported = context.acquisition_manager.import_files(source, tuple(pdfs))
    print(f"imported {len(imported)} new archive objects ({len(pdfs)} candidates, dedup by hash)")


def campaign_report(context) -> dict:
    layout = context.workspace_store.layout_for(context.current_workspace.id)
    events_path = layout.telemetry_dir / "events.jsonl"
    counts: dict[str, int] = {}
    batch_exclusions = 0
    total_bytes = events_path.stat().st_size if events_path.exists() else 0
    lines = 0
    if events_path.exists():
        with events_path.open(encoding="utf-8") as handle:
            for line in handle:
                lines += 1
                record = json.loads(line)
                counts[record["kind"]] = counts.get(record["kind"], 0) + 1
                if record["kind"] == "CandidateExcludedBatch":
                    batch_exclusions += len(record.get("exclusions", ()))

    progress = {"runs": 0, "documents_completed": 0, "documents_failed": 0, "durations": []}
    processing_path = layout.telemetry_dir / "processing.jsonl"
    if processing_path.exists():
        with processing_path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if record["kind"] == "ProcessingRunStarted":
                    progress["runs"] += 1
                elif record["kind"] == "DocumentProcessingCompleted":
                    if record.get("had_failure"):
                        progress["documents_failed"] += 1
                    else:
                        progress["documents_completed"] += 1
                    if record.get("duration_seconds") is not None:
                        progress["durations"].append(record["duration_seconds"])

    docs = progress["documents_completed"] + progress["documents_failed"]
    durations = progress.pop("durations")
    return {
        "workspace": context.current_workspace.name,
        "events_lines": lines,
        "events_bytes": total_bytes,
        "event_kinds": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        "individual_CandidateExcluded": counts.get("CandidateExcluded", 0),
        "CandidateExcludedBatch_events": counts.get("CandidateExcludedBatch", 0),
        "exclusion_pairs_inside_batches": batch_exclusions,
        "events_per_document": round(lines / docs, 1) if docs else None,
        "bytes_per_document": round(total_bytes / docs) if docs else None,
        "documents": progress,
        "seconds_per_document": {
            "min": round(min(durations), 2) if durations else None,
            "max": round(max(durations), 2) if durations else None,
            "mean": round(sum(durations) / len(durations), 2) if durations else None,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=int, default=5, help="How many documents to process this invocation")
    parser.add_argument("--import-limit", type=int, default=500, help="How many corpus PDFs to register")
    parser.add_argument("--report", action="store_true", help="Print the campaign report as JSON")
    args = parser.parse_args()

    from archivetrust.acquisition.queue_runner import QueueRunCallbacks, run_queue

    context = open_campaign_context()
    ensure_imported(context, args.import_limit)
    enabled = sorted(context.enabled_provider_ids())
    print(f"enabled+available providers: {enabled}")

    if args.docs > 0:
        pending = context.acquisition_manager.pending
        print(f"pending: {len(pending)}; processing up to {args.docs} this invocation")
        del pending[args.docs:]  # bound this invocation; re-running continues the campaign
        started = time.monotonic()
        outcome = run_queue(
            context,
            callbacks=QueueRunCallbacks(
                document_completed=lambda name, failed: print(
                    f"  done {name} {'FAILED' if failed else 'ok'}", flush=True
                )
            ),
        )
        print(
            f"run {outcome.run_id}: {outcome.documents_done}/{outcome.documents_total} docs in "
            f"{time.monotonic() - started:.1f}s (cancelled={outcome.cancelled}, "
            f"paused_for_health={outcome.paused_for_health})"
        )

    if args.report or args.docs > 0:
        print(json.dumps(campaign_report(context), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
