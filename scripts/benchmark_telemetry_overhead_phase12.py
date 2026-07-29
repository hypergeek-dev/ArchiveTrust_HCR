"""Phase 12 (`ROADMAP_TELEMETRY_STANDARD.md`): measures `Journal.replay` overhead per document,
before vs. after Phases 2-9's new telemetry (basis codes, mapping-table refs, per-packet outcome
tracking, `CandidateExcluded`, Research Telemetry).

Read-only over `archivetrust_data/` (Article 34's standing constraint) -- no document is
reprocessed, no provider is invoked, nothing is written back. Reads a fixed-size sample of the real
2026-07-13 benchmark corpus's own telemetry (`benchmarks/BENCHMARK_1_REPORT.md`'s workspace),
disclosed explicitly (never silently extrapolated from the full ~977MB file, most of which this
measurement doesn't need).

Because this session has made no commits, `git HEAD` *is* the pre-Phase-2 baseline and the working
tree *is* the post-Phase-9 state -- so the intended "before/after" comparison is literally
`git stash` / `git stash pop` around two runs of this same script, each writing its own timing to
`benchmarks/telemetry_overhead_<label>.json` for comparison. This script itself only measures and
writes to `benchmarks/` -- it never invokes `git`.

Run: `python scripts/benchmark_telemetry_overhead_phase12.py <label>` where `<label>` is `before` or
`after` (any string is accepted; results are written keyed by that label).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

WORKSPACE_DIR = REPO_ROOT / "archivetrust_data" / "workspaces" / "46e94f15d73347c2be79f1527e646c11"
TELEMETRY_PATH = WORKSPACE_DIR / "telemetry" / "events.jsonl"
OUTPUT_DIR = REPO_ROOT / "benchmarks"

SAMPLE_DOCUMENT_COUNT = 30  # disclosed sample size, not the full ~1000-document corpus


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: benchmark_telemetry_overhead_phase12.py <label>")
    label = sys.argv[1]

    from archivetrust.application.journal import Journal
    from archivetrust.infrastructure.storage.telemetry_sink import FileTelemetrySink

    if not TELEMETRY_PATH.exists():
        raise SystemExit(f"expected real corpus telemetry at {TELEMETRY_PATH}, not found")

    _log(f"Streaming {TELEMETRY_PATH.name}, collecting the first {SAMPLE_DOCUMENT_COUNT} documents...")
    sink = FileTelemetrySink(TELEMETRY_PATH)
    events_by_document: dict[str, list] = {}
    for event in sink.all_events():
        ref = event.document_ref
        if ref not in events_by_document and len(events_by_document) >= SAMPLE_DOCUMENT_COUNT:
            continue
        events_by_document.setdefault(ref, []).append(event)

    document_count = len(events_by_document)
    total_events = sum(len(events) for events in events_by_document.values())
    _log(f"Collected {document_count} documents, {total_events} events. Replaying...")

    journal = Journal()
    per_document_seconds: list[float] = []
    for events in events_by_document.values():
        start = time.perf_counter()
        journal.replay(events)
        per_document_seconds.append(time.perf_counter() - start)

    total_seconds = sum(per_document_seconds)
    result = {
        "label": label,
        "document_count": document_count,
        "total_events": total_events,
        "total_replay_seconds": total_seconds,
        "mean_replay_seconds_per_document": total_seconds / document_count if document_count else None,
        "mean_events_per_document": total_events / document_count if document_count else None,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"telemetry_overhead_{label}.json"
    output_path.write_text(json.dumps(result, indent=2))
    _log(f"Wrote {output_path}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
