#!/usr/bin/env python
"""Exports the committed baseline research knowledge to JSON, CSV, Markdown and an evidence bundle.

Reads **only** the three committed append-only logs under
`docs/experiments/baseline-comparison/`, replays them through `HtrJournal`, and renders whatever the
replayed projection actually contains. No entity is constructed here, no metric is recomputed, and no
`baseline_knowledge.py` builder is called -- so the exported files are a view of the durable record,
not a second independent statement of it. If the logs and this script ever disagreed, the logs would
win, because this script has nothing of its own to disagree with.

    htr_research_events.jsonl           the 2026-07-30 run (100 events) -- supplies the entities the
                                        knowledge records' evidence references resolve against
    htr_knowledge_events.jsonl          the 5 observations + 5 findings + 2 review workflows
    htr_knowledge_feedback_events.jsonl the research question, its hypothesis, and the drafted
                                        (unexecuted) experiment

**Files written** (all under `docs/experiments/baseline-comparison/knowledge-export/`, all intended to
be committed):

| File | What it is |
|---|---|
| `research-knowledge.json` | the round-trippable export (`knowledge_from_json` reconstructs it) |
| `research-knowledge-findings.csv` | one row per finding; `review_status` is column 2 |
| `research-knowledge-findings_observations.csv` | one row per observation (written beside the above by `write_knowledge`) |
| `research-knowledge.md` | structured Markdown; every finding's status in its heading, a badge, and the quoted statement |
| `bundle/` | the machine-readable evidence bundle: per-entity JSON, `evidence_index.json`, and `manifest.json` with a SHA-256 per file |
| `research-knowledge-external.json` | the **reviewer-identity-stripped** variant (see below) |

**Reproducible byte-for-byte.** `generated_at` is the fixed `EXTRACTED_AT` constant the knowledge
records themselves carry, not a clock read, and the bundle writer takes no clock either -- so re-running
this script produces identical files and a diff means the logs changed. Same discipline as
`scripts/register_baseline_knowledge.py`.

**Why an `-external` variant, and why only one.** `docs/telemetry-retention.md` §7 records the finding
that the committed knowledge log carries a real personal identity (`hypergeek-dev`, 23 occurrences) in
`ResearchFinding.reviewer`, `FindingRevision.actor` and `ContradictoryEvidence.recorded_by`. The four
main exports keep it: they live in this repository, alongside the log that already carries it, and
stripping attribution from an internal research artifact would weaken accountability for no privacy
gain. `research-knowledge-external.json` is the variant for anything leaving -- same statuses, same
statements, same limitations, same revision reasoning, reviewer identity replaced by a stable
pseudonym. The pseudonym-to-identity ledger is **not written by this script**, deliberately: printed to
stdout for the operator to place wherever their own accountability records live, never into the
directory the redacted export sits in.

Usage (from the repo root):

    PYTHONPATH=src .venv/Scripts/python.exe scripts/export_baseline_knowledge.py

Idempotent -- unlike `register_baseline_knowledge.py`, re-running is safe and needs no `--fresh`: these
are rendered views of an append-only log, overwritten in place, not appended-to records.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from archivetrust.application.htr_journal import HtrJournal  # noqa: E402
from archivetrust.htr.knowledge.baseline_knowledge import EXTRACTED_AT  # noqa: E402
from archivetrust.htr.knowledge.export import (  # noqa: E402
    collect_knowledge,
    knowledge_to_json,
    write_knowledge,
)
from archivetrust.htr.knowledge.redaction import (  # noqa: E402
    DEMONSTRATION_SALT,
    identity_values,
    probable_component_identity,
    probable_human_identities,
    strip_reviewer_identities,
)
from archivetrust.infrastructure.storage.telemetry_sink import FileTelemetrySink  # noqa: E402

BASELINE_DIR = REPO_ROOT / "docs" / "experiments" / "baseline-comparison"
OUTPUT_DIR = BASELINE_DIR / "knowledge-export"

SOURCE_LOGS = (
    "htr_research_events.jsonl",
    "htr_knowledge_events.jsonl",
    "htr_knowledge_feedback_events.jsonl",
)
"""Read in this order. The run log first so the corpus/experiment/metric entities exist in the
projection before the knowledge records that reference them are replayed -- `HtrJournal`'s
`advance_*` methods refuse an entity whose registration event they have not seen, so a wrong order
would surface as a replay error rather than a silently thinner export."""

SOURCE_DESCRIPTION = (
    "Replayed from the committed append-only logs "
    "docs/experiments/baseline-comparison/{htr_research_events, htr_knowledge_events, "
    "htr_knowledge_feedback_events}.jsonl (2026-07-30 baseline run)"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--salt",
        default=DEMONSTRATION_SALT,
        help=(
            "salt for the reviewer pseudonyms in the -external variant. The default is the published "
            "DEMONSTRATION_SALT, which keeps the committed artifact reproducible and is NOT a privacy "
            "control -- pass a secret salt for a real external release."
        ),
    )
    args = parser.parse_args()

    events = []
    for name in SOURCE_LOGS:
        path = BASELINE_DIR / name
        if not path.exists():
            raise SystemExit(f"Missing committed source log: {path}")
        log_events = tuple(FileTelemetrySink(path).all_events())
        print(f"  read {len(log_events):3d} events from {name}")
        events.extend(log_events)

    store = HtrJournal().replay(tuple(events))
    export = collect_knowledge(
        store, generated_at=EXTRACTED_AT, source_description=SOURCE_DESCRIPTION
    )

    print(
        f"\nReplayed projection holds {len(export.observations)} observation(s), "
        f"{len(export.findings)} finding(s), {len(export.questions)} research question(s)."
    )
    print("Finding statuses, as recorded (never as re-derived):")
    for status, count in export.status_counts().items():
        print(f"  {count:3d}  {status}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    written = [
        write_knowledge(export, OUTPUT_DIR / "research-knowledge.json", export_format="json"),
        write_knowledge(
            export, OUTPUT_DIR / "research-knowledge-findings.csv", export_format="csv"
        ),
        OUTPUT_DIR / "research-knowledge-findings_observations.csv",
        write_knowledge(export, OUTPUT_DIR / "research-knowledge.md", export_format="markdown"),
        write_knowledge(export, OUTPUT_DIR / "bundle", export_format="bundle"),
    ]

    print("\nIdentity-bearing field values found in the export:")
    for value in identity_values(export):
        kind = "component (left intact)" if probable_component_identity(value) else "HUMAN"
        print(f"  {kind:24s} {value}")

    humans = probable_human_identities(export)
    redacted, ledger = strip_reviewer_identities(export, human_identities=humans, salt=args.salt)
    external_path = OUTPUT_DIR / "research-knowledge-external.json"
    external_path.write_text(knowledge_to_json(redacted), encoding="utf-8")
    written.append(external_path)

    print("\nWrote:")
    for path in written:
        print(f"  {path.relative_to(REPO_ROOT)}")

    print(
        "\nPseudonym ledger for research-knowledge-external.json -- the internal accountability "
        "record.\nDeliberately NOT written to disk by this script: place it wherever your own "
        "accountability\nrecords live, never beside the redacted export."
    )
    for pseudonym, identity in ledger.entries:
        print(f"  {pseudonym}  ->  {identity}")
    if args.salt == DEMONSTRATION_SALT:
        print(
            "\n  NOTE: the published DEMONSTRATION_SALT was used. The pseudonyms above are\n"
            "  enumerable by anyone who can guess the reviewer population and is therefore not a\n"
            "  privacy control -- it exists so the committed artifact is reproducible. Pass --salt\n"
            "  with a secret value for a real external release."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
