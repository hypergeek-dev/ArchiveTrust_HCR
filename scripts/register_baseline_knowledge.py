#!/usr/bin/env python
"""Writes the real research-knowledge artifact for the committed 2026-07-30 baseline run.

Registers the five real `ResearchObservation`s and five candidate `ResearchFinding`s from
`archivetrust.htr.knowledge.baseline_knowledge` into a `DurableHtrResearchStore` backed by a real
`FileTelemetrySink`, then runs two of the findings through the real status workflow. Every id and
every measured value comes from `docs/experiments/baseline-comparison/htr_research_events.jsonl`; no
model is run and no metric is recomputed.

**Why a second log file rather than appending to the run's.** `htr_research_events.jsonl` is a
committed research artifact documented as exactly one run's 100 events, and
`tests/htr/persistence/test_real_baseline_reconstruction.py` reads it as the evidence for that run.
Appending knowledge records would falsify its own README's description of it. Knowledge therefore goes
to `htr_knowledge_events.jsonl` beside it -- `docs/architecture/htr-telemetry.md` §7's "two streams,
one mechanism", the same `FileTelemetrySink` class, the same append-only guarantees, the same
hash-chain sidecar. The `causation_id`s point across into the run's log, and replaying the two
concatenated reconstructs the whole graph (proven by
`tests/htr/knowledge/test_baseline_knowledge.py`).

**Files written** (all under `docs/experiments/baseline-comparison/`, all intended to be committed):

| File | What it is |
|---|---|
| `htr_knowledge_events.jsonl` | the durable, append-only knowledge log -- the source of truth for the observations and findings |
| `htr_knowledge_events.jsonl.chain.jsonl` | `HashChainAppender`'s tamper-evidence sidecar |

Reproducible byte-for-byte apart from the freshly-minted entity ids (`observation_id`, `finding_id`,
`revision_id`, `contradiction_id`, `event_id`): every timestamp is the fixed `EXTRACTED_AT` constant
rather than a clock read.

Usage (from the repo root):

    PYTHONPATH=src .venv/Scripts/python.exe scripts/register_baseline_knowledge.py [--fresh]

`--fresh` is required when the log already exists -- same "one pass per log" rule, and for the same
reason, as `scripts/run_baseline_comparison.py`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from archivetrust.htr.knowledge.baseline_knowledge import EXTRACTED_AT  # noqa: E402
from archivetrust.htr.knowledge.registration import (  # noqa: E402
    demonstrate_review_workflow,
    register_baseline_knowledge,
)
from archivetrust.htr.persistence import DurableHtrResearchStore  # noqa: E402
from archivetrust.infrastructure.storage.telemetry_sink import FileTelemetrySink  # noqa: E402

OUTPUT_DIR = REPO_ROOT / "docs" / "experiments" / "baseline-comparison"
KNOWLEDGE_PATH = OUTPUT_DIR / "htr_knowledge_events.jsonl"
CHAIN_PATH = OUTPUT_DIR / "htr_knowledge_events.jsonl.chain.jsonl"

DEFAULT_REVIEWER = "hypergeek-dev"
"""The repository maintainer, who is the human actually performing this phase's reviews.

Attributed honestly rather than with a plausible-looking placeholder, and stated plainly in
`docs/knowledge-lifecycle.md`: there is no independent second reviewer and no domain expert in the
loop for these two transitions. `reviewer` is a required argument to the transition function precisely
so this had to be a decision rather than a default."""


def _prepare(*, fresh: bool) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not KNOWLEDGE_PATH.exists():
        return
    if not fresh:
        raise SystemExit(
            f"Refusing to run: {KNOWLEDGE_PATH} already exists.\n"
            "  Pass --fresh to delete it and record a single clean pass (what you want for a\n"
            "  committed artifact). Appending would put two sets of observations about one run in\n"
            "  one file with nothing saying which the documentation quotes."
        )
    for path in (KNOWLEDGE_PATH, CHAIN_PATH):
        if path.exists():
            path.unlink()
    print(f"      --fresh: removed the previous knowledge log at {KNOWLEDGE_PATH}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh", action="store_true", help="delete an existing knowledge log first")
    parser.add_argument("--reviewer", default=DEFAULT_REVIEWER, help="attributable human reviewer")
    args = parser.parse_args()

    _prepare(fresh=args.fresh)

    sink = FileTelemetrySink(KNOWLEDGE_PATH)
    store = DurableHtrResearchStore(sink, actor_id="scripts.register_baseline_knowledge")

    knowledge = register_baseline_knowledge(store, at=EXTRACTED_AT)
    print(f"Registered {len(knowledge.observations)} research observations:")
    for name, observation in knowledge.observations.items():
        print(f"  [{observation.observation_type.value}] {name}")
        print(f"      id        {observation.observation_id}")
        print(f"      event     {knowledge.observation_event_ids[name]}")
        print(f"      scope     {observation.scope.describe()}")
        print(f"      evidence  {len(observation.supporting_evidence)} typed references")
        if observation.unverified_hypothesis is not None:
            print("      hypothesis (UNVERIFIED, held separately from every factual field)")

    print(f"\nRegistered {len(knowledge.findings)} candidate findings:")
    for name, finding in knowledge.findings.items():
        print(f"  [{finding.review_status.value}] {name}")
        print(f"      id        {finding.finding_id}")
        print(f"      event     {knowledge.finding_event_ids[name]}")
        print(f"      observ.   {', '.join(finding.supporting_observations)}")

    transitioned = demonstrate_review_workflow(
        store, knowledge, reviewer=args.reviewer, at=EXTRACTED_AT
    )
    print("\nStatus workflow (real transitions, real reasoning):")
    for name, finding in transitioned.items():
        print(f"  {name} -> {finding.review_status.value}")
        for revision in finding.revision_history:
            print(
                f"      {revision.from_status.value} -> {revision.to_status.value} "
                f"by {revision.actor}"
            )
        for contradiction in finding.contradictory_evidence:
            print(
                f"      contradicted by {contradiction.source_kind.value} "
                f"{contradiction.source_id}"
            )

    events = tuple(sink.all_events())
    kinds = sorted({event.kind.value for event in events})
    print(f"\nWrote {len(events)} knowledge events ({len(kinds)} kinds) to {KNOWLEDGE_PATH}")
    for kind in kinds:
        print(f"  {sum(1 for e in events if e.kind.value == kind):3d}  {kind}")
    print(f"Hash chain sidecar: {CHAIN_PATH}")
    print(
        "\nNo Superseded example was produced: superseding needs a second experiment run and this "
        "repository has one.\nThe mechanism is proven by "
        "tests/htr/knowledge/test_lifecycle.py instead."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
