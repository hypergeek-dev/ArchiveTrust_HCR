#!/usr/bin/env python
"""Raises the one real research question the committed baseline knowledge supports, and drafts one
real, unexecuted experiment from it.

Phase 11 of `docs/knowledge-lifecycle.md`'s "Known gaps for the next phase": "the observation ->
question -> experiment-draft feedback loop".

**Reads the committed logs; does not regenerate them.** The question links to the observation and
finding the documentation actually quotes
(`research_observation_067cd6a4ae80439289aa8256efffa835` and
`research_finding_0f77a4f5443e4e79b429d8076596c4eb`), so this script replays
`htr_research_events.jsonl` + `htr_knowledge_events.jsonl` and takes the entities it finds there. It
never runs `register_baseline_knowledge` itself: doing so would mint fresh observation and finding ids
and silently invalidate every id quoted in `docs/research-observations.md`,
`docs/research-findings.md` and `docs/knowledge-lifecycle.md`.

**A fourth stream, for the same reason there was a third.** These records go to
`htr_knowledge_feedback_events.jsonl`, beside the other two, because
`docs/architecture/htr-telemetry.md` §12 documents `htr_knowledge_events.jsonl` as exactly 18 events
of 4 kinds and `tests/htr/knowledge/test_baseline_knowledge.py` reads it as that. Appending would
falsify its own description. Same `FileTelemetrySink` class, same append-only guarantees, same
hash-chain sidecar -- §7's "two streams, one mechanism", now four. The `causation_id` points across
into the knowledge log, exactly as that log's own point across into the run log, and
`HtrJournal().replay(run + knowledge + feedback)` reconstructs the whole graph.

**Selection is by observation type, not by a pasted id.** The confidence-anomaly observation is found
by `ObservationType.CONFIDENCE_ANOMALY` and the script fails if the committed log does not contain
exactly one, rather than trusting a hardcoded id that could drift from the artifact.

Nothing is executed: the drafted `ExperimentVersion` has no `ExperimentRun`, no method is invoked, and
no metric is computed.

Usage (from the repo root):

    PYTHONPATH=src .venv/Scripts/python.exe scripts/register_research_question.py [--fresh]
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import chain
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from archivetrust.application.htr_journal import HtrJournal  # noqa: E402
from archivetrust.htr.knowledge.baseline_knowledge import EXTRACTED_AT  # noqa: E402
from archivetrust.htr.knowledge.models import ObservationType  # noqa: E402
from archivetrust.htr.knowledge.registration import register_feedback_loop  # noqa: E402
from archivetrust.htr.persistence import DurableHtrResearchStore  # noqa: E402
from archivetrust.infrastructure.storage.telemetry_sink import FileTelemetrySink  # noqa: E402

OUTPUT_DIR = REPO_ROOT / "docs" / "experiments" / "baseline-comparison"
RUN_PATH = OUTPUT_DIR / "htr_research_events.jsonl"
KNOWLEDGE_PATH = OUTPUT_DIR / "htr_knowledge_events.jsonl"
FEEDBACK_PATH = OUTPUT_DIR / "htr_knowledge_feedback_events.jsonl"
CHAIN_PATH = OUTPUT_DIR / "htr_knowledge_feedback_events.jsonl.chain.jsonl"

DEFAULT_ASKER = "hypergeek-dev"
"""The repository maintainer, who is the human actually raising this question.

Attributed honestly rather than with a placeholder, the same decision
`scripts/register_baseline_knowledge.py` makes about its reviewer. `created_by` is a required argument
with no default on `register_feedback_loop` precisely so this had to be a decision.
"""


def _prepare(*, fresh: bool) -> None:
    if not FEEDBACK_PATH.exists():
        return
    if not fresh:
        raise SystemExit(
            f"Refusing to run: {FEEDBACK_PATH} already exists.\n"
            "  Pass --fresh to delete it and record a single clean pass (what you want for a\n"
            "  committed artifact). Appending would put two questions about one observation in one\n"
            "  file with nothing saying which the documentation quotes."
        )
    for path in (FEEDBACK_PATH, CHAIN_PATH):
        if path.exists():
            path.unlink()
    print(f"      --fresh: removed the previous feedback log at {FEEDBACK_PATH}")


def _committed_knowledge():
    """The committed run + knowledge logs, replayed into one projection.

    Both are needed: the observation and finding live in the knowledge log, but the evidence they
    reference (metric results, method runs, the input crop) lives only in the run log, and a
    projection holding one without the other could not resolve an evidence reference.
    """
    for path in (RUN_PATH, KNOWLEDGE_PATH):
        if not path.exists():
            raise SystemExit(
                f"Missing committed log {path}. Run scripts/run_baseline_comparison.py and "
                "scripts/register_baseline_knowledge.py first."
            )
    run_events = tuple(FileTelemetrySink(RUN_PATH).all_events())
    knowledge_events = tuple(FileTelemetrySink(KNOWLEDGE_PATH).all_events())
    projection = HtrJournal().replay(chain(run_events, knowledge_events))
    return projection, run_events, knowledge_events


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh", action="store_true", help="delete an existing feedback log first")
    parser.add_argument("--asker", default=DEFAULT_ASKER, help="attributable human asker")
    args = parser.parse_args()

    _prepare(fresh=args.fresh)
    projection, _run_events, knowledge_events = _committed_knowledge()

    anomalies = projection.research_observations(
        observation_type=ObservationType.CONFIDENCE_ANOMALY.value
    )
    if len(anomalies) != 1:
        raise SystemExit(
            f"Expected exactly one {ObservationType.CONFIDENCE_ANOMALY.value} observation in the "
            f"committed knowledge log, found {len(anomalies)}. This script raises a question about "
            "that specific recorded gap; it will not guess which one."
        )
    observation = anomalies[0]
    findings = [
        finding
        for finding in projection.findings()
        if observation.observation_id in finding.supporting_observations
        and len(finding.supporting_observations) == 1
    ]
    if len(findings) != 1:
        raise SystemExit(
            f"Expected exactly one committed finding resting solely on {observation.observation_id}, "
            f"found {len(findings)}."
        )
    finding = findings[0]

    causing_event = next(
        (
            event
            for event in knowledge_events
            if getattr(event, "finding_ref", None) == finding.finding_id
            and event.kind.value == "CandidateFindingCreated"
        ),
        None,
    )
    if causing_event is None:
        raise SystemExit(
            f"No CandidateFindingCreated for {finding.finding_id} in {KNOWLEDGE_PATH}: the causal "
            "edge this question hangs off does not exist, so it will not be invented."
        )

    sink = FileTelemetrySink(FEEDBACK_PATH)
    store = DurableHtrResearchStore(sink, actor_id="scripts.register_research_question")
    # The projection is hydrated from the committed logs so the store refuses a duplicate registration
    # and so `advance_research_question`'s "was it ever registered?" check is meaningful. Adopting the
    # projection deliberately bypasses the emitting `register_*` overrides -- see
    # `HtrResearchStore.adopt_projection`; replaying through them would append a second copy of every
    # committed event into this new file.
    store.adopt_projection(projection)

    loop = register_feedback_loop(
        store,
        observation=observation,
        finding=finding,
        created_by=args.asker,
        at=EXTRACTED_AT,
        caused_by=causing_event.event_id,
    )

    print("Research question raised from the committed baseline knowledge:")
    print(f"  question   {loop.question.question_id}")
    print(f"  event      {loop.question_event_id}")
    print(f"  caused by  {causing_event.event_id}  (CandidateFindingCreated)")
    print(f"  from obs   {loop.question.originating_observation_id}")
    print(f"  from find  {loop.question.originating_finding_id}")
    print(f"  statement  {loop.question.statement}")
    for hypothesis in loop.question.hypotheses:
        print(f"  hypothesis {hypothesis.hypothesis_id}")
        print(f"      {hypothesis.statement}")
        print(f"      refuted if: {hypothesis.falsification_criterion}")

    draft = loop.draft
    print("\nDrafted (UNEXECUTED) experiment:")
    print(f"  experiment {draft.experiment.experiment_id}  {draft.experiment.name}")
    print(f"  event      {loop.experiment_event_id}")
    print(f"  version    {draft.experiment_version.experiment_version_id} "
          f"(v{draft.experiment_version.version})")
    print(f"  event      {loop.experiment_version_event_id}")
    print(f"  link event {loop.draft_event_id}  (ExperimentDraftedFromQuestion)")
    print(f"  methods    {', '.join(draft.experiment_version.method_ids)}")
    print(f"  revisions  {draft.definition.method_versions}")
    print(f"  dataset v  {draft.experiment_version.dataset_version_id}")
    print(f"  runs       {len(store.experiment_runs(experiment_version_id=draft.experiment_version.experiment_version_id))}"
          "  (a draft has none, by construction)")
    ref = json.loads(draft.experiment_version.pipeline_configuration_ref)
    print(f"  ref names  research_question_id={ref['research_question_id']}")
    print(f"             originating_observation_id={ref['originating_observation_id']}")
    print(f"             originating_finding_id={ref['originating_finding_id']}")
    print(f"  question   {loop.draft.question.status.value}")

    events = tuple(sink.all_events())
    kinds = sorted({event.kind.value for event in events})
    print(f"\nWrote {len(events)} feedback events ({len(kinds)} kinds) to {FEEDBACK_PATH}")
    for kind in kinds:
        print(f"  {sum(1 for e in events if e.kind.value == kind):3d}  {kind}")
    print(f"Hash chain sidecar: {CHAIN_PATH}")
    print(
        "\nNo ExperimentRun was created and no inference was performed. Answering this question "
        "needs a\nlarger ground-truthed corpus than this repository contains -- stated in the "
        "draft's own\nsampling_strategy and scope_caveats."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
