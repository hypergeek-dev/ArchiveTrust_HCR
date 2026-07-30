#!/usr/bin/env python
"""Durably records the SATRN output-repetition observation from the bounded diagnostic.

Reads the real diagnostic artifact
(`docs/experiments/technical-reliability-screening/diagnostic/satrn_repetition_diagnostic.json`),
verifies `htr/knowledge/satrn_repetition_knowledge.py` has not drifted from it, then appends the
experiment records and the `ResearchObservation` to a **new** telemetry stream.

**Nothing existing is overwritten.** The Checkpoint 3 smoke test's `smoke_test_events.jsonl`,
`smoke_test_events.jsonl.chain.jsonl` and `smoke_test_results.json` are opened read-only here and by
the diagnostic; this script writes only to `diagnostic/htr_knowledge_events.jsonl`. The
`ExperimentVersion` created below is version 1 of a *new* `Experiment` for the diagnostic itself and
supersedes nothing -- the smoke test never created an `Experiment`, so there is nothing to supersede
and nothing to mutate. `assert_experiment_mutable` is called anyway, before the version is created,
so the supersede-not-overwrite discipline is enforced by the same mechanism the rest of the codebase
uses rather than by this docstring asserting it.

Re-running this script is refused if the stream already exists, so a second run cannot silently
append a duplicate observation with a fresh id.

Run:

    PYTHONPATH=src .venv/Scripts/python.exe scripts/register_satrn_repetition_observation.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from archivetrust.application.htr_journal import HtrJournal  # noqa: E402
from archivetrust.htr.corpus.models import (  # noqa: E402
    Collection,
    Dataset,
    DatasetVersion,
    ResearchProject,
)
from archivetrust.htr.experiment.models import (  # noqa: E402
    Experiment,
    ExperimentRun,
    ExperimentVersion,
    assert_experiment_mutable,
)
from archivetrust.htr.knowledge.satrn_repetition_knowledge import (  # noqa: E402
    DIAGNOSTIC_REPORT,
    FLORENCE2_METHOD_ID,
    SATRN_METHOD_ID,
    satrn_output_repetition_observation,
    verify_against_diagnostic_report,
)
from archivetrust.htr.persistence import DurableHtrResearchStore  # noqa: E402
from archivetrust.infrastructure.storage.telemetry_sink import FileTelemetrySink  # noqa: E402

DIAGNOSTIC_DIR = REPO_ROOT / "docs" / "experiments" / "technical-reliability-screening" / "diagnostic"
STREAM_PATH = DIAGNOSTIC_DIR / "htr_knowledge_events.jsonl"
SMOKE_DIR = REPO_ROOT / "docs" / "experiments" / "technical-reliability-screening" / "smoke-test"
ACTOR = "scripts/register_satrn_repetition_observation.py"


def main() -> int:
    report_path = REPO_ROOT / DIAGNOSTIC_REPORT
    if not report_path.exists():
        print(f"FATAL: diagnostic report not found at {report_path}")
        print("Run scripts/run_satrn_repetition_diagnostic.py first.")
        return 1
    report = json.loads(report_path.read_text(encoding="utf-8"))

    problems = verify_against_diagnostic_report(report)
    if problems:
        print("FATAL: knowledge module has drifted from the diagnostic artifact:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(f"knowledge module verified against {DIAGNOSTIC_REPORT}: no drift")

    if STREAM_PATH.exists():
        print(f"FATAL: {STREAM_PATH} already exists -- refusing to append a duplicate observation.")
        print("Delete it deliberately if you really intend to re-register.")
        return 1

    # Prove the smoke-test evidence is untouched, before and after.
    smoke_files = sorted(path for path in SMOKE_DIR.glob("*") if path.is_file())
    before = {path.name: (path.stat().st_size, path.stat().st_mtime_ns) for path in smoke_files}

    now = datetime.now(timezone.utc).isoformat()
    store = DurableHtrResearchStore(FileTelemetrySink(STREAM_PATH), actor_id=ACTOR)

    project = ResearchProject.create(
        name="Swedish Historical HTR Technical Reliability Screening",
        description=(
            "Reference-free technical-reliability screening of HTR methods on the 766-page Swedish "
            "witchcraft-trial court-record corpus. No ground truth exists for this corpus, so no "
            "CER/WER/accuracy is computed at any stage."
        ),
        created_at=now,
    )
    store.register_project(project)

    dataset = Dataset.create(
        project_id=project.project_id,
        name="dataset-rgb (Swedish witchcraft-trial court records)",
        description=(
            "The real 766-page corpus at the repository root. This diagnostic reads 5 of its pages, "
            "via the 15 line crops the Checkpoint 3 smoke test already cut from them."
        ),
        created_at=now,
    )
    store.register_dataset(dataset)

    collection = Collection.create(
        dataset_id=dataset.dataset_id,
        name=(
            "Checkpoint 3 smoke-test pages -- 2 structurally difficult spreads, 2 ordinary pages, "
            "and the corpus's single technically difficult page"
        ),
        created_at=now,
    )
    store.register_collection(collection)

    dataset_version = DatasetVersion.create(
        dataset_id=dataset.dataset_id,
        version=1,
        collection_ids=(collection.collection_id,),
        created_at=now,
    )
    store.register_dataset_version(dataset_version)

    experiment = Experiment.create(
        name="SATRN output-repetition diagnostic (Checkpoint 3 follow-up)",
        research_project_id=project.project_id,
        description=(
            "Bounded investigation of one signal from the Checkpoint 3 smoke test: SATRN returning "
            "the identical string for 6 of 15 byte-distinct line crops. Tests the integration-defect "
            "hypotheses (temp-file reuse, stale subprocess state, cross-call model state, parsing "
            "fallback, IPC response reuse) against real evidence. Not a benchmark and not an "
            "accuracy evaluation."
        ),
        created_at=now,
    )
    store.register_experiment(experiment)

    # The supersede-not-overwrite guard, run *before* creating the version. This experiment is new,
    # so it has no runs and no versions and the call passes trivially -- which is the point: the
    # discipline is exercised by the same mechanism every other stage uses, not asserted in prose.
    assert_experiment_mutable(
        experiment.experiment_id, store.experiment_runs(), store.experiment_versions()
    )

    version = ExperimentVersion.create(
        experiment_id=experiment.experiment_id,
        version=1,
        dataset_version_id=dataset_version.dataset_version_id,
        method_ids=(SATRN_METHOD_ID, FLORENCE2_METHOD_ID),
        segmentation_configuration_ref=(
            "nazounoryuu/florence_base__mixed__page__line_od -- crops reused byte-identically from "
            "the Checkpoint 3 smoke test, not re-detected"
        ),
        pipeline_configuration_ref=DIAGNOSTIC_REPORT,
        created_at=now,
        supersedes=None,
    )
    store.register_experiment_version(version)

    run = ExperimentRun.create(
        experiment_version_id=version.experiment_version_id,
        is_end_to_end=False,
        started_at=report["generated_at"],
        completed_at=now,
    )
    with store.correlated_to(run.experiment_run_id):
        run_event = store.register_experiment_run(run)

        observation = satrn_output_repetition_observation(
            experiment_id=experiment.experiment_id,
            experiment_version_id=version.experiment_version_id,
            experiment_run_id=run.experiment_run_id,
            dataset_id=dataset.dataset_id,
            dataset_version_id=dataset_version.dataset_version_id,
        )
        store.register_research_observation(observation, caused_by=run_event)
        store.complete_experiment_run(run)

    after = {path.name: (path.stat().st_size, path.stat().st_mtime_ns) for path in smoke_files}
    untouched = before == after

    # Durability: throw the in-process store away and rebuild from the file alone.
    del store
    replayed = HtrJournal().replay(FileTelemetrySink(STREAM_PATH).all_events())
    replayed_observations = list(replayed.research_observations())
    round_tripped = (
        len(replayed_observations) == 1 and replayed_observations[0] == observation
    )

    print()
    print("=" * 100)
    print("REGISTERED -- SATRN output-repetition observation")
    print("=" * 100)
    print(f"  project            {project.project_id}")
    print(f"  dataset            {dataset.dataset_id}")
    print(f"  dataset_version    {dataset_version.dataset_version_id}")
    print(f"  experiment         {experiment.experiment_id}")
    print(
        f"  experiment_version {version.experiment_version_id}  "
        f"(version {version.version}, supersedes {version.supersedes})"
    )
    print(f"  experiment_run     {run.experiment_run_id}")
    print(f"  observation        {observation.observation_id}")
    print(f"  type               {observation.observation_type.value}")
    print(f"  confidence         {observation.observation_confidence.value}")
    print(f"  review_status      {observation.review_status.value}")
    print(f"  evidence refs      {len(observation.supporting_evidence)}")
    print(f"  scope sample_size  {observation.scope.sample_size}")
    print()
    print(f"  scope: {observation.scope.describe()}")
    print()
    print(f"  stream                              {STREAM_PATH.relative_to(REPO_ROOT)}")
    print(f"  events written                      {len(FileTelemetrySink(STREAM_PATH).all_events())}")
    print(f"  replay round-trips the observation  {round_tripped}")
    print(f"  smoke-test files untouched          {untouched}  ({len(smoke_files)} files checked)")
    for name, (size, _) in sorted(before.items()):
        print(f"      {name:<44} {size:>8} bytes  unchanged={before[name] == after[name]}")
    return 0 if (round_tripped and untouched) else 1


if __name__ == "__main__":
    raise SystemExit(main())
