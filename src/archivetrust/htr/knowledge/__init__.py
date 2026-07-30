"""Research-knowledge entities: `ResearchObservation`, `ResearchFinding`, and their lifecycle.

Layers 10-12 of `docs/architecture/htr-event-model.md` §1 -- the three layers the durable-telemetry
phase landed the event *schema* for and deliberately left producerless
(`docs/architecture/htr-telemetry.md` §6: "five event kinds have no producer ... by design, pending
the knowledge lifecycle"). This package is that lifecycle.

**Why a sibling package rather than living in `htr/experiment/` or `htr/corpus/`.** This codebase's
`htr/` convention is one package per concern, each owning its own `models.py`
(`htr/corpus/models.py`, `htr/experiment/models.py`, `htr/evaluation/*`). Research knowledge is a
distinct concern from experiment *execution*: an `ExperimentRun` records what a machine did, a
`ResearchFinding` records what a human is willing to claim, and event-model doc §1's hard rule is
that the second never follows automatically from the first. Folding findings into
`htr/experiment/models.py` would put an entity whose whole point is *not* being auto-derived from
execution into the module that models execution. `htr/knowledge/` keeps the seam visible.

The three modules:

* `models.py` -- the frozen entities (`ResearchObservation`, `ResearchFinding`, `ResearchScope`,
  `EvidenceReference`, `FindingRevision`, `ContradictoryEvidence`) and their enums. Imports nothing
  but `pydantic` and `domain.shared.ids`, which is what lets `domain/telemetry/events.py` embed them
  as typed objects rather than `record` dicts (`docs/architecture/htr-telemetry.md` §5's rule).
* `lifecycle.py` -- the status state machine and `transition_finding_status`, a pure function.
* `baseline_knowledge.py` -- the real observations and candidate findings extracted from the
  committed 2026-07-30 baseline run. Real ids, real measured values, no placeholders.
"""
