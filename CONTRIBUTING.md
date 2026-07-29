# Contributing to ArchiveTrust

ArchiveTrust is an accountable machine-reading system. Contributions are welcome when they preserve
the core architecture: provider independence, evidence-first reasoning, replayability, and a clear
separation between the Trust Engine and learning/research surfaces.

## License

Code contributions are accepted under the Apache License 2.0. By submitting a contribution, you
agree that it may be incorporated under the repository license unless a separate written agreement
exists.

Dataset contributions are different. Do not submit private, municipal, personal, copyrighted, or
PII-containing data unless a maintainer has explicitly approved the data license and handling plan.

## Before You Start

Read:

- `ARCHITECTURAL_CONSTITUTION.md`
- `README.md`
- `docs/MASTER_EXECUTION_PROGRAM_2026-07-16.md`
- `docs/DATA_HANDLING_POLICY.md` if your change touches data, exports, benchmarks, or corpora

Open an issue or design note before large changes, provider additions, telemetry schema changes,
or anything touching `domain/comparison/`, `domain/alignment/`, or `domain/confidence/`.

## Development Workflow

1. Keep changes narrowly scoped.
2. Add tests that exercise the behavior you changed.
3. Run the focused tests for the touched area.
4. Run dependency-direction or architecture tests when changing package boundaries.
5. Update docs when changing user-visible behavior, public APIs, telemetry, or governance policy.

Useful commands:

```powershell
pytest -q
pytest -q tests/domain/test_dependency_direction.py
git diff --check
```

## Architectural Guardrails

Do not:

- reason over raw provider output downstream of the adapter boundary;
- let provider-specific vocabulary leak into canonical or comparison layers;
- mutate Evidence, Observations, Canonical Observations, or Canonical Documents in place;
- add a canonical fact without an Evidence to Observation to Comparison to Canonical chain;
- use calibration or learning output to write back into production state without an explicit
  constitutional and policy amendment;
- implement roadmap milestones before their dependencies are complete.

Changes touching `domain/comparison/`, `domain/alignment/`, or `domain/confidence/` need especially
careful review because they can appear to work while weakening the scientific hypothesis.

## Tests and Review

Every contribution should include the smallest meaningful test set. For behavior changes, include a
regression test that would fail without the change.

Reviewers should check:

- the Constitution article most directly affected;
- dependency direction;
- replay compatibility;
- telemetry compatibility;
- whether the change creates a new source of truth.

## Constitutional Amendments

The Constitution changes rarely. A proposed amendment must cite evidence: an audit, incident,
benchmark, investigation, or accepted roadmap decision. See the governance section in
`ARCHITECTURAL_CONSTITUTION.md`.

## Security and Sensitive Data

Do not include secrets, live credentials, private archives, or PII in issues, pull requests, tests,
fixtures, screenshots, or benchmark artifacts. Use synthetic or cleared data.

Report sensitive findings privately to the maintainer until a dedicated security process exists.
