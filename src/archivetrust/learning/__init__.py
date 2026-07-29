"""The Learning Platform (ROADMAP_V2.md S5.2).

The Learning Platform is the second of ArchiveTrust's two independent, cooperating systems. It
*consumes* the telemetry and canonical artifacts the Trust Engine produces (ROADMAP.md S12,
S5.10) and produces *understanding about* that knowledge -- provider health, quality trends,
ontology/prompt pressure, and evidence-backed evolution candidates.

**The separation contract (ROADMAP_V2.md S5, Guiding Principles 2 and 6):**

- This package may import Trust Engine domain *types* (to read telemetry) but must never construct,
  mutate, or persist a Canonical Document, Canonical Observation, Evidence record, or any other
  Trust Engine domain object. It has no write path into the Trust Engine, by design.
- Nothing in the Trust Engine (`archivetrust.domain`, `.providers`, `.application`,
  `.infrastructure`) may import from this package. The dependency is strictly one-directional so
  the Trust Engine can be run with the Learning Platform deleted, disabled, or broken and be
  entirely unaffected (ROADMAP_V2.md S5.1). `tests/learning/test_separation.py` enforces both
  halves of this.
- The only output path back toward production is an *evidence-backed candidate*
  (`archivetrust.learning.analytics.evolution.EvolutionCandidate`) -- advisory input to a human
  architectural decision, never an automatic write (ROADMAP_V2.md S7, S11).

**Two telemetry streams, kept distinct.** The Trust Engine's knowledge-evolution telemetry
(`archivetrust.domain.telemetry`) is frozen and timeless -- replay excludes wall-clock time for
determinism (ROADMAP.md S12, Milestone 7). The Learning Platform has its *own* passive-telemetry
stream for human review (`archivetrust.learning.review`), which is fundamentally temporal (review
duration, edit duration). Mixing the two would either pollute the Trust Engine's frozen event set
with process-timing (forbidden by ROADMAP.md S12) or force the Learning Platform to discard the
timing its research mission depends on. Their separation is exactly the Dual Architecture (S5)
made concrete at the telemetry layer.
"""
