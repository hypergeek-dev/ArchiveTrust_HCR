"""Alignment Observability (ROADMAP.md S5.12; Constitution Article 24).

Makes the Comparison Engine's grouping step — the hypothesis that the Observations entering a
comparison group refer to the same semantic object — an explicit, first-class, replayable artifact,
rather than a silent assumption. This package does **not** improve, replace, or redesign the
grouping algorithm: the existing clustering (`domain.comparison.clustering`) remains the sole,
authoritative grouping mechanism, byte-for-byte unchanged. This package only *records what that
mechanism already decides*, behind a stable, replaceable seam.

Two pieces:
- `service.AlignmentService` — the stable interface between Observations and Comparison. The
  Comparison Engine consumes an `AlignmentService`'s output instead of calling clustering directly,
  so a future alignment strategy (heuristic, PDF-anchor, graph-based, ML-assisted) can replace the
  current one without any change to downstream systems.
- `models.AlignmentResult` — the observability record: one `AlignmentAttempt` per comparison group
  (candidates considered, Observations selected, rationale, algorithm version), and every
  Observation's terminal `AlignmentOutcome` (Aligned / Unaligned) so none silently disappears.

Explicitly out of scope (ROADMAP.md S5.12 non-goals): no Alignment Confidence, no scoring, no new
heuristic, no ML. Those are future milestones, after empirical evidence from this layer exists.
"""
