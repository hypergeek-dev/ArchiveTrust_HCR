"""The Human Review System's data layer (ROADMAP_V2.md S9).

Operational review produces attributable correction and effort telemetry, not evaluation truth.
Its records may inform workflow research, but they become evaluation references only through the
separate blinded assignment and adjudication workflow. This subpackage models the *passive telemetry*
that captures those byproducts without adding to the reviewer's work (S9.4), plus the effort
measures derived from it (S9).

This is the Learning Platform's own telemetry stream, deliberately separate from the Trust
Engine's frozen knowledge-evolution events -- see `archivetrust.learning` for why the two streams
must not be merged.
"""
