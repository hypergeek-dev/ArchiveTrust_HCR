"""The Learning Platform's analytics layer (ROADMAP_V2.md S13).

A dedicated analytics layer, architecturally separate from canonical archive data. Every analyzer
here is a **pure function of the Trust Engine telemetry stream** (`TelemetrySource`): given the
same stored events it yields the same result, and it holds no canonical copy of Evidence,
Observations, or the Canonical Document -- only derived, aggregated views reconstructible from the
stream at any time (S13: "append-only and reproducible from telemetry"; Guiding Principle 7).

No analyzer writes anything back to the Trust Engine. Output types (quality metrics, provider
health, evolution candidates) are advisory artifacts consumed by humans, never by the pipeline
(Guiding Principle 2).

Storage backends (PostgreSQL for production, SQLite for standalone deployments, ROADMAP_V2.md
S13) are a later, deliberately-deferred infrastructure milestone: because every view here is
reconstructible from the telemetry stream, persisting the *computed* views is an optimization, not
a correctness requirement, and is not built at this stage (Guiding Principle 10, sequential
milestones).
"""
