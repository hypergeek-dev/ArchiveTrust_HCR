"""Workspace Management (ROADMAP.md §5.13.1, Revision 5; Constitution Article 25) — infrastructure
only, a sibling package to `runtime/`.

A Workspace is a complete archival project: the highest-level operational unit, owning its own
configuration, telemetry, and storage independently of every other Workspace. This package never
imports `archivetrust.domain`, `archivetrust.providers`, `archivetrust.application`,
`archivetrust.learning`, or `archivetrust.review` — it composes `runtime/`'s existing
`DeploymentLayout` at a Workspace's own root rather than reimplementing it (Article 25: no global
singleton configuration; Article 20's provider-independence discipline extended to configuration
independence between Workspaces).
"""
