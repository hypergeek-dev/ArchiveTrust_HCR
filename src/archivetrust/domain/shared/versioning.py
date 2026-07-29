"""Version constants for the Canonical Observation Ontology and its serialization schema.

Ontology versioning (ROADMAP.md S5.4, Constitution Article 19) and schema versioning are
independent axes: ``ONTOLOGY_VERSION`` tracks the *meaning* of Observation payload types,
``SCHEMA_VERSION`` tracks the *serialization shape* used to persist them. Both must be recorded
on every Observation and Canonical Observation so replay can select the correct migration path
for each axis independently.
"""

from __future__ import annotations

CURRENT_ONTOLOGY_VERSION = 1
CURRENT_SCHEMA_VERSION = 1
