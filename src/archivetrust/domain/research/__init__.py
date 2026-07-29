"""Research Telemetry (Constitution Article 34, `ARCHITECTURE_TELEMETRY_STANDARD.md` §9).

ArchiveTrust's third telemetry ontology -- self-knowledge about its own reasoning across a corpus
(audits, benchmarks, investigatory replays), deliberately independent of both document-scoped Trust
Engine telemetry (`domain/telemetry/`) and Workspace-scoped Acquisition telemetry
(`acquisition/events.py`). See `events.py` for the event vocabulary and `sink.py` for the
persistence Protocol.
"""

from __future__ import annotations
