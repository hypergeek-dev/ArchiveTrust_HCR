"""The Human Review System (HUMAN_REVIEW_SPECIFICATION.md; ROADMAP_V2.md §9).

**This package deliberately spans the Dual Architecture boundary** (HUMAN_REVIEW_SPECIFICATION.md
§0 Frozen Constraint 1, §3). A single review decision fans out into three effects across two
systems, and this package is the only component allowed to touch both sides:

- It **reads** the Trust Engine telemetry stream (via `application.journal` / a `TelemetrySource`)
  to assemble a `ReviewPacket` — read-only (LP-2).
- On a decision, it invokes the **Trust Engine's** frozen Milestone-6 correction contract
  (`domain.feedback`), which supersedes a `CanonicalObservation` *as new human evidence* — a Trust
  Engine operation exercised by an authorized human decision, never an out-of-band edit (§3
  effects A/B). This is why the review services live here and not in `archivetrust.learning`,
  whose invariants forbid holding a write path to any Trust Engine domain object (`ROADMAP_V2.md`
  LP-1/LP-5).
- It emits the **Learning Platform's** own passive review telemetry
  (`learning.review.ReviewInteractionSink`) — a separate, temporal stream (§3 effect C, LP-3).

**Dependency direction.** This package may import both `archivetrust.domain` / `.application`
(Trust Engine) and `archivetrust.learning` (Learning Platform). Nothing in the Trust Engine may
import this package — the same one-directional rule that protects `learning` (`ROADMAP_V2.md`
LP-1), enforced by `tests/review/test_separation.py`.

**What lives here vs. above it.** This package is framework-independent Python: it defines the
service contracts (`ReviewService`), the read-only presentation data the interface consumes
(`ReviewPacket`), triage, and packet assembly. It contains no GUI code. ViewModels and the PySide6
client sit *above* it and consume these services (the GUI → ViewModels → Services → Trust Engine
layering); they never reach around it into the Trust Engine directly.
"""
