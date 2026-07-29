"""Passive review telemetry (ROADMAP_V2.md S9.4: "observe rather than ask").

Every interaction a reviewer has with one uncertainty is recorded automatically, at the moment it
happens, requiring zero reviewer effort -- review duration, edit duration, accept-vs-edit outcome,
undo/zoom/pan/overlay/navigation behavior. Behavior recorded live is more honest, more complete,
and more comparable across reviewers than a post-hoc questionnaire (S9.4, Guiding Principle 1);
these events are the raw material every downstream effort and Evolution-Center signal is computed
from.

**Why these events carry a timestamp when no Trust Engine domain type does.** The Trust Engine is
timeless by design -- its replay excludes wall-clock time so it stays deterministic (ROADMAP.md
Milestone 7). The Learning Platform's review telemetry is the opposite: its entire research value
is *timing* (how long a reviewer looked, how long an edit took). Recording time here does not
weaken Trust Engine determinism, because this is a separate stream the Trust Engine never reads
(ROADMAP_V2.md S5.2). Reproducibility (Guiding Principle 7) still holds: a `ReviewSession`
recomputed from the same stored interactions is byte-identical every time.

`timestamp` is monotonic seconds (e.g. `time.monotonic()`), not a calendar time -- durations are
the only thing any consumer derives from it, and monotonic time is immune to clock adjustments
that would corrupt a duration.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict

LEARNING_TELEMETRY_SCHEMA_VERSION = 1
"""Versioned independently of the Trust Engine's `schema_version` (ROADMAP_V2.md Guiding
Principle 4 / Guiding Principle 7): the review-telemetry shape may evolve without touching the
Trust Engine's frozen event set, and every long-term metric must know which shape it was computed
from."""


class ReviewInteractionKind(str, Enum):
    """What the reviewer did. Extensible with an `OTHER` catch-all, following the same
    extensible-enum precedent as `CorrectionCategory`/`RiskType` in the Trust Engine -- an
    unenumerated interaction must never be unrepresentable, or passive telemetry would silently
    drop exactly the novel behavior the Evolution Center exists to notice.
    """

    REVIEW_OPENED = "review_opened"
    """The reviewer began looking at one uncertainty (starts the review-duration clock)."""

    REVIEW_COMPLETED = "review_completed"
    """The reviewer resolved the uncertainty (stops the review-duration clock)."""

    VALUE_ACCEPTED = "value_accepted"
    """The reviewer accepted a provider's presented value -- the fast path (S9.2, GP 9)."""

    EDIT_STARTED = "edit_started"
    """The reviewer began entering a value no provider presented (starts edit-duration clock)."""

    EDIT_COMMITTED = "edit_committed"
    """The reviewer committed a manual edit (stops edit-duration clock)."""

    EDIT_UNDONE = "edit_undone"
    """The reviewer undid an edit -- friction signal, not just a keystroke (S9.4)."""

    ZOOMED = "zoomed"
    PANNED = "panned"
    OVERLAY_TOGGLED = "overlay_toggled"
    """The reviewer consulted a provider/confidence/agreement overlay before deciding (S9.3)."""

    NAVIGATED = "navigated"
    """The reviewer moved context (a cost the interface aims to minimize, S9.1)."""

    OTHER = "other"


class ReviewInteraction(BaseModel):
    """One recorded interaction within the review of one uncertainty.

    `review_id` groups every interaction belonging to a single uncertainty's review (one cluster /
    Canonical Observation, S9.2). `target_canonical_observation_id` ties the review back to the
    exact Trust Engine artifact under review, so effort and outcome can be attributed per
    observation-type and per provider downstream -- without the Learning Platform ever writing to
    that Canonical Observation.
    """

    model_config = ConfigDict(frozen=True)

    interaction_id: str
    review_id: str
    target_canonical_observation_id: str
    reviewer_ref: str
    kind: ReviewInteractionKind
    timestamp: float
    schema_version: int = LEARNING_TELEMETRY_SCHEMA_VERSION


class ReviewerMetadata(BaseModel):
    """Optional, lightweight, never-blocking self-report (ROADMAP_V2.md S9.5).

    Deliberately separate from `ReviewInteraction`: passive interactions are the primary record
    (S9.4), and none of these fields may ever gate completing a review action. All three are
    nullable so a reviewer who supplies nothing is fully represented -- the absence is itself
    honest data, not a hole to be filled.

    `failure_category` reuses the Trust Engine's `CorrectionCategory` string values rather than
    inventing a parallel taxonomy (ROADMAP_V2.md S9.5: "the same structured taxonomy Milestone 6
    already defines"). Stored as a bare string here to keep this package's optional-metadata layer
    from importing the feedback domain's enum as a hard structural dependency.
    """

    model_config = ConfigDict(frozen=True)

    review_id: str
    reviewer_confidence: float | None = None
    failure_category: str | None = None
    rationale: str | None = None
    schema_version: int = LEARNING_TELEMETRY_SCHEMA_VERSION
