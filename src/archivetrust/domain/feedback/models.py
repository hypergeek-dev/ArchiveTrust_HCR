"""The Human Feedback domain model (ROADMAP.md Milestone 6, S1 Vision item 8).

**No prior document in this repository enumerates a categories/actions taxonomy.**
ROADMAP.md S8's Milestone 6 acceptance criteria says "all listed structured categories and
actions," but `MILESTONE0_REVIEW.md` W6, `ARCHITECTURE_VALIDATION_REPORT.md`, and
`ARCHITECTURE_READINESS_REVIEW.md` all independently and consistently state that
`HumanCorrection`/`DatasetCandidate` are telemetry event names with no domain object behind them,
and that "Milestone 6 will need to design these from scratch" (`MILESTONE0_REVIEW.md` W6). This
module is that design, originated here per that explicit license -- not a lookup of something that
was supposed to already exist. See IMPLEMENTATION_STATUS.md for the full citation trail.

`CorrectionCategory` and `CorrectionAction` follow the same extensible-enum-with-catch-all shape
already established for `RiskType` (`domain/comparison/reporting.py`) -- an unenumerated
correction type must never be unrepresentable, consistent with that precedent.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict


class CorrectionCategory(str, Enum):
    """What aspect of the canonical fact was wrong -- grounded in concepts the domain model
    already distinguishes (payload content vs. classification vs. graph structure vs. clustering/
    segmentation), so this taxonomy adds no new concept, only names which existing layer a human
    reviewer identified a problem in.
    """

    TRANSCRIPTION_ERROR = "transcription_error"
    """The accepted text/value itself is wrong (S5's text-reconciliation territory)."""

    CLASSIFICATION_ERROR = "classification_error"
    """Wrong `observation_type`/payload kind (e.g. a Heading should have been a Caption)."""

    STRUCTURAL_ERROR = "structural_error"
    """Wrong parent/child/related edge (S4's structural-reconciliation territory)."""

    SEGMENTATION_ERROR = "segmentation_error"
    """Wrong clustering or table-lattice segmentation -- should have been merged/split
    differently (S2, S6's territory)."""

    MISSING_CONTENT = "missing_content"
    """Something present in the archive was never captured by any provider."""

    SPURIOUS_CONTENT = "spurious_content"
    """The canonical fact asserts something not actually present (e.g. a VLM hallucination)."""

    DIFFERENT_THINGS = "different_things"
    """Both readings may be valid, but they describe different archive objects/regions/scopes."""

    METADATA_ERROR = "metadata_error"

    OTHER = "other"


class CorrectionAction(str, Enum):
    """What the human reviewer did about it. `MERGE`/`SPLIT` are recorded as legitimate proposals
    but their structural execution (rewriting `ReconciledObservationGraph` edges across multiple
    Canonical Observations) is explicitly out of Milestone 6's scope -- see
    `feedback/engine.py`'s module docstring -- consistent with ROADMAP.md's own Risk-table entry:
    "Milestone 6 ships the data model + capture contract only; a full review UI is a separate,
    later, explicitly scoped effort."
    """

    ACCEPT = "accept"
    REJECT = "reject"
    EDIT = "edit"
    MERGE = "merge"
    SPLIT = "split"
    FLAG_FOR_REVIEW = "flag_for_review"
    DIFFERENT_THINGS = "different_things"


class HumanCorrection(BaseModel):
    """One human review action against one semantic slot's current Canonical Observation.

    Required fields per ROADMAP.md S8's explicit list: category, action, raw AI output, raw
    corrected output, rationale. `raw_ai_output` is the verbatim text the system had accepted
    before this correction; `raw_corrected_output` is the human's replacement text, required for
    `EDIT` and meaningless (left `None`) for `ACCEPT`/`REJECT`/`FLAG_FOR_REVIEW`.
    """

    model_config = ConfigDict(frozen=True)

    correction_id: str
    target_canonical_observation_id: str
    category: CorrectionCategory
    action: CorrectionAction
    raw_ai_output: str
    raw_corrected_output: str | None = None
    rationale: str | None = None


class DatasetCandidate(BaseModel):
    """A pointer marking one `HumanCorrection` as a candidate training/calibration example.

    Deliberately minimal: Dataset Generation itself is a documented, not-yet-built future
    milestone (ROADMAP.md S14) -- this type is the seam Milestone 6 is asked to leave (Guiding
    Principle 9), not an implementation of dataset export/curation logic.
    """

    model_config = ConfigDict(frozen=True)

    candidate_id: str
    correction_id: str
    archive_object_ref: str
    description: str
