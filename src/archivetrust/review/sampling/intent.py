"""Review intent (Milestone 11, Phase 1): why a slot was placed in front of a reviewer.

Operational, calibration, and research review are separate objectives (Milestone 11 Guiding
Principles) and must stay distinguishable end to end -- from selection (`review/sampling/queue.py`)
through to which corrections may feed calibration statistics (`domain/calibration/progress.py`).
"""

from __future__ import annotations

from enum import Enum


class ReviewIntent(str, Enum):
    OPERATIONAL = "operational"
    """Correct today's archive. Driven by the current confidence policy, review thresholds, and
    production needs -- exactly today's `review/triage.py` behavior, unchanged."""

    CALIBRATION = "calibration"
    """Improve confidence calibration. Driven by statistical uncertainty, evidence gaps,
    calibration progress, and information gain (`domain/calibration/`), not by what a document
    happens to need fixed today."""

    RESEARCH = "research"
    """Evaluate new providers, new reasoning, rare evidence patterns, or newly discovered
    observation types. Not intended for routine production review."""
