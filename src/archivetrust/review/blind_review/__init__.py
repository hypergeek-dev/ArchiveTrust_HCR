"""Blind dual-review workflow (Migration Plan Stage 10; `docs/htr-domain-design.md` §1/§5,
`docs/htr-transformation-audit.md`'s "blinded assignments -> two reviews -> adjudication").

This package is the SERVICE/workflow layer over `review/htr_models.py`'s already-implemented
Pydantic models (`ReviewAssignment`, `ReviewSubmission`, `AgreementResult`, `Adjudication`) -- it
does not redefine any of them. `htr_models.py` says explicitly that blind isolation and submission
locking are "enforced at the store/service layer (a later migration stage), not representable as a
field" on the models themselves. This package is that later stage.

**Why built on `review/htr_models.py`'s own enums, not `evaluation/ground_truth.py`'s.**
`evaluation/ground_truth.py` already defines `AdjudicationStatus` (UNADJUDICATED/ADJUDICATED/
DISPUTED), `VerificationStatus`, `ReviewerIndependenceState`, and `BlindingState` -- but these
describe one `GroundTruthAnnotation` record's own lifecycle in the *general* evaluation-reference
store (which also covers single-review, non-blind, and AI-assisted annotation paths). They are
coarser than what this task brief requires: `AdjudicationStatus` has no "minor" vs. "material"
distinction and no CER-based thresholds, and nothing in `ground_truth.py` computes agreement
between two specific submissions or preserves per-span disagreement locations. `review/
htr_models.py`'s `ReviewAssignment`/`ReviewSubmission`/`AgreementResult`/`Adjudication` are the
purpose-built shapes for exactly this workflow (two blind roles, one `AgreementResult` per target,
one optional `Adjudication`), so this package operates on those, and adds only what they do not yet
carry: a `BenchmarkStatus` classification (`agreement.py`), disagreement locations
(`agreement.py`), and an `ExclusionRecord` (`exclusion.py`, a genuinely new concept with no
existing analogue in either module). `evaluation/ground_truth.py` is left untouched; a
`GroundTruthAnnotation.annotation_id` is the natural value for `ReviewAssignment.target_ref`
(its own docstring already says so), so the two modules compose without either redefining the
other.

Submodules:

- `assignment.py` -- `create_blind_review_pair`: two independent `ReviewAssignment`s (reviewer A,
  reviewer B) for one target.
- `store.py` -- `BlindReviewStore`: the mechanical enforcement of blind isolation (a reviewer
  cannot read the other's submission before their own is finalized) and submission locking (a
  finalized submission is never mutated or replaced).
- `agreement.py` -- `BenchmarkStatus`, `AgreementPolicy` (documented CER thresholds),
  `compute_agreement`: builds an `AgreementResult` plus a benchmark classification and disagreement
  locations from two finalized submissions, reusing `htr/evaluation/recognition.py`'s CER/WER/
  edit-op primitives rather than reimplementing them.
- `adjudication.py` -- `adjudicate`: resolves a `Requires adjudication` item, recording an
  `Adjudication` with a required, non-empty rationale.
- `exclusion.py` -- `ExclusionRecord`, `exclude_from_benchmark`: excludes a target from the
  benchmark, only with a required, non-empty reason.
- `outcome.py` -- `BenchmarkOutcome`, `resolve_benchmark_outcome`: the final, queryable resolution
  for one target (exclusion always wins; otherwise the computed classification plus any
  adjudication), never overwriting reviewer A's, reviewer B's, or the adjudicator's original
  recorded result (Constitution Article 15).
- `queries.py` -- `reviewer_workload`, `batch_completion_status`: plain data-returning queries for
  a future Review Center UI. Disagreements are never dropped from these listings -- only an
  explicit `ExclusionRecord` removes a target from benchmark use, and even then it stays visible in
  `target_statuses`.
"""

from __future__ import annotations

from archivetrust.review.blind_review.adjudication import adjudicate
from archivetrust.review.blind_review.agreement import (
    AgreementAssessment,
    AgreementPolicy,
    BenchmarkStatus,
    DEFAULT_AGREEMENT_POLICY,
    DisagreementSpan,
    compute_agreement,
)
from archivetrust.review.blind_review.assignment import create_blind_review_pair
from archivetrust.review.blind_review.exclusion import ExclusionRecord, exclude_from_benchmark
from archivetrust.review.blind_review.outcome import BenchmarkOutcome, resolve_benchmark_outcome
from archivetrust.review.blind_review.queries import (
    BatchCompletionStatus,
    ReviewerWorkload,
    batch_completion_status,
    reviewer_workload,
)
from archivetrust.review.blind_review.store import (
    AlreadyFinalizedError,
    BlindIsolationError,
    BlindReviewStore,
    DuplicateExclusionError,
    UnknownAssignmentError,
)

__all__ = [
    "adjudicate",
    "AgreementAssessment",
    "AgreementPolicy",
    "BenchmarkStatus",
    "DEFAULT_AGREEMENT_POLICY",
    "DisagreementSpan",
    "compute_agreement",
    "create_blind_review_pair",
    "ExclusionRecord",
    "exclude_from_benchmark",
    "BenchmarkOutcome",
    "resolve_benchmark_outcome",
    "BatchCompletionStatus",
    "ReviewerWorkload",
    "batch_completion_status",
    "reviewer_workload",
    "AlreadyFinalizedError",
    "BlindIsolationError",
    "BlindReviewStore",
    "DuplicateExclusionError",
    "UnknownAssignmentError",
]
