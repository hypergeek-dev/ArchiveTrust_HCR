# Loghi Full-Corpus Retraining Readiness Audit — Revised

**This document updates `retraining_readiness_audit.md` after an independent adversarial review
(`opus_independent_review.md`). The original is preserved unmodified. Read the independent review for
full evidence and reasoning — this document summarizes the resulting change in decision.**

**Revision date:** 2026-08-01
**Code state:** commit `bf9a53a8863d6ad4f4faf74b38e2aa9e9cc8fcf0`
**Fresh prepared run:** `training/full-corpus-20260801T213023Z`

## Revised decision

## RETRAINING READINESS DECISION: CONDITIONAL GO

*(was: NO-GO)*

The original NO-GO was driven by one CRITICAL, quantified finding: insufficient disk capacity
(R-018). The operator has since freed disk space; this review independently measured the result
(322.73GB free, not merely accepted the operator's ~300GB claim) and recalculated the requirement
from a more precise, non-averaged per-shard figure (127.44GB for the full 171-shard plan). The
capacity blocker is resolved with a 61% margin.

During independent verification, this review also found and fixed two more real, launch-relevant
gaps (random-seed validation on resume; a self-referential shard-integrity check), and — most
significantly — found that the original audit's characterization of learning-rate/optimizer
continuity as "architecturally sound" was **empirically wrong**, via a real, bounded test against
already-existing pilot checkpoints (zero new training). The corrected finding is more severe in what
it reveals (no cross-shard optimizer/LR continuity exists, ever) but **not more severe in
consequence**, because the pilot's own real, completed, successful 22-epoch run is direct proof the
same mechanism already works.

No remaining finding meets the bar for a launch blocker. Remaining open items are governance
(success criteria, approving authority), scientific-interpretation (document-level leakage
disclosure, LR-policy relabeling), and deferrable hardening (checkpoint retention, near-duplicate
detection) — exactly the category compatible with CONDITIONAL GO under the review's own decision
rules, provided human review gates exist after the first shard and first full corpus lap. Both gates
are specified in the accompanying checklists.

## What changed since the original audit

| Item | Original audit | This revision |
|---|---|---|
| Disk capacity | ~95.4GB free vs ~124.4GB required — CRITICAL blocker | ~322.73GB free vs ~127.44GB required — 61% margin, RESOLVED |
| Resume of terminal (COMPLETED/FAILED) run | Fixed during the original audit | Reconfirmed working |
| Code-revision drift at launch | Fixed during the original audit | Reconfirmed working; caught its own evidence run going stale a second time |
| Random-seed drift on resume | Open (R-006) | **Fixed** in this revision |
| Shard-integrity self-reference at launch | Open (R-007) | **Fixed** in this revision (real rehash + test-overlap re-check) |
| Functional checkpoint loadability | Unproven (R-008) | **Empirically proven** in this revision (real GPU load of 2 real checkpoints) |
| Optimizer/LR continuity | Assessed "architecturally sound... unverified at scale" | **Empirically disproven as claimed** — no continuity exists — but validated as safe by the pilot's own real success (R-019, new) |
| Licensing | "No LICENSE file located" | **Real MIT LICENSE files found and read** for all code components (R-020, new) |

## Scorecard deltas

Categories whose score changes as a direct result of this review (others unchanged from the original
audit — see `retraining_readiness_audit.md` §12 for the full table):

| Category | Original score | Revised score | Why |
|---|---|---|---|
| Infrastructure | 1 | **4** | Disk capacity resolved with a large, measured margin; Docker/GPU already solid. |
| Checkpoint/resume | 2 | **3** | Functional loadability now proven; seed-consistency gap fixed; global-step/optimizer framing corrected and now understood precisely rather than merely flagged as unverified. |
| Training configuration | 4 | **4** (unchanged score, corrected substance) | The LR-decay claim is now known to be wrong as previously stated, but the corrected understanding does not reduce confidence in launch safety — it changes what documentation must say. |
| Launch safety | 4 | **5** | Two more real gaps closed and proven with tamper tests; zero known bypass paths remain. |
| Security | 4 | **4** (unchanged score, new evidence) | Licensing evidence substantially improved (real files found), though this is governance, not security in the strict sense. |

A single unresolved critical issue still means NO-GO under this review's own rules — there are none
remaining, so the aggregate score is no longer overridden by a veto.

## Decision gates — status update

| Gate | Original status | Current status |
|---|---|---|
| 1. Audit complete | Complete (NO-GO) | Complete (this revision, CONDITIONAL GO) |
| 2. Preflight complete | 21/21 (at stale commit) | **21/21, reconfirmed at current HEAD**, twice, in this revision |
| 3. Prepared state | `training/full-corpus-20260801T192341Z` (now stale) | `training/full-corpus-20260801T213023Z` (fresh, matches HEAD) |
| 4. First shard review | Not reached | **Required before unattended continuation** — see `first_shard_review_checklist.md` |
| 5. First full lap review | Not reached | **Required** — see `first_epoch_review_checklist.md`; must also confirm the corrected LR/optimizer understanding does not surprise the operator (loss/CER behavior should look like the pilot's own, not like a run with continuous LR decay) |
| 6. Model-selection review | Not reached | Unchanged requirement |
| 7. Final acceptance | Not reached; no approving authority defined | Unchanged requirement; see `opus_independent_review.md` Phase 4 for a proposed (not yet operator-approved) evaluation policy |

## Final launch conditions (revised)

1. ~~Resolve R-018 (disk capacity)~~ — **DONE.**
2. ~~Fix R-006 (seed validation)~~ — **DONE.**
3. ~~Fix R-007 (shard-integrity self-reference)~~ — **DONE.**
4. **(New, documentation, not blocking)** Correct the `learning_rate_policy` label and any monitoring
   documentation that implies continuous cross-shard LR decay, per R-019.
5. **(Unchanged)** Disclose the R-009 line-level-split limitation in any subgroup/aggregate CER
   reporting.
6. **(Unchanged, now doubly demonstrated)** Do not use either stale prepared run
   (`training/full-corpus-20260801T171555Z` or `training/full-corpus-20260801T192341Z`). Use
   `training/full-corpus-20260801T213023Z`.
7. **(Unchanged)** Commit or explicitly acknowledge (`--allow-dirty-repository`) the working-tree
   state immediately before launch.
8. **(Unchanged, governance)** Define success criteria, minimum-improvement threshold, and approving
   authority (proposed structure in `opus_independent_review.md` Phase 4).

Once items 4-8 are explicitly acknowledged or completed (all are governance/documentation, none are
code fixes), this system supports **GO**. This revision stops at CONDITIONAL GO specifically because
items 4 and 8 have not yet received an actual human decision — that decision belongs to the operator/
ML lead, not to this review.
