"""One-time correction of a policy artifact in run `full-corpus-20260802T044250Z`.

The run stopped at shard 53 of 57 with `no_val_cer_improvement` and was marked COMPLETED. Under the
policy in force at the time, patience was allowed to act once `min_exposure_steps` was reached --
which happens around shard 20, because that threshold was derived from a 9,999-line pilot epoch. The
operator has since ruled that patience must remain informational until one full corpus lap (57
shards) has completed. Under the corrected policy this stop should never have occurred, so
`completed` does not describe reality: the lap is unfinished at 53/57.

This rewrites the status to STOPPED via the normal `mark_stopped` transition (not a raw JSON edit),
which is the resumable state the run should have been left in. Nothing else is touched: no metric,
no checkpoint, no shard count, no history. The run then resumes under the lap-gated code to finish
shards 54-57.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from archivetrust.htr.training.full_run.run_state import (
    STATUS_COMPLETED,
    load_run_state,
    mark_stopped,
    save_run_state,
)

RUN_STATE_DIR = Path(r"d:\ArchiveTrust_HCR\training\full-corpus-20260802T044250Z\run-state")

state = load_run_state(RUN_STATE_DIR)
print(f"before: status={state.status} stop_reason={state.stop_reason}")
print(f"        shards_completed_in_current_epoch={state.shards_completed_in_current_epoch}"
      f" / {state.shards_per_epoch}  epochs_completed={state.epochs_completed}")

if state.status != STATUS_COMPLETED:
    print("not marked completed; nothing to correct")
    raise SystemExit(0)

if state.epochs_completed >= 1:
    raise SystemExit("REFUSING: a full lap really did complete -- this is not a policy artifact")

corrected = mark_stopped(
    state,
    stop_reason="premature_completion_corrected_lap_incomplete_53_of_57",
)
save_run_state(RUN_STATE_DIR, corrected)

after = load_run_state(RUN_STATE_DIR)
print(f"after:  status={after.status} stop_reason={after.stop_reason}")
print(f"        global_shards_completed={after.global_shards_completed} (unchanged)")
assert after.global_shards_completed == state.global_shards_completed
assert after.best_metrics == state.best_metrics
assert after.best_checkpoint == state.best_checkpoint
print("metrics and checkpoints unchanged; run is resumable")
