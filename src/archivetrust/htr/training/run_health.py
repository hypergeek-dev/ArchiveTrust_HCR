"""`compute_health`: the dashboard's health-summary panel (brief §"Health summary") -- one check per
listed warning condition, each backed by a real signal `run_status.py`/`checkpoint_index.py`/
`run_profile.py`/the telemetry sampler already computes. Pure function, no I/O of its own -- the
ViewModel gathers the real inputs (validation history, checkpoint entries, recent GPU samples) and
passes them in, so this module stays trivially testable without any fixture files.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from archivetrust.htr.training.checkpoint_index import CheckpointEntry
from archivetrust.htr.training.run_status import RunStatus

WARNING_STALENESS_SECONDS = 20.0
"""4x the sampler's default 5s tick -- long enough to absorb one missed tick's jitter, short enough
to flag a real problem well before `run_status.py`'s own 60s `STALE_THRESHOLD_SECONDS` reclassifies
the whole run as `"interrupted"`."""
LOW_DISK_FREE_GB_THRESHOLD = 5.0
VAL_CER_WORSENED_EPOCH_THRESHOLD = 2
"""`epochs_since_improvement >= this` is flagged as a warning -- deliberately lower than a typical
early-stopping `patience` (5 in this project's own pilot profile default), so the dashboard surfaces
the trend before training actually stops because of it."""

HEALTHY = "healthy"
WARNING = "warning"
STALLED = "stalled"
FAILED = "failed"
COMPLETED = "completed"


class HealthFinding(BaseModel):
    model_config = ConfigDict(frozen=True)

    reason: str
    """A short, stable code (e.g. `"stale_heartbeat"`, `"cap_shorter_than_epoch"`) -- for tests and
    for a caller that wants to filter/group findings, distinct from the free-text `message`."""
    message: str


class HealthAssessment(BaseModel):
    model_config = ConfigDict(frozen=True)

    level: str
    """`"healthy" | "warning" | "stalled" | "failed" | "completed"`."""
    findings: tuple[HealthFinding, ...]


def compute_health(
    *,
    status: RunStatus,
    validation_history: tuple[dict, ...] = (),
    checkpoint_entries: tuple[CheckpointEntry, ...] = (),
    recent_gpu_samples: tuple[dict, ...] = (),  # oldest first, most recent last (e.g. gpu_samples.jsonl's tail)
    cap_warning: str | None = None,  # run_profile.cap_shorter_than_one_epoch_warning(...)'s result, precomputed by the caller
    seconds_since_last_heartbeat: float | None = None,
    disk_free_gb_threshold: float = LOW_DISK_FREE_GB_THRESHOLD,
) -> HealthAssessment:
    findings: list[HealthFinding] = []

    if status.status == "failed":
        findings.append(HealthFinding(reason="run_failed", message=f"Run failed: {status.stop_reason}"))
        return HealthAssessment(level=FAILED, findings=tuple(findings))
    if status.status == "interrupted":
        findings.append(HealthFinding(
            reason="interrupted",
            message="No fresh telemetry heartbeat and no recorded clean stop -- the process appears "
            "to have been killed without a chance to record why.",
        ))
        return HealthAssessment(level=STALLED, findings=tuple(findings))

    # -- Warning-level checks (every check below runs regardless of terminal status, since a
    # completed run can still have things worth flagging, e.g. a mid-epoch stop). --

    if status.status in ("running", "stopping") and seconds_since_last_heartbeat is not None:
        if seconds_since_last_heartbeat > WARNING_STALENESS_SECONDS:
            findings.append(HealthFinding(
                reason="stale_heartbeat",
                message=f"No telemetry update in {seconds_since_last_heartbeat:.0f}s.",
            ))

    if status.cumulative_epoch > 0 and len(validation_history) < status.cumulative_epoch:
        findings.append(HealthFinding(
            reason="validation_missing",
            message=f"{status.cumulative_epoch} epoch(s) completed but only "
            f"{len(validation_history)} validation record(s) present.",
        ))

    if checkpoint_entries:
        most_recent = max(checkpoint_entries, key=lambda e: (e.epoch, e.created_at))
        if most_recent.verification_status == "verification_failed":
            findings.append(HealthFinding(
                reason="checkpoint_save_failed",
                message=f"Checkpoint at epoch {most_recent.epoch} failed verification.",
            ))

    if recent_gpu_samples:
        last_disk = recent_gpu_samples[-1].get("system", {}).get("disk_free_gb")
        if last_disk is not None and last_disk < disk_free_gb_threshold:
            findings.append(HealthFinding(
                reason="low_disk_space",
                message=f"Only {last_disk:.1f}GB free at the run's output location.",
            ))

        gpu_names = [s.get("gpu", {}).get("name") for s in recent_gpu_samples]
        if any(n is not None for n in gpu_names[:-1]) and gpu_names[-1] is None:
            findings.append(HealthFinding(
                reason="gpu_telemetry_disappeared",
                message="GPU telemetry was present in recent samples but is now absent.",
            ))

    if cap_warning is not None:
        findings.append(HealthFinding(reason="cap_shorter_than_epoch", message=cap_warning))

    if status.epochs_since_improvement >= VAL_CER_WORSENED_EPOCH_THRESHOLD:
        findings.append(HealthFinding(
            reason="val_cer_not_improving",
            message=f"Validation CER has not improved for {status.epochs_since_improvement} "
            "consecutive epoch(s).",
        ))

    if status.stopped_mid_epoch:
        findings.append(HealthFinding(
            reason="stopped_mid_epoch",
            message="The run stopped mid-epoch (the failing epoch never reached a verified checkpoint).",
        ))

    if status.cumulative_epoch > 0 and status.best_checkpoint_path is None:
        findings.append(HealthFinding(
            reason="best_checkpoint_missing",
            message="At least one epoch has completed but no best-validation checkpoint is recorded.",
        ))
    elif status.best_checkpoint_path is not None:
        from pathlib import Path

        if not Path(status.best_checkpoint_path).exists():
            findings.append(HealthFinding(
                reason="best_checkpoint_missing",
                message=f"Recorded best checkpoint path does not exist: {status.best_checkpoint_path}",
            ))

    if status.status == "completed" and not findings:
        return HealthAssessment(level=COMPLETED, findings=())

    if not findings:
        return HealthAssessment(level=HEALTHY, findings=())
    return HealthAssessment(level=WARNING, findings=tuple(findings))
