"""The hard guard between a *prepared* full-corpus run and an actually *started* one (brief §10).
`cmd_start` calls `enforce_launch_guard` before it does anything else -- every rejection condition
below is checked against real, freshly re-read evidence (the persisted `launch_manifest.json`,
`run_state.json`, a fresh `docker ps`, a fresh disk-space read, a fresh dataset/manifest hash), never
assumed unchanged since `prepare` ran.

No confirmation flag is ever supplied by anything in this codebase -- `--confirm-full-corpus-run` is
an operator-typed flag on the real CLI invocation, the one manual action the whole workflow exists to
gate.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pyarrow.parquet as pq

from archivetrust.htr.training.full_run.launch_manifest import LaunchManifest
from archivetrust.htr.training.full_run.run_state import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_RUNNING,
    STATUS_STOPPING,
    FullRunState,
)


class LaunchGuardRejected(RuntimeError):
    """Raised with every violated condition listed -- an operator sees the whole picture in one
    failed `start` attempt, not one-problem-at-a-time whack-a-mole."""


def _docker_reachable() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        completed = subprocess.run(["docker", "ps"], capture_output=True, text=True, timeout=15.0, check=False)
    except (subprocess.SubprocessError, OSError):
        return False
    return completed.returncode == 0


def _container_image_available(image_tag: str, image_digest: str | None) -> bool:
    ref = f"{image_tag}@{image_digest}" if image_digest else image_tag
    try:
        completed = subprocess.run(
            ["docker", "image", "inspect", ref], capture_output=True, text=True, timeout=15.0, check=False,
        )
    except (subprocess.SubprocessError, OSError):
        return False
    return completed.returncode == 0


CRASHED_RUN_TELEMETRY_STALE_SECONDS = 300.0
"""How stale `telemetry/status.json` must be before a run still marked `running`/`stopping` is
treated as *crashed* rather than *live*. `TelemetrySampler` writes that file every ~5s for the whole
duration of a container invocation, so 300s is 60x its normal cadence -- generous enough that a
merely-slow host is never mistaken for a dead one, while still detecting a real crash quickly. This
is deliberately independent of `run_state.json`'s own heartbeat, which only advances once per
*shard* (~8 minutes here) and so cannot distinguish "mid-shard" from "dead"."""


def looks_like_a_crashed_run(run_state_dir: str | Path, *, now: float | None = None) -> bool:
    """`True` when a run's live-telemetry heartbeat has gone stale, i.e. the process that was
    writing it is gone. Used only to *explain* a blocked resume and to gate the explicit
    `--force-resume-after-crash` override -- never to silently permit one.

    Fails closed: any unreadable/absent/malformed telemetry, or a parse failure, returns `False`
    ("cannot prove it crashed"), so an ambiguous state keeps the run blocked rather than allowing a
    second process to attach to the same directories."""
    import calendar
    import json as _json
    import time

    status_path = Path(run_state_dir) / "telemetry" / "status.json"
    if not status_path.exists():
        # Never started a container (so nothing to conflict with) is indistinguishable here from
        # telemetry never having been wired up. Fail closed.
        return False
    try:
        payload = _json.loads(status_path.read_text(encoding="utf-8"))
        last = payload.get("last_sample_at")
        if not last:
            return False
        parsed = calendar.timegm(time.strptime(last, "%Y-%m-%dT%H:%M:%SZ"))
    except (OSError, ValueError, _json.JSONDecodeError):
        return False
    now = now if now is not None else time.time()
    return (now - parsed) > CRASHED_RUN_TELEMETRY_STALE_SECONDS


def _shard_line_id_overlap_with_manifest(shards_dir: Path, manifest_path: Path) -> int:
    from archivetrust.htr.training.full_run.corpus_sharding import load_sharding_summary

    summary = load_sharding_summary(shards_dir)
    manifest_ids = set(pq.read_table(manifest_path, columns=["line_id"]).column("line_id").to_pylist())
    overlap = 0
    for shard in summary.shards:
        shard_ids = set(pq.read_table(shard.manifest_path, columns=["line_id"]).column("line_id").to_pylist())
        overlap += len(shard_ids & manifest_ids)
    return overlap


def recompute_real_shard_hash(shards_dir: str | Path) -> str | None:
    """Independent-review finding: the previous version of this check only compared
    `sharding_summary.json`'s own recorded `line_id_set_hash` against itself (re-reading the same
    file the frozen `launch_manifest.json` hash was originally copied from) -- a tampered or corrupted
    individual shard `.parquet` file would never be caught, since nothing ever rehashed real shard
    bytes. This function reads every real lap-0 shard file's actual `line_id` column, reproduces
    `corpus_sharding.py::build_full_corpus_shards`'s own hash method exactly (sha256 of the sorted,
    JSON-dumped line-ID list), and returns a hash that only matches the recorded one if the real,
    current shard file contents genuinely still match what was hashed at `prepare` time. Lap 0 alone
    is used because it is exactly the usable line-ID set by construction (each usable line appears
    there exactly once); later laps are reshuffled repeats of the same set, not additional lines."""
    import hashlib
    import json

    from archivetrust.htr.training.full_run.corpus_sharding import load_sharding_summary

    shards_dir = Path(shards_dir)
    if not shards_dir.exists():
        return None
    summary = load_sharding_summary(shards_dir)
    lap0_shards = [s for s in summary.shards if s.lap == 0]
    if not lap0_shards:
        return None
    line_ids: set[str] = set()
    for shard in lap0_shards:
        line_ids.update(pq.read_table(shard.manifest_path, columns=["line_id"]).column("line_id").to_pylist())
    return hashlib.sha256(json.dumps(sorted(line_ids)).encode("utf-8")).hexdigest()


def evaluate_launch_guard(
    *,
    confirmed: bool,
    run_dir: str | Path,
    run_state: FullRunState | None,
    launch_manifest: LaunchManifest | None,
    current_dataset_hash: str | None,
    current_training_manifest_hash: str | None,
    preflight_passed: bool,
    preflight_age_seconds: float | None,
    max_preflight_age_seconds: float,
    min_free_disk_gb: float,
    container_image_tag: str,
    container_image_digest: str | None,
    allow_dirty_repository: bool = False,
    check_docker: bool = True,
    check_train_val_overlap: bool = True,
    current_code_revision: str | None = None,
    allow_code_revision_drift: bool = False,
    is_resume: bool = False,
    test_manifest_path: str | Path | None = None,
    force_resume_after_crash: bool = False,
) -> list[str]:
    """Returns every violated condition (empty list means the guard passes). Never raises itself --
    `enforce_launch_guard` below is the raising wrapper, so callers that just want the full list (e.g.
    a `--dry-run` report) can inspect it without a `try/except`.

    **Two conditions added after a real audit finding, not hypothetical**: (1) `run_state.status` in
    `{COMPLETED, FAILED}` is now rejected exactly like `{RUNNING, STOPPING}` -- previously `resume`
    had no guard against re-entering a run this codebase itself had already marked finished, since
    `cli.py`'s own `state.status != STATUS_PREPARED` early-return only applied to `start`, never to
    `resume`. (2) `current_code_revision` (freshly read `git rev-parse HEAD` at start/resume time) is
    now compared against `launch_manifest.code_commit_hash` (frozen at `prepare` time) -- previously
    the guard checked `repository_dirty` but never whether the *committed* code itself had moved on
    since preparation, which a real prepared run in this repository was found to have done (prepared
    at one commit, a real bug fix landed in a later commit, and nothing would have caught the drift)."""
    run_dir = Path(run_dir)
    problems: list[str] = []

    if not confirmed:
        problems.append("Missing explicit confirmation (--confirm-full-corpus-run was not supplied).")

    if run_state is None:
        problems.append(f"No run_state.json found under {run_dir} -- run `prepare` first.")
    elif run_state.status in (STATUS_RUNNING, STATUS_STOPPING):
        # A crash (power loss, host reboot, Ctrl+C, killed container) leaves this status behind with
        # no process attached. Blocking unconditionally -- as this guard originally did -- made the
        # documented recovery path (`resume`) impossible and forced hand-editing of run_state.json.
        # Blocking a genuinely *live* run is still correct, so the two cases are separated by real
        # telemetry liveness, and the override only ever applies to the provably-dead case.
        crashed = is_resume and looks_like_a_crashed_run(run_dir / "run-state")
        if not crashed:
            problems.append(
                f"An active run already exists at {run_dir} with status {run_state.status!r}."
                + (
                    " Its live telemetry is still fresh, so a training process appears to be "
                    "genuinely attached -- refusing to start a second one."
                    if is_resume else ""
                )
            )
        elif not force_resume_after_crash:
            problems.append(
                f"Run at {run_dir} is still marked {run_state.status!r}, but its live telemetry has "
                f"been stale for over {CRASHED_RUN_TELEMETRY_STALE_SECONDS:.0f}s -- this looks like a "
                "crashed run, not a live one. Confirm no container is still attached "
                "(`docker ps`), then re-run with --force-resume-after-crash to recover from the "
                "last verified checkpoint."
            )
    elif is_resume and run_state.status in (STATUS_COMPLETED, STATUS_FAILED):
        problems.append(
            f"Refusing to resume a run already marked {run_state.status!r} -- a completed or failed "
            "run is terminal; start a new prepared run instead of resuming this one."
        )

    if launch_manifest is None:
        problems.append(f"No launch_manifest.json found under {run_dir} -- run `prepare` first.")
    else:
        if launch_manifest.pilot_checkpoint_used_as_parent:
            problems.append("launch_manifest.json records the pilot checkpoint as the parent -- refusing to launch.")
        if current_dataset_hash is None or launch_manifest.dataset_hash is None:
            problems.append("Dataset hash unavailable (recorded or current) -- cannot confirm the dataset is unchanged.")
        elif current_dataset_hash != launch_manifest.dataset_hash:
            problems.append(
                f"Dataset hash changed since preparation: prepared={launch_manifest.dataset_hash[:16]}..., "
                f"now={current_dataset_hash[:16]}..."
            )
        if current_training_manifest_hash is None:
            problems.append("Could not recompute the current training manifest hash.")
        elif current_training_manifest_hash != launch_manifest.training_manifest_hash:
            problems.append(
                f"Training manifest hash changed since preparation: "
                f"prepared={launch_manifest.training_manifest_hash[:16]}..., now={current_training_manifest_hash[:16]}..."
            )
        if launch_manifest.repository_dirty and not allow_dirty_repository:
            problems.append(
                "The repository had uncommitted changes at preparation time -- reproducibility policy "
                "requires a clean commit (pass allow_dirty_repository=True to override)."
            )
        if (
            not allow_code_revision_drift
            and launch_manifest.code_commit_hash is not None
            and current_code_revision is not None
            and current_code_revision != launch_manifest.code_commit_hash
        ):
            problems.append(
                f"Code commit changed since preparation: prepared at {launch_manifest.code_commit_hash[:12]}, "
                f"now at {current_code_revision[:12]} -- the code that will actually run no longer matches "
                "the audited/prepared configuration. Re-run `prepare` (and `preflight`) at the current "
                "commit, or pass allow_code_revision_drift=True to override with an explicit, informed "
                "acknowledgement."
            )
        if check_train_val_overlap:
            shards_dir = run_dir / "shards"
            val_path = Path(launch_manifest.validation_manifest_path)
            if shards_dir.exists() and val_path.exists():
                overlap = _shard_line_id_overlap_with_manifest(shards_dir, val_path)
                if overlap:
                    problems.append(f"{overlap} training line ID(s) overlap with the validation manifest.")
            if shards_dir.exists() and test_manifest_path is not None and Path(test_manifest_path).exists():
                test_overlap = _shard_line_id_overlap_with_manifest(shards_dir, Path(test_manifest_path))
                if test_overlap:
                    problems.append(f"{test_overlap} training line ID(s) overlap with the reserved test manifest.")

    if not preflight_passed:
        problems.append("No passing preflight recorded -- run `preflight` and get all critical checks passing first.")
    elif preflight_age_seconds is not None and preflight_age_seconds > max_preflight_age_seconds:
        problems.append(
            f"Last passing preflight is {preflight_age_seconds:.0f}s old, older than the allowed "
            f"{max_preflight_age_seconds:.0f}s -- rerun `preflight` before launching."
        )

    if not run_dir.exists() or not run_dir.is_dir():
        problems.append(f"Output directory is not valid: {run_dir}")
    else:
        free_gb = shutil.disk_usage(run_dir).free / (1024**3)
        if free_gb < min_free_disk_gb:
            problems.append(f"Only {free_gb:.1f}GB free at {run_dir}, below the required {min_free_disk_gb}GB.")

    if check_docker:
        if not _docker_reachable():
            problems.append("Docker daemon is not reachable (docker ps failed).")
        elif not _container_image_available(container_image_tag, container_image_digest):
            problems.append(f"Pinned container image not available locally: {container_image_tag}.")

    return problems


def enforce_launch_guard(**kwargs) -> None:
    problems = evaluate_launch_guard(**kwargs)
    if problems:
        raise LaunchGuardRejected(
            "Refusing to start full-corpus training -- " + str(len(problems)) + " guard condition(s) failed:\n"
            + "\n".join(f"  - {p}" for p in problems)
        )
