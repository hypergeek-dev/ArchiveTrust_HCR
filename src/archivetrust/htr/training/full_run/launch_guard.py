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


def _shard_line_id_overlap_with_validation(shards_dir: Path, validation_manifest_path: Path) -> int:
    from archivetrust.htr.training.full_run.corpus_sharding import load_sharding_summary

    summary = load_sharding_summary(shards_dir)
    val_ids = set(pq.read_table(validation_manifest_path, columns=["line_id"]).column("line_id").to_pylist())
    overlap = 0
    for shard in summary.shards:
        shard_ids = set(pq.read_table(shard.manifest_path, columns=["line_id"]).column("line_id").to_pylist())
        overlap += len(shard_ids & val_ids)
    return overlap


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
        problems.append(f"An active run already exists at {run_dir} with status {run_state.status!r}.")
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
                overlap = _shard_line_id_overlap_with_validation(shards_dir, val_path)
                if overlap:
                    problems.append(f"{overlap} training line ID(s) overlap with the validation manifest.")

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
