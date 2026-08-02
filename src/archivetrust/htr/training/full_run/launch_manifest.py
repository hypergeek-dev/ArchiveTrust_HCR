"""`LaunchManifest` (`launch_manifest.json`) -- the full, immutable record of exactly what a prepared
full-corpus run *would* start with, written once by `prepare` and never modified afterward (brief §6,
§7). This is the artifact the launch guard (`launch_guard.py`) re-derives against at `start` time to
detect drift (a changed dataset hash, a changed manifest hash) between preparation and the operator's
eventual, separate, manual launch decision.

Every field is a real, already-computed fact at `prepare` time -- nothing here is invented or
recomputed later; `prepare` calls `build_launch_manifest` exactly once, using values it already holds
from `identity.py`, `corpus_sharding.py`, `monitoring_config.py`, and `run_state.py`.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict

TELEMETRY_SCHEMA_VERSION = "telemetry-standard-v1"
"""Matches `docs/TELEMETRY_STANDARD_V1.md` -- the one telemetry schema version this project has ever
defined; named here so a future v2 has somewhere real to be declared."""

PREPARED_MARKER_NAME = "PREPARED_NOT_STARTED"
"""A zero-byte marker file dropped directly in the run directory -- readable by a human `ls`/`dir`
without opening any JSON, and by any tooling that just wants to confirm 'has this run ever been
launched' without parsing `run_state.json`. Removed by nothing in this codebase once training
actually starts (`orchestrator.py` never deletes it) -- its *presence* stops being the source of
truth the moment `run_state.json.status` leaves `"prepared"`, but it is deliberately left in place as
a permanent breadcrumb of "this run was, at some point, prepared but not yet started" rather than
being cleaned up."""


class LaunchManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    parent_model_type: str
    original_loghi_base_checkpoint_path: str
    base_checkpoint_hash: str
    pilot_checkpoint_used_as_parent: bool
    pilot_checkpoint_rejection_evidence: str

    training_manifest_dir: str
    training_manifest_hash: str
    validation_manifest_path: str
    validation_manifest_hash: str
    dataset_hash: str | None

    train_line_count: int
    validation_line_count: int
    excluded_test_line_count: int

    code_commit_hash: str | None
    repository_dirty: bool

    container_image_name: str
    container_image_digest: str | None

    batch_size: int
    optimizer: str
    scheduler: str
    learning_rate: float
    random_seed: int

    max_epochs: int
    max_epochs_basis: str
    early_stopping_patience: int
    early_stopping_basis: str
    wall_clock_policy: str

    shard_count: int
    checkpoint_policy: str

    telemetry_schema_version: str
    dashboard_config: str

    preparation_timestamp: str
    exact_launch_command: str


def _is_repository_dirty() -> bool:
    """**Fails closed.** An earlier version returned `False` (i.e. "clean") whenever the probe failed,
    which is the permissive answer to a question it could not answer. That fired for real: once the
    shared training-data pool existed, `git status --porcelain` took ~44s to walk 562k generated
    files, blew the 10s timeout, and a prepared run recorded `repository_dirty: False` while the tree
    was genuinely dirty -- a reproducibility claim that was simply false. Unknown now means dirty, so
    the launch guard asks for an explicit `--allow-dirty-repository` rather than silently waving a
    run through.

    The timeout is also raised, and `--untracked-files=normal` is left at its default so generated
    output still counts; the real fix for the slow case is `.gitignore`, not a looser probe."""
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=60.0, check=True,
        )
    except (subprocess.SubprocessError, OSError):
        return True
    return bool(completed.stdout.strip())


def build_launch_manifest(
    *,
    run_id: str,
    original_loghi_base_checkpoint_path: str | Path,
    base_checkpoint_hash: str,
    known_pilot_run_dirs: tuple[str | Path, ...],
    training_manifest_dir: str | Path,
    training_manifest_hash: str,
    validation_manifest_path: str | Path,
    validation_manifest_hash: str,
    dataset_hash: str | None,
    train_line_count: int,
    validation_line_count: int,
    excluded_test_line_count: int,
    code_commit_hash: str | None,
    container_image_name: str,
    container_image_digest: str | None,
    batch_size: int,
    optimizer: str,
    scheduler: str,
    learning_rate: float,
    random_seed: int,
    max_epochs: int,
    max_epochs_basis: str,
    early_stopping_patience: int,
    early_stopping_basis: str,
    wall_clock_policy: str,
    shard_count: int,
    checkpoint_policy: str,
    dashboard_config: str,
    exact_launch_command: str,
) -> LaunchManifest:
    pilot_dirs_str = ", ".join(str(p) for p in known_pilot_run_dirs) or "(none known)"
    return LaunchManifest(
        run_id=run_id,
        parent_model_type="loghi generic pretrained checkpoint (original, pristine -- never a pilot output)",
        original_loghi_base_checkpoint_path=str(original_loghi_base_checkpoint_path),
        base_checkpoint_hash=base_checkpoint_hash,
        pilot_checkpoint_used_as_parent=False,
        pilot_checkpoint_rejection_evidence=(
            f"parent_checkpoint_dir={original_loghi_base_checkpoint_path} was checked against known "
            f"pilot run directories ({pilot_dirs_str}) via "
            "identity.assert_parent_checkpoint_is_not_a_pilot_path and did not match any of them."
        ),
        training_manifest_dir=str(training_manifest_dir),
        training_manifest_hash=training_manifest_hash,
        validation_manifest_path=str(validation_manifest_path),
        validation_manifest_hash=validation_manifest_hash,
        dataset_hash=dataset_hash,
        train_line_count=train_line_count,
        validation_line_count=validation_line_count,
        excluded_test_line_count=excluded_test_line_count,
        code_commit_hash=code_commit_hash,
        repository_dirty=_is_repository_dirty(),
        container_image_name=container_image_name,
        container_image_digest=container_image_digest,
        batch_size=batch_size,
        optimizer=optimizer,
        scheduler=scheduler,
        learning_rate=learning_rate,
        random_seed=random_seed,
        max_epochs=max_epochs,
        max_epochs_basis=max_epochs_basis,
        early_stopping_patience=early_stopping_patience,
        early_stopping_basis=early_stopping_basis,
        wall_clock_policy=wall_clock_policy,
        shard_count=shard_count,
        checkpoint_policy=checkpoint_policy,
        telemetry_schema_version=TELEMETRY_SCHEMA_VERSION,
        dashboard_config=dashboard_config,
        preparation_timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        exact_launch_command=exact_launch_command,
    )


def write_launch_manifest(manifest: LaunchManifest, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(output_path.parent), prefix=".tmp-launch-manifest-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(manifest.model_dump_json(indent=2))
        os.replace(tmp_path, str(output_path))
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def load_launch_manifest(path: str | Path) -> LaunchManifest:
    return LaunchManifest.model_validate_json(Path(path).read_text(encoding="utf-8"))


def write_prepared_marker(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    marker = run_dir / PREPARED_MARKER_NAME
    marker.write_text(
        "This run was PREPARED but has NOT been started. See launch_manifest.json for the exact, "
        "unexecuted launch command. No training process has run against this directory.\n",
        encoding="utf-8",
    )
    return marker
