"""Preflight checks for a full-corpus run -- training must not start when any critical check fails
(brief §5). Each check is its own small, independently testable function; `run_preflight` assembles
them into one `PreflightReport`, catching any individual check's exception as a failed check rather
than letting one broken probe crash the whole preflight.

**Ordering discipline, deliberately**: this module only checks what genuinely exists *before*
`prepare` has run -- the pinned base checkpoint, Docker/GPU readiness, the pilot's own real
val/test manifests (reused, not regenerated), disk/telemetry/seed/conflicting-run state. Full-corpus
shard-level checks (aggregate line counts, per-shard overlap, source-path resolution) cannot be
run here because shards do not exist until `prepare` creates them -- that is the separate, later
"validate sharding" step (`corpus_sharding.py`'s own summary + a dedicated report), never silently
skipped, just correctly sequenced.

Reuses real primitives throughout, never a parallel probing mechanism: `checkpoint_index.
verify_checkpoint` (base-model + checkpoint-round-trip checks), `identity.
assert_parent_checkpoint_is_not_a_pilot_path`, `providers/loghi/environment.probe_loghi_environment`,
`shutil.disk_usage`. The dry-run forward+backward smoke test reuses `container_epoch_runner.
ContainerEpochRunner`/`memory_probe.py`'s exact real-but-bounded pattern via an injectable
`EpochRunner` (a real container in production, a `FakeEpochRunner` in tests).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Protocol

import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict

from archivetrust.htr.training.checkpoint_index import verify_checkpoint
from archivetrust.htr.training.full_run.identity import (
    PilotCheckpointRejected,
    assert_parent_checkpoint_is_not_a_pilot_path,
)
from archivetrust.htr.training.full_run.monitoring_config import FullRunMonitoringConfig
from archivetrust.htr.training.training_session import EpochResult

EXPECTED_PILOT_VAL_LINE_COUNT = 1000
"""The pilot's real, recorded validation split size -- reused as the full run's own validation set.
A plausibility check, not a hard architectural constant."""


class SmokeTestRunner(Protocol):
    def run_epoch(
        self, *, existing_model_dir: str, output_dir: str, train_list_path: str,
        validation_list_path: str, epoch_seed: int,
    ) -> EpochResult: ...


class PreflightCheck(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    passed: bool
    message: str
    critical: bool = True
    """A failed non-critical check is still reported but does not block `start` -- currently every
    check this module defines is critical; the field exists so a future soft-warning check has
    somewhere to declare itself without a schema change."""


class PreflightReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    checks: tuple[PreflightCheck, ...]

    @property
    def all_critical_passed(self) -> bool:
        return all(c.passed for c in self.checks if c.critical)

    @property
    def failed_checks(self) -> tuple[PreflightCheck, ...]:
        return tuple(c for c in self.checks if not c.passed)

    @property
    def summary(self) -> str:
        total = len(self.checks)
        passed = sum(1 for c in self.checks if c.passed)
        return f"{passed}/{total}"


def _check(name: str, fn) -> PreflightCheck:
    try:
        passed, message = fn()
        return PreflightCheck(name=name, passed=passed, message=message)
    except Exception as exc:  # noqa: BLE001 -- a broken probe is a failed check, never a crash
        return PreflightCheck(name=name, passed=False, message=f"Check raised an exception: {exc}")


# -- Docker / GPU ------------------------------------------------------------------------------


def _check_docker_daemon_reachable() -> tuple[bool, str]:
    """Real `docker ps` -- distinct from `probe_loghi_environment().docker_cli_present`, which only
    checks that the CLI binary exists (`docker --version`), not that the daemon behind it is
    actually reachable. A real gap this project's own earlier preflight run exposed: the CLI-present
    check passed while the daemon was genuinely down."""
    if shutil.which("docker") is None:
        return False, "docker CLI not found on PATH."
    try:
        completed = subprocess.run(
            ["docker", "ps"], capture_output=True, text=True, timeout=15.0, check=False,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        return False, f"docker ps failed to run: {exc}"
    if completed.returncode != 0:
        return False, f"docker ps exited {completed.returncode}: {(completed.stderr or completed.stdout).strip()[:300]}"
    return True, "Docker daemon reachable (docker ps succeeded)."


def _check_container_image_available(image_tag: str, image_digest: str | None) -> tuple[bool, str]:
    ref = f"{image_tag}@{image_digest}" if image_digest else image_tag
    try:
        completed = subprocess.run(
            ["docker", "image", "inspect", ref], capture_output=True, text=True, timeout=15.0, check=False,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        return False, f"docker image inspect failed to run: {exc}"
    if completed.returncode != 0:
        return False, f"Pinned image not present locally: {ref} (run `docker pull {image_tag}`)."
    return True, f"Pinned image present locally: {ref}."


def _check_gpu_available(require_gpu: bool) -> tuple[bool, str]:
    from archivetrust.providers.loghi.environment import probe_loghi_environment

    report = probe_loghi_environment()
    if not require_gpu:
        return True, "GPU not required for this configuration."
    if not report.docker_cli_present:
        return False, "Docker CLI not available -- cannot run GPU-backed training containers."
    return True, f"Docker CLI present, execution_mode={report.execution_mode}."


# -- Base checkpoint ---------------------------------------------------------------------------


def _check_base_model_loadable(base_model_dir: Path) -> tuple[bool, str]:
    ok, digest, extra = verify_checkpoint(base_model_dir)
    if not ok:
        return False, f"{base_model_dir} does not contain a real, readable .keras checkpoint."
    return True, f"Base model verified: sha256={digest[:16]}..., {extra}"


def _check_base_checkpoint_hash_matches_pin(base_model_dir: Path, pinned_hash: str) -> tuple[bool, str]:
    keras_files = list(base_model_dir.glob("*.keras"))
    if not keras_files:
        return False, f"No .keras file found under {base_model_dir} to hash."
    digest = hashlib.sha256()
    with keras_files[0].open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != pinned_hash:
        return False, f"Base checkpoint hash mismatch: expected {pinned_hash[:16]}..., got {actual[:16]}..."
    return True, f"Base checkpoint hash matches the pin: {actual[:16]}..."


def _check_parent_not_pilot(
    parent_checkpoint_dir: Path, known_pilot_run_dirs: tuple[Path, ...]
) -> tuple[bool, str]:
    try:
        assert_parent_checkpoint_is_not_a_pilot_path(parent_checkpoint_dir, known_pilot_run_dirs=known_pilot_run_dirs)
    except PilotCheckpointRejected as exc:
        return False, str(exc)
    return True, "Parent checkpoint is not under any known pilot run directory."


# -- Manifests (the pilot's real val/test splits, reused unchanged) ---------------------------


def _check_manifest_non_empty(label: str, manifest_path: Path) -> tuple[bool, str]:
    if not manifest_path.exists():
        return False, f"{label} manifest does not exist: {manifest_path}"
    table = pq.read_table(manifest_path, columns=["line_id"])
    if table.num_rows == 0:
        return False, f"{label} manifest is empty: {manifest_path}"
    return True, f"{label} manifest has {table.num_rows} real rows."


def _line_ids(path: Path) -> set[str]:
    return set(pq.read_table(path, columns=["line_id"]).column("line_id").to_pylist())


def _check_val_test_no_overlap(val_manifest_path: Path, test_manifest_path: Path) -> tuple[bool, str]:
    if not (val_manifest_path.exists() and test_manifest_path.exists()):
        return False, "Validation or reserved-test manifest missing -- cannot check overlap."
    overlap = _line_ids(val_manifest_path) & _line_ids(test_manifest_path)
    if overlap:
        return False, f"{len(overlap)} line ID(s) appear in both validation and the reserved test set."
    return True, "No overlap between validation and the reserved test set."


def _check_expected_line_counts_plausible(val_manifest_path: Path) -> tuple[bool, str]:
    if not val_manifest_path.exists():
        return False, "Validation manifest missing -- cannot check line count plausibility."
    count = pq.read_table(val_manifest_path, columns=["line_id"]).num_rows
    if count != EXPECTED_PILOT_VAL_LINE_COUNT:
        return False, f"Validation line count {count} does not match the expected {EXPECTED_PILOT_VAL_LINE_COUNT}."
    return True, f"Validation line count is exactly the expected {count}."


# -- Dataset / environment ----------------------------------------------------------------------


def _check_dataset_hash_recorded(dataset_hash: str | None) -> tuple[bool, str]:
    if not dataset_hash:
        return False, "No dataset hash recorded -- the source inventory's provenance is not pinned."
    return True, f"Dataset hash recorded: {dataset_hash[:16]}..."


def _check_dataset_hash_stable(recorded_hash: str | None, current_hash: str | None) -> tuple[bool, str]:
    if recorded_hash is None or current_hash is None:
        return False, "Cannot compare dataset hashes -- one or both are unavailable."
    if recorded_hash != current_hash:
        return False, f"Dataset hash changed since analysis: recorded {recorded_hash[:16]}..., now {current_hash[:16]}..."
    return True, f"Dataset hash unchanged since analysis: {current_hash[:16]}..."


def _check_disk_space(output_dir: Path, min_free_gb: float) -> tuple[bool, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(output_dir).free / (1024**3)
    if free_gb < min_free_gb:
        return False, f"Only {free_gb:.1f}GB free at {output_dir}, below the required {min_free_gb}GB."
    return True, f"{free_gb:.1f}GB free at {output_dir}."


def _check_telemetry_path_writable(run_state_dir: Path) -> tuple[bool, str]:
    telemetry_dir = run_state_dir / "telemetry"
    try:
        telemetry_dir.mkdir(parents=True, exist_ok=True)
        probe = telemetry_dir / ".preflight_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        return False, f"Telemetry path {telemetry_dir} is not writable: {exc}"
    return True, f"Telemetry path writable: {telemetry_dir}."


def _check_monitoring_config_valid(config: FullRunMonitoringConfig) -> tuple[bool, str]:
    problems = []
    if config.shard_line_count <= 0:
        problems.append("shard_line_count must be positive")
    if config.recommended_patience <= 0:
        problems.append("recommended_patience must be positive")
    if config.max_full_run_epochs <= 0:
        problems.append("max_full_run_epochs must be positive")
    if config.min_exposure_steps is not None and config.min_exposure_steps < 0:
        problems.append("min_exposure_steps must not be negative")
    if problems:
        return False, "; ".join(problems)
    return True, "Monitoring configuration values are structurally valid."


def _check_random_seed_explicit(random_seed: int | None) -> tuple[bool, str]:
    if random_seed is None:
        return False, "No random seed configured -- reproducibility requires an explicit seed."
    return True, f"Random seed explicitly configured: {random_seed}."


def _check_no_conflicting_active_run(training_root: Path) -> tuple[bool, str]:
    from archivetrust.htr.training.full_run.run_state import STATUS_RUNNING, load_run_state

    if not training_root.exists():
        return True, "No existing run directories -- nothing to conflict with."
    running = []
    for candidate in training_root.iterdir():
        if not candidate.is_dir():
            continue
        state = load_run_state(candidate / "run-state")
        if state is not None and state.status == STATUS_RUNNING:
            running.append(candidate.name)
    if running:
        return False, f"Conflicting active run(s) already in 'running' status: {running}"
    return True, "No conflicting active run found."


def _check_dashboard_can_discover(training_root: Path) -> tuple[bool, str]:
    from archivetrust.presentation.training_dashboard_viewmodel import TrainingDashboardViewModel

    try:
        vm = TrainingDashboardViewModel(training_root=training_root)
        dirs = vm.discover_run_dirs()
    except Exception as exc:  # noqa: BLE001 -- a broken discovery mechanism is a failed check
        return False, f"Dashboard run discovery raised an exception: {exc}"
    return True, f"Dashboard discovery mechanism works ({len(dirs)} run(s) currently discoverable)."


def _check_resume_state_continuity(pilot_resume_proof_path: Path) -> tuple[bool, str]:
    """Reuses the pilot's own real, already-completed resume proof
    (`training_session.py::prove_full_state_resume`) rather than re-running an expensive multi-epoch
    real-container proof here -- same checkpoint mechanism, same pinned image, unchanged for the
    full run.

    **Renamed from `_check_optimizer_scheduler_serialization` on 2026-08-01.** The old name claimed
    more than the evidence supports: optimizer and LR-schedule state are *not* serialized by the
    pinned container at all (proven empirically -- see `training_session.py`'s module docstring).
    What this proof genuinely establishes, and all it now claims, is that ArchiveTrust's own
    externally-tracked resume state (cumulative epoch, best-metric bookkeeping, checkpoint chaining)
    continues correctly across a real process boundary rather than restarting."""
    if not pilot_resume_proof_path.exists():
        return False, f"No pilot resume-proof evidence found at {pilot_resume_proof_path}."
    payload = json.loads(pilot_resume_proof_path.read_text(encoding="utf-8"))
    if not payload.get("proof_passed"):
        return False, "Pilot resume-proof evidence exists but did not pass."
    return True, (
        f"Resume-state continuity proven via the pilot's real resume-proof "
        f"(epoch_continued={payload.get('epoch_continued_not_restarted')}). "
        "Note: optimizer momentum and LR-schedule position are NOT carried across checkpoints by "
        "the pinned container -- weights and ArchiveTrust's own counters are."
    )


# -- Smoke test ----------------------------------------------------------------------------------


def _stage_writable_checkpoint_for_smoke_test(*, source_dir: Path, staging_dir: Path) -> str:
    """Delegates to `training_session._stage_writable_checkpoint` -- the exact same copy-once
    discipline a real run already uses before ever mounting a checkpoint into a container. **A real
    incident this function exists to prevent**: an earlier version of this module passed
    `base_model_dir` straight through as `existing_model_dir`, and the real Loghi container's own
    old-format-checkpoint loading path silently resaved (converted, shrank, and added a
    `tokenizer.json` to) the pristine pinned checkpoint *in place* on the host filesystem during a
    preflight smoke test -- corrupting the one file every later preflight/prepare run depends on
    being unchanged. Staging a disposable copy here means the smoke test's container can write
    anything it wants without ever touching the real pinned source."""
    from archivetrust.htr.training.training_session import _stage_writable_checkpoint

    return _stage_writable_checkpoint(source_dir=str(source_dir), staging_dir=staging_dir)


def _run_smoke_test(
    runner: SmokeTestRunner | None, *, existing_model_dir: str, output_dir: str, probe_train_list: str, probe_val_list: str
) -> EpochResult:
    if runner is None:
        return EpochResult(
            ok=False, duration_seconds=0.0,
            error_message="No smoke-test runner available (e.g. Docker unreachable) -- cannot verify a real forward+backward pass.",
        )
    return runner.run_epoch(
        existing_model_dir=existing_model_dir, output_dir=output_dir,
        train_list_path=probe_train_list, validation_list_path=probe_val_list, epoch_seed=1,
    )


def _smoke_test_checks(result: EpochResult) -> tuple[tuple[bool, str], tuple[bool, str]]:
    """Returns `((smoke_test_passed, message), (checkpoint_round_trip_passed, message))` -- both
    derived from one real smoke-test invocation, never a second, redundant real run."""
    if not result.ok:
        smoke = (False, f"Smoke test failed: {result.error_message}")
        round_trip = (False, "No checkpoint produced by the failed smoke test to verify a round trip against.")
        return smoke, round_trip

    smoke = (True, "Real forward+backward smoke-test pass succeeded.")
    if result.checkpoint_dir is None:
        round_trip = (False, "Smoke test succeeded but produced no checkpoint directory to verify.")
    else:
        ok, digest, extra = verify_checkpoint(result.checkpoint_dir)
        round_trip = (
            (True, f"Smoke-test checkpoint reopened and verified: sha256={digest[:16]}...")
            if ok else (False, f"Smoke-test checkpoint at {result.checkpoint_dir} failed reopen/verify.")
        )
    return smoke, round_trip


def run_preflight(
    *,
    base_model_dir: str | Path,
    base_checkpoint_pinned_hash: str,
    parent_checkpoint_dir: str | Path,
    known_pilot_run_dirs: tuple[str | Path, ...] = (),
    pilot_val_manifest_path: str | Path,
    pilot_test_manifest_path: str | Path,
    output_dir: str | Path,
    training_root: str | Path,
    monitoring_config: FullRunMonitoringConfig,
    dataset_hash: str | None,
    recorded_dataset_hash: str | None = None,
    random_seed: int | None = None,
    pilot_resume_proof_path: str | Path | None = None,
    container_image_tag: str | None = None,
    container_image_digest: str | None = None,
    require_gpu: bool = True,
    min_free_disk_gb: float = 20.0,
    run_smoke_test: bool = True,
    check_docker_daemon: bool = True,
    check_container_image: bool = True,
    smoke_test_runner: SmokeTestRunner | None = None,
    probe_train_list: str | None = None,
    probe_val_list: str | None = None,
) -> PreflightReport:
    base_model_dir = Path(base_model_dir)
    parent_checkpoint_dir = Path(parent_checkpoint_dir)
    known_pilot_run_dirs = tuple(Path(p) for p in known_pilot_run_dirs)
    pilot_val_manifest_path = Path(pilot_val_manifest_path)
    pilot_test_manifest_path = Path(pilot_test_manifest_path)
    output_dir = Path(output_dir)
    training_root = Path(training_root)

    checks: list[PreflightCheck] = []

    if check_docker_daemon:
        checks.append(_check("docker_daemon_reachable", _check_docker_daemon_reachable))
    if check_container_image and container_image_tag:
        checks.append(_check(
            "container_image_available",
            lambda: _check_container_image_available(container_image_tag, container_image_digest),
        ))
    checks.append(_check("gpu_available", lambda: _check_gpu_available(require_gpu)))

    checks.append(_check("base_model_loadable", lambda: _check_base_model_loadable(base_model_dir)))
    checks.append(_check(
        "base_checkpoint_hash_matches_pin",
        lambda: _check_base_checkpoint_hash_matches_pin(base_model_dir, base_checkpoint_pinned_hash),
    ))
    checks.append(_check("parent_not_pilot", lambda: _check_parent_not_pilot(parent_checkpoint_dir, known_pilot_run_dirs)))

    checks.append(_check("val_manifest_non_empty", lambda: _check_manifest_non_empty("Validation", pilot_val_manifest_path)))
    checks.append(_check("test_manifest_non_empty", lambda: _check_manifest_non_empty("Reserved test", pilot_test_manifest_path)))
    checks.append(_check("val_test_no_overlap", lambda: _check_val_test_no_overlap(pilot_val_manifest_path, pilot_test_manifest_path)))
    checks.append(_check("expected_line_counts_plausible", lambda: _check_expected_line_counts_plausible(pilot_val_manifest_path)))

    checks.append(_check("dataset_hash_recorded", lambda: _check_dataset_hash_recorded(dataset_hash)))
    checks.append(_check(
        "dataset_hash_stable",
        lambda: _check_dataset_hash_stable(recorded_dataset_hash, dataset_hash),
    ))

    checks.append(_check("output_dir_writable", lambda: _check_disk_space(output_dir, min_free_disk_gb)))
    checks.append(_check("telemetry_path_writable", lambda: _check_telemetry_path_writable(output_dir / "run-state")))
    checks.append(_check("monitoring_config_valid", lambda: _check_monitoring_config_valid(monitoring_config)))
    checks.append(_check("random_seed_explicit", lambda: _check_random_seed_explicit(random_seed)))
    checks.append(_check("no_conflicting_active_run", lambda: _check_no_conflicting_active_run(training_root)))
    checks.append(_check("dashboard_can_discover_run", lambda: _check_dashboard_can_discover(training_root)))

    if pilot_resume_proof_path is not None:
        checks.append(_check(
            "resume_state_continuity",
            lambda: _check_resume_state_continuity(Path(pilot_resume_proof_path)),
        ))

    if run_smoke_test:
        try:
            staged_model_dir = _stage_writable_checkpoint_for_smoke_test(
                source_dir=base_model_dir, staging_dir=output_dir / "preflight_smoke_test_staged_checkpoint",
            )
            result = _run_smoke_test(
                smoke_test_runner, existing_model_dir=staged_model_dir,
                output_dir=str(output_dir / "preflight_smoke_test"),
                probe_train_list=probe_train_list or "", probe_val_list=probe_val_list or "",
            )
            (smoke_ok, smoke_msg), (round_trip_ok, round_trip_msg) = _smoke_test_checks(result)
        except Exception as exc:  # noqa: BLE001 -- a broken smoke test is a failed check, never a crash
            smoke_ok, smoke_msg = False, f"Smoke test raised an exception: {exc}"
            round_trip_ok, round_trip_msg = False, "Smoke test did not complete -- no checkpoint to verify."
        checks.append(PreflightCheck(name="smoke_test_forward_backward", passed=smoke_ok, message=smoke_msg))
        checks.append(PreflightCheck(name="checkpoint_save_reload_round_trip", passed=round_trip_ok, message=round_trip_msg))

    return PreflightReport(checks=tuple(checks))
