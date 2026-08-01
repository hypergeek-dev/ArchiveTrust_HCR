"""Preflight checks for a full-corpus run -- training must not start when any critical check fails
(brief §5). Each check is its own small, independently testable function; `run_preflight` assembles
them into one `PreflightReport`, catching any individual check's exception as a failed check rather
than letting one broken probe crash the whole preflight.

Reuses real primitives throughout, never a parallel probing mechanism: `checkpoint_index.
verify_checkpoint` (base-model + checkpoint-round-trip checks), `identity.
assert_parent_checkpoint_is_not_a_pilot_path`, `providers/loghi/environment.probe_loghi_environment`
(GPU/CUDA), `shutil.disk_usage` (the same pattern `telemetry_sampler.py` already uses). The dry-run
forward+backward smoke test reuses `container_epoch_runner.ContainerEpochRunner`/`memory_probe.py`'s
exact real-but-bounded pattern via an injectable `EpochRunner` (a real container in production, a
`FakeEpochRunner` in tests -- the same seam `training_session.py` already established).
"""

from __future__ import annotations

import shutil
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


def _check(name: str, fn) -> PreflightCheck:
    try:
        passed, message = fn()
        return PreflightCheck(name=name, passed=passed, message=message)
    except Exception as exc:  # noqa: BLE001 -- a broken probe is a failed check, never a crash
        return PreflightCheck(name=name, passed=False, message=f"Check raised an exception: {exc}")


def _check_base_model_loadable(base_model_dir: Path) -> tuple[bool, str]:
    ok, digest, extra = verify_checkpoint(base_model_dir)
    if not ok:
        return False, f"{base_model_dir} does not contain a real, readable .keras checkpoint."
    return True, f"Base model verified: sha256={digest[:16]}..., {extra}"


def _check_parent_not_pilot(
    parent_checkpoint_dir: Path, known_pilot_run_dirs: tuple[Path, ...]
) -> tuple[bool, str]:
    try:
        assert_parent_checkpoint_is_not_a_pilot_path(parent_checkpoint_dir, known_pilot_run_dirs=known_pilot_run_dirs)
    except PilotCheckpointRejected as exc:
        return False, str(exc)
    return True, "Parent checkpoint is not under any known pilot run directory."


def _check_path_exists(label: str, path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, f"{label} does not exist: {path}"
    return True, f"{label} exists: {path}"


def _check_manifest_non_empty(label: str, manifest_path: Path) -> tuple[bool, str]:
    if not manifest_path.exists():
        return False, f"{label} manifest does not exist: {manifest_path}"
    table = pq.read_table(manifest_path, columns=["line_id"])
    if table.num_rows == 0:
        return False, f"{label} manifest is empty: {manifest_path}"
    return True, f"{label} manifest has {table.num_rows} real rows."


def _check_train_val_no_overlap(train_manifest_paths: tuple[Path, ...], val_manifest_path: Path) -> tuple[bool, str]:
    if not val_manifest_path.exists():
        return False, f"Validation manifest does not exist: {val_manifest_path}"
    val_ids = set(pq.read_table(val_manifest_path, columns=["line_id"]).column("line_id").to_pylist())
    overlap: set[str] = set()
    for path in train_manifest_paths:
        if not path.exists():
            continue
        train_ids = set(pq.read_table(path, columns=["line_id"]).column("line_id").to_pylist())
        overlap |= train_ids & val_ids
    if overlap:
        sample = sorted(overlap)[:5]
        return False, f"{len(overlap)} line ID(s) appear in both a training shard and validation (e.g. {sample})."
    return True, "No overlap between training shards and validation line IDs."


def _check_disk_space(output_dir: Path, min_free_gb: float) -> tuple[bool, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(output_dir).free / (1024**3)
    if free_gb < min_free_gb:
        return False, f"Only {free_gb:.1f}GB free at {output_dir}, below the required {min_free_gb}GB."
    return True, f"{free_gb:.1f}GB free at {output_dir}."


def _check_gpu_available(require_gpu: bool) -> tuple[bool, str]:
    from archivetrust.providers.loghi.environment import probe_loghi_environment

    report = probe_loghi_environment()
    if not require_gpu:
        return True, "GPU not required for this configuration."
    if not report.docker_cli_present:
        return False, "Docker CLI not available -- cannot run GPU-backed training containers."
    return True, f"Docker CLI present, execution_mode={report.execution_mode}."


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


def _check_dataset_hash_recorded(dataset_hash: str | None) -> tuple[bool, str]:
    if not dataset_hash:
        return False, "No dataset hash recorded -- the source inventory's provenance is not pinned."
    return True, f"Dataset hash recorded: {dataset_hash[:16]}..."


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
    parent_checkpoint_dir: str | Path,
    known_pilot_run_dirs: tuple[str | Path, ...] = (),
    train_manifest_paths: tuple[str | Path, ...],
    val_manifest_path: str | Path,
    output_dir: str | Path,
    monitoring_config: FullRunMonitoringConfig,
    dataset_hash: str | None,
    require_gpu: bool = True,
    min_free_disk_gb: float = 20.0,
    run_smoke_test: bool = True,
    smoke_test_runner: SmokeTestRunner | None = None,
    probe_train_list: str | None = None,
    probe_val_list: str | None = None,
) -> PreflightReport:
    base_model_dir = Path(base_model_dir)
    parent_checkpoint_dir = Path(parent_checkpoint_dir)
    known_pilot_run_dirs = tuple(Path(p) for p in known_pilot_run_dirs)
    train_manifest_paths = tuple(Path(p) for p in train_manifest_paths)
    val_manifest_path = Path(val_manifest_path)
    output_dir = Path(output_dir)

    checks = [
        _check("base_model_loadable", lambda: _check_base_model_loadable(base_model_dir)),
        _check("parent_not_pilot", lambda: _check_parent_not_pilot(parent_checkpoint_dir, known_pilot_run_dirs)),
        _check("output_dir_writable", lambda: _check_disk_space(output_dir, min_free_disk_gb)),
        _check("gpu_available", lambda: _check_gpu_available(require_gpu)),
        _check("monitoring_config_valid", lambda: _check_monitoring_config_valid(monitoring_config)),
        _check("dataset_hash_recorded", lambda: _check_dataset_hash_recorded(dataset_hash)),
        _check("val_manifest_non_empty", lambda: _check_manifest_non_empty("Validation", val_manifest_path)),
        _check("train_val_no_overlap", lambda: _check_train_val_no_overlap(train_manifest_paths, val_manifest_path)),
    ]
    for i, path in enumerate(train_manifest_paths):
        checks.append(_check(f"train_manifest_{i}_non_empty", lambda p=path: _check_manifest_non_empty(f"Train shard {i}", p)))

    if run_smoke_test:
        try:
            result = _run_smoke_test(
                smoke_test_runner, existing_model_dir=str(base_model_dir),
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
