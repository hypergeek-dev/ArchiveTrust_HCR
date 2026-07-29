"""Provider Health Guard (Phase 22, 2026-07-15): prevents a research/calibration run from
silently continuing once a required provider becomes unavailable mid-run.

Root incident: Docker became unavailable while a calibration run was in progress; the pipeline
kept processing documents with an incomplete provider set, and the resulting corpus had to be
discarded because provider availability changed during execution. `run_pipeline`'s existing
per-invocation "one provider fails, this document keeps going" behavior
(`application/pipeline.py`) is correct for a single transient hiccup on one document and is left
untouched; this module adds the separate, coarser, run-level check that incident actually needed.

Framework-independent (no PySide6 import) so it is cheaply unit-testable without Qt/threads.
`probes`/`recovery` are injected callables, never hardcoded provider-identity branches (`if
provider == "docling"` etc.) -- Constitution Article 20, provider independence, applies here too:
this guard classifies providers only by whether a health probe and/or a recovery function exists
for them, never by name.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from collections.abc import Callable, Mapping

from pydantic import BaseModel, ConfigDict

PROVIDER_HEALTH_GUARD_POLICY_VERSION = 1


class ProviderHealthCheck(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider_id: str
    healthy: bool
    detail: str


class RecoveryAttempt(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider_id: str
    attempted: bool
    succeeded: bool
    detail: str


class ProviderHealthGuardResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    healthy: bool
    checks: tuple[ProviderHealthCheck, ...]
    recovery_attempts: tuple[RecoveryAttempt, ...]
    reason: str
    """One human-readable summary -- empty string when `healthy` is True. Used directly as both
    the pause reason and the operator notification text, so there is exactly one place that turns
    a health-check result into words a human reads."""


class ProviderHealthGuardPolicy(BaseModel):
    """Versioned exactly like `TriagePolicy`/`ConfidencePolicy`/`ObservationScopePolicy`."""

    model_config = ConfigDict(frozen=True)

    version: int = PROVIDER_HEALTH_GUARD_POLICY_VERSION
    max_recovery_attempts: int = 1
    """Deliberately conservative (per this phase's own instruction): one bounded attempt, never
    indefinite retry."""
    recovery_wait_seconds: float = 5.0
    """Wait between a recovery attempt and re-probing, so a just-started daemon has a moment to
    come up before being re-checked."""


class ProviderHealthGuard:
    """Re-probes every provider a run started with, before each document. Never re-reads live
    provider configuration for its baseline -- `required_provider_ids` is a fixed snapshot the
    caller took once at run start (`QueueWorker.run()`'s `running_ids`), which is what makes "the
    provider set changed mid-run" detectable at all: a provider silently disabled or made
    unreachable after the run started is still checked against, even though it would no longer
    appear in a fresh `enabled_provider_ids()` call.
    """

    def __init__(
        self,
        *,
        required_provider_ids: frozenset[str],
        probes: Mapping[str, Callable[[], bool]],
        recovery: Mapping[str, Callable[[], RecoveryAttempt]] | None = None,
        policy: ProviderHealthGuardPolicy | None = None,
    ) -> None:
        self._required_provider_ids = required_provider_ids
        self._probes = probes
        self._recovery = recovery or {}
        self._policy = policy or ProviderHealthGuardPolicy()

    def check(self) -> ProviderHealthGuardResult:
        checks = tuple(self._probe(provider_id) for provider_id in sorted(self._required_provider_ids))
        unhealthy = [c for c in checks if not c.healthy]
        if not unhealthy:
            return ProviderHealthGuardResult(healthy=True, checks=checks, recovery_attempts=(), reason="")

        recovery_attempts: list[RecoveryAttempt] = []
        for check in unhealthy:
            recover = self._recovery.get(check.provider_id)
            if recover is None:
                continue
            recovery_attempts.append(self._attempt_recovery(check.provider_id, recover))

        # Re-probe only the providers a recovery function was actually attempted for -- a provider
        # with no recovery function registered stays exactly as unhealthy as first found.
        final_checks = list(checks)
        for attempt in recovery_attempts:
            if attempt.succeeded:
                idx = next(i for i, c in enumerate(final_checks) if c.provider_id == attempt.provider_id)
                final_checks[idx] = self._probe(attempt.provider_id)

        final_unhealthy = [c for c in final_checks if not c.healthy]
        if not final_unhealthy:
            return ProviderHealthGuardResult(
                healthy=True, checks=tuple(final_checks), recovery_attempts=tuple(recovery_attempts), reason=""
            )

        reason = "; ".join(
            f"{c.provider_id}: {c.detail}" for c in final_unhealthy
        )
        return ProviderHealthGuardResult(
            healthy=False,
            checks=tuple(final_checks),
            recovery_attempts=tuple(recovery_attempts),
            reason=f"Required provider(s) unavailable -- {reason}",
        )

    def _probe(self, provider_id: str) -> ProviderHealthCheck:
        probe = self._probes.get(provider_id)
        if probe is None:
            return ProviderHealthCheck(
                provider_id=provider_id, healthy=False, detail="no health probe registered for this provider"
            )
        try:
            healthy = bool(probe())
        except Exception as exc:  # noqa: BLE001 -- a probe raising is itself a health-check failure, never a crash
            return ProviderHealthCheck(provider_id=provider_id, healthy=False, detail=f"health probe raised: {exc}")
        return ProviderHealthCheck(
            provider_id=provider_id,
            healthy=healthy,
            detail="reachable" if healthy else "not currently reachable",
        )

    def _attempt_recovery(
        self, provider_id: str, recover: Callable[[], RecoveryAttempt]
    ) -> RecoveryAttempt:
        last = RecoveryAttempt(provider_id=provider_id, attempted=False, succeeded=False, detail="not attempted")
        for attempt_number in range(self._policy.max_recovery_attempts):
            last = recover()
            if last.succeeded:
                return last
            if attempt_number < self._policy.max_recovery_attempts - 1:
                time.sleep(self._policy.recovery_wait_seconds)
        return last


def attempt_docker_recovery(provider_id: str) -> RecoveryAttempt:
    """Conservative, bounded recovery for a Docker/vLLM-backed provider: verify Docker is even
    reachable, and if not, try starting Docker Desktop (Windows) once, then re-check. Never its
    own retry loop -- `ProviderHealthGuard._attempt_recovery` owns bounding/waiting.
    """
    if shutil.which("docker") is None:
        return RecoveryAttempt(
            provider_id=provider_id, attempted=False, succeeded=False, detail="docker is not on PATH"
        )

    if _docker_reachable():
        return RecoveryAttempt(provider_id=provider_id, attempted=False, succeeded=True, detail="docker already reachable")

    docker_desktop = _find_docker_desktop_executable()
    if docker_desktop is None:
        return RecoveryAttempt(
            provider_id=provider_id,
            attempted=False,
            succeeded=False,
            detail="docker daemon unreachable and no Docker Desktop executable found to start",
        )

    try:
        subprocess.Popen([str(docker_desktop)])  # noqa: S603 -- fixed, non-shell, no user input
    except OSError as exc:
        return RecoveryAttempt(
            provider_id=provider_id, attempted=True, succeeded=False, detail=f"failed to launch Docker Desktop: {exc}"
        )

    succeeded = _docker_reachable()
    return RecoveryAttempt(
        provider_id=provider_id,
        attempted=True,
        succeeded=succeeded,
        detail="docker reachable after starting Docker Desktop" if succeeded else "docker still unreachable after starting Docker Desktop",
    )


def _docker_reachable() -> bool:
    try:
        result = subprocess.run(
            ["docker", "info"], capture_output=True, timeout=10, check=False  # noqa: S603, S607
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _find_docker_desktop_executable():
    from pathlib import Path

    candidate = Path("C:/Program Files/Docker/Docker/Docker Desktop.exe")
    return candidate if candidate.exists() else None
