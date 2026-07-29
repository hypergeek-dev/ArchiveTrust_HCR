# ArchiveTrust Windows Bootstrap

Status: development bootstrap / installation candidate -- **not a production installer**.
Created: 2026-07-20

## What this is

`scripts/setup-archivetrust.ps1` is a reproducible entry point for getting a development or
trial ArchiveTrust environment running on Windows. It is explicitly not qualified as a P8
production installer -- see `docs/P4_P10_CLOSURE_ROADMAP.md` section 6.5 for what P8 still
requires (SBOM, code-signing decision, clean-machine install proof, upgrade/rollback/uninstall
qualification, and more).

## What is implemented today

- **Preflight checks** (`src/archivetrust/bootstrap/preflight.py`): Windows platform/architecture,
  elevation state, disk space, total RAM, Python version, installation/deployment path safety
  (rejects dangerous roots via the same `is_dangerous_root` the ACL policy uses), `.env` safety,
  and presence/health checks for Git, Docker Desktop, the Docker daemon, Docker Compose, WSL2, and
  an NVIDIA GPU/driver. Checks that shell out do so through an injectable `CommandRunner`, so the
  test suite (`tests/bootstrap/test_preflight.py`) never actually invokes Docker or WSL.
- **Durable, resumable install-state** (`src/archivetrust/bootstrap/install_state.py`): a versioned
  JSON file at `<deployment_root>/bootstrap/install_state.json`, written atomically, one record
  per phase (status, attempt count, timestamps, last error classification). Corruption (invalid
  JSON, unsupported schema version) is detected and surfaced, never silently discarded and
  recreated. Every phase this bootstrap will eventually need
  (`docker_desktop_detected` ... `smoke_test_executed`, see `Phase` enum) already exists in the
  schema so later work does not need a breaking migration -- but only `preflight_complete` has
  real logic behind it right now. `-Mode Verify`/the `status` CLI command report every other phase
  as "not implemented yet," never as if it had silently run.
- **CLI** (`archivetrust-bootstrap` / `python -m archivetrust.bootstrap.cli`): `preflight` and
  `status` subcommands. This is what `setup-archivetrust.ps1` calls into -- the actual logic lives
  in Python so it is testable with pytest; PowerShell is the operator-facing shell and does real
  OS-level orchestration (elevation awareness, safe argument-array construction), not a
  reimplementation of the state machine.
- **`scripts/setup-archivetrust.ps1`**: `-Mode Install|Resume|Verify|Repair`. `Install`/`Resume`/
  `Repair` all currently run the same real preflight (Resume and Repair are kept as distinct modes
  now, before there is a difference in behavior, so later phases can add real resume/repair logic
  without a breaking CLI change). `Verify` is read-only and only reports state.

## Provider start/stop/status/logs

```powershell
.\scripts\providers.ps1 status
.\scripts\providers.ps1 start paddleocr-vl
.\scripts\providers.ps1 stop surya
.\scripts\providers.ps1 logs paddleocr-vl
```

Deliberately **not** a Docker Compose file. ArchiveTrust's real container-lifecycle owner for
`paddleocr-vl`/`surya` is already `archivetrust.runtime.vllm_runtime.VLLMRuntime` (deterministic
container names/ports, pinned `vllm/vllm-openai` image, tuned GPU/max-model-len settings,
live-verified on real hardware). `scripts/providers.ps1` and
`src/archivetrust/bootstrap/providers.py` are a thin wrapper around that same production code
(`warm_up()`/`is_healthy()`/`stop()`), not a second, independently-tracked container lifecycle --
see `artifacts/provider_environment_20260720/provider_environment_summary.md` for why a Compose
file would have duplicated (and risked drifting from) the real one. `status`/`logs` are
non-destructive; `start` downloads the model checkpoint on first run (~1.8GB for `paddleocr-vl`,
~1.3GB for `surya`) and reserves real GPU memory.

## What is not implemented yet

Python virtual environment setup, dependency locking, deployment-root initialization,
administrator bootstrap, ACL policy application during setup (the policy engine itself is real and
separately usable via `archivetrust-admin acl-plan`/`acl-apply`/`acl-verify` -- see
`artifacts/acl_implementation_validation_20260720/`), the installation smoke test, and the
clean-environment validation harness. Every one of these phases exists in the install-state schema
and is reported honestly as `not implemented yet` by `-Mode Verify`. Provider start/stop/status/logs
(above) are implemented; wiring provider startup into `-Mode Install`'s automatic flow is not.

## Operator commands

```powershell
.\scripts\setup-archivetrust.ps1 -Mode Install
.\scripts\setup-archivetrust.ps1 -Mode Verify -DeploymentRoot D:\ArchiveTrust\archivetrust_data
.\scripts\setup-archivetrust.ps1 -Mode Repair
```

Requirements: Python 3.11+ with `pip install -e .` already run (the script checks for this and
gives an exact next action if it is missing). No administrator rights are required for the phases
implemented today.

## Tests

`tests/bootstrap/` (47 tests): install-state schema/versioning/atomic-write/corruption/resume
behavior, preflight checks (with injected fake command runners, no real Docker/WSL invocation),
provider start/stop/status/logs (with injected fake container lifecycle/health probe, reusing
`tests/runtime/_fakes.py` -- the same doubles `vllm_runtime`'s own suite uses), and the CLI's
`preflight`/`status`/`providers` commands. Full repository suite: 1216 passed, 2 skipped.

## Known gaps

- No installer packaging, code-signing, or SBOM -- P8 scope, not this bootstrap.
- `-Mode Repair` does not yet repair anything beyond re-running preflight, because nothing else is
  implemented yet to repair.
- PowerShell version and execution-policy checks are not yet wired into `preflight` (Python cannot
  reliably self-report the calling shell's version); deferred to when the PowerShell entry point
  needs to gate on them.
- This has been exercised on one development machine only; no clean-machine evidence exists yet.
