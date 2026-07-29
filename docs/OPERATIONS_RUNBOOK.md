# ArchiveTrust Operations Runbook

Status: Current, incomplete pending deployment qualification  
Scope: Single managed Windows workstation  
Governs: Operator/admin procedures  
Applies to version: 0.1.0 / Production Closure working tree  
Supersedes: scattered operational command notes  
Superseded by: none  
Last verified against code: 2026-07-17

There is no qualified installer. Controlled use requires a Python 3.11+ virtual environment,
`.[gui]`, and explicitly selected provider extras. Start with
`python -m archivetrust.clients.desktop`; first production startup explicitly creates an admin.
Processing launches a separate local worker command process. Worker commands and controls are
deployment-bound authenticated envelopes; there is no service stop command.

Run `archivetrust-admin health --deployment-root <root>` and `monitor` daily. Resolve critical
integrity, writability, disk, queue, worker-command security, config-lock, and backup alerts before
work. Acknowledge only after resolution with `acknowledge-alert <id> --actor <account>`.

- Retry/restart: reopen the client; incomplete runs reconcile before a new run.
- Historical closure: dry-run then `reconcile-runs <workspace> ... --apply`.
- Telemetry epoch reset: after a stable release boundary is accepted, run
  `freeze-telemetry-history <workspace> --label <label> --report <file>`. This hashes the current
  telemetry files, records byte/line cursors, writes `config/active_telemetry_epoch.json`, and
  leaves the historical JSONL streams untouched. New qualification runs should be measured from
  the active epoch cursors rather than from pre-stability history.
- Qualification evidence: edit `qualification_run_config.json`, then run
  `archivetrust-admin qualification` for the three-option interactive menu. The non-interactive
  resume/start surface is `archivetrust-admin qualification-run --config <file> [--resume]`.
  While running, the runner prints a live processed-document counter. From another terminal, run
  `archivetrust-admin qualification-status --config <file>` to see the latest checkpoint progress.
  Full P4/P5/P10 qualification must keep `required_providers` enabled for `docling`,
  `tesseract_layoutparser`, `paddleocr-vl`, and `surya`; CPU-only runs are partial development
  evidence and must not be counted as full qualification. Reports are evidence only and must not be
  treated as production approval.
- Stale truth: dry-run `regenerate-current-state`; apply requires an outside-workspace backup and
  supports checkpoints.
- Config: `lock-config` after approval; `verify-config` before processing.
- Worker command boundary: installation must create/verify restrictive permissions for
  `worker/commands`, `worker/control`, `worker/results`, `worker/freshness_state.json`,
  `identity/`, workspace telemetry/blob/config paths, and qualification output. Treat rejected
  commands, replay attempts, stale commands without results, missing freshness state, or worker
  filesystem-boundary warnings as incidents until reviewed. Preserve command, result, freshness,
  audit, health, and diagnostic evidence before cleanup. If freshness state is damaged, stop
  processing, preserve the deployment root, restore from backup where possible, or perform an
  explicitly approved reset that records which commands may need reconciliation.
- Backup: stop activity, run `backup`, then `verify-backup`. Restore only to an empty target, run
  health, open/replay, inspect a reviewed document, and export. Backups are not encrypted.
- Incident: preserve deployment/backup and create a sanitized `diagnostic-bundle`.

Upgrade, rollback, uninstall, service installation, and clean-machine production restoration remain
unqualified and must not be improvised for production data.
