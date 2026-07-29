#Requires -Version 5.1
<#
.SYNOPSIS
    ArchiveTrust Windows development bootstrap / installation candidate. NOT a production
    installer -- see docs/WINDOWS_BOOTSTRAP.md for what is implemented versus still manual.

.DESCRIPTION
    Currently implements: preflight checks (Windows version/architecture/admin state, disk/RAM,
    Python version, Git/Docker/WSL/GPU presence via injected checks, port availability, path
    safety, .env safety) and a durable, resumable install-state file
    (<DeploymentRoot>\bootstrap\install_state.json). All real logic lives in the
    `archivetrust-bootstrap` Python CLI (src/archivetrust/bootstrap/) so it is unit-tested with
    pytest; this script is the operator-facing entry point and does real OS-level orchestration
    (elevation awareness, safe argument construction), not a reimplementation of that logic.

    Python environment setup, Docker/provider container orchestration, deployment-root
    initialization, administrator bootstrap, ACL application, and the installation smoke test are
    NOT implemented by this script yet -- their phases exist in the install-state schema
    (reported by -Mode Verify) but are explicitly marked "not implemented yet" rather than
    silently skipped or falsely reported complete.

.PARAMETER Mode
    Install   - run preflight and report current bootstrap state; safe to run on a fresh machine.
    Resume    - identical to Install for the phases currently implemented (only preflight); kept
                as a distinct mode now so scripts and documentation that reference it do not need
                to change once later phases add real resume behavior.
    Verify    - read-only: print the current install-state phases without running preflight again.
    Repair    - re-runs preflight (the only phase with real logic yet) and reports the result;
                does not attempt to repair phases that are not implemented.

.PARAMETER DeploymentRoot
    Where the durable install-state file and (eventually) the ArchiveTrust deployment live.
    Defaults to .\archivetrust_data relative to the current directory.

.EXAMPLE
    .\setup-archivetrust.ps1 -Mode Install
.EXAMPLE
    .\setup-archivetrust.ps1 -Mode Verify -DeploymentRoot D:\ArchiveTrust\archivetrust_data
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('Install', 'Resume', 'Verify', 'Repair')]
    [string] $Mode,

    [string] $DeploymentRoot,

    [int[]] $Port = @()
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

Import-Module (Join-Path $PSScriptRoot 'bootstrap\Common.psm1') -Force

if (-not $DeploymentRoot) {
    $DeploymentRoot = Get-DefaultDeploymentRoot
}

Write-Host "ArchiveTrust Windows bootstrap -- development bootstrap / installation candidate, not a production installer." -ForegroundColor Cyan
Write-Host ("Mode: {0}" -f $Mode)
Write-Host ("Deployment root: {0}" -f $DeploymentRoot)
Write-Host ""

if (-not (Test-BootstrapCliAvailable)) {
    Write-Host "BLOCKED: 'python -m archivetrust.bootstrap.cli' did not run." -ForegroundColor Red
    Write-Host "Exact next action: install Python 3.11+ and run 'pip install -e .' from the repository root, then re-run this script." -ForegroundColor Yellow
    exit 1
}

switch ($Mode) {
    'Verify' {
        $status = Get-BootstrapStatus -DeploymentRoot $DeploymentRoot
        if (-not $status.Success) {
            Write-Host ("Could not read install state: {0}" -f $status.Error) -ForegroundColor Red
            exit 2
        }
        Write-Host "Current phase state:"
        Write-PhaseSummary -Report $status.Report
        if ($status.Report.overall_blocked) {
            Write-Host "BLOCKED: at least one phase is in a failed/blocked state." -ForegroundColor Red
            exit 1
        }
        exit 0
    }

    { $_ -in @('Install', 'Resume', 'Repair') } {
        Write-Host "Running preflight checks (real, not simulated)..."
        $preflight = Invoke-BootstrapPreflight -DeploymentRoot $DeploymentRoot -Port $Port
        if (-not $preflight.Success -and $preflight.Error) {
            Write-Host ("Preflight failed to run: {0}" -f $preflight.Error) -ForegroundColor Red
            exit 2
        }
        foreach ($check in $preflight.Report.checks) {
            $color = if ($check.status -eq 'ok') { 'Green' } else { 'Yellow' }
            Write-Host ("  [{0}] {1}: {2}" -f $check.status, $check.name, $check.detail) -ForegroundColor $color
        }
        Write-Host ""
        Write-Host ("Preflight overall status: {0}" -f $preflight.Report.overall_status)

        $status = Get-BootstrapStatus -DeploymentRoot $DeploymentRoot
        if ($status.Success) {
            Write-Host ""
            Write-Host "Full bootstrap phase state (most phases are schema-only placeholders until later work lands):"
            Write-PhaseSummary -Report $status.Report
        }

        Write-Host ""
        Write-Host "Automated by this run: preflight checks, durable state recording." -ForegroundColor Cyan
        Write-Host "NOT automated yet (manual/future work): Python virtual environment setup, Docker/provider container orchestration, deployment-root initialization, administrator bootstrap, ACL application, installation smoke test." -ForegroundColor Yellow
        Write-Host ("Exact resume command: .\setup-archivetrust.ps1 -Mode Resume -DeploymentRoot `"{0}`"" -f $DeploymentRoot)

        if ($preflight.Report.overall_status -eq 'completed') {
            exit 0
        }
        else {
            Write-Host "PARTIAL: preflight found issues that should be resolved before continuing (see above)." -ForegroundColor Yellow
            exit 1
        }
    }
}
