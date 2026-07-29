#Requires -Version 5.1
<#
    Shared helpers for scripts\setup-archivetrust.ps1. Kept in its own module (rather than one
    monolithic top-level script) so each function can be dot-sourced and exercised individually.

    This is a development bootstrap / installation-candidate, not a production installer. It
    delegates the state machine and preflight logic to the `archivetrust-bootstrap` Python CLI
    (src/archivetrust/bootstrap/) rather than reimplementing it in PowerShell, so that logic stays
    testable with pytest. PowerShell's job here is real OS-level orchestration (elevation,
    Docker/WSL presence at a glance, argument construction) and operator-facing output.
#>

Set-StrictMode -Version Latest

function Get-DefaultDeploymentRoot {
    [CmdletBinding()]
    param()
    return (Join-Path -Path (Get-Location) -ChildPath 'archivetrust_data')
}

function Test-BootstrapCliAvailable {
    <# Returns $true only if `python -m archivetrust.bootstrap.cli` actually runs -- distinct
       from "python is on PATH", which the CLI's own preflight check reports separately. #>
    [CmdletBinding()]
    param()
    try {
        $null = & python -m archivetrust.bootstrap.cli --help 2>$null
        return ($LASTEXITCODE -eq 0)
    }
    catch {
        return $false
    }
}

function Invoke-BootstrapPreflight {
    <# Calls the real preflight implementation. All arguments are passed as an array, never
       string-concatenated, so a deployment-root path containing spaces or special characters
       cannot be mis-parsed or used for command injection. #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string] $DeploymentRoot,
        [string] $InstallationPath,
        [int[]] $Port = @()
    )
    $argumentList = @('-m', 'archivetrust.bootstrap.cli', 'preflight', '--deployment-root', $DeploymentRoot)
    if ($InstallationPath) {
        $argumentList += @('--installation-path', $InstallationPath)
    }
    foreach ($p in $Port) {
        $argumentList += @('--port', [string] $p)
    }
    $output = & python @argumentList
    $exitCode = $LASTEXITCODE
    try {
        $parsed = $output | ConvertFrom-Json
    }
    catch {
        return [pscustomobject]@{ Success = $false; ExitCode = $exitCode; Error = "preflight output was not valid JSON: $output" }
    }
    return [pscustomobject]@{ Success = ($exitCode -eq 0); ExitCode = $exitCode; Report = $parsed }
}

function Get-BootstrapStatus {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string] $DeploymentRoot
    )
    $output = & python -m archivetrust.bootstrap.cli status --deployment-root $DeploymentRoot
    $exitCode = $LASTEXITCODE
    try {
        $parsed = $output | ConvertFrom-Json
    }
    catch {
        return [pscustomobject]@{ Success = $false; ExitCode = $exitCode; Error = "status output was not valid JSON: $output" }
    }
    return [pscustomobject]@{ Success = ($exitCode -eq 0); ExitCode = $exitCode; Report = $parsed }
}

function Write-PhaseSummary {
    <# Prints one line per phase from a `status` report, distinguishing implemented phases (real
       logic ran) from placeholder phases (schema exists, no logic yet) -- never presents a
       placeholder as if it had actually executed. #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)] $Report
    )
    $phaseNames = $Report.phases.PSObject.Properties.Name | Sort-Object
    foreach ($name in $phaseNames) {
        $phase = $Report.phases.$name
        if (-not $phase.implemented) {
            Write-Host ("  [not implemented yet] {0}" -f $name) -ForegroundColor DarkGray
            continue
        }
        $color = switch ($phase.status) {
            'completed' { 'Green' }
            'failed' { 'Red' }
            'blocked' { 'Red' }
            'reboot_required' { 'Yellow' }
            'manual_action_required' { 'Yellow' }
            default { 'Gray' }
        }
        Write-Host ("  [{0}] {1} (attempts={2})" -f $phase.status, $name, $phase.attempts) -ForegroundColor $color
    }
}

Export-ModuleMember -Function `
    Get-DefaultDeploymentRoot, `
    Test-BootstrapCliAvailable, `
    Invoke-BootstrapPreflight, `
    Get-BootstrapStatus, `
    Write-PhaseSummary
