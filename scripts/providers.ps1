#Requires -Version 5.1
<#
.SYNOPSIS
    Start/stop/status/logs for ArchiveTrust's Docker/vLLM-served providers (paddleocr-vl, surya).

.DESCRIPTION
    Thin wrapper around `archivetrust-bootstrap providers` -- deliberately NOT a Docker Compose
    file. ArchiveTrust's real container-lifecycle owner for these providers is already
    `archivetrust.runtime.vllm_runtime.VLLMRuntime` (deterministic container names/ports, pinned
    image, tuned GPU/max-model-len settings). This script calls into that same production code
    through the `archivetrust-bootstrap` CLI so there is exactly one place that knows how these
    containers are actually started -- see `src/archivetrust/bootstrap/providers.py`.

.PARAMETER Action
    start  - warm up (start if needed, wait for health) the named provider.
    stop   - stop the named provider's container.
    status - report running/healthy state without starting or stopping anything. Omit -Provider
             to report both.
    logs   - print the container's recent log tail.

.PARAMETER Provider
    'paddleocr-vl' or 'surya'. Required for start/stop/logs; optional for status.

.EXAMPLE
    .\providers.ps1 status
.EXAMPLE
    .\providers.ps1 start paddleocr-vl
.EXAMPLE
    .\providers.ps1 logs surya
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet('start', 'stop', 'status', 'logs')]
    [string] $Action,

    [Parameter(Position = 1)]
    [ValidateSet('paddleocr-vl', 'surya')]
    [string] $Provider,

    [int] $Tail = 200
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if ($Action -in @('start', 'stop', 'logs') -and -not $Provider) {
    Write-Host "Action '$Action' requires -Provider paddleocr-vl|surya." -ForegroundColor Red
    exit 2
}

$argumentList = @('-m', 'archivetrust.bootstrap.cli', 'providers', $Action)
if ($Provider) {
    $argumentList += $Provider
}
if ($Action -eq 'logs') {
    $argumentList += @('--tail', [string] $Tail)
}

& python @argumentList
exit $LASTEXITCODE
