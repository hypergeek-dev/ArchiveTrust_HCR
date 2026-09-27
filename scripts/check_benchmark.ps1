<#
.SYNOPSIS
    One-command readiness check for the independent Loghi-vs-Lion HTR benchmark.

.DESCRIPTION
    1. Runs the benchmark harness test suite (tests/htr/benchmark).
    2. Runs `python -m archivetrust.htr.benchmark check`: Python and packages, the frozen Loghi
       checkpoint's SHA-256 identity, Docker daemon and the pinned Loghi container image, GPU, Lion
       dependencies, uncommitted benchmark code, incoming deliveries' inspection verdicts, and
       (with -Benchmark) the integrity of a frozen benchmark.
    Exit code 0 only if the tests pass and no check FAILs. Never modifies data or models.

.EXAMPLE
    .\scripts\check_benchmark.ps1
    .\scripts\check_benchmark.ps1 -Benchmark friend-2026-v1
    .\scripts\check_benchmark.ps1 -Python C:\path\to\venv\Scripts\python.exe
#>
param(
    [string]$Python = "",
    [string]$Benchmark = ""
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot

if (-not $Python) {
    $candidate = Join-Path $repo ".venv\Scripts\python.exe"
    if (Test-Path $candidate) { $Python = $candidate } else { $Python = "python" }
}
try {
    & $Python -c "import sys; assert sys.version_info >= (3, 11)" | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "python < 3.11" }
} catch {
    Write-Host "[FAIL] Python interpreter '$Python' is missing or broken (needs >= 3.11). Pass -Python <path>." -ForegroundColor Red
    exit 1
}

$env:PYTHONPATH = Join-Path $repo "src"
Push-Location $repo
try {
    Write-Host "== Benchmark harness tests ==" -ForegroundColor Cyan
    & $Python -m pytest -q -p no:cacheprovider tests/htr/benchmark
    $testsOk = ($LASTEXITCODE -eq 0)

    Write-Host "`n== Readiness ==" -ForegroundColor Cyan
    $checkArgs = @("-m", "archivetrust.htr.benchmark", "check")
    if ($Benchmark) { $checkArgs += @("--benchmark", $Benchmark) }
    & $Python @checkArgs
    $checkOk = ($LASTEXITCODE -eq 0)
} finally {
    Pop-Location
}

if ($testsOk -and $checkOk) {
    Write-Host "`nREADY" -ForegroundColor Green
    exit 0
}
Write-Host "`nNOT READY (tests ok: $testsOk, checks ok: $checkOk)" -ForegroundColor Yellow
exit 1
