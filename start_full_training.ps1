# Activates the project's virtual environment and opens the full-corpus training GUI.
# Run from the repository root: .\start_full_training.ps1
#
# Equivalent, without this script:
#   .venv\Scripts\Activate.ps1
#   $env:PYTHONPATH = "src"
#   python -m archivetrust.htr.training.full_run gui

$ErrorActionPreference = "Stop"

$RepoRoot = $PSScriptRoot
Set-Location $RepoRoot

$VenvActivate = Join-Path $RepoRoot ".venv\Scripts\Activate.ps1"
if (-not (Test-Path $VenvActivate)) {
    Write-Error "No .venv found at $VenvActivate -- create it first (python -m venv .venv) and install the project with the 'gui' extra: pip install -e `".[gui]`""
}
& $VenvActivate

$env:PYTHONPATH = Join-Path $RepoRoot "src"

python -m archivetrust.htr.training.full_run gui
