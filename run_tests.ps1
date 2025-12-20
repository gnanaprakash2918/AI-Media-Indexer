param (
    [switch]$Nuke
)

$ErrorActionPreference = "Stop"

if ($Nuke) {
    $nukeResponse = 'y'
} else {
    $nukeResponse = Read-Host "Nuke venv for reproducibility check? (y/n)"
}

if ($nukeResponse -eq 'y') {
    if (Test-Path .venv) {
        Write-Host "Removing .venv..." -ForegroundColor Yellow
        Remove-Item .venv -Recurse -Force
    }
    Write-Host "Repulling dependencies..." -ForegroundColor Green
    uv sync
}

# Use uv run to ensure environment consistency
uv run python scripts/run_tests.py $args
