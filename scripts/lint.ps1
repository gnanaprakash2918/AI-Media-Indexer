
# Script to run linting and formatting
# Usage: ./lint.ps1

Write-Host "🧹 Running Ruff Linter & Formatter..." -ForegroundColor Cyan

# Check for ruff
if (-not (Get-Command "ruff" -ErrorAction SilentlyContinue)) {
    Write-Host "⚠️  Ruff not found. Installing via pip..."
    & "uv" pip install ruff
}

Write-Host "🔍 Checking code functionality..." -ForegroundColor Yellow
# Run check with fix
ruff check . --fix

if ($LASTEXITCODE -eq 0) {
    Write-Host "OK - Lint check passed!" -ForegroundColor Green
} else {
    Write-Host "WARN - Lint issues found (some fixes applied)." -ForegroundColor Yellow
}

Write-Host "INFO - Formatting code style..." -ForegroundColor Yellow
ruff format .

if ($LASTEXITCODE -eq 0) {
    Write-Host "OK - Code formatted!" -ForegroundColor Green
} else {
    Write-Host "FAIL - Formatting failed." -ForegroundColor Red
}
