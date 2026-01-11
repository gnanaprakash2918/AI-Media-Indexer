
# Script to install Native AI4Bharat ASR dependencies
# Usage: ./install_native_nemo.ps1

Write-Host "📦 Installing Native AI4Bharat NeMo Dependencies..." -ForegroundColor Cyan

# Check if uv is installed
if (-not (Get-Command "uv" -ErrorAction SilentlyContinue)) {
    Write-Error "uv is not installed. Please install it first."
    exit 1
}

# Install optional group 'indic'
# This pulls nemo-toolkit[asr] and compatible protobuf
Write-Host "Running: uv sync --extra indic" -ForegroundColor Yellow
uv sync --extra indic

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Native NeMo Installed!" -ForegroundColor Green
    Write-Host "   You can now set use_native_nemo = True in config.py (Default)"
} else {
    Write-Error "Failed to install dependencies. Check log for conflicts."
}
