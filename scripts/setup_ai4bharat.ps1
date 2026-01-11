
# Script to setup and run AI4Bharat ASR Docker container
# Usage: ./setup_ai4bharat.ps1

Write-Host "🚀 Setting up AI4Bharat ASR Service..." -ForegroundColor Cyan

# check if docker is installed
if (-not (Get-Command "docker" -ErrorAction SilentlyContinue)) {
    Write-Error "Docker is not installed or not in PATH. Please install Docker Desktop."
    exit 1
}

# Build Image
Write-Host "📦 Building Docker Image (ai-media-asr)... This may take a while." -ForegroundColor Yellow
docker build -t ai-media-asr -f Dockerfile.asr .

if ($LASTEXITCODE -ne 0) {
    Write-Error "Build failed."
    exit $LASTEXITCODE
}

# Run Container
Write-Host "🏃 Starting Container on Port 8001..." -ForegroundColor Green
docker run -d -p 8001:8000 --gpus all --name ai4bharat-asr ai-media-asr

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ AI4Bharat Service Started!" -ForegroundColor Green
    Write-Host "   URL: http://localhost:8001"
    Write-Host "   Health: http://localhost:8001/health"
} else {
    Write-Error "Failed to start container."
}
