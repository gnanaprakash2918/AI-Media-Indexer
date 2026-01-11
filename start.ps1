<#
.SYNOPSIS
    Universal Startup Script for AI Media Indexer
.DESCRIPTION
    1. Checks Hardware (GPU/CPU/RAM).
    2. Manages ASR Mode (Native vs Docker).
    3. Syncs dependencies (including optional 'indic' if Native).
    4. Launches Qdrant, Agents, Backend, and Frontend.
.PARAMETER DockerASR
    If set, skips Native NeMo installation and starts the AI4Bharat Docker container instead.
.PARAMETER Reset
    If set, wipes the database before starting (WARNING: Destructive).
#>
param (
    [switch]$DockerASR,
    [switch]$Reset,
    [switch]$Lint,
    [switch]$Distributed
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot

if ($Lint) {
    if (Test-Path "$Root\scripts\lint.ps1") {
        & "$Root\scripts\lint.ps1"
    } else {
        Write-Warning "Lint script not found."
    }
}

Write-Host "`n🚀 AI Media Indexer: Production Start" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor DarkCyan
Write-Host "   Mode:          $($DockerASR ? 'Docker ASR' : 'Native ASR')"
Write-Host "   Ingestion:     $($Distributed ? 'Distributed (Celery+Redis)' : 'Local')"

# 1. Database Reset (Optional)
if ($Reset) {
    Write-Host "⚠️  Reset flag detected. Wiping Database..." -ForegroundColor Red
    if (Test-Path "$Root\scripts\reset_db.py") {
        & "$Root\.venv\Scripts\python.exe" "$Root\scripts\reset_db.py" --force
    } elseif (Test-Path "$Root\scripts\resetdb.py") {
         & "$Root\.venv\Scripts\python.exe" "$Root\scripts\resetdb.py" --force
    }
}

# 2. ASR Strategy Selection
if ($DockerASR) {
    Write-Host "🎙️  ASR Mode: DOCKER (AI4Bharat Container)" -ForegroundColor Yellow
    $Env:USE_NATIVE_NEMO = "False"
    
    # Start Docker Container
    Write-Host "   Starting indic-conformer container..."
    docker ps -q -f name=ai4bharat-asr | ForEach-Object { Write-Host "   Container running." }
} else {
    Write-Host "🎙️  ASR Mode: NATIVE (Local Python)" -ForegroundColor Green
    $Env:USE_NATIVE_NEMO = "True"
    
    # Check if 'indic' extra is synced
    Write-Host "   Verifying Native Dependencies (including NeMo)..."
    uv sync --extra indic
}

# 3. Infrastructure (Qdrant + Redis)
Write-Host "📦 Infrastructure Check..." -ForegroundColor Yellow
$QdrantStatus = docker ps -q -f name=media-indexer-qdrant
if (-not $QdrantStatus) {
    Write-Host "   Starting Qdrant..."
    docker compose up -d qdrant
}

if ($Distributed) {
    Write-Host "   Starting Redis for Distributed Ingestion..."
    docker compose up -d redis
    $Env:ENABLE_DISTRIBUTED_INGESTION = "True"
} else {
    $Env:ENABLE_DISTRIBUTED_INGESTION = "False"
}

Start-Sleep -Seconds 3

# 4. Hardware Profile
Write-Host "🔍 Hardware Check..." -ForegroundColor Yellow
& "uv" run python -c "import torch; import pynvml; print(f'   GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else '   GPU: None (CPU Mode)');"

# 5. Start Celery Worker (If Distributed)
$WorkerProc = $null
if ($Distributed) {
    Write-Host "👷 Starting Celery Worker..." -ForegroundColor Magenta
    # -P threads is usually better for I/O bound tasks on Windows, but let's stick to default or threads
    # Using 'solo' or 'threads' is often required on Windows to avoid spawning issues
    $WorkerProc = Start-Process -FilePath "uv" `
        -ArgumentList "run", "celery", "-A", "core.ingestion.celery_app", "worker", "--loglevel=info", "-P", "threads" `
        -PassThru -NoNewWindow
}

# 6. Start Backend (Async)
Write-Host "🔌 Starting Backend API (Port 8000)..." -ForegroundColor Green
$BackendProc = Start-Process -FilePath "uv" `
    -ArgumentList "run", "uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000", "--reload" `
    -PassThru -NoNewWindow

# 7. Start Frontend (Async)
Write-Host "🎨 Starting Frontend (Port 5173)..." -ForegroundColor Green
Set-Location "$Root\web"
if (-not (Test-Path "node_modules")) { npm install }
$FrontendProc = Start-Process -FilePath "npm" -ArgumentList "run", "dev" -PassThru -NoNewWindow

Write-Host "`n✅ System Online!" -ForegroundColor Cyan
Write-Host "   Backend API:   http://localhost:8000/docs"
Write-Host "   Frontend UI:   http://localhost:5173"
Write-Host "   Press Ctrl+C to stop all services..."

try {
    Wait-Process -Id $BackendProc.Id
}
finally {
    Write-Host "`n🛑 Shutting down..."
    Stop-Process -Id $BackendProc.Id -ErrorAction SilentlyContinue
    Stop-Process -Id $FrontendProc.Id -ErrorAction SilentlyContinue
    if ($WorkerProc) {
        Write-Host "   Stopping Celery Worker..."
        Stop-Process -Id $WorkerProc.Id -ErrorAction SilentlyContinue
    }
    Set-Location $Root
}
