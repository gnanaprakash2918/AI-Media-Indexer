#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Full system startup script for AI-Media-Indexer (Windows)
.DESCRIPTION
    - Optionally recreates virtual environment with uv
    - Clears caches (pycache, face cache, pytest cache)
    - Stops and removes Docker containers (including orphans)
    - Starts Ollama
    - Starts Docker services
    - Launches backend and frontend in separate terminal windows (default)

.PARAMETER Quick
    Quick start - skip cleanup, keep all data (same as option 1)
.PARAMETER Fresh
    Clear caches but keep indexed data (same as option 2)
.PARAMETER Nuclear
    Wipe all data including Qdrant (same as option 3)
.PARAMETER Distributed
    Enable Distributed Ingestion (Redis + Celery Worker)
.PARAMETER InstallIndic
    Install NeMo toolkit for Indic language ASR (Tamil, Hindi, etc.)
.PARAMETER Full
    Full setup: Nuclear + Dev Mode + Indic ASR + Pull Images
.PARAMETER SkipOllama
    Skip Ollama startup
.PARAMETER SkipDocker
    Skip Docker operations
.PARAMETER SkipClean
    Skip cache cleanup
.PARAMETER RecreateVenv
    Remove .venv and recreate with uv sync
.PARAMETER NukeQdrant
    Delete Qdrant data directory (qdrant_data)
.PARAMETER PullImages
    Pull latest Docker images before starting
.PARAMETER Integrated
    Run backend in foreground with frontend as background process
.PARAMETER NoInteractive
    Skip interactive menu, use defaults

.EXAMPLE
    ./start.ps1 -Quick
    Quick start, keep all data

.EXAMPLE
    ./start.ps1 -Nuclear
    Wipe all data and start fresh

.EXAMPLE
    ./start.ps1 -Nuclear -Distributed
    Wipe data + start with Celery workers

.EXAMPLE
    ./start.ps1 -InstallIndic
    Install Tamil/Hindi ASR support

.EXAMPLE
    ./start.ps1 -Full
    Full setup with all features (Nuclear + Dev + Indic + Images)

.EXAMPLE
    ./start.ps1 -Nuclear -InstallIndic
    Wipe data + install Indic ASR
#>

param(
    # Combined convenience flags
    [switch]$Quick,
    [switch]$Fresh,
    [switch]$Nuclear,
    [switch]$Full,
    
    # Feature flags
    [switch]$Distributed,
    [switch]$InstallIndic,
    
    # Granular flags
    [switch]$SkipOllama,
    [switch]$SkipDocker,
    [switch]$SkipClean,
    [switch]$RecreateVenv,
    [switch]$NukeQdrant,
    [switch]$PullImages,
    [switch]$Integrated,
    [switch]$NoInteractive,
    
    # Agent mode
    [switch]$Agent,
    [string]$AgentTask = "analyze"
)

$ErrorActionPreference = "Continue"
$ProjectRoot = $PSScriptRoot

# Process combined flags
if ($Quick) {
    $SkipClean = $true
    $NoInteractive = $true
}
if ($Fresh) {
    $SkipClean = $false
    $NoInteractive = $true
}
if ($Nuclear) {
    $SkipClean = $false
    $NukeQdrant = $true
    $NoInteractive = $true
}
if ($Full) {
    $SkipClean = $false
    $NukeQdrant = $true
    $RecreateVenv = $true
    $PullImages = $true
    $InstallIndic = $true
    $NoInteractive = $true
}

function Check-Port-Availability {
    param (
        [int]$Port,
        [string]$ServiceName
    )

    $process = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($process) {
        $procId = $process.OwningProcess
        try {
            $procName = (Get-Process -Id $procId -ErrorAction SilentlyContinue).ProcessName
        } catch {
            $procName = "Unknown"
        }
        
        Write-Host "Warning: Port $Port ($ServiceName) is currently in use by process '$procName' (PID: $procId)" -ForegroundColor Yellow
        $choice = Read-Host "Do you want to kill this process to free the port? (Y/N)"
        if ($choice -eq 'Y' -or $choice -eq 'y') {
            try {
                Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
                Write-Host "Process $procId terminated. Port $Port is free." -ForegroundColor Green
            } catch {
                Write-Host "Failed to terminate process: $_" -ForegroundColor Red
            }
        } else {
            Write-Host "Skipping kill. Note: Application may fail to start." -ForegroundColor DarkYellow
        }
        Write-Host ""
    }
}

Write-Host ">>> AI-Media-Indexer Full System Startup" -ForegroundColor Cyan
Write-Host ""

# Show active flags
if ($Nuclear) { Write-Host "   Mode: NUCLEAR (wipe all data)" -ForegroundColor Red }
if ($Distributed) { Write-Host "   Mode: Distributed (Redis + Celery)" -ForegroundColor Magenta }
if ($InstallIndic) { Write-Host "   Mode: Indic ASR (Tamil/Hindi)" -ForegroundColor Cyan }
if ($Full) { Write-Host "   Mode: FULL SETUP (All features)" -ForegroundColor DarkRed }

# Check critical ports before starting
Check-Port-Availability -Port 8000 -ServiceName "Backend API"
Check-Port-Availability -Port 3000 -ServiceName "Frontend UI"
Check-Port-Availability -Port 6333 -ServiceName "Qdrant Vector DB"
Check-Port-Availability -Port 6379 -ServiceName "Redis"

# Change to project root
Set-Location $ProjectRoot

# Interactive menu (unless -NoInteractive or any flags are passed)
$anyFlagsSet = $SkipOllama -or $SkipDocker -or $SkipClean -or $RecreateVenv -or $NukeQdrant -or $PullImages -or $Integrated -or $Distributed -or $InstallIndic -or $Quick -or $Fresh -or $Nuclear -or $Full

if (-not $NoInteractive -and -not $anyFlagsSet) {
    Write-Host "============================================================" -ForegroundColor Gray
    Write-Host "  STARTUP OPTIONS" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Gray
    Write-Host ""
    
    # Detect current system state for smart recommendations
    $hasVenv = Test-Path (Join-Path $ProjectRoot ".venv")
    # Check for both embedded and docker data folders
    $hasQdrant = (Test-Path (Join-Path $ProjectRoot "qdrant_data")) -or (Test-Path (Join-Path $ProjectRoot "qdrant_data_embedded"))
    $hasCache = Test-Path (Join-Path $ProjectRoot ".cache")
    $cacheSize = 0
    if ($hasCache) {
        $cacheSize = [math]::Round((Get-ChildItem (Join-Path $ProjectRoot ".cache") -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum / 1MB, 1)
    }
    
    Write-Host "  Current system state:" -ForegroundColor Yellow
    Write-Host "    - Virtual env: $(if ($hasVenv) { 'exists' } else { 'NOT FOUND' })" -ForegroundColor Gray
    Write-Host "    - Qdrant data: $(if ($hasQdrant) { 'exists' } else { 'fresh' })" -ForegroundColor Gray
    Write-Host "    - Cache size:  $($cacheSize) MB" -ForegroundColor Gray
    Write-Host ""
    
    Write-Host "  Choose startup mode:" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  [1] Quick Start (RECOMMENDED)" -ForegroundColor Green
    Write-Host "      - Keeps caches and data intact" -ForegroundColor Gray
    Write-Host "      - Fastest startup time" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  [2] Fresh Start" -ForegroundColor Yellow
    Write-Host "      - Clears all caches" -ForegroundColor Gray
    Write-Host "      - Keeps Qdrant data (your indexed videos)" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  [3] NUCLEAR RESET" -ForegroundColor Red
    Write-Host "      - Wipes caches AND Qdrant data" -ForegroundColor Gray
    Write-Host "      - You'll need to re-ingest all videos" -ForegroundColor Gray
    Write-Host "      - Local processing (no Celery)" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  [4] Distributed Start (Redis + Celery)" -ForegroundColor Magenta
    Write-Host "      - Starts Redis, Qdrant, Celery Worker, Backend, Frontend" -ForegroundColor Gray
    Write-Host "      - Best for processing many videos in parallel" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  [5] Dev Mode (Recreate Venv)" -ForegroundColor Blue
    Write-Host "      - Recreate virtual environment" -ForegroundColor Gray
    Write-Host "      - Pull latest Docker images" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  [6] NUCLEAR + Dev Mode" -ForegroundColor DarkRed
    Write-Host "      - Wipe ALL data + recreate venv + pull images" -ForegroundColor Gray
    Write-Host "      - Complete fresh start from scratch" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  [7] NUCLEAR + Distributed" -ForegroundColor Magenta
    Write-Host "      - Complete Data Wipe + Celery workers" -ForegroundColor Gray
    Write-Host "      - Best for restarting a large batch job from zero" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  [8] Install Indic ASR (Tamil/Hindi)" -ForegroundColor Cyan
    Write-Host "      - Installs NeMo toolkit for SOTA Indic transcription" -ForegroundColor Gray
    Write-Host "      - Quick start after installation" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  [9] NUCLEAR + Indic ASR" -ForegroundColor DarkCyan
    Write-Host "      - Wipe all data + Install NeMo toolkit" -ForegroundColor Gray
    Write-Host "      - Fresh start with Indic language support" -ForegroundColor Gray
    Write-Host ""
    
    $choice = Read-Host "Enter choice [1-9] or press Enter for Quick Start"
    
    switch ($choice) {
        "1" { 
            $SkipClean = $true
            Write-Host "`n  >> Quick Start selected" -ForegroundColor Green
        }
        "2" { 
            $SkipClean = $false
            Write-Host "`n  >> Fresh Start selected (clearing caches)" -ForegroundColor Yellow
        }
        "3" { 
            $SkipClean = $false
            $NukeQdrant = $true
            Write-Host "`n  >> NUCLEAR RESET selected (wiping all data, local processing)" -ForegroundColor Red
        }
        "4" { 
            $SkipClean = $true
            $Distributed = $true
            Write-Host "`n  >> Distributed Mode selected (Redis + Celery)" -ForegroundColor Magenta
        }
        "5" { 
            $RecreateVenv = $true
            $PullImages = $true
            Write-Host "`n  >> Dev Mode selected (recreating venv, pulling images)" -ForegroundColor Blue
        }
        "6" {
            $SkipClean = $false
            $NukeQdrant = $true
            $RecreateVenv = $true
            $PullImages = $true
            Write-Host "`n  >> NUCLEAR + Dev Mode selected (wiping everything + fresh venv)" -ForegroundColor DarkRed
        }
        "7" {
            $SkipClean = $false
            $NukeQdrant = $true
            $Distributed = $true
            Write-Host "`n  >> NUCLEAR + Distributed selected (Wipe + Celery)" -ForegroundColor Magenta
        }
        "8" {
            $SkipClean = $true
            $InstallIndic = $true
            Write-Host "`n  >> Install Indic ASR selected (NeMo for Tamil/Hindi)" -ForegroundColor Cyan
        }
        "9" {
            $SkipClean = $false
            $NukeQdrant = $true
            $InstallIndic = $true
            Write-Host "`n  >> NUCLEAR + Indic ASR selected (Wipe + NeMo)" -ForegroundColor DarkCyan
        }
        "" { 
            $SkipClean = $true
            Write-Host "`n  >> Quick Start (default)" -ForegroundColor Green
        }
        default { 
            $SkipClean = $true
            Write-Host "`n  >> Invalid choice, using Quick Start" -ForegroundColor Yellow
        }
    }
    Write-Host ""
}

Write-Host "[1/8] Working directory: $ProjectRoot" -ForegroundColor Yellow

# Handle virtual environment
if ($RecreateVenv) {
    Write-Host ""
    Write-Host "[2/8] Recreating virtual environment..." -ForegroundColor Yellow
    
    $venvPath = Join-Path $ProjectRoot ".venv"
    if (Test-Path $venvPath) {
        Write-Host "  Removing existing .venv..." -ForegroundColor Gray
        Remove-Item -Path $venvPath -Recurse -Force
    }
    
    Write-Host "  Creating new venv with uv..." -ForegroundColor Gray
    uv venv
    
    Write-Host "  Syncing dependencies with uv..." -ForegroundColor Gray
    uv sync
    
    Write-Host "  Virtual environment recreated!" -ForegroundColor Green
} else {
    # Check if venv exists, if not create it
    $venvPath = Join-Path $ProjectRoot ".venv"
    if (-not (Test-Path $venvPath)) {
        Write-Host ""
        Write-Host "[2/8] Creating virtual environment (not found)..." -ForegroundColor Yellow
        uv venv
        uv sync
        Write-Host "  Virtual environment created and synced!" -ForegroundColor Green
    } else {
        Write-Host "[2/8] Virtual environment exists" -ForegroundColor Gray
    }
}

# Install Indic ASR (NeMo) if requested
if ($InstallIndic) {
    Write-Host ""
    Write-Host "[2.5/8] Installing Indic ASR (NeMo toolkit)..." -ForegroundColor Cyan
    Write-Host "  This includes NeMo + dependencies for Tamil/Hindi transcription" -ForegroundColor Gray
    Write-Host "  Note: Using Whisper large-v3-turbo as primary (best quality)" -ForegroundColor Gray
    
    try {
        uv sync --extra indic
        Write-Host "  Indic ASR dependencies installed!" -ForegroundColor Green
        Write-Host "  Transcription uses: Whisper large-v3-turbo (Tamil, Hindi, Telugu, Malayalam)" -ForegroundColor Gray
    } catch {
        Write-Host "  WARNING: Indic ASR installation failed: $_" -ForegroundColor Yellow
        Write-Host "  Whisper fallback will still work for transcription" -ForegroundColor Yellow
    }
}

# Clean caches (project only, not .venv)
if (-not $SkipClean) {
    Write-Host ""
    Write-Host "[3/8] Cleaning caches..." -ForegroundColor Yellow
    
    $cacheDirs = @(
        ".cache",
        ".face_cache", 
        ".pytest_cache",
        "__pycache__"
    )
    
    foreach ($dir in $cacheDirs) {
        $path = Join-Path $ProjectRoot $dir
        if (Test-Path $path) {
            Write-Host "  Removing: $dir" -ForegroundColor Gray
            Remove-Item -Path $path -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
    
    # Clean log files (keep directory)
    $logFiles = Get-ChildItem -Path (Join-Path $ProjectRoot "logs") -Filter "*.log" -ErrorAction SilentlyContinue
    if ($logFiles) {
        Write-Host "  Removing: log files" -ForegroundColor Gray
        $logFiles | Remove-Item -Force -ErrorAction SilentlyContinue
    }
    
    # Clean pycache recursively in project subdirectories (EXCLUDE .venv)
    Write-Host "  Removing: __pycache__ (recursively)" -ForegroundColor Gray
    
    $excludePaths = @(".venv", "node_modules", ".git")
    Get-ChildItem -Path $ProjectRoot -Directory -Recurse -Filter "__pycache__" -ErrorAction SilentlyContinue | 
        Where-Object { 
            $path = $_.FullName
            -not ($excludePaths | Where-Object { $path -like "*\$_\*" })
        } | 
        ForEach-Object {
            Remove-Item -Path $_.FullName -Recurse -Force -ErrorAction SilentlyContinue
        }
    
    Write-Host "  Cache cleanup complete!" -ForegroundColor Green
} else {
    Write-Host "[3/8] Skipping cache cleanup (--SkipClean)" -ForegroundColor Gray
}

# Configure OTEL auth for Langfuse (if needed)
$envFile = Join-Path $ProjectRoot ".env"
if (Test-Path $envFile) {
    $envContent = Get-Content $envFile -Raw
    
    # Check if Langfuse is enabled (docker or cloud)
    $langfuseBackend = [regex]::Match($envContent, '^LANGFUSE_BACKEND\s*=\s*(\w+)', [System.Text.RegularExpressions.RegexOptions]::Multiline)
    
    if ($langfuseBackend.Success -and $langfuseBackend.Groups[1].Value -in @("docker", "cloud")) {
        Write-Host ""
        Write-Host "[3.5/8] Checking Langfuse OTEL configuration..." -ForegroundColor Yellow
        
        # Extract keys
        $publicKeyMatch = [regex]::Match($envContent, 'LANGFUSE_PUBLIC_KEY\s*=\s*"?([^"\r\n]+)"?')
        $secretKeyMatch = [regex]::Match($envContent, 'LANGFUSE_SECRET_KEY\s*=\s*"?([^"\r\n]+)"?')
        $otelHeaderMatch = [regex]::Match($envContent, 'OTEL_EXPORTER_OTLP_HEADERS\s*=\s*([^\r\n]+)')
        
        if ($publicKeyMatch.Success -and $secretKeyMatch.Success) {
            $publicKey = $publicKeyMatch.Groups[1].Value.Trim().Trim('"')
            $secretKey = $secretKeyMatch.Groups[1].Value.Trim().Trim('"')
            
            # Generate expected base64 auth (use space, not %20)
            $authString = "$publicKey`:$secretKey"
            $expectedBase64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($authString))
            $expectedHeader = "Authorization=Basic $expectedBase64"
            
            # Check if current header matches
            $currentHeader = if ($otelHeaderMatch.Success) { $otelHeaderMatch.Groups[1].Value.Trim() } else { "" }
            
            if ($currentHeader -ne $expectedHeader) {
                Write-Host "  Updating OTEL auth header with Langfuse credentials..." -ForegroundColor Gray
                
                # Update or add the header in .env
                $lines = Get-Content $envFile
                $newLines = @()
                $found = $false
                
                foreach ($line in $lines) {
                    if ($line -match '^OTEL_EXPORTER_OTLP_HEADERS\s*=') {
                        $newLines += "OTEL_EXPORTER_OTLP_HEADERS=$expectedHeader"
                        $found = $true
                    } else {
                        $newLines += $line
                    }
                }
                
                if (-not $found) {
                    # Add after OTEL_EXPORTER_OTLP_ENDPOINT
                    $insertLines = @()
                    $inserted = $false
                    foreach ($line in $newLines) {
                        $insertLines += $line
                        if (-not $inserted -and $line -match 'OTEL_EXPORTER_OTLP_ENDPOINT') {
                            $insertLines += "OTEL_EXPORTER_OTLP_HEADERS=$expectedHeader"
                            $inserted = $true
                        }
                    }
                    if (-not $inserted) {
                        $insertLines += "OTEL_EXPORTER_OTLP_HEADERS=$expectedHeader"
                    }
                    $newLines = $insertLines
                }
                
                $newLines | Set-Content $envFile -Encoding UTF8
                Write-Host "  OTEL auth configured!" -ForegroundColor Green
            } else {
                Write-Host "  OTEL auth already configured" -ForegroundColor Green
            }
        } else {
            Write-Host "  Langfuse keys not found in .env - OTEL tracing may not work" -ForegroundColor Yellow
            Write-Host "  Add LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY from Langfuse Settings" -ForegroundColor Gray
        }
    }
}

# Nuke Qdrant data
# Nuke Qdrant data
if ($NukeQdrant) {
    Write-Host ""
    Write-Host "[4/8] Performing Complete Data Reset..." -ForegroundColor Red
    
    # 1. Kill any existing backend/frontend processes to release locks
    Write-Host "  Terminating existing AI-Media-Indexer processes..." -ForegroundColor Gray
    Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like "*api/server.py*" } | Stop-Process -Force -ErrorAction SilentlyContinue
    Get-Process node -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like "*vite*" } | Stop-Process -Force -ErrorAction SilentlyContinue

    # 2. Stop Docker and remove named volumes (Critical for MinIO/ClickHouse)
    Write-Host "  Stopping Docker containers and removing volumes..." -ForegroundColor Gray
    try {
        # Using & (call operator) to ensure flags are passed correctly
        & docker-compose down -v --remove-orphans 2>&1 | Out-Null
        
        # NUCLEAR OPTION: Force remove specific containers if they are stuck
        # This addresses the issue where Qdrant would remain running despite 'down'
        $containers = @(
            "media_agent_qdrant", 
            "media_agent_postgres", 
            "media_agent_minio", 
            "media_agent_redis", 
            "media_agent_clickhouse", 
            "media_agent_langfuse", 
            "media_agent_langfuse_worker", 
            "media_agent_createbuckets"
        )
        foreach ($c in $containers) {
            docker rm -f $c 2>&1 | Out-Null
        }
        
        Write-Host "  Docker services stopped and volumes removed." -ForegroundColor Green
    } catch {
        Write-Host "  Warning: Docker down failed: $_" -ForegroundColor Yellow
    }

    # 3. Wipe local bind-mount directories
    $dataDirs = @("qdrant_data", "qdrant_data_embedded", "thumbnails", "logs", ".cache", "jobs.db", "identity.db")
    
    # Optional: Wipe Postgres (Langfuse)
    Write-Host ""
    $wipeLangfuse = Read-Host "  [?] Also wipe Langfuse/Postgres data (Resets API keys)? (y/N)"
    if ($wipeLangfuse -match "^[yY]") {
        $dataDirs += "postgres_data"
        Write-Host "  >> Including Postgres/Langfuse in wipe list." -ForegroundColor Yellow
    } else {
        Write-Host "  >> Preserving Postgres/Langfuse data (Keys kept safe)." -ForegroundColor Green
    }

    foreach ($dir in $dataDirs) {
        $path = Join-Path $ProjectRoot $dir
        if (Test-Path $path) {
            Write-Host "  Removing: $dir" -ForegroundColor Gray
            # Use rd /s /q for aggressive Windows deletion
            cmd /c "rd /s /q `"$path`"" 2>&1 | Out-Null
            
            # Verification
            if (Test-Path $path) {
                # Try PowerShell native as backup
                Remove-Item -Path $path -Recurse -Force -ErrorAction SilentlyContinue
                if (Test-Path $path) {
                    Write-Host "  ERROR: Could not delete $dir. Folder is locked." -ForegroundColor Red
                    Write-Host "  Please close all applications (Explorer, VS Code) open in that folder." -ForegroundColor Yellow
                } else {
                    Write-Host "  $dir deleted successfully (backup method)." -ForegroundColor Green
                }
            } else {
                Write-Host "  $dir deleted successfully." -ForegroundColor Green
            }
        }
    }
    
    Write-Host "  Data reset complete!" -ForegroundColor Green
} else {
    Write-Host "[4/8] Keeping Qdrant data (use -NukeQdrant to delete)" -ForegroundColor Gray
}

# Check and Start Docker
if (-not $SkipDocker) {
    Write-Host ""
    Write-Host "[5/8] Checking Docker status..." -ForegroundColor Yellow
    
    $dockerRunning = $false
    try {
        $null = docker info 2>&1
        if ($LASTEXITCODE -eq 0) {
            $dockerRunning = $true
        }
    } catch {
        $dockerRunning = $false
    }
    
    if (-not $dockerRunning) {
        Write-Host "  Docker Daemon is not running. Attempting to start Docker Desktop..." -ForegroundColor Yellow
        $dockerPath = "C:\Program Files\Docker\Docker\Docker Desktop.exe"
        if (Test-Path $dockerPath) {
            Start-Process -FilePath $dockerPath -WindowStyle Hidden
            Write-Host "  Docker Desktop launched. Waiting for daemon to start (this may take a minute)..." -ForegroundColor Gray
            
            # Wait for Docker to be ready
            $retries = 60
            while ($retries -gt 0) {
                Write-Host "." -NoNewline -ForegroundColor Gray
                Start-Sleep -Seconds 2
                try {
                    $null = docker info 2>&1
                    if ($LASTEXITCODE -eq 0) {
                        Write-Host ""
                        Write-Host "  Docker Daemon is ready!" -ForegroundColor Green
                        $dockerRunning = $true
                        break
                    }
                } catch {}
                $retries--
            }
            if (-not $dockerRunning) {
                Write-Host ""
                Write-Host "  ERROR: Docker failed to start within timeout. Please start Docker Desktop manually." -ForegroundColor Red
                exit 1
            }
        } else {
            Write-Host "  ERROR: Docker Desktop not found at default location ($dockerPath). Please start it manually." -ForegroundColor Red
            exit 1
        }
    } else {
        Write-Host "  Docker Daemon is already running" -ForegroundColor Green
    }

    # Detect valid Docker Compose command
    $dockerComposeCmd = "docker-compose"
    if (-not (Get-Command "docker-compose" -ErrorAction SilentlyContinue)) {
        if (docker compose version 2>&1 | Select-String "Docker Compose") {
             $dockerComposeCmd = "docker compose"
             Write-Host "  Using 'docker compose'..." -ForegroundColor Gray
        } else {
             Write-Host "  WARNING: docker-compose not found. Docker operations might fail." -ForegroundColor Red
        }
    }

    Write-Host ""
    Write-Host "[5.5/8] Stopping Docker containers..." -ForegroundColor Yellow
    
    # Stop all containers
    # Use Invoke-Expression to handle the command string with arguments correctly
    Invoke-Expression "$dockerComposeCmd down --remove-orphans" 2>&1 | Out-Null
    
    Write-Host "  Docker containers stopped and orphans removed" -ForegroundColor Green
    
    # Pull latest images if requested
    if ($PullImages) {
        Write-Host ""
        Write-Host "[6/8] Pulling latest Docker images..." -ForegroundColor Yellow
        Invoke-Expression "$dockerComposeCmd pull"
        Write-Host "  Docker images updated!" -ForegroundColor Green
    } else {
        # Check if Langfuse images exist, pull only if missing
        $langfuseImage = docker images "langfuse/langfuse" --format "{{.Repository}}" 2>$null
        if (-not $langfuseImage) {
            Write-Host "[6/8] Langfuse images not found, pulling..." -ForegroundColor Yellow
            Invoke-Expression "$dockerComposeCmd pull langfuse langfuse-worker"
            Write-Host "  Langfuse images pulled!" -ForegroundColor Green
        } else {
            Write-Host "[6/8] Langfuse images already present (use -PullImages to update)" -ForegroundColor Gray
        }
    }
    
    Write-Host ""
    Write-Host "[7/8] Starting Docker services..." -ForegroundColor Yellow
    
    # Setup Distributed ENV - Persist to .env file so backend reads it correctly
    if ($Distributed) {
         $Env:ENABLE_DISTRIBUTED_INGESTION = "True"
         Write-Host "  Distributed Ingestion Enabled: Starting Redis + Celery..." -ForegroundColor Magenta
         
         # Persist to .env file
         $envFilePath = Join-Path $ProjectRoot ".env"
         if (Test-Path $envFilePath) {
             $envLines = Get-Content $envFilePath
             $found = $false
             $newLines = @()
             foreach ($line in $envLines) {
                 if ($line -match '^ENABLE_DISTRIBUTED_INGESTION\s*=') {
                     $newLines += "ENABLE_DISTRIBUTED_INGESTION=True"
                     $found = $true
                 } else {
                     $newLines += $line
                 }
             }
             if (-not $found) {
                 $newLines += ""
                 $newLines += "# Distributed Ingestion (Celery + Redis)"
                 $newLines += "ENABLE_DISTRIBUTED_INGESTION=True"
             }
             $newLines | Set-Content $envFilePath -Encoding UTF8
             Write-Host "  Updated .env: ENABLE_DISTRIBUTED_INGESTION=True" -ForegroundColor Green
         }
    } else {
         $Env:ENABLE_DISTRIBUTED_INGESTION = "False"
         
         # Ensure .env has it set to False (or remove) to avoid confusion
         $envFilePath = Join-Path $ProjectRoot ".env"
         if (Test-Path $envFilePath) {
             $envLines = Get-Content $envFilePath
             $newLines = @()
             foreach ($line in $envLines) {
                 if ($line -match '^ENABLE_DISTRIBUTED_INGESTION\s*=') {
                     $newLines += "ENABLE_DISTRIBUTED_INGESTION=False"
                 } else {
                     $newLines += $line
                 }
             }
             $newLines | Set-Content $envFilePath -Encoding UTF8
         }
    }

    # Start containers
    Write-Host "  Starting containers with $dockerComposeCmd..." -ForegroundColor Gray
    
    if ($Distributed) {
         # Ensure Redis is up
         if ($dockerComposeCmd -eq "docker-compose") {
             Invoke-Expression "$dockerComposeCmd up -d qdrant redis"
         } else {
             Invoke-Expression "$dockerComposeCmd up -d --wait qdrant redis"
         }
    } else {
         # Standard Start - Just Qdrant for now? Or everything?
         # Original script just did 'up' for whatever is in compose?
         # No, wait. Original script didn't select services, it did 'up'. 
         # But usually we only need Qdrant minimal.
         # Let's start Qdrant by default.
         if ($dockerComposeCmd -eq "docker-compose") {
             Invoke-Expression "$dockerComposeCmd up -d qdrant"
         } else {
             Invoke-Expression "$dockerComposeCmd up -d --wait qdrant"
         }
    }

    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ERROR: Docker start failed" -ForegroundColor Red
    } else {
        Write-Host "  Docker services started and healthy!" -ForegroundColor Green
    }
} else {
    Write-Host "[5/8] Skipping Docker cleanup (--SkipDocker)" -ForegroundColor Gray
    Write-Host "[6/8] Skipping Docker pull (--SkipDocker)" -ForegroundColor Gray
    Write-Host "[7/8] Skipping Docker startup (--SkipDocker)" -ForegroundColor Gray
}

# Start Ollama
if (-not $SkipOllama) {
    Write-Host ""
    Write-Host "[8/8] Checking Ollama status..." -ForegroundColor Yellow
    
    # Check if Ollama is running by checking the port (more reliable than process name)
    $ollamaRunning = $false
    try {
        $tcpConn = Test-NetConnection -ComputerName localhost -Port 11434 -InformationLevel Quiet
        if ($tcpConn) {
            $ollamaRunning = $true
        }
    } catch {
        $ollamaRunning = $false
    }

    if ($ollamaRunning) {
        Write-Host "  Ollama is already running (Port 11434)" -ForegroundColor Green
    } else {
        Write-Host "  Ollama not detected on port 11434. Starting..." -ForegroundColor Yellow
        
        # Try to start Ollama
        $ollamaPath = Get-Command "ollama" -ErrorAction SilentlyContinue
        if ($ollamaPath) {
            Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
            Write-Host "  Ollama launched in background. Waiting for API..." -ForegroundColor Gray
            
            # Wait for Ollama to be ready
            $retries = 10
            while ($retries -gt 0) {
                 Start-Sleep -Seconds 1
                 if (Test-NetConnection -ComputerName localhost -Port 11434 -InformationLevel Quiet) {
                     Write-Host "  Ollama API is ready!" -ForegroundColor Green
                     break
                 }
                 $retries--
            }
        } else {
            Write-Host "  WARNING: Ollama not found in PATH. Please start it manually." -ForegroundColor Yellow
        }
    }
    
    # Auto-pull Ollama model if not present
    Write-Host "  Checking Ollama vision model..." -ForegroundColor Gray
    
    # Read model from .env or use default
    $ollamaModel = "moondream"  # Default lightweight model
    $envFile = Join-Path $ProjectRoot ".env"
    if (Test-Path $envFile) {
        $envContent = Get-Content $envFile -Raw
        $modelMatch = [regex]::Match($envContent, 'OLLAMA_MODEL\s*=\s*"?([^"\r\n]+)"?')
        if ($modelMatch.Success) {
            $ollamaModel = $modelMatch.Groups[1].Value.Trim()
        }
    }
    
    # Check if model exists, pull if not
    $modelList = ollama list 2>&1
    if ($modelList -notmatch [regex]::Escape($ollamaModel)) {
        Write-Host "  Model '$ollamaModel' not found. Pulling..." -ForegroundColor Yellow
        ollama pull $ollamaModel
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  Model '$ollamaModel' pulled successfully!" -ForegroundColor Green
        } else {
            Write-Host "  WARNING: Failed to pull model. Vision features may not work." -ForegroundColor Red
        }
    } else {
        Write-Host "  Model '$ollamaModel' is available." -ForegroundColor Green
    }
} else {
    Write-Host "[8/8] Skipping Ollama startup (--SkipOllama)" -ForegroundColor Gray
}

# Start Backend and Frontend
Write-Host ""
Write-Host "[9/9] Starting Backend and Frontend..." -ForegroundColor Yellow

$frontendDir = Join-Path $ProjectRoot "web"

# Detect PowerShell executable (pwsh for Core, powershell for Windows PowerShell)
$psExe = if (Get-Command pwsh -ErrorAction SilentlyContinue) { "pwsh" } else { "powershell" }

# Start Celery Worker if Distributed
$workerProcess = $null
if ($Distributed) {
    Write-Host "  Starting Celery Worker (Distributed)..." -ForegroundColor Magenta
    
    if ($Integrated) {
        # Background worker
        $workerProcess = Start-Process $psExe -ArgumentList "-NoProfile", "-Command", "cd '$ProjectRoot'; uv run celery -A core.ingestion.celery_app worker --loglevel=info -P threads" -PassThru -WindowStyle Hidden
        Write-Host "  Celery Worker started (PID: $($workerProcess.Id))" -ForegroundColor Green
    } else {
        # New terminal worker
        $workerCmd = "cd '$ProjectRoot'; Write-Host 'Celery Worker' -ForegroundColor Magenta; Write-Host '=============' -ForegroundColor Magenta; uv run celery -A core.ingestion.celery_app worker --loglevel=info -P threads"
        Start-Process $psExe -ArgumentList "-NoExit", "-Command", $workerCmd
        Write-Host "  Celery Worker started in new terminal" -ForegroundColor Green
    }
}

if ($Integrated) {
    # Launch in integrated terminal using background process
    Write-Host "  Mode: Integrated terminal (-Integrated)" -ForegroundColor Gray
    
    # Start Frontend as background process
    if (-not (Test-Path "$frontendDir\node_modules")) { 
        Write-Host "  Installing frontend dependencies..." -ForegroundColor Gray
        & npm.cmd install 
    }
    $frontendProcess = Start-Process $psExe -ArgumentList "-NoProfile", "-Command", "cd '$frontendDir'; npm.cmd run dev" -PassThru -WindowStyle Hidden
    Write-Host "  Frontend started (PID: $($frontendProcess.Id))" -ForegroundColor Green
    
    Write-Host ""
    Write-Host ">>> System startup complete!" -ForegroundColor Green
    Write-Host ""
    Write-Host "  Backend:  http://localhost:8000" -ForegroundColor White
    Write-Host "  Frontend: http://localhost:5173" -ForegroundColor White
    Write-Host "  Langfuse: http://localhost:3300" -ForegroundColor White
    Write-Host "  Qdrant:   http://localhost:6333" -ForegroundColor White
    Write-Host ""
    Write-Host "  Frontend running in background (PID: $($frontendProcess.Id))." -ForegroundColor Gray
    Write-Host "  Press Ctrl+C to stop the backend. Frontend will continue running." -ForegroundColor Gray
    Write-Host "  To stop frontend: Stop-Process -Id $($frontendProcess.Id)" -ForegroundColor Gray
    Write-Host ""
    
    # Start Backend in foreground (blocking)
    Set-Location $ProjectRoot
    try {
        uv run uvicorn api.server:app --host 0.0.0.0 --port 8000
    } finally {
        # Cleanup when backend stops
        Write-Host ""
        Write-Host "Backend stopped. Stopping frontend and workers..." -ForegroundColor Yellow
        Stop-Process -Id $frontendProcess.Id -Force -ErrorAction SilentlyContinue
        if ($workerProcess) {
             Stop-Process -Id $workerProcess.Id -Force -ErrorAction SilentlyContinue
        }
        Write-Host "Cleanup complete." -ForegroundColor Green
    }
} else {
    # DEFAULT: Launch in separate terminal windows
    Write-Host "  Mode: Separate terminal windows (default)" -ForegroundColor Gray
    
    # Start Backend in new terminal
    $backendCmd = "cd '$ProjectRoot'; Write-Host 'AI-Media-Indexer Backend' -ForegroundColor Cyan; Write-Host '========================' -ForegroundColor Cyan; uv run uvicorn api.server:app --port 8000"
    Start-Process $psExe -ArgumentList "-NoExit", "-Command", $backendCmd
    Write-Host "  Backend started in new terminal (port 8000)" -ForegroundColor Green
    
    # Start Frontend in new terminal
    if (-not (Test-Path "$frontendDir\node_modules")) { 
        Write-Host "  Installing frontend dependencies..." -ForegroundColor Gray
        Set-Location $frontendDir
        & npm.cmd install
        Set-Location $ProjectRoot
    }
    $frontendCmd = "cd '$frontendDir'; Write-Host 'AI-Media-Indexer Frontend' -ForegroundColor Cyan; Write-Host '=========================' -ForegroundColor Cyan; npm.cmd run dev"
    Start-Process $psExe -ArgumentList "-NoExit", "-Command", $frontendCmd
    Write-Host "  Frontend started in new terminal (port 5173)" -ForegroundColor Green
    
    Write-Host ""
    Write-Host ">>> System startup complete!" -ForegroundColor Green
    Write-Host ""
    Write-Host "  Backend:  http://localhost:8000" -ForegroundColor White
    Write-Host "  Frontend: http://localhost:5173" -ForegroundColor White
    Write-Host "  Langfuse: http://localhost:3300" -ForegroundColor White
    Write-Host "  Qdrant:   http://localhost:6333" -ForegroundColor White
    Write-Host ""
    Write-Host "  Two new terminal windows have been opened." -ForegroundColor Gray
    Write-Host "  Close them manually or use: Get-Process powershell | Stop-Process" -ForegroundColor Gray
    Write-Host ""
}
