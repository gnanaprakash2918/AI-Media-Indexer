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
    Run backend in foreground with frontend as background process (instead of separate windows)
#>

param(
    [switch]$SkipOllama,
    [switch]$SkipDocker,
    [switch]$SkipClean,
    [switch]$RecreateVenv,
    [switch]$NukeQdrant,
    [switch]$PullImages,
    [switch]$Integrated
)

$ErrorActionPreference = "Continue"
$ProjectRoot = $PSScriptRoot

Write-Host ">>> AI-Media-Indexer Full System Startup" -ForegroundColor Cyan
Write-Host ""

# Change to project root
Set-Location $ProjectRoot
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
    $excludePaths = @(".venv", "node_modules", ".git")
    Get-ChildItem -Path $ProjectRoot -Directory -Recurse -Filter "__pycache__" -ErrorAction SilentlyContinue | 
        Where-Object { 
            $path = $_.FullName
            -not ($excludePaths | Where-Object { $path -like "*\$_\*" })
        } | 
        ForEach-Object {
            Write-Host "  Removing: $($_.FullName -replace [regex]::Escape($ProjectRoot), '.')" -ForegroundColor Gray
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
if ($NukeQdrant) {
    Write-Host ""
    Write-Host "[4/8] Nuking Qdrant data..." -ForegroundColor Yellow
    
    $qdrantDirs = @("qdrant_data", "qdrant_data_embedded")
    foreach ($dir in $qdrantDirs) {
        $path = Join-Path $ProjectRoot $dir
        if (Test-Path $path) {
            Write-Host "  Removing: $dir" -ForegroundColor Gray
            Remove-Item -Path $path -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
    
    Write-Host "  Qdrant data nuked!" -ForegroundColor Green
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

    Write-Host ""
    Write-Host "[5.5/8] Stopping Docker containers..." -ForegroundColor Yellow
    
    # Stop all containers (redirect stderr to null to avoid PowerShell treating progress as error)
    $null = docker compose down --remove-orphans 2>&1
    
    Write-Host "  Docker containers stopped and orphans removed" -ForegroundColor Green
    
    # Pull latest images if requested
    if ($PullImages) {
        Write-Host ""
        Write-Host "[6/8] Pulling latest Docker images..." -ForegroundColor Yellow
        docker compose pull
        Write-Host "  Docker images updated!" -ForegroundColor Green
    } else {
        # Check if Langfuse images exist, pull only if missing
        $langfuseImage = docker images "langfuse/langfuse" --format "{{.Repository}}" 2>$null
        if (-not $langfuseImage) {
            Write-Host "[6/8] Langfuse images not found, pulling..." -ForegroundColor Yellow
            docker compose pull langfuse langfuse-worker
            Write-Host "  Langfuse images pulled!" -ForegroundColor Green
        } else {
            Write-Host "[6/8] Langfuse images already present (use -PullImages to update)" -ForegroundColor Gray
        }
    }
    
    Write-Host ""
    Write-Host "[7/8] Starting Docker services..." -ForegroundColor Yellow
    
    # Start containers
    $dockerOutput = docker compose up -d 2>&1
    $dockerOutput | ForEach-Object { Write-Host "  $_" -ForegroundColor Gray }
    
    # Wait for services to be healthy
    Write-Host "  Waiting for services to be healthy..." -ForegroundColor Gray
    Start-Sleep -Seconds 5
    
    docker compose ps --format "table {{.Name}}\t{{.Status}}"
    Write-Host "  Docker services started!" -ForegroundColor Green
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
} else {
    Write-Host "[8/8] Skipping Ollama startup (--SkipOllama)" -ForegroundColor Gray
}

# Start Backend and Frontend
Write-Host ""
Write-Host "[9/9] Starting Backend and Frontend..." -ForegroundColor Yellow

$frontendDir = Join-Path $ProjectRoot "web"

# Detect PowerShell executable (pwsh for Core, powershell for Windows PowerShell)
$psExe = if (Get-Command pwsh -ErrorAction SilentlyContinue) { "pwsh" } else { "powershell" }

if ($Integrated) {
    # Launch in integrated terminal using background process
    Write-Host "  Mode: Integrated terminal (-Integrated)" -ForegroundColor Gray
    
    # Start Frontend as background process
    $frontendProcess = Start-Process $psExe -ArgumentList "-NoProfile", "-Command", "cd '$frontendDir'; npm run dev" -PassThru -WindowStyle Hidden
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
        uv run uvicorn api.server:create_app --factory --host 0.0.0.0 --port 8000
    } finally {
        # Cleanup when backend stops
        Write-Host ""
        Write-Host "Backend stopped. Stopping frontend..." -ForegroundColor Yellow
        Stop-Process -Id $frontendProcess.Id -Force -ErrorAction SilentlyContinue
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
    $frontendCmd = "cd '$frontendDir'; Write-Host 'AI-Media-Indexer Frontend' -ForegroundColor Cyan; Write-Host '=========================' -ForegroundColor Cyan; npm run dev"
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
