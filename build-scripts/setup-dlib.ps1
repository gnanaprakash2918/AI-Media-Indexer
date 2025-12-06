
$ProjectRoot = (Get-Item .).FullName
$VenvDir = ".\.venv"
$DlibSourceDir = "C:\dlib"
$DlibVersion = "v19.24.6"
$CudaPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
$BuildScript = ".\run-dlib-cuda-build.cmd"
$TestFile = "t.py"
$ForceAsciiEncoding = "Ascii"


function Quick-Cleanup {
    Write-Host "## Quick Cleanup: Removing temporary files..."
    if (Test-Path $BuildScript) { Remove-Item -Path $BuildScript -Force | Out-Null }
    if (Test-Path $TestFile) { Remove-Item -Path $TestFile -Force | Out-Null }
    Write-Host "Done."
}

function Full-Cleanup-Dlib {
    Write-Host "## Performing Full Cleanup (Removing installed package and source clone)..."

    Quick-Cleanup # Clean temporary build files

    # Uninstall dlib from venv
    try {
        if (Test-Path ".\.venv\Scripts\activate.ps1") {
            & ".\.venv\Scripts\activate.ps1" 2> $null | Out-Null
            uv pip uninstall dlib -y 2> $null | Out-Null
            deactivate 2> $null
        }
    } catch {
        Write-Host "Could not uninstall dlib cleanly. Proceeding..."
    }
    
    # Remove dlib source clone
    if (Test-Path $DlibSourceDir) {
        Write-Host "Removing dlib source directory: $DlibSourceDir"
        Remove-Item -Path $DlibSourceDir -Recurse -Force
    }

    Write-Host "✅ Full Cleanup finished." -ForegroundColor Green
}

function Setup-DlibBuild {
    Quick-Cleanup

    Write-Host "## 1. Project Setup and UV Clean..."
    cd $ProjectRoot

    if (-not (Test-Path $VenvDir)) {
        uv venv $VenvDir
    }

    if (Test-Path "uv.lock") { Remove-Item -Path "uv.lock" -Force | Out-Null }
    uv lock | Out-Null
    if (Select-String -Path "uv.lock" -Pattern "dlib" -ErrorAction SilentlyContinue) {
        Write-Host " WARNING: 'dlib' still found in uv.lock. Check pyproject.toml!" -ForegroundColor Red
    }

    Write-Host "## 2. Cloning dlib Source..."
    if (Test-Path $DlibSourceDir) {
        cd $DlibSourceDir
        git checkout $DlibVersion | Out-Null
    } else {
        git clone https://github.com/davisking/dlib.git $DlibSourceDir | Out-Null
        cd $DlibSourceDir
        git checkout $DlibVersion | Out-Null
    }

    Write-Host "## 3. Creating verification script t.py..."
    cd $ProjectRoot
    $testScript = @"
import dlib
print("version:", dlib.__version__)
print("CUDA enabled:", dlib.DLIB_USE_CUDA)
print("Devices:", dlib.cuda.get_num_devices() if dlib.DLIB_USE_CUDA else "No CUDA")
"@
    $testScript | Set-Content -LiteralPath $TestFile -Encoding UTF8 -Force

    Write-Host "## 4. Creating CMD build script: $BuildScript"
    $cmdScript = @"
@echo off
rem This script MUST be run inside the Developer Command Prompt for VS 2019 or 2022.

echo --- Activating VENV and Setting Env Vars ---
cd /d "$ProjectRoot"
.\.venv\Scripts\activate.bat

echo Verifying nvcc version...
nvcc --version

set CUDA_PATH=$CudaPath
set PATH=%CUDA_PATH%\bin;%CUDA_PATH%\libnvvp;%PATH%

set DLIB_USE_CUDA=1
set DLIB_USE_CUBLAS=1

rem -T v143 for VS 2022, change to -T v142 for VS 2019
set CMAKE_ARGS=-DDLIB_USE_CUDA=1 -DDLIB_USE_CUBLAS=1 -DCUDA_TOOLKIT_ROOT_DIR="%CUDA_PATH%" -T v143

echo --- Building and Installing dlib from source ---
cd /d "$DlibSourceDir"
rmdir /S /Q build 2>nul
uv pip install . --no-build-isolation --no-binary dlib

echo --- Verification ---
cd /d "$ProjectRoot"
python $TestFile

echo.
echo --- Build Process Complete (Check output for 'CUDA enabled: True') ---
pause
"@
    $cmdScript | Set-Content -LiteralPath $BuildScript -Encoding $ForceAsciiEncoding -Force
}


param(
    [switch]$Cleanup
)

if ($Cleanup) {
    Full-Cleanup-Dlib
} else {
    Setup-DlibBuild
    Write-Host ""
    Write-Host "=================================================================================="
    Write-Host "Setup Complete. NEXT STEP REQUIRED (IMPORTANT):" -ForegroundColor Green
    Write-Host "The build environment must be initialized by Visual Studio."
    Write-Host ""
    Write-Host "1. Open the **Developer Command Prompt for VS 2022** (or 2019)."
    Write-Host "2. Navigate to your project root: cd /d $ProjectRoot"
    Write-Host "3. Run the generated build script:"
    Write-Host "   $BuildScript" -ForegroundColor Yellow
    Write-Host "=================================================================================="
}