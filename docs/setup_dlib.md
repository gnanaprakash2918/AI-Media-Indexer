# Setup Guide: Dlib with CUDA

This guide covers installing **CUDA-enabled `dlib`** into a `uv`-managed virtual environment.

## Linux (Ubuntu/Debian)

### Prerequisites

Ensure you have the CUDA toolkit installed. Check with:

```bash
nvcc --version
```

Install build dependencies:

```bash
sudo apt-get update
sudo apt-get install build-essential cmake
```

### Installation

If you are using `uv`:

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dlib (should auto-detect CUDA if nvcc is on path)
uv pip install dlib
```

---

## Windows (Python 3.12)

This detailed section walks through installing **CUDA-enabled `dlib`** on Windows using `uv`.

### 0. Prerequisites

#### 0.1. CUDA Toolkit 11.8 (or compatible)
Install from NVIDIA. Verify:

<!-- carousel -->
#### PowerShell
```powershell
nvcc --version
```
<!-- slide -->
#### CMD
```cmd
nvcc --version
```
<!-- slide -->
#### Bash
```bash
nvcc --version
```
<!-- /carousel -->

#### 0.2. Visual Studio C++ Build Tools
Install **VS 2019** or **VS 2022** with "**Desktop development with C++**".

### 1. Project Setup (with `uv`)

Assume project root is `D:\AI-Media-Indexer`.

<!-- carousel -->
#### PowerShell
```powershell
cd D:\AI-Media-Indexer
& .\.venv\Scripts\Activate.ps1
```
<!-- slide -->
#### CMD
```cmd
cd /d D:\AI-Media-Indexer
call .venv\Scripts\activate.bat
```
<!-- slide -->
#### Bash (Git Bash / WSL)
```bash
cd /mnt/d/AI-Media-Indexer
source .venv/Scripts/activate
```
<!-- /carousel -->

### 2. Clean Existing Dlib

Remove `face-recognition` or any existing `dlib` references from `pyproject.toml` and `uv.lock`.

<!-- carousel -->
#### PowerShell
```powershell
del uv.lock
uv lock
```
<!-- slide -->
#### CMD
```cmd
del uv.lock
uv lock
```
<!-- slide -->
#### Bash
```bash
rm uv.lock
uv lock
```
<!-- /carousel -->

### 3. Clone `dlib` Source

<!-- carousel -->
#### PowerShell
```powershell
git clone https://github.com/davisking/dlib.git C:\dlib
cd C:\dlib
git fetch --all --tags
git checkout v19.24.6
```
<!-- slide -->
#### CMD
```cmd
git clone https://github.com/davisking/dlib.git C:\dlib
cd C:\dlib
git fetch --all --tags
git checkout v19.24.6
```
<!-- slide -->
#### Bash
```bash
git clone https://github.com/davisking/dlib.git /c/dlib
cd /c/dlib
git fetch --all --tags
git checkout v19.24.6
```
<!-- /carousel -->

### 4. Build and Install

**IMPORTANT**: For Windows, use the **Developer Command Prompt** for VS 2019/2022. The following commands are specific to that environment (CMD-like).

#### Developer Command Prompt (Windows Only)

1. Open **Developer Command Prompt**.
2. Run the following:

```cmd
cd /d D:\AI-Media-Indexer
call .venv\Scripts\activate.bat

set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8
set PATH=%CUDA_PATH%\bin;%CUDA_PATH%\libnvvp;%PATH%

set DLIB_USE_CUDA=1
set DLIB_USE_CUBLAS=1
set CMAKE_ARGS=-DDLIB_USE_CUDA=1 -DDLIB_USE_CUBLAS=1 -DCUDA_TOOLKIT_ROOT_DIR="%CUDA_PATH%" -T v143

cd /d C:\dlib
rmdir /S /Q build 2>nul

uv pip install . --no-build-isolation --no-binary dlib
```

### 5. Verify Installation

Create `t.py`:

```python
import dlib
print("version:", dlib.__version__)
print("CUDA enabled:", dlib.DLIB_USE_CUDA)
print("Devices:", dlib.cuda.get_num_devices() if dlib.DLIB_USE_CUDA else "No CUDA")
```

Run it:

<!-- carousel -->
#### PowerShell
```powershell
python t.py
```
<!-- slide -->
#### CMD
```cmd
python t.py
```
<!-- slide -->
#### Bash
```bash
python t.py
```
<!-- /carousel -->
