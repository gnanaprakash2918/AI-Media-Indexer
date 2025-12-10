````markdown
# Installing CUDA-enabled `dlib` with `uv` on Windows (Python 3.12)

This guide walks through installing **CUDA-enabled `dlib`** into a `uv`-managed virtualenv on Windows using:

- Python 3.12
- CUDA 11.8
- Visual Studio (C++ build tools)
- `uv` for package management
- `dlib` **v19.24.6** from source

It also includes all the gotchas that break CUDA builds and how to avoid them.

---

## 0. Prerequisites

### 0.1. CUDA Toolkit 11.8

Install **CUDA 11.8** from NVIDIA.

Verify:

```powershell
nvcc --version
```
````

Output should contain:

```text
Cuda compilation tools, release 11.8, V11.8.x
```

### 0.2. Visual Studio C++ Build Tools

Install either:

- **Visual Studio 2019 Community** with:

  - Workload: **Desktop development with C++**
  - MSVC v142 (14.29)

- or **Visual Studio 2022 Build Tools** with:

  - Workload: **Desktop development with C++**
  - MSVC v143 (14.3x)

You must be able to open:

- **Developer Command Prompt for VS 2019** or
- **Developer Command Prompt for VS 2022**

---

## 1. Project Setup (with `uv`)

Assume project root:

```text
D:\AI-Media-Indexer
```

And a `uv` venv:

```powershell
D:
cd D:\AI-Media-Indexer
dir
.\.venv\Scripts\activate.ps1   # PowerShell
or
& D:\AI-Media-Indexer\.venv\Scripts\Activate.ps1
```

```cmd
D:
cd D:\AI-Media-Indexer
call .venv\Scripts\activate.bat
```

---

## 2. Remove Anything That Forces CPU-only `dlib`

### 2.1. Remove `face-recognition` Dependency

`face-recognition` **always** pulls in `dlib` from PyPI (CPU-only).
If you want full control over `dlib`, remove it.

In `pyproject.toml`:

```toml
[project]
dependencies = [
    # "face-recognition>=1.3.0",  # ← REMOVE THIS LINE
    ...
]
```

If you need its functionality later, you can re-implement using CUDA `dlib` directly.

---

### 2.2. Remove `dlib` as Direct Dependency or Source Override

In `pyproject.toml`, ensure there is **no** `dlib` anywhere:

- No `"dlib"` in `[project.dependencies]`
- No `dlib = { url = "..." }` in `[tool.uv.sources]`

Example of a **clean** `[tool.uv.sources]`:

```toml
[tool.uv.sources]
torch = [{ index = "pytorch" }]
torchvision = [{ index = "pytorch" }]
torchaudio = [{ index = "pytorch" }]
```

No `dlib` entry.

---

### 2.3. Clean `uv.lock` from `dlib`

From project root:

```powershell
cd D:\AI-Media-Indexer
del uv.lock
uv lock
```

Check that `dlib` is no longer in the lockfile:

```powershell
Select-String -Path uv.lock -Pattern "dlib"
```

Expected: **no output**.
If you still see something like `dlib-20.0.0.tar.gz`, it is still being pulled by a dependency; fix that before continuing.

---

## 3. Clone `dlib` Source (v19.24.6)

Use a separate directory, e.g. `C:\dlib`:

```powershell
git clone https://github.com/davisking/dlib.git C:\dlib
cd C:\dlib

git fetch --all --tags
git checkout v19.24.6

git describe --tags
# should print: v19.24.6
```

---

## 4. Build Environment: Use the Developer Command Prompt

1. Open:

   - **Developer Command Prompt for VS 2019** _or_
   - **Developer Command Prompt for VS 2022**

2. Change to your project and activate the venv:

   ```cmd
   cd /d D:\AI-Media-Indexer
   .\.venv\Scripts\activate.bat
   ```

   Prompt should show `(.venv)`.

---

## 5. Set CUDA-related Environment Variables

In the same **Developer Command Prompt** with venv active:

```cmd
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8
set PATH=%CUDA_PATH%\bin;%CUDA_PATH%\libnvvp;%PATH%

set DLIB_USE_CUDA=1
set DLIB_USE_CUBLAS=1

rem Optional: force toolset; works with VS 2022 (v143) or falls back to VS 2019 config
set CMAKE_ARGS=-DDLIB_USE_CUDA=1 -DDLIB_USE_CUBLAS=1 -DCUDA_TOOLKIT_ROOT_DIR="%CUDA_PATH%" -T v143
```

> Do **not** use PowerShell syntax (`$env:...`) inside the Developer Command Prompt; only `set`.

---

## 6. Build and Install `dlib` from Local Source with `uv`

From the same Developer Command Prompt:

```cmd
cd /d C:\dlib
rmdir /S /Q build 2>nul

uv pip install . --no-build-isolation --no-binary dlib
```

Expected output includes something like:

```text
Built dlib @ file:///C:/dlib
Installed 1 package
 + dlib==19.24.6 (from file:///C:/dlib)
```

If `uv` reports installing from **PyPI** instead of `file:///C:/dlib`, the dependency graph is still pulling `dlib` externally; fix `pyproject.toml` and `uv.lock` as in step 2.

---

## 7. Verify CUDA in `dlib`

Create a test file `t.py` under `D:\AI-Media-Indexer`:

```python
import dlib

print("version:", dlib.__version__)
print("CUDA enabled:", dlib.DLIB_USE_CUDA)
print("Devices:", dlib.cuda.get_num_devices() if dlib.DLIB_USE_CUDA else "No CUDA")
```

Run from the same venv:

```powershell
cd D:\AI-Media-Indexer
python t.py
```

Expected:

```text
version: 19.24.6
CUDA enabled: True
Devices: 1  # or more, depending on your GPUs
```

If `CUDA enabled: False`, then `dlib` built without CUDA. Common causes:

- `nvcc` not usable in the current Developer Command Prompt
- CUDA version not compatible with MSVC toolchain
- Environment variables `DLIB_USE_CUDA` / `CMAKE_ARGS` were not set in the build shell

Re-check:

```cmd
nvcc --version
cl
```

Both commands should work inside the same Developer Command Prompt.

---

## 8. Key Things to Avoid

- Do **not** install `face-recognition` if you want to control `dlib` yourself; it pulls CPU-only `dlib` from PyPI.
- Do **not** keep any `dlib` URL or version in:

  - `pyproject.toml`
  - `[tool.uv.sources]`
  - `requirements.txt`

- Do **not** run Python code (`import dlib`, `print(...)`) directly in CMD; use `python` and then run code inside the interpreter or scripts (`python t.py`).
- Do **not** mix PowerShell heredoc syntax (`<<EOF`) inside CMD.

---

## 9. Summary of Commands

```cmd
rem Developer Command Prompt for VS 2019/2022
cd /d D:\AI-Media-Indexer
.\.venv\Scripts\activate.bat

del uv.lock
uv lock

set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8
set PATH=%CUDA_PATH%\bin;%CUDA_PATH%\libnvvp;%PATH%
set DLIB_USE_CUDA=1
set DLIB_USE_CUBLAS=1
set CMAKE_ARGS=-DDLIB_USE_CUDA=1 -DDLIB_USE_CUBLAS=1 -DCUDA_TOOLKIT_ROOT_DIR="%CUDA_PATH%" -T v143

cd /d C:\dlib
git fetch --all --tags
git checkout v19.24.6
rmdir /S /Q build 2>nul

uv pip install . --no-build-isolation --no-binary dlib

cd /d D:\AI-Media-Indexer
python t.py
```
