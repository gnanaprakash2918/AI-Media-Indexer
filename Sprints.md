# Sprints for the Project

## Repository Information

- **Repository**: [gnanaprakash2918/AI-Media-Indexer](https://github.com/gnanaprakash2918/AI-Media-Indexer)
- **Main Branches**: `main`, `sprint-1`, `sprint-2`, `sprint-2-bug`
- **Tech Stack**:
  - Orchestrator: Google ADK (Agent Development Kit)
  - Ingestion Engine: Python with FFmpeg
  - LLM: Ollama (local) + Google Gemini API
  - Memory: Qdrant with Docker for Vector Store
  - Backend: FastAPI

---

## Sprint 1 (Branch: `sprint-1`)

### Commit History

#### Commit `28eb302` - Initial Project Setup (2025-11-23)

**Author**: gnanaprakash2918  
**Message**: Initial commit

**Changes**:

- Created initial repository structure
- Added `.gitignore` with basic Python exclusions
- Created `README.md` with project overview
- Established project foundation

**Files Added**:

- `.gitignore`
- `README.md`

---

#### Commit `acfa0d6` - Sprint 1 Folder Structure (2025-11-23)

**Author**: gnanaprakash2918  
**Message**: Added Sprint 1, Folder Structure

**Changes**:

- Established modular project structure
- Created core processing modules
- Set up ingestion and storage layers
- Added initial Sprints.md documentation

**Files Added/Modified**:

- `core/` directory structure
- `core/ingestion/scanner.py`
- `core/processing/prober.py`
- `core/processing/extractor.py`
- `core/processing/transcriber.py`
- `core/schemas.py`
- `Sprints.md`
- `pyproject.toml`

---

#### Commit `7d35ca6` - Prober Refactoring (2025-11-24)

**Author**: Gnana Prakash M  
**Message**: Replaces If-else checks with simple loop

**Changes**:

- Refactored `MediaProber` to use loop-based binary checking
- Improved code readability and maintainability
- Simplified FFmpeg binary validation logic

**Files Modified**:

- `core/processing/prober.py`

---

#### Commit `a5d3e40` - FFprobe Binary Validation (2025-11-24)

**Author**: Gnana Prakash M  
**Message**: Added a decorator to perform ffprobe binary checks

**Changes**:

- Implemented decorator pattern for FFprobe binary validation
- Added `@require_ffprobe` decorator
- Enhanced error handling for missing FFmpeg binaries
- Improved robustness of media probing operations

**Files Modified**:

- `core/processing/prober.py` (31 insertions, 4 deletions)

---

#### Commit `289fa39` - Prober Testing (2025-11-24)

**Author**: Gnana Prakash M  
**Message**: Added Test Main block for prober

**Changes**:

- Added test/demo code to `prober.py`
- Implemented `if __name__ == "__main__"` block
- Enabled standalone testing of MediaProber functionality

**Files Modified**:

- `core/processing/prober.py`

---

#### Commit `f8e38be` - Sprint 1 Completion (2025-11-24)

**Author**: Gnana Prakash M  
**Message**: SPRINT 1 COMPLETE

**Changes**:

- Finalized Sprint 1 deliverables
- Updated Sprints.md with completed tasks
- Consolidated all Sprint 1 features
- Prepared for Sprint 2 development

**Files Modified**:

- `Sprints.md` (71 insertions)
- `.idea/AI-Media-Indexer.iml`
- Multiple configuration files

**Total Sprint 1 Changes**: 677 insertions, 279 deletions across 8 files

---

### Sprint 1 Completed Tasks

#### Task 1.1 - Project Initialization ✅

- ✅ Project scaffold and basic tooling
  - Created repository structure: `core/`, `tests/`, `main.py`
  - Added `pyproject.toml` with project metadata
  - Runtime dependency: `pydantic`
  - Dev tools: `ruff` (linting), `black` (formatting)
  - Ruff configured for Google-style docstrings (80-char line length)

#### Task 1.2 - Media Prober ✅

- ✅ Implemented `MediaProber` in `core/processing/prober.py`
  - Provides `probe(file_path: str) -> dict` using `ffprobe`
  - JSON-formatted metadata extraction (streams + format)
  - Uses `subprocess` for transparent FFmpeg calls
  - Custom `MediaProbeError` exception handling
  - Google-style docstrings with Args, Returns, Raises
  - Decorator-based FFprobe binary validation

**Technical Details**:

- `subprocess.run` with `capture_output=True` and `text=True`
- FFmpeg flags: `-v quiet`, `-print_format json`, `-show_format`, `-show_streams`
- Robust error handling and binary availability checks

#### Task 1.3 - Library Scanner ✅

- ✅ Implemented `LibraryScanner` in `core/ingestion/scanner.py`
  - Generator-based file discovery: `scan(root_path) -> Generator[Path, None, None]`
  - Uses `pathlib.Path` for cross-platform compatibility
  - Configurable directory exclusions (system folders, node_modules, etc.)
  - Graceful `PermissionError` handling
  - Optional `python-magic` integration for MIME type detection
  - Memory-efficient streaming approach

---

## Sprint 2 (Branch: `sprint-2`)

### Commit History

#### Commit `fa49693` - Sprint 2 Implementation (2025-11-26)

**Author**: Gnana Prakash M  
**Message**: SPRINT 2 COMPLETE

**Changes**:

- Implemented LLM factory pattern with multiple providers
- Added configuration management system
- Enhanced vision processing capabilities
- Updated transcriber with improved settings
- Comprehensive Sprint 2 documentation

**Files Added**:

- `config.py` - Centralized configuration management
- `llm/__init__.py` - LLM package initialization
- `llm/interface.py` - Abstract LLM interface
- `llm/factory.py` - LLM factory pattern implementation
- `llm/gemini.py` - Google Gemini API adapter
- `llm/ollama.py` - Ollama local model adapter

**Files Modified**:

- `.gitignore` (3 insertions)
- `Sprints.md` (156 insertions, 85 deletions)
- `core/ingestion/scanner.py`
- `core/processing/extractor.py`
- `core/processing/transcriber.py`
- `core/processing/vision.py`
- `core/schemas.py`
- `pyproject.toml`
- `uv.lock`

**Total Sprint 2 Changes**: 1,126 insertions, 156 deletions across 15 files

---

#### Commit `4822777` - Transcriber Class Addition (2025-11-24)

**Author**: Gnana Prakash M  
**Message**: Added Transcriber Class

**Changes**:

- Implemented `AudioTranscriber` class in `core/processing/transcriber.py`
- Added audio transcription capabilities using faster-whisper
- Integrated GPU acceleration support
- Enhanced project dependencies for audio processing

**Files Added**:

- `core/processing/transcriber.py`

**Files Modified**:

- `pyproject.toml` - Added transcription dependencies
- `uv.lock` - Updated dependency lock file

---

#### Commit `4cd0da2` - WSL Configuration (2025-12-03)

**Author**: Gnana Prakash M  
**Message**: Add wslconfig to control ram usage

**Changes**:

- Added WSL2 configuration file to optimize memory usage
- Configured memory limit to 12GB for WSL2
- Added 8GB swap space for heavy PyTorch compilation
- Enabled automatic memory reclamation to Windows

**Files Added**:

- `.wslconfig` - WSL2 resource management configuration

**Configuration Details**:

- Memory limit: 12GB
- Swap space: 8GB
- Auto memory reclaim: `drop_cache` mode

---

#### Commit `cd45046` - Qdrant Data Exclusion (2025-12-04)

**Author**: Gnana Prakash M  
**Message**: Added Qdrant data to gitignore

**Changes**:

- Updated `.gitignore` to exclude Qdrant vector database data
- Improved repository cleanliness by ignoring generated data files
- Prevented accidental commits of large vector store files

**Files Modified**:

- `.gitignore` - Added `qdrant_data/` exclusion

---

#### Commit `336bac6` - Qdrant Data Fix (2025-12-04)

**Author**: Gnana Prakash M  
**Message**: Fix the qdrant_data/

**Changes**:

- Fixed `.gitignore` pattern for Qdrant data directory
- Added output file for debugging/logging purposes
- Ensured proper exclusion of vector database files

**Files Modified**:

- `.gitignore` - Refined Qdrant data exclusion pattern

**Files Added**:

- `output.txt` - Debug/logging output file

**Total Additional Sprint 2 Changes**: 576+ insertions across multiple commits

---

### Sprint 2 Completed Tasks

#### Task 2.1 - Audio Transcription ✅

- ✅ Implemented `AudioTranscriber` in `core/processing/transcriber.py`
  - Hugging Face `faster-whisper` integration
  - GPU acceleration via CUDA support
  - Configurable model selection via environment variables

**CUDA Setup**:

- Verified CUDA 13.0 with `nvidia-smi`
- Installed PyTorch with CUDA 12.1:
  ```bash
  uv add torch --index-url https://download.pytorch.org/whl/cu121
  ```

**Environment Configuration**:

- Hugging Face cache: `$env:HF_HOME = "D:\huggingface_cache"`
- Unicode support: `chcp 65001` (Tamil/multilingual text)

**Code Quality**:

- Black formatter: `black . --line-length 88`
- Ruff linting: `ruff check . --fix --unsafe-fixes`

**References**:

- [PyTorch CUDA Downloads](https://download.pytorch.org/whl/cu121)
- [Faster Whisper](https://github.com/guillaumekln/faster-whisper)

---

#### Task 2.2 - Frame Extraction ✅

- ✅ Implemented `FrameExtractor` in `core/processing/extractor.py`
  - FFmpeg-based video frame extraction
  - Direct `subprocess` invocation (no abstraction layers)
  - Configurable extraction intervals and keyframe detection
  - `FrameCache` class for temporary storage management
  - Non-blocking process execution
  - JPEG output format for efficiency

---

#### Task 2.3 - Vision Processing ✅

- ✅ Implemented `VisionProcessor` in `core/processing/vision.py`
  - Multi-modal vision analysis of video frames
  - LLM factory integration for flexible backend selection
  - Support for both Gemini and Ollama vision models
  - Google-style docstrings and error handling

---

#### Task 2.4 - LLM Factory Pattern ✅

- ✅ Created `llm/` package with abstraction layer

**`llm/interface.py`**:

- Abstract `LLMInterface` base class
- Prompt loading with in-memory caching
- JSON parsing utilities for structured LLM responses
- Common error handling patterns

**`llm/gemini.py`**:

- Google Gemini API adapter
- Implements `LLMInterface` contract
- Supports text and vision models
- API key management via environment variables

**`llm/ollama.py`**:

- Local Ollama model adapter
- Implements `LLMInterface` contract
- Configurable base URL and model selection
- Supports vision models (e.g., `llava:7b`)

**`llm/factory.py`**:

- `LLMFactory` class for provider instantiation
- Dynamic provider selection based on configuration
- Singleton pattern for efficient resource usage

**`config.py`**:

- Centralized `Settings` class
- `LLMProvider` enum (GEMINI, OLLAMA)
- Environment-backed configuration:
  - `GEMINI_API_KEY` / `GOOGLE_API_KEY`
  - `GEMINI_MODEL` (default: `gemini-1.5-flash`)
  - `OLLAMA_BASE_URL` (default: `http://localhost:11434`)
  - `OLLAMA_MODEL` (default: `llava:7b`)
  - `WHISPER_MODEL`, `WHISPER_DEVICE`, `WHISPER_COMPUTE_TYPE`
  - `PROMPT_DIR` for external prompt templates

# Running Ingestion pipeline

Gotcha — Windows, and you only care about **Qdrant as a service**, no app container. Perfect, that’s actually simpler. 🙌

You **do not need any Dockerfile** if you’re just running Qdrant — the official image already _is_ the Dockerfile. Your `docker-compose.yml` is enough.

I’ll still show you:

- How to **stop + clean everything** on Docker.
- How to **run only Qdrant** via your `docker-compose.yml`.
- (Optional) a minimal Dockerfile for Qdrant if you _really_ want one, but it’s redundant.

---

## 1️⃣ Stop everything + clean Docker (Windows)

### 📌 In PowerShell

Run these one by one:

```powershell
# 1. Stop all running containers
docker ps -q | ForEach-Object { docker stop $_ }

# 2. Remove all containers (stopped + running)
docker ps -aq | ForEach-Object { docker rm $_ }

# 3. Prune EVERYTHING unused: containers, images, networks, volumes, build cache
#    WARNING: This will delete unused images and volumes for ALL projects.
docker system prune -af --volumes
```

If you want to **wipe Qdrant data** as well (fresh DB):

```powershell
Remove-Item -Recurse -Force .\qdrant_data -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path .\qdrant_data | Out-Null
```

> Run those from your **project root** (where `docker-compose.yml` lives).

---

## 2️⃣ Run only Qdrant with docker-compose

From the same folder:

```powershell
# If your Docker Desktop uses the new syntax:
docker compose up -d

# If that errors, use the old one:
# docker-compose up -d
```

Check it’s running:

```powershell
docker ps
```

You should see `media_agent_qdrant` up with ports `6333` and `6334`.

**References**:

- [Google Gemini API](https://ai.google.dev/)
- [Ollama](https://ollama.ai/)
- [Ollama Python Library](https://github.com/ollama/ollama-python)

---

#### Task 2.5 - Schemas and Data Validation ✅

- ✅ Enhanced `core/schemas.py` with Pydantic models
  - Media metadata schemas
  - Transcription result models
  - Vision analysis output schemas
  - Runtime validation and IDE type hints

---

#### Task 2.6 - Code Quality and Linting ✅

- ✅ Applied Google Python Style Guide project-wide
  - Google-style docstrings in all modules
  - 80-character line length (Ruff)
  - 88-character line length (Black)
  - Proper import organization
  - Unused code removal

**Configuration** (`pyproject.toml`):

- Ruff: `pydocstyle` rules, 80-char limit
- Black: 88-char limit
- Type checking preparation

---

## Sprint 2 Bug Fixes (Branch: `sprint-2-bug`)

### Commit History

#### Commit `c4ddbd3` - .gitignore Improvements (2025-11-30)

**Author**: Gnana Prakash M
**Message**: Using .gitignore to get rid of .idea files Improved .gitignore to get rid of .idea files

**Changes**:

- Enhanced `.gitignore` to exclude IDE-specific files
- Added comprehensive JetBrains `.idea/` exclusions
- Improved repository cleanliness
- Updated Sprints.md with bug fix documentation

**Files Modified**:

- `Sprints.md` (5 insertions, 1 deletion)
- `config.py` (110 insertions, 64 deletions)
- `core/processing/transcriber.py`
- `pyproject.toml`
- `uv.lock`

**Total Bug Fix Changes**: Focused on configuration refinement and IDE file exclusions

---

## Sprint 3 (Planned)

### Task 3.1 - Database and Storage Layer

- [ ] Implement SQLite database schema for media metadata
- [ ] Create `core/storage/db.py` with database operations
- [ ] CRUD operations for media entries, transcriptions, analysis results
- [ ] Database migration system

### Task 3.2 - Main Processing Pipeline

- [ ] Unified `main.py` orchestration:
  - File scanning via `LibraryScanner`
  - Frame extraction via `FrameExtractor`
  - Transcription via `AudioTranscriber`
  - Vision analysis via `VisionProcessor`
  - Results storage via database layer
- [ ] Batch processing with progress tracking
- [ ] Error recovery and comprehensive logging
- [ ] Async/concurrent processing optimization

### Task 3.3 - API and Web Interface

- [ ] FastAPI endpoints:
  - Media library browsing
  - Search and filtering
  - Analysis result retrieval
  - Indexing status monitoring
- [ ] Web UI for media index visualization
- [ ] RESTful API documentation (OpenAPI/Swagger)

### Task 3.4 - Testing and Documentation

- [ ] Comprehensive unit tests for each module
- [ ] Integration tests for full pipeline
- [ ] API documentation (auto-generated)
- [ ] User guide and deployment instructions
- [ ] Performance benchmarking

---

## Development Tools and Dependencies

### Build Tools (Windows)

```powershell
winget install --id Microsoft.VisualStudio.2022.BuildTools -e
winget install --id Kitware.CMake -e
winget install --id NinjaBuild.Ninja -e
winget install Ninja-build.Ninja
```

### Special Dependencies

**dlib (Windows Python 3.12)**:

- Repository: [z-mahmud22/Dlib_Windows_Python3.x](https://github.com/z-mahmud22/Dlib_Windows_Python3.x)
- Installation:
  ```bash
  uv add "dlib @ https://github.com/z-mahmud22/Dlib_Windows_Python3.x/raw/main/dlib-19.24.99-cp312-cp312-win_amd64.whl"
  ```

### Key Python Packages

- `pydantic` - Data validation
- `torch` - Deep learning (CUDA 12.1)
- `faster-whisper` - Audio transcription
- `google-generativeai` - Gemini API
- `ollama` - Local LLM integration
- `python-magic` - MIME type detection
- `ruff` - Linting
- `black` - Code formatting

---

## Useful Commands

### Environment Setup

```powershell
# Set Hugging Face cache directory
$env:HF_HOME = "D:\huggingface_cache"

# Enable Unicode support in PowerShell
chcp 65001
```

### Code Quality

```bash
# Format code with Black
black . --line-length 88

# Lint and auto-fix with Ruff
ruff check . --fix --unsafe-fixes

# Check types (future)
mypy .
```

### Git Workflow

```bash
# View all branches
git branch -a

# View commit history
git log --all --graph --oneline --decorate

# Compare branches
git diff sprint-1..sprint-2 --stat
```

---

## References and Documentation

- **FFmpeg**: [https://ffmpeg.org/](https://ffmpeg.org/)
- **PyTorch CUDA**: [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
- **Faster Whisper**: [https://github.com/guillaumekln/faster-whisper](https://github.com/guillaumekln/faster-whisper)
- **Google Gemini API**: [https://ai.google.dev/](https://ai.google.dev/)
- **Ollama**: [https://ollama.ai/](https://ollama.ai/)
- **Pydantic**: [https://docs.pydantic.dev/](https://docs.pydantic.dev/)
- **FastAPI**: [https://fastapi.tiangolo.com/](https://fastapi.tiangolo.com/)
- **Qdrant**: [https://qdrant.tech/](https://qdrant.tech/)
- **Google ADK**: [https://developers.google.com/adk](https://developers.google.com/adk)

- `docker compose -f docker-compose.qdrant.yaml up -d`

https://developer.nvidia.com/cudnn-downloads?ta rget_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local
https://developer.nvidia.com/cuda-12-4-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local

uv pip uninstall dlib
set DLIB_USE_CUDA=1
uv run pip install dlib --no-binary :all:

$env:CUDA_PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
$env:PATH="$env:CUDA_PATH\bin;$env:CUDA_PATH\libnvvp;$env:PATH"

# Tell dlib explicitly to use CUDA + cuBLAS

$env:DLIB_USE_CUDA="1"
$env:DLIB_USE_CUBLAS="1"
