# Sprints for the Project

## Sprint 1

### Task 1.1 - Project Initialization (COMPLETED)

- ✅ Project scaffold and basic tooling.
  - Created the repository structure with `core/`, `tests/`, and top-level `main.py`.
  - Added `pyproject.toml` with project metadata and dev tooling.
  - Runtime dependency: `pydantic`.
  - Dev tools: `ruff` (linting) and `black` (optional formatting).
  - Ruff configured to enforce Google-style docstrings and an 80-character line length.

### Task 1.2 - Media Prober (COMPLETED)

- ✅ Implemented `MediaProber` in `core/processing/prober.py`.

  - Provides `probe(file_path: str) -> dict` which runs `ffprobe` to produce
    JSON-formatted metadata (streams + format).
  - Uses `subprocess` directly for transparent, reproducible FFmpeg calls.
  - Errors from `ffprobe` are captured and re-raised as `MediaProbeError` to
    avoid crashing callers.
  - Added Google-style docstrings to `MediaProber`, `MediaProbeError`, and the
    `probe` method, documenting Args, Returns, and Raises.

  Topics covered:

  - `subprocess.run` vs `subprocess.Popen` and error handling patterns.
  - `capture_output=True` and `text=True` for capturing `ffprobe` output.
  - FFmpeg/ffprobe flags: `-v quiet`, `-print_format json`, `-show_format`,
    `-show_streams`.

### Task 1.3 - Library Scanner (COMPLETED)

- ✅ Implemented `LibraryScanner` in `core/ingestion/scanner.py`.

  - Implements `scan(self, root_path: str) -> Generator[Path, None, None]`
    to recursively discover media files without loading results into memory.
  - Uses `pathlib.Path` for path operations and yields `Path` objects.
  - Skips excluded directories and handles `PermissionError` gracefully.
  - Uses `python-magic` where available to determine mime types and file
    thresholds for media detection.
  - Google-style docstrings and linting applied.

  Notes:

  - The scanner yields `(Path, mime_type)` in some helper modes to allow
    downstream code to make quick decisions without re-probing the file.
  - Excluded directories are configurable to avoid scanning large system
    folders (e.g., program files, node_modules, etc.).

## Sprint 2

### Task 2.1 - Audio Transcription (COMPLETED)

- ✅ Add a transcriber file to `core/processing/transcriber.py`.
  - Implemented `AudioTranscriber` class using Hugging Face's `faster-whisper` model.
  - Supports GPU acceleration via CUDA.
- ✅ CUDA Setup:
  - Used `nvidia-smi` to check CUDA version (13.0 in this case).
  - Downloaded PyTorch with CUDA 12.1 support: `uv add torch --index-url https://download.pytorch.org/whl/cu121`
- ✅ Environment Configuration:
  - Set Hugging Face cache path: `$env:HF_HOME = "D:\huggingface_cache"`
  - Enabled Unicode support: `chcp 65001` in PowerShell for proper Tamil/Unicode display.
- ✅ Code Formatting:
  - Applied Black formatter with 88-character line length: `black . --line-length 88`
  - Applied Ruff linting and fixes: `ruff check . --fix --unsafe-fixes`

### Task 2.2 - Frame Extraction (COMPLETED)

- ✅ Add frame extraction to `core/processing/extractor.py`.
  - Implemented `FrameExtractor` class using FFmpeg to extract video frames.
  - Uses `subprocess` to invoke FFmpeg for transparent, non-abstracted control.
- ✅ Features:
  - Extracts frames at specified intervals or as keyframes.
  - Handles frame caching using `FrameCache` class for temporary storage.
  - Non-blocking process execution with proper error handling.
- ✅ Integration:
  - Proper error handling and path validation.
  - JPEG frame output format for efficient storage and processing.

### Task 2.3 - Vision Processing (COMPLETED)

- ✅ Add vision processing to `core/processing/vision.py`.
  - Implemented `VisionProcessor` class for analyzing extracted video frames.
  - Supports multiple LLM backends for vision analysis.
- ✅ Features:
  - Processes video frames using state-of-the-art vision models.
  - Integrates with LLM factory for flexible model selection.
  - Google-style docstrings and proper error handling.

### Task 2.4 - LLM Factory Pattern (COMPLETED)

- ✅ Implement LLM abstraction layer in `llm/` package.
  - Created `llm/interface.py` with abstract `LLMInterface` base class.
  - Implements prompt loading with in-memory caching.
  - Provides JSON parsing utilities for structured responses.
- ✅ LLM Implementations:
  - `llm/gemini.py`: Google Gemini API adapter.
  - `llm/ollama.py`: Local Ollama model adapter.
  - Both implement the `LLMInterface` contract.
- ✅ Factory Pattern:
  - Created `llm/factory.py` with `LLMFactory` for provider instantiation.
  - Dynamic provider selection based on configuration.
- ✅ Configuration Management:
  - `config.py`: Centralized settings with `Settings` class and `LLMProvider` enum.
  - Environment-backed configuration for API keys, model names, and URLs.

### Task 2.5 - Schemas and Data Validation (COMPLETED)

- ✅ Created `core/schemas.py` with Pydantic models for data validation.
  - Defined schemas for media metadata, transcription results, and analysis outputs.
  - Used Pydantic for runtime validation and IDE type hints.

### Task 2.6 - Code Quality and Linting (COMPLETED)

- ✅ Applied Google Python Style Guide throughout:
  - All files conform to Google-style docstrings.
  - 80-character line length enforced via Ruff configuration.
  - Proper import organization and unused code removal.
- ✅ Configuration in `pyproject.toml`:
  - Ruff configured with `pydocstyle` rule set for Google-style docstrings.
  - Line length set to 80 characters.
  - Black formatter set to 88 characters for additional formatting.

## Sprint 3

### Task 3.1 - Database and Storage Layer

- [ ] Implement SQLite database schema for media metadata storage.
- [ ] Create `core/storage/db.py` for database operations.
- [ ] Implement CRUD operations for media entries, transcriptions, and analysis results.

### Task 3.2 - Main Processing Pipeline

- [ ] Create unified `main.py` that orchestrates:
  - File scanning via `LibraryScanner`
  - Frame extraction via `FrameExtractor`
  - Transcription via `AudioTranscriber`
  - Vision analysis via `VisionProcessor`
  - Results storage via database layer.
- [ ] Implement batch processing with progress tracking.
- [ ] Add error recovery and logging.

### Task 3.3 - API and Web Interface

- [ ] Implement FastAPI endpoints for:
  - Media library browsing
  - Search and filtering
  - Analysis result retrieval
  - Indexing status monitoring
- [ ] Create web UI for media index visualization.

### Task 3.4 - Testing and Documentation

- [ ] Write comprehensive unit tests for each module.
- [ ] Add integration tests for the full pipeline.
- [ ] Create API documentation.
- [ ] Write user guide and deployment instructions.

- `winget install --id Microsoft.VisualStudio.2022.BuildTools -e`
- `winget install --id Kitware.CMake -e`
- `winget install --id NinjaBuild.Ninja -e`
- `winget install Ninja-build.Ninja`
- `https://github.com/z-mahmud22/Dlib_Windows_Python3.x`
- `uv add "dlib @ https://github.com/z-mahmud22/Dlib_Windows_Python3.x/raw/main/dlib-19.24.99-cp312-cp312-win_amd64.whl"`
