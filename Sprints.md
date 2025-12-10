# Sprints for the Project

## Repository Information

- **Repository**: [gnanaprakash2918/AI-Media-Indexer](https://github.com/gnanaprakash2918/AI-Media-Indexer)
- **Main Branches**: `main`, `sprint-1`, `sprint-2`, `sprint-2-bug`, `sprint-3`
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

## Sprint 3 (Branch: `sprint-3`)

### Commit History

#### Commit `c874ca6` - Sprint 3 Complete (2025-12-06)

**Author**: Gnana Prakash M  
**Message**: Use dlib instead of face recognition and set use cpu as default

**Changes**:

- Replaced `face_recognition` library with direct `dlib` implementation
- Implemented custom face detection and encoding pipeline
- Added GPU/CPU fallback with MediaPipe integration
- Set CPU as default device to avoid memory issues
- Enhanced face detection with multiple backend support

**Files Modified**:

- `core/processing/identity.py` - Complete rewrite with dlib
- `config.py` - Updated device configuration
- `pyproject.toml` - Updated dependencies

---

#### Major Sprint 3 Commits (2025-11-25 to 2025-12-06)

**Database and Vector Store Implementation**:

- Implemented `VectorDB` class in `core/storage/db.py`
- Integrated Qdrant for vector storage and similarity search
- Added sentence-transformers for text embedding
- Created collections for media segments and face embeddings
- Implemented search functionality with filtering

**Ingestion Pipeline**:

- Created `IngestionPipeline` in `core/ingestion/pipeline.py`
- Orchestrated end-to-end video processing workflow
- Integrated all processing components (prober, transcriber, extractor, vision, identity)
- Added async processing for frames
- Implemented memory cleanup and resource management

**Face Detection and Identity**:

- Implemented `FaceManager` in `core/processing/identity.py`
- Added face detection using dlib with GPU/CPU/MediaPipe fallback
- Implemented 128-d face encoding computation
- Added DBSCAN clustering for identity grouping
- Automatic dlib model download and caching

**Configuration Updates**:

- Enhanced `config.py` with comprehensive settings
- Added device detection (CUDA/MPS/CPU)
- Configured Whisper model mapping for multiple languages
- Added computed fields for paths and hardware settings
- Integrated pydantic-settings for environment variable management

**Schema Updates**:

- Added `DetectedFace` model to `core/schemas.py`
- Defined face bounding box and encoding structure
- Added confidence field for future enhancements

**Main Entry Point**:

- Updated `main.py` with CLI interface
- Integrated `IngestionPipeline` for video processing
- Added path validation and async execution
- Windows-specific event loop policy handling

**Dependencies**:

- Added `qdrant-client` for vector database
- Added `sentence-transformers` for embeddings
- Added `mediapipe` for face detection fallback
- Added `scikit-learn` for DBSCAN clustering
- Added `pydantic-settings` for configuration
- Updated PyTorch to CUDA 12.4 support

---

### Sprint 3 Completed Tasks

#### Task 3.1 - Database and Storage Layer ✅

- ✅ Implemented Qdrant vector database integration in `core/storage/db.py`
  - `VectorDB` class for managing vector storage
  - Support for embedded (memory) and Docker-based Qdrant
  - Automatic collection creation for media and faces
  - SentenceTransformer integration for text embeddings
  - Local model caching in `project_root/models`

**Key Features**:

- **Media Segment Storage**:
  - `insert_media_segments()` - Index transcription segments
  - `search_media()` - Semantic search with filtering
  - Metadata: video path, timestamps, segment type
- **Face Embedding Storage**:
  - `insert_face()` - Store 128-d face encodings
  - `search_face()` - Find similar faces
  - Metadata: name, cluster ID, confidence
- **Frame Storage**:
  - `upsert_media_frame()` - Store frame embeddings
  - Metadata: video path, timestamp, action, dialogue

**Technical Details**:

- Vector dimensions: 384 (text), 128 (faces)
- Distance metric: Cosine similarity
- Collections: `media_segments`, `faces`, `media_frames`
- Automatic model download and caching

---

#### Task 3.2 - Main Processing Pipeline ✅

- ✅ Implemented `IngestionPipeline` in `core/ingestion/pipeline.py`
  - End-to-end video processing orchestration
  - Async frame processing with memory management
  - Integration of all processing components
  - Comprehensive error handling and logging

**Pipeline Steps**:

1. **Media Probing**: Extract metadata (duration, streams) via `MediaProber`
2. **Audio Transcription**: Transcribe audio to text segments via `AudioTranscriber`
3. **Segment Indexing**: Store transcription segments in Qdrant
4. **Frame Extraction**: Extract frames at fixed intervals via `FrameExtractor`
5. **Frame Processing** (async for each frame):
   - Vision analysis via `VisionAnalyzer`
   - Face detection via `FaceManager`
   - Vector embedding and Qdrant indexing
6. **Memory Cleanup**: Force garbage collection and CUDA cache clearing

**Configuration**:

- Qdrant backend: memory or docker
- Frame interval: configurable (default 5 seconds)
- Async processing for parallel frame analysis
- Automatic resource cleanup

**CLI Interface** (`main.py`):

```bash
uv run python main.py <path_to_video>
```

- Path validation and resolution
- Windows asyncio policy handling
- Single video processing

---

#### Task 3.3 - Face Detection and Identity ✅

- ✅ Implemented `FaceManager` in `core/processing/identity.py`
  - Multi-backend face detection (GPU → MediaPipe → HOG)
  - dlib-based face encoding (128-d ResNet embeddings)
  - DBSCAN clustering for identity grouping
  - Automatic model download and caching

**Face Detection Pipeline**:

1. **Load Image**: Convert to RGB numpy array via PIL
2. **Detect Faces**:
   - GPU: dlib CNN detector (if CUDA available)
   - Fallback 1: MediaPipe face detection
   - Fallback 2: dlib HOG detector
3. **Compute Encodings**: dlib ResNet model (128-d vectors)
4. **Cluster Identities**: DBSCAN clustering on encodings

**Models** (auto-downloaded to `project_root/models`):

- `shape_predictor_68_face_landmarks.dat`
- `dlib_face_recognition_resnet_model_v1.dat`
- `mmod_human_face_detector.dat` (CNN detector)

**Configuration**:

- DBSCAN epsilon: 0.5 (default)
- DBSCAN min samples: 3 (default)
- Distance metric: Euclidean
- GPU support: Optional (CPU default to avoid OOM)

**Integration**:

- Returns `DetectedFace` Pydantic models
- Stores face embeddings in Qdrant
- Supports identity clustering across videos

---

#### Task 3.4 - Configuration and Settings ✅

- ✅ Enhanced `config.py` with comprehensive application settings
  - Hardware detection (CUDA/MPS/CPU)
  - LLM provider configuration
  - Whisper model mapping for multiple languages
  - Path management with computed fields
  - Environment variable integration

**Key Settings**:

- **Paths**:

  - `project_root` - Auto-detected from .git or .env
  - `model_cache_dir` - `project_root/models`
  - `prompt_dir` - `project_root/prompts`

- **LLM Configuration**:

  - Provider: Gemini or Ollama
  - Timeout: 120s default
  - Model selection per provider

- **ASR Configuration**:

  - Language-specific Whisper models
  - Batch size and chunk length
  - Hugging Face token support

- **Hardware**:
  - Auto device detection
  - Compute type (float16/float32)
  - Device index for Pipeline

---

#### Task 3.5 - Schema Updates ✅

- ✅ Added `DetectedFace` model to `core/schemas.py`
  - Bounding box: (top, right, bottom, left) tuple
  - Encoding: 128-d face embedding as list
  - Confidence: Placeholder for future use

**Schema Definition**:

```python
class DetectedFace(BaseModel):
    box: tuple[int, int, int, int]
    encoding: list[float]
    confidence: float = 1.0
```

---

#### Task 3.6 - Dependency Management ✅

- ✅ Updated `pyproject.toml` with Sprint 3 dependencies
  - Vector database: `qdrant-client>=1.16.1`
  - Embeddings: `sentence-transformers>=5.1.2`
  - Face detection: `mediapipe>=0.10.21`
  - Clustering: `scikit-learn>=1.5.0`
  - Configuration: `pydantic-settings>=2.12.0`
  - PyTorch: CUDA 12.4 support

**Removed Dependencies**:

- `face-recognition` (replaced with direct dlib)
- Commented out `nemo-toolkit` (not used yet)

---

### Sprint 3 Technical Highlights

#### Vector Database Architecture

- **Qdrant Integration**: Embedded and Docker modes
- **Collections**:
  - `media_segments`: Text-based search (384-d)
  - `faces`: Face similarity search (128-d)
  - `media_frames`: Frame-level indexing (384-d)
- **Automatic Setup**: Collection creation on initialization
- **Local Caching**: Models stored in project directory

#### Async Processing

- Frame processing parallelized with `asyncio`
- Memory cleanup after each frame
- CUDA cache clearing to prevent OOM
- Windows event loop policy handling

#### Multi-Backend Face Detection

- **Primary**: dlib CNN (GPU)
- **Fallback 1**: MediaPipe (CPU-optimized)
- **Fallback 2**: dlib HOG (CPU)
- Automatic model download from dlib.net
- Bz2 decompression and caching

#### Configuration Management

- Pydantic-based settings with validation
- Environment variable support via `.env`
- Computed fields for dynamic paths
- Hardware auto-detection
- Language-specific model mapping

---

### Sprint 3 Future Enhancements

- [ ] API endpoints for search and retrieval
- [ ] Web UI for media library visualization
- [ ] Batch video processing
- [ ] Progress tracking and status monitoring
- [ ] Unit and integration tests
- [ ] Performance benchmarking
- [ ] Documentation and user guides

---

## Sprint 4 (Branch: `sprint-4`)

### Commit History

#### Commit `27025dd` - Documentation Updates (2025-12-08)

**Author**: Gnana Prakash M  
**Message**: Added Doc string for text_utils

**Changes**:

- Added comprehensive documentation strings to `text_utils` module
- Improved code readability and developer experience

---

#### Commit `dcd706d` - Resource Optimization (2025-12-08)

**Author**: Gnana Prakash M  
**Message**: Fix log ordering for the model loading to free VRAM

**Changes**:

- Optimized log ordering during model initialization
- Improved VRAM management to prevent widespread allocation issues
- Enhanced application stability during startup

---

#### Commit `eb33d49` - Robust Ingestion Pipeline (2025-12-08)

**Author**: Gnana Prakash M  
**Message**: feat(pipeline): implement robust ingestion with metadata enrichment

**Changes**:

- Implemented metadata enrichment logic for ingested media
- Enhanced pipeline robustness against failures
- Improved data consistency in the vector store

---

#### Commit `baa9fb5` - Schema Enhancements (2025-12-08)

**Author**: Gnana Prakash M  
**Message**: Added new schemas and metadata engine

**Changes**:

- Defined new Pydantic schemas for improved type safety
- Introduced a dedicated metadata engine for handling complex metadata

---

#### Commit `393cf31` - Prompt Management (2025-12-06)

**Author**: Gnana Prakash M  
**Message**: Added Prompts separately and made vision.py load from it

**Changes**:

- Decoupled system prompts from codebase
- Updated `vision.py` to load prompts dynamically
- Easier prompt tuning and management

---

#### Commit `5cefdd9` - Sidecar & SRT Support (2025-12-06)

**Author**: Gnana Prakash M  
**Message**: Updated the ingestion pipeline to handle sidecars / srt ingestion properly

**Changes**:

- Added support for sidecar files (e.g., subtitles)
- Improved SRT ingestion processing

---

#### Commit `00dc903` - Retrieval Search CLI (2025-12-06)

**Author**: Gnana Prakash M  
**Message**: Developed a cli to invoke the retrieval search on embeddings

**Changes**:

- Created a CLI interface for searching embeddings
- Enabled direct verification of vector search capabilities

---

### Sprint 4 Technical Highlights

#### Ingestion Pipeline Enhancements

- **Metadata Enrichment**: The pipeline now actively enriches media metadata, ensuring richer search capabilities.
- **Sidecar Support**: Proper handling of `.srt` and other sidecar files during ingestion.
- **Robustness**: Improved error handling and stability in the ingestion process.

#### CLI & Interactive Features

- **Embedding Search**: New CLI tools to perform retrieval tasks directly on the vector database.
- **Interactive Resolution**: Mechanism to interactively resolve ambiguous filenames or paths.

#### PowerShell & File Path Handling (Learnings)

During this sprint, significant issues were identified and resolved regarding file path handling in PowerShell, specifically with special characters.

- **The Issue**: PowerShell aggressively interprets brackets `[]` as wildcard/glob patterns. A file named `Movie [2025].mkv` would fail to resolve because PowerShell tries to match `[2025]` as a pattern.
- **Solution 1 (Recommended)**: Rename files to remove special characters (`[]`, `()`, space) and use underscores (e.g., `Movie_2025.mkv`).
- **Solution 2 (Escaping)**: If renaming is impossible, use backticks to escape special chars: ` uv run python main.py "Movie ``[2025``].mkv" `.
- **Best Practice**: Always enclose file paths in quotes when passing them as arguments: `uv run python main.py "path/to/file.mkv"`.

#### Resource Optimization

- **VRAM Management**: Optimized model loading to prevent unnecessary VRAM consumption.

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
- Pre-built wheel installation:
  ```bash
  uv add "dlib @ https://github.com/z-mahmud22/Dlib_Windows_Python3.x/raw/main/dlib-19.24.99-cp312-cp312-win_amd64.whl"
  ```

**dlib with CUDA Support (Optional)**:

For GPU-accelerated face detection, you can build dlib from source with CUDA support:

1. **Install NVIDIA CUDA Toolkit** (see References section for download links)

   - CUDA 12.4 or CUDA 11.8
   - cuDNN libraries

2. **Set CUDA environment variables** (PowerShell):

   ```powershell
   # Set CUDA path (adjust version as needed)
   $env:CUDA_PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
   $env:PATH="$env:CUDA_PATH\bin;$env:CUDA_PATH\libnvvp;$env:PATH"

   # Tell dlib explicitly to use CUDA + cuBLAS
   $env:DLIB_USE_CUDA="1"
   $env:DLIB_USE_CUBLAS="1"
   ```

3. **Build dlib from source**:

   ```bash
   # Uninstall pre-built version if installed
   uv pip uninstall dlib

   # CMD (Windows Command Prompt)
   set DLIB_USE_CUDA=1
   uv run pip install dlib --no-binary :all:

   # PowerShell
   $env:DLIB_USE_CUDA="1"
   uv run pip install dlib --no-binary :all:
   ```

> **Note**: Building dlib with CUDA requires Visual Studio Build Tools, CMake, and Ninja (see Build Tools section above). The build process may take 10-30 minutes.

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

- Centralize the python cache

```powershell
mkdir .cache
$env:PYTHONPYCACHEPREFIX = "D:\AI-Media-Indexer\.cache"
```

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

## Sprint 5 (Branch: `sprint-5`)

### Commit History

#### Commit `5eab3d6` - Agent CLI Implementation (2025-12-09)

**Author**: Gnana Prakash M  
**Message**: Added Agent server to schemas

**Changes**:

- Implemented `agent_cli.py` for MCP server interaction
- Added CLI entry point for agent operations

**Files Added**:

- `agent_cli.py`

### Sprint 5 Completed Tasks

#### Task 5.1 - MCP Agent Client ✅

- ✅ Implemented `AgentCLI` in `agent_cli.py`
  - Interactive command-line interface for the Media Indexer Agent
  - Integration with Model Context Protocol (MCP) via `mcp` library
  - Seamless communication with `core.agent.server`
  - Powered by Ollama (default: `llama3.1`) for natural language interaction
  - Support for tool calling (`search_media`, `ingest_media`)
  - Dynamic tool discovery and schema mapping

**Key Features**:

- **REPL Interface**: Interactive chat loop with state preservation
- **Tool Chaining**: Model can call tools, receive output, and synthesize answers
- **Stdio Transport**: Robust local communication between client and agent server

#### Task 5.2 - Schema Enhancements ✅

- ✅ Updated `core/schemas.py` with standard response models
  - **`SearchResponse`**: Standardized structure for visual and dialogue search results
  - **`IngestResponse`**: Unified response format for ingestion operations
  - Improved type safety for MCP tool returns

---

### Sprint 5 Technical Highlights

#### MCP Client-Server Architecture

- **Protocol**: Adopted the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) standard.
- **Client**: `agent_cli.py` acts as the host, managing the LLM (Ollama) and the connection to the server.
- **Server**: `core.agent.server` provides the actual tools (`search`, `ingest`) and executes them.
- **Flow**: User -> CLI -> LLM -> (Tool Call) -> CLI -> Server -> CLI -> (Tool Result) -> LLM -> CLI -> User.

This architecture decouples the "brain" (LLM/CLI) from the "body" (Tools/Server), allowing for easier switching of LLM backends or tool implementations.

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
- **NVIDIA CUDA 12.4**: [https://developer.nvidia.com/cuda-12-4-0-download-archive](https://developer.nvidia.com/cuda-12-4-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local)
- **NVIDIA cuDNN**: [https://developer.nvidia.com/cudnn-downloads](https://developer.nvidia.com/cudnn-downloads?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local)

- https://pypi.org/project/a2a-sdk/
