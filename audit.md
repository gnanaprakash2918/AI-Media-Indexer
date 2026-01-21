# AI-Media-Indexer Code Audit Report

**Audit Date**: 2026-01-20  
**Auditor**: Senior Software Architect  
**Project**: AI-Media-Indexer  
**Version**: 2.1.0  

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Architecture Overview](#architecture-overview)
3. [Pipeline Flow](#pipeline-flow)
4. [File-by-File Audit](#file-by-file-audit)
5. [Critical Issues](#critical-issues)
6. [Security Vulnerabilities](#security-vulnerabilities)
7. [Performance Bottlenecks](#performance-bottlenecks)
8. [Code Quality Issues](#code-quality-issues)
9. [Recommendations Summary](#recommendations-summary)

---

## Executive Summary

The AI-Media-Indexer is a sophisticated multimodal media indexing system that processes video/audio content through multiple AI pipelines including:
- **Transcription** (Whisper, AI4Bharat IndicConformer)
- **Face Detection/Recognition** (InsightFace, SFace, YuNet fallback)
- **Voice Diarization** (Pyannote)
- **Vision Analysis** (Local VLM via Ollama/Gemini)
- **Audio Event Detection** (CLAP)
- **Vector Search** (Qdrant)

### Key Strengths
- Robust fallback chains for model loading
- Comprehensive multimodal search with hybrid BM25+vector fusion
- Well-structured Pydantic schemas for data validation
- Lazy loading patterns for memory management

### Critical Issues Identified
1. ~~**Duplicate field definitions** in `config.py` (batch_size defined twice)~~ ✅ Fixed
2. **Potential infinite retry loops** in db.py retry decorator
3. **Missing async context manager** handling in pipeline
4. **Hardcoded credentials** in docker-compose.yaml
5. **Race conditions** in face track builder

> **Note**: See [Resolution Status](#resolution-status-2026-01-21) for completed fixes.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              AI-Media-Indexer                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────┐   ┌────────────────┐   ┌───────────────┐   ┌─────────────┐ │
│  │   API      │──▶│   Ingestion    │──▶│  Processing   │──▶│   Storage   │ │
│  │  (FastAPI) │   │   Pipeline     │   │   Modules     │   │  (Qdrant)   │ │
│  └────────────┘   └────────────────┘   └───────────────┘   └─────────────┘ │
│       │                   │                   │                   ▲         │
│       │                   ▼                   ▼                   │         │
│       │          ┌────────────────┐   ┌───────────────┐          │         │
│       │          │  Audio Branch  │   │ Visual Branch │          │         │
│       │          │  - Transcribe  │   │ - Frame Extract│         │         │
│       │          │  - Diarize     │   │ - VLM Caption  │         │         │
│       │          │  - CLAP Events │   │ - Face Detect  │         │         │
│       │          └────────────────┘   └───────────────┘          │         │
│       │                                       │                   │         │
│       │          ┌─────────────────────────────────────────────────┐       │
│       └─────────▶│               Retrieval System                  │       │
│                  │  - Agentic Search (LLM Query Expansion)         │       │
│                  │  - Hybrid Search (Vector + BM25 RRF)            │       │
│                  │  - Council Reranking (VLM + CrossEncoder)       │       │
│                  └─────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Pipeline Flow

### Ingestion Pipeline (core/ingestion/pipeline.py)

The ingestion pipeline follows a sequential processing flow with checkpointing for crash recovery:

```
Video Input
    │
    ▼
┌───────────────────┐
│ 1. Media Probe    │  ← FFprobe for metadata (duration, format)
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ 2. Audio Process  │  ← Whisper/IndicASR transcription
│   (5% → 30%)      │  ← SRT sidecar generation
│                   │  ← CLAP audio event detection
│                   │  ← Loudness analysis
│                   │  ← Music structure analysis
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ 3. Voice Process  │  ← Pyannote diarization
│   (35% → 50%)     │  ← Voice embedding extraction
│                   │  ← Speaker registry matching
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ 4. Frame Process  │  ← Frame extraction at interval
│   (55% → 90%)     │  ← InsightFace detection
│                   │  ← VLM captioning (Ollama)
│                   │  ← OCR extraction
│                   │  ← Temporal context building
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ 5. Scene Caption  │  ← PySceneDetect boundaries
│   (90%)           │  ← VLM scene summarization
│                   │  ← Multi-vector scene storage
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ 6. Post-Process   │  ← Global context aggregation
│   (95% → 100%)    │  ← SAM3 concept tracking
│                   │  ← Thumbnail generation
└───────────────────┘
```

---

## File-by-File Audit

### 1. `main.py` (Entry Point)

**File**: `d:\AI-Media-Indexer\main.py`  
**Lines**: 161  
**Purpose**: CLI entrypoint for single-video ingestion

#### Functions

| Function | Lines | Purpose |
|----------|-------|---------|
| `_ask_media_type()` | 24-45 | Interactive media type selection |
| `_run(video_path, media_type)` | 48-64 | Async pipeline execution |
| `_interactive_resolve(raw_path)` | 67-135 | Fuzzy file path resolution |
| `main()` | 138-160 | CLI entry point |

#### Issues

| Issue | Line | Severity | Description | Recommendation |
|-------|------|----------|-------------|----------------|
| Hardcoded Qdrant settings | 57-60 | Medium | Pipeline created with hardcoded localhost:6333 | Use settings from config.py |
| No error handling for asyncio.run | 156 | Low | Raw exception propagation | Wrap in try/except with user-friendly message |
| Windows-specific policy hardcoded | 153-154 | Low | May not be needed on newer Python | Check Python version before applying |

---

### 2. `config.py` (Configuration)

**File**: `d:\AI-Media-Indexer\config.py`  
**Lines**: 526  
**Purpose**: Pydantic Settings with hardware detection

#### Classes

| Class | Lines | Purpose |
|-------|-------|---------|
| `HardwareProfile` | 14-24 | Enum for hardware tiers |
| `LLMProvider` | 82-87 | Enum for LLM backends |
| `Settings` | 89-523 | Main configuration class |

#### Functions

| Function | Lines | Purpose |
|----------|-------|---------|
| `get_hardware_profile()` | 27-76 | Detect GPU VRAM and set batch sizes |

#### Issues

| Issue | Line | Severity | Description | Recommendation |
|-------|------|----------|-------------|----------------|
| **Duplicate field: batch_size** | 142-145, 207 | **Critical** | `batch_size` defined twice with different defaults (profile-based vs 24) | Remove duplicate on line 207 |
| Broad exception handling | 73-74 | Medium | Catches all exceptions in hardware detection | Catch specific exceptions (RuntimeError, OSError) |
| Missing validation for paths | 408-410 | Low | `arcface_model_path` not validated for existence | Add validator or handle missing file gracefully |
| Model validator mutates self | 461-486 | Medium | `adjust_dimensions` mutates fields, can cause confusion | Return new instance or use `__post_init__` |

---

### 3. `core/ingestion/pipeline.py` (Main Pipeline)

**File**: `d:\AI-Media-Indexer\core\ingestion\pipeline.py`  
**Lines**: 2537  
**Purpose**: Orchestrates full video ingestion workflow

#### Class: `IngestionPipeline`

| Method | Lines | Purpose |
|--------|-------|---------|
| `__init__` | 48-102 | Initialize all processors |
| `process_video` | 137-312 | Main orchestration method |
| `_process_audio` | 314-818 | Audio transcription + CLAP + loudness |
| `_process_voice` | 948-1075 | Speaker diarization + embedding |
| `_process_frames` | 1077-1380 | Visual analysis loop |
| `_process_scene_captions` | 1382-1558 | Scene-level aggregation |
| `_process_single_frame` | 1560-2093 | Per-frame VLM + face detection |
| `_post_process_video` | 2264-2344 | Global enrichment + SAM3 |

#### Issues

| Issue | Line | Severity | Description | Recommendation |
|-------|------|----------|-------------|----------------|
| Duplicate Sam3Tracker instantiation | 68, 99-101 | Medium | `Sam3Tracker()` created twice in `__init__` | Remove line 68 |
| Unused import `uuid` inside method | 2077 | Low | Import already at module level | Remove local import |
| Exception swallowed silently | 1596-1597 | Medium | Face detection exception caught with `pass` | Log warning at minimum |
| Variable shadowing in loop | 672 | Low | `i` used for loop, zip doesn't need it | Use `_` for unused variable |
| Indentation bug in SAM3 tracking | 2488-2497 | **Critical** | `if track_key not in tracks` is outside the loop | Fix indentation to be inside the for loop |
| Hardcoded `random.randint` for cluster IDs | 1000, 1004 | Medium | Random IDs can collide | Use UUID or atomic counter |
| Missing await for subprocess | 1038 | Low | `subprocess.run` is blocking in async context | Use `asyncio.create_subprocess_exec` |
| Variables possibly undefined | 714, 768, 774 | Medium | `audio_array`, `sr` used in `locals()` check | Initialize at start of method |

---

### 4. `core/storage/db.py` (Vector Database)

**File**: `d:\AI-Media-Indexer\core\storage\db.py`  
**Lines**: 6140  
**Purpose**: Qdrant interface for all vector operations

#### Class: `VectorDB`

| Method Category | Description |
|-----------------|-------------|
| **Initialization** | Lazy encoder loading, collection setup |
| **Insert** | insert_media_segments, insert_face, insert_voice_segment, etc. |
| **Search** | search_frames, search_scenes, hybrid search |
| **Identity** | get_face_name_by_cluster, match_speaker, etc. |

#### Issues

| Issue | Line | Severity | Description | Recommendation |
|-------|------|----------|-------------|----------------|
| Method `_lazy_load_encoder()` referenced but undefined | 367 | **Critical** | `self._lazy_load_encoder()` called but method doesn't exist | Should be `self._ensure_encoder_loaded()` |
| Retry decorator could loop forever | 27-54 | Medium | If `max_retries` is 0, behavior is undefined | Add check for max_retries > 0 |
| `close()` method may not exist | 157 (lifespan) | Medium | Server calls `pipeline.db.close()` but method not shown | Ensure close() method exists |
| Numpy type serialization | 57-74 | Low | `sanitize_numpy_types` good but not called everywhere | Apply decorator to all upsert methods |
| Large scroll limits | 2149, 2189 | Medium | Scrolling 1000-5000 items may OOM | Use pagination with iterators |

---

### 5. `core/retrieval/agentic_search.py` (Search Agent)

**File**: `d:\AI-Media-Indexer\core\retrieval\agentic_search.py`  
**Lines**: 1285  
**Purpose**: LLM-powered search with query expansion

#### Class: `SearchAgent`

| Method | Lines | Purpose |
|--------|-------|---------|
| `parse_query` | 81-103 | LLM query expansion |
| `search_scenes` | 146-264 | Scene-level search |
| `search` | 266-436 | Frame-level legacy search |
| `hybrid_search` | 471-511 | BM25+vector RRF fusion |
| `rerank_with_llm` | 517-618 | Second-stage verification |
| `sota_search` | 620-end | Full SOTA pipeline |

#### Issues

| Issue | Line | Severity | Description | Recommendation |
|-------|------|----------|-------------|----------------|
| Nested class definition in loop | 566-572 | **Critical** | `RerankResult` class defined inside for loop | Move class definition outside loop |
| Duplicate filter logic | 312-390 | Medium | Filter building duplicated between search methods | Extract to helper method |
| Traceback import inside except | 792-793 | Low | Import should be at module level | Move `import traceback` to top |
| `hasattr` checks repeated | 680-697 | Low | Multiple hasattr checks for same attributes | Use getattr with defaults |

---

### 6. `api/server.py` (FastAPI Server)

**File**: `d:\AI-Media-Indexer\api\server.py`  
**Lines**: 239  
**Purpose**: API server composition with CORS and middleware

#### Functions

| Function | Lines | Purpose |
|----------|-------|---------|
| `lifespan()` | 127-158 | App startup/shutdown |
| `create_app()` | 161-223 | FastAPI factory |

#### Issues

| Issue | Line | Severity | Description | Recommendation |
|-------|------|----------|-------------|----------------|
| Duplicate imports | 24, 30-31 | Low | `asyncio`, `logging` imported twice | Remove duplicates |
| Wildcard CORS origins | 171 | **Security** | `allow_origins=["*"]` allows any domain | Restrict to known origins in production |
| Trace ID unused | 181 | Low | Trace ID extracted but assigned to `_` | Use or remove |
| Optional import without full path | 114-117 | Low | `overlays` import could fail silently | Log warning on import failure |

---

### 7. `core/knowledge/schemas.py` (Knowledge Models)

**File**: `d:\AI-Media-Indexer\core\knowledge\schemas.py`  
**Lines**: 829  
**Purpose**: Pydantic schemas for LLM structured output

#### Classes

| Class | Lines | Purpose |
|-------|-------|---------|
| `DynamicEntity` | 18-80 | Flexible entity representation |
| `DynamicParsedQuery` | 83-199 | Complex query parser |
| `EntityDetail` | 207-233 | Legacy entity model |
| `FrameAnalysis` | 261-383 | Per-frame analysis schema |
| `SceneData` | 405-590 | Scene-level aggregation |
| `ParsedQuery` | 658-829 | Search query parser |

#### Issues

| Issue | Line | Severity | Description | Recommendation |
|-------|------|----------|-------------|----------------|
| Duplicate person name weighting | 758-760 | Low | `person.name` added twice to parts | Intentional for weighting, add comment |
| Large schema complexity | All | Medium | Many overlapping schemas | Consider consolidating DynamicParsedQuery and ParsedQuery |

---

### 8. `core/schemas.py` (Core Models)

**File**: `d:\AI-Media-Indexer\core\schemas.py`  
**Lines**: 378  
**Purpose**: Core Pydantic models and enums

#### Classes

| Class | Lines | Purpose |
|-------|-------|---------|
| `MediaType` | 11-20 | Media type enum |
| `EntityDetail` | 26-60 | Entity with visual details |
| `FrameAnalysis` | 90-171 | Frame analysis (duplicate) |
| `MediaMetadata` | 174-197 | File metadata |
| `DetectedFace` | 225-235 | Face detection result |
| `SpeakerSegment` | 301-311 | Voice segment |

#### Issues

| Issue | Line | Severity | Description | Recommendation |
|-------|------|----------|-------------|----------------|
| Duplicate EntityDetail class | 26-60 | Medium | Same class exists in knowledge/schemas.py | Use single source of truth |
| Duplicate FrameAnalysis class | 90-171 | Medium | Different implementation than knowledge/schemas.py | Consolidate or rename |
| `visual_details` accepts multiple types | 37-40 | Low | Union type may cause validation issues | Consider normalizing at input |

---

### 9. `core/processing/identity.py` (Face Processing)

**File**: `d:\AI-Media-Indexer\core\processing\identity.py`  
**Lines**: 1134  
**Purpose**: Face detection, recognition, and tracking

#### Classes

| Class | Lines | Purpose |
|-------|-------|---------|
| `ActiveFaceTrack` | 122-140 | Dataclass for active tracks |
| `FaceTrackBuilder` | 143-389 | Temporal face grouping |
| `FaceManager` | 433-1112 | Face detection/embedding |

#### Issues (from outline review)

| Issue | Severity | Description | Recommendation |
|-------|----------|-------------|----------------|
| Nested class in function | Medium | `MEMORYSTATUS` class defined inside `_detect_system_capabilities` | Move to module level |
| GPU_SEMAPHORE global | Low | Global semaphore may cause issues with multiple pipelines | Use dependency injection |

---

### 10. `core/processing/transcriber.py` (Audio Transcription)

**File**: `d:\AI-Media-Indexer\core\processing\transcriber.py`  
**Lines**: 887  
**Purpose**: Whisper-based transcription with model management

#### Class: `AudioTranscriber`

| Method | Lines | Purpose |
|--------|-------|---------|
| `__init__` | 99-117 | Initialize compute settings |
| `transcribe` | 593-658 | Main transcription entry |
| `detect_language` | 810-883 | Language detection |
| `_load_model` | 392-495 | Model loading with fallback |

#### Issues (from outline review)

| Issue | Severity | Description | Recommendation |
|-------|----------|-------------|----------------|
| Shared model state | Medium | `_SHARED_MODEL` class variable may cause issues | Use instance-level caching with weak references |
| Context manager doesn't unload | Low | `__exit__` noted as not unloading | Document this behavior clearly |

---

### 11. `core/processing/voice.py` (Voice Diarization)

**File**: `d:\AI-Media-Indexer\core\processing\voice.py`  
**Lines**: 402  
**Purpose**: Pyannote-based speaker diarization

#### Class: `VoiceProcessor`

| Method | Lines | Purpose |
|--------|-------|---------|
| `__init__` | 74-101 | Initialize with settings check |
| `process` | 214-302 | Main diarization pipeline |
| `cleanup` | 192-212 | Model unloading |

#### Issues (from outline review)

| Issue | Severity | Description | Recommendation |
|-------|----------|-------------|----------------|
| Monkey-patched torch.load | Medium | Line 28 patches `torch.load` globally | Use local wrapper instead of global patch |

---

## Critical Issues

### 1. Duplicate `batch_size` Field (config.py:142, 207)

**Severity**: Critical  
**Impact**: The second definition (line 207) overrides the hardware-profile-based setting, nullifying automatic tuning.

```python
# Line 142-145 (correct - profile based)
batch_size: int = Field(
    default=_HW_PROFILE["batch_size"],
    description="Batch size for inference",
)

# Line 207 (duplicate - overrides profile)
batch_size: int = Field(default=24)  # REMOVE THIS
```

**Fix**: Remove line 207.

---

### 2. Indentation Bug in SAM3 Tracking (pipeline.py:2488-2497)

**Severity**: Critical  
**Impact**: Track aggregation logic is outside the for loop, processing only the last `obj_id`.

```python
# CURRENT (BROKEN):
for obj_id in obj_ids:
    if obj_id < len(prompts):
        concept = prompts[obj_id]
    else:
        concept = f"object_{obj_id}"
    track_key = f"{concept}_{obj_id}"

if track_key not in tracks:  # WRONG INDENTATION - outside loop
    tracks[track_key] = {...}

# FIXED:
for obj_id in obj_ids:
    if obj_id < len(prompts):
        concept = prompts[obj_id]
    else:
        concept = f"object_{obj_id}"
    track_key = f"{concept}_{obj_id}"

    if track_key not in tracks:  # CORRECT - inside loop
        tracks[track_key] = {...}
    else:
        tracks[track_key]["end"] = max(...)
```

---

### 3. Undefined Method Reference (db.py:367)

**Severity**: Critical  
**Impact**: Runtime error when encoder needs re-initialization.

```python
# Line 367 calls undefined method:
self._lazy_load_encoder()

# Should be:
self._ensure_encoder_loaded()
```

---

### 4. Nested Class in Loop (agentic_search.py:566-572)

**Severity**: Critical  
**Impact**: Performance degradation - class definition creates new type on each iteration.

```python
# CURRENT (BROKEN):
for candidate in candidates[:top_k]:
    ...
    class RerankResult(BaseModel):  # Created on EVERY iteration
        match_score: float = ...
```

**Fix**: Move class definition outside the method.

---

## Security Vulnerabilities

### 1. Wildcard CORS (api/server.py:171)

**Risk Level**: High  
**Description**: `allow_origins=["*"]` allows requests from any domain.

**Recommendation**:
```python
allowed_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    ...
)
```

### 2. Hardcoded Redis Password (config.py:352)

**Risk Level**: Medium  
**Description**: Default `redis_auth = "redispass"` in code.

**Recommendation**: Remove default, require environment variable.

### 3. HuggingFace Token Exposure (transcriber.py:39-44)

**Risk Level**: Low  
**Description**: Token logged on startup if login fails.

**Recommendation**: Don't log token-related failures with full context.

---

## Performance Bottlenecks

### 1. Synchronous subprocess.run in Async Context (pipeline.py:1038)

**Impact**: Blocks event loop during FFmpeg calls.

**Current**:
```python
subprocess.run(cmd, capture_output=True, text=True)
```

**Recommended**:
```python
proc = await asyncio.create_subprocess_exec(
    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
)
await proc.communicate()
```

### 2. Large Scroll Operations (db.py:2149, 2189)

**Impact**: Memory pressure when scrolling 1000-5000 items.

**Recommendation**: Use cursor-based pagination with smaller batches:
```python
async for batch in paginate_scroll(collection, batch_size=100):
    yield from batch
```

### 3. Repeated Model Loading (identity.py, voice.py)

**Impact**: Multiple instances may load same models.

**Recommendation**: Implement singleton pattern or model registry.

---

## Code Quality Issues

### 1. Code Duplication

| Location A | Location B | Description |
|------------|------------|-------------|
| core/schemas.py:EntityDetail | core/knowledge/schemas.py:EntityDetail | Same class, different implementations |
| core/schemas.py:FrameAnalysis | core/knowledge/schemas.py:FrameAnalysis | Same class name, different fields |
| pipeline.py filter building | agentic_search.py filter building | Qdrant filter construction duplicated |

### 2. Missing Type Hints

| File | Issue |
|------|-------|
| pipeline.py:1616 | `idx` should be `int` |
| db.py scroll methods | Return types not fully annotated |

### 3. Inconsistent Error Handling

| Pattern | Files | Issue |
|---------|-------|-------|
| Silent `pass` | pipeline.py | Exceptions ignored |
| Bare `except:` | Some files | Catches all including SystemExit |

---

## Recommendations Summary

### Immediate Fixes (Critical)

1. **Remove duplicate batch_size** in config.py:207
2. **Fix SAM3 indentation** in pipeline.py:2488
3. **Fix undefined method** in db.py:367
4. **Move RerankResult class** outside loop in agentic_search.py

### Security Hardening

1. Restrict CORS origins to known domains
2. Move secrets to environment variables only
3. Add rate limiting to API endpoints

### Performance Improvements

1. Replace subprocess.run with async subprocess
2. Implement cursor-based pagination for large scrolls
3. Add model registry for singleton pattern

### Code Quality

1. Consolidate duplicate schemas
2. Add comprehensive type hints
3. Implement consistent logging/error handling patterns
4. Add unit tests for critical paths

### Documentation

1. Add docstrings to all public methods
2. Create API documentation with OpenAPI
3. Document configuration options

---

## Appendix A: File Statistics

| Category | Count | Total Lines |
|----------|-------|-------------|
| Python Files | 158+ | ~50,000+ |
| Core Pipeline | 1 | 2,537 |
| Core Storage | 1 | 6,140 |
| Core Retrieval | 13 | ~15,000 |
| Processing | 44 | ~25,000 |
| API Routes | 15 | ~5,000 |

---

## Resolution Status (2026-01-21)

### Critical Fixes ✅

| Issue | File | Status | Notes |
|-------|------|--------|-------|
| Duplicate `batch_size` | `config.py:207` | ✅ **FIXED** | Removed duplicate, hardware profile now used |
| Undefined method `_lazy_load_encoder` | `db.py:367` | ✅ **FIXED** | Changed to `_ensure_encoder_loaded()` |
| SAM3 indentation bug | `pipeline.py:2488` | ✅ **FIXED** | Track logic now inside for loop |
| Nested class in loop | `agentic_search.py:566` | ✅ **FIXED** | `RerankResult` moved to module level |

### Security Fixes ✅

| Issue | Status | Notes |
|-------|--------|-------|
| Wildcard CORS | ✅ **FIXED** | Now uses `CORS_ORIGINS` env var with safe defaults |
| Hardcoded Redis password | ⏳ Deferred | Requires deployment changes |

### Performance Fixes ✅

| Issue | Status | Notes |
|-------|--------|-------|
| Blocking subprocess | ✅ **FIXED** | `_generate_main_thumbnail` now async |
| Large scroll pagination | ⏳ Deferred | Lower priority |

### Code Quality ✅

| Issue | Status | Notes |
|-------|--------|-------|
| Duplicate schemas | ✅ **DOCUMENTED** | Cross-references added between flexible/strict versions |
| Missing type hints | ⏳ Deferred | Low impact |
| Silent exception handling | ⏳ Deferred | Requires broader refactor |

---

*End of Audit Report*
