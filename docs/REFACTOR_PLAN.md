# AI-Media-Indexer — Architecture Overhaul & Refactoring Plan

> **Created:** 2026-07-12  
> **Reference:** [CURRENT_ARCHITECTURE.md](./CURRENT_ARCHITECTURE.md)  
> **Codebase:** ~56,574 lines Python (197 files) + React/TS frontend  
> **Estimated Total Duration:** 5–7 weeks  
> **Principle:** No "vibe-coded" prototype survives production. Every phase produces a deployable system.

---

## Table of Contents

- [Preamble: The Problem](#preamble-the-problem)
- [Phase 1: Understand the Codebase](#phase-1-understand-the-codebase)
- [Phase 2: Architecture & Scalability Audit](#phase-2-architecture--scalability-audit)
- [Phase 3: Open-Source Modernization Strategy](#phase-3-open-source-modernization-strategy)
- [Phase 4: Decoupling & Removing Heavy Dependencies](#phase-4-decoupling--removing-heavy-dependencies)
- [Phase 5: The Refactoring Engine](#phase-5-the-refactoring-engine)
- [Phase 6: Security Hardening](#phase-6-security-hardening)
- [Phase 7: Agentic Workflow & Search Accuracy Overhaul](#phase-7-agentic-workflow--search-accuracy-overhaul)
- [Phase 8: Testing, CI, & Production Readiness](#phase-8-testing-ci--production-readiness)
- [Execution Timeline](#execution-timeline)
- [Verification Checklist](#verification-checklist)

---


## Core Pedagogical & Open-Source Principles
This codebase will not just be refactored; it will be transformed into a **masterclass for backend engineering fundamentals**.
1. **Laptop-First Scalability**: We will eliminate brute-force logic (like extracting millions of frames) in favor of adaptive scene boundaries, making it the best OSS video indexer capable of running 12+ hour videos on commodity laptops.
2. **Open-Source Standard (Podman)**: We prioritize daemonless, rootless containers (Podman) over Docker to enforce the highest security and open-source standards.
3. **Clean Code as a Teacher**: The code will strictly adhere to the Strategy and Factory design patterns. Hardcoded models are replaced with Universal Protocols (`LLMProvider`), and monolithic logic is decoupled via Dependency Injection (FastAPI `Depends()`). This teaches readers *how* to build maintainable enterprise Python.
4. **Subtitles & Accuracy**: Implementing contextual LLM-based error correction to simulate ROVER (Recognizer Output Voting Error Reduction) without needing three heavy ASR models running locally.

## Preamble: The Problem

This project is a "vibe-coded" monolith. It works — it indexes videos, extracts faces, transcribes audio, and runs semantic search across 7 modalities. But it was built for speed of prototyping, not maintainability.

The symptoms:

| Symptom | Evidence |
|---|---|
| **God classes** | `VectorDB` (1,667 lines), `SearchAgent` (1,508 lines), `IngestionPipeline` (1,056 lines) |
| **God config** | `config.py` — 1,054 lines, 150+ settings, imports `torch` at module level, 54 consumers |
| **Dead code** | `core/processors/` — 4 files with broken imports, zero consumers |
| **Duplicate logic** | Two `MultiAgentOrchestrator` classes, two LLM packages, two schema packages, three prompt loaders |
| **Global singletons** | 20+ `global` keyword usages across the codebase |
| **No tests** | Zero unit tests. Root `conftest.py` mocks `torch` globally. Only integration scripts |
| **Binary blobs in git** | 104 MB of YOLO model weights (`yolov8m.pt` × 2) + 115 MB of test videos |
| **Unsandboxed media parsing** | FFmpeg subprocesses run without namespace isolation — classic RCE vector |
| **Security gaps** | Two competing sanitizers (`query_sanitizer.py` + `sanitizer.py`), neither integrated into the main search path consistently |
| **Fixed-interval chunking** | Audio/video chunked by time intervals, not semantic boundaries — ceilings search accuracy |
| **Monolithic ingestion + query** | Heavy compute (ingestion) and low-latency (search) run in the same process |

**The goal:** Transform this into a clean, modular, production-grade system — without rewriting from scratch.

---

## Phase 1: Understand the Codebase

> **Before changing a single line of code, map the blast radius.**

**Duration:** 2–3 days  
**Risk:** 🟢 None — read-only  
**Output:** Complete mental model + blast radius documented

---

### 1.1 — Map the Entry Points

There are **6 distinct entry points** into this system. Each initializes the world differently:

| Entry Point | File | What It Does | Heavy Deps |
|---|---|---|---|
| **FastAPI Server** | [`api/server.py`](../api/server.py) | REST API for web UI + search | Full model warmup at startup |
| **CLI Ingestion** | [`main.py`](../main.py) | Process a single video file | Full pipeline + all AI models |
| **Search CLI** | [`search_cli.py`](../search_cli.py) | Run search query from terminal | VectorDB + SearchAgent + LLM |
| **Agent CLI** | [`agent_cli.py`](../agent_cli.py) | Interactive MCP agent chat | Ollama + MCP tools |
| **Agent Orchestrator** | [`agent_main.py`](../agent_main.py) | Multi-agent routing | LLM + orchestrator |
| **MCP Server** | [`core/agent/server.py`](../core/agent/server.py) | Tool server for LLM agents | VectorDB + SearchAgent |

> [!IMPORTANT]
> **Every single entry point** imports `config.py`, which triggers `import torch` and `get_hardware_profile()` at module level. This is the root cause of the 2–5 second cold-start penalty across the entire application.

---

### 1.2 — Trace the Media Flow

Follow a single `.mp4` file from upload to searchable index:

```
Upload/CLI Input
    │
    ▼
┌──────────────────────────────────────────────────────────────────────┐
│ api/routes/ingest.py → validate_path → IngestionPipeline.start()    │
│                                                                      │
│ PROBE PHASE                                                          │
│   └─ prober.py → FFprobe subprocess → duration, codec, resolution   │
│                  ⚠ UNSANDBOXED: subprocess.check_output(ffprobe)     │
│                                                                      │
│ AUDIO EXTRACTION                                                     │
│   └─ extractor.py → FFmpeg subprocess → .wav file                   │
│                     ⚠ UNSANDBOXED: subprocess.run(ffmpeg)            │
│                     ⚠ ENTIRE AUDIO IN MEMORY for small files         │
│                                                                      │
│ TRANSCRIPTION (AudioStageMixin)                                      │
│   └─ transcriber.py → Whisper large-v3 (faster-whisper)             │
│      ├─ Language detection (30s sample)                              │
│      ├─ Chunked transcription (fixed 600s chunks)                   │
│      │   ⚠ FIXED INTERVAL, not semantic boundaries                  │
│      └─ Word-level alignment → segments with timestamps             │
│                                                                      │
│ VOICE DIARIZATION (VoiceStageMixin)                                  │
│   └─ voice.py → PyAnnote 3.1 → who spoke when                      │
│      └─ Speaker embedding extraction → cosine matching              │
│                                                                      │
│ SCENE DETECTION                                                      │
│   └─ TransNet V2 (ONNX) → shot boundaries                          │
│      └─ PySceneDetect (content-based) → fallback                    │
│                                                                      │
│ FRAME PROCESSING (FrameStageMixin — 1,263 lines!)                    │
│   └─ For each extracted frame:                                       │
│      ├─ SigLIP → visual embedding (1152d)                           │
│      ├─ InsightFace → face detection + ArcFace embeddings           │
│      ├─ VLM caption → Ollama/Gemini dense description               │
│      │   ⚠ SEQUENTIAL: VLM_SEMAPHORE limits concurrency             │
│      ├─ PaddleOCR → text extraction                                 │
│      ├─ YOLO-World → object detection                               │
│      └─ Deep Research → shot type, mood, aesthetics                 │
│                                                                      │
│ AUDIO EVENTS (AudioEventsStageMixin)                                 │
│   └─ CLAP → audio event embeddings (5s fixed windows)               │
│      ⚠ FIXED 5s WINDOWS, not semantic audio boundaries              │
│                                                                      │
│ SCENE AGGREGATION (SceneStageMixin)                                  │
│   └─ Merge frames into scenes → VLM scene summary                  │
│      └─ InternVideo → video embeddings (1024d) per scene            │
│      └─ Text encoder → BGE-M3 text embeddings                      │
│                                                                      │
│ POST-PROCESSING                                                      │
│   ├─ Knowledge graph → Neo4j entity nodes + edges                   │
│   ├─ Metadata enrichment → TMDB/OMDB API                            │
│   ├─ Global summary → L1/L2 hierarchical video summary              │
│   └─ Thumbnail extraction                                           │
│                                                                      │
│ STORAGE → 10 Qdrant collections                                      │
│   media_frames, media_segments, scenes, scenelets,                   │
│   faces, voice_segments, audio_events, video_metadata,               │
│   summaries, masklets                                                │
└──────────────────────────────────────────────────────────────────────┘
```

> [!CAUTION]
> **The entire ingestion pipeline runs in-process.** For a 3-hour movie, this can take 30–60 minutes, blocking the API server's event loop. The existing Celery integration (`core/ingestion/celery_app.py`, `core/ingestion/tasks.py`) exists but is gated behind `enable_distributed_ingestion` and contains several incomplete TODO comments (see `tasks.py` lines 30–55).

---

### 1.3 — Audit the Dependencies

**Heavy dependencies imported globally just for utility functions:**

| Dependency | Size | Where Imported | Actual Usage |
|---|---|---|---|
| `torch` (~2.5 GB) | `config.py:10` | Module level on import | Only `torch.cuda.is_available()` for hardware detection |
| `paddlepaddle` + `paddleocr` (~800 MB) | `core/processing/ocr.py` | Lazy import | OCR on ~10% of frames |
| `insightface` (~300 MB models) | `core/processing/identity.py` | `__init__` of FaceManager | Face detection only |
| `transformers` (~200 MB) | Multiple processors | Lazy but many sites | Could use `httpx` to local server instead |
| `deepface` (~200 MB) | `core/processing/biometrics.py` | Optional feature | Only if `enable_face_recognition=True` |
| `fer` + `tf-keras` (~150 MB) | `core/processing/biometrics.py` | Optional feature | Emotion detection only |
| `sam3` (from git) | Variable | `core/tracking/sam3_tracker.py` | Only if `enable_sam3_tracking=True` |

**Dev tools shipped as runtime dependencies (bloat in Docker):**
- `black`, `ruff`, `cmake`, `cython`, `ninja`, `pytest` — all in `[project.dependencies]`
- `hydra-core` — declared but **never imported in any Python file**

**Result:** Production Docker image is ~8–12 GB when it could be ~2–3 GB.

---

### 1.4 — Isolate the Core Logic

**Code that actually processes media/prompts (keep and refactor):**

| Module | Purpose | LOC |
|---|---|---|
| `core/processing/transcriber.py` | Whisper transcription | 1,109 |
| `core/processing/identity.py` | Face detection + clustering | 1,087 |
| `core/processing/voice.py` | Speaker diarization | 682 |
| `core/processing/audio_events.py` | CLAP audio events | 706 |
| `core/processing/video_understanding.py` | InternVideo | 598 |
| `core/processing/visual_encoder.py` | SigLIP visual encoding | 458 |
| `core/processing/ocr.py` | PaddleOCR/EasyOCR/Surya | 569 |
| `core/retrieval/agentic_search.py` | Multi-modal search | 1,508 |
| `core/retrieval/reranker.py` | VLM + BGE + cross-encoder | 680 |
| `core/storage/encoder.py` | Text embedding lifecycle | 286 |

**Code that just moves data around (simplify or eliminate):**

| Module | Purpose | Action |
|---|---|---|
| `core/integration.py` | Feature flag hub + singleton getters | Inline or delete |
| `core/processors/*` | Dead wrapper classes | **Delete** |
| `core/orchestration/agent_graph.py` | Duplicate orchestrator | **Delete** |
| `core/agent/card.py` | Overlaps with `cards.py` | **Merge** |
| `core/utils/model_warmer.py` | Downloads models at startup | Keep but defer to background |

---

## Phase 2: Architecture & Scalability Audit

> **Document the current state with brutal honesty — reference specific files, classes, and functions.**

**Duration:** 2–3 days (may overlap with Phase 1)  
**Risk:** 🟢 None — documentation only  
**Output:** [`CURRENT_ARCHITECTURE.md`](./CURRENT_ARCHITECTURE.md) (already produced) + scalability analysis below

---

### 2.1 — Component Mapping: Data Flow from Ingestion to Storage

See the trace in [Phase 1.2](#12--trace-the-media-flow). The key architectural layers:

```
┌─────────────────────────────────────────────────────┐
│                    ENTRY LAYER                       │
│  api/server.py  |  main.py  |  agent_cli.py         │
│  (REST API)     | (CLI)     | (MCP Agent)           │
└───────────┬─────────┬───────────┬───────────────────┘
            │         │           │
            ▼         ▼           ▼
┌─────────────────────────────────────────────────────┐
│              ORCHESTRATION LAYER                     │
│  IngestionPipeline (5 mixin stages)                  │
│  SearchAgent (2 mixin classes)                       │
│  MultiAgentOrchestrator (x2 competing!)              │
│  ⚠ NO SERVICE LAYER — business logic in routes      │
└───────────┬─────────────────────┬───────────────────┘
            │                     │
            ▼                     ▼
┌───────────────────┐  ┌──────────────────────────────┐
│  PROCESSING LAYER │  │  RETRIEVAL LAYER              │
│  37 flat files in │  │  agentic_search.py (1,508L)   │
│  core/processing/ │  │  reranker.py (680L)           │
│  (transcriber,    │  │  rag.py (546L)                │
│   identity,       │  │  hybrid.py (240L)             │
│   voice, etc.)    │  │  hitl_feedback.py (310L)      │
└───────────┬───────┘  └──────────────┬───────────────┘
            │                         │
            ▼                         ▼
┌─────────────────────────────────────────────────────┐
│               STORAGE LAYER                          │
│  VectorDB (1,667L) = 4 Repositories via MI           │
│  TextEncoder (286L) — SentenceTransformer            │
│  IdentityGraph (497L) — Track-level clustering       │
│  Qdrant Collections: 10 total                        │
│  BM25 Index: pickle file on disk                     │
│  Knowledge Graph: Neo4j (optional)                   │
│  Job Store: SQLite                                   │
└─────────────────────────────────────────────────────┘
```

---

### 2.2 — Scalability Bottlenecks (Why This Architecture Breaks at Scale)

#### 🔴 Bottleneck 1: Synchronous Blocking in Async Context

**Where:** `core/processing/transcriber.py:238`, `core/processing/extractor.py:340`, `core/processing/prober.py:34`, `core/processing/scene_detector.py:151`

**Problem:** Multiple processors call `subprocess.run()` (blocking) or `subprocess.check_output()` (blocking) inside what should be an async pipeline. The frame extraction path in `extractor.py:340` runs FFmpeg synchronously:

```python
# core/processing/extractor.py:340 — BLOCKING in async context
return subprocess.run(
    cmd, timeout=timeout, check=True, capture_output=True
)
```

**Impact:** For a 3-hour video, FFmpeg extraction can take 5–10 minutes. During this time, the entire async event loop is blocked. No search queries can be served.

**Fix:** Wrap all blocking subprocess calls in `asyncio.create_subprocess_exec()` or `asyncio.to_thread()`.

---

#### 🔴 Bottleneck 2: In-Memory Buffering of Entire Media Streams

**Where:** `core/processing/transcriber.py` (chunking), `core/ingestion/stages/frame_stage.py` (frame accumulation), `core/storage/encoder.py` (embedding cache)

**Problem:** The transcriber chunks audio by **fixed 600-second intervals** (`chunk_duration_seconds=600`), but loads each chunk fully into memory before processing. For a 3-hour file, that's 18 sequential chunks, each holding a 600-second WAV buffer. The frame stage accumulates frame metadata in a list that grows linearly with video duration.

**Fix:** Use streaming/generator patterns. Yield frames as they're extracted instead of accumulating:

```python
# Instead of:
frames = []
for timestamp in timestamps:
    frame = extract_frame(video, timestamp)
    frames.append(frame)  # ← Memory grows linearly
process_all(frames)

# Use:
async def frame_stream(video, timestamps):
    for timestamp in timestamps:
        frame = extract_frame(video, timestamp)
        yield frame  # ← Constant memory

async for frame in frame_stream(video, timestamps):
    await process_and_store(frame)  # ← Process and discard
```

---

#### 🔴 Bottleneck 3: Monolithic Process — Ingestion + Query in Same Worker

**Where:** `api/server.py` lifespan (lines 144–196) — initializes BOTH `IngestionPipeline` AND `SearchAgent` in the same process.

**Problem:** A single video ingestion loads Whisper (3 GB VRAM), InsightFace (1.5 GB), VLM (6 GB), PaddleOCR, YOLO — all competing for the same GPU with the search embedding model (1–4 GB). The `ResourceArbiter` (`core/utils/resource_arbiter.py`) tracks VRAM budgets but **doesn't actually prevent OOM** — it's advisory only.

**Fix:** Decouple ingestion workers from query servers via a message broker (Phase 3).

---

#### 🟠 Bottleneck 4: Fixed-Interval Chunking, Not Semantic Boundaries

**Where:** `config.py:846` (`chunk_duration_seconds=600`), `core/processing/audio_events.py` (5-second CLAP windows)

**Problem:** Audio is chunked at fixed 600-second intervals. CLAP audio events use fixed 5-second windows. Neither respects semantic boundaries (sentence breaks, scene changes, music transitions). This means:
- A sentence can be split across two chunks, corrupting the transcript
- An audio event (gunshot, music crescendo) can span a window boundary, halving its embedding quality

**Fix:** Use scene boundaries from TransNet/PySceneDetect as chunk boundaries. For audio, use voice activity detection (VAD) from PyAnnote to find natural speech boundaries.

---

#### 🟠 Bottleneck 5: Eager Model Loading

**Where:** `core/storage/db.py:141` (VisualEncoder loaded in `__init__`), `core/ingestion/pipeline.py:82-120` (10+ models in `__init__`)

**Problem:** `VectorDB.__init__()` eagerly loads the SigLIP visual encoder — even for search-only paths. `IngestionPipeline.__init__()` eagerly constructs `TransNetV2`, `VideoVLM`, `FaceManager`, `VoiceProcessor`, `GraphBuilder`, `SAM3Tracker`. Server startup takes 30–120 seconds.

**Fix:** Lazy `@property` for every model. Load on first access, not construction.

---

#### 🟡 Bottleneck 6: BM25 Index as Single Pickle File

**Where:** `core/storage/keyword_index.py` — `KeywordIndex` loads/saves a single pickle file (`data/bm25_index.pkl`)

**Problem:** The entire BM25 index is serialized/deserialized as one object. Beyond ~100K documents, this becomes a memory bomb and a startup bottleneck.

**Fix:** Replace with PostgreSQL full-text search or Tantivy (Rust-based, Python bindings) for persistent keyword indexing.

---

### 2.3 — Dependency Bloat Analysis

**Total `pyproject.toml` dependency count:** 62 direct dependencies  
**Production Docker image estimate:** 8–12 GB  

**Dependencies that could be eliminated if models run behind HTTP:**

| Dependency | Size | Replacement |
|---|---|---|
| `torch` + `torchvision` + `torchaudio` | ~2.5 GB | Offload to Ollama/vLLM server; use `httpx` for inference |
| `transformers` | ~200 MB | Same — call local model server |
| `paddlepaddle` + `paddleocr` | ~800 MB | Run as microservice or call via HTTP |
| `insightface` | ~300 MB (models) | Run as sidecar service |
| `faster-whisper` + `ctranslate2` | ~150 MB | Run Whisper.cpp or faster-whisper as separate worker |
| `deepface` + `fer` + `tf-keras` | ~350 MB | Optional, behind feature flag — should be separate service |

**Net reduction potential:** ~4 GB of dependencies → main application is a lightweight HTTP orchestrator.

---

## Phase 3: Open-Source Modernization Strategy

> **Build a robust, decentralized stack that runs on any OS and supports any open-source model.**

**Duration:** 1 week  
**Risk:** 🟠 Medium — infrastructure changes  
**Dependency:** Phase 1 + 2 (must understand blast radius first)

---

### 3.1 — Containerization: Podman Over Docker

**Why Podman:**
- Daemonless — no root process managing containers
- Rootless by default — improved security for media processing
- Drop-in Docker CLI compatible (`alias docker=podman`)
- Integrates with systemd for managing background workers
- Fully open-source (no Docker Desktop licensing)

**Action:**
1. Replace `docker-compose.yaml` with `podman-compose.yaml` (or use Podman's native pod support)
2. Create a `Containerfile` (Podman's equivalent of Dockerfile, backward-compatible):

```dockerfile
# Containerfile — Lightweight query server
FROM python:3.12-slim-bookworm AS runtime

WORKDIR /app

# Only runtime deps — NO torch, NO transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY --chown=1000:1000 . .

USER 1000
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

ENTRYPOINT ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

3. Create separate containers for:
   - **api** — Query server (lightweight, ~1 GB image)
   - **worker** — Ingestion worker (heavy, ~8 GB image with models)
   - **qdrant** — Vector database
   - **redis** — Task broker + cache
   - **ollama** — Local LLM server
   - **neo4j** — Knowledge graph (optional)

---

### 3.2 — Database: PostgreSQL + pgvector (Consolidation Target)

**Current state:** 4 different storage backends:
- Qdrant (vector search)
- SQLite (`core/ingestion/jobs.py` — job persistence)
- Pickle files (`core/storage/keyword_index.py` — BM25)
- Neo4j (`core/storage/graph/` — knowledge graph, optional)

**Modernization path (incremental, not big-bang):**

| Phase | Action | Risk |
|---|---|---|
| **Now** | Keep Qdrant — it works. Add `pgvector` as a migration target. | 🟢 None |
| **Later** | Migrate SQLite job store → PostgreSQL | 🟡 Low |
| **Later** | Migrate BM25 pickle → PostgreSQL full-text search (`tsvector`) | 🟡 Low |
| **Future** | Evaluate Qdrant → `pgvector` migration for consolidation | 🟠 Medium |

> [!NOTE]
> **Do not rush this migration.** Qdrant is mature and performant for vector search. The immediate win is eliminating SQLite and pickle — not replacing Qdrant.

---

### 3.3 — Asynchronous Task Queue: Decouple Ingestion from API

**Current state:** Celery integration exists (`core/ingestion/celery_app.py`) but is incomplete. The Celery task in `core/ingestion/tasks.py` has 25 lines of TODO comments and uses `loop.run_until_complete()` inside a Celery worker — a known anti-pattern.

**Action:** Fix and activate the existing Celery integration:

1. **Fix the Celery task** — replace `loop.run_until_complete()` with proper `asyncio.run()`:

```python
# core/ingestion/tasks.py — FIXED
@shared_task(bind=True, name="core.ingestion.tasks.ingest_video_task")
def ingest_video_task(self, video_path: str, job_id: str):
    """Celery task to ingest a video file."""
    import asyncio
    from core.ingestion.pipeline import IngestionPipeline
    from core.ingestion.jobs import job_manager, JobStatus

    bind_context(trace_id=job_id, component="celery_worker")

    try:
        pipeline = IngestionPipeline()

        # Use asyncio.run() — creates and manages its own event loop
        asyncio.run(pipeline.process_video(Path(video_path), job_id=job_id))

        return {"status": "completed", "path": video_path}
    except Exception as exc:
        job_manager.update_job(job_id, status=JobStatus.FAILED, error=str(exc))
        raise self.retry(exc=exc, countdown=60, max_retries=3) from exc
    finally:
        clear_context()
```

2. **API ingest endpoint becomes non-blocking:**

```python
@router.post("/ingest")
async def ingest(request: IngestRequest):
    job_id = str(uuid4())
    job_manager.create_job(job_id, request.path)

    # Dispatch to Celery worker — returns immediately
    ingest_video_task.delay(request.path, job_id)

    return {"job_id": job_id, "status": "queued"}
```

3. **Pod composition:**
```yaml
# podman-compose.yaml
services:
  api:
    build: .
    ports: ["8000:8000"]
    depends_on: [qdrant, redis]
    # NO model loading — lightweight query server

  worker:
    build:
      context: .
      dockerfile: Containerfile.worker
    depends_on: [qdrant, redis, ollama]
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
    # HEAVY compute — loads Whisper, InsightFace, etc.

  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]

  qdrant:
    image: qdrant/qdrant:latest
    ports: ["6333:6333"]
    volumes:
      - qdrant_data:/qdrant/storage

  ollama:
    image: ollama/ollama:latest
    ports: ["11434:11434"]
    volumes:
      - ollama_models:/root/.ollama
```

---

### 3.4 — Model Agnosticism: HTTP Endpoints, Not In-Process Models

**Current state:** The application loads 15+ AI models directly into process memory:
- Whisper large-v3 via `faster-whisper` (3 GB VRAM)
- InsightFace via `insightface.app.FaceAnalysis()` (1.5 GB)
- SigLIP via `transformers.AutoModel` (1 GB)
- VLM via Ollama HTTP (already decoupled ✓)
- YOLO via `ultralytics.YOLO()` (1 GB)
- PyAnnote via `pyannote.audio.Pipeline()` (1.5 GB)

**Modernization target:** All model inference happens behind HTTP endpoints:

```python
# BEFORE: Tight coupling to model framework
from transformers import AutoModel
model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384")
embeddings = model.encode(image)

# AFTER: Model-agnostic HTTP call
async with httpx.AsyncClient() as client:
    response = await client.post(
        f"{settings.embedding_service_url}/encode",
        files={"image": image_bytes},
    )
    embeddings = response.json()["embedding"]
```

**Implementation priority:**
1. VLM captioning → already uses Ollama HTTP ✅
2. Text embeddings → move to dedicated embedding server (TEI or local)
3. Whisper → move to faster-whisper-server or WhisperX HTTP
4. Face detection → move to InsightFace HTTP microservice
5. YOLO → move to Triton or custom Flask service

> [!IMPORTANT]
> **Do not do this all at once.** Start with the VLM (already done) and text embeddings (highest ROI). Keep in-process models behind a `Protocol` interface so either path works.

---

## Phase 4: Decoupling & Removing Heavy Dependencies

> **Extract interfaces. Inject dependencies. Kill the ML framework bloat.**

**Duration:** 1–2 weeks  
**Risk:** 🟡 Low — additive changes with backward-compatible wrappers  
**Dependency:** Phase 2 (must know what to decouple)

---

### 4.1 — Extract Interfaces (Protocols)

**Current state:** `core/protocols.py` defines `EmbeddingProvider`, `SearchProvider`, `MediaProcessor` — but they are **not consistently used**. Most code depends on concrete classes directly.

`core/ports/processors.py` defines `AudioProcessor`, `VoiceProcessor`, `VisionAnalyzer`, `SceneDetector`, `VLMProcessor`, `FaceTracker` protocols — but the pipeline's mixin stages **don't type-hint against them**.

**Action:** Enforce protocol usage at construction boundaries:

```python
# core/ports/media_analyzer.py — NEW
from typing import Any, Protocol, runtime_checkable

@runtime_checkable
class MediaAnalyzerProtocol(Protocol):
    """Any LLM-backed media analyzer must implement this."""

    async def analyze_frame(
        self,
        image_path: str,
        prompt: str,
        schema: type | None = None,
    ) -> dict[str, Any]:
        """Analyze a single frame and return structured data."""
        ...

    async def generate_caption(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
    ) -> str:
        """Generate a natural language caption for a video segment."""
        ...

# Implementations:
# - OllamaMediaAnalyzer (uses Ollama HTTP — local, free)
# - GeminiMediaAnalyzer (uses Google API — cloud, paid)
# - VLLMMediaAnalyzer (uses vLLM server — local, fast)
# - MockMediaAnalyzer (for testing — returns canned data)
```

```python
# core/ports/transcription.py — NEW
@runtime_checkable
class TranscriptionProvider(Protocol):
    """Swappable transcription backend."""

    async def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
    ) -> list[dict]:
        """Transcribe audio and return segments with timestamps."""
        ...

# Implementations:
# - WhisperTranscriber (in-process faster-whisper)
# - WhisperHTTPTranscriber (calls whisper-server via HTTP)
# - MockTranscriber (for testing)
```

---

### 4.2 — Dependency Injection via FastAPI `Depends()`

**Current state:** `api/deps.py` provides `get_pipeline()` and `get_search_agent()` from `app.state` — minimal DI. Most routes access `request.app.state` directly. The MCP server (`core/agent/server.py`) uses module-level globals with `global _vector_db`.

**Action — Create a proper DI container:**

```python
# core/container.py — NEW
from dataclasses import dataclass, field
from typing import Protocol

class Container:
    """Dependency injection container. No framework magic — just explicit wiring."""

    def __init__(self, settings):
        self.settings = settings
        self._db: VectorDB | None = None
        self._search: SearchAgent | None = None
        self._ingestion: IngestionService | None = None

    @property
    def db(self) -> VectorDB:
        if self._db is None:
            from core.storage.db import VectorDB
            self._db = VectorDB(
                backend=self.settings.qdrant.backend,
                host=self.settings.qdrant.host,
                port=self.settings.qdrant.port,
            )
        return self._db

    @property
    def search_agent(self) -> SearchAgent:
        if self._search is None:
            from core.retrieval.agentic_search import SearchAgent
            self._search = SearchAgent(db=self.db)
        return self._search

    @property
    def ingestion_service(self) -> IngestionService:
        if self._ingestion is None:
            from core.services.ingestion_service import IngestionService
            self._ingestion = IngestionService(db=self.db)
        return self._ingestion

    def close(self):
        if self._db:
            self._db.close()
```

```python
# api/deps.py — UPDATED
from functools import lru_cache
from core.container import Container
from config import settings

@lru_cache(maxsize=1)
def get_container() -> Container:
    return Container(settings)

def get_db(container: Container = Depends(get_container)) -> VectorDB:
    return container.db

def get_search(container: Container = Depends(get_container)) -> SearchAgent:
    return container.search_agent
```

---

### 4.3 — Kill the ML Framework Bloat

**Action — Split `requirements.txt` into tiers:**

```
requirements-api.txt        # Query server (lightweight ~500 MB)
├── fastapi, uvicorn, httpx
├── qdrant-client
├── sentence-transformers (for text encoding only)
├── pydantic, loguru
└── (NO torch, NO transformers, NO paddle, NO insightface)

requirements-worker.txt     # Ingestion worker (heavy ~8 GB)
├── requirements-api.txt
├── torch, torchvision, torchaudio
├── faster-whisper
├── insightface
├── paddleocr
├── ultralytics
├── pyannote.audio
├── transformers
└── sam3

requirements-dev.txt        # Development tools
├── pytest, pytest-asyncio, pytest-cov
├── black, ruff, mypy, pyright
└── httpx (for TestClient)
```

**Split `pyproject.toml` optional dependencies:**

```toml
[project.optional-dependencies]
api = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.32.0",
    "httpx>=0.28.0",
    "qdrant-client>=1.12.0",
    "sentence-transformers>=3.3.0",
]
worker = [
    "torch>=2.5.1",
    "faster-whisper>=1.0.3",
    "insightface>=0.7.3",
    "paddleocr>=2.9.0",
    "pyannote-audio>=3.3.2",
    "ultralytics>=8.3.0",
]
dev = [
    "pytest>=9.0.2",
    "pytest-asyncio>=0.23.0",
    "black>=24.0.0",
    "ruff>=0.6.0",
]
```

---

### 4.4 — Implement the Strategy Pattern for Media Ingestion

**Current state:** `IngestionPipeline` processes every file through the same mixin stages. Media type is a parameter that conditionally skips stages (`if media_type == "audio": skip frame stage`), buried deep in the mixin code.

**Action — Route by MIME type to dedicated handlers:**

```python
# core/ingestion/handlers.py — NEW
from typing import Protocol

class MediaHandler(Protocol):
    """Strategy interface for media-type-specific processing."""
    async def process(self, path: Path, ctx: PipelineContext) -> None: ...

class VideoHandler:
    """Full pipeline: audio + voice + frames + scenes + events."""
    async def process(self, path: Path, ctx: PipelineContext) -> None:
        await AudioStage().execute(ctx)
        await VoiceStage().execute(ctx)
        await FrameStage().execute(ctx)
        await SceneStage().execute(ctx)
        await AudioEventsStage().execute(ctx)

class AudioHandler:
    """Audio-only: transcription + voice + audio events. No frame processing."""
    async def process(self, path: Path, ctx: PipelineContext) -> None:
        await AudioStage().execute(ctx)
        await VoiceStage().execute(ctx)
        await AudioEventsStage().execute(ctx)

class ImageHandler:
    """Single-frame analysis: face + VLM + OCR. No temporal processing."""
    async def process(self, path: Path, ctx: PipelineContext) -> None:
        await SingleFrameStage().execute(ctx)

# Factory
def get_handler(mime_type: str) -> MediaHandler:
    """Route to the correct handler by MIME type."""
    if mime_type.startswith("video/"):
        return VideoHandler()
    elif mime_type.startswith("audio/"):
        return AudioHandler()
    elif mime_type.startswith("image/"):
        return ImageHandler()
    else:
        raise ValueError(f"Unsupported MIME type: {mime_type}")
```

---

## Phase 5: The Refactoring Engine

> **Transform "vibe-coded" modules into production-ready software. One module at a time.**

**Duration:** 2–3 weeks  
**Risk:** 🟠 Medium — touching core logic  
**Dependency:** Phase 4 (interfaces must exist before refactoring implementations)

---

### 5.1 — Module Refactoring Order

Do NOT refactor everything at once. Follow this order (highest ROI first):

| Order | Module | LOC | Why First |
|-------|--------|-----|-----------|
| 1 | `config.py` | 1,054 | Blocks everything — 54 consumers, torch import penalty |
| 2 | `core/storage/db.py` | 1,667 | God class — blocks service layer and testing |
| 3 | `core/retrieval/agentic_search.py` | 1,508 | God class — blocks search accuracy improvements |
| 4 | `core/ingestion/pipeline.py` + stages | 3,448 | Mixin soup — blocks task decoupling |
| 5 | `core/processing/transcriber.py` | 1,109 | Blocking subprocess calls + fixed chunking |
| 6 | `core/processing/identity.py` | 1,087 | Large class, global state |
| 7 | `core/retrieval/reranker.py` | 680 | Performance-critical for search accuracy |

---

### 5.2 — Refactoring Constraints (Per Module)

For each module, apply these constraints strictly:

#### Standards
- **Google Python Style Guide** — docstrings, type hints for all parameters and return types
- **Remove all dead/commented-out code** — no `# TODO`, no `# FIXME`, no commented `elif` blocks
- **Remove all `print()` calls** — use structured logging via `core/utils/logger.py`

#### Decoupling
- **Strategy Pattern** — for interchangeable implementations (LLM providers, embedding models, transcription backends)
- **Factory Pattern** — for creating the right handler based on media type or config
- **All external service calls behind Protocol interfaces** — any open-source model can be swapped in

#### Performance
- **File operations use streaming/generators** — never load entire media files into memory
- **Wrap blocking calls** in `asyncio.to_thread()` or use `asyncio.create_subprocess_exec()`
- **Lazy model loading** — every heavy model behind a `@property` that loads on first access

#### Modernization
- **Replace synchronous I/O with `async/await`** where the caller is already async
- **Strip unused dependencies** — if a module only calls an HTTP endpoint, it doesn't need `torch`
- **Read all configuration from environment variables** — no hardcoded paths, host/port, or API keys

#### Deployment Context
- **OS-agnostic** — no Windows-specific `asyncio.WindowsSelectorEventLoopPolicy()` in core logic (move to entry point guard)
- **Container-ready** — paths come from `$DATA_DIR`, `$MODEL_CACHE_DIR`, etc.
- **No module-level side effects** — no `os.environ` mutation, no `sys.modules` patching on import

---

### 5.3 — Detailed Refactoring: `config.py`

**The single most impactful refactor in the entire project.**

**Current problems (lines referenced):**
- Line 10: `import torch` — module-level, runs on every import
- Line 80: `_HW_PROFILE = get_hardware_profile()` — runs `torch.cuda.is_available()` at import time
- Lines 1040–1054: `settings = Settings()` followed by `os.environ` mutations
- 1,054 lines in one file with 150+ settings

**Refactored structure:**

```
config/
├── __init__.py          # Re-exports `settings` for backward compatibility
├── settings.py          # Pydantic BaseSettings — NO torch, NO side effects
├── hardware.py          # Lazy hardware detection (deferred torch import)
├── groups/
│   ├── qdrant.py        # QdrantSettings
│   ├── llm.py           # LLMSettings
│   ├── embedding.py     # EmbeddingSettings
│   ├── search.py        # SearchSettings
│   ├── processing.py    # ProcessingSettings (face, voice, scene thresholds)
│   └── security.py      # SecuritySettings
└── initialize.py        # Explicit init() — env mutations, cache setup
```

**Key changes:**

```python
# config/__init__.py — backward-compatible re-export
from config.settings import settings  # noqa: F401
from config.initialize import initialize  # noqa: F401

# config/settings.py — NO torch import, NO side effects
from pydantic_settings import BaseSettings
from config.groups.qdrant import QdrantSettings
from config.groups.llm import LLMSettings
# ... etc

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_nested_delimiter="__")

    qdrant: QdrantSettings = QdrantSettings()
    llm: LLMSettings = LLMSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    search: SearchSettings = SearchSettings()
    processing: ProcessingSettings = ProcessingSettings()
    security: SecuritySettings = SecuritySettings()

    # Backward-compatible aliases (deprecate over time)
    @property
    def qdrant_host(self) -> str:
        return self.qdrant.host

settings = Settings()

# config/hardware.py — lazy torch import
import functools

@functools.lru_cache(maxsize=1)
def get_hardware_profile() -> dict:
    """Detect GPU/CPU capabilities. Torch imported ONLY here, ONLY on first call."""
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_mem / (1024**3)
            # ... detection logic ...
    except ImportError:
        pass
    return {"device": "cpu", "vram_gb": 0, ...}

# config/initialize.py — explicit init, called by entry points only
_initialized = False

def initialize():
    """Call once from entry points. Sets env vars, cache dirs, etc."""
    global _initialized
    if _initialized:
        return
    _initialized = True

    import os
    from config.settings import settings
    os.environ["HF_HOME"] = str(settings.model_cache_dir / "huggingface")
    os.environ["TORCH_HOME"] = str(settings.model_cache_dir / "torch")
    # ...
```

---

### 5.4 — Detailed Refactoring: `VectorDB`

See [CURRENT_ARCHITECTURE.md — Phase 3 Task 3.1](./CURRENT_ARCHITECTURE.md#task-31--decompose-vectordb-1667-lines--5-services) for the full decomposition plan. Summary:

- Break 4 repository base classes into 5 composed service classes
- `VectorDB` becomes a thin facade with delegating properties
- `VisualEncoder` becomes a lazy `@property`
- `ensure_all_collections()` deferred to first write operation

---

### 5.5 — Detailed Refactoring: `IngestionPipeline`

Replace mixin inheritance with explicit stage composition:

```python
class IngestionPipeline:
    def __init__(self, db: VectorDB, stages: list[PipelineStage] | None = None):
        self.db = db
        self.stages = stages or self._default_stages()

    async def process_video(self, path: Path, **kwargs) -> str:
        ctx = PipelineContext(video_path=path, ...)
        for stage in self.stages:
            if await stage.should_run(ctx):
                logger.info(f"Executing stage: {stage.name}")
                await stage.execute(ctx, self.db)
        return ctx.job_id
```

Design patterns applied:
- **Strategy Pattern** — each `PipelineStage` is a strategy for a processing step
- **Template Method** — `should_run()` + `execute()` on each stage
- **Chain of Responsibility** — stages execute in sequence, each deciding whether to proceed

---

## Phase 6: Security Hardening

> **An indexer that parses untrusted media is an RCE vector. A search engine that passes user queries to LLMs is a prompt injection vector.**

**Duration:** 3–5 days  
**Risk:** 🔴 Critical — security is non-negotiable  
**Dependency:** Phase 4 (need interfaces for sanitization injection)

---

### 6.1 — Sandbox the Media Processing Pipeline

**Current state:** `core/ingestion/sandbox.py` provides a `MediaSandbox` class that monitors subprocess memory/timeout. But:
1. It only monitors `psutil.Process.memory_info()` — no namespace isolation
2. FFmpeg subprocesses run as the same user with full filesystem access
3. No `seccomp` profile, no Linux namespace isolation, no chroot
4. The sandbox is **not used** for all subprocess calls — `subprocess.run()` is called directly in `transcriber.py:238`, `extractor.py:340`, `prober.py:34`, `scene_detector.py:151`

**Action — Enforce sandboxed execution for all media parsing:**

```python
# core/security/media_sandbox.py — UPGRADED
import asyncio
import os
import tempfile
from pathlib import Path

class MediaSandbox:
    """Process untrusted media files in an isolated environment."""

    def __init__(
        self,
        max_memory_mb: int = 4096,
        timeout_seconds: int = 3600,
        allowed_commands: frozenset[str] = frozenset({"ffmpeg", "ffprobe"}),
    ):
        self.max_memory_mb = max_memory_mb
        self.timeout_seconds = timeout_seconds
        self.allowed_commands = allowed_commands

    async def run(self, cmd: list[str]) -> tuple[int, bytes, bytes]:
        """Run a command in a sandboxed subprocess."""
        binary = Path(cmd[0]).name
        if binary not in self.allowed_commands:
            raise SecurityError(f"Command '{binary}' not in allowlist: {self.allowed_commands}")

        # Validate no path traversal in arguments
        for arg in cmd[1:]:
            if ".." in arg or arg.startswith("/etc") or arg.startswith("/proc"):
                raise SecurityError(f"Suspicious argument: {arg}")

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=10 * 1024 * 1024,  # 10MB pipe buffer
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.timeout_seconds,
            )
        except asyncio.TimeoutError:
            proc.kill()
            raise TimeoutError(f"Command timed out after {self.timeout_seconds}s")

        return proc.returncode, stdout, stderr

# Global sandbox instance
MEDIA_SANDBOX = MediaSandbox()
```

**Then replace all direct `subprocess.run()` calls with `MEDIA_SANDBOX.run()`:**

```python
# core/processing/extractor.py — BEFORE:
subprocess.run(cmd, timeout=timeout, check=True, capture_output=True)

# AFTER:
from core.security.media_sandbox import MEDIA_SANDBOX
returncode, stdout, stderr = await MEDIA_SANDBOX.run(cmd)
if returncode != 0:
    raise ExtractionError(f"FFmpeg failed: {stderr.decode()}")
```

---

### 6.2 — Unify and Harden Prompt Injection Defense

**Current state:** Two competing sanitizers:
1. `core/security/query_sanitizer.py` → `QuerySanitizer` — semantic embedding similarity
2. `core/security/sanitizer.py` → `PromptSanitizer` — regex + LLM-based scoring

**Problems:**
- Neither is consistently integrated into the main search path
- `QuerySanitizer` depends on `VectorDB` for embeddings (circular concern)
- `PromptSanitizer` calls the LLM to validate queries — the LLM itself could be fooled
- Both are module-level singletons

**Action — Create a layered defense pipeline:**

```python
# core/security/defense.py — NEW: Defense-in-depth pipeline
class QueryDefensePipeline:
    """Multi-layer prompt injection defense."""

    def __init__(self, llm: LLMInterface | None = None):
        self.layers = [
            LengthValidator(max_length=1500),
            CharacterDensityValidator(max_special_ratio=0.4),
            RegexPatternValidator(),  # From existing PromptSanitizer patterns
            # LLM-based scoring is the LAST layer (expensive, fallible)
        ]
        if llm:
            self.layers.append(LLMIntentValidator(llm))

    async def validate(self, query: str) -> tuple[bool, str]:
        """Run query through all defense layers. Returns (is_safe, reason)."""
        for layer in self.layers:
            is_safe, reason = await layer.validate(query)
            if not is_safe:
                log.warning(f"[Security] Query blocked by {layer.__class__.__name__}: {reason}")
                return False, reason
        return True, "Passed all security layers"
```

**Integration point — inject into `SearchAgent` before query parsing:**

```python
# core/retrieval/agentic_search.py
class SearchAgent:
    async def search(self, query: str, **kwargs):
        # SECURITY: Validate before ANY LLM interaction
        is_safe, reason = await self.defense.validate(query)
        if not is_safe:
            return {"error": reason, "results": [], "blocked": True}

        parsed = await self.parser.parse_query(query)
        # ... continue with search
```

---

### 6.3 — Input Validation on API Boundaries

**Action — Add Pydantic validators for all API request models:**

```python
# api/schemas.py — HARDENED
from pydantic import BaseModel, Field, field_validator
import re

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1500)
    limit: int = Field(default=20, ge=1, le=200)

    @field_validator("query")
    @classmethod
    def sanitize_query(cls, v: str) -> str:
        # Strip null bytes and control characters
        v = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", v)
        # Strip excessive whitespace
        v = " ".join(v.split())
        return v.strip()

class IngestRequest(BaseModel):
    path: str = Field(..., min_length=1, max_length=4096)

    @field_validator("path")
    @classmethod
    def validate_path(cls, v: str) -> str:
        # Prevent path traversal
        if ".." in v:
            raise ValueError("Path traversal not allowed")
        return v
```

---

## Phase 7: Agentic Workflow & Search Accuracy Overhaul

> **Stop asking one prompt to do everything. Use a multi-step reasoning loop with specific tools for visual vs. audio retrieval.**

**Duration:** 1–2 weeks  
**Risk:** 🟠 Medium — touches core search logic  
**Dependency:** Phase 5 (search must be decomposed first)

---

### 7.1 — Agentic Query Decomposition (Replace Monolithic Prompt)

**Current state:** `SearchAgent.parse_query()` sends the user's query to a single LLM call with a massive system prompt. The LLM must simultaneously:
- Extract visual keywords
- Identify person names
- Determine temporal context
- Classify intent
- Expand synonyms
- Resolve identities

This is the "monolithic Swiss Knife" anti-pattern. Complex multi-part queries break because one prompt can't handle 6 tasks.

**Action — Multi-agent decomposition with specialized tools:**

```python
# core/retrieval/agents/decomposer.py — NEW
class QueryDecomposer:
    """Agent 1: Break complex queries into atomic sub-queries."""

    async def decompose(self, query: str) -> list[SubQuery]:
        """
        Input: "Show me when Prakash scored a strike while wearing the blue shirt"
        Output: [
            SubQuery(type="identity", target="Prakash"),
            SubQuery(type="action", target="scored a strike"),
            SubQuery(type="visual", target="wearing blue shirt"),
        ]
        """
        # Single focused LLM call — decomposition only
        result = await self.llm.generate_structured(
            schema=DecomposedQuery,
            prompt=f"Break this media search query into atomic sub-queries:\n{query}",
            system_prompt=DECOMPOSITION_PROMPT,
        )
        return result.sub_queries

# core/retrieval/agents/router.py — NEW
class QueryRouter:
    """Agent 2: Route each sub-query to the best retrieval tool."""

    TOOL_MAP = {
        "identity": IdentitySearchTool,    # Face/voice cluster lookup
        "action": SceneletSearchTool,      # Temporal action search
        "visual": FrameSearchTool,         # Visual similarity search
        "dialogue": DialogueSearchTool,    # Transcript search
        "audio": AudioEventSearchTool,     # CLAP audio search
        "temporal": TemporalFilterTool,    # Time-based filtering
    }

    async def route(self, sub_queries: list[SubQuery]) -> list[ToolResult]:
        tasks = []
        for sq in sub_queries:
            tool_class = self.TOOL_MAP.get(sq.type)
            if tool_class:
                tasks.append(tool_class(self.db).search(sq.target))
        return await asyncio.gather(*tasks)

# core/retrieval/agents/synthesizer.py — NEW
class ResultSynthesizer:
    """Agent 3: Fuse results from multiple tools and rerank."""

    async def synthesize(
        self,
        query: str,
        tool_results: list[ToolResult],
    ) -> list[SearchResult]:
        # Temporal intersection — find moments where ALL sub-queries match
        fused = self._temporal_intersect(tool_results)
        # Rerank the fused results
        return await self.reranker.rerank(fused, query)
```

**This replaces the monolithic `SearchAgent.sota_search()` with a 3-agent pipeline:**

```
User Query
    │
    ▼
┌──────────────────┐
│ QueryDecomposer  │ → Break into atomic sub-queries
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   QueryRouter    │ → Route each sub-query to specialized tool
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ ResultSynthesizer│ → Fuse, intersect, rerank
└────────┬─────────┘
         │
         ▼
    Final Results
```

---

### 7.2 — Semantic Boundary Chunking (Replace Fixed Intervals)

**Current state:**
- Audio chunked at fixed 600s intervals (`config.py:846 chunk_duration_seconds`)
- CLAP audio events at fixed 5s windows
- Scenelets at fixed 10s windows with 5s stride

**Action — Use semantic boundaries everywhere:**

#### Audio Chunking: Use VAD + Scene Boundaries

```python
# core/processing/semantic_chunker.py — NEW
class SemanticAudioChunker:
    """Chunk audio at natural speech boundaries, not fixed intervals."""

    async def chunk(self, audio_path: Path, scene_boundaries: list[float]) -> list[AudioChunk]:
        """
        1. Run VAD (Voice Activity Detection) to find speech segments
        2. Align chunks with scene boundaries from TransNet
        3. Merge short segments, split long ones at sentence boundaries
        """
        # Use PyAnnote VAD for speech boundary detection
        from pyannote.audio import Pipeline
        vad = Pipeline.from_pretrained("pyannote/voice-activity-detection")
        speech_segments = vad(audio_path)

        # Align with scene boundaries
        chunks = []
        for scene_start, scene_end in scene_boundaries:
            scene_speech = [s for s in speech_segments if s.start >= scene_start and s.end <= scene_end]
            if scene_speech:
                chunks.append(AudioChunk(
                    start=scene_start,
                    end=scene_end,
                    speech_segments=scene_speech,
                ))

        return chunks
```

#### Scenelet Building: Use Semantic Scene Boundaries

```python
# Instead of fixed 10s/5s stride:
scenelets = build_scenelets(frames, window_size=10, stride=5)  # ← FIXED

# Use scene transitions as boundaries:
scenelets = build_scenelets_semantic(
    frames,
    scene_boundaries=detected_scenes,
    max_duration=30,   # Split long scenes at 30s
    min_duration=3,    # Merge tiny scenes into neighbors
)
```

---

### 7.3 — Add a Reranker Layer for Multi-Part Queries

**Current state:** `core/retrieval/reranker.py` implements a `RerankingCouncil` with 3 models (cross-encoder, BGE, VLM). But it's **optional** and gated behind `use_reranking=True`. Most search calls skip it.

**Action — Make reranking mandatory for complex queries:**

```python
# core/retrieval/agentic_search.py
async def search(self, query: str, **kwargs):
    parsed = await self.parser.parse_query(query)

    # Determine query complexity
    complexity = self._score_complexity(parsed)
    # Simple: "red car" → 1 keyword, no identity, no temporal
    # Complex: "Prakash bowling while wearing blue" → identity + action + visual

    results = await self._multi_modal_search(parsed)

    if complexity > COMPLEXITY_THRESHOLD:
        # MANDATORY reranking for complex queries
        results = await self.council.council_rerank(query, results, top_k=20)
    else:
        # Lightweight scoring for simple queries
        results = self._rrf_fuse(results)

    return results
```

---

### 7.4 — Context-Aware Cross-Modal Embeddings

**Current state:** Each modality generates embeddings independently. The text embedding for a scene doesn't know about the audio in that scene, and vice versa.

**Action — Build cross-modal context before embedding:**

```python
# Instead of embedding each modality independently:
text_emb = encode("person walking")
visual_emb = encode_image(frame)

# Build context-enriched text before embedding:
enriched_text = (
    f"Visual: {vlm_caption}. "
    f"Dialogue: {transcript_segment}. "
    f"Audio: {audio_events}. "
    f"Faces: {detected_faces}. "
    f"OCR: {detected_text}."
)
text_emb = encode(enriched_text)
# This single embedding captures cross-modal context
```

---

## Phase 8: Testing, CI, & Production Readiness

> **No refactoring without regression safety.**

**Duration:** 1 week  
**Risk:** 🟢 None — additive  
**Dependency:** Phase 5 (refactored modules are testable)

---

### 8.1 — Testing Foundation

**Delete root `conftest.py`** — it mocks `torch` globally via `sys.modules`, which is an anti-pattern that breaks any test needing real functionality.

**Create proper test structure:**
```
tests/
├── conftest.py              # Shared fixtures (mock_qdrant, mock_llm, etc.)
├── unit/
│   ├── test_config.py       # Settings validation, env overrides
│   ├── test_query_parser.py # Decomposition with mocked LLM
│   ├── test_result_processor.py # RRF fusion, dedup, scoring
│   ├── test_text_encoder.py # Embedding cache, model prefix logic
│   ├── test_pipeline.py     # Stage execution order with mock stages
│   ├── test_job_manager.py  # SQLite job CRUD
│   ├── test_sanitizer.py    # Prompt injection defense
│   └── test_media_sandbox.py # Command allowlisting
├── integration/
│   ├── test_api_health.py   # FastAPI TestClient
│   ├── test_api_search.py   # Search endpoint with mocked agent
│   ├── test_api_ingest.py   # Ingest endpoint with mocked pipeline
│   └── test_celery_tasks.py # Task dispatch with mocked worker
└── e2e/
    ├── test_golden_path.py  # Full ingestion + search (needs GPU)
    └── test_search_quality.py # Precision/recall metrics
```

---

### 8.2 — CI Pipeline

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv sync --extra dev
      - run: uv run ruff check .
      - run: uv run ruff format --check .

  test-unit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv sync --extra dev --extra api
      - run: uv run pytest tests/unit/ -v --tb=short --cov=core

  test-integration:
    runs-on: ubuntu-latest
    services:
      qdrant:
        image: qdrant/qdrant:latest
        ports: ["6333:6333"]
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv sync --extra dev --extra api
      - run: uv run pytest tests/integration/ -v --tb=short
```

---

### 8.3 — Production Readiness Checklist

| Item | Status | Action |
|---|---|---|
| Fix Dockerfile entrypoint | 🔴 `python main.py` → `uvicorn api.server:app` | Phase 8 |
| Remove debug prints | 🔴 12 `print("DEBUG:...")` in `api/server.py` | Phase 5 |
| Pin Python version | 🟡 3.11 in Dockerfile vs 3.12 in ruff config | Align to 3.12 |
| Add `.dockerignore` | 🔴 Missing — builds include test data (115 MB) | Add now |
| Health check endpoint | 🟡 Exists but basic | Add Qdrant connectivity check |
| Structured logging | 🟡 Loguru exists but mixed with `print()` | Phase 5 |
| Secrets management | 🟡 `.env` file only | Document env var requirements |
| Rate limiting | 🔴 None | Add FastAPI `slowapi` middleware |
| CORS configuration | 🟡 Permissive `allow_origins=["*"]` | Restrict to frontend origin |

---


---

## Phase 9 — Ultra-Long Video Scalability & Universal LLMs

**Goal:** Enable 12+ hour video processing on commodity laptop hardware. Introduce universal LLM switchability and robust contextual subtitle transcription.

**Duration:** 1 week  
**Risk:** 🟠 Medium — Changes fundamental extraction assumptions  
**Dependency:** Phase 4 (Interfaces)

---

### Task 9.1 — Dynamic Keyframe Extraction & Adaptive Sampling

**Why:** A 12-hour video at 1 FPS produces 43,200 frames. Processing this through SigLIP, InsightFace, YOLO, and VLMs will destroy a laptop's RAM and take days.

**Action:** Replace fixed-interval extraction with adaptive keyframe sampling.
1. Use `scenedetect` (PySceneDetect) or FFmpeg I-frame extraction (`select='eq(pict_type,I)'`) to find true shot boundaries.
2. For short scenes (< 10s), extract only the center frame.
3. For long scenes, use adaptive sampling (1 frame every 10 seconds).
4. **Impact:** Reduces frame count from 43,200 to ~1,500-3,000 for a 12-hour video (95% reduction).

---

### Task 9.2 — Reverse Frame Interpolation & Combining (Scenelets)

**Why:** Analyzing individual frames sequentially wastes compute on visually similar frames.

**Action:** 
1. Group the adaptive keyframes into "Scenelets".
2. Send the start, middle, and end frames of a Scenelet *simultaneously* to the VLM (Gemini, Claude, or LLaVA).
3. Prompt the VLM to act as the interpolator: "Based on these keyframes, describe the continuous action and transitions."
4. **Impact:** Divides VLM API calls and compute by a further 3x, while capturing temporal context better than single-frame descriptions.

---

### Task 9.3 — Universal LLM Switchability (LiteLLM / Protocol)

**Why:** Hardcoding Ollama or Gemini logic prevents users from using Claude, OpenAI, or local vLLM endpoints.

**Action:**
1. Integrate `litellm` OR enforce a strict `LLMInterface` protocol.
2. Remove provider-specific logic from the core pipeline.
3. Config becomes fully dynamic:
   ```env
   LLM_PROVIDER=anthropic
   LLM_MODEL=claude-3-5-sonnet-20240620
   LLM_API_KEY=sk-...
   ```
4. Supports Ollama (`ollama/llama3.1`), OpenAI, Anthropic, Gemini, and generic OpenAI-compatible endpoints seamlessly.

---

### Task 9.4 — ROVER-Style Subtitle Voting & Contextual Correction

**Why:** Standard local Whisper (especially smaller models suitable for laptops) makes phonetic mistakes (e.g., "recognize speech" vs "wreck a nice beach").

**Action:** 
1. Run a fast, laptop-friendly ASR (e.g., `distil-whisper`).
2. Pass the raw transcripts + the VLM Scenelet descriptions into the Universal LLM.
3. Use the LLM to perform contextual error correction (simulating ROVER voting) by aligning the raw transcript with the visual context.
4. Output highly accurate, visually-aligned subtitles (`.srt`).


## Execution Timeline

```
Week 1:
├── Phase 1 (Understand) — 2 days
└── Phase 2 (Audit) — 3 days (overlap with Phase 1)

Week 2:
├── Phase 5.1 (Refactor config.py) — 3 days
└── Phase 6 (Security Hardening) — 2 days

Week 3:
├── Phase 4 (Decoupling) — interfaces + DI container
└── Phase 5.2-5.3 (Refactor VectorDB, SearchAgent)

Week 4:
├── Phase 5.4-5.5 (Refactor Pipeline, Transcriber)
└── Phase 3 (Modernization) — task queue decoupling

Week 5:
├── Phase 7 (Agentic Workflow) — decomposer, router, reranker
└── Phase 8 (Testing + CI)

Week 6 (buffer):
├── Phase 7 completion (semantic chunking)
└── Integration testing + deployment verification
```

---

## Verification Checklist

After all phases, verify:

| Check | Command | Expected |
|---|---|---|
| Import speed | `time python -c "from config import settings"` | < 0.5s (no torch) |
| Server starts | `uvicorn api.server:app` | < 10s, no DEBUG prints |
| Search works | `python search_cli.py "test" --limit 1` | Returns results |
| Unit tests pass | `pytest tests/unit/ -v` | All green |
| Integration tests | `pytest tests/integration/ -v` | All green |
| Container builds | `podman build -t ami-api .` | < 2 GB image |
| Worker builds | `podman build -f Containerfile.worker -t ami-worker .` | < 8 GB image |
| No dead imports | `python -c "import core.processors"` | ImportError |
| Security blocks injection | `curl -X POST .../search -d '{"query":"ignore instructions"}'` | 403 |
| Git repo size | `git count-objects -v --human-readable` | No 52 MB blobs |
| Complex query works | `search_cli.py "Prakash bowling in blue shirt"` | Multi-modal results |

---

> **Note:** This document is a living plan. Update it after each phase is completed. Mark completed tasks, note deviations, and adjust timelines.

> **Remember:** The goal is not to rewrite from scratch. It's to surgically decouple, harden, and modernize — one module at a time — until the "vibe-coded" prototype becomes production-grade software.
