# AI-Media-Indexer — Current Architecture Analysis

> **Analysis Date:** 2026-07-12  
> **Codebase:** ~56,574 lines of Python across 197 files  
> **Frontend:** React + TypeScript + MUI + TailwindCSS (Vite)  
> **Principle:** This document describes only what **actually exists**. No invented architecture.

---

## Table of Contents

- [Executive Summary](#executive-summary)
- [Directory Structure](#directory-structure)
- [Runtime Architecture](#runtime-architecture)
- [Module Architecture](#module-architecture)
- [Execution Flow](#execution-flow)
- [Dependency Graph](#dependency-graph)
- [Performance Bottlenecks](#performance-bottlenecks)
- [Scalability Problems](#scalability-problems)
- [Design Problems](#design-problems)
- [Testing](#testing)
- [Technical Debt](#technical-debt)
- [Refactor Priority](#refactor-priority)

---

## Executive Summary

### What the Project Does

AI-Media-Indexer is a **multimodal video understanding and search system**. It ingests video files, extracts rich structured metadata (transcription, face detection, object detection, scene detection, OCR, audio events, VLM captioning), stores it as vector embeddings in Qdrant, and provides semantic search with LLM-powered query expansion, reranking, and identity resolution. It exposes a FastAPI REST API, an MCP tool server for LLM agents, an A2A (Agent-to-Agent) protocol server, multiple CLI interfaces, and a React web frontend.

### Primary Goals

1. Index any media file with SOTA AI models (Whisper, InsightFace, SigLIP, CLAP, YOLO, PaddleOCR, InternVideo, etc.)
2. Enable natural language search across visual, dialogue, audio, and identity modalities
3. Support face/voice identity clustering with Human-in-the-Loop (HITL) correction
4. Provide agentic search with LLM query decomposition and multi-vector fusion
5. Build a knowledge graph linking entities, scenes, and identities

### Current Strengths

| Strength | Details |
|---|---|
| **Model breadth** | Integrates 15+ SOTA AI models (Whisper, InsightFace, SigLIP, CLAP, YOLO, PaddleOCR, InternVideo, TransNet, PyAnnote, etc.) |
| **Multi-modal search** | RRF fusion across 7 modalities (scenes, frames, scenelets, voice, dialogue, audio, metadata) |
| **Hardware adaptivity** | Auto-detects VRAM and adjusts batch sizes/parallelism via `HardwareProfile` |
| **Protocol surface** | FastAPI REST, MCP stdio, A2A HTTP — three integration pathways |
| **VRAM management** | `ResourceArbiter` tracks model VRAM budgets with lazy load/unload lifecycle |
| **Port/adapter separation** | `core/ports/` defines protocol interfaces for DI in the ingestion pipeline |
| **Prompt externalization** | All LLM prompts stored as text files in `prompts/` — no hardcoded strings |
| **Job persistence** | SQLite-backed job manager with crash recovery and pause/resume |
| **Domain value objects** | `VideoPath`, `Timestamp`, `ClusterId`, `JobId` reduce primitive obsession |

### Major Weaknesses

| Weakness | Severity |
|---|---|
| **God config module** — `config.py` is 1,054 lines with 150+ settings, module-level `torch` import, module-level side effects | 🔴 Critical |
| **God classes** — `VectorDB` (1,667 lines), `SearchAgent` (1,508 lines), `IngestionPipeline` (1,056 lines) | 🔴 Critical |
| **Duplicate packages** — `core/processing/` (37 files) vs `core/processors/` (4 files) overlap responsibilities | 🔴 Critical |
| **Duplicate orchestrators** — `core/orchestration/orchestrator.py` and `core/orchestration/agent_graph.py` define competing `MultiAgentOrchestrator` classes | 🟠 High |
| **Pervasive global singletons** — 20+ `global` keyword usages, module-level singletons everywhere | 🟠 High |
| **Config imported at module level by 54+ files** — creates an import-time torch initialization bottleneck | 🟠 High |
| **No unit tests** — only integration/e2e scripts in `tests/`, root `conftest.py` mocks torch globally | 🟡 Medium |
| **52MB YOLO model committed to git** — `yolov8m.pt` duplicated in root AND `api/` directory | 🟡 Medium |
| **Dead code** — `core/processors/` package imports non-existent modules (`core.ingestion.diarization`, `core.ingestion.faces`, `core.ingestion.vision`) | 🟠 High |
| **Requirements.txt drift** — hardcoded Windows path (`-e file:///D:/AI-Media-Indexer`), version conflicts with `pyproject.toml` | 🟡 Medium |
| **Print-driven debugging** — `api/server.py` has 12 `print("DEBUG: ...")` statements in production code | 🟡 Medium |

---

## Directory Structure

```
AI-Media-Indexer/
├── api/                    # FastAPI web server + REST routes
│   ├── routes/             # 17 route modules (search, ingest, faces, etc.)
│   ├── server.py           # App factory, lifespan, middleware
│   ├── deps.py             # FastAPI DI (get_pipeline, get_search_agent)
│   ├── schemas.py          # Pydantic request/response models
│   └── yolov8m.pt          # ⚠ DUPLICATE 52MB model file
├── core/                   # Main application logic
│   ├── agent/              # MCP server + A2A handler (6 files)
│   ├── domain/             # Pydantic schemas + value objects (2 files)
│   ├── ingestion/          # Ingestion pipeline + stages (10 files + stages/)
│   │   └── stages/         # Mixin stages: audio, voice, frames, scenes (5 files)
│   ├── knowledge/          # Knowledge graph schemas + builder (4 files)
│   ├── llm/                # Core LLM factories (text + VLM) (3 files)
│   ├── manipulation/       # Video inpainting/editing pipeline (5 files)
│   ├── orchestration/      # Multi-agent routing (3 files)
│   ├── ports/              # Protocol interfaces for DI (2 files)
│   ├── processing/         # ⚠ 37 AI processor modules (LARGEST package)
│   ├── processors/         # ⚠ DUPLICATE package — 4 files, broken imports
│   ├── retrieval/          # Search, reranking, RAG, hybrid (14 files)
│   ├── security/           # Query + input sanitization (2 files)
│   ├── storage/            # Qdrant DB, encoder, repositories (15 files)
│   │   ├── graph/          # Neo4j graph managers (6 files)
│   │   └── repositories/   # Repository pattern for DB (6 files, LARGE)
│   ├── tools/              # Privacy tool for manipulation (1 file)
│   ├── tracking/           # SAM3 object tracker (1 file)
│   └── utils/              # Utilities (18 files — DUMPING GROUND)
├── llm/                    # Top-level LLM abstraction layer
│   ├── interface.py        # ABC for all LLM providers
│   ├── factory.py          # Factory for Ollama/Gemini
│   ├── gemini.py           # Gemini adapter
│   └── ollama.py           # Ollama adapter (610 lines)
├── tools/                  # Dev/ops tools (convert, diagnose, reset, etc.)
├── scripts/                # Maintenance scripts (cleanup, lint, health)
├── tests/                  # Integration test scripts (no real unit tests)
├── test/                   # Raw test media files (115MB)
├── prompts/                # 19 LLM prompt templates (text files)
├── web/                    # React frontend (Vite + MUI + Tailwind)
├── docs/                   # Documentation (partially stale)
├── config.py               # ⚠ GOD CONFIG — 1,054 lines, module-level side effects
├── main.py                 # CLI entrypoint for single video ingestion
├── agent_main.py           # CLI entrypoint for agent orchestration
├── agent_cli.py            # Interactive MCP agent CLI (Ollama + tools)
├── search_cli.py           # CLI entrypoint for search queries
├── conftest.py             # Root pytest config (mocks torch globally)
├── yolov8m.pt              # ⚠ 52MB YOLO model (should not be in git)
├── start.sh / start.ps1    # Platform startup scripts
├── Dockerfile              # CPU production image
├── Dockerfile.cpu          # CPU-only image
├── Dockerfile.cuda         # CUDA GPU image
├── docker-compose.yaml     # Full stack (Qdrant, Redis, Langfuse, etc.)
├── docker-compose.lite.yaml # Minimal stack
├── docker-compose.graph.yaml # Neo4j graph stack
├── pyproject.toml          # Project config (uv + hatch)
├── requirements.txt        # ⚠ Stale, has Windows-only path, version conflicts
└── uv.lock                 # UV lockfile (1.1MB)
```

### Responsibility Analysis

| Directory | Responsibility | Problem |
|---|---|---|
| `core/processing/` | All 37 AI model processors | Too many files, no sub-organization by domain |
| `core/processors/` | "Refactored" processor wrappers | Dead code — imports non-existent modules, never consumed |
| `core/utils/` | 18 utility files | Dumping ground — mixes logging, hardware, VRAM, retry, progress, model warming |
| `core/storage/repositories/` | 4 repository files | Each is 1,200-1,500 lines — god repositories |
| `core/retrieval/` | Search, reranking, RAG | `agentic_search.py` alone is 1,508 lines |
| `config.py` | ALL configuration | Single 1,054-line file with module-level `torch` import and `os.environ` mutation |

---

## Runtime Architecture

### Startup Sequence

There are **4 distinct entry points**, each with its own initialization:

#### 1. FastAPI Server (`api/server.py`)

```
Module load:
  1. config.py imports torch (global) → get_hardware_profile() runs → Settings() instantiated
  2. config.py sets os.environ[HF_HOME], TORCH_HOME, XDG_CACHE_HOME
  3. server.py suppresses TF warnings, sets Windows event loop
  4. All 14 route modules imported (some trigger heavy model imports)

Lifespan:
  5. init_langfuse() — observability init
  6. warmup_models() — downloads TransNet, BGE, InsightFace, YOLO, SigLIP, SAM3
  7. QueryPipeline() → VectorDB() → TextEncoder (lazy) + VisualEncoder (eager!) + Qdrant connection
  8. SearchAgent(db) → LLMFactory.create_llm() → OllamaLLM()
  9. job_manager.recover_on_startup() — crash recovery
```

#### 2. CLI Ingestion (`main.py`)

```
  1. config.py loaded (same torch boot)
  2. Optional: run_startup_checks() — checks FFmpeg, Qdrant, Ollama
  3. Optional: warmup_models() — same as server
  4. IngestionPipeline() — instantiates VectorDB, VisionAnalyzer, FaceManager, VoiceProcessor, TransNet, VideoVLM, SAM3Tracker, GraphBuilder, etc.
  5. pipeline.process_video() — full ingestion
```

#### 3. Agent CLI (`agent_cli.py`)

```
  1. config.py loaded
  2. MCP stdio_client connects to `uv run python -m core.agent.server`
  3. Server discovers tools (search_media, ingest_media, etc.)
  4. Ollama chat loop with function calling
```

#### 4. Agent Orchestrator (`agent_main.py`)

```
  1. config.py loaded
  2. get_orchestrator() → MultiAgentOrchestrator() → LLMFactory → Ollama
  3. orchestrator.execute(query) — routes to agents
```

### Configuration Loading

- **Mechanism:** Pydantic `BaseSettings` reads from `.env` file and environment variables
- **Module-level execution at import:** `config.py` line 80: `_HW_PROFILE = get_hardware_profile()` calls `torch.cuda.is_available()` — this happens **on any `import config`**
- **Side effects at module bottom (lines 1040-1054):** After `settings = Settings()`, the module mutates `os.environ` for `HF_HOME`, `TORCH_HOME`, `XDG_CACHE_HOME` and sets `sys.pycache_prefix`
- **54 files import `from config import settings`** — all trigger this chain

### Dependency Injection

- **Partial DI in `IngestionPipeline.__init__`**: Accepts optional `db`, `vision_analyzer`, `face_manager`, `voice_processor`, `video_vlm` via keyword args, falls back to concrete classes
- **FastAPI DI:** `api/deps.py` provides `get_pipeline()` and `get_search_agent()` from `app.state`
- **No DI container:** All wiring is manual. Most modules instantiate their own dependencies via module-level globals or lazy `@property` initialization

### Global State & Singletons

| Singleton | Location | Mechanism |
|---|---|---|
| `settings` | `config.py:1040` | Module-level `Settings()` |
| `pipeline` | `api/server.py:141` | Module-level `None`, set in lifespan via `global` |
| `_orchestrator` | `core/orchestration/orchestrator.py:215` | Module-level `None` + `get_orchestrator()` |
| `_orchestrator` | `core/orchestration/agent_graph.py:288` | **Duplicate** singleton with same name |
| `_vector_db`, `_pipeline`, `_agentic_search` | `core/agent/server.py:47-49` | Module-level `None` + getter functions |
| `_enhanced_config` | `core/integration.py:233` | Module-level singleton |
| `RESOURCE_ARBITER` | `core/utils/resource_arbiter.py` | Module-level instance |
| `_AUDIO_DETECTOR` | `core/processing/audio_events.py:703` | Global singleton |
| `_DEFAULT_ENCODER` | `core/processing/visual_encoder.py:453` | Global singleton |
| `_linker` | `core/processing/identity_linker.py:368` | Global singleton |
| `_engine` | `core/processing/metadata.py:391` | Global singleton |
| `_deep_research_processor` | `core/processing/deep_research.py:373` | Global singleton |
| `_analyzer` | `core/processing/audio_structure.py:540` | Global singleton |
| `_inpainter` | `core/manipulation/inpainting.py:335` | Global singleton |
| `_pipeline` | `core/manipulation/pipeline.py:221` | Global singleton |
| `_GRAPH_STORE` | `core/knowledge/graph_store.py:98` | Global singleton |
| `_orchestrator` | `core/retrieval/rag.py:543` | Global singleton |
| `_default_manager` | `core/retrieval/hitl_feedback.py:310` | Global singleton |
| `_cached_profile` | `core/utils/hardware.py:456` | Global singleton |
| `identity_graph` | `core/storage/identity_graph.py` | Module-level instance |
| `job_manager` | `core/ingestion/jobs.py` | Module-level instance |
| `progress_tracker` | `core/utils/progress.py` | Module-level instance |

**Total: 20+ module-level singletons managed via `global` keyword.**

---

## Module Architecture

### `config.py` (Root)

- **Purpose:** Central configuration for the entire application
- **Responsibilities:** Hardware detection, all 150+ settings, env var override, device detection, embedding dimension auto-adjustment, HuggingFace cache redirection
- **Public API:** `settings` (singleton `Settings` instance), `HardwareProfile`, `LLMProvider`, `get_hardware_profile()`
- **Dependencies:** `torch`, `pydantic`, `pydantic_settings`
- **Consumers:** 54 files across all packages
- **Problems:** Module-level `torch` import; module-level `get_hardware_profile()` execution; module-level `os.environ` mutation; 1,054 lines

---

### `llm/` (Top-level LLM Layer)

- **Purpose:** Abstract LLM interface + concrete adapters
- **Modules:**
  - `interface.py` — `LLMInterface` ABC with `generate()`, `generate_structured()`, `describe_image()`, prompt loading, JSON repair
  - `factory.py` — `LLMFactory` with `create_llm()`, `get_default_llm()`, `create_vision_llm()`, `create_text_llm()`
  - `ollama.py` — `OllamaLLM` adapter (610 lines) — handles text, vision, structured output
  - `gemini.py` — `GeminiLLM` adapter using `google-generativeai`
- **Public API:** `LLMFactory.create_llm()`, `LLMInterface` protocol
- **Dependencies:** `ollama`, `google-generativeai`, `pydantic`, `config.settings`
- **Consumers:** `core/retrieval/`, `core/orchestration/`, `core/agent/`, `core/processing/`

---

### `core/ingestion/` (Ingestion Pipeline)

- **Purpose:** Orchestrate media file ingestion through all processing stages
- **Key Files:**
  - `pipeline.py` (1,056 lines) — `IngestionPipeline` class using mixin inheritance from 5 stage files
  - `stages/frame_stage.py` (1,263 lines) — Frame processing mixin
  - `stages/audio_stage.py` (535 lines) — Audio transcription mixin
  - `stages/voice_stage.py` (338 lines) — Voice diarization mixin
  - `stages/scene_stage.py` (562 lines) — Scene detection + VLM captioning
  - `stages/audio_events_stage.py` (290 lines) — CLAP audio event detection
  - `jobs.py` (424 lines) — SQLite job persistence + crash recovery
  - `scanner.py` — Directory scanning for batch ingestion
  - `sandbox.py` — Sandboxed path validation
  - `celery_app.py` / `tasks.py` — Celery distributed ingestion (optional)
- **Design Pattern:** `IngestionPipeline` inherits from 5 `*StageMixin` classes, each contributing a `_process_*` method
- **Dependencies:** Virtually everything — all processing, storage, LLM, utils
- **Problems:** Pipeline is a 1,056-line god class built via mixin soup; each mixin accesses `self.*` attributes from the pipeline, creating implicit coupling

---

### `core/processing/` (AI Processors)

- **Purpose:** 37 individual AI model wrappers for every processing capability
- **Key Modules (by size):**

| Module | Lines | Responsibility |
|---|---|---|
| `transcriber.py` | 1,109 | Whisper transcription with chunking, language detection |
| `identity.py` | 1,087 | Face detection, clustering, track building (FaceManager) |
| `voice.py` | 682 | Speaker diarization (PyAnnote), voice embedding |
| `audio_events.py` | 706 | CLAP + AST audio event detection |
| `video_understanding.py` | 598 | InternVideo/LanguageBind video embeddings |
| `ocr.py` | 569 | PaddleOCR/EasyOCR/Surya text extraction |
| `cinematography.py` | 507 | Shot type, mood, aesthetics classification |
| `visual_encoder.py` | 458 | SigLIP/CLIP visual embedding |
| `summarizer.py` | 441 | Hierarchical video summarization (L1/L2) |
| `deep_research.py` | 382 | Zero-shot cinematography classification |
| `enrichment.py` | 337 | External web search enrichment |
| `scene_aggregator.py` | 428 | Global context aggregation |
| `object_detection.py` | 359 | YOLO-World object detection |
| `extractor.py` | 441 | Frame extraction from video |

- **Dependencies:** `torch`, `transformers`, `insightface`, `paddleocr`, `ultralytics`, `pyannote`, etc.
- **Problems:** No sub-organization; 37 files flat in one directory; some modules have module-level singleton globals

---

### `core/processors/` (⚠ Dead Package)

- **Purpose:** Attempted refactor to create higher-level processor classes
- **Files:** `audio.py`, `video.py`, `identity.py`
- **Status:** **DEAD CODE** — These files import from non-existent modules:
  - `from core.ingestion.diarization import VoiceProcessor` — module does not exist
  - `from core.ingestion.faces import FaceManager, FaceTrackBuilder` — module does not exist
  - `from core.ingestion.vision import VisionAnalyzer` — module does not exist
  - `from core.processing.scenelets import SceneletBuilder` — module does not exist
- **Consumers:** Zero. No file imports from `core.processors`
- **Verdict:** Should be deleted entirely

---

### `core/storage/` (Database Layer)

- **Purpose:** Qdrant vector database operations, encoding, schema management
- **Key Files:**
  - `db.py` (1,667 lines) — `VectorDB` god class inheriting from 4 repositories
  - `repositories/face_repository.py` (1,532 lines) — All face CRUD
  - `repositories/search_repository.py` (1,463 lines) — All search queries
  - `repositories/scene_repository.py` (1,428 lines) — Scene/scenelet/summary CRUD
  - `repositories/voice_repository.py` (1,294 lines) — Voice CRUD
  - `encoder.py` (286 lines) — SentenceTransformer lifecycle
  - `schema.py` (303 lines) — Qdrant collection definitions
  - `identity_graph.py` (497 lines) — Track-level identity graph
  - `keyword_index.py` (173 lines) — BM25 keyword index
  - `graph/` — Neo4j graph managers (6 files)
- **Architecture:** `VectorDB` uses **multiple inheritance** from 4 Repository classes — each repository is 1,200-1,500 lines
- **Problems:** `VectorDB` has 100+ methods across inherited repositories; no clear separation between read/write; eager loading of `VisualEncoder` in `__init__`

---

### `core/retrieval/` (Search Layer)

- **Purpose:** Query parsing, multi-modal search, reranking, RAG
- **Key Files:**
  - `agentic_search.py` (1,508 lines) — `SearchAgent` via mixin inheritance (`QueryParserMixin` + `ResultProcessorMixin`)
  - `reranker.py` (680 lines) — `RerankingCouncil` with BGE + LLM verification
  - `rag.py` (546 lines) — `VideoRAGOrchestrator` with query decomposition
  - `hybrid.py` (212 lines) — BM25 + vector hybrid search
  - `hitl_feedback.py` (310 lines) — Human-in-the-loop feedback scoring
  - `privacy.py` (160 lines) — Privacy filter for personal/movie mode
  - `query_parser.py` (107 lines) — `QueryParserMixin`
  - `result_processor.py` (436 lines) — `ResultProcessorMixin`
- **Problems:** `SearchAgent` at 1,508 lines is a god class; mixin pattern makes the call chain hard to trace

---

### `core/agent/` (Agent Layer)

- **Purpose:** MCP tool server + A2A protocol handler
- **Files:**
  - `server.py` (408 lines) — FastMCP server with 7 tools
  - `handler.py` (601 lines) — A2A request handler with Ollama function calling
  - `cards.py` (227 lines) — Agent capability cards
  - `card.py` (62 lines) — Agent card data class (⚠ overlaps with `cards.py`)
  - `a2a_server.py` (69 lines) — A2A HTTP server entry point
- **Problems:** `card.py` and `cards.py` have overlapping responsibilities; `handler.py` duplicates tool schemas that `server.py` already defines

---

### `core/orchestration/` (Agent Routing)

- **Purpose:** Route user queries to appropriate specialized agents
- **Files:**
  - `orchestrator.py` (231 lines) — `MultiAgentOrchestrator` #1 (uses tool schemas from `cards.py`)
  - `agent_graph.py` (301 lines) — `MultiAgentOrchestrator` #2 (uses `AgentCard` class)
- **Problems:** **Two competing implementations of the same class with the same name.** Both have `get_orchestrator()` singletons. `orchestrator.py` does actual tool execution; `agent_graph.py` only does routing. `agent_main.py` imports from `orchestrator.py`; no file imports from `agent_graph.py`'s `get_orchestrator`

---

### `core/utils/` (Utilities)

- **Purpose:** Cross-cutting concerns (logging, hardware, VRAM, retry, etc.)
- **18 files including:**
  - `logger.py` (367 lines) — Loguru + stdlib bridge + Loki integration
  - `progress.py` (879 lines) — Real-time pipeline progress tracking
  - `resource_arbiter.py` (442 lines) — VRAM budget manager
  - `hardware.py` (472 lines) — GPU/CPU detection and monitoring
  - `model_warmer.py` (290 lines) — Model pre-downloading
  - `startup_checks.py` (186 lines) — Pre-flight dependency checks
  - `observability.py` (237 lines) — Langfuse tracing integration
  - `batch_extract.py` (196 lines) — Batch audio extraction
  - `cancellation.py` (88 lines) — Job cancellation tokens
- **Problems:** This is a dumping ground. `resource_arbiter.py`, `hardware.py`, and `resource.py` overlap significantly. `progress.py` at 879 lines is doing too much

---

### `api/routes/` (REST API)

- **Purpose:** 17 route modules covering all API endpoints
- **Largest files:** `faces.py` (594 lines), `media.py` (609 lines), `search.py` (440 lines), `identities.py` (407 lines), `voices.py` (397 lines)
- **Problems:** Routes contain significant business logic (e.g., face cluster merging in `faces.py`, search parameter tuning in `search.py`); should delegate to service layer

---

### `web/` (React Frontend)

- **Purpose:** Browser UI for search, library management, identity management
- **Stack:** React 19 + TypeScript + MUI 7 + TailwindCSS 4 + React Query + Framer Motion + React Router
- **Structure:** `components/`, `pages/`, `api/`, `types/`, `theme.ts`
- **Dependencies:** MUI AND Tailwind simultaneously — competing styling systems

---

## Execution Flow

### How a Media File Moves Through the System

```
1. INPUT
   ┌─────────────────┐
   │  Video File      │ (e.g., movie.mp4)
   │  + media_type    │ (movie/tv/personal/unknown)
   └───────┬─────────┘
           │
2. PROBING & VALIDATION
           ▼
   ┌─────────────────┐
   │  MediaProber     │ → FFprobe metadata (duration, codec, resolution)
   │  Path validation │ → Sandbox check
   └───────┬─────────┘
           │
3. AUDIO EXTRACTION & TRANSCRIPTION (AudioStageMixin)
           ▼
   ┌─────────────────┐
   │  FFmpeg extract  │ → .wav audio file
   │  Language detect │ → Whisper language detection (30s sample)
   │  Transcriber     │ → Whisper large-v3 (faster-whisper/CTranslate2)
   │                  │ → Segments with timestamps + word-level alignment
   │  Store segments  │ → VectorDB.insert_media_segments()
   └───────┬─────────┘
           │
4. VOICE DIARIZATION (VoiceStageMixin)
           ▼
   ┌─────────────────┐
   │  PyAnnote 3.1   │ → Speaker segments (who spoke when)
   │  Voice embeddings│ → Speaker clustering (cosine similarity)
   │  Global identity │ → Match against existing speakers
   │  Store voices    │ → VectorDB voice collection
   └───────┬─────────┘
           │
5. SCENE DETECTION
           ▼
   ┌─────────────────┐
   │  TransNet V2     │ → Shot boundaries (ONNX)
   │  PySceneDetect   │ → Content-based scene detection (fallback)
   └───────┬─────────┘
           │
6. FRAME PROCESSING (FrameStageMixin — LARGEST STAGE)
           ▼
   ┌─────────────────┐
   │  Frame extract   │ → OpenCV frame sampling at configured interval
   │  For each frame: │
   │    ├─ SigLIP     │ → Visual embedding (1152d)
   │    ├─ InsightFace│ → Face detection + ArcFace embeddings
   │    ├─ VLM caption│ → Ollama/Gemini dense frame description
   │    ├─ PaddleOCR  │ → Text extraction (throttled)
   │    ├─ YOLO-World │ → Object detection
   │    └─ Deep Res.  │ → Shot type, mood, aesthetics (per-scene)
   │  Face clustering │ → HDBSCAN → Track-level clusters
   │  Identity link   │ → Cross-video identity matching
   └───────┬─────────┘
           │
7. AUDIO EVENTS (AudioEventsStageMixin)
           ▼
   ┌─────────────────┐
   │  CLAP            │ → Audio event embeddings (5s windows)
   │  AST             │ → Audio classification (AudioSet labels)
   │  Store events    │ → VectorDB audio_events collection
   └───────┬─────────┘
           │
8. SCENE AGGREGATION & STORAGE (SceneStageMixin)
           ▼
   ┌─────────────────┐
   │  Scene aggregate │ → Merge frames into scenes
   │  Scenelet build  │ → Sliding window action sequences (10s/5s stride)
   │  VLM scene cap   │ → Scene-level VLM summary
   │  InternVideo     │ → Video embeddings (1024d) per scene
   │  Text embeddings │ → BGE-M3 / NV-Embed-v2 (1024d/4096d)
   │  Store all       │ → scenes, scenelets, video_metadata collections
   └───────┬─────────┘
           │
9. POST-PROCESSING
           ▼
   ┌─────────────────┐
   │  Knowledge graph │ → GraphBuilder → Neo4j entity nodes + edges
   │  Metadata enrich │ → TMDB/OMDB API (for movies/TV)
   │  Global summary  │ → L1 video summary (optional)
   │  Thumbnail       │ → Extract representative frame
   └───────┬─────────┘
           │
10. OUTPUT → Qdrant Collections
           ▼
   ┌─────────────────────────────────────────┐
   │  media_frames    → visual + text embeddings  │
   │  media_segments  → dialogue embeddings        │
   │  scenes          → scene-level text embeddings │
   │  scenelets       → action sequence embeddings  │
   │  faces           → face embeddings (512d)      │
   │  voice_segments  → speaker embeddings (256d)   │
   │  audio_events    → CLAP embeddings             │
   │  video_metadata  → video-level metadata        │
   │  summaries       → hierarchical summaries      │
   │  masklets        → SAM3 segmentation masks     │
   └─────────────────────────────────────────┘
```

### Search Flow

```
User Query → SearchAgent.sota_search()
  → QueryParserMixin.parse_query()
    → LLM structured output → ParsedQuery (person, action, objects, etc.)
    → Identity resolution (name → face cluster ID)
  → Multi-modal parallel search:
    → Scene search (text + visual embedding)
    → Frame search (text + visual embedding)
    → Scenelet search (text embedding)
    → Voice search (text embedding + identity filter)
    → Dialogue search (text embedding)
    → Audio event search (text embedding)
    → Video metadata search (text embedding)
    → Graph search (Neo4j)
  → ResultProcessorMixin.process_results()
    → RRF fusion across modalities
    → Temporal bucketing (2s windows)
    → VLM reranking (optional)
    → Identity boost
    → Deduplication
  → Response with scored, ranked results
```

---

## Dependency Graph

### Internal Dependencies (Simplified)

```
config.py ←──── [54 files across all packages]
    │
    ▼
core/storage/db.py ←── core/ingestion/pipeline.py
    │                    core/retrieval/agentic_search.py
    │                    core/agent/server.py
    │                    api/server.py
    ▼
core/storage/encoder.py ←── core/storage/db.py
core/storage/schema.py  ←── core/storage/db.py
core/storage/repositories/* ←── core/storage/db.py (via multiple inheritance)
    │
    ▼
core/processing/* ←── core/ingestion/stages/*
                       core/ingestion/pipeline.py
    │
    ▼
llm/* ←── core/retrieval/agentic_search.py
           core/orchestration/orchestrator.py
           core/agent/handler.py
           core/processing/* (VLM captioning)
```

### External Dependencies (Heavy)

| Dependency | Size (Approx) | Used By |
|---|---|---|
| `torch` (2.5.1 + CUDA) | ~2.5GB | Everything (via config.py import) |
| `transformers` (4.46.3) | ~200MB | Transcriber, VLM, video understanding |
| `paddlepaddle` + `paddleocr` | ~800MB | OCR processing |
| `insightface` | ~300MB (models) | Face detection/recognition |
| `pyannote.audio` | ~200MB | Speaker diarization |
| `ultralytics` | ~100MB | YOLO object detection |
| `sentence-transformers` | ~100MB | Text embedding |
| `faster-whisper` | ~50MB (+ models) | Speech transcription |
| `sam3` (from git) | Variable | Object tracking |
| `deepface` | ~200MB | Facial analysis |
| `fer` | ~50MB | Facial emotion recognition |
| `neo4j` | ~10MB | Knowledge graph |
| `celery[redis]` | ~10MB | Distributed processing |

### Circular Dependencies

No hard circular imports detected (Python would crash). However, there are **soft circular patterns** via lazy imports:

1. `core/ingestion/pipeline.py` → lazy imports `core/integration.py` → references types from `core/processing/*`
2. `core/storage/db.py` → imports `core/processing/visual_encoder.py` at init time → which imports `config.settings` → which imports `torch`
3. `core/retrieval/agentic_search.py` → lazy imports `core/retrieval/hybrid.py` → imports `core/storage/db.py`

### Unused Dependencies (in `pyproject.toml`)

| Dependency | Status |
|---|---|
| `hydra-core` | Declared but never imported in any Python file |
| `sam3 @ git` | Only used if `enable_sam3_tracking=True` |
| `deepface` | Only used in `core/processing/biometrics.py` |
| `fer` | Only used in `core/processing/biometrics.py` |
| `neo4j` | Only used in `core/storage/graph/` |
| `celery[redis]` | Only used if `enable_distributed_ingestion=True` |
| `black` | Dev tool listed as runtime dependency |
| `ruff` | Dev tool listed as runtime dependency |
| `cmake`, `cython`, `ninja` | Build tools listed as runtime deps |
| `pytest` | Test tool listed as runtime dependency |

### Potentially Removable Dependencies

| Dependency | Reason |
|---|---|
| `black` | Dev tool, should be in `[project.optional-dependencies.dev]` only |
| `ruff` | Dev tool, should be in `[project.optional-dependencies.dev]` only |
| `cmake`, `cython`, `ninja` | Build tools, should be build-time only |
| `pytest` | Already in dev deps separately |
| `hydra-core` | Not imported anywhere in codebase |
| `tf-keras` | Heavy compatibility shim, only needed for `fer` package |

---

## Performance Bottlenecks

### 1. Slow Imports — `config.py` Torch Boot (🔴 Critical)

Every `import config` triggers:
```python
import torch                          # ~2-5 seconds
_HW_PROFILE = get_hardware_profile()  # Calls torch.cuda.is_available()
settings = Settings()                 # Pydantic validation
os.environ["HF_HOME"] = ...          # Side effects
```

54 files import `config.settings`. **Any test, script, or module load pays this 2-5 second penalty.**

**Solution:** Move `torch` imports to lazy properties. Make `get_hardware_profile()` a cached `@property` on `Settings`. Move env var mutations to an explicit `init()` function.

### 2. Eager Visual Encoder Loading (🟠 High)

`VectorDB.__init__()` line 141:
```python
from core.processing.visual_encoder import get_default_visual_encoder
self.visual_encoder = get_default_visual_encoder()
```
This loads the SigLIP model into VRAM **every time a VectorDB is constructed**, even for search-only operations that may not need visual encoding.

**Solution:** Make `visual_encoder` a lazy `@property` that loads on first access.

### 3. Repeated Model Initialization (🟠 High)

The ingestion pipeline `__init__` eagerly instantiates:
- `TransNetV2()` — loads ONNX model
- `VideoVLM()` — loads Qwen2-VL
- `FaceManager()` — loads InsightFace
- `VoiceProcessor()` — loads PyAnnote
- `GraphBuilder()` — connects to Neo4j
- `SAM3Tracker()` — loads SAM3 model

For a search-only server startup, none of these are needed.

**Solution:** All processors should be lazy-loaded behind `@property` or factory methods.

### 4. Sequential Frame Processing (🟡 Medium)

In `frame_stage.py`, frames are processed with limited parallelism:
```python
VLM_SEMAPHORE = asyncio.Semaphore(settings.vlm_concurrency)
```
VLM captioning is the bottleneck. Each frame requires a full VLM inference round-trip (200ms-5s depending on model/hardware).

**Solution:** Batch VLM calls where possible; pre-filter frames using motion detection before expensive VLM calls.

### 5. Embedding Cache Size (🟡 Medium)

`TextEncoder._embedding_cache_max_size = 1000` — this is an in-memory `OrderedDict` with no persistence. Cache is lost on restart.

**Solution:** Use disk-backed cache (e.g., `diskcache`) or Redis for embedding cache persistence.

### 6. Blocking Synchronous Operations (🟡 Medium)

Several operations block the async event loop:
- `self.encoder.encode()` in `TextEncoder` — synchronous SentenceTransformer call
- `InsightFace.get()` — synchronous face detection
- `PaddleOCR` — synchronous OCR

**Solution:** Wrap blocking calls in `asyncio.to_thread()` or use `loop.run_in_executor()`.

---

## Scalability Problems

### Large Files

- Processing a 3-hour video loads all frames into memory progressively. Memory grows linearly with video duration.
- Chunking is configurable (`chunk_duration_seconds=600`) but the default `min_media_length_for_chunking=1800` means videos under 30 minutes are never chunked.

### Large Folders / Millions of Media Files

- No batch ingestion pipeline. `scanner.py` finds files but processes them one at a time.
- Qdrant scroll operations in repositories use `limit=100` per page but may return millions of results for large libraries.
- BM25 keyword index is stored as a single pickle file — won't scale past ~100K documents.

### GPU Utilization

- Only single-GPU support (`torch.cuda.get_device_properties(0)`)
- No multi-GPU model sharding or pipeline parallelism
- `ResourceArbiter` tracks VRAM but doesn't actually prevent OOM — it's advisory only

### Database Bottlenecks

- `VectorDB` creates a new `QdrantClient` per construction. Multiple entry points may create multiple connections.
- Identity graph operations do full collection scans for cluster matching.
- No connection pooling for Qdrant.

### Startup Latency

- Server startup takes 30-120 seconds due to model warming (`warmup_models()`)
- Model warming downloads models synchronously before the server can accept requests

### Model Loading

- 15+ AI models must be loaded. Even with lazy loading, first-use latency is significant.
- No model caching between process restarts (except HuggingFace cache on disk)

---

## Design Problems

### God Classes

| Class | Lines | Methods (est.) | Diagnosis |
|---|---|---|---|
| `VectorDB` | 1,667 | 100+ | Inherits from 4 repositories. Should be split into focused services. |
| `SearchAgent` | 1,508 | 30+ | Inherits from 2 mixins. Search, parsing, and processing should be separate. |
| `IngestionPipeline` | 1,056 | 20+ | Inherits from 5 stage mixins. Orchestration mixed with processing. |
| `FaceRepository` | 1,532 | 40+ | All face CRUD in one class. |
| `SearchRepository` | 1,463 | 30+ | All search queries in one class. |

### God Modules

| Module | Lines | Problem |
|---|---|---|
| `config.py` | 1,054 | Every setting in one file with side effects |
| `frame_stage.py` | 1,263 | Frame processing stage as a mixin |
| `transcriber.py` | 1,109 | Transcription with language detection, chunking, retry |
| `identity.py` | 1,087 | Face management with detection, clustering, tracking |

### Duplicate Logic

1. **Two `MultiAgentOrchestrator` classes** — `core/orchestration/orchestrator.py` and `core/orchestration/agent_graph.py`
2. **Two `core/llm/` packages** — `llm/` (top-level) and `core/llm/` (factories for text/VLM)
3. **`core/processing/` vs `core/processors/`** — overlapping package names, the latter is dead code
4. **`card.py` vs `cards.py`** in `core/agent/` — overlapping agent card definitions
5. **`core/domain/schemas.py` vs `core/knowledge/schemas.py`** — both define frame analysis schemas, one "flexible" and one "strict"
6. **Prompt loading** — `LLMInterface.load_prompt()`, `core/utils/prompt_loader.py`, `agent_graph._load_prompt()` — three implementations

### Global State

20+ module-level singletons (documented above). This makes:
- Testing nearly impossible without mocking at module level
- Process forking (Celery workers) unreliable
- Memory management unpredictable

### Business Logic in Wrong Layers

| Problem | Location |
|---|---|
| Search parameter tuning in route handler | `api/routes/search.py` |
| Face cluster merging logic in route handler | `api/routes/faces.py` |
| Agent tool schemas hardcoded in handler | `core/agent/handler.py` |
| VectorDB connection details in MCP server | `core/agent/server.py` (hardcoded `localhost:6333`) |

### Hidden Side Effects

1. `config.py` mutates `os.environ` and `sys.pycache_prefix` on import
2. `api/server.py` calls `asyncio.set_event_loop_policy()` at module level on Windows
3. `core/agent/server.py` redirects `sys.stdout` to `sys.stderr` at module level
4. `VectorDB.__init__` triggers `ensure_all_collections()` — creates Qdrant collections as a side effect of construction

### Magic Constants

- `VLM_SEMAPHORE = asyncio.Semaphore(settings.vlm_concurrency)` — module-level semaphore
- `MEDIA_VECTOR_SIZE`, `FACE_VECTOR_SIZE` etc. in `constants.py` — well-named but tightly coupled to model choices
- Hardcoded `"docker"`, `"localhost"`, `6333` in multiple files (server.py, pipeline.py, agent/server.py)

### Dead Code

| Code | Location | Why Dead |
|---|---|---|
| Entire `core/processors/` package | 4 files | Imports non-existent modules; never consumed |
| `core/orchestration/agent_graph.py` | 301 lines | Competing implementation, never imported by entry points |
| `core/agent/card.py` | 62 lines | Overlaps with `cards.py` |
| `core/retrieval/query_pipeline.py` uses it but `late_interaction.py` | 214 lines | May be unused depending on search mode |
| `Dockerfile` entrypoint is `main.py` | - | But `main.py` is a CLI tool, not a server |

---

## Testing

### Current Testing Strategy

- **Root `conftest.py`**: Mocks `torch` globally via `sys.modules` — prevents any real model testing
- **`tests/conftest.py`**: Sets `TEST_MODE=true` env var, provides `mock_qdrant` fixture
- **Test files are integration/verification scripts**, not unit tests:
  - `tests/verify_infrastructure.py` — Checks if Qdrant, Ollama are running
  - `tests/verify_agents.py` — Tests agent routing end-to-end
  - `tests/verify_clustering.py` — Tests face clustering with real models
  - `tests/golden_run.py` — Full pipeline golden path test
  - `tests/stress_test_memory.py` — Memory stress testing
  - `tests/e2e_search_quality.py` — Search quality evaluation
  - `tests/mocks/` — Contains `processors.py` and `storage.py` mock classes

### Missing Tests

- **Zero unit tests** for any module
- No tests for:
  - `config.py` settings validation
  - `LLMInterface` / `LLMFactory` — no mocked LLM tests
  - `VectorDB` methods — no tests with mocked Qdrant
  - `IngestionPipeline` stages — no stage-level tests
  - `SearchAgent` query parsing — no tests with mocked LLM
  - API route handlers — no FastAPI `TestClient` tests
  - Error handling paths — no exception testing
  - Edge cases (empty video, corrupted file, network failure)

### Hard-to-Test Code

| Issue | Why |
|---|---|
| `config.py` imports `torch` at module level | Can't import config without torch |
| 20+ module-level singletons | Must mock at `sys.modules` level |
| `IngestionPipeline.__init__` instantiates 10+ heavy objects | Can't construct without mocking everything |
| `VectorDB.__init__` calls `ensure_all_collections()` | Requires running Qdrant to construct |
| `api/server.py` runs `create_app()` at module level | Importing the module starts the app |

### Mocking Issues

- Root `conftest.py` puts a `MagicMock` for `torch` in `sys.modules` — this breaks any test that actually needs torch
- No factory/DI pattern for most dependencies — must mock at module level
- Tight coupling between `IngestionPipeline` and concrete implementations

---

## Technical Debt

### 🔴 Critical

| Issue | Impact | Location |
|---|---|---|
| `config.py` module-level torch import + side effects | 2-5s penalty on every import; blocks testing | `config.py:10,80,1040-1054` |
| `core/processors/` dead package with broken imports | Confusion, import errors if accidentally loaded | `core/processors/*` |
| VectorDB god class (1,667 lines, 100+ methods) | Unmaintainable, untestable | `core/storage/db.py` |
| Two competing `MultiAgentOrchestrator` implementations | Confusion about which is active | `core/orchestration/*` |
| 52MB YOLO model checked into git (x2 copies) | Repository bloat (104MB of binary in git) | `yolov8m.pt`, `api/yolov8m.pt` |

### 🟠 High

| Issue | Impact | Location |
|---|---|---|
| 20+ global singletons via `global` keyword | Memory leaks, test isolation impossible | Throughout codebase |
| IngestionPipeline mixin soup (5 mixins, 1,056 lines) | Implicit coupling, hard to trace execution | `core/ingestion/pipeline.py` + `stages/` |
| SearchAgent god class (1,508 lines, 2 mixins) | Untestable, hard to extend | `core/retrieval/agentic_search.py` |
| Eager model loading in VectorDB.__init__ | Wasted VRAM for search-only paths | `core/storage/db.py:141` |
| Duplicate LLM packages (`llm/` + `core/llm/`) | Confusion about which to use | Two locations |
| Business logic in API routes | Tight coupling, untestable business rules | `api/routes/faces.py`, `search.py` |
| Debug print statements in production | Log noise, unprofessional | `api/server.py` (12 `print("DEBUG:...")`) |
| `requirements.txt` with Windows path and version drift | Broken cross-platform builds | `requirements.txt:3` |

### 🟡 Medium

| Issue | Impact | Location |
|---|---|---|
| No unit tests whatsoever | No regression safety | `tests/` |
| Prompt loading implemented 3 times | Maintenance burden, inconsistency | `llm/interface.py`, `core/utils/prompt_loader.py`, `core/orchestration/agent_graph.py` |
| `core/processing/` — 37 flat files | Hard to navigate, no domain grouping | `core/processing/` |
| Dev tools as runtime dependencies | Bloated production image | `pyproject.toml` |
| `core/domain/schemas.py` vs `core/knowledge/schemas.py` | Duplicate schema definitions | Two locations |
| Hardcoded connection details (`localhost:6333`) | Can't change without code modification | Multiple files |
| Dockerfile entrypoint is CLI, not server | Container can't serve API without override | `Dockerfile:60` |
| MUI + Tailwind in frontend | Competing styling systems | `web/package.json` |

### 🟢 Low

| Issue | Impact | Location |
|---|---|---|
| `use_indic_asr` referenced in `agent_main.py` but not in Settings | Runtime AttributeError possible | `agent_main.py:41` |
| `hydra-core` dependency unused | Unnecessary install | `pyproject.toml` |
| Stale docs (architecture before/after, roadmap) | Misleading for new developers | `docs/` |
| `core/agent/card.py` overlaps `cards.py` | Minor confusion | `core/agent/` |
| Windows-specific code in cross-platform modules | Unnecessary complexity on Linux/Mac | `api/server.py:42-100` |
| `test/` directory with 115MB test media | Repository bloat | `test/` |

---

## Refactor Priority

Ordered from **highest ROI** (least effort, most impact) to lowest ROI.

### Phase 1: Remove Dead Weight (1-2 days)

**ROI: 🔴🔴🔴🔴🔴 — Immediate cleanup, zero risk**

1. **Delete `core/processors/` entirely** — Dead package, broken imports, zero consumers
2. **Delete `core/orchestration/agent_graph.py`** — Duplicate, never imported by active code
3. **Delete duplicate `api/yolov8m.pt`** — Same 52MB file duplicated
4. **Add `yolov8m.pt` to `.gitignore`** and remove from git history (use `git filter-repo`)
5. **Move `test/` media files** to an external storage/LFS — 115MB of test videos
6. **Delete debug print statements** from `api/server.py` — replace with logger calls
7. **Move dev dependencies** (`black`, `ruff`, `cmake`, `cython`, `ninja`, `pytest`) from `[project.dependencies]` to `[project.optional-dependencies.dev]`
8. **Remove `hydra-core`** from dependencies — unused
9. **Delete or regenerate `requirements.txt`** — has Windows path, version conflicts with `pyproject.toml`

### Phase 2: Fix Config Module-Level Side Effects (2-3 days)

**ROI: 🔴🔴🔴🔴 — Unblocks testing, fixes 2-5s import penalty**

1. **Defer `torch` import in `config.py`** — Move `get_hardware_profile()` to a lazy cached property:
   ```python
   # Before (runs at import time):
   import torch
   _HW_PROFILE = get_hardware_profile()

   # After (runs on first access):
   @functools.lru_cache(maxsize=1)
   def get_hardware_profile(): ...
   ```
2. **Move `os.environ` mutations** to an explicit `config.initialize()` function called by entry points only
3. **Move `sys.pycache_prefix`** mutation to entry points
4. **Split settings into logical groups** using nested Pydantic models:
   ```python
   class QdrantSettings(BaseModel): ...
   class LLMSettings(BaseModel): ...
   class ProcessingSettings(BaseModel): ...
   class SearchSettings(BaseModel): ...
   class Settings(BaseSettings):
       qdrant: QdrantSettings = QdrantSettings()
       llm: LLMSettings = LLMSettings()
       processing: ProcessingSettings = ProcessingSettings()
       search: SearchSettings = SearchSettings()
   ```

### Phase 3: Break God Classes (1-2 weeks)

**ROI: 🔴🔴🔴 — Enables testing, improves maintainability**

1. **Split `VectorDB`** — Stop using multiple inheritance. Create focused service classes:
   - `FrameService(client, encoder)` — frame CRUD
   - `FaceService(client)` — face CRUD
   - `VoiceService(client)` — voice CRUD
   - `SceneService(client, encoder)` — scene CRUD
   - `SearchService(client, encoder, visual_encoder)` — search queries
   - `VectorDB` becomes a thin facade that composes these services

2. **Split `SearchAgent`** — Replace mixin inheritance with composition:
   - `QueryParser(llm)` — parse + expand queries
   - `ResultProcessor(llm)` — rerank + fuse results
   - `SearchAgent(db, query_parser, result_processor)` — thin orchestrator

3. **Refactor `IngestionPipeline`** — Replace mixin inheritance with explicit stage composition:
   ```python
   class IngestionPipeline:
       def __init__(self, stages: list[PipelineStage]):
           self.stages = stages

       async def process(self, video_path):
           context = PipelineContext(video_path)
           for stage in self.stages:
               await stage.execute(context)
   ```

4. **Make `VectorDB.__init__` lazy** — Don't load `VisualEncoder` or call `ensure_all_collections()` in constructor

### Phase 4: Introduce Service Layer (1 week)

**ROI: 🟠🟠🟠 — Moves business logic out of routes/CLI**

1. **Create `core/services/` package:**
   - `ingestion_service.py` — wraps pipeline construction + execution
   - `search_service.py` — wraps SearchAgent construction + search
   - `identity_service.py` — face/voice cluster management
   - `library_service.py` — media library operations

2. **API routes become thin**: Parse request → call service → return response
3. **CLI entry points become thin**: Parse args → call service → print output

### Phase 5: Testing Foundation (1 week)

**ROI: 🟠🟠🟠 — Enables confident refactoring**

1. **Remove root `conftest.py`** torch mock — it's a testing anti-pattern
2. **Add proper pytest fixtures** with factory functions for DI
3. **Unit test `config.py`** with env var overrides
4. **Unit test `SearchAgent.parse_query()`** with mocked LLM
5. **Unit test `VectorDB` methods** with mocked Qdrant client
6. **Add `pytest-asyncio`** for async test support
7. **Add FastAPI `TestClient`** tests for key routes

### Phase 6: Organize Processing Modules (3-5 days)

**ROI: 🟡🟡 — Improves navigability**

1. **Group `core/processing/` into subdirectories:**
   ```
   core/processing/
   ├── audio/          # transcriber, audio_events, audio_structure, speech_emotion
   ├── vision/         # extractor, vision, visual_encoder, cinematography, deep_research
   ├── identity/       # identity, clustering, biometrics, biometric_arbitrator, celebrity_lookup
   ├── text/           # ocr, ocr_factory, text_utils
   ├── scene/          # scene_detector, scene_aggregator, transnet_detector
   ├── video/          # video_understanding, frame_sampling, temporal
   └── enrichment/     # enrichment, metadata, content_classifier
   ```

### Phase 7: Consolidate Duplicates (3-5 days)

**ROI: 🟡🟡 — Reduces confusion**

1. **Merge `llm/` and `core/llm/`** into a single `core/llm/` package
2. **Merge `core/domain/schemas.py` and `core/knowledge/schemas.py`** — one canonical schema set
3. **Consolidate prompt loading** into `core/utils/prompt_loader.py` — one implementation
4. **Merge `core/agent/card.py` into `core/agent/cards.py`**
5. **Consolidate `core/utils/resource.py`, `resource_arbiter.py`, `hardware.py`** into a coherent hardware management package

### Phase 8: Fix Dockerfile + Docker Compose (1-2 days)

**ROI: 🟡 — Production readiness**

1. **Fix `Dockerfile` entrypoint** — should be `uvicorn api.server:app` not `python main.py`
2. **Remove `print("DEBUG:")` statements** from server.py
3. **Pin Python version** consistently across all Dockerfiles
4. **Add `.dockerignore` entries** for `test/`, `tests/`, `*.pt`, `uv.lock`

---

> **Note:** This document is a snapshot of the architecture as of 2026-07-12. It should be updated after each refactoring phase is completed.
