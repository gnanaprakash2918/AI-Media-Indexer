# AI-Media-Indexer Architecture

This document describes the system architecture, component interactions, and data flows.

## System Overview

AI-Media-Indexer is a multimodal video search and indexing system that combines:
- **Visual Analysis**: Frame extraction, object detection, face recognition
- **Audio Processing**: Transcription, speaker diarization, voice clustering  
- **Semantic Search**: Hybrid vector+keyword search with LLM reranking
- **Identity Resolution**: Face/voice cluster to named person mapping

---

## High-Level Architecture

```mermaid
graph TB
    subgraph Frontend["Frontend (React + Vite)"]
        UI[Web UI]
        RC[React Query Client]
    end

    subgraph Backend["Backend (FastAPI)"]
        API[API Routes]
        PIPE[Ingestion Pipeline]
        RAG[Search Engine]
    end

    subgraph Storage["Storage Layer"]
        QD[(Qdrant Vector DB)]
        FS[File System]
        IG[(Identity Graph SQLite)]
    end

    subgraph ML["ML Services"]
        OL[Ollama LLM]
        EMB[Embedding Models]
        ASR[Whisper ASR]
        INS[InsightFace]
    end

    UI --> RC --> API
    API --> PIPE
    API --> RAG
    PIPE --> QD
    PIPE --> FS
    RAG --> QD
    RAG --> OL
    PIPE --> EMB
    PIPE --> ASR
    PIPE --> INS
    PIPE --> IG
```

---

## Component Diagram

```mermaid
graph LR
    subgraph api["api/"]
        server[server.py]
        routes[routes/]
        deps[deps.py]
        schemas[schemas.py]
    end

    subgraph core["core/"]
        ingestion[ingestion/]
        processing[processing/]
        retrieval[retrieval/]
        storage[storage/]
        utils[utils/]
    end

    subgraph web["web/"]
        pages[pages/]
        components[components/]
        apiclient[api/client.ts]
    end

    server --> routes
    routes --> deps
    deps --> ingestion
    ingestion --> processing
    processing --> storage
    retrieval --> storage
    routes --> retrieval
```

---

## Ingestion Flow

```mermaid
flowchart TD
    A[Video Upload] --> B{File Validation}
    B -->|Valid| C[Media Probing]
    B -->|Invalid| Z[Error Response]
    
    C --> D[Parallel Processing]
    
    subgraph D[Parallel Processing]
        D1[Frame Extraction]
        D2[Audio Extraction]
        D3[Subtitle Detection]
    end
    
    D1 --> E[Scene Detection]
    E --> F[Visual Analysis]
    F --> G[Face Detection]
    F --> H[Object/Action Detection]
    
    D2 --> I[Transcription]
    I --> J[Speaker Diarization]
    J --> K[Voice Clustering]
    
    D3 --> L[Subtitle Parsing]
    
    G --> M[Face Clustering]
    M --> N[Identity Resolution]
    
    subgraph VectorDB["Vector Database"]
        O[(media_frames)]
        P[(voice_segments)]
        Q[(media_transcripts)]
        R[(scenelets)]
    end
    
    H --> O
    K --> P
    L --> Q
    I --> Q
    N --> R
```

---

## Search Flow

```mermaid
flowchart TD
    A[Search Query] --> B[Query Parser]
    B --> C{Search Type}
    
    C -->|Hybrid| D[Hybrid Search]
    C -->|Agentic| E[Agentic Search]
    C -->|Granular| F[Granular Search]
    
    D --> G[Vector Search]
    D --> H[Keyword Search]
    G --> I[RRF Fusion]
    H --> I
    
    E --> J[LLM Query Expansion]
    J --> K[Identity Resolution]
    K --> L[Multi-Vector Search]
    L --> M[LLM Reranking]
    
    F --> N[Query Decomposition]
    N --> O[Constraint Matching]
    O --> P[Chain-of-Thought Scoring]
    
    I --> Q[Results]
    M --> Q
    P --> Q
    
    Q --> R[Response Formatting]
    R --> S[Thumbnail/Playback URLs]
```

---

## Key Components

### 1. Ingestion Pipeline (`core/ingestion/`)
| File | Purpose |
|------|---------|
| `pipeline.py` | Main orchestrator for video processing |
| `jobs.py` | Job queue management and status tracking |
| `celery_app.py` | Distributed processing with Celery |

### 2. Processing (`core/processing/`)
| File | Purpose |
|------|---------|
| `transcriber.py` | Whisper-based transcription |
| `indic_transcriber.py` | Tamil/Hindi ASR with NeMo |
| `vision_analyzer.py` | LLM vision analysis via Ollama |
| `face_clustering.py` | HDBSCAN face clustering |
| `voice_clustering.py` | Speaker clustering |
| `segmentation.py` | Scene/shot detection |

### 3. Retrieval (`core/retrieval/`)
| File | Purpose |
|------|---------|
| `agentic_search.py` | SOTA search with LLM reranking |
| `advanced_query.py` | Multi-constraint query handling |
| `rag.py` | VideoRAG orchestrator |
| `engine.py` | Core search engine |

### 4. Storage (`core/storage/`)
| File | Purpose |
|------|---------|
| `db.py` | Qdrant vector database wrapper |
| `identity_graph.py` | SQLite identity mapping |

---

## API Endpoints

### Media
- `GET /media` - Stream video with range support
- `GET /media/segment` - Extract clip segment
- `GET /media/thumbnail` - Dynamic frame thumbnail

### Search
- `GET /search` - Multi-modal semantic search
- `GET /search/hybrid` - SOTA hybrid search
- `POST /search/granular` - Complex query handling

### Ingestion
- `POST /ingest` - Start video ingestion
- `GET /ingest/jobs` - List job statuses
- `DELETE /ingest/jobs/{id}` - Cancel job

### Identity Management
- `GET /faces/clusters` - Get face clusters
- `PUT /faces/{id}/identity` - Assign identity
- `POST /faces/merge` - Merge clusters
- `GET /voices/clusters` - Get speaker clusters

---

## Technology Stack

| Layer | Technology |
|-------|------------|
| Frontend | React, Vite, TanStack Query, TailwindCSS |
| Backend | FastAPI, Uvicorn, Python 3.12 |
| Vector DB | Qdrant |
| LLM | Ollama (local), Gemini (optional) |
| Embeddings | BAAI/bge-m3 |
| Vision | InsightFace, OpenCV |
| ASR | Whisper, NeMo |
| Task Queue | Celery + Redis (optional) |
| Observability | Langfuse (optional) |
