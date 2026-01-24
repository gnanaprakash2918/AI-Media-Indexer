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
    subgraph Frontend["Frontend (Reac)"]
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

## Ingestion Flow (Ultra-High-Fidelity)

```mermaid
flowchart TD
    A[Video Upload] --> B{File Validation}
    B -->|Valid| C[Media Probing]
    
    C --> D[Parallel Pipeline Orchestration]
    
    subgraph D[GPU-Aware Pipeline]
        D1[Frame Extraction]
        D2[Audio Demuxing]
    end
    
    subgraph AudioLogic[Audio Council]
        AL[Multi-pass Lang Detect] --> AC[ASR Council: ROVER voting]
        AC -->|Whisper v3, Indic, Seamless| AT[Final Transcript]
        AT --> AD[Analysis: Pyannote 3.1, SER, CLAP Events]
    end
    
    subgraph VideoLogic[Video Council]
        VS[Text-Gated Smart Sampler] --> VC[Video Council: InternVideo2.5]
        VC --> VT[SAM 2 Temporal Tracking]
        VT --> VM[Metadata: InsightFace 512D, PP-OCRv5]
    end
    
    D1 --> VS
    D2 --> AL
    
    subgraph Memory[3-Tier Temporal Memory]
        SW[Sensory Sliding Window]
        WM[Working Entity Map]
        LT[Long-term Identity Graph]
    end
    
    VT --> SW
    AT --> SW
    
    subgraph VectorDB["Qdrant Multi-Vector Store"]
        O[(media_frames: 512D)]
        S[(scenes: Visual/Motion/Dialogue)]
        P[(masklets: SAM 2)]
    end
    
    SW --> S
    VM --> O
    VT --> P
```

---

## Search Flow (Agentic & Hybrid)

```mermaid
flowchart TD
    A[User Query] --> B[LLM Query Expansion]
    B --> C[Constraint Decomposition: 7 Types]
    
    subgraph Retrieval[Retriever Council]
        V[Vector Search: NV-Embed-v2]
        K[Keyword Search: BM25 Dense]
        F[Face/Identity Search: Identity Graph]
    end
    
    C --> V
    C --> K
    C --> F
    
    subgraph Scoring[Reranker Council]
        RRF[Hybrid RRF Fusion: 0.7/0.3]
        CE[Cross-Encoder: MiniLM-L-12-v2]
        BGE[BGE-Reranker v2: M3]
        VV[VLM Visual Verification: Gemini 1.5]
    end
    
    V --> RRF
    K --> RRF
    RRF --> CE
    CE --> BGE
    BGE --> VV
    
    VV --> Q[Final Ranked Results]
    Q --> R[Reasoning Traces]
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
| `asr_council.py` | ROVER-based multi-ASR consensus |
| `vlm_council.py` | Multi-model frame description with synthesis |
| `temporal_context.py` | 3-tier XMem temporal memory management |
| `identity.py` | InsightFace 512D + temporal track building |
| `scene_aggregator.py` | Fusing visual/dialogue into Scenelets |
| `indic_transcriber.py` | AI4Bharat/NeMo Indic ASR wrapper |

### 3. Retrieval (`core/retrieval/`)
| File | Purpose |
|------|---------|
| `reranker.py` | RRF fusion + Cross-Encoder reranking council |
| `agentic_search.py` | Constraint decomposition & search agents |
| `late_interaction.py` | ColBERT-style MaxSim scoring |
| `hitl_feedback.py` | Human-in-the-loop scoring boosts |

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
| Vector DB | Qdrant (Multi-Vector Scene Schema) |
| LLM | Gemini 1.5 Pro (Reasoning), Ollama (Local) |
| Embeddings | NV-Embed-v2 (7B), BGE-M3 (Hybrid) |
| Vision | InternVideo2.5, SAM 2, InsightFace ArcFace |
| ASR | Whisper v3, AI4Bharat, SeamlessM4T v2 |
| Reranking | MiniLM-L-12-v2, BGE-Reranker v2 |
| Memory | 3-Tier XMem Memory (Sensory/Working/LT) |
