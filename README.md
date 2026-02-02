# AI-MEDIA-INDEXER

*Multimodal Intelligence for Media Discovery and Understanding*

![last commit](https://img.shields.io/github/last-commit/gnanaprakash2918/AI-Media-Indexer)
![python](https://img.shields.io/badge/python-3.12-blue)
![qdrant](https://img.shields.io/badge/Qdrant-Vector%20DB-red)

*Built with state-of-the-art AI technologies:*

![InternVideo2.5](https://img.shields.io/badge/InternVideo2.5-SOTA_VLM-blue)
![Whisper v3](https://img.shields.io/badge/Whisper_v3-General_ASR-green)
![IndicConformer](https://img.shields.io/badge/IndicConformer-AI4Bharat-orange)
![SAM 2](https://img.shields.io/badge/SAM_2-Visual_Tracking-purple)
![NV-Embed-v2](https://img.shields.io/badge/NV--Embed--v2-7B_SOTA-red)
![BGE-M3](https://img.shields.io/badge/BGE--M3-Hybrid_Retrieval-blue)
![Pyannote 3.1](https://img.shields.io/badge/Pyannote_3.1-Diarization-yellow)
![InsightFace](https://img.shields.io/badge/InsightFace-ArcFace_512D-green)
![PP-OCRv5](https://img.shields.io/badge/PP--OCRv5-Indic_OCR-red)

---

## Architecture Overview

The AI-Media-Indexer implements a massively parallel ingestion pipeline coordinated by a centralized resource arbiter. It processes audio and video tracks independently before fusing them into a temporal context for hyper-granular hybrid search.

### High-Fidelity Technical Schematic

### Current State Demo
> **Note**: This is the current state of the system. I am still strictly working on improving accuracy. Leaving a star on the repository helps ⭐!

![Demo](assets/demo.gif)

> **Note**: This architecture diagram reflects the **actual running code**, removing theoretical voting blocks for transparency.

### Ingestion Logic Flow
```mermaid
flowchart TD
    A[Video Upload] --> B{File Validation}
    B -->|Valid| C[Media Probing]
    
    C --> D[Parallel Pipeline Orchestration]
    
    subgraph D["GPU-Aware Pipeline"]
        D1[Frame Extraction]
        D2[Audio Demuxing]
    end
    
    subgraph AudioLogic["Audio Handler"]
        AL[Input Audio] --> AC{Sidecar/Embedded?}
        AC -->|Yes| AT[Parse Subtitles]
        AC -->|No| AD{Language Detect}
        
        AD -->|Indic > 20%| AI["IndicConformer (AI4Bharat)"]
        AD -->|Other| AW[Whisper v3 Large]
        
        AI --> AT
        AW --> AT
        
        AT --> AN[Diarization & Analysis]
        AN --> S1[Pyannote 3.1 - Speakers]
        AN --> S2[Speech Emotion Recognition]
        AN --> S3["CLAP Event Detection (Streaming)"]
    end
    
    subgraph VideoLogic["Video Loop"]
        VS[Scene Detection] --> VSAMP[Text-Gated Sampler]
        VSAMP --> VLM[InternVideo2.5 / Qwen-VL]
        VLM --> VT[SAM 2 Temporal Tracking]
        VT --> VM[Metadata: InsightFace, OCR]
    end
    
    D1 --> VS
    D2 --> AL
    
    subgraph Memory["Temporal Context Manager"]
        SW[Sensory Sliding Window]
        WM[Working Entity Map]
        LT[Long-term Identity Graph]
    end
    
    VT --> SW
    AT --> SW
```

### Search & Retrieval Flow
```mermaid
flowchart TD
    A[User Query] --> B[LLM Query Expansion]
    B --> C[Constraint Decomposition]
    
    subgraph Retrieval["Retriever (Parallel)"]
        V[Vector Search: NV-Embed-v2]
        K[Keyword Search: BM25]
        F[Graph Search: Identity Graph]
    end
    
    C --> V
    C --> K
    C --> F
    
    subgraph Scoring["Reranking"]
        RRF["Hybrid Fusion (Weighted)"]
        CE[Cross-Encoder Verification]
        LLM["VLM Visual Verification (Gemini)"]
    end
    
    V --> RRF
    K --> RRF
    RRF --> CE
    CE --> LLM
    LLM --> Q[Final Ranked Results]
```

---

## Core Features

- **Parallel Ingestion Pipeline**: Independent audio/video processing with GPU-aware resource orchestration.
- **Multimodal Intelligence**:
    - **Audio**: **CLAP** for text-audio retrieval and **Whisper v3/IndicConformer** for ASR (Fallback strategy).
    - **Vision**: **InternVideo2.5** for temporal video understanding and **SAM 2** for tracking.
    - **Identity**: Temporal face tracking with **InsightFace** and **HDBSCAN** clustering.
- **Advanced Temporal Fusion**:
    - **3-Tier Memory**: Tracks entities across short-term (Window) and long-term (Graph) contexts.
    - **Fused Scenelets**: 5s window fusion of visual descriptions and dialogue transcripts.
- **Agentic Search Engine**:
    - **Hybrid Retrieval**: RRF Fusion (Vector + Keyword) across Multi-Vector Scenes.
    - **LLM Reranking**: Reasoning-based verification using **Gemini 1.5**.

## Capabilities

| Module | Technology | Capability |
|--------|------------|------------|
| **VLM Intelligence** | Gemini 1.5 Pro | Narrative synthesis & reasoning traces |
| **Action Recognition** | InternVideo2.5 | Dense motion description |
| **Face Identity** | InsightFace ArcFace | 512D biometric vectors |
| **ASR** | Whisper/AI4Bharat | Multi-lingual transcription (Fallback logic) |
| **Search Fusion** | RRF | Hybrid ranking (Dense + Sparse) |

## Tech Stack

- **Backend**: Python 3.12, FastAPI, Celery, Redis
- **Vector Database**: Qdrant with Multi-Vector support
- **frontend**: React 19, Vite, Tailwind CSS 4.0

## Getting Started

### 1. Requirements
Ensure you have Docker, Python 3.12, and Node.js 20+ installed.

### 2. Initialization
```powershell
./init.ps1
```

### 3. Running
```powershell
python run.py
cd web && npm run dev
```

---

## Configuration (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_HOST` | `localhost` | Vector DB host |
| `USE_INDIC_ASR` | `True` | Enable AI4Bharat for Indic langs |
| `ENABLE_HYBRID_SEARCH`| `True` | Use weighted RRF for ranking |

---

## Search Examples

- "Person in red shirt running fast" (Visual)
- "Crowd cheering in background" (Audio Events)
- "Prakash speaking near the door" (Identity + Voice)
- "Text 'EXIT' visible on sign" (OCR)