# AI-Media-Indexer

Commercial-grade AI engine for hyper-specific video search using identity, visual semantics, and audio.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                           CORE STACK                                │
├─────────────────────────────────────────────────────────────────────┤
│  FastAPI (REST API)  │  Qdrant (Vectors)  │  SQLite (Identity Graph)│
│  Ollama/Gemini (VLM) │  Whisper (ASR)     │  InsightFace (Faces)    │
└─────────────────────────────────────────────────────────────────────┘
```

### Ingestion Pipeline

```
Video → Extract → Diarize → Track-Cluster → VLM Caption → Index
        ▼          ▼           ▼              ▼            ▼
      Frames    Speakers    FaceTracks    Scenes      Qdrant
      Audio    (Pyannote)   (IoU+Cosine)  (Dense)     Vectors
```

| Stage | File | Purpose |
|-------|------|---------|
| Extract | `core/processing/extractor.py` | Frame extraction at intervals |
| Transcribe | `core/processing/transcriber.py` | Whisper ASR with language lock |
| Diarize | `core/processing/voice.py` | Speaker segmentation + RMS silence filter |
| Face Track | `core/processing/identity.py` | FaceTrackBuilder (IoU + cosine) |
| Scene Caption | `core/processing/scene_detector.py` | PySceneDetect + VLM dense captions |
| Index | `core/storage/db.py` | Qdrant hybrid vectors |

### Search Pipeline

```
Query → Intent Parse → Graph Filter → Vector Search → VLM Rerank
        ▼               ▼              ▼               ▼
    SearchIntent    video_ids     Candidates   SearchResultDetail
    (LLM)         (IdentityGraph)  (Qdrant)     (Calibrated 0-1.0)
```

| Component | File | Purpose |
|-----------|------|---------|
| Text LLM Factory | `core/llm/text_factory.py` | Strategy: Ollama/Gemini |
| Query Parser | `core/retrieval/query_parser.py` | NL → SearchIntent |
| Identity Graph | `core/storage/identity_graph.py` | SQLite pre-filtering |
| VLM Reranker | `core/retrieval/reranker.py` | Keyframe verification |
| Score Calibration | `core/retrieval/calibration.py` | Normalize to 0-1.0 |

### Key APIs

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/search/advanced` | POST | Agentic search with reranking |
| `/identities` | GET | List all identities |
| `/identities/{id}/merge` | POST | Merge two identities |
| `/jobs/{id}/pause` | POST | Pause job (SQLite-backed) |
| `/tools/redact` | POST | Smart blur identity |
| `/api/media/thumbnail` | GET | Dynamic frame extraction |

---

## Quick Start

```bash
# Install
git clone https://github.com/gnanaprakash2918/AI-Media-Indexer.git
cd AI-Media-Indexer
uv sync

# Run
./start.ps1  # or ./start.sh

# Web UI
cd web && npm install && npm run dev
```

Open http://localhost:3000

---

## Configuration

Copy `.env.example` to `.env`:

| Variable | Default | Purpose |
|----------|---------|---------|
| `OLLAMA_MODEL` | `llava:7b` | Vision model |
| `AI_PROVIDER_VISION` | `ollama` | VLM provider (ollama/gemini) |
| `AI_PROVIDER_TEXT` | `ollama` | Text LLM provider |
| `HF_TOKEN` | - | Voice analysis |
| `QDRANT_HOST` | `localhost` | Vector DB |

---

## Response Schema

```json
{
  "video_id": "abc123",
  "file_path": "/path/to/video.mp4",
  "start_time": 10.0,
  "end_time": 15.0,
  "score": 0.87,
  "match_reasons": ["identity_face", "semantic_visual", "vlm_verified"],
  "explanation": "Man in blue shirt bowling shown clearly",
  "thumbnail_url": "/api/media/thumbnail?path=...&time=10.0"
}
```

Score is normalized 0.0-1.0 (80% VLM weight when reranking enabled).

---

For detailed docs: `docs/development.md`