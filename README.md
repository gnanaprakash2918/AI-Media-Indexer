# AI-Media-Indexer
AI engine for searching and navigating large media libraries using text, visuals, and audio.

## Documentation

- [**Developer Guide**](docs/development.md): Setup instructions, MCP Inspector workflow, and contribution guidelines.
- [**Dlib Setup Guide**](docs/setup_dlib.md): Detailed instructions for installing CUDA-enabled `dlib` on Windows and Linux.
- [**Troubleshooting**](docs/troubleshooting.md): Common errors, A2A server testing, and useful snippets.
- [**Sprint History**](Sprints.md): Detailed changelogs for each sprint.

---

## Current Progress & Features

The project has evolved into a robust Modular Monolith with the following capabilities:

### 1. Ingestion Engine
- **Multi-Modal Scanning**: Automatically discovers Video, Audio, and Image files.
- **Smart Probing**: Extracts metadata (resolution, codecs, duration) using FFmpeg.
- **Audio Transcription**: Uses `faster-whisper` (CUDA-accelerated) to generate accurate transcripts.
- **Visual Analysis**: Extracts frames at intervals and analyzes them using Vision LLMs (Gemini/Ollama).
- **Face Detection**: Identifies faces using `dlib` (CNN/HOG) or MediaPipe, computes 128-d encodings, and clusters identities via DBSCAN.

### 2. Search & Retrieval
- **Vector Database**: Powered by Qdrant (Docker/Local).
- **Semantic Search**:
  - **Visual Search**: Find scenes by description (e.g., "red car on a bridge").
  - **Dialogue Search**: Search within spoken conversations.
  - **Face Search**: Find appearances of specific people.
- **Hybrid Indexing**: Combines text embeddings (`sentence-transformers`) and face encodings.

### 3. Agentic Workflow (Sprint 5)
- **Model Context Protocol (MCP)**: Implements standard MCP server-client architecture.
- **Agent CLI**: Interactive command-line agent (`agent_cli.py`) that can:
  - Understand natural language queries.
  - Call tools (`search_media`, `ingest_media`) autonomously.
  - Synthesize complex answers from database results.

### 4. Tech Stack
- **Languages**: Python 3.12+ (Type-safe with Pydantic)
- **AI/ML**:
  - **LLM**: Ollama (Llama 3, LLaVA), Google Gemini
  - **ASR**: Faster-Whisper
  - **Vision**: Dlib, MediaPipe
- **Infrastructure**:
  - **Vector Store**: Qdrant
  - **Orchestration**: Google ADK / MCP
  - **Backend**: FastAPI (Server) & Typer (CLI)
  - **Frontend**: React 19 + MUI 7 + Vite

---

## Quick Start

### 0. Prerequisites
- Python 3.12+
- `uv` package manager

### 1. Installation

<!-- carousel -->
#### PowerShell
```powershell
git clone https://github.com/gnanaprakash2918/AI-Media-Indexer.git
cd AI-Media-Indexer
uv sync
```
<!-- slide -->
#### CMD
```cmd
git clone https://github.com/gnanaprakash2918/AI-Media-Indexer.git
cd AI-Media-Indexer
uv sync
```
<!-- slide -->
#### Bash
```bash
git clone https://github.com/gnanaprakash2918/AI-Media-Indexer.git
cd AI-Media-Indexer
uv sync
```
<!-- /carousel -->

### 2. Run Backend
```bash
uv run uvicorn api.server:app --port 8000
```

### 3. Run Web UI
```bash
cd web
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

**Required for full functionality:**

| Variable | Description |
|----------|-------------|
| `OLLAMA_MODEL` | Vision model (e.g., `llava`, `llama3.2-vision`) |
| `HF_TOKEN` | Hugging Face token for voice analysis |
| `QDRANT_HOST` | Vector database host (default: `localhost`) |

See [Configuration Guide](docs/configuration.md) for complete documentation.

### Docker Services

Start required services:

```bash
docker compose up qdrant -d
```

---

For detailed development workflows, see the [Developer Guide](docs/development.md).