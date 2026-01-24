# Configuration Guide

This document describes all configuration options for AI-Media-Indexer.

## Environment Setup

All configuration is managed through environment variables. Copy `.env.example` to `.env` and update the values as needed.

```bash
cp .env.example .env
```

## Complete Configuration Reference

### AI Providers

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `OLLAMA_BASE_URL` | string | `http://localhost:11434` | Ollama API endpoint |
| `OLLAMA_MODEL` | string | `llava:7b` | Model for vision analysis (must be multimodal) |
| `AGENT_MODEL` | string | `llama3.1` | Model for Agent CLI queries |
| `LLM_PROVIDER` | enum | `ollama` | LLM provider: `ollama` or `gemini` |
| `GOOGLE_API_KEY` | string | None | Google Gemini API key (required if using gemini) |
| `GEMINI_MODEL` | string | `gemini-1.5-pro` | Gemini model name (Flash or Pro) |
| `HF_TOKEN` | string | None | Hugging Face token for pyannote models |
| `AI_PROVIDER_VISION` | string | `gemini` | Primary VLM for "Chairman" synthesis |
| `USE_INDIC_ASR` | bool | `True` | Enable IndicConformer in ASR Council |

### Vector Database (Qdrant)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `QDRANT_HOST` | string | `localhost` | Qdrant server hostname |
| `QDRANT_PORT` | int | `6333` | Qdrant HTTP API port |
| `QDRANT_BACKEND` | enum | `docker` | Backend type: `memory` or `docker` |
| `QDRANT_RETRY_COUNT`| int | `3` | Retries for WinError 10053 resilience |

### Hardware Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DEVICE_OVERRIDE` | enum | Auto | Force device: `cuda`, `cpu`, or `mps` |
| `GPU_SEMAPHORE_LIMIT`| int | `1` | Max concurrent GPU tasks (OOM prevention) |

### Audio/Video Processing

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `FRAME_INTERVAL` | int | `15` | Seconds between extracted frames |
| `BATCH_SIZE` | int | `24` | Batch size for processing |
| `LANGUAGE` | string | `ta` | Default language code for transcription |
| `FALLBACK_MODEL_ID` | string | `distil-whisper/distil-large-v3` | Fallback Whisper model |

### Voice Analysis (Pyannote)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ENABLE_VOICE_ANALYSIS` | bool | `true` | Enable speaker diarization |
| `PYANNOTE_MODEL` | string | `pyannote/speaker-diarization-3.1` | Diarization model |
| `VOICE_EMBEDDING_MODEL` | string | `pyannote/wespeaker-voxceleb-resnet34-LM` | Voice embedding model |
| `MIN_SPEAKERS` | int | None | Minimum expected speakers |
| `MAX_SPEAKERS` | int | None | Maximum expected speakers |

### Resource Monitoring

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ENABLE_RESOURCE_MONITORING` | bool | `true` | Enable CPU/RAM/temperature monitoring |
| `MAX_CPU_PERCENT` | float | `90.0` | Pause processing above this CPU usage |
| `MAX_RAM_PERCENT` | float | `85.0` | Pause processing above this RAM usage |
| `MAX_TEMP_CELSIUS` | float | `85.0` | Pause processing above this temperature |
| `COOL_DOWN_SECONDS` | int | `30` | Seconds to wait when overheated |

### Metadata Enrichment

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `TMDB_API_KEY` | string | None | TMDB API key for movie/TV metadata |
| `OMDB_API_KEY` | string | None | OMDB API key for additional metadata |

### Langfuse Observability

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `LANGFUSE_BACKEND` | enum | `disabled` | Backend: `disabled`, `docker`, or `cloud` |
| `LANGFUSE_DOCKER_HOST` | string | `http://localhost:3300` | Local Langfuse URL |
| `LANGFUSE_HOST` | string | `https://cloud.langfuse.com` | Cloud Langfuse URL |
| `LANGFUSE_PUBLIC_KEY` | string | None | Cloud public key |
| `LANGFUSE_SECRET_KEY` | string | None | Cloud secret key |

## Prerequisites

### 1. Python Environment

Python 3.12+ is required. Use `uv` for dependency management:

```bash
uv sync
```

### 2. Ollama (Required for Vision Analysis)

Install Ollama and pull a vision-capable model:

```bash
# Install Ollama from https://ollama.com/download

# Pull a multimodal model (required for vision)
ollama pull llava

# Alternative: newer llama3.2-vision
ollama pull llama3.2-vision
```

Set in your `.env`:
```env
OLLAMA_MODEL=llava
```

### 3. Qdrant Vector Database

**Option A: Docker (Recommended)**
```bash
docker compose up qdrant -d
```

**Option B: Embedded (Development Only)**
```env
QDRANT_BACKEND=memory
```

### 4. Hugging Face Token (Required for Voice Analysis)

Voice analysis uses `pyannote.audio` which requires a Hugging Face token.

1. Create an account at [huggingface.co](https://huggingface.co)
2. Accept the model licenses:
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
3. Generate a token at [Settings > Access Tokens](https://huggingface.co/settings/tokens)
4. Add to `.env`:
```env
HF_TOKEN=hf_your_token_here
```

### 5. FFmpeg (Required)

FFmpeg is required for audio/video processing.

**Windows:**
```powershell
winget install ffmpeg
```

**Linux (Debian/Ubuntu):**
```bash
sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

## Running the Application

### Starting Docker Services

**Required: Qdrant Vector Database**
```bash
docker compose up qdrant -d
```

**Optional: Full Langfuse Stack (Observability)**
```bash
docker compose --profile docker up -d
```

This starts:
- Qdrant (port 6333)
- PostgreSQL (port 5435)
- ClickHouse (port 8123)
- Redis (port 6379)
- MinIO (port 9090)
- Langfuse (port 3300)

**Stop All Services:**
```bash
docker compose --profile docker down
```

**View Service Logs:**
```bash
docker compose logs -f langfuse
```

### Development Mode

This project uses `uv` for dependency management (not pip/venv).

**Install Dependencies:**
```bash
uv sync
```

**Terminal 1 - Backend API:**
```bash
uv run uvicorn api.server:app --port 8000
```

**Terminal 2 - Frontend Dev Server:**
```bash
cd web
npm install
npm run dev
```

Access the application at http://localhost:3000

### Production Mode

**Backend (with multiple workers):**
```bash
uv run uvicorn api.server:app --host 0.0.0.0 --port 8000 --workers 4
```

**Frontend (Build and Serve):**
```bash
cd web
npm run build
npx serve -s dist -l 3000
```

### Docker Compose (Full Stack)

```bash
docker compose up -d
```

This starts:
- Qdrant (port 6333)
- Langfuse (optional, port 3300)

## Feature-Specific Configuration

### Disable Voice Analysis

If you do not have a Hugging Face token, voice analysis is automatically disabled. No configuration change is required.

To explicitly disable:
```env
ENABLE_VOICE_ANALYSIS=false
```

### Use Gemini Instead of Ollama

```env
LLM_PROVIDER=gemini
GOOGLE_API_KEY=your_google_api_key
```

### Adjust Frame Extraction Interval

Default is 15 seconds. For more detailed analysis:
```env
FRAME_INTERVAL=5
```

For faster processing of long videos:
```env
FRAME_INTERVAL=30
```

### Enable Langfuse Observability

Langfuse provides distributed tracing and observability for your AI application.

#### Docker (Local Self-Hosted)

**Step 1: Start the Langfuse stack**
```bash
docker compose --profile docker up -d
```

This starts PostgreSQL, ClickHouse, Redis, MinIO, and Langfuse.

**Step 2: Create a Langfuse account**

1. Open http://localhost:3300
2. Click "Sign up" and create an account (any email works for local)
3. Create a new project (e.g., "AI-Media-Indexer")

**Step 3: Get your API keys**

1. In your project, go to **Settings → API Keys**
2. Copy the **Public Key** (starts with `pk-lf-...`)
3. Copy the **Secret Key** (starts with `sk-lf-...`)

**Step 4: Generate the OTEL auth header**

<!-- carousel -->
##### PowerShell
```powershell
$publicKey = "pk-lf-YOUR_KEY"
$secretKey = "sk-lf-YOUR_KEY"
$auth = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes("$publicKey`:$secretKey"))
Write-Host "OTEL_EXPORTER_OTLP_HEADERS=Authorization=Basic%20$auth"
```
<!-- slide -->
##### Bash
```bash
PUBLIC_KEY="pk-lf-YOUR_KEY"
SECRET_KEY="sk-lf-YOUR_KEY"
AUTH=$(echo -n "$PUBLIC_KEY:$SECRET_KEY" | base64)
echo "OTEL_EXPORTER_OTLP_HEADERS=Authorization=Basic%20$AUTH"
```
<!-- /carousel -->

**Step 5: Update your `.env` file**

```env
LANGFUSE_BACKEND=docker
LANGFUSE_DOCKER_HOST=http://localhost:3300
LANGFUSE_PUBLIC_KEY=pk-lf-YOUR_KEY
LANGFUSE_SECRET_KEY=sk-lf-YOUR_KEY
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:3300/api/public/otel/v1/traces
OTEL_EXPORTER_OTLP_HEADERS=Authorization=Basic%20YOUR_BASE64_AUTH_STRING
```

> **Important**: The `%20` in the header is URL-encoded space. Do not use a regular space.

**Step 6: Restart your backend**

The OTEL 401 errors should now be resolved and traces will appear in Langfuse.

#### Cloud (Langfuse Cloud)

```env
LANGFUSE_BACKEND=cloud
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
OTEL_EXPORTER_OTLP_ENDPOINT=https://cloud.langfuse.com/api/public/otel/v1/traces
OTEL_EXPORTER_OTLP_HEADERS=Authorization=Basic%20YOUR_BASE64_AUTH_STRING
```

## Troubleshooting

### "0 Results" After Ingestion

1. **Check Voice Analysis**: Ensure `HF_TOKEN` is set and valid
2. **Check Vision Analysis**: Ensure `OLLAMA_MODEL` is a vision model (`llava`, `llama3.2-vision`)
3. **Check Ollama**: Ensure Ollama is running (`ollama serve`)
4. **Check Logs**: Review `logs/app.log` for errors

### Langfuse 404 Error

If you see `Failed to export span batch code: 404`:
- Set `LANGFUSE_BACKEND=disabled` in your `.env` file
- Or start the Langfuse Docker container: `docker compose --profile docker up -d`

### Ollama Connection Error

Ensure Ollama is running:
```bash
ollama serve
```

### CUDA Out of Memory

Reduce batch size or switch to CPU:
```env
DEVICE_OVERRIDE=cpu
BATCH_SIZE=8
```

### Whisper Model Download Issues

Models are cached in `models/`. If download fails:
1. Check internet connection
2. Try a smaller model: `FALLBACK_MODEL_ID=openai/whisper-small`
