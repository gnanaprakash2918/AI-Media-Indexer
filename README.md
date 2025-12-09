# AI-Media-Indexer
AI engine for searching and navigating large media libraries using text, visuals, and audio.

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

---

## Developer Guide

## MCP Inspector Guide

Here’s a clean “from zero” flow you can follow every time.

---

### 0. Pre-checks (one-time-ish)

Do this once (or when things feel weird):

1. **Make sure you’re in the project root**

```powershell
cd D:\AI-Media-Indexer
```

2. **Activate your venv**

```powershell
& .\.venv\Scripts\Activate.ps1
```

3. **Make sure Qdrant + Ollama are running**

   - Qdrant (Docker): container up and listening on `localhost:6333`
   - Ollama: `ollama list` works in a terminal

You don’t need to run `uv run python -m core.agent.server` manually anymore. The Inspector will do that.

---

### 1. Start the MCP Inspector (this also starts your server)

In **PowerShell**, from `D:\AI-Media-Indexer`:

```powershell
npx @modelcontextprotocol/inspector uv run python -m core.agent.server
```

What this does:

- Starts the Inspector web UI
- Starts your MCP server via: `uv run python -m core.agent.server`
- Shows logs (from `log(...)`) in this same terminal

You should see something like:

- `MCP Inspector is up and running at: http://localhost:6274/?MCP_PROXY_AUTH_TOKEN=...`
- Some “New STDIO connection request” lines
- Optionally your `[MCP] MediaIndexer server starting...` log if you added that

> Leave this terminal **open**. It’s running both the Inspector proxy and your MCP server.

---

### 2. Open the Inspector UI

1. Your browser should open automatically.
   If not, copy the printed URL, e.g.:

   ```
   http://localhost:6274/?MCP_PROXY_AUTH_TOKEN=...
   ```

2. In the UI, on the **left** you’ll see:

   - A **Connections** section
   - An entry for the `uv run python -m core.agent.server` command (if it auto-detected)

If it doesn’t auto-connect:

- Click **“New Connection”**
- Choose **STDIO**
- Command: `uv`
- Args: `run python -m core.agent.server`
- Click **Connect**

If the connection succeeds, you’ll see:

- The connection icon turn green
- A list of **Tools** like:

  - `search_media`
  - `ingest_media`

---

### 3. Test `search_media` (fast sanity check)

This doesn’t re-index anything, it just queries Qdrant.

1. In the Inspector UI, click the **Tools** tab.

2. Click on **`search_media`**.

3. In the “Inputs” panel, fill:

   - `query`: something like
     `red car` or `hello` or `arguing in a kitchen`
   - `limit`: `5` (or keep default)

4. Click **Run Tool**.

If everything is wired correctly:

- The **Result** panel will show a JSON object:

  ```json
  {
    "visual_matches": [
      {
        "score": "0.92",
        "time": "123.45s",
        "file": "D:\\SomeMovie.mkv",
        "content": "A red car driving through traffic"
      }
    ],
    "dialogue_matches": [
      {
        "score": "0.88",
        "time": "321.00s",
        "file": "D:\\SomeMovie.mkv",
        "content": "Character says: 'We need to talk'"
      }
    ]
  }
  ```

- The terminal where you ran `npx @modelcontextprotocol/inspector ...` will show logs like:

  ```text
  [Search] Querying: 'red car'...
  ```

If you get empty arrays, it just means Qdrant doesn’t have much indexed yet.

---

### 4. Test `ingest_media` (use a small file first)

Because ingestion is heavy, use a **short clip** (1–5 min) for testing, not the 5.8 GB movie.

1. In the Inspector, click **`ingest_media`**.

2. Fill the inputs:

   - `file_path`:
     Example:
     `D:\clips\test.mp4`
     (use the real path; in the Inspector UI you don’t need to escape backslashes)
   - `media_type` (optional): `movie` or `personal` or leave `unknown`

3. Click **Run Tool**.

What to expect:

- The tool call will stay “pending” while it runs.

- In the **Result** panel, when done, you’ll see something like:

  ```json
  {
    "file_path": "D:\\clips\\test.mp4",
    "media_type_hint": "movie",
    "message": "Ingestion complete."
  }
  ```

- In your terminal logs you’ll see the whole pipeline:
  Once that completes, you can go back to **`search_media`** and query for something that appears in that test clip.

---

### 5. When you’re done

- To stop everything, go to the terminal running `npx @modelcontextprotocol/inspector` and hit **Ctrl + C**.
- That shuts down:

  - The Inspector web UI
  - The MCP proxy
  - The `uv run python -m core.agent.server` subprocess

Next time you want to use it, just repeat:

```powershell
cd D:\AI-Media-Indexer
& .\.venv\Scripts\Activate.ps1
npx @modelcontextprotocol/inspector uv run python -m core.agent.server
```

Then:

- Open the browser URL it prints
- Use `search_media` and `ingest_media` from the Inspector