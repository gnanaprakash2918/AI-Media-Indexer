
# Manual Testing Standard Operating Procedure (SOP)

This document outlines the manual verification steps to ensure the **AI Media Indexer** is Production Ready.

## 1. Helper Scripts
Use these scripts to manage the environment:
- `start.ps1`: Starts the entire stack (Frontend, Backend, Qdrant).

---

## 2. Ingestion Verification
**Goal**: Verify end-to-end processing (ASR, Vision, VectorDB).

1. **Prepare Data**:
   - Place a test video (e.g., `keladi.webm` or `test.mp4`) in a known folder.
2. **Launch System**:
   ```powershell
   ./start.ps1
   # Ensure "System Online".
   ```
3. **Trigger Ingest**:
   - Go to Web UI: `http://localhost:5173`.
   - Use "Upload" or drag-drop the file.
   - **Check logs** in the terminal running `start.ps1`.
     - Look for: `[Whisper] Transcribing...`.
     - Look for: `[Vision] Analyzed...`.
     - Look for: `[VectorDB] Upserted...`.

---

## 3. Search Verification
**Goal**: Verify Hybrid Search, VideoRAG, and Intersection Logic.

1. **Complex Query**:
   - Search: `"Prakash bowling"`
   - **Expected**: Results where visual is "bowling" AND identity/person is "Prakash" (if trained).
2. **Intersection Test ("Rain + Smile")**:
   - Search: `"Someone smiling while it rains"`
   - **Log Check**: Look for `[VideoRAG] Performing Strict Intersection` in backend logs.
   - **Expected**: Valid timestamp range returned.
3. **Agent Search**:
   - Click "Ask Agent" or use Agent UI.
   - Query: `"Summarize the events involving Prakash"`
   - **Expected**: Text summary generated from retrieved clips.

---

## 4. Identity & Clustering
**Goal**: Verify Face Clustering accuracy.

1. **Check Clusters**:
   - Go to `http://localhost:5173/faces` (or equivalent UI tab).
   - Ensure you do NOT see 40+ clusters for the same person (Bug fixed in Phase 12).
2. **Merge (If needed)**:
   - Select multiple clusters -> Click "Merge".
   - Verify they combine into one ID.

## 5. Observability
1. **Health Check**:
   - Visit: `http://localhost:8000/health`.
   - Check JSON: `observability: "connected"` (if Langfuse configured) and `asr_mode`.

---

## Troubleshooting
- **OOM Errors**: Check `config.py` Hardware Profile. Reduce `batch_size`.
- **Docker Fail**: Ensure Docker Desktop is running.
