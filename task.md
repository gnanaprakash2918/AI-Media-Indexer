[ ] Phase 1: Dependency & Stability (Immediate)
    [x] Run `uv sync` to generate a valid `uv.lock` and `requirements.txt` with resolvable versions (Fix invalid Pillow/Transformers versions).
    [x] Verify `Dockerfile.cuda` builds successfully with the new lockfile.

[ ] Phase 2: Core Refactoring (High Impact)
    [x] **Refactor `core/ingestion/pipeline.py`**
        [x] Extract `AudioProcessor` (ASR, Diarization).
        [x] Extract `VideoProcessor` (Frame Sampling, VLM).
        [x] Extract `IdentityProcessor` (Face/Voice Clustering).
    [x] **Refactor `core/storage/db.py`**
        [x] Extract `QdrantHandler` (Pure DB ops).
        [x] Extract `ClusterManager` (Biometric logic).
        [x] Extract `ModelLoader` (SentenceTransformer logic).

[ ] Phase 3: Security & API Hardening
    [x] Bind `uvicorn` to `127.0.0.1` by default in `main.py` (prevent LAN exposure).
    [x] Add basic API Key authentication middleware (optional, configurable).
    [x] Fix `tkinter` thread safety in `api/routes/system.py` (Replaced with API).

[ ] Phase 4: Performance Optimization
    [x] Implement true async processing for `IngestionPipeline` using a ProcessPool (SKIPPED: Avoid overengineering).
    [x] Create `docker-compose.lite.yaml` for low-resource environments.

[x] Phase 5: uPDATE SCRIPT.PS1
    [x] at end of every phase make sure to update SCRIPT.PS1 BASED ON CHANGES if needed