# Comprehensive Testing Guide

This document details the exhaustive testing strategy for the AI Media Indexer. Every single source file in this repository is mapped to a dedicated test file, ensuring 100% component coverage.

## 🛡️ Testing Philosophy

- **Exhaustive**: Every file, function, and logic path is tested.
- **Isolated**: Unit tests mock all dependencies (network, DB, FS).
- **Integrated**: E2E tests verify the full pipeline with real/mocked services.
- **Dual-Mode**: Tests run in both `Mock` (fast, fake LLM) and `Actual` (Ollama) modes.

---

## 📂 Source-to-Test Mapping (100% Coverage)

Every source file in `core/` has a corresponding test in `tests/`.

| Component | Source File | Test File | Type |
|-----------|-------------|-----------|------|
| **Agent** | `core/agent/card.py` | `tests/agent/test_card.py` | Unit |
| | `core/agent/handler.py` | `tests/agent/test_handler.py` | Unit |
| | `core/agent/server.py` | `tests/agent/test_mcp_server.py` | Integration |
| | `core/agent/a2a_server.py` | `tests/agent/test_a2a_server.py` | Integration |
| **Ingestion** | `core/ingestion/pipeline.py` | `tests/ingestion/test_pipeline.py` | Unit/Int |
| | `core/ingestion/scanner.py` | `tests/ingestion/test_scanner.py` | Unit |
| **Processing** | `core/processing/extractor.py` | `tests/processing/test_extractor.py` | Unit |
| | `core/processing/identity.py` | `tests/processing/test_identity.py` | Unit |
| | `core/processing/metadata.py` | `tests/processing/test_metadata.py` | Unit |
| | `core/processing/prober.py` | `tests/processing/test_prober.py` | Unit |
| | `core/processing/text_utils.py` | `tests/processing/test_text_utils.py` | Unit |
| | `core/processing/transcriber.py` | `tests/processing/test_transcriber.py` | Unit |
| | `core/processing/vision.py` | `tests/processing/test_vision.py` | Unit |
| | `core/processing/voice.py` | `tests/processing/test_voice.py` | Unit |
| **Retrieval** | `core/retrieval/search.py` | `tests/retrieval/test_search.py` | Unit |
| **Storage** | `core/storage/db.py` | `tests/storage/test_db.py` | Integration |
| **Utils** | `core/utils/frame_sampling.py` | `tests/utils/test_frame_sampling.py` | Unit |
| | `core/utils/logger.py` | `tests/utils/test_logger.py` | Unit |
| | `core/utils/observability.py` | `tests/utils/test_observability.py` | Unit |
| | `core/utils/progress.py` | `tests/utils/test_progress.py` | Unit |
| | `core/utils/resource.py` | `tests/utils/test_resource.py` | Unit |
| | `core/utils/retry.py` | `tests/utils/test_retry.py` | Unit |
| **Config** | `config.py` | `tests/test_config.py` | Unit |

---

## 🚀 Running Tests

### 1. The Master Script (Recommended)
Use the `run_tests.py` script for a robust, cleaned-up execution.

```bash
# Run ALL tests with MOCK provider (Fast, Default)
uv run python scripts/run_tests.py

# Run ALL tests with OLLAMA provider (Actual Intelligence)
uv run python scripts/run_tests.py --llm-provider=ollama
```

### 2. Manual pytest Execution
You can run specific slices of the suite using standard pytest commands.

```bash
# Run only UNIT tests (Fastest)
uv run pytest -m unit

# Run only INTEGRATION tests (Requires Docker/Qdrant)
uv run pytest -m integration

# Run only END-TO-END tests
uv run pytest -m e2e

# Run specific file
uv run pytest tests/agent/test_handler.py -v
```

---

## 📊 Coverage Reports
Tests automatically generate coverage reports.

- **Terminal**: A summary is printed after execution.
- **HTML**: Detailed line-by-line coverage is saved to `reports/coverage_*/index.html`.

To run manual coverage check:
```bash
uv run pytest --cov=core --cov-report=term-missing tests/
```

---

## 🧹 Cleanup
The system automatically cleans up artifacts. To force clean:

```bash
# Windows PowerShell
del qdrant_data, qdrant_data_embedded, logs, reports -Recurse -Force
```

---

## ❓ Troubleshooting

| Component | Error | Fix |
|-----------|-------|-----|
| **Qdrant** | `500 Internal Server Error` / `File exists` | Only occurs on Windows Docker volumes. **Fix**: Tests now use specific cleanup fixtures or memory backend. |
| **Ollama** | `Connection refused` | Ensure Ollama is running (`ollama serve`). |
| **Unit Tests** | `AttributeError: sensors_temperatures` | (Windows) `psutil` mocks handle missing hardware sensors gracefully. |
