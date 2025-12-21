# Comprehensive Testing Guide

This document details the exhaustive testing strategy for the AI Media Indexer. Every single source file in this repository is mapped to a dedicated test file and covered by various testing layers, ensuring 100% component coverage and production readiness.

## Testing Philosophy

- **Exhaustive**: Every file, function, and logic path is tested.
- **Isolated**: Unit tests mock all dependencies (network, DB, FS).
- **Integrated**: E2E tests verify the full pipeline with real/mocked services.
- **Production-Ready**: Includes Smoke, Performance, and Acceptance testing.
- **Dual-Mode**: Tests run in both `Mock` (fast, fake LLM) and `Actual` (Ollama) modes.

---

## Source-to-Test Mapping

The codebase tests are consolidated into domain-specific test files for better maintainability.

| Component | Source Files | Test File | Type |
|-----------|-------------|-----------|------|
| **Agent** | `core/agent/*.py` | `tests/test_agent.py` | Unit & Integration |
| **Ingestion** | `core/ingestion/*.py` | `tests/test_ingestion.py` | Unit & Integration |
| **Processing** | `core/processing/*.py` | `tests/test_processing.py` | Unit |
| **Storage / Search** | `core/storage/`, `core/retrieval/` | `tests/test_storage.py` | Integration |
| **Utils** | `core/utils/*.py`, `config.py` | `tests/test_utils.py` | Unit |
| **End-to-End** | *Whole System* | `tests/test_e2e.py` | E2E |
| **Sanity** | *System Boot* | `tests/test_smoke.py` | Smoke |
| **Performance** | *Critical Paths* | `tests/test_performance.py` | Benchmark |
| **Acceptance** | *User Stories* | `tests/test_acceptance.py` | Acceptance |

---

## Key Test Types

1. **Unit Tests**: Low-level verification of individual functions/classes (e.g., `test_processing.py`).
2. **Integration Tests**: Verify interactions between components, like Pipeline -> VectorDB (e.g., `test_ingestion.py`).
3. **End-to-End (E2E) Tests**: Full workflow simulations from file input to search result (e.g., `test_e2e.py`).
4. **Smoke Tests**: Quick sanity checks to ensure the system environment is sound and components can instantiate (e.g., `test_smoke.py`).
5. **Performance Tests**: benchmarks for critical operations like VectorDB insertions (e.g., `test_performance.py`).
6. **Acceptance Tests**: High-level user story verifications (e.g., `test_acceptance.py`).

---

## Running Tests

### 1. The Master Script (Recommended)
Use the `run_tests.py` script for a robust, cleaned-up execution.

```bash
# Run ALL tests with MOCK provider (Fast, Default)
uv run python scripts/run_tests.py

# Run ALL tests with OLLAMA provider (Actual Intelligence)
# Note: Requires Ollama running locally
uv run python scripts/run_tests.py --llm-provider=ollama

# Force cleanup of old data before running
uv run python scripts/run_tests.py --force
```

### 2. Manual pytest Execution
You can run specific slices of the suite using standard pytest commands.

```bash
# Run only SMOKE tests
uv run pytest -m smoke

# Run only PERFORMANCE tests
uv run pytest -m performance

# Run only ACCEPTANCE tests
uv run pytest -m acceptance

# Run E2E tests
uv run pytest -m e2e

# Run specific file
uv run pytest tests/test_agent.py -v
```

---

## Coverage Reports
Tests automatically generate coverage reports.

- **Terminal**: A summary is printed after execution.
- **HTML**: Detailed line-by-line coverage is saved to `reports/coverage_*/index.html`.

To run manual coverage check (Note: May be slow):
```bash
uv run pytest --cov=core --cov-report=term-missing tests/
```

---

## Cleanup
The system automatically cleans up artifacts. To force clean:

```bash
# Windows PowerShell
uv run python scripts/run_tests.py --dry-run
# Then run without dry-run to execute
```

---

## Troubleshooting

| Component | Error | Fix |
|-----------|-------|-----|
| **Qdrant** | `500 Internal Server Error` / `File exists` | Only occurs on Windows Docker volumes. **Fix**: Tests now use specific cleanup fixtures or memory backend. |
| **Ollama** | `Connection refused` | Ensure Ollama is running (`ollama serve`). |
| **Performance** | Tests running slowly | Use `-p no:cov` to disable coverage if not needed. |
