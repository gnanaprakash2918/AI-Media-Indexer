# Troubleshooting & Developer Notes

This document contains troubleshooting tips, common errors, and useful snippets.

## Model Context Protocol (MCP) Server

### Testing FastMCP Server (`core/agent/server.py`)

The MCP server exposes `search_media` and `ingest_media` tools over stdio for LLM integration.

```bash
uv run python core/agent/server.py
```

## Common Errors & Diagnostics

### Database Connectivity
- Ensure Qdrant vector database is running on port 6333 (`http://localhost:6333`).
- For vector DB troubleshooting:
  ```bash
  python3 tools/diagnose_search.py
  ```
