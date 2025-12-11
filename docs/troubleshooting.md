# Troubleshooting & Developer Notes

This document contains troubleshooting tips, common errors, and useful snippets.

## Agent-to-Agent (A2A) SDK & Server

### Testing `a2a_server`

Use `curl` to test the server endpoint.

<!-- carousel -->
#### Bash / WSL
```bash
curl -X POST "http://localhost:8000/a2a/v1/message:send" \
  -H "Content-Type: application/json" \
  -d '{
    "message": {
      "role": "ROLE_USER",
      "message_id": "msg-1",
      "content": [
        {
          "text": "Find the red car"
        }
      ]
    }
  }'
```
<!-- slide -->
#### PowerShell
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/a2a/v1/message:send" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{
    "message": {
      "role": "ROLE_USER",
      "message_id": "msg-1",
      "content": [
        {
          "text": "Find the red car"
        }
      ]
    }
  }'
```
<!-- /carousel -->

### Installation/Update Commands

<!-- carousel -->
#### PowerShell
```powershell
# Centralize cache and run server
$env:PYTHONPYCACHEPREFIX="D:\AI-Media-Indexer\.cache\pycache"
uv run python -m core.agent.a2a_server

# Upgrade SDK
uv pip install --upgrade "a2a-sdk[all]"
```
<!-- slide -->
#### CMD
```cmd
set PYTHONPYCACHEPREFIX=D:\AI-Media-Indexer\.cache\pycache
uv run python -m core.agent.a2a_server

uv pip install --upgrade "a2a-sdk[all]"
```
<!-- slide -->
#### Bash
```bash
export PYTHONPYCACHEPREFIX="/mnt/d/AI-Media-Indexer/.cache/pycache"
uv run python -m core.agent.a2a_server

uv pip install --upgrade "a2a-sdk[all]"
```
<!-- /carousel -->

## Common Errors

### Protobuf Enum Parsing Error
**Error**: `Invalid enum value USER for enum type a2a.v1.Role`

**Cause**: `google.protobuf.json_format.Parse` requires exact enum identifiers (e.g., `ROLE_USER`).

**Resolution**:
1. Check `http://localhost:8000/openapi.json` for allowed values.
2. Verify `.proto` definitions.
