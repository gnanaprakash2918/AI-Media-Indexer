# Developer Guide

## MCP Inspector Guide

Here’s a clean “from zero” flow you can follow every time to debug using the MCP Inspector.

### 0. Pre-checks

1. **Navigate to Project Root**

<!-- carousel -->
#### PowerShell
```powershell
cd D:\AI-Media-Indexer
```
<!-- slide -->
#### CMD
```cmd
cd /d D:\AI-Media-Indexer
```
<!-- slide -->
#### Bash
```bash
cd /path/to/AI-Media-Indexer
```
<!-- /carousel -->

2. **Activate Virtual Environment**

<!-- carousel -->
#### PowerShell
```powershell
& .\.venv\Scripts\Activate.ps1
```
<!-- slide -->
#### CMD
```cmd
call .venv\Scripts\activate.bat
```
<!-- slide -->
#### Bash
```bash
source .venv/bin/activate
```
<!-- /carousel -->

3. **Ensure Qdrant + Ollama are running**
   - Qdrant: `localhost:6333`
   - Ollama: `ollama list` works

### 1. Start the MCP Inspector

Run the inspector which will auto-start your server.

<!-- carousel -->
#### PowerShell
```powershell
npx @modelcontextprotocol/inspector uv run python -m core.agent.server
```
<!-- slide -->
#### CMD
```cmd
npx @modelcontextprotocol/inspector uv run python -m core.agent.server
```
<!-- slide -->
#### Bash
```bash
npx @modelcontextprotocol/inspector uv run python -m core.agent.server
```
<!-- /carousel -->

> **Note**: This starts the Inspector web UI (typically port 3000 or similar) and pipes your server's STDIO to it.

### 2. Using the Inspector UI

1. Open the URL provided in the terminal (e.g., `http://localhost:6274/...`).
2. If not auto-connected, click **New Connection** -> **STDIO** -> Command: `uv`, Args: `run python -m core.agent.server`.
3. You should see tools like `search_media` and `ingest_media`.

### 3. Testing Tools

#### `search_media`
- **Query**: "red car"
- **Limit**: 5
- **Expectation**: JSON result with "visual_matches" or "dialogue_matches".

#### `ingest_media`
- **File Path**: Use absolute paths.
- **Example**: `D:\clips\test.mp4`

---

## Development Tools and Dependencies

### Build Tools (Windows)

<!-- carousel -->
#### PowerShell
```powershell
winget install --id Microsoft.VisualStudio.2022.BuildTools -e
winget install --id Kitware.CMake -e
winget install --id NinjaBuild.Ninja -e
```
<!-- slide -->
#### CMD
```cmd
winget install --id Microsoft.VisualStudio.2022.BuildTools -e
winget install --id Kitware.CMake -e
winget install --id NinjaBuild.Ninja -e
```
<!-- /carousel -->

### Git Workflow

<!-- carousel -->
#### PowerShell
```powershell
# View all branches
git branch -a
# Compare branches
git diff sprint-1..sprint-2 --stat
```
<!-- slide -->
#### CMD
```cmd
git branch -a
git diff sprint-1..sprint-2 --stat
```
<!-- slide -->
#### Bash
```bash
git branch -a
git diff sprint-1..sprint-2 --stat
```
<!-- /carousel -->
