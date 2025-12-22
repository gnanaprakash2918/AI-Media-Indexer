# Developer Guide

## Quick Start Scripts

The project includes startup scripts that automate the full development environment setup.

### Basic Usage

<!-- carousel -->
#### Windows (PowerShell)
```powershell
.\start.ps1
```
<!-- slide -->
#### Linux / macOS (Bash)
```bash
chmod +x start.sh  # First time only
./start.sh
```
<!-- /carousel -->

### What the Scripts Do

| Step | Description |
|------|-------------|
| 1 | Navigate to project root |
| 2 | Check/create virtual environment with `uv` |
| 3 | Clean caches (pycache, face cache, pytest cache) |
| 4 | Optionally nuke Qdrant data |
| 5 | Stop Docker containers and remove orphans |
| 6 | Optionally pull latest Docker images |
| 7 | Start Docker services (Qdrant, Langfuse, etc.) |
| 8 | Start Ollama (if not running) |
| 9 | Launch backend and frontend in separate terminals |

### Available Options

| Option | Windows | Linux/macOS | Description |
|--------|---------|-------------|-------------|
| Skip Ollama | `-SkipOllama` | `--skip-ollama` | Don't start Ollama |
| Skip Docker | `-SkipDocker` | `--skip-docker` | Skip all Docker operations |
| Skip Cleanup | `-SkipClean` | `--skip-clean` | Don't clean cache directories |
| Recreate Venv | `-RecreateVenv` | `--recreate-venv` | Delete `.venv` and recreate with `uv sync` |
| Nuke Qdrant | `-NukeQdrant` | `--nuke-qdrant` | Delete `qdrant_data` directories |
| Pull Images | `-PullImages` | `--pull-images` | Run `docker compose pull` before starting |
| Integrated | `-Integrated` | `--integrated` | Run in single terminal (default: opens separate windows) |

> **Note**: The scripts auto-configure Langfuse OTEL authentication if `LANGFUSE_BACKEND=docker` and `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` are set in `.env`.

### Common Scenarios

#### Fresh Start (Full Reset)
Delete everything and start clean:

<!-- carousel -->
#### Windows
```powershell
.\start.ps1 -RecreateVenv -NukeQdrant -PullImages
```
<!-- slide -->
#### Linux/macOS
```bash
./start.sh --recreate-venv --nuke-qdrant --pull-images
```
<!-- /carousel -->

#### Update Docker Images Only
```powershell
.\start.ps1 -PullImages
```

#### Quick Restart (Skip Cleanup)
When you just want to restart services quickly:

<!-- carousel -->
#### Windows
```powershell
.\start.ps1 -SkipClean
```
<!-- slide -->
#### Linux/macOS
```bash
./start.sh --skip-clean
```
<!-- /carousel -->

#### Development Without Docker
If Docker services are already running elsewhere:

<!-- carousel -->
#### Windows
```powershell
.\start.ps1 -SkipDocker
```
<!-- slide -->
#### Linux/macOS
```bash
./start.sh --skip-docker
```
<!-- /carousel -->

### Service URLs After Startup

| Service | URL |
|---------|-----|
| Backend API | http://localhost:8000 |
| Frontend UI | http://localhost:5173 |
| Langfuse | http://localhost:3300 |
| Qdrant | http://localhost:6333 |

---

## MCP Inspector Guide

Here's a clean "from zero" flow you can follow every time to debug using the MCP Inspector.

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
