#!/bin/bash
#
# Full system startup script for AI-Media-Indexer (Linux/macOS)
#
# Usage: ./start.sh [options]
#   --skip-ollama        Skip Ollama startup
#   --skip-docker        Skip Docker operations
#   --skip-clean         Skip cache cleanup
#   --recreate-venv      Remove .venv and recreate with uv sync
#   --nuke-qdrant        Delete Qdrant data directories
#   --pull-images        Pull latest Docker images before starting
#   --integrated         Run in single terminal (backend foreground, frontend background)
#   --no-interactive     Skip interactive menu, use defaults

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
GRAY='\033[0;90m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# Parse arguments
SKIP_OLLAMA=false
SKIP_DOCKER=false
SKIP_CLEAN=false
RECREATE_VENV=false
NUKE_QDRANT=false
PULL_IMAGES=false
INTEGRATED=false
NO_INTERACTIVE=false
AGENT_MODE=false
AGENT_TASK="analyze"
ANY_FLAGS=false

while [[ $# -gt 0 ]]; do
    ANY_FLAGS=true
    case $1 in
        --skip-ollama) SKIP_OLLAMA=true; shift ;;
        --skip-docker) SKIP_DOCKER=true; shift ;;
        --skip-clean) SKIP_CLEAN=true; shift ;;
        --recreate-venv) RECREATE_VENV=true; shift ;;
        --nuke-qdrant) NUKE_QDRANT=true; shift ;;
        --pull-images) PULL_IMAGES=true; shift ;;
        --integrated) INTEGRATED=true; shift ;;
        --no-interactive|-y) NO_INTERACTIVE=true; shift ;;
        --agent) AGENT_MODE=true; shift ;;
        --task) AGENT_TASK="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "${CYAN}>>> AI-Media-Indexer Full System Startup${NC}"
echo ""

# Interactive menu (unless --no-interactive or any flags are passed)
if [ "$NO_INTERACTIVE" = false ] && [ "$ANY_FLAGS" = false ]; then
    echo -e "${GRAY}============================================================${NC}"
    echo -e "${CYAN}  STARTUP OPTIONS${NC}"
    echo -e "${GRAY}============================================================${NC}"
    echo ""
    
    # Detect current system state for smart recommendations
    HAS_VENV="NOT FOUND"
    [ -d "$SCRIPT_DIR/.venv" ] && HAS_VENV="exists"
    
    HAS_QDRANT="fresh"
    if [ -d "$SCRIPT_DIR/qdrant_data" ] || [ -d "$SCRIPT_DIR/qdrant_data_embedded" ]; then
        HAS_QDRANT="exists"
    fi
    
    CACHE_SIZE="0"
    if [ -d "$SCRIPT_DIR/.cache" ]; then
        CACHE_SIZE=$(du -sm "$SCRIPT_DIR/.cache" 2>/dev/null | cut -f1 || echo "0")
    fi
    
    echo -e "${YELLOW}  Current system state:${NC}"
    echo -e "${GRAY}    - Virtual env: $HAS_VENV${NC}"
    echo -e "${GRAY}    - Qdrant data: $HAS_QDRANT${NC}"
    echo -e "${GRAY}    - Cache size:  ${CACHE_SIZE} MB${NC}"
    echo ""
    
    echo -e "${CYAN}  Choose startup mode:${NC}"
    echo ""
    echo -e "${GREEN}  [1] Quick Start (RECOMMENDED)${NC}"
    echo -e "${GRAY}      - Keeps caches and data intact${NC}"
    echo -e "${GRAY}      - Fastest startup time${NC}"
    echo ""
    echo -e "${YELLOW}  [2] Fresh Start${NC}"
    echo -e "${GRAY}      - Clears all caches${NC}"
    echo -e "${GRAY}      - Keeps Qdrant data (your indexed videos)${NC}"
    echo ""
    echo -e "${RED}  [3] Complete Reset${NC}"
    echo -e "${GRAY}      - Clears caches AND Qdrant data${NC}"
    echo -e "${GRAY}      - You'll need to re-ingest all videos${NC}"
    echo ""
    echo -e "${MAGENTA}  [4] Dev Mode${NC}"
    echo -e "${GRAY}      - Recreate virtual environment${NC}"
    echo -e "${GRAY}      - Pull latest Docker images${NC}"
    echo ""
    echo -e "${WHITE}  [5] Custom (use command line flags)${NC}"
    echo -e "${GRAY}      - Run: ./start.sh --help for options${NC}"
    echo ""
    echo -e "${RED}  [6] NUCLEAR (Reset + Dev Mode)${NC}"
    echo -e "${GRAY}      - Wipe ALL data + recreate venv + pull images${NC}"
    echo -e "${GRAY}      - Complete fresh start from scratch${NC}"
    echo ""
    
    read -p "Enter choice [1-6] or press Enter for Quick Start: " choice
    
    case $choice in
        1)
            SKIP_CLEAN=true
            echo -e "\n${GREEN}  >> Quick Start selected${NC}"
            ;;
        2)
            SKIP_CLEAN=false
            echo -e "\n${YELLOW}  >> Fresh Start selected (clearing caches)${NC}"
            ;;
        3)
            SKIP_CLEAN=false
            NUKE_QDRANT=true
            echo -e "\n${RED}  >> Complete Reset selected (clearing everything)${NC}"
            ;;
        4)
            RECREATE_VENV=true
            PULL_IMAGES=true
            echo -e "\n${MAGENTA}  >> Dev Mode selected (recreating venv, pulling images)${NC}"
            ;;
        5)
            echo -e "\n${CYAN}  Startup flags:${NC}"
            echo -e "${GRAY}    --skip-ollama      Skip Ollama startup${NC}"
            echo -e "${GRAY}    --skip-docker      Skip Docker operations${NC}"
            echo -e "${GRAY}    --skip-clean       Skip cache cleanup${NC}"
            echo -e "${GRAY}    --recreate-venv    Recreate virtual environment${NC}"
            echo -e "${GRAY}    --nuke-qdrant      Delete all indexed data${NC}"
            echo -e "${GRAY}    --pull-images      Pull latest Docker images${NC}"
            echo -e "${GRAY}    --integrated       Run in single terminal${NC}"
            echo -e "${GRAY}    --no-interactive   Skip this menu${NC}"
            echo ""
            echo -e "${WHITE}  Example: ./start.sh --skip-docker --skip-ollama${NC}"
            echo ""
            echo -e "${CYAN}  Video Conversion Tool (for faster playback):${NC}"
            echo -e "${GRAY}    # Convert single file${NC}"
            echo -e "${GRAY}    python -m tools.convert \"/path/to/video.webm\"${NC}"
            echo ""
            echo -e "${GRAY}    # Batch convert entire directory (4 parallel workers)${NC}"
            echo -e "${GRAY}    python -m tools.convert \"/Downloads/Videos\" --workers 4${NC}"
            echo ""
            echo -e "${GRAY}    # Faster conversion (lower quality)${NC}"
            echo -e "${GRAY}    python -m tools.convert \"/path/to/video.webm\" --preset ultrafast${NC}"
            echo ""
            exit 0
            ;;
        6)
            SKIP_CLEAN=false
            NUKE_QDRANT=true
            RECREATE_VENV=true
            PULL_IMAGES=true
            echo -e "\n${RED}  >> NUCLEAR selected (wiping everything + fresh venv + pulling images)${NC}"
            ;;
        "")
            SKIP_CLEAN=true
            echo -e "\n${GREEN}  >> Quick Start (default)${NC}"
            ;;
        *)
            SKIP_CLEAN=true
            echo -e "\n${YELLOW}  >> Invalid choice, using Quick Start${NC}"
            ;;
    esac
    echo ""
fi

echo -e "${YELLOW}[1/9] Working directory: $SCRIPT_DIR${NC}"

# Handle virtual environment
if [ "$RECREATE_VENV" = true ]; then
    echo ""
    echo -e "${YELLOW}[2/9] Recreating virtual environment...${NC}"
    
    if [ -d ".venv" ]; then
        echo -e "${GRAY}  Removing existing .venv...${NC}"
        rm -rf .venv
    fi
    
    echo -e "${GRAY}  Creating new venv with uv...${NC}"
    uv venv
    
    echo -e "${GRAY}  Syncing dependencies with uv...${NC}"
    uv sync
    
    echo -e "${GREEN}  Virtual environment recreated!${NC}"
else
    # Check if venv exists, if not create it
    if [ ! -d ".venv" ]; then
        echo ""
        echo -e "${YELLOW}[2/9] Creating virtual environment (not found)...${NC}"
        uv venv
        uv sync
        echo -e "${GREEN}  Virtual environment created and synced!${NC}"
    else
        echo -e "${GRAY}[2/9] Virtual environment exists${NC}"
    fi
fi

# Clean caches (project only, not .venv)
if [ "$SKIP_CLEAN" = false ]; then
    echo ""
    echo -e "${YELLOW}[3/9] Cleaning caches...${NC}"
    
    CACHE_DIRS=(".cache" ".face_cache" ".pytest_cache" "__pycache__")
    
    for dir in "${CACHE_DIRS[@]}"; do
        if [ -d "$dir" ]; then
            echo -e "${GRAY}  Removing: $dir${NC}"
            rm -rf "$dir"
        fi
    done
    
    # Clean log files (keep directory)
    rm -f logs/*.log 2>/dev/null || true
    
    # Clean pycache recursively (EXCLUDE .venv and node_modules)
    find . -type d -name "__pycache__" \
        -not -path "./.venv/*" \
        -not -path "./node_modules/*" \
        -not -path "./.git/*" \
        -exec rm -rf {} + 2>/dev/null || true
    
    find . -type f -name "*.pyc" \
        -not -path "./.venv/*" \
        -not -path "./node_modules/*" \
        -delete 2>/dev/null || true
    
    echo -e "${GREEN}  Cache cleanup complete!${NC}"
else
    echo -e "${GRAY}[3/9] Skipping cache cleanup (--skip-clean)${NC}"
fi

# Configure OTEL auth for Langfuse (if needed)
ENV_FILE="$SCRIPT_DIR/.env"
if [ -f "$ENV_FILE" ]; then
    # Check if Langfuse is enabled
    LANGFUSE_BACKEND=$(grep -E '^LANGFUSE_BACKEND\s*=' "$ENV_FILE" | head -1 | sed 's/.*=\s*//' | tr -d '"')
    
    if [ "$LANGFUSE_BACKEND" = "docker" ] || [ "$LANGFUSE_BACKEND" = "cloud" ]; then
        echo ""
        echo -e "${YELLOW}[3.5/9] Checking Langfuse OTEL configuration...${NC}"
        
        # Extract keys
        PUBLIC_KEY=$(grep -E '^LANGFUSE_PUBLIC_KEY\s*=' "$ENV_FILE" | head -1 | sed 's/.*=\s*"\?\([^"]*\)"\?/\1/' | tr -d ' ')
        SECRET_KEY=$(grep -E '^LANGFUSE_SECRET_KEY\s*=' "$ENV_FILE" | head -1 | sed 's/.*=\s*"\?\([^"]*\)"\?/\1/' | tr -d ' ')
        CURRENT_HEADER=$(grep -E '^OTEL_EXPORTER_OTLP_HEADERS\s*=' "$ENV_FILE" | head -1 | sed 's/.*=\s*//')
        
        if [ -n "$PUBLIC_KEY" ] && [ -n "$SECRET_KEY" ]; then
            # Generate expected base64 auth (use space, not %20)
            EXPECTED_BASE64=$(echo -n "$PUBLIC_KEY:$SECRET_KEY" | base64 | tr -d '\n')
            EXPECTED_HEADER="Authorization=Basic $EXPECTED_BASE64"
            
            if [ "$CURRENT_HEADER" != "$EXPECTED_HEADER" ]; then
                echo -e "${GRAY}  Updating OTEL auth header with Langfuse credentials...${NC}"
                
                # Update or add the header
                if grep -q '^OTEL_EXPORTER_OTLP_HEADERS' "$ENV_FILE"; then
                    sed -i "s|^OTEL_EXPORTER_OTLP_HEADERS.*|OTEL_EXPORTER_OTLP_HEADERS=$EXPECTED_HEADER|" "$ENV_FILE"
                else
                    # Add after OTEL_EXPORTER_OTLP_ENDPOINT if it exists
                    if grep -q 'OTEL_EXPORTER_OTLP_ENDPOINT' "$ENV_FILE"; then
                        sed -i "/OTEL_EXPORTER_OTLP_ENDPOINT/a OTEL_EXPORTER_OTLP_HEADERS=$EXPECTED_HEADER" "$ENV_FILE"
                    else
                        echo "OTEL_EXPORTER_OTLP_HEADERS=$EXPECTED_HEADER" >> "$ENV_FILE"
                    fi
                fi
                echo -e "${GREEN}  OTEL auth configured!${NC}"
            else
                echo -e "${GREEN}  OTEL auth already configured${NC}"
            fi
        else
            echo -e "${YELLOW}  Langfuse keys not found in .env - OTEL tracing may not work${NC}"
            echo -e "${GRAY}  Add LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY from Langfuse Settings${NC}"
        fi
    fi
fi

# Nuke Qdrant data
if [ "$NUKE_QDRANT" = true ]; then
    echo ""
    echo -e "${RED}[4/9] Performing Complete Data Reset...${NC}"
    
    # 1. Kill any existing backend/frontend processes to release locks
    echo -e "${GRAY}  Terminating existing AI-Media-Indexer processes...${NC}"
    pkill -f "python.*api/server.py" || true
    pkill -f "node.*vite" || true

    # 2. Stop Docker and remove named volumes
    echo -e "${GRAY}  Stopping Docker containers and removing volumes...${NC}"
    # -v is essential to wipe named volumes
    if docker compose down -v --remove-orphans; then
        echo -e "${GREEN}  Docker services stopped and volumes removed.${NC}"
    else
        echo -e "${YELLOW}  Warning: Docker down failed. Attempting force removal...${NC}"
    fi
    
    # Force remove specific containers to ensure nothing is stuck
    CONTAINERS=("media_agent_qdrant" "media_agent_postgres" "media_agent_minio" "media_agent_redis" "media_agent_clickhouse" "media_agent_langfuse" "media_agent_langfuse_worker" "media_agent_createbuckets")
    for container in "${CONTAINERS[@]}"; do
        docker rm -f "$container" >/dev/null 2>&1 || true
    done

    # 3. Wipe local bind-mount directories
    DATA_DIRS=("qdrant_data" "qdrant_data_embedded" "thumbnails" "logs" ".cache")
    
    echo ""
    read -p "  [?] Also wipe Langfuse/Postgres data (Resets API keys)? (y/N): " wipe_langfuse
    if [[ "$wipe_langfuse" =~ ^[yY] ]]; then
        DATA_DIRS+=("postgres_data")
        echo -e "${YELLOW}  >> Including Postgres/Langfuse in wipe list.${NC}"
    else
        echo -e "${GREEN}  >> Preserving Postgres/Langfuse data (Keys kept safe).${NC}"
    fi

    for dir in "${DATA_DIRS[@]}"; do
        if [ -d "$dir" ]; then
            echo -e "${GRAY}  Removing: $dir${NC}"
            if rm -rf "$dir"; then
                 # Verify deletion
                 if [ -d "$dir" ]; then
                     echo -e "${RED}  ERROR: Failed to delete $dir. Folder still exists.${NC}"
                 else
                     echo -e "${GREEN}  $dir deleted successfully.${NC}"
                 fi
            else
                 echo -e "${RED}  ERROR: Failed to delete $dir.${NC}"
            fi
        fi
    done
    
    echo -e "${GREEN}  Data reset complete!${NC}"
else
    echo -e "${GRAY}[4/9] Keeping Qdrant data (use --nuke-qdrant to delete)${NC}"
fi

# Check and Start Docker
if [ "$SKIP_DOCKER" = false ]; then
    echo ""
    echo -e "${YELLOW}[5/9] Checking Docker status...${NC}"
    
    if ! docker info > /dev/null 2>&1; then
        echo -e "${YELLOW}  Docker Daemon is not running. Attempting to start...${NC}"
        
        if [[ "$OSTYPE" == "darwin"* ]]; then
            open -a Docker
            echo -e "${GRAY}  Launched Docker Desktop (macOS). Waiting for daemon...${NC}"
        elif command -v systemctl &> /dev/null; then
            echo -e "${GRAY}  Attempting: sudo systemctl start docker${NC}"
            sudo systemctl start docker || true
        elif command -v service &> /dev/null; then
             echo -e "${GRAY}  Attempting: sudo service docker start${NC}"
             sudo service docker start || true
        else
             echo -e "${RED}  Could not auto-start Docker. Please start it manually.${NC}"
             # Don't exit, just let it fail at next step or user interaction
        fi
        
        # Wait for Docker (up to 60s)
        RETRIES=60
        while [ $RETRIES -gt 0 ]; do
            echo -n "."
            sleep 2
            if docker info > /dev/null 2>&1; then
                echo ""
                echo -e "${GREEN}  Docker Daemon is ready!${NC}"
                break
            fi
            RETRIES=$((RETRIES-1))
        done
        
        if ! docker info > /dev/null 2>&1; then
             echo ""
             echo -e "${RED}  ERROR: Docker failed to become ready.${NC}"
             exit 1
        fi
    else
        echo -e "${GREEN}  Docker Daemon is already running${NC}"
    fi

    echo ""
    echo -e "${YELLOW}[5.5/9] Stopping Docker containers...${NC}"
    
    docker compose down --remove-orphans 2>/dev/null || true
    
    echo -e "${GREEN}  Docker containers stopped and orphans removed${NC}"
    
    # Pull latest images if requested
    if [ "$PULL_IMAGES" = true ]; then
        echo ""
        echo -e "${YELLOW}[6/9] Pulling latest Docker images...${NC}"
        docker compose pull
        echo -e "${GREEN}  Docker images updated!${NC}"
    else
        # Check if Langfuse images exist, pull only if missing
        LANGFUSE_IMAGE=$(docker images "langfuse/langfuse" --format "{{.Repository}}" 2>/dev/null)
        if [ -z "$LANGFUSE_IMAGE" ]; then
            echo -e "${YELLOW}[6/9] Langfuse images not found, pulling...${NC}"
            docker compose pull langfuse langfuse-worker
            echo -e "${GREEN}  Langfuse images pulled!${NC}"
        else
            echo -e "${GRAY}[6/9] Langfuse images already present (use --pull-images to update)${NC}"
        fi
    fi
    
    echo ""
    echo -e "${YELLOW}[7/9] Starting Docker services...${NC}"
    docker compose up -d
    
    echo -e "${GRAY}  Waiting for services to be healthy...${NC}"
    sleep 5
    
    docker compose ps
    echo -e "${GREEN}  Docker services started!${NC}"
else
    echo -e "${GRAY}[5/9] Skipping Docker cleanup (--skip-docker)${NC}"
    echo -e "${GRAY}[6/9] Skipping Docker pull (--skip-docker)${NC}"
    echo -e "${GRAY}[7/9] Skipping Docker startup (--skip-docker)${NC}"
fi

# Start Ollama
if [ "$SKIP_OLLAMA" = false ]; then
    echo ""
    echo -e "${YELLOW}[8/9] Checking Ollama status...${NC}"
    
    # Check if port 11434 is open (using nc or curl)
    OLLAMA_READY=false
    if command -v nc &> /dev/null; then
        if nc -z localhost 11434 2>/dev/null; then OLLAMA_READY=true; fi
    elif command -v curl &> /dev/null; then
        if curl -s http://localhost:11434 > /dev/null; then OLLAMA_READY=true; fi
    elif pgrep -x "ollama" > /dev/null; then
        # Fallback to process check
        OLLAMA_READY=true
    fi
    
    if [ "$OLLAMA_READY" = true ]; then
        echo -e "${GREEN}  Ollama is already running${NC}"
    else
        echo -e "${YELLOW}  Ollama not detected on port 11434. Starting...${NC}"
        if command -v ollama &> /dev/null; then
            ollama serve &>/dev/null &
            echo -e "${GREEN}  Ollama launched in background. Waiting for API...${NC}"
            
            # Wait for API
            RETRIES=10
            while [ $RETRIES -gt 0 ]; do
                sleep 1
                if curl -s http://localhost:11434 > /dev/null 2>&1; then
                    echo -e "${GREEN}  Ollama API is ready!${NC}"
                    break
                fi
                RETRIES=$((RETRIES-1))
            done
        else
            echo -e "${YELLOW}  WARNING: Ollama not found in PATH. Please start it manually.${NC}"
        fi
    fi
    
    # Auto-pull Ollama model if not present
    echo -e "${GRAY}  Checking Ollama vision model...${NC}"
    
    # Read model from .env or use default
    OLLAMA_MODEL="moondream"  # Default lightweight model
    if [ -f "$SCRIPT_DIR/.env" ]; then
        MODEL_MATCH=$(grep -E '^OLLAMA_MODEL\s*=' "$SCRIPT_DIR/.env" | cut -d'=' -f2 | tr -d '"' | tr -d ' ')
        if [ -n "$MODEL_MATCH" ]; then
            OLLAMA_MODEL="$MODEL_MATCH"
        fi
    fi
    
    # Check if model exists, pull if not
    if ! ollama list 2>/dev/null | grep -q "$OLLAMA_MODEL"; then
        echo -e "${YELLOW}  Model '$OLLAMA_MODEL' not found. Pulling...${NC}"
        if ollama pull "$OLLAMA_MODEL"; then
            echo -e "${GREEN}  Model '$OLLAMA_MODEL' pulled successfully!${NC}"
        else
            echo -e "${RED}  WARNING: Failed to pull model. Vision features may not work.${NC}"
        fi
    else
        echo -e "${GREEN}  Model '$OLLAMA_MODEL' is available.${NC}"
    fi
else
    echo -e "${GRAY}[8/9] Skipping Ollama startup (--skip-ollama)${NC}"
fi

# Start Backend and Frontend
echo ""
echo -e "${YELLOW}[9/9] Starting Backend and Frontend...${NC}"

FRONTEND_DIR="$SCRIPT_DIR/web"

if [ "$INTEGRATED" = true ]; then
    # Launch in integrated terminal (background process for frontend, foreground for backend)
    echo -e "${GRAY}  Mode: Integrated terminal (--integrated)${NC}"
    
    # Start Frontend in background
    cd "$FRONTEND_DIR"
    npm run dev &
    FRONTEND_PID=$!
    echo -e "${GREEN}  Frontend started (PID: $FRONTEND_PID)${NC}"
    
    # Trap to cleanup on exit
    cleanup() {
        echo ""
        echo -e "${YELLOW}Stopping frontend (PID: $FRONTEND_PID)...${NC}"
        kill $FRONTEND_PID 2>/dev/null || true
        echo -e "${GREEN}Cleanup complete.${NC}"
    }
    trap cleanup EXIT
    
    echo ""
    echo -e "${CYAN}>>> System startup complete!${NC}"
    echo ""
    echo -e "  Backend:  http://localhost:8000"
    echo -e "  Frontend: http://localhost:5173"
    echo -e "  Langfuse: http://localhost:3300"
    echo -e "  Qdrant:   http://localhost:6333"
    echo ""
    echo -e "${GRAY}  Frontend running in background (PID: $FRONTEND_PID)${NC}"
    echo -e "${GRAY}  Press Ctrl+C to stop both servers.${NC}"
    echo ""
    
    # Start Backend in foreground (blocking)
    cd "$SCRIPT_DIR"
    uv run uvicorn api.server:create_app --factory --host 0.0.0.0 --port 8000
else
    # DEFAULT: Launch in separate terminal windows
    echo -e "${GRAY}  Mode: Separate terminal windows (default)${NC}"
    
    # Detect terminal emulator
    if command -v gnome-terminal &> /dev/null; then
        TERMINAL="gnome-terminal"
    elif command -v konsole &> /dev/null; then
        TERMINAL="konsole"
    elif command -v xterm &> /dev/null; then
        TERMINAL="xterm"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        TERMINAL="osascript"
    else
        TERMINAL="none"
    fi
    
    # Start Backend
    if [ "$TERMINAL" = "gnome-terminal" ]; then
        gnome-terminal -- bash -c "cd '$SCRIPT_DIR' && echo -e '${CYAN}AI-Media-Indexer Backend${NC}' && uv run uvicorn api.server:create_app --factory --host 0.0.0.0 --port 8000; exec bash"
    elif [ "$TERMINAL" = "konsole" ]; then
        konsole -e bash -c "cd '$SCRIPT_DIR' && uv run uvicorn api.server:create_app --factory --host 0.0.0.0 --port 8000; exec bash" &
    elif [ "$TERMINAL" = "xterm" ]; then
        xterm -hold -e "cd '$SCRIPT_DIR' && uv run uvicorn api.server:create_app --factory --host 0.0.0.0 --port 8000" &
    elif [ "$TERMINAL" = "osascript" ]; then
        osascript -e "tell application \"Terminal\" to do script \"cd '$SCRIPT_DIR' && uv run uvicorn api.server:create_app --factory --host 0.0.0.0 --port 8000\""
    else
        echo -e "${YELLOW}  No terminal emulator found. Falling back to integrated mode...${NC}"
        cd "$SCRIPT_DIR" && uv run uvicorn api.server:create_app --factory --host 0.0.0.0 --port 8000 &
    fi
    echo -e "${GREEN}  Backend started (port 8000)${NC}"
    
    # Start Frontend
    if [ "$TERMINAL" = "gnome-terminal" ]; then
        gnome-terminal -- bash -c "cd '$FRONTEND_DIR' && echo -e '${CYAN}AI-Media-Indexer Frontend${NC}' && npm run dev; exec bash"
    elif [ "$TERMINAL" = "konsole" ]; then
        konsole -e bash -c "cd '$FRONTEND_DIR' && npm run dev; exec bash" &
    elif [ "$TERMINAL" = "xterm" ]; then
        xterm -hold -e "cd '$FRONTEND_DIR' && npm run dev" &
    elif [ "$TERMINAL" = "osascript" ]; then
        osascript -e "tell application \"Terminal\" to do script \"cd '$FRONTEND_DIR' && npm run dev\""
    else
        cd "$FRONTEND_DIR" && npm run dev &
    fi
    echo -e "${GREEN}  Frontend started (port 5173)${NC}"
    
    echo ""
    echo -e "${CYAN}>>> System startup complete!${NC}"
    echo ""
    echo -e "  Backend:  http://localhost:8000"
    echo -e "  Frontend: http://localhost:5173"
    echo -e "  Langfuse: http://localhost:3300"
    echo -e "  Qdrant:   http://localhost:6333"
    echo ""
    echo -e "${GRAY}  Two new terminal windows have been opened.${NC}"
    echo ""
fi

# Agent Mode info
if [ "$AGENT_MODE" = true ]; then
    echo ""
    echo -e "${MAGENTA}>>> Antigravity Agent Mode${NC}"
    echo ""
    echo -e "${GRAY}  To run the agent:${NC}"
    echo -e "${WHITE}    uv run python agent_main.py video.mp4 --task='$AGENT_TASK'${NC}"
    echo ""
fi
