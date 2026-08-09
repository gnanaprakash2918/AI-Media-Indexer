#!/bin/bash
# =============================================================================
# AI-Media-Indexer Full System Startup Script (Linux/macOS/WSL)
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
GRAY='\033[0;90m'
WHITE='\033[1;37m'
DARK_RED='\033[0;31m'
DARK_YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Base directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Default flag values
QUICK=false
FRESH=false
NUCLEAR=false
FULL=false
DISTRIBUTED=false
SKIP_OLLAMA=false
SKIP_DOCKER=false
SKIP_CLEAN=false
RECREATE_VENV=false
NUKE_QDRANT=false
PULL_IMAGES=false
INTEGRATED=false
NO_INTERACTIVE=false
SHOW_HELP=false

# Parse Arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -Quick|-q|--quick) QUICK=true ;;
        -Fresh|-f|--fresh) FRESH=true ;;
        -Nuclear|-n|--nuclear) NUCLEAR=true ;;
        -Full|--full) FULL=true ;;
        -Distributed|-d|--distributed) DISTRIBUTED=true ;;
        -SkipOllama|--skip-ollama) SKIP_OLLAMA=true ;;
        -SkipDocker|--skip-docker) SKIP_DOCKER=true ;;
        -SkipClean|--skip-clean) SKIP_CLEAN=true ;;
        -RecreateVenv|--recreate-venv) RECREATE_VENV=true ;;
        -NukeQdrant|--nuke-qdrant) NUKE_QDRANT=true ;;
        -PullImages|--pull-images) PULL_IMAGES=true ;;
        -Integrated|--integrated) INTEGRATED=true ;;
        -NoInteractive|--no-interactive) NO_INTERACTIVE=true ;;
        -Help|-h|--help) SHOW_HELP=true ;;
        *) echo -e "${RED}Unknown parameter: $1${NC}"; exit 1 ;;
    esac
    shift
done

# Show Help
if [ "$SHOW_HELP" = true ]; then
    echo ""
    echo -e "${CYAN}AI-Media-Indexer Startup Script${NC}"
    echo -e "${CYAN}================================${NC}"
    echo ""
    echo -e "${YELLOW}USAGE:${NC}"
    echo "  ./start.sh [flags]"
    echo ""
    echo -e "${YELLOW}QUICK START MODES:${NC}"
    echo "  -q, --quick, -Quick          Fast start (skip cleanup, keep all data)"
    echo "  -f, --fresh, -Fresh          Clear caches, keep indexed videos"
    echo "  -n, --nuclear, -Nuclear      Wipe ALL data (caches + Qdrant)"
    echo "  --full, -Full                Nuclear + Venv + Pull Images"
    echo ""
    echo -e "${YELLOW}FEATURE FLAGS:${NC}"
    echo "  -d, --distributed, -Distributed Enable Redis + Celery workers (parallel processing)"
    echo "  --integrated, -Integrated    Run backend & frontend in single terminal session"
    echo ""
    echo -e "${YELLOW}GRANULAR FLAGS:${NC}"
    echo "  --recreate-venv, -RecreateVenv Delete and recreate virtual environment"
    echo "  --pull-images, -PullImages     Pull latest Docker images"
    echo "  --nuke-qdrant, -NukeQdrant     Delete Qdrant data directory"
    echo "  --skip-ollama, -SkipOllama     Skip Ollama startup"
    echo "  --skip-docker, -SkipDocker     Skip all Docker operations"
    echo "  --skip-clean, -SkipClean       Skip cache cleanup"
    echo "  --no-interactive, -NoInteractive Skip menu, use defaults"
    echo ""
    echo -e "${YELLOW}EXAMPLES:${NC}"
    echo "  ./start.sh -q                    # Fastest startup"
    echo "  ./start.sh -n -d                 # Fresh start + parallel workers"
    echo "  ./start.sh -n --recreate-venv    # Complete reset"
    echo ""
    exit 0
fi

# Convenience flag mapping
if [ "$QUICK" = true ]; then
    SKIP_CLEAN=true
    NO_INTERACTIVE=true
fi
if [ "$FRESH" = true ]; then
    SKIP_CLEAN=false
    NO_INTERACTIVE=true
fi
if [ "$NUCLEAR" = true ]; then
    SKIP_CLEAN=false
    NUKE_QDRANT=true
    NO_INTERACTIVE=true
fi
if [ "$FULL" = true ]; then
    SKIP_CLEAN=false
    NUKE_QDRANT=true
    RECREATE_VENV=true
    PULL_IMAGES=true
    NO_INTERACTIVE=true
fi

# Function: Port check and conflict management
check_port_availability() {
    local PORT=$1
    local SERVICE_NAME=$2
    local PID=""
    local PROC_NAME=""

    if command -v lsof >/dev/null 2>&1; then
        PID=$(lsof -t -i:$PORT 2>/dev/null | head -n 1 || true)
    elif command -v fuser >/dev/null 2>&1; then
        PID=$(fuser $PORT/tcp 2>/dev/null | awk '{print $1}' || true)
    fi

    if [ -n "$PID" ]; then
        PROC_NAME=$(ps -p "$PID" -o comm= 2>/dev/null || echo "Unknown")

        if [[ "$PROC_NAME" =~ docker ]] || [[ "$PROC_NAME" =~ com.docker ]]; then
            local CONTAINER_ID=$(docker ps -q --filter "publish=$PORT" 2>/dev/null || true)
            if [ -n "$CONTAINER_ID" ]; then
                echo -e "${YELLOW}Note: Port $PORT ($SERVICE_NAME) is used by Docker container ($CONTAINER_ID). Stopping container...${NC}"
                docker stop "$CONTAINER_ID" >/dev/null 2>&1 || true
                sleep 1
            fi
            PID=$(lsof -t -i:$PORT 2>/dev/null | head -n 1 || true)
            if [ -z "$PID" ]; then
                echo -e "${GREEN}Port $PORT is now free.${NC}"
                return 0
            fi
        fi

        echo -e "${YELLOW}Warning: Port $PORT ($SERVICE_NAME) is currently in use by process '$PROC_NAME' (PID: $PID)${NC}"

        local SHOULD_KILL=false
        if [ "$NUCLEAR" = true ] || [ "$NO_INTERACTIVE" = true ]; then
            echo -e "${RED}  [NUCLEAR/AUTO] Automatically terminating process '$PROC_NAME' on port $PORT...${NC}"
            SHOULD_KILL=true
        else
            read -p "Do you want to kill this process to free the port? (Y/N) " CHOICE
            case "$CHOICE" in
                y|Y) SHOULD_KILL=true ;;
                *) SHOULD_KILL=false ;;
            esac
        fi

        if [ "$SHOULD_KILL" = true ]; then
            kill -9 "$PID" 2>/dev/null || true
            sleep 1
            if lsof -t -i:$PORT >/dev/null 2>&1; then
                echo -e "${RED}  Failed to kill process $PID (may require root privileges).${NC}"
                echo -e "${YELLOW}  If docker fails to start, run: sudo systemctl restart containerd docker${NC}"
            else
                echo -e "${GREEN}Process $PID terminated. Port $PORT is free.${NC}"
            fi
        else
            echo -e "${DARK_YELLOW}Skipping kill. Note: Application may fail to start.${NC}\n"
        fi
    fi
}

echo -e "${CYAN}>>> AI-Media-Indexer Full System Startup${NC}"
echo -e "${GREEN}    Agentic Search: ENABLED (LLM query expansion)${NC}\n"

# Check if any flags were set to determine interactive menu
ANY_FLAGS_SET=false
if [ "$SKIP_OLLAMA" = true ] || [ "$SKIP_DOCKER" = true ] || [ "$SKIP_CLEAN" = true ] || \
   [ "$RECREATE_VENV" = true ] || [ "$NUKE_QDRANT" = true ] || [ "$PULL_IMAGES" = true ] || \
   [ "$INTEGRATED" = true ] || [ "$DISTRIBUTED" = true ] || [ "$QUICK" = true ] || \
   [ "$FRESH" = true ] || [ "$NUCLEAR" = true ] || [ "$FULL" = true ]; then
    ANY_FLAGS_SET=true
fi

# Interactive Menu
if [ "$NO_INTERACTIVE" = false ] && [ "$ANY_FLAGS_SET" = false ]; then
    echo -e "${GRAY}============================================================${NC}"
    echo -e "${CYAN}  STARTUP OPTIONS${NC}"
    echo -e "${GRAY}============================================================${NC}\n"

    HAS_VENV="NOT FOUND"
    if [ -d ".venv" ]; then HAS_VENV="exists"; fi

    HAS_QDRANT="fresh"
    if [ -d "qdrant_data" ] || [ -d "qdrant_data_embedded" ]; then HAS_QDRANT="exists"; fi

    CACHE_SIZE="0"
    if [ -d ".cache" ]; then
        CACHE_SIZE=$(du -sm .cache 2>/dev/null | cut -f1 || echo "0")
    fi

    echo -e "${YELLOW}  Current system state:${NC}"
    echo -e "${GRAY}    - Virtual env: $HAS_VENV${NC}"
    echo -e "${GRAY}    - Qdrant data: $HAS_QDRANT${NC}"
    echo -e "${GRAY}    - Cache size:  ${CACHE_SIZE} MB${NC}\n"

    echo -e "${CYAN}  Choose startup mode:${NC}\n"
    echo -e "${GREEN}  [1] Quick Start (RECOMMENDED)${NC}"
    echo -e "${GRAY}      - Keeps caches and data intact${NC}"
    echo -e "${GRAY}      - Fastest startup time${NC}\n"
    echo -e "${YELLOW}  [2] Fresh Start${NC}"
    echo -e "${GRAY}      - Clears all caches${NC}"
    echo -e "${GRAY}      - Keeps Qdrant data (your indexed videos)${NC}\n"
    echo -e "${RED}  [3] NUCLEAR RESET${NC}"
    echo -e "${GRAY}      - Wipes caches AND Qdrant data${NC}"
    echo -e "${GRAY}      - You'll need to re-ingest all videos${NC}"
    echo -e "${GRAY}      - Local processing (no Celery)${NC}\n"
    echo -e "${MAGENTA}  [4] Distributed Start (Redis + Celery)${NC}"
    echo -e "${GRAY}      - Starts Redis, Qdrant, Celery Worker, Backend, Frontend${NC}"
    echo -e "${GRAY}      - Best for processing many videos in parallel${NC}\n"
    echo -e "${BLUE}  [5] Dev Mode (Recreate Venv)${NC}"
    echo -e "${GRAY}      - Recreate virtual environment${NC}"
    echo -e "${GRAY}      - Pull latest Docker images${NC}\n"
    echo -e "${RED}  [6] NUCLEAR + Dev Mode${NC}"
    echo -e "${GRAY}      - Wipe ALL data + recreate venv + pull images${NC}"
    echo -e "${GRAY}      - Complete fresh start from scratch${NC}\n"
    echo -e "${MAGENTA}  [7] NUCLEAR + Distributed${NC}"
    echo -e "${GRAY}      - Complete Data Wipe + Celery workers${NC}"
    echo -e "${GRAY}      - Best for restarting a large batch job from zero${NC}\n"

    read -p "Enter choice [1-7] or press Enter for Quick Start: " CHOICE

    case "$CHOICE" in
        1)
            SKIP_CLEAN=true
            echo -e "\n  >> Quick Start selected"
            ;;
        2)
            SKIP_CLEAN=false
            echo -e "\n  >> Fresh Start selected (clearing caches)"
            ;;
        3)
            NUCLEAR=true
            SKIP_CLEAN=false
            NUKE_QDRANT=true
            echo -e "\n  >> NUCLEAR RESET selected (wiping all data, local processing)"
            ;;
        4)
            SKIP_CLEAN=true
            DISTRIBUTED=true
            echo -e "\n  >> Distributed Mode selected (Redis + Celery)"
            ;;
        5)
            RECREATE_VENV=true
            PULL_IMAGES=true
            echo -e "\n  >> Dev Mode selected (recreating venv, pulling images)"
            ;;
        6)
            NUCLEAR=true
            SKIP_CLEAN=false
            NUKE_QDRANT=true
            RECREATE_VENV=true
            PULL_IMAGES=true
            echo -e "\n  >> NUCLEAR + Dev Mode selected (wiping everything + fresh venv)"
            ;;
        7)
            NUCLEAR=true
            SKIP_CLEAN=false
            NUKE_QDRANT=true
            DISTRIBUTED=true
            echo -e "\n  >> NUCLEAR + Distributed selected (Wipe + Celery)"
            ;;
        "")
            SKIP_CLEAN=true
            echo -e "\n  >> Quick Start (default)"
            ;;
        *)
            SKIP_CLEAN=true
            echo -e "\n  >> Invalid choice, using Quick Start"
            ;;
    esac
    echo ""
fi

if [ "$NUCLEAR" = true ]; then echo -e "   ${RED}Mode: NUCLEAR (wipe all data)${NC}"; fi
if [ "$DISTRIBUTED" = true ]; then echo -e "   ${MAGENTA}Mode: Distributed (Redis + Celery)${NC}"; fi
if [ "$FULL" = true ]; then echo -e "   ${RED}Mode: FULL SETUP (All features)${NC}"; fi

# Check critical ports after menu choice is finalized
check_port_availability 8000 "Backend API"
check_port_availability 3000 "Frontend UI"
check_port_availability 6333 "Qdrant Vector DB (HTTP)"
check_port_availability 6334 "Qdrant Vector DB (gRPC)"
check_port_availability 6379 "Redis"
check_port_availability 7474 "Neo4j Graph DB (HTTP)"
check_port_availability 7687 "Neo4j Graph DB (Bolt)"
check_port_availability 8001 "vLLM Inference Server"

echo -e "${YELLOW}[1/8] Working directory: $PROJECT_ROOT${NC}"

# Handle Virtual Environment
if [ "$RECREATE_VENV" = true ]; then
    echo -e "\n${YELLOW}[2/8] Recreating virtual environment...${NC}"
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
    if [ ! -d ".venv" ]; then
        echo -e "\n${YELLOW}[2/8] Creating virtual environment (not found)...${NC}"
        uv venv
        uv sync
        echo -e "${GREEN}  Virtual environment created and synced!${NC}"
    else
        echo -e "${GRAY}[2/8] Virtual environment exists${NC}"
    fi
fi

# Clean Caches
if [ "$SKIP_CLEAN" = false ]; then
    echo -e "\n${YELLOW}[3/8] Cleaning caches...${NC}"
    for dir in ".cache" ".face_cache" ".pytest_cache" "__pycache__"; do
        if [ -d "$dir" ]; then
            echo -e "${GRAY}  Removing: $dir${NC}"
            rm -rf "$dir" 2>/dev/null || true
        fi
    done

    # Clean log files
    if [ -d "logs" ]; then
        LOG_FILES=$(ls logs/*.log 2>/dev/null || true)
        if [ -n "$LOG_FILES" ]; then
            echo ""
            if [ "$NO_INTERACTIVE" = true ]; then
                echo -e "${GREEN}  [SAFETY] Preserving logs in Nuclear/Fresh mode (Use manual delete if needed)${NC}"
            else
                read -p "  [?] Delete existing log files? (y/N) " DEL_LOGS
                if [[ "$DEL_LOGS" =~ ^[yY] ]]; then
                    echo -e "${GRAY}  Removing: log files${NC}"
                    rm -f logs/*.log 2>/dev/null || true
                else
                    echo -e "${GREEN}  Preserving logs${NC}"
                fi
            fi
        fi
    fi

    # Clean __pycache__ recursively excluding .venv, node_modules, .git
    echo -e "${GRAY}  Removing: __pycache__ (recursively)${NC}"
    find . -type d -name "__pycache__" \
        -not -path "./.venv/*" \
        -not -path "./node_modules/*" \
        -not -path "./.git/*" \
        -exec rm -rf {} + 2>/dev/null || true

    echo -e "${GREEN}  Cache cleanup complete!${NC}"
else
    echo -e "${GRAY}[3/8] Skipping cache cleanup (--SkipClean)${NC}"
fi

# Configure OTEL auth for Langfuse
if [ -f ".env" ]; then
    LANGFUSE_BACKEND=$(grep -E '^LANGFUSE_BACKEND\s*=' .env | cut -d '=' -f2 | tr -d ' "' || true)
    if [[ "$LANGFUSE_BACKEND" == "docker" ]] || [[ "$LANGFUSE_BACKEND" == "cloud" ]]; then
        echo -e "\n${YELLOW}[3.5/8] Checking Langfuse OTEL configuration...${NC}"
        PUB_KEY=$(grep -E '^LANGFUSE_PUBLIC_KEY\s*=' .env | cut -d '=' -f2 | tr -d ' "' || true)
        SEC_KEY=$(grep -E '^LANGFUSE_SECRET_KEY\s*=' .env | cut -d '=' -f2 | tr -d ' "' || true)

        if [ -n "$PUB_KEY" ] && [ -n "$SEC_KEY" ]; then
            AUTH_STR="${PUB_KEY}:${SEC_KEY}"
            EXPECTED_B64=$(echo -n "$AUTH_STR" | base64 | tr -d '\n')
            EXPECTED_HEADER="Authorization=Basic $EXPECTED_B64"
            CURRENT_HEADER=$(grep -E '^OTEL_EXPORTER_OTLP_HEADERS\s*=' .env | cut -d '=' -f2- | tr -d '"' || true)

            if [ "$CURRENT_HEADER" != "$EXPECTED_HEADER" ]; then
                echo -e "${GRAY}  Updating OTEL auth header with Langfuse credentials...${NC}"
                if grep -q '^OTEL_EXPORTER_OTLP_HEADERS\s*=' .env; then
                    sed -i "s|^OTEL_EXPORTER_OTLP_HEADERS=.*|OTEL_EXPORTER_OTLP_HEADERS=$EXPECTED_HEADER|" .env
                else
                    echo "OTEL_EXPORTER_OTLP_HEADERS=$EXPECTED_HEADER" >> .env
                fi
                echo -e "${GREEN}  OTEL auth configured!${NC}"
            else
                echo -e "${GREEN}  OTEL auth already configured${NC}"
            fi
        else
            echo -e "${YELLOW}  Langfuse keys not found in .env - OTEL tracing may not work${NC}"
        fi
    fi
fi

# Nuke Qdrant & Reset Data
if [ "$NUKE_QDRANT" = true ]; then
    echo -e "\n${RED}[4/8] Performing Complete Data Reset...${NC}"
    echo -e "${GRAY}  Terminating existing AI-Media-Indexer processes...${NC}"

    pkill -f "ffmpeg" 2>/dev/null || true
    pkill -f "ffprobe" 2>/dev/null || true
    pkill -f "api/server.py" 2>/dev/null || true
    pkill -f "celery" 2>/dev/null || true
    pkill -f "vite" 2>/dev/null || true

    echo -e "${GRAY}  Stopping Docker containers and removing volumes...${NC}"
    docker compose -f docker-compose.yaml -f docker-compose.graph.yaml down -v --remove-orphans 2>/dev/null || true

    CONTAINERS=(
        "media_agent_qdrant"
        "media_agent_postgres"
        "media_agent_minio"
        "media_agent_redis"
        "media_agent_clickhouse"
        "media_agent_langfuse"
        "media_agent_langfuse_worker"
        "media_agent_createbuckets"
        "aimI_knowledge_graph"
        "media_agent_vllm"
    )
    for c in "${CONTAINERS[@]}"; do
        docker stop "$c" 2>/dev/null || true
        docker rm -f "$c" 2>/dev/null || true
    done
    echo -e "${GREEN}  Docker services stopped and volumes removed.${NC}"

    WIPE_ITEMS=(
        "qdrant_data"
        "qdrant_data_embedded"
        "qdrant_storage"
        "temp"
        "data"
        "thumbnails"
        ".cache"
        ".face_cache"
        "langfuse_data"
        "jobs.db"
        "identity.db"
        "identity_graph.db"
        "clusters.json"
        "agent_err.json"
        "output.txt"
        "debug_output.txt"
        "test_output.txt"
        "ruff_report.txt"
        "eslint_report.txt"
        "test_results.log"
    )

    # Nuclear/Full mode wipes Postgres too (matches start.ps1 behaviour)
    if [ "$NUCLEAR" = true ] || [ "$FULL" = true ]; then
        WIPE_ITEMS+=("postgres_data")
        echo -e "${RED}  >> Including postgres_data in cleanup (Nuclear/Full mode).${NC}"
    elif [ -d "postgres_data" ]; then
        WIPE_ITEMS+=("postgres_data")
        echo -e "${RED}  >> Removing postgres_data...${NC}"
    fi

    for item in "${WIPE_ITEMS[@]}"; do
        if [ -e "$item" ]; then
            echo -e "${GRAY}  Removing: $item${NC}"
            rm -rf "$item" 2>/dev/null || true
        fi
    done
    echo -e "${GREEN}  Data reset complete!${NC}"
else
    echo -e "${GRAY}[4/8] Keeping Qdrant data (use -NukeQdrant to delete)${NC}"
fi

# Check and Start Docker
DOCKER_COMPOSE_CMD="docker compose -f docker-compose.yaml -f docker-compose.graph.yaml"
if [ "$SKIP_DOCKER" = false ]; then
    echo -e "\n${YELLOW}[5/8] Checking Docker status...${NC}"
    if ! docker info >/dev/null 2>&1; then
        echo -e "${RED}  ERROR: Docker Daemon is not running or accessible. Please start Docker.${NC}"
        exit 1
    else
        echo -e "${GREEN}  Docker Daemon is running${NC}"
        if [ -f "/etc/docker/daemon.json" ] && grep -q '"default-runtime"\s*:\s*"nvidia"' /etc/docker/daemon.json 2>/dev/null; then
            echo -e "${YELLOW}  WARNING: /etc/docker/daemon.json has 'default-runtime: nvidia'.${NC}"
            echo -e "${YELLOW}  This can cause OCI runtime errors for standard containers (redis/qdrant/neo4j).${NC}"
        fi
    fi

    echo -e "\n${YELLOW}[5.5/8] Stopping Docker containers...${NC}"
    $DOCKER_COMPOSE_CMD down --remove-orphans >/dev/null 2>&1 || true
    echo -e "${GREEN}  Docker containers stopped and orphans removed${NC}"

    if [ "$PULL_IMAGES" = true ]; then
        echo -e "\n${YELLOW}[6/8] Pulling latest Docker images...${NC}"
        $DOCKER_COMPOSE_CMD pull
        echo -e "${GREEN}  Docker images updated!${NC}"
    else
        echo -e "${GRAY}[6/8] Skipping Docker pull (use -PullImages to update)${NC}"
    fi

    echo -e "\n${YELLOW}[7/8] Starting Docker services...${NC}"

    if [ "$DISTRIBUTED" = true ]; then
        export ENABLE_DISTRIBUTED_INGESTION="True"
        echo -e "${MAGENTA}  Distributed Ingestion Enabled: Starting Redis + Celery...${NC}"
        if [ -f ".env" ]; then
            if grep -q '^ENABLE_DISTRIBUTED_INGESTION=' .env; then
                sed -i 's/^ENABLE_DISTRIBUTED_INGESTION=.*/ENABLE_DISTRIBUTED_INGESTION=True/' .env
            else
                echo -e "\n# Distributed Ingestion\nENABLE_DISTRIBUTED_INGESTION=True" >> .env
            fi
        fi
    else
        export ENABLE_DISTRIBUTED_INGESTION="False"
        if [ -f ".env" ] && grep -q '^ENABLE_DISTRIBUTED_INGESTION=' .env; then
            sed -i 's/^ENABLE_DISTRIBUTED_INGESTION=.*/ENABLE_DISTRIBUTED_INGESTION=False/' .env
        fi
    fi

    echo -e "${GRAY}  Starting database containers with $DOCKER_COMPOSE_CMD...${NC}"
    if ! $DOCKER_COMPOSE_CMD up -d --wait qdrant redis neo4j; then
        echo -e "${YELLOW}  WARNING: Docker start failed. Cleaning up and attempting auto-recovery (Pull & Build)...${NC}"
        $DOCKER_COMPOSE_CMD down --remove-orphans 2>/dev/null || true
        $DOCKER_COMPOSE_CMD pull
        $DOCKER_COMPOSE_CMD build
        if ! $DOCKER_COMPOSE_CMD up -d --wait qdrant redis neo4j; then
            echo -e "${RED}  ERROR: Docker start failed after recovery attempt.${NC}"
            exit 1
        fi
    fi

    if command -v nvidia-smi >/dev/null 2>&1; then
        echo -e "${GREEN}  NVIDIA GPU detected: Launching vLLM container service...${NC}"
        $DOCKER_COMPOSE_CMD up -d vllm 2>/dev/null || true
    fi
    echo -e "${GREEN}  Docker containers started.${NC}"

    echo -e "\n${GRAY}  Waiting for database containers to initialize (5s)...${NC}"
    sleep 5
    echo -e "${GREEN}  Databases should be ready.${NC}"
else
    echo -e "${GRAY}[5/8] Skipping Docker startup (--SkipDocker)${NC}"
fi

# Check LLM / VLM Provider (vLLM or Ollama)
CURRENT_PROVIDER="vllm"
if [ -f ".env" ]; then
    ENV_PROVIDER=$(grep -E '^LLM_PROVIDER\s*=' .env | cut -d '=' -f2 | tr -d ' "' || true)
    if [ -n "$ENV_PROVIDER" ]; then CURRENT_PROVIDER="$ENV_PROVIDER"; fi
fi

if [[ "$CURRENT_PROVIDER" == "vllm" ]]; then
    echo -e "\n${YELLOW}[8/8] Checking vLLM Provider status...${NC}"
    VLLM_URL="http://localhost:8001"
    if [ -f ".env" ]; then
        ENV_VLLM_URL=$(grep -E '^VLLM_BASE_URL\s*=' .env | cut -d '=' -f2 | tr -d ' "' || true)
        if [ -n "$ENV_VLLM_URL" ]; then VLLM_URL="$ENV_VLLM_URL"; fi
    fi

    # Read model name from .env (fallback to AWQ default)
    VLLM_MODEL="Qwen/Qwen2.5-VL-3B-Instruct-AWQ"
    if [ -f ".env" ]; then
        ENV_MODEL=$(grep -E '^VLM_ENDPOINT_MODEL_NAME\s*=' .env | cut -d '=' -f2 | tr -d ' "' || true)
        if [ -n "$ENV_MODEL" ]; then VLLM_MODEL="$ENV_MODEL"; fi
    fi

    # -----------------------------------------------------------------------
    # VRAM budget (7.6 GiB card):
    #   --gpu-memory-utilization 0.70  → vLLM KV cache  ~5.33 GiB
    #   Headroom for other processes   →                 ~2.27 GiB
    #     SigLIP embeddings            ~0.50 GiB
    #     InsightFace / face recog.    ~0.30 GiB
    #     PyTorch CUDA allocator slack ~0.30 GiB
    #   Whisper + CLAP run on CPU when VRAM < threshold (auto-fallback)
    # -----------------------------------------------------------------------
    VLLM_GPU_UTIL="0.70"
    VLLM_MAX_LEN="4096"
    VLLM_EXTRA_ARGS="--quantization awq_marlin --enable-prefix-caching --disable-log-requests"
    VLLM_PORT="8001"

    echo -e "${GRAY}  Testing vLLM connection at $VLLM_URL...${NC}"
    VLLM_READY=false
    if curl -s "$VLLM_URL/health" >/dev/null 2>&1 || curl -s "$VLLM_URL/v1/models" >/dev/null 2>&1; then
        VLLM_READY=true
    fi

    if [ "$VLLM_READY" = true ]; then
        echo -e "${GREEN}  vLLM Inference Server is already running at $VLLM_URL${NC}"
    else
        # Try to auto-launch vllm serve if the binary is available
        if command -v vllm >/dev/null 2>&1 && command -v nvidia-smi >/dev/null 2>&1; then
            echo -e "${YELLOW}  vLLM not running — launching in background with VRAM-safe settings...${NC}"
            echo -e "${GRAY}  Model: $VLLM_MODEL${NC}"
            echo -e "${GRAY}  GPU util: $VLLM_GPU_UTIL  max-model-len: $VLLM_MAX_LEN  port: $VLLM_PORT${NC}"

            # Launch vllm serve as a background daemon
            mkdir -p "$PROJECT_ROOT/logs"
            nohup vllm serve "$VLLM_MODEL" \
                --port "$VLLM_PORT" \
                --gpu-memory-utilization "$VLLM_GPU_UTIL" \
                --max-model-len "$VLLM_MAX_LEN" \
                --limit-mm-per-prompt image=8 \
                --trust-remote-code \
                $VLLM_EXTRA_ARGS \
                > "$PROJECT_ROOT/logs/vllm.log" 2>&1 &
            VLLM_PID=$!
            echo -e "${GRAY}  vLLM started (PID $VLLM_PID). Logs: logs/vllm.log${NC}"

            # Wait up to 5 minutes for vLLM to become ready
            echo -e "${GRAY}  Waiting for vLLM to load model weights (this may take a few minutes)...${NC}"
            RETRIES=30
            MAX_RETRIES=30
            while [ $RETRIES -gt 0 ]; do
                if curl -s "$VLLM_URL/health" >/dev/null 2>&1 || curl -s "$VLLM_URL/v1/models" >/dev/null 2>&1; then
                    VLLM_READY=true
                    break
                fi
                ATTEMPT=$((MAX_RETRIES - RETRIES + 1))
                echo -e "${GRAY}  Waiting for vLLM engine initialization (check $ATTEMPT/$MAX_RETRIES)...${NC}"
                sleep 10
                RETRIES=$((RETRIES-1))
            done

            if [ "$VLLM_READY" = true ]; then
                echo -e "${GREEN}  vLLM Inference Server is ready at $VLLM_URL!${NC}"
            else
                echo -e "${YELLOW}  WARNING: vLLM did not respond after 5 minutes.${NC}"
                echo -e "${GRAY}  Check logs/vllm.log for details. The app will start anyway.${NC}"
            fi
        else
            # vllm binary not found — show the exact command to run manually
            echo -e "${YELLOW}  WARNING: vLLM is not running and 'vllm' binary not found in PATH.${NC}"
            echo -e "${YELLOW}  Start vLLM manually with these VRAM-safe flags:${NC}"
            echo -e "${CYAN}    vllm serve $VLLM_MODEL \\
      --port $VLLM_PORT \\
      --gpu-memory-utilization $VLLM_GPU_UTIL \\
      --max-model-len $VLLM_MAX_LEN \\
      --limit-mm-per-prompt image=8 \\
      --trust-remote-code \\
      $VLLM_EXTRA_ARGS${NC}"
            echo -e ""
            echo -e "${GRAY}  Or via Docker: docker compose up -d vllm${NC}"
            echo -e "${GRAY}  (To switch to Ollama instead: set LLM_PROVIDER=ollama in .env)${NC}"
        fi
    fi

elif [ "$SKIP_OLLAMA" = false ]; then
    echo -e "\n${YELLOW}[8/8] Checking Ollama status...${NC}"
    OLLAMA_RUNNING=false
    if command -v nc >/dev/null 2>&1; then
        if nc -z localhost 11434 2>/dev/null; then OLLAMA_RUNNING=true; fi
    elif command -v curl >/dev/null 2>&1; then
        if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then OLLAMA_RUNNING=true; fi
    fi

    if [ "$OLLAMA_RUNNING" = true ]; then
        echo -e "${GREEN}  Ollama is already running (Port 11434)${NC}"
    else
        echo -e "${YELLOW}  Ollama not detected on port 11434. Starting...${NC}"
        if command -v ollama >/dev/null 2>&1; then
            ollama serve >/dev/null 2>&1 &
            echo -e "${GRAY}  Ollama launched in background. Waiting for API...${NC}"
            RETRIES=10
            while [ $RETRIES -gt 0 ]; do
                sleep 1
                if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
                    echo -e "${GREEN}  Ollama API is ready!${NC}"
                    break
                fi
                RETRIES=$((RETRIES-1))
            done
        else
            echo -e "${YELLOW}  WARNING: Ollama not found in PATH. Please start it manually.${NC}"
        fi
    fi

    # Auto-pull Ollama vision model
    echo -e "${GRAY}  Checking Ollama vision model...${NC}"
    OLLAMA_MODEL="moondream"
    if [ -f ".env" ]; then
        MODEL_FROM_ENV=$(grep -E '^OLLAMA_MODEL\s*=' .env | cut -d '=' -f2 | tr -d ' "' || true)
        if [ -n "$MODEL_FROM_ENV" ]; then OLLAMA_MODEL="$MODEL_FROM_ENV"; fi
    fi

    if command -v ollama >/dev/null 2>&1; then
        MODEL_LIST=$(ollama list 2>&1 || true)
        if echo "$MODEL_LIST" | grep -q "$OLLAMA_MODEL"; then
            echo -e "${GREEN}  Model '$OLLAMA_MODEL' is available.${NC}"
        else
            echo -e "${YELLOW}  Model '$OLLAMA_MODEL' not found. Pulling...${NC}"
            ollama pull "$OLLAMA_MODEL"
            echo -e "${GREEN}  Model '$OLLAMA_MODEL' pulled successfully!${NC}"
        fi
    fi
else
    echo -e "${GRAY}[8/8] Skipping Ollama startup (--SkipOllama)${NC}"
fi

# Start Backend and Frontend
echo -e "\n${YELLOW}[9/9] Starting Backend and Frontend...${NC}"

FRONTEND_DIR="$PROJECT_ROOT/web"

# Function: Terminal launcher for separate windows
launch_terminal() {
    local TITLE="$1"
    local WORK_DIR="$2"
    local CMD="$3"
    local LAUNCHED=false

    if command -v ptyxis >/dev/null 2>&1; then
        ptyxis --working-directory="$WORK_DIR" -T "$TITLE" -- bash -c "$CMD; exec bash" >/dev/null 2>&1 &
        LAUNCHED=true
    elif command -v gnome-terminal >/dev/null 2>&1; then
        gnome-terminal --working-directory="$WORK_DIR" --title="$TITLE" -- bash -c "$CMD; exec bash" >/dev/null 2>&1 &
        LAUNCHED=true
    elif command -v konsole >/dev/null 2>&1; then
        konsole --workdir "$WORK_DIR" --title "$TITLE" -e bash -c "$CMD; exec bash" >/dev/null 2>&1 &
        LAUNCHED=true
    elif command -v xfce4-terminal >/dev/null 2>&1; then
        xfce4-terminal --working-directory="$WORK_DIR" --title="$TITLE" -e "bash -c \"$CMD; exec bash\"" >/dev/null 2>&1 &
        LAUNCHED=true
    elif command -v kitty >/dev/null 2>&1; then
        kitty --directory "$WORK_DIR" --title "$TITLE" bash -c "$CMD; exec bash" >/dev/null 2>&1 &
        LAUNCHED=true
    elif command -v alacritty >/dev/null 2>&1; then
        alacritty --working-directory "$WORK_DIR" --title "$TITLE" -e bash -c "$CMD; exec bash" >/dev/null 2>&1 &
        LAUNCHED=true
    elif command -v x-terminal-emulator >/dev/null 2>&1; then
        (cd "$WORK_DIR" && x-terminal-emulator -e bash -c "$CMD; exec bash") >/dev/null 2>&1 &
        LAUNCHED=true
    elif command -v xterm >/dev/null 2>&1; then
        (cd "$WORK_DIR" && xterm -title "$TITLE" -e bash -c "$CMD; exec bash") >/dev/null 2>&1 &
        LAUNCHED=true
    elif command -v osascript >/dev/null 2>&1; then
        osascript -e "tell application \"Terminal\" to do script \"cd '$WORK_DIR' && $CMD\"" >/dev/null 2>&1 &
        LAUNCHED=true
    elif command -v wt.exe >/dev/null 2>&1; then
        wt.exe -d "$WORK_DIR" bash -c "$CMD; exec bash" >/dev/null 2>&1 &
        LAUNCHED=true
    fi

    if [ "$LAUNCHED" = true ]; then
        return 0
    else
        return 1
    fi
}

can_launch_terminals() {
    if [ -z "$DISPLAY" ] && [ -z "$WAYLAND_DISPLAY" ] && [ -z "$SSH_CONNECTION" ]; then
        return 1
    fi
    for t in ptyxis gnome-terminal konsole xfce4-terminal kitty alacritty x-terminal-emulator xterm osascript wt.exe; do
        if command -v "$t" >/dev/null 2>&1; then
            return 0
        fi
    done
    return 1
}

# Start Backend and Frontend
echo -e "\n${YELLOW}[9/9] Starting Backend and Frontend...${NC}"

FRONTEND_DIR="$PROJECT_ROOT/web"

# Ensure frontend dependencies
if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
    echo -e "${GRAY}  Installing frontend dependencies...${NC}"
    (cd "$FRONTEND_DIR" && npm install)
fi

CELERY_PID=""
FRONTEND_PID=""
BACKEND_PID=""

cleanup() {
    echo -e "\n${YELLOW}Shutting down system services...${NC}"
    if [ -n "$BACKEND_PID" ]; then kill -9 "$BACKEND_PID" 2>/dev/null || true; fi
    if [ -n "$FRONTEND_PID" ]; then kill -9 "$FRONTEND_PID" 2>/dev/null || true; fi
    if [ -n "$CELERY_PID" ]; then kill -9 "$CELERY_PID" 2>/dev/null || true; fi
    echo -e "${GREEN}Cleanup complete.${NC}"
    exit 0
}

USE_INTEGRATED=false
if [ "$INTEGRATED" = true ] || ! can_launch_terminals; then
    USE_INTEGRATED=true
fi

if [ "$USE_INTEGRATED" = true ]; then
    echo -e "${GRAY}  Mode: Integrated terminal (-Integrated or no GUI terminal found)${NC}"
    trap cleanup SIGINT SIGTERM EXIT

    if [ "$DISTRIBUTED" = true ]; then
        echo -e "${MAGENTA}  Starting Celery Worker (Distributed)...${NC}"
        uv run celery -A core.ingestion.celery_app worker --loglevel=info -P threads &
        CELERY_PID=$!
        echo -e "${GREEN}  Celery Worker started (PID: $CELERY_PID)${NC}"
    fi

    echo -e "${GREEN}  Starting Frontend...${NC}"
    (cd "$FRONTEND_DIR" && npm run dev) &
    FRONTEND_PID=$!
    echo -e "${GREEN}  Frontend started (PID: $FRONTEND_PID)${NC}"

    echo -e "\n${GREEN}>>> System startup complete!${NC}\n"
    echo -e "${WHITE}  Backend:  http://localhost:8000${NC}"
    echo -e "${WHITE}  Frontend: http://localhost:5173${NC}"
    echo -e "${WHITE}  Langfuse: http://localhost:3300${NC}"
    echo -e "${WHITE}  Qdrant:   http://localhost:6333${NC}\n"
    echo -e "${GRAY}  Frontend running in background (PID: $FRONTEND_PID).${NC}"
    echo -e "${GRAY}  Press Ctrl+C to stop backend.${NC}\n"

    echo -e "${GREEN}  Starting Backend (Port 8000)...${NC}"
    uv run uvicorn api.server:app --host 0.0.0.0 --port 8000 &
    BACKEND_PID=$!

    wait $BACKEND_PID
else
    echo -e "${GRAY}  Mode: Separate terminal windows (default)${NC}"

    if [ "$DISTRIBUTED" = true ]; then
        WORKER_CMD="cd '$PROJECT_ROOT' && echo -e '\\033[0;35mCelery Worker\\033[0m' && echo '=============' && uv run celery -A core.ingestion.celery_app worker --loglevel=info -P threads"
        launch_terminal "AI-Media-Indexer Celery Worker" "$PROJECT_ROOT" "$WORKER_CMD"
        echo -e "${GREEN}  Celery Worker started in new terminal${NC}"
    fi

    BACKEND_CMD="cd '$PROJECT_ROOT' && echo -e '\\033[0;36mAI-Media-Indexer Backend\\033[0m' && echo '========================' && uv run uvicorn api.server:app --port 8000"
    launch_terminal "AI-Media-Indexer Backend" "$PROJECT_ROOT" "$BACKEND_CMD"
    echo -e "${GREEN}  Backend started in new terminal (port 8000)${NC}"

    FRONTEND_CMD="cd '$FRONTEND_DIR' && echo -e '\\033[0;36mAI-Media-Indexer Frontend\\033[0m' && echo '=========================' && npm run dev"
    launch_terminal "AI-Media-Indexer Frontend" "$FRONTEND_DIR" "$FRONTEND_CMD"
    echo -e "${GREEN}  Frontend started in new terminal (port 5173)${NC}"

    echo -e "\n${GREEN}>>> System startup complete!${NC}\n"
    echo -e "${WHITE}  Backend:  http://localhost:8000${NC}"
    echo -e "${WHITE}  Frontend: http://localhost:5173${NC}"
    echo -e "${WHITE}  Langfuse: http://localhost:3300${NC}"
    echo -e "${WHITE}  Qdrant:   http://localhost:6333${NC}\n"
    echo -e "${GRAY}  Two new terminal windows have been opened.${NC}"
    echo -e "${GRAY}  Close them manually when done.${NC}\n"
fi

