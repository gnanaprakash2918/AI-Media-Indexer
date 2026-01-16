#!/bin/bash
# =============================================================================
# AI-Media-Indexer Startup Script (Linux/macOS/WSL)
# Mirrors start.ps1 functionality with CLI flags
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Defaults
QUICK=false
FRESH=false
NUCLEAR=false
DISTRIBUTED=false
INSTALL_INDIC=false
SKIP_OLLAMA=false
SKIP_DOCKER=false
SKIP_CLEAN=false
RECREATE_VENV=false
PULL_IMAGES=false
SHOW_HELP=false

# Parse Arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -q|--quick) QUICK=true; SKIP_CLEAN=true ;;
        -f|--fresh) FRESH=true ;;
        -n|--nuclear) NUCLEAR=true; SKIP_CLEAN=false ;;
        -d|--distributed) DISTRIBUTED=true ;;
        -i|--indic) INSTALL_INDIC=true ;;
        --skip-ollama) SKIP_OLLAMA=true ;;
        --skip-docker) SKIP_DOCKER=true ;;
        --skip-clean) SKIP_CLEAN=true ;;
        --recreate-venv) RECREATE_VENV=true ;;
        --pull-images) PULL_IMAGES=true ;;
        -h|--help) SHOW_HELP=true ;;
        *) echo -e "${RED}Unknown parameter: $1${NC}"; exit 1 ;;
    esac
    shift
done

# Show help
if [ "$SHOW_HELP" = true ]; then
    echo ""
    echo -e "${CYAN}AI-Media-Indexer Startup Script${NC}"
    echo -e "${CYAN}================================${NC}"
    echo ""
    echo -e "${YELLOW}USAGE:${NC}"
    echo "  ./start.sh [flags]"
    echo ""
    echo -e "${YELLOW}QUICK START MODES:${NC}"
    echo "  -q, --quick        Fast start (skip cleanup, keep all data)"
    echo "  -f, --fresh        Clear caches, keep indexed videos"
    echo "  -n, --nuclear      Wipe ALL data (caches + Qdrant)"
    echo ""
    echo -e "${YELLOW}FEATURE FLAGS:${NC}"
    echo "  -d, --distributed  Enable Redis + Celery workers"
    echo "  -i, --indic        Install NeMo for Tamil/Hindi transcription"
    echo ""
    echo -e "${YELLOW}GRANULAR FLAGS:${NC}"
    echo "  --recreate-venv    Delete and recreate virtual environment"
    echo "  --pull-images      Pull latest Docker images"
    echo "  --skip-ollama      Skip Ollama startup"
    echo "  --skip-docker      Skip all Docker operations"
    echo "  --skip-clean       Skip cache cleanup"
    echo ""
    echo -e "${YELLOW}EXAMPLES:${NC}"
    echo "  ./start.sh -q                      # Fastest startup"
    echo "  ./start.sh -n -d                   # Fresh start + parallel workers"
    echo "  ./start.sh -f -i                   # Clear caches + Tamil support"
    echo "  ./start.sh -n --recreate-venv      # Complete reset"
    echo ""
    exit 0
fi

echo -e "\n${CYAN}🚀 AI-Media-Indexer Startup (Linux/macOS/WSL)${NC}"
echo "================================================"

# Show active flags
if [ "$NUCLEAR" = true ]; then echo -e "   ${RED}Mode: NUCLEAR (wipe all data)${NC}"; fi
if [ "$DISTRIBUTED" = true ]; then echo -e "   ${MAGENTA}Mode: Distributed (Redis + Celery)${NC}"; fi
if [ "$INSTALL_INDIC" = true ]; then echo -e "   ${CYAN}Mode: Indic ASR (Tamil/Hindi)${NC}"; fi

# 1. Virtual Environment
echo -e "\n${YELLOW}[1/6] Checking virtual environment...${NC}"
if [ "$RECREATE_VENV" = true ] && [ -d ".venv" ]; then
    echo "  Removing existing .venv..."
    rm -rf .venv
fi

if [ ! -d ".venv" ]; then
    echo "  Creating virtual environment with uv..."
    uv venv
    uv sync
    echo -e "  ${GREEN}Virtual environment created!${NC}"
else
    echo -e "  ${GREEN}Virtual environment exists${NC}"
fi

# 2. Install Indic ASR
if [ "$INSTALL_INDIC" = true ]; then
    echo -e "\n${CYAN}[2/6] Installing Indic ASR (NeMo toolkit)...${NC}"
    uv sync --extra indic
    echo -e "  ${GREEN}Indic ASR installed!${NC}"
fi

# 3. Clean caches
if [ "$SKIP_CLEAN" = false ]; then
    echo -e "\n${YELLOW}[3/6] Cleaning caches...${NC}"
    rm -rf .cache .face_cache .pytest_cache __pycache__ 2>/dev/null || true
    find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
    rm -f logs/*.log 2>/dev/null || true
    echo -e "  ${GREEN}Cache cleanup complete!${NC}"
else
    echo -e "${YELLOW}[3/6] Skipping cache cleanup (--skip-clean)${NC}"
fi

# 4. Nuke Qdrant data
if [ "$NUCLEAR" = true ]; then
    echo -e "\n${RED}[4/6] Performing complete data reset...${NC}"
    rm -rf qdrant_data qdrant_data_embedded thumbnails jobs.db identity.db 2>/dev/null || true
    docker-compose down -v --remove-orphans 2>/dev/null || true
    echo -e "  ${GREEN}Data reset complete!${NC}"
else
    echo -e "${YELLOW}[4/6] Keeping Qdrant data (use -n to delete)${NC}"
fi

# 5. Docker
if [ "$SKIP_DOCKER" = false ]; then
    echo -e "\n${YELLOW}[5/6] Starting Docker services...${NC}"
    
    if [ "$PULL_IMAGES" = true ]; then
        echo "  Pulling latest images..."
        docker-compose pull
    fi
    
    if [ "$DISTRIBUTED" = true ]; then
        docker-compose up -d qdrant redis
        echo -e "  ${GREEN}Qdrant + Redis started (Distributed mode)${NC}"
    else
        docker-compose up -d qdrant
        echo -e "  ${GREEN}Qdrant started${NC}"
    fi
else
    echo -e "${YELLOW}[5/6] Skipping Docker (--skip-docker)${NC}"
fi

# 6. Start Services
echo -e "\n${YELLOW}[6/6] Starting services...${NC}"

# Hardware check
uv run python -c "import torch; print(f'  GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else '  GPU: None (CPU Mode)');" 2>/dev/null || true

# Start Celery worker if distributed
if [ "$DISTRIBUTED" = true ]; then
    echo -e "  ${MAGENTA}Starting Celery Worker...${NC}"
    uv run celery -A core.ingestion.celery_app worker --loglevel=info -P threads &
    CELERY_PID=$!
fi

# Start Backend
echo -e "  ${GREEN}Starting Backend (Port 8000)...${NC}"
uv run uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# Start Frontend
echo -e "  ${GREEN}Starting Frontend (Port 5173)...${NC}"
cd web
if [ ! -d "node_modules" ]; then
    npm install
fi
npm run dev &
FRONTEND_PID=$!
cd ..

echo -e "\n${GREEN}✅ System Online!${NC}"
echo "   Backend:  http://localhost:8000"
echo "   Frontend: http://localhost:5173"
echo "   Qdrant:   http://localhost:6333"
echo ""
echo "   Press Ctrl+C to stop..."

# Cleanup handler
cleanup() {
    echo -e "\n${YELLOW}Shutting down...${NC}"
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
    if [ -n "$CELERY_PID" ]; then
        kill $CELERY_PID 2>/dev/null || true
    fi
    echo -e "${GREEN}Cleanup complete.${NC}"
    exit 0
}

trap cleanup SIGINT SIGTERM
wait
