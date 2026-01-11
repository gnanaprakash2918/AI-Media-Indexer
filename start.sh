#!/bin/bash
# Universal Startup Script for AI Media Indexer (Linux/WSL)
# Handles Hybrid ASR, Hardware Checks, and Launch.

set -e

# Defaults
DOCKER_ASR=false
RESET=false
LINT=false

# Parse Arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --docker-asr) DOCKER_ASR=true ;;
        --reset) RESET=true ;;
        --lint) LINT=true ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

echo -e "\n🚀 AI Media Indexer: Production Start (Linux/WSL)"
echo "================================================"

# 0. Linting (Optional)
if [ "$LINT" = true ]; then
    echo -e "🧹 Running Lint & Format..."
    if [ -f "scripts/lint.sh" ]; then
        ./scripts/lint.sh
    elif command -v ruff &> /dev/null; then
        ruff check . --fix
        ruff format .
    else
        echo "⚠️  Ruff not found. Skipping lint."
    fi
fi

# 1. Database Reset
if [ "$RESET" = true ]; then
    echo -e "⚠️  Reset flag detected. Wiping Database..."
    if [ -f "scripts/reset_db.py" ]; then
        uv run python scripts/reset_db.py --force
    fi
fi

# 2. ASR Strategy
if [ "$DOCKER_ASR" = true ]; then
    echo -e "🎙️  ASR Mode: DOCKER (AI4Bharat Container)"
    export USE_NATIVE_NEMO="False"
    # Check/Start Docker Container if needed (optional)
else
    echo -e "🎙️  ASR Mode: NATIVE (Local Python)"
    export USE_NATIVE_NEMO="True"
    echo "   Verifying Native Dependencies..."
    # We use uv sync to ensure extras are present
    uv sync --extra indic
fi

# 3. Infrastructure
echo -e "📦 Infrastructure Check..."
if ! docker ps -q -f name=media-indexer-qdrant > /dev/null; then
    echo "   Starting Qdrant..."
    docker compose up -d qdrant
    sleep 3
fi

# 4. Hardware
echo -e "🔍 Hardware Check..."
uv run python -c "import torch; import pynvml; print(f'   GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else '   GPU: None (CPU Mode)');"

# 5. Start Services
echo -e "🔌 Starting Backend (Port 8000)..."
uv run uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

echo -e "🎨 Starting Frontend (Port 5173)..."
cd web
if [ ! -d "node_modules" ]; then
    npm install
fi
npm run dev &
FRONTEND_PID=$!

echo -e "\n✅ System Online!"
echo "   Backend: http://localhost:8000/docs"
echo "   UI:      http://localhost:5173"
echo "   Press Ctrl+C to stop..."

# Cleanup
trap "kill $BACKEND_PID $FRONTEND_PID; exit" SIGINT SIGTERM
wait
