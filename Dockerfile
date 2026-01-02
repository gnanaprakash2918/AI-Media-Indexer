# ==============================================================================
# AI-Media-Indexer Production Dockerfile
# Multi-stage build for GPU-optimized deployment
# ==============================================================================

# --- Stage 1: Builder (Heavy) ---
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS builder

WORKDIR /app

# Install system build tools
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    gcc \
    g++ \
    make \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install dependencies
COPY requirements.txt .

# Use --no-cache-dir to save space
RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 && \
    pip install --no-cache-dir -r requirements.txt

# --- Stage 2: Runtime (Lightweight) ---
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

WORKDIR /app

# Install runtime libs (ffmpeg needed for audio/video processing)
RUN apt-get update && apt-get install -y \
    python3 \
    python3-venv \
    ffmpeg \
    libsndfile1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Security: Create non-root user
RUN useradd -m -u 1000 appuser
USER appuser

# Copy application code
COPY --chown=appuser:appuser . .

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Expose API port
EXPOSE 8000

# Default entrypoint (can be overridden)
ENTRYPOINT ["python", "main.py"]
