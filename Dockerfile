# ==============================================================================
# AI-Media-Indexer Production Dockerfile (Optimized)
# Multi-stage build: ~60% smaller image via slim runtime
# ==============================================================================

# --- Stage 1: Builder ---
FROM python:3.11-slim-bookworm AS builder

WORKDIR /build

# Install build deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ make git \
    libsndfile1-dev libgl1-mesa-dev libglib2.0-dev \
    && rm -rf /var/lib/apt/lists/*

# Create venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python deps with no cache
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# --- Stage 2: Runtime (Slim) ---
FROM python:3.11-slim-bookworm AS runtime

WORKDIR /app

# Runtime deps only (no compilers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Non-root user
RUN useradd -m -u 1000 appuser
USER appuser

# Copy app code
COPY --chown=appuser:appuser . .

# Env
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

ENTRYPOINT ["python", "main.py"]
