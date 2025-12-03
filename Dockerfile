# ==========================================
# STAGE 1: Builder (Compiling & Installing)
# ==========================================
FROM nvidia/cuda:12.4.1-base-ubuntu22.04 AS builder

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# Install compilers (Deleted in final stage)
# Squashed into one layer to save space
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-dev \
    python3.10-venv \
    build-essential \
    git \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fastest installer)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Create virtual environment
RUN python3.10 -m venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# --- CACHING MAGIC ---
# Copy ONLY requirements first. Docker caches this layer.
COPY requirements.docker.txt requirements.txt

# Install dependencies into the venv
# --no-cache-dir prevents saving the 2GB download cache to the image
RUN uv pip install --no-cache-dir -r requirements.txt \
    --index-url https://download.pytorch.org/whl/cu124 \
    --extra-index-url https://pypi.org/simple

# Optimization: Remove compiled pycache to save space
RUN find /app/.venv -name "__pycache__" -type d -exec rm -rf {} +

# ==========================================
# STAGE 2: Runtime (Production Image)
# ==========================================
# CHANGE: Switch from 'base' to 'cudnn-runtime'
# This adds the ~600MB cuDNN libraries required by CTranslate2
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

WORKDIR /app

# Install ONLY runtime libs (ffmpeg for audio)
# Added libcudnn8 just in case, though the base image should handle it
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy ONLY the virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Activate environment
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app

# Copy Code LAST (Maximizes caching)
COPY config.py /app/
COPY core/ /app/core/

CMD ["tail", "-f", "/dev/null"]