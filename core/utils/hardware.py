"""Hardware detection utilities for VRAM-aware model selection.

Provides automatic detection of available GPU VRAM and selects
appropriate models to prevent OOM on laptops with limited resources.
"""

from __future__ import annotations

import gc
from typing import Literal

import torch

from core.utils.logger import log


def get_available_vram() -> float:
    """Get total VRAM in GB, 0 if CPU-only."""
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        return props.total_memory / (1024**3)
    return 0.0


def get_used_vram() -> float:
    """Get currently allocated VRAM in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024**3)
    return 0.0


def get_vram_usage_percent() -> float:
    """Get VRAM usage as percentage (0-100)."""
    total = get_available_vram()
    if total == 0:
        return 0.0
    return (get_used_vram() / total) * 100


def cleanup_vram() -> None:
    """Force garbage collection and clear CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def select_embedding_model() -> tuple[str, int]:
    vram = get_available_vram()
    
    # FAANG-level quality: prioritize accuracy over memory
    if vram >= 8:
        # State-of-the-art for retrieval (MTEB top tier)
        model = "intfloat/e5-large-v2"
        dim = 1024
    elif vram >= 4:
        model = "BAAI/bge-base-en-v1.5"
        dim = 768
    elif vram >= 2:
        model = "BAAI/bge-small-en-v1.5"
        dim = 384
    else:
        model = "sentence-transformers/all-MiniLM-L6-v2"
        dim = 384
    
    log(f"Selected embedding model: {model} ({dim}d) for {vram:.1f}GB VRAM")
    return model, dim


def select_vision_model() -> str:
    """Select vision model based on available VRAM."""
    vram = get_available_vram()
    
    if vram >= 12:
        model = "llava:34b"
    elif vram >= 8:
        model = "llava:13b"
    else:
        model = "llava:7b"
    
    log(f"Selected vision model: {model} for {vram:.1f}GB VRAM")
    return model


VramTier = Literal["low", "medium", "high"]


def get_vram_tier() -> VramTier:
    """Categorize VRAM availability for pipeline decisions."""
    vram = get_available_vram()
    if vram >= 10:
        return "high"
    elif vram >= 6:
        return "medium"
    return "low"


def can_load_model(estimated_vram_gb: float, safety_margin: float = 0.75) -> bool:
    """Check if we can safely load a model of given size.
    
    Args:
        estimated_vram_gb: Estimated VRAM needed for the model.
        safety_margin: Leave this fraction of VRAM free (default 75%).
    
    Returns:
        True if safe to load, False if would risk OOM.
    """
    total = get_available_vram()
    used = get_used_vram()
    available = total - used
    threshold = estimated_vram_gb / safety_margin
    
    return available >= threshold


def log_vram_status(context: str = "") -> None:
    """Log current VRAM usage for debugging."""
    if torch.cuda.is_available():
        total = get_available_vram()
        used = get_used_vram()
        percent = get_vram_usage_percent()
        log(
            f"VRAM [{context}]: {used:.2f}/{total:.2f}GB ({percent:.1f}%)",
            vram_used=used,
            vram_total=total,
        )
