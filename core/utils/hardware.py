"""Hardware detection and system profiling for VRAM-aware resource management.

STRATEGY: SOTA Quality Always - Never downgrade models, only throttle resources.
- Batch sizes reduce on low VRAM
- Concurrency reduces on low VRAM  
- Lazy unload enables on low VRAM
- Model selection stays SOTA unless user explicitly overrides
"""
from __future__ import annotations

import gc
import os
import subprocess
from dataclasses import dataclass, field
from typing import Literal, Optional

import torch

from core.utils.logger import log


@dataclass
class SystemProfile:
    """Dynamic resource profile based on hardware detection."""
    vram_gb: float
    ram_gb: float
    tier: Literal["low", "medium", "high"]
    
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024
    vision_model: str = "moondream:latest"
    
    batch_size: int = 1
    max_concurrent_jobs: int = 1
    frame_batch_size: int = 4
    lazy_unload: bool = True
    aggressive_cleanup: bool = True
    
    def to_dict(self) -> dict:
        return {
            "vram_gb": self.vram_gb,
            "ram_gb": self.ram_gb,
            "tier": self.tier,
            "embedding_model": self.embedding_model,
            "embedding_dim": self.embedding_dim,
            "vision_model": self.vision_model,
            "batch_size": self.batch_size,
            "max_concurrent_jobs": self.max_concurrent_jobs,
            "frame_batch_size": self.frame_batch_size,
            "lazy_unload": self.lazy_unload,
            "aggressive_cleanup": self.aggressive_cleanup,
        }


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


def get_available_ram() -> float:
    """Get total system RAM in GB."""
    try:
        import psutil
        return psutil.virtual_memory().total / (1024**3)
    except ImportError:
        return 16.0


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


VramTier = Literal["low", "medium", "high"]


def get_vram_tier(vram_gb: Optional[float] = None) -> VramTier:
    """Categorize VRAM availability."""
    vram = vram_gb if vram_gb is not None else get_available_vram()
    if vram >= 12:
        return "high"
    elif vram >= 6:
        return "medium"
    return "low"


def get_system_profile(
    embedding_override: Optional[str] = None,
    vision_override: Optional[str] = None,
) -> SystemProfile:
    """Create dynamic resource profile based on hardware.
    
    STRATEGY: SOTA models always, throttle resources for low VRAM.
    """
    vram = get_available_vram()
    ram = get_available_ram()
    tier = get_vram_tier(vram)
    
    embedding_model = embedding_override or os.getenv("EMBEDDING_MODEL_OVERRIDE") or "BAAI/bge-m3"
    vision_model = vision_override or os.getenv("OLLAMA_VISION_MODEL") or "moondream:latest"
    
    if "bge-m3" in embedding_model or "large" in embedding_model:
        embedding_dim = 1024
    elif "base" in embedding_model:
        embedding_dim = 768
    else:
        embedding_dim = 384
    
    if tier == "high":
        profile = SystemProfile(
            vram_gb=vram,
            ram_gb=ram,
            tier=tier,
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
            vision_model=vision_model,
            batch_size=32,
            max_concurrent_jobs=4,
            frame_batch_size=16,
            lazy_unload=False,
            aggressive_cleanup=False,
        )
    elif tier == "medium":
        profile = SystemProfile(
            vram_gb=vram,
            ram_gb=ram,
            tier=tier,
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
            vision_model=vision_model,
            batch_size=8,
            max_concurrent_jobs=2,
            frame_batch_size=8,
            lazy_unload=True,
            aggressive_cleanup=False,
        )
    else:
        profile = SystemProfile(
            vram_gb=vram,
            ram_gb=ram,
            tier=tier,
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
            vision_model=vision_model,
            batch_size=1,
            max_concurrent_jobs=1,
            frame_batch_size=4,
            lazy_unload=True,
            aggressive_cleanup=True,
        )
    
    log(f"SystemProfile: {tier} tier | VRAM={vram:.1f}GB | Batch={profile.batch_size} | Concurrency={profile.max_concurrent_jobs}")
    return profile


def select_embedding_model() -> tuple[str, int]:
    """Select embedding model - ALWAYS SOTA unless overridden."""
    override = os.getenv("EMBEDDING_MODEL_OVERRIDE")
    if override:
        if "large" in override or "m3" in override:
            dim = 1024
        elif "base" in override:
            dim = 768
        else:
            dim = 384
        log(f"Embedding model (override): {override} ({dim}d)")
        return override, dim
    
    model = "BAAI/bge-m3"
    dim = 1024
    log(f"Embedding model (SOTA): {model} ({dim}d)")
    return model, dim


def select_vision_model() -> str:
    """Select vision model based on VRAM."""
    vram = get_available_vram()
    override = os.getenv("OLLAMA_VISION_MODEL")
    if override:
        log(f"Vision model (override): {override}")
        return override
    
    if vram >= 12:
        model = "llava:13b"
    elif vram >= 8:
        model = "llava:7b"
    else:
        model = "moondream:latest"
    
    log(f"Vision model: {model} for {vram:.1f}GB VRAM")
    return model


def can_load_model(estimated_vram_gb: float, safety_margin: float = 0.75) -> bool:
    """Check if we can safely load a model of given size."""
    total = get_available_vram()
    used = get_used_vram()
    available = total - used
    threshold = estimated_vram_gb / safety_margin
    return available >= threshold


def log_vram_status(context: str = "") -> None:
    """Log current VRAM usage."""
    if torch.cuda.is_available():
        total = get_available_vram()
        used = get_used_vram()
        percent = get_vram_usage_percent()
        log(f"VRAM [{context}]: {used:.2f}/{total:.2f}GB ({percent:.1f}%)")


class VRAMManager:
    """Manages GPU model lifecycle."""
    
    _instance: Optional["VRAMManager"] = None
    _models: dict[str, object]
    _current_model: Optional[str]
    
    def __new__(cls) -> "VRAMManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._models = {}
            cls._instance._current_model = None
        return cls._instance
    
    def register(self, name: str, model: object) -> None:
        self._models[name] = model
        self._current_model = name
        log(f"VRAMManager: Registered '{name}'")
    
    def unload(self, name: str) -> None:
        if name in self._models:
            model = self._models.pop(name)
            if hasattr(model, 'to'):
                try:
                    model.to('cpu')
                except Exception:
                    pass
            if hasattr(model, 'unload'):
                try:
                    model.unload()
                except Exception:
                    pass
            del model
            cleanup_vram()
            log(f"VRAMManager: Unloaded '{name}'")
            if self._current_model == name:
                self._current_model = None
    
    def unload_all_except(self, keep: Optional[str] = None) -> None:
        to_unload = [n for n in list(self._models.keys()) if n != keep]
        for name in to_unload:
            self.unload(name)
    
    def prepare_for_model(self, name: str, estimated_vram_gb: float = 2.0) -> None:
        if not can_load_model(estimated_vram_gb):
            log(f"VRAMManager: Low VRAM, unloading all for '{name}'")
            self.unload_all_except(None)
        cleanup_vram()
        log_vram_status(f"before_{name}")
    
    def cleanup_before_ollama(self) -> None:
        self.unload_all_except(None)
        cleanup_vram()
        log_vram_status("before_ollama")


vram_manager = VRAMManager()
_cached_profile: Optional[SystemProfile] = None


def get_cached_profile() -> SystemProfile:
    """Get cached system profile (singleton)."""
    global _cached_profile
    if _cached_profile is None:
        _cached_profile = get_system_profile()
    return _cached_profile


def refresh_profile() -> SystemProfile:
    """Force refresh the cached profile."""
    global _cached_profile
    _cached_profile = get_system_profile()
    return _cached_profile
