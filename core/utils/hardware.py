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
from dataclasses import dataclass
from typing import Literal

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
        """Converts the system profile to a dictionary.

        Returns:
            A dictionary representation of the profile.
        """
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
    """Retrieves the total VRAM available on the primary CUDA device.

    Returns:
        The total VRAM in gigabytes, or 0.0 if CUDA is not available.
    """
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        return props.total_memory / (1024**3)
    return 0.0


def get_used_vram() -> float:
    """Retrieves the amount of VRAM currently allocated by PyTorch.

    Returns:
        The allocated VRAM in gigabytes, or 0.0 if CUDA is not available.
    """
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024**3)
    return 0.0


def get_available_ram() -> float:
    """Retrieves the total system RAM.

    Returns:
        The total RAM in gigabytes, defaulting to 16.0 if psutil is unavailable.
    """
    try:
        import psutil

        return psutil.virtual_memory().total / (1024**3)
    except ImportError:
        return 16.0


def get_vram_usage_percent() -> float:
    """Calculates the current VRAM usage percentage.

    Returns:
        The usage percentage (0.0 to 100.0).
    """
    total = get_available_vram()
    if total == 0:
        return 0.0
    return (get_used_vram() / total) * 100


def cleanup_vram() -> None:
    """Forces garbage collection and clears the PyTorch CUDA cache.

    This is used to free up VRAM between heavy processing stages.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


VramTier = Literal["low", "medium", "high"]


def get_vram_tier(vram_gb: float | None = None) -> VramTier:
    """Categorizes hardware into performance tiers based on VRAM.

    Args:
        vram_gb: Optional total VRAM value to use for categorization.
            If None, the system total is detected automatically.

    Returns:
        The VRAM tier ('low', 'medium', or 'high').
    """
    vram = vram_gb if vram_gb is not None else get_available_vram()
    if vram >= 12:
        return "high"
    elif vram >= 6:
        return "medium"
    return "low"


def get_system_profile(
    embedding_override: str | None = None,
    vision_override: str | None = None,
) -> SystemProfile:
    """Creates a dynamic resource profile based on hardware detection.

    Implements the 'SOTA Quality Always' strategy by choosing state-of-the-art
    models and throttling concurrency/batch sizes rather than downgrading
    model quality on lower-end hardware.

    Args:
        embedding_override: Optional override for the embedding model.
        vision_override: Optional override for the vision model.

    Returns:
        An initialized SystemProfile tailored to the current hardware.
    """
    vram = get_available_vram()
    ram = get_available_ram()
    tier = get_vram_tier(vram)

    embedding_model = (
        embedding_override
        or os.getenv("EMBEDDING_MODEL_OVERRIDE")
        or "BAAI/bge-m3"
    )
    vision_model = (
        vision_override
        or os.getenv("OLLAMA_VISION_MODEL")
        or "moondream:latest"
    )

    if "nv-embed-v2" in embedding_model.lower():
        embedding_dim = 4096
    elif (
        "sfr-embedding-2" in embedding_model.lower()
        or "bge-m3" in embedding_model
        or "large" in embedding_model
    ):
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

    log(
        f"SystemProfile: {tier} tier | VRAM={vram:.1f}GB | Batch={profile.batch_size} | Concurrency={profile.max_concurrent_jobs}"
    )
    return profile


def select_embedding_model() -> tuple[str, int]:
    """Selects the optimal embedding model based on available VRAM.

    STRATEGY: SOTA Quality Always.
    - 24GB+ VRAM: NV-Embed-v2 (4096d)
    - 12GB+ VRAM: Salesforce/SFR-Embedding-2_R (1024d)
    - Else: BGE-M3 (1024d)

    Returns:
        A tuple of (model_name, embedding_dimension).
    """
    override = os.getenv("EMBEDDING_MODEL_OVERRIDE")
    if override:
        if "nv-embed-v2" in override.lower():
            dim = 4096
        elif "large" in override or "m3" in override or "sfr" in override:
            dim = 1024
        elif "base" in override:
            dim = 768
        else:
            dim = 384
        log(f"Embedding model (override): {override} ({dim}d)")
        return override, dim

    vram = get_available_vram()
    if vram >= 23.0:  # 24GB cards (3090/4090/A100)
        model = "nvidia/NV-Embed-v2"
        dim = 4096
    elif vram >= 11.0:  # 12GB+ cards (3060 12GB, 4070, etc)
        model = "Salesforce/SFR-Embedding-2_R"
        dim = 1024
    else:
        model = "BAAI/bge-m3"
        dim = 1024

    log(f"Embedding model (Auto-SOTA for {vram:.1f}GB VRAM): {model} ({dim}d)")
    return model, dim


def select_vision_model() -> str:
    """Selects the optimal vision model based on available VRAM.

    Returns:
        The name of the vision model to use.
    """
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


def can_load_model(
    estimated_vram_gb: float, safety_margin: float = 0.75
) -> bool:
    """Checks if a model of a given size can be safely loaded into VRAM.

    Args:
        estimated_vram_gb: The estimated VRAM requirement of the model.
        safety_margin: The fraction of total available memory to consider safe.

    Returns:
        True if the model can be loaded with the specified safety margin.
    """
    total = get_available_vram()
    used = get_used_vram()
    available = total - used
    threshold = estimated_vram_gb / safety_margin
    return available >= threshold


def log_vram_status(context: str = "") -> None:
    """Logs the current VRAM usage status.

    Args:
        context: Optional string to provide context in the log message.
    """
    if torch.cuda.is_available():
        total = get_available_vram()
        used = get_used_vram()
        percent = get_vram_usage_percent()
        log(f"VRAM [{context}]: {used:.2f}/{total:.2f}GB ({percent:.1f}%)")


class VRAMManager:
    """Manages the lifecycle and memory distribution of GPU-resident models.

    Implements a singleton pattern to ensure coordinated control over which
    models are currently occupying VRAM, allowing for proactive unloading
    before loading heavy models like LLVs or Ollama.
    """

    _instance: VRAMManager | None = None
    _models: dict[str, object]
    _current_model: str | None

    def __new__(cls) -> VRAMManager:
        """Ensures that only one instance of VRAMManager exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._models = {}
            cls._instance._current_model = None
        return cls._instance

    def register(self, name: str, model: object) -> None:
        """Registers a newly loaded model with the manager.

        Args:
            name: A unique identifier for the model.
            model: The model object (expected to be a PyTorch model or similar).
        """
        self._models[name] = model
        self._current_model = name
        log(f"VRAMManager: Registered '{name}'")

    def unload(self, name: str) -> None:
        """Unloads a specific model from VRAM and releases its resources.

        Args:
            name: The unique identifier of the model to unload.
        """
        if name in self._models:
            model = self._models.pop(name)
            if hasattr(model, "to"):
                try:
                    model.to("cpu")  # type: ignore
                except Exception:
                    pass
            if hasattr(model, "unload"):
                try:
                    model.unload()  # type: ignore
                except Exception:
                    pass
            del model
            cleanup_vram()
            log(f"VRAMManager: Unloaded '{name}'")
            if self._current_model == name:
                self._current_model = None

    def unload_all_except(self, keep: str | None = None) -> None:
        """Unloads all registered models except for the specified one.

        Args:
            keep: Optional identifier of the model to remain loaded.
        """
        to_unload = [n for n in list(self._models.keys()) if n != keep]
        for name in to_unload:
            self.unload(name)

    def prepare_for_model(
        self, name: str, estimated_vram_gb: float = 2.0
    ) -> None:
        """Ensures sufficient VRAM is available before loading a new model.

        Args:
            name: The identifier of the model about to be loaded.
            estimated_vram_gb: Estimated VRAM requirement for the new model.
        """
        if not can_load_model(estimated_vram_gb):
            log(f"VRAMManager: Low VRAM, unloading all for '{name}'")
            self.unload_all_except(None)
        cleanup_vram()
        log_vram_status(f"before_{name}")

    def cleanup_before_ollama(self) -> None:
        """Clears all GPU models to free maximum VRAM for external Ollama calls.

        Ollama runs in a separate process, so we must manually release all
        PyTorch-allocated VRAM for it to function correctly on low-memory GPUs.
        """
        self.unload_all_except(None)
        cleanup_vram()
        log_vram_status("before_ollama")


vram_manager = VRAMManager()
_cached_profile: SystemProfile | None = None


def get_cached_profile() -> SystemProfile:
    """Retrieves the globally cached system profile.

    Uses lazy initialization to detect hardware on the first call.

    Returns:
        The cached SystemProfile.
    """
    global _cached_profile
    if _cached_profile is None:
        _cached_profile = get_system_profile()
    return _cached_profile


def refresh_profile() -> SystemProfile:
    """Forces a refresh of the cached hardware profile.

    Use this if hardware state might have changed (e.g., after loading models).

    Returns:
        The updated SystemProfile.
    """
    global _cached_profile
    _cached_profile = get_system_profile()
    return _cached_profile
