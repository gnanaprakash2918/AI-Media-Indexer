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


def get_global_vram_usage_percent() -> float:
    """Calculates the GLOBAL VRAM usage percentage (all processes).

    Uses torch.cuda.mem_get_info() to check actual free memory,
    accounting for other apps (Browser, Docker, etc.) and driver overhead.
    """
    if not torch.cuda.is_available():
        return 0.0

    try:
        # mem_get_info returns (free, total) in bytes
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        used_bytes = total_bytes - free_bytes
        return (used_bytes / total_bytes) * 100
    except Exception as e:
        log(f"Warning: Failed to get global VRAM info: {e}")
        # Fallback to local usage
        return get_vram_usage_percent()


def cleanup_vram() -> None:
    """Forces garbage collection and clears the PyTorch CUDA cache.

    This is used to free up VRAM between heavy processing stages.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
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
        get_available_vram()
        used = get_used_vram()
        percent = get_vram_usage_percent()
        global_percent = get_global_vram_usage_percent()
        log(
            f"VRAM [{context}]: Local={used:.2f}GB ({percent:.1f}%) | Global={global_percent:.1f}%"
        )


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


# --- Merged from resource_arbiter.py ---


import asyncio
import gc
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING  # noqa: F401 (used by type checkers)

from config import settings


def safe_cleanup_vram() -> None:
    """Safely clear GPU VRAM cache — handles missing/unavailable torch gracefully.

    Use this instead of bare ``torch.cuda.empty_cache()`` in cleanup/unload
    methods so they never crash with ``NameError: name 'torch' is not defined``.
    """
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except (ImportError, NameError):
        pass


@dataclass
class ModelVRAM:
    """Approximate VRAM usage per model in GB (fp16 where applicable)."""

    whisper_large: float = 3.0
    whisper_small: float = 1.0
    insightface: float = 1.5
    vlm_7b: float = 6.0
    vlm_3b: float = 3.0
    timesformer: float = 1.0  # fp16
    nv_embed_v2: float = 16.0
    yolo_world: float = 1.0
    pyannote: float = 1.5
    clap: float = 1.0
    depth_anything: float = 1.0  # fp16
    raft: float = 0.5  # fp16


class ResourceArbiter:
    """Global GPU resource manager with VRAM tracking.

    Usage:
        async with RESOURCE_ARBITER.acquire("vlm", vram_gb=6.0, job_id="job_123"):
            result = await vlm.predict(frame)
        # Model automatically released, VRAM cleaned
    """

    def __init__(self, total_vram_gb: float | None = None):
        """Initialize ResourceArbiter.

        Args:
            total_vram_gb: Total VRAM available. Auto-detected if None.
        """
        if total_vram_gb is None:
            try:
                import torch

                if torch.cuda.is_available():
                    total_vram_gb = torch.cuda.get_device_properties(
                        0
                    ).total_memory / (1024**3)
            except ImportError:
                total_vram_gb = 8.0
            except Exception:
                total_vram_gb = 8.0

        self.total_vram = total_vram_gb or 8.0
        self.current_usage = 0.0
        self._lock = asyncio.Lock()

        # Track loaded models and their unload callbacks
        # Format: {model_name: {"vram": float, "unload_fn": callable, "last_used": float}}
        self.registry: dict[str, dict] = {}
        self._gpu_semaphore = asyncio.Semaphore(1)
        self._initialized = True

    def register_model(self, model_name: str, unload_fn: Callable) -> None:
        """Register a model's unload function for VRAM management."""
        if model_name not in self.registry:
            self.registry[model_name] = {
                "vram": 0.0,
                "unload_fn": unload_fn,
                "last_used": 0.0,
                "active": False,
            }

    @asynccontextmanager
    async def acquire(
        self,
        model_name: str,
        vram_gb: float,
        job_id: str | None = None,
        cleanup_fn: Callable | None = None,
    ):
        """Acquire GPU resources for a model.

        Args:
            model_name: Name of the model.
            vram_gb: Approximate VRAM required in GB.
            job_id: Optional job ID for cancellation support.
            cleanup_fn: Optional function to call to unload the model when done.
                       If provided, enables automatic lazy unloading.
        """
        import time

        from core.utils.cancellation import (
            CancellationError,
            get_or_create_token,
        )
        from core.utils.logger import get_logger, log_verbose

        logger = get_logger(__name__)

        token = get_or_create_token(job_id) if job_id else None

        logger.info(f"[Arbiter] Acquiring {model_name} ({vram_gb:.1f}GB)")
        log_verbose(
            f"[Arbiter] Acquire request: model={model_name}, vram={vram_gb}GB, "
            f"job_id={job_id}, has_cleanup={cleanup_fn is not None}, "
            f"current_usage={self.current_usage:.1f}GB/{self.total_vram:.1f}GB"
        )

        async with self._lock:
            # Check cancellation
            if token and token.is_cancelled:
                logger.warning(f"[Arbiter] Job {job_id} already cancelled")
                raise CancellationError(f"Job {job_id} cancelled")

            # Update registry - always update cleanup_fn if provided
            if model_name not in self.registry:
                # TRANSIENT ALLOCATION (original behavior)
                # Wait for VRAM availability
                limit = self.total_vram * (settings.max_vram_percent / 100)

                max_offload_attempts = 10
                offload_attempts = 0
                while self.current_usage + vram_gb > limit:
                    offload_attempts += 1
                    if offload_attempts > max_offload_attempts:
                        logger.warning(
                            f"[Arbiter] Exceeded {max_offload_attempts} offload attempts, proceeding anyway"
                        )
                        break
                    logger.info(
                        f"[Arbiter] VRAM full ({self.current_usage:.1f}/{limit:.1f}GB), offloading...(attempt {offload_attempts})"
                    )
                    log_verbose(
                        f"[Arbiter] VRAM pressure: need={vram_gb}GB, "
                        f"current={self.current_usage:.1f}GB, limit={limit:.1f}GB, "
                        f"registry={list(self.registry.keys())}"
                    )
                    offloaded = await self._offload_least_recent()
                    if not offloaded:
                        logger.warning(
                            "[Arbiter] No models to offload, proceeding anyway"
                        )
                        break

                self.current_usage += vram_gb
                self.registry[model_name] = {
                    "vram": vram_gb,
                    "unload_fn": cleanup_fn,
                    "last_used": time.time(),
                    "active": True,
                    "is_persistent": False,
                }
                log_verbose(f"[Arbiter] Allocated transient: {model_name}")
            else:
                # PERSISTENT/EXISTING ALLOCATION
                # Model already allocated (e.g. by ensure_loaded).
                # We just mark it active without changing current_usage.
                log_verbose(
                    f"[Arbiter] Using existing allocation: {model_name}"
                )
                if cleanup_fn:
                    self.registry[model_name]["unload_fn"] = cleanup_fn
                self.registry[model_name]["active"] = True
                self.registry[model_name]["last_used"] = time.time()
                # Ensure VRAM info is up to date if changed?
                # For now assume it's consistent.

        try:
            log_verbose(f"[Arbiter] Waiting for GPU semaphore: {model_name}")
            async with self._gpu_semaphore:
                log_verbose(f"[Arbiter] GPU semaphore acquired: {model_name}")
                yield token
                log_verbose(f"[Arbiter] GPU semaphore releasing: {model_name}")
        except CancellationError:
            logger.warning(f"[Arbiter] {model_name} cancelled mid-execution")
            raise
        finally:
            log_verbose(f"[Arbiter] Cleanup starting: {model_name}")
            async with self._lock:
                # Mark as inactive AND release VRAM allocation
                self.registry[model_name]["active"] = False
                self.registry[model_name]["last_used"] = time.time()

                # CRITICAL: Only decrement usage if this was a TRANSIENT acquisition
                # If the model is marked as persistent (managed via ensure_loaded),
                # we do NOT free the VRAM budget here. It stays allocated until unload_fn is called.
                is_persistent = self.registry[model_name].get(
                    "is_persistent", False
                )

                if not is_persistent:
                    model_vram = self.registry[model_name].get("vram", vram_gb)
                    self.current_usage = max(0, self.current_usage - model_vram)

                    # LAZY UNLOAD: Actually unload the model if setting enabled
                    unload_fn = self.registry[model_name].get("unload_fn")
                    if settings.lazy_unload and unload_fn:
                        try:
                            logger.info(
                                f"[Arbiter] Lazy unloading {model_name}"
                            )
                            log_verbose(
                                f"[Arbiter] Calling unload_fn for {model_name}, "
                                f"is_async={asyncio.iscoroutinefunction(unload_fn)}"
                            )
                            if asyncio.iscoroutinefunction(unload_fn) or (
                                callable(unload_fn)
                                and asyncio.iscoroutinefunction(
                                    unload_fn.__call__
                                )
                            ):
                                await unload_fn()
                            else:
                                unload_fn()
                            self._cleanup_vram()
                            log_verbose(
                                f"[Arbiter] {model_name} unloaded, VRAM cleaned"
                            )
                        except Exception as e:
                            logger.warning(
                                f"[Arbiter] Failed to unload {model_name}: {e}"
                            )
                            log_verbose(
                                f"[Arbiter] Unload exception: {type(e).__name__}: {e}"
                            )

                    log_verbose(
                        f"[Arbiter] Released: model={model_name}, freed={model_vram}GB, "
                        f"new_usage={self.current_usage:.1f}GB"
                    )

    async def ensure_loaded(
        self,
        model_name: str,
        vram_gb: float,
        cleanup_fn: Callable,
    ) -> bool:
        """Register and allocate VRAM for a persistent model (lazy loaded).

        This keeps the VRAM declared as 'used' even when the model is not
        actively running inference, preventing overcommitment.

        Args:
            model_name: Unique name.
            vram_gb: VRAM requirement.
            cleanup_fn: Function to unload the model.

        Returns:
            True if allocated successfully.
        """
        import time

        from core.utils.logger import get_logger

        logger = get_logger(__name__)

        async with self._lock:
            # If already loaded and allocated, just return True
            if model_name in self.registry:
                # Update metadata
                self.registry[model_name]["last_used"] = time.time()
                self.registry[model_name]["unload_fn"] = cleanup_fn
                self.registry[model_name]["is_persistent"] = (
                    True  # Mark as persistent
                )
                return True

            logger.info(f"[Arbiter] ensuring loaded {model_name} ({vram_gb}GB)")

            # Check limits
            limit = self.total_vram * (settings.max_vram_percent / 100)

            while self.current_usage + vram_gb > limit:
                logger.info(
                    f"[Arbiter] VRAM full for persistent load ({self.current_usage:.1f}/{limit:.1f}GB), offloading..."
                )
                if not await self._offload_least_recent():
                    logger.warning(
                        "[Arbiter] Failed to make space for persistent model"
                    )
                    return False

            # Allocate
            self.current_usage += vram_gb
            self.registry[model_name] = {
                "vram": vram_gb,
                "unload_fn": cleanup_fn,
                "last_used": time.time(),
                "active": False,  # Idle but loaded
                "is_persistent": True,
            }
            return True

    async def _offload_least_recent(self) -> bool:
        """Finds and unloads the least recently used inactive model."""
        candidates = [
            (name, data)
            for name, data in self.registry.items()
            if not data["active"] and data["unload_fn"]
        ]

        if not candidates:
            return False

        # Sort by last_used (oldest first)
        candidates.sort(key=lambda x: x[1]["last_used"])

        name, data = candidates[0]
        unload_fn = data["unload_fn"]
        vram = data["vram"]

        try:
            # Call the unload callback
            import inspect

            if inspect.iscoroutinefunction(unload_fn) or (
                callable(unload_fn)
                and inspect.iscoroutinefunction(unload_fn.__call__)
            ):
                await unload_fn()
            else:
                unload_fn()

            self.current_usage = max(0, self.current_usage - vram)
            # Mark as unloaded to prevent double-unload on next pressure loop
            data["unload_fn"] = None
            data["vram"] = 0

            self._cleanup_vram()
            return True
        except Exception as e:
            import logging

            logging.getLogger(__name__).warning(
                f"[Arbiter] Failed to release model: {e}"
            )
            return False

    def _cleanup_vram(self) -> None:
        """Force garbage collection and CUDA cache clear."""
        safe_cleanup_vram()

    async def force_release_all(self) -> None:
        """Emergency release all resources by calling all registered unload functions."""
        import inspect
        import logging

        logger = logging.getLogger(__name__)

        unloaded_count = 0

        # Snapshot registry items to avoid concurrent modification issues
        for name, data in list(self.registry.items()):
            unload_fn = data.get("unload_fn")
            if unload_fn:
                try:
                    logger.info(f"[Arbiter] Force unloading {name}...")

                    if inspect.iscoroutinefunction(unload_fn) or (
                        callable(unload_fn)
                        and inspect.iscoroutinefunction(unload_fn.__call__)
                    ):
                        await unload_fn()
                    else:
                        unload_fn()

                    unloaded_count += 1
                except Exception as e:
                    logger.warning(f"[Arbiter] Failed to unload {name}: {e}")

        # Reset tracking
        self.current_usage = 0
        self.registry.clear()  # Clear registry since models are unloaded
        self._cleanup_vram()

        if unloaded_count > 0:
            logger.info(f"[Arbiter] Force released {unloaded_count} models")

    def get_status(self) -> dict:
        """Get current resource status."""
        return {
            "total_vram_gb": self.total_vram,
            "current_usage_gb": self.current_usage,
            "available_gb": self.total_vram - self.current_usage,
            "models": {
                k: {"active": v["active"], "last": v["last_used"]}
                for k, v in self.registry.items()
            },
        }


# Global singleton instance
RESOURCE_ARBITER = ResourceArbiter()

# Expose GPU semaphore as a proper public property
import warnings as _warnings


def _get_gpu_semaphore():
    """Get the GPU semaphore with a deprecation warning."""
    _warnings.warn(
        "GPU_SEMAPHORE is deprecated. Use RESOURCE_ARBITER.gpu_semaphore instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return RESOURCE_ARBITER._gpu_semaphore


# DEPRECATED: Use RESOURCE_ARBITER.gpu_semaphore instead.
# This alias exists only for backward compatibility.
GPU_SEMAPHORE = RESOURCE_ARBITER._gpu_semaphore
