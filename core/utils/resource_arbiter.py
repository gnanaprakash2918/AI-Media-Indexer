"""Global GPU resource manager with VRAM tracking and cancellation support.

Prevents OOM crashes by strictly managing VRAM allocation across models.
Integrates with CancellationToken for preemptive job cancellation.
"""

import asyncio
import gc
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING  # noqa: F401 (used by type checkers)


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
        if total_vram_gb is None and torch.cuda.is_available():
            try:
                total_vram_gb = torch.cuda.get_device_properties(
                    0
                ).total_memory / (1024**3)
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
                from config import settings as _s
                limit = self.total_vram * (_s.max_vram_percent / 100)
                
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
                    "is_persistent": False 
                }
                log_verbose(f"[Arbiter] Allocated transient: {model_name}")
            else:
                # PERSISTENT/EXISTING ALLOCATION
                # Model already allocated (e.g. by ensure_loaded). 
                # We just mark it active without changing current_usage.
                log_verbose(f"[Arbiter] Using existing allocation: {model_name}")
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
                is_persistent = self.registry[model_name].get("is_persistent", False)
                
                if not is_persistent:
                    model_vram = self.registry[model_name].get("vram", vram_gb)
                    self.current_usage = max(0, self.current_usage - model_vram)
                
                    # LAZY UNLOAD: Actually unload the model if setting enabled
                    from config import settings
                    unload_fn = self.registry[model_name].get("unload_fn")
                    if settings.lazy_unload and unload_fn:
                        try:
                            logger.info(f"[Arbiter] Lazy unloading {model_name}")
                            log_verbose(
                                f"[Arbiter] Calling unload_fn for {model_name}, "
                                f"is_async={asyncio.iscoroutinefunction(unload_fn)}"
                            )
                            if asyncio.iscoroutinefunction(unload_fn) or (hasattr(unload_fn, '__call__') and asyncio.iscoroutinefunction(unload_fn.__call__)):
                                await unload_fn()
                            else:
                                unload_fn()
                            self._cleanup_vram()
                            log_verbose(f"[Arbiter] {model_name} unloaded, VRAM cleaned")
                        except Exception as e:
                            logger.warning(f"[Arbiter] Failed to unload {model_name}: {e}")
                            log_verbose(f"[Arbiter] Unload exception: {type(e).__name__}: {e}")
                    
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
        from core.utils.logger import get_logger, log_verbose
        logger = get_logger(__name__)
        
        async with self._lock:
            # If already loaded and allocated, just return True
            if model_name in self.registry:
                 # Update metadata
                self.registry[model_name]["last_used"] = time.time()
                self.registry[model_name]["unload_fn"] = cleanup_fn
                self.registry[model_name]["is_persistent"] = True # Mark as persistent
                return True
                
            logger.info(f"[Arbiter] ensuring loaded {model_name} ({vram_gb}GB)")
            
            # Check limits
            from config import settings as _s
            limit = self.total_vram * (_s.max_vram_percent / 100)
            
            while self.current_usage + vram_gb > limit:
                logger.info(f"[Arbiter] VRAM full for persistent load ({self.current_usage:.1f}/{limit:.1f}GB), offloading...")
                if not await self._offload_least_recent():
                    logger.warning("[Arbiter] Failed to make space for persistent model")
                    return False
            
            # Allocate
            self.current_usage += vram_gb
            self.registry[model_name] = {
                "vram": vram_gb,
                "unload_fn": cleanup_fn,
                "last_used": time.time(),
                "active": False, # Idle but loaded
                "is_persistent": True
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
            if inspect.iscoroutinefunction(unload_fn) or (hasattr(unload_fn, '__call__') and inspect.iscoroutinefunction(unload_fn.__call__)):
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
            logging.getLogger(__name__).warning(f"[Arbiter] Failed to release model: {e}")
            return False

    def _cleanup_vram(self) -> None:
        """Force garbage collection and CUDA cache clear."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    async def force_release_all(self) -> None:
        """Emergency release all resources by calling all registered unload functions."""
        import logging
        import inspect
        logger = logging.getLogger(__name__)
        
        unloaded_count = 0
        
        # Snapshot registry items to avoid concurrent modification issues
        for name, data in list(self.registry.items()):
            unload_fn = data.get("unload_fn")
            if unload_fn:
                try:
                    logger.info(f"[Arbiter] Force unloading {name}...")
                    
                    if inspect.iscoroutinefunction(unload_fn) or (hasattr(unload_fn, '__call__') and inspect.iscoroutinefunction(unload_fn.__call__)):
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
